import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from scripts.run_retrieval_smoke import derive_smoke_config, validate_smoke_config
from src import agents, models, prompts
from src.contracts import (
    EXPERIMENT_SCHEMA,
    QWEN3_HYBRID_MODELS,
    validate_model_contract,
)
from src.runner import _order_batches_largest_first, _preflight_stage


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "experiment.yaml"


class SpyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    model_max_length = 1024

    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(dict(kwargs))
        return " ".join(str(message.get("content", "")) for message in messages)

    def __call__(
        self,
        texts,
        *,
        return_tensors=None,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ):
        del truncation, add_special_tokens
        is_single = isinstance(texts, str)
        values = [texts] if is_single else list(texts)
        lengths = [max(1, len(value.split())) for value in values]
        if return_tensors == "pt":
            width = max(lengths)
            input_ids = torch.zeros((len(values), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, length in enumerate(lengths):
                input_ids[row, width - length :] = 1
                attention_mask[row, width - length :] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        encoded = [list(range(length)) for length in lengths]
        return {"input_ids": encoded[0] if is_single else encoded}

    def decode(self, row, *, skip_special_tokens=True):
        del row, skip_special_tokens
        return '{"answer":"ok"}'


class FakeGenerationModel:
    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=1024)
    generation_config = SimpleNamespace(eos_token_id=2)

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        generated = torch.full(
            (input_ids.shape[0], int(kwargs["max_new_tokens"])),
            3,
            dtype=input_ids.dtype,
        )
        return torch.cat((input_ids, generated), dim=1)


class MemoryStore:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, records) -> None:
        self.records.extend(records)

    def durable_flush(self) -> None:
        pass


class ThinkingModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_config_pins_exact_qwen3_hybrid_family_with_thinking_off(self) -> None:
        validate_model_contract(self.config)
        self.assertEqual(EXPERIMENT_SCHEMA, "open_corpus_marag_v4")
        self.assertEqual(
            {
                "large": self.config["models"]["large"],
                "base": self.config["model_id"],
                "mid": self.config["models"]["mid"],
                "small": self.config["models"]["small"],
                "tiny": self.config["models"]["tiny"],
            },
            QWEN3_HYBRID_MODELS,
        )
        self.assertIs(self.config["thinking_mode"], False)

    def test_model_contract_rejects_thinking_or_family_drift(self) -> None:
        mutations = []

        missing_mode = copy.deepcopy(self.config)
        missing_mode.pop("thinking_mode")
        mutations.append(missing_mode)

        thinking_on = copy.deepcopy(self.config)
        thinking_on["thinking_mode"] = True
        mutations.append(thinking_on)

        dedicated_variant = copy.deepcopy(self.config)
        dedicated_variant["model_id"] = "Qwen/Qwen3-8B-Instruct-2507"
        mutations.append(dedicated_variant)

        alias_drift = copy.deepcopy(self.config)
        alias_drift["models"]["small"] = "Qwen/Qwen3-4B-Instruct-2507"
        mutations.append(alias_drift)

        foreign_literal = copy.deepcopy(self.config)
        foreign_literal["runs"]["baseline"]["planner"] = {
            "model": "OtherOrg/OtherModel",
            "precision": "fp16",
        }
        mutations.append(foreign_literal)

        for mutated in mutations:
            with self.subTest(model_id=mutated.get("model_id")):
                with self.assertRaises(ValueError):
                    validate_model_contract(mutated)

    def test_generate_batch_disables_thinking_for_all_six_prompt_roles(self) -> None:
        tokenizer = SpyTokenizer()
        prior_steps = [{
            "step_number": 1,
            "sub_question": "Who directed Jaws?",
            "answer": "Steven Spielberg",
            "answer_grounded": True,
            "success": "yes",
            "rating": 5,
        }]
        role_fields = {
            "planner": agents.build_planner_fields("Which university?"),
            "step_definer": agents.build_step_definer_fields(
                "Which university?",
                "Who directed Jaws?",
                full_plan=["Who directed Jaws?", "Which university?"],
            ),
            "extractor": agents.build_extractor_fields(
                "[1] Jaws: Steven Spielberg directed Jaws.",
                "Who directed Jaws?",
            ),
            "qa": agents.build_qa_fields(
                "Which university?",
                [("Who directed Jaws?", ["Steven Spielberg directed Jaws."])],
                sub_question="Who directed Jaws?",
            ),
            "plan_summary": agents.build_plan_summary_fields(
                "Which university?",
                ["Who directed Jaws?", "Which university?"],
                prior_steps,
                "plan_complete",
            ),
            "solo": agents.build_solo_fields(
                "Which university?", "[1] Jaws: Steven Spielberg directed Jaws."
            ),
        }
        conversations = [
            prompts.build_messages(prompt_role, **role_fields[prompt_role])
            for prompt_role in role_fields
        ]
        rows = models.generate_batch(
            FakeGenerationModel(),
            tokenizer,
            conversations,
            max_new_tokens=1,
            batch_size=2,
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(tokenizer.template_calls), 6)
        self.assertTrue(all(
            call == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            for call in tokenizer.template_calls
        ))

    def test_batch_ordering_uses_the_non_thinking_renderer(self) -> None:
        tokenizer = SpyTokenizer()
        batches = [
            (
                0,
                [{
                    "question_id": "q1",
                    "call_index": 0,
                    "fields": {"question": "A short question?"},
                }],
                [],
            )
        ]

        self.assertEqual(
            _order_batches_largest_first(tokenizer, "planner", batches), batches
        )
        self.assertEqual(tokenizer.template_calls[0]["enable_thinking"], False)

    def test_preflight_shape_rendering_uses_the_non_thinking_renderer(self) -> None:
        tokenizer = SpyTokenizer()
        call = {
            "question_id": "q1",
            "call_index": 0,
            "fields": {"question": "Which university?"},
        }

        def fake_run_calls(*args, **kwargs):
            stage = args[2]
            calls = args[3]
            prompt_role = prompts.prompt_for(stage)
            records = [
                {
                    "question_id": item["question_id"],
                    "call_index": item["call_index"],
                }
                for item in calls
            ]
            return records, {
                "forced_full_generation": kwargs["force_full_generation"],
                "generated_sequence_tokens_total": (
                    len(calls) * prompts.MAX_NEW_TOKENS[prompt_role]
                ),
            }

        store = MemoryStore()
        with patch("src.runner.agents.run_calls", side_effect=fake_run_calls) as mocked:
            _preflight_stage(
                FakeGenerationModel(),
                tokenizer,
                "planner",
                [call],
                "fp16",
                "baseline",
                2,
                store,
                {},
                model_id="Qwen/Qwen3-8B",
                execution_session_id="session",
                gpu_metadata={},
                config_fingerprint="model-fingerprint",
                model_revision="revision",
                tokenizer_revision="revision",
                question_manifest_sha256="questions",
                preflight_manifest_sha256="preflight",
                experiment_fingerprint="experiment",
            )

        self.assertTrue(tokenizer.template_calls)
        self.assertTrue(all(
            call_kwargs["enable_thinking"] is False
            for call_kwargs in tokenizer.template_calls
        ))
        self.assertIs(mocked.call_args.kwargs["force_full_generation"], True)
        self.assertTrue(store.records)

    def test_runner_contains_no_direct_chat_template_calls(self) -> None:
        tree = ast.parse((ROOT / "src" / "runner.py").read_text(encoding="utf-8"))
        direct_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_chat_template"
        ]
        self.assertEqual(direct_calls, [])

    def test_local_smoke_preserves_hybrid_family_and_thinking_off(self) -> None:
        for alias in ("tiny", "small"):
            with self.subTest(alias=alias):
                smoke = derive_smoke_config(
                    self.config,
                    source_config_path=CONFIG,
                    model_alias=alias,
                    batch_size=1,
                )
                with self.assertRaises(ValueError):
                    validate_model_contract(smoke)
                validate_model_contract(smoke, allow_local_smoke=True)
                summary = validate_smoke_config(smoke)
                self.assertIs(smoke["thinking_mode"], False)
                self.assertEqual(summary["model_id"], QWEN3_HYBRID_MODELS[alias])

    def test_local_smoke_rejects_mutated_approved_alias(self) -> None:
        drifted = copy.deepcopy(self.config)
        drifted["models"]["tiny"] = "Qwen/Qwen3-0.6B-Instruct-2507"
        with self.assertRaises(ValueError):
            derive_smoke_config(
                drifted,
                source_config_path=CONFIG,
                model_alias="tiny",
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
