import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import prompts
from src import agents
from src import mechanism
from src import models
from src.pipeline import (
    build_answer_records,
    build_stage_calls,
    load_id_manifest,
)
from src.runner import (
    JsonlStore,
    _batch_hash,
    _certified_batch_ids,
    _content_hash,
    _derived_config_matches_lock,
    _run_stage,
    _resolve_cohort_seeds,
    _validate_architecture_contract,
    _validate_gold_sentence_coverage,
    result_slug,
)


ROOT = Path(__file__).resolve().parents[1]


class MemoryStore:
    def __init__(self):
        self.records = []

    def write(self, records):
        self.records.extend(records)

    def flush(self):
        pass

    def durable_flush(self):
        pass


class FakeModel:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return iter(self._parameters)


def fake_parameter(class_name, n, dtype):
    cls = type(class_name, (), {})
    value = cls()
    value.numel = lambda: n
    value.dtype = dtype
    return value


class ExecutionIntegrityTests(unittest.TestCase):
    def test_extractor_budget_restores_pre_merge_ceiling(self):
        self.assertEqual(prompts.MAX_NEW_TOKENS["extractor"], 320)

    def test_generation_budgets_are_json_stable(self):
        self.assertEqual(
            json.loads(json.dumps(prompts.MAX_NEW_TOKENS, sort_keys=True)),
            prompts.MAX_NEW_TOKENS,
        )

    def test_ma_qa_uses_evidence_first_but_attempts_a_short_answer(self):
        self.assertIn("Use the evidence first", prompts.QA_SYSTEM)
        self.assertIn("best short answer", prompts.QA_SYSTEM)
        self.assertIn("general knowledge", prompts.QA_SYSTEM)
        self.assertIn(
            "only when you cannot produce a usable short answer",
            prompts.QA_SYSTEM,
        )
        self.assertEqual(prompts.ROLE_PROMPT_VERSIONS["qa"], "marag-v3")
        self.assertEqual(prompts.ROLE_PROMPT_VERSIONS["solo"], "solo-v1")

    def test_extractor_fields_match_the_one_document_runtime_shape(self):
        fields = agents.build_extractor_fields("[1] Doc: Sentence.", "Find it.")
        self.assertEqual(fields, {
            "document": "[1] Doc: Sentence.",
            "sub_question": "Find it.",
        })
        rendered = prompts.build_messages("extractor", **fields)[1]["content"]
        self.assertIn("Document:\n[1] Doc: Sentence.", rendered)
        self.assertNotIn("Look for:", rendered)
        self.assertNotIn("Keywords:", rendered)

    def test_qa_example_matches_current_step_document_layout(self):
        self.assertIn("1. Document 1: Jaws", prompts.QA_SYSTEM)
        self.assertIn("Current sub-question: Who directed", prompts.QA_SYSTEM)
        self.assertIn('\"answer\": \"Steven Spielberg\"', prompts.QA_SYSTEM)
        self.assertNotIn(
            "Which university did that director attend?\n   -",
            prompts.QA_SYSTEM,
        )

    def test_summary_marks_unsupported_guesses_as_candidates_not_evidence(self):
        self.assertIn(
            "Unsupported guesses are candidates, not evidence",
            prompts.PLAN_SUMMARY_SYSTEM,
        )
        self.assertEqual(
            prompts.ROLE_PROMPT_VERSIONS["plan_summary"], "marag-v2"
        )

    def test_loaded_precision_accepts_fp16_without_quantized_parameters(self):
        model = FakeModel([
            fake_parameter("Parameter", 100, models.torch.float16)
        ])
        proof = models.validate_loaded_precision(model, "fp16", "model")
        self.assertTrue(proof["precision_validation_passed"])
        self.assertEqual(proof["quantized_fraction"], 0.0)

    def test_loaded_precision_rejects_metadata_only_q4(self):
        model = FakeModel([
            fake_parameter("Parameter", 100, models.torch.float16)
        ])
        with self.assertRaisesRegex(RuntimeError, "requested 4bit.*quantized fraction"):
            models.validate_loaded_precision(model, "4bit", "model")

    def test_loaded_precision_accepts_material_q4(self):
        model = FakeModel([
            fake_parameter("Params4bit", 40, models.torch.uint8),
            fake_parameter("Parameter", 20, models.torch.float16),
        ])
        proof = models.validate_loaded_precision(model, "4bit", "model")
        self.assertGreaterEqual(proof["quantized_fraction"], 0.5)

    def test_loaded_precision_rejects_partial_q8(self):
        model = FakeModel([
            fake_parameter("Int8Params", 20, models.torch.int8),
            fake_parameter("Parameter", 80, models.torch.float16),
        ])
        with self.assertRaisesRegex(RuntimeError, "requested 8bit.*quantized fraction"):
            models.validate_loaded_precision(model, "8bit", "model")

    def test_loaded_precision_rejects_wrong_quantization_kind(self):
        model = FakeModel([
            fake_parameter("Int8Params", 80, models.torch.int8),
            fake_parameter("Parameter", 20, models.torch.float16),
        ])
        with self.assertRaisesRegex(RuntimeError, "requested 4bit.*4bit fraction"):
            models.validate_loaded_precision(model, "4bit", "model")

    def test_pinned_bitsandbytes_configs_match_the_quantization_contract(self):
        int8 = models._bnb_config("8bit").to_dict()
        self.assertTrue(int8["load_in_8bit"])
        self.assertFalse(int8["load_in_4bit"])
        self.assertEqual(int8["llm_int8_threshold"], 6.0)

        nf4 = models._bnb_config("4bit").to_dict()
        self.assertTrue(nf4["load_in_4bit"])
        self.assertFalse(nf4["load_in_8bit"])
        self.assertEqual(nf4["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(nf4["bnb_4bit_compute_dtype"], "float16")
        self.assertTrue(nf4["bnb_4bit_use_double_quant"])

    def test_pilot_artifact_seed_does_not_change_final_manifest_load_seed(self):
        dataset = {"n": 1500, "eval_seed": 20260805}
        pilot = {"n": 200, "seed": 20260806}
        self.assertEqual(
            _resolve_cohort_seeds(dataset, pilot, None, None, pilot_mode=True),
            (1500, 20260806, 20260805),
        )

    def test_gold_sentence_coverage_detects_same_title_split_drift(self):
        from src.retrieval import Passage, RetrievalContext

        question = {
            "question_id": "q1",
            "supporting_facts": {"title": ["Gold"], "sent_id": [1]},
            "sentence_index": [
                {"title": "Gold", "sent_id": 0, "text": "First."},
                {"title": "Gold", "sent_id": 1, "text": "Exact gold."},
            ],
        }
        good = RetrievalContext([Passage("Gold", ["First.", "Exact gold."])])
        self.assertEqual(
            _validate_gold_sentence_coverage([question], good)["gold_sentence_coverage"],
            1.0,
        )
        nbsp = RetrievalContext([Passage("Gold", ["First.", "Exact\u00a0gold."])])
        self.assertTrue(
            _validate_gold_sentence_coverage([question], nbsp)[
                "gold_sentence_text_nfkc_whitespace_equivalent"
            ]
        )
        drifted = RetrievalContext([Passage("Gold", ["First. Exact gold."])])
        with self.assertRaisesRegex(AssertionError, "supporting-sentence"):
            _validate_gold_sentence_coverage([question], drifted)

    def test_architecture_contract_fails_closed_on_every_ma_rag_principle(self):
        architecture = {
            "framework": "ma-rag",
            "conceptual_agents": prompts.ROLES,
            "max_plan_steps": prompts.MAX_PLAN_STEPS,
            "plan_step_routing": ["question-answering", "aggregate"],
            "extractor_granularity": "per_document",
            "semantic_stop_on_qa_success_no": True,
            "finalizer": "step_definer_plan_summary",
        }
        _validate_architecture_contract({"architecture": architecture})
        for key in architecture:
            broken = {"architecture": {**architecture, key: "wrong"}}
            with self.assertRaisesRegex(ValueError, key):
                _validate_architecture_contract(broken)

    def test_plan_summary_protocol_enforces_short_answer_contract(self):
        short = {"output": "Successful", "answer": "Paris", "score": 10}
        verbose = {
            "output": "Successful",
            "answer": "one two three four five six seven eight nine ten eleven twelve thirteen",
            "score": 10,
        }
        self.assertTrue(
            mechanism.protocol_ok("plan_summary", json.dumps(short), short)
        )
        self.assertFalse(
            mechanism.protocol_ok("plan_summary", json.dumps(verbose), verbose)
        )

    def test_frozen_manifest_and_exclusions_validate(self):
        ids, meta = load_id_manifest(
            ROOT / "config/manifests/final_n1500_seed20260805.json"
        )
        self.assertEqual(len(ids), 1500)
        self.assertEqual(
            meta["question_ids_sha256"],
            "5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af",
        )
        excluded = meta["exclusions"]["question_ids"]
        self.assertEqual(len(excluded), 3031)
        self.assertFalse(set(ids) & set(excluded))

    def test_qa_consumes_salvaged_extractor_payload(self):
        question = {
            "question_id": "q1", "question": "Q?", "answer": "A",
            "paragraphs": "[1] T: supporting sentence", "level": "easy",
            "type": "bridge", "supporting_facts": {"title": [], "sent_id": []},
            "sentence_index": [],
        }
        idx = {
            ("q1", "planner", 0): {
                "parsed": {"sub_questions": ["SQ?"]}, "salvaged": None,
            },
            ("q1", "extractor", 0): {
                "parsed": None,
                "salvaged": {"spans": ["supporting sentence"]},
            },
        }
        call = build_stage_calls("qa", [question], idx)[0]
        self.assertEqual(call["consumer_payload_source"], "salvaged")
        self.assertIn("supporting sentence", call["fields"]["evidence"])
        self.assertEqual(
            call["consumer_input"]["evidence_blocks"][0]["spans"],
            ["supporting sentence"],
        )

    def test_final_answer_comes_from_plan_summary(self):
        question = {
            "question_id": "q1", "question": "Q?", "answer": "final",
            "level": "hard", "type": "bridge",
            "supporting_facts": {"title": [], "sent_id": []},
            "sentence_index": [], "retrieval_stratum": "hidden_bridge",
        }
        idx = {
            ("q1", "planner", 0): {
                "parsed": {"sub_questions": ["first?", "second?"]},
                "salvaged": None,
            },
            ("q1", "qa", 0): {"stage": "qa", "parsed": {
                "answer": "bridge", "success": "yes", "rating": 8,
            }},
            ("q1", "qa_step2", 1): {
                "stage": "qa_step2", "parsed": {
                    "answer": "final", "success": "yes", "rating": 9,
                },
            },
            ("q1", "plan_summary", 0): {
                "stage": "plan_summary",
                "parsed": {"output": "Successful", "answer": "final", "score": 9},
                "consumer_input": {"stop_reason": "plan_complete"},
            },
        }
        answer = build_answer_records([question], idx, "baseline")[0]
        self.assertEqual(answer["answer_stage"], "plan_summary")
        self.assertEqual(answer["predicted_answer"], "final")
        self.assertEqual(answer["plan_depth"], 2)

    def test_solo_answer_has_no_fake_subquestions_or_evidence_metric(self):
        question = {
            "question_id": "q1", "question": "Q?", "answer": "A",
            "paragraphs": "[1] T: A", "level": "easy", "type": "bridge",
            "supporting_facts": {"title": ["T"], "sent_id": [0]},
            "sentence_index": [],
        }
        idx = {("q1", "solo", 0): {
            "stage": "solo", "parsed": {"answer": "A"}, "salvaged": None,
        }}
        answer = build_answer_records([question], idx, "single_fp16", "manifest")[0]
        self.assertIsNone(answer["n_sub_questions"])
        self.assertEqual(answer["evidence_status"], "not_applicable")
        self.assertEqual(answer["question_manifest_sha256"], "manifest")

    def test_partial_resume_regenerates_canonical_neighbors(self):
        calls = [
            {"question_id": f"q{i}", "call_index": 0, "fields": {}}
            for i in range(4)
        ]
        idx = {("q0", "qa", 0): {"question_id": "q0", "stage": "qa", "call_index": 0}}
        generated = []

        def fake_run_calls(_model, _tok, role, chunk, precision, run_id, **kwargs):
            generated.append([c["question_id"] for c in chunk])
            records = [{
                "record_type": "agent_call", "run_id": run_id,
                "question_id": c["question_id"], "stage": role,
                "call_index": c["call_index"],
            } for c in chunk]
            batch = {
                "record_type": "batch", "stage": role,
                "batch_id": kwargs["batch_id"], "phase": "scored", "oom": False,
                "members": [
                    {"question_id": c["question_id"], "call_index": c["call_index"]}
                    for c in chunk
                ],
                "batch_size_requested": kwargs["batch_size"],
                "batch_ordinal": kwargs["batch_ordinal"],
                "config_fingerprint": kwargs["config_fingerprint"],
                "question_manifest_sha256": kwargs["question_manifest_sha256"],
            }
            return records, batch

        store = MemoryStore()
        with patch("src.runner.agents.run_calls", side_effect=fake_run_calls):
            _run_stage(
                object(), object(), "qa", calls, "fp16", "r", 2, 2,
                store, idx, model_id="m", config_fingerprint="fp",
                model_revision="rev", tokenizer_revision="rev",
                question_manifest_sha256="manifest", certified_batch_ids=set(),
            )
        self.assertEqual(generated, [["q0", "q1"], ["q2", "q3"]])
        written_calls = [r["question_id"] for r in store.records if r.get("record_type") == "agent_call"]
        self.assertEqual(written_calls, ["q1", "q2", "q3"])
        certificates = [r for r in store.records if r.get("record_type") == "batch"]
        self.assertEqual(len(certificates), 2)
        self.assertEqual(certificates[0]["canonical_batch_sha256"], _batch_hash(calls[:2]))
        self.assertEqual(
            _certified_batch_ids(
                certificates, "qa", calls, 2,
                config_fingerprint="fp", question_manifest_sha256="manifest",
            ),
            {"qa:000000", "qa:000001"},
        )

    def test_jsonl_repairs_only_torn_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_bytes(b'{"ok": 1}\n{"torn":')
            store = JsonlStore(path).open("session")
            records = store.read_existing()
            store.close()
            self.assertEqual(records[0], {"ok": 1})
            self.assertEqual(records[1]["record_type"], "store_repair")
            self.assertFalse(Path(str(path) + ".lock").exists())

    def test_jsonl_rejects_complete_corrupt_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_bytes(b'{"ok": 1}\nnot-json\n{"ok": 2}\n')
            store = JsonlStore(path).open("session")
            try:
                with self.assertRaises(RuntimeError):
                    store.read_existing()
            finally:
                store.close()

    def test_mixed_model_slug_and_frozen_prompt_hashes(self):
        treatments = {
            "planner": {"model_id": "Qwen/Qwen2.5-3B-Instruct"},
            "qa": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
        }
        self.assertIn(
            "qwen2.5-3b+qwen2.5-1.5b",
            result_slug("x", 10, 1234, "Qwen/Qwen2.5-3B-Instruct", treatments),
        )
        self.assertEqual(prompts.prompt_template_hashes() | {}, {
            "planner": "c91c48626dd0b17c8ba3d29ce30ea216762404b24da3f88ddd689c71056bbc03",
            "step_definer": "9ef3f1c68808719f05c8f860a98c6e1cf4bc61df370419ec3be291b2b8c8b156",
            "extractor": "4a9f59f3a1c464bc9a64743bf7fe254adb46428b68a6caba6120a54b8d88b441",
            "qa": "88dfe73142f7fbcf76ca54a431188f265e25e24a0ff4595661e32f798fcf8ae5",
            "plan_summary": "c12dc220a69d842c75be944273bfeb90fa6c64f2bc7a2d0f2bf9402935b1ba9e",
            "solo": "337626135fa3a5054bb5a065cc638a9ca05c4e1f21b44977b4700c5a0cba94cb",
        })

    def test_derived_allocation_link_rejects_unrelated_config_tamper(self):
        base = {
            "models": {"small": "m1"},
            "dataset": {"manifest_sha256": "manifest"},
            "allocation_selector": {"enabled": True, "status": "pending"},
            "timing": {"run_ids": ["baseline"]},
            "runs": {"baseline": {"qa": "fp16"}},
        }
        run_def = {"qa": "8bit"}
        allocation = {"qa": "base_8bit"}
        artifact = {
            "run_id": "ma_optimized_exploratory",
            "source_config_sha256": _content_hash(base),
            "execution_run_id": "ma_optimized_exploratory",
            "materialized_new_run": True,
            "claim_status": "exploratory_same_sample_post_selection",
            "question_manifest_sha256": "manifest",
            "selection": {"selected": {"allocation": allocation}},
            "run_definition": run_def,
            "run_config_sha256": _content_hash(run_def),
        }
        artifact["artifact_sha256"] = _content_hash(artifact)
        derived = json.loads(json.dumps(base))
        derived["runs"]["ma_optimized_exploratory"] = run_def
        derived["allocation_selector"].update({
            "status": "frozen",
            "selected_execution_run_id": "ma_optimized_exploratory",
            "selected_allocation": allocation,
            "selected_run_config_sha256": artifact["run_config_sha256"],
        })
        derived["timing"].update({
            "selected_execution_run_id": "ma_optimized_exploratory",
            "selected_system_timing_required": True,
        })
        derived["frozen_allocation"] = {
            "run_id": "ma_optimized_exploratory",
            "execution_run_id": "ma_optimized_exploratory",
            "materialized_new_run": True,
            "selection_artifact_sha256": artifact["artifact_sha256"],
            "run_config_sha256": artifact["run_config_sha256"],
            "claim_status": artifact["claim_status"],
            "question_manifest_sha256": "manifest",
        }
        artifact["executable_config_sha256"] = _content_hash(derived)
        lock = {
            "experiment_config_contract": base,
            "experiment_config_content_sha256": _content_hash(base),
            "final_manifest_ids_sha256": "manifest",
        }
        self.assertTrue(_derived_config_matches_lock(derived, lock, artifact))
        tampered = json.loads(json.dumps(derived))
        tampered["models"]["small"] = "malicious-other-model"
        tampered_artifact = dict(artifact)
        tampered_artifact["executable_config_sha256"] = _content_hash(tampered)
        self.assertFalse(
            _derived_config_matches_lock(tampered, lock, tampered_artifact)
        )

    def test_derived_allocation_link_authorizes_selected_static_timing_only(self):
        baseline_definition = {"qa": "fp16"}
        base = {
            "dataset": {"manifest_sha256": "manifest"},
            "allocation_selector": {"enabled": True, "status": "pending"},
            "timing": {"run_ids": ["baseline"]},
            "runs": {"baseline": baseline_definition},
        }
        allocation = {"qa": "base_fp16"}
        artifact = {
            "run_id": "ma_optimized_exploratory",
            "source_config_sha256": _content_hash(base),
            "execution_run_id": "baseline",
            "materialized_new_run": False,
            "claim_status": "exploratory_same_sample_post_selection",
            "question_manifest_sha256": "manifest",
            "selection": {"selected": {"allocation": allocation}},
            "run_definition": baseline_definition,
            "run_config_sha256": _content_hash(baseline_definition),
        }
        artifact["artifact_sha256"] = _content_hash(artifact)
        derived = json.loads(json.dumps(base))
        derived["allocation_selector"].update({
            "status": "frozen",
            "selected_execution_run_id": "baseline",
            "selected_allocation": allocation,
            "selected_run_config_sha256": artifact["run_config_sha256"],
        })
        derived["timing"].update({
            "selected_execution_run_id": "baseline",
            "selected_system_timing_required": True,
        })
        derived["frozen_allocation"] = {
            "run_id": "ma_optimized_exploratory",
            "execution_run_id": "baseline",
            "materialized_new_run": False,
            "run_config_sha256": artifact["run_config_sha256"],
            "selection_artifact_sha256": artifact["artifact_sha256"],
            "claim_status": artifact["claim_status"],
            "question_manifest_sha256": "manifest",
        }
        artifact["executable_config_sha256"] = _content_hash(derived)
        lock = {
            "experiment_config_contract": base,
            "experiment_config_content_sha256": _content_hash(base),
            "final_manifest_ids_sha256": "manifest",
        }
        self.assertTrue(_derived_config_matches_lock(derived, lock, artifact))
        unauthorized = json.loads(json.dumps(derived))
        unauthorized["timing"]["run_ids"].append("unselected_tiny")
        altered_artifact = dict(artifact)
        altered_artifact["executable_config_sha256"] = _content_hash(unauthorized)
        self.assertFalse(
            _derived_config_matches_lock(unauthorized, lock, altered_artifact)
        )


if __name__ == "__main__":
    unittest.main()
