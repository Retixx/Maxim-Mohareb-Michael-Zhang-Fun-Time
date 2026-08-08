from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import check_pilot as gate
from scripts import run_retrieval_smoke as smoke_script
from scripts.run_retrieval_smoke import (
    SMOKE_PROFILE,
    derive_smoke_config,
    validate_smoke_config,
)
from src import models, prompts
from src.contracts import (
    EXPERIMENT_SCHEMA,
    QWEN3_HYBRID_FAMILY,
    QWEN3_HYBRID_MODELS,
)
from src.metrics import exact_match, f1_score
from src.runner import _validate_local_smoke_contract, resolve_treatments


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "experiment.yaml"


class LocalSmokeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def derive(self, *, model_alias: str = "tiny", batch_size: int = 4) -> dict:
        return derive_smoke_config(
            self.source,
            source_config_path=CONFIG,
            model_alias=model_alias,
            batch_size=batch_size,
        )

    def test_derivation_is_isolated_and_preserves_frozen_inputs(self) -> None:
        before = copy.deepcopy(self.source)
        smoke = self.derive()

        self.assertEqual(self.source, before)
        self.assertEqual(smoke["dataset"], self.source["dataset"])
        self.assertEqual(smoke["retrieval"], self.source["retrieval"])
        for key in (
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "n",
            "seed",
            "retrieval_strata_counts",
        ):
            self.assertEqual(smoke["pilot"][key], self.source["pilot"][key])
        self.assertEqual(smoke["local_smoke"]["profile"], SMOKE_PROFILE)
        self.assertTrue(smoke["results_dir"].startswith("analysis/local_smoke/"))
        self.assertTrue(
            smoke["pilot"]["gate_artifact"].startswith("analysis/local_smoke/")
        )

    def test_tiny_and_small_are_memory_matched_uniform_4bit_pairs(self) -> None:
        for alias in ("tiny", "small"):
            with self.subTest(alias=alias):
                smoke = self.derive(model_alias=alias)
                summary = validate_smoke_config(smoke)
                ma = resolve_treatments(smoke, "baseline")
                single = resolve_treatments(smoke, "single_fp16")
                expected_model = self.source["models"][alias]

                self.assertEqual(summary["model_id"], expected_model)
                self.assertEqual({row["model_id"] for row in ma.values()}, {expected_model})
                self.assertEqual({row["precision"] for row in ma.values()}, {"4bit"})
                self.assertEqual({row["model_id"] for row in single.values()}, {expected_model})
                self.assertEqual({row["precision"] for row in single.values()}, {"4bit"})
                self.assertEqual(
                    {row["config_fingerprint"] for row in ma.values()},
                    {row["config_fingerprint"] for row in single.values()},
                )

    def test_batch_is_fixed_at_one_through_four(self) -> None:
        for batch_size in (1, 2, 3, 4):
            with self.subTest(batch_size=batch_size):
                smoke = self.derive(batch_size=batch_size)
                summary = validate_smoke_config(smoke)
                self.assertEqual(summary["batch_size"], batch_size)
                self.assertEqual(smoke["generation"]["min_batch_size"], batch_size)
        for batch_size in (0, 5, 32):
            with self.subTest(rejected=batch_size):
                with self.assertRaises(ValueError):
                    self.derive(batch_size=batch_size)

    def test_unapproved_model_alias_is_rejected(self) -> None:
        for alias in ("base", "mid", "large"):
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    self.derive(model_alias=alias)

    def test_smoke_contract_rejects_precision_or_cohort_drift(self) -> None:
        smoke = self.derive()
        smoke["runs"]["single_fp16"] = {"solo": "fp16"}
        with self.assertRaises(ValueError):
            validate_smoke_config(smoke)

        smoke = self.derive()
        smoke["pilot"]["n"] = 199
        with self.assertRaises(ValueError):
            validate_smoke_config(smoke)

        smoke = self.derive()
        smoke["retrieval"]["k"] = 11
        with self.assertRaises(ValueError):
            validate_smoke_config(smoke)

    def test_runner_smoke_boundary_is_pilot_only_and_batch_capped(self) -> None:
        smoke = self.derive()
        treatments = resolve_treatments(smoke, "baseline")
        _validate_local_smoke_contract(
            smoke,
            treatments,
            pilot_mode=True,
            batch_size=4,
            min_batch_size=4,
        )
        with self.assertRaises(ValueError):
            _validate_local_smoke_contract(
                smoke,
                treatments,
                pilot_mode=False,
                batch_size=4,
                min_batch_size=4,
            )
        with self.assertRaises(ValueError):
            _validate_local_smoke_contract(
                smoke,
                treatments,
                pilot_mode=True,
                batch_size=5,
                min_batch_size=5,
            )

    def test_tbd_revision_is_relaxed_only_for_explicit_smoke_metadata(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123"))
        tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "abc123"})
        with self.assertRaises(RuntimeError):
            models.resolved_revision_metadata(model, tokenizer, "TBD", "TBD")
        got = models.resolved_revision_metadata(
            model,
            tokenizer,
            "TBD",
            "TBD",
            allow_unpinned_tbd=True,
        )
        self.assertEqual(got["resolved_model_revision"], "abc123")
        self.assertEqual(got["resolved_tokenizer_revision"], "abc123")
        self.assertEqual(got["revision_pin_status"], "unpinned_smoke_TBD")

    def test_full_200_question_local_report_is_not_production_go(self) -> None:
        (ROOT / "results").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="local-smoke-gate-test-", dir=ROOT / "results"
        ) as directory:
            root = Path(directory)
            (root / "config").mkdir()
            question_ids = [f"q-{index:03d}" for index in range(200)]
            manifest_hash = hashlib.sha256(
                "".join(f"{question_id}\n" for question_id in question_ids).encode()
            ).hexdigest()
            manifest_path = root / "config" / "pilot.json"
            manifest_path.write_text(
                json.dumps({
                    "question_ids": question_ids,
                    "question_ids_sha256": manifest_hash,
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            source = copy.deepcopy(self.source)
            source["pilot"]["manifest_path"] = "config/pilot.json"
            source["pilot"]["manifest_sha256"] = manifest_hash
            source["pilot"]["manifest_file_sha256"] = gate.sha256_file(manifest_path)
            source["pilot"]["n"] = 200
            source["pilot"]["seed"] = 20260806
            source["pilot"]["retrieval_strata_counts"] = {
                "hidden_bridge": 160,
                "fully_named": 40,
            }
            source_path = root / "config" / "source.yaml"
            source_path.write_text(
                yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
            )

            with mock.patch.object(smoke_script, "ROOT", root):
                config = derive_smoke_config(
                    source,
                    source_config_path=source_path,
                    model_alias="tiny",
                    batch_size=4,
                )
            config_path = root / "analysis" / "local_smoke" / "experiment.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            results_dir = root / config["results_dir"]
            results_dir.mkdir(parents=True)

            retrieval_cfg = config["retrieval"]
            retrieval_identity = {
                "corpus_passages": retrieval_cfg["expected_corpus_passages"],
                "corpus_sha256": retrieval_cfg["expected_corpus_sha256"],
                "algorithm": retrieval_cfg["algorithm"],
                "query_policy": retrieval_cfg["query_policy"],
                "k_per_step": retrieval_cfg["k"],
                "gold_sentence_coverage": 1.0,
                "initial_query_source": retrieval_cfg["initial_query_source"],
                "grounded_followup_k": retrieval_cfg["grounded_followup_k"],
                "anchor_k": retrieval_cfg["anchor_k"],
                "task_k": retrieval_cfg["task_k"],
                "grounded_followup_requires_evidence": False,
                "gold_sentence_text_nfkc_whitespace_equivalent": True,
            }
            payload = {
                "schema": EXPERIMENT_SCHEMA,
                "thinking_mode": False,
                "model_family": {
                    "name": QWEN3_HYBRID_FAMILY,
                    "models": dict(QWEN3_HYBRID_MODELS),
                },
                "architecture": config["architecture"],
                "pipeline_stages": prompts.PIPELINE_STAGES,
                "stage_role": prompts.STAGE_ROLE,
                "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
                "prompt_template_sha256": prompts.prompt_template_hashes(),
                "max_new_tokens": dict(prompts.MAX_NEW_TOKENS),
                "retrieval": retrieval_identity,
                "dataset_revision": config["dataset"]["revision"],
                "local_smoke_profile_sha256": gate.content_hash(
                    config["local_smoke"]
                ),
            }
            fingerprint = gate.content_hash(payload)
            resolved_revision = "a" * 40
            for run_id in ("baseline", "single_fp16"):
                records = [{
                    "record_type": "model_load",
                    "run_id": run_id,
                    "resolved_model_revision": resolved_revision,
                    "resolved_tokenizer_revision": resolved_revision,
                }, {
                    "record_type": "agent_call",
                    "run_id": run_id,
                    "question_id": question_ids[0],
                    "question_manifest_sha256": manifest_hash,
                    "experiment_fingerprint": fingerprint,
                    "conceptual_role": "planner" if run_id == "baseline" else "solo",
                    "parse_status": "ok",
                    "strict_format_ok": True,
                    "protocol_ok": True,
                }]
                for index, question_id in enumerate(question_ids):
                    hidden = index < 160
                    gold = "correct answer"
                    prediction = (
                        gold
                        if run_id == "baseline" or not hidden
                        else "wrong"
                    )
                    records.append({
                        "record_type": "answer",
                        "run_id": run_id,
                        "question_id": question_id,
                        "question_manifest_sha256": manifest_hash,
                        "experiment_fingerprint": fingerprint,
                        "retrieval_stratum": (
                            "hidden_bridge" if hidden else "fully_named"
                        ),
                        "predicted_answer": prediction,
                        "gold_answer": gold,
                        "f1": f1_score(prediction, gold),
                        "em": exact_match(prediction, gold),
                        "answer_stage": (
                            "plan_summary" if run_id == "baseline" else "solo"
                        ),
                        "final_answer_source": (
                            "summary_parsed" if run_id == "baseline" else "solo_parsed"
                        ),
                        "retrieval_gold_title_recall": 1.0,
                        "retrieval_all_gold": 1.0,
                        "retrieval_step_count": 2.0 if run_id == "baseline" else 1.0,
                        "retrieval_anchor_gold_title_recall": 1.0,
                        "retrieval_query_count": 3.0 if run_id == "baseline" else 1.0,
                        "retrieval_task_query_count": 1.0 if run_id == "baseline" else 0.0,
                        "retrieval_followup_eligible_step_count": 1 if run_id == "baseline" else 0,
                        "retrieval_followup_fired_step_count": 1 if run_id == "baseline" else 0,
                        "retrieval_grounded_followup_firing_rate": 1.0 if run_id == "baseline" else 0.0,
                        "retrieval_incremental_task_gold_title_recall": 0.5 if run_id == "baseline" else 0.0,
                        "retrieval_zero_result_query_count": 0.0,
                        "retrieval_aggregate_step_count": 0.0,
                        "retrieval_passage_exposures": 20.0 if run_id == "baseline" else 10.0,
                        "retrieval_unique_titles": 12.0 if run_id == "baseline" else 8.0,
                        **({
                            "executed_steps": 2.0,
                            "planner_emitted_depth": 2.0,
                            "plan_was_clamped": 0.0,
                            "stop_reason": "plan_complete",
                        } if run_id == "baseline" else {}),
                    })
                jsonl_path = results_dir / f"{run_id}.jsonl"
                jsonl_path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                treatments = resolve_treatments(config, run_id)
                meta = {
                    "run_id": run_id,
                    "pilot_mode": True,
                    "local_smoke_mode": True,
                    "local_smoke": config["local_smoke"],
                    "metadata_complete": True,
                    "n": 200,
                    "seed": config["pilot"]["seed"],
                    "batch_size": 4,
                    "question_ids_sha256": manifest_hash,
                    "pilot_manifest_file_sha256": config["pilot"][
                        "manifest_file_sha256"
                    ],
                    "jsonl_sha256": gate.sha256_file(jsonl_path),
                    "environment_lock_sha256": None,
                    "git_commit": "synthetic-local-smoke-commit",
                    "thinking_mode": False,
                    "model_family": {
                        "name": QWEN3_HYBRID_FAMILY,
                        "models": dict(QWEN3_HYBRID_MODELS),
                    },
                    "gpu_name": "NVIDIA GeForce RTX 3050",
                    "execution_sessions": [{"gpu_uuid": "GPU-synthetic-3050"}],
                    "deduplicated_concurrent_model_footprint_mib": 512.0,
                    "stage_config_fingerprints": {
                        stage: treatment["config_fingerprint"]
                        for stage, treatment in treatments.items()
                    },
                    "experiment_fingerprint": fingerprint,
                    "experiment_fingerprint_payload": payload,
                    "retrieval": retrieval_identity,
                }
                (results_dir / f"{run_id}.meta.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )

            with mock.patch.object(gate, "ROOT", root):
                artifact = gate.check_pilot(config_path)
                self.assertEqual(artifact["status"], "PASS_LOCAL_SMOKE")
                self.assertTrue(artifact["passed"])
                self.assertFalse(artifact["production_eligible"])
                gate_path = root / config["pilot"]["gate_artifact"]
                gate.write_atomic(gate_path, artifact)
                with self.assertRaises(RuntimeError):
                    gate.verify_gate(config_path, gate_path)


if __name__ == "__main__":
    unittest.main()
