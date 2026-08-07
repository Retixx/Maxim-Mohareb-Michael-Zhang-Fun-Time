from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src import prompts
from src.runner import resolve_treatments

from scripts.run_campaign import (
    STATIC_RUNS,
    TINY_RUNS,
    build_plan,
    completed_run_ids,
)


ROOT = Path(__file__).resolve().parents[1]


class CampaignPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8")
        )

    def test_accuracy_plan_assigns_every_static_arm_once(self) -> None:
        plan = build_plan(self.config, kind="accuracy", workers=7, seed=42)
        assigned = [run_id for values in plan["assignments"].values() for run_id in values]
        self.assertEqual(len(assigned), 22)
        self.assertEqual(set(assigned), STATIC_RUNS)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_plan_is_worker_count_independent_in_global_order(self) -> None:
        one = build_plan(self.config, kind="accuracy", workers=1, seed=7)
        many = build_plan(self.config, kind="accuracy", workers=5, seed=7)
        self.assertEqual(one["ordered_run_ids"], many["ordered_run_ids"])

    def test_timing_is_one_worker_and_excludes_tiny(self) -> None:
        plan = build_plan(self.config, kind="timing", workers=1, seed=99)
        self.assertTrue(set(plan["ordered_run_ids"]).isdisjoint(TINY_RUNS))
        with self.assertRaises(ValueError):
            build_plan(self.config, kind="timing", workers=2, seed=99)

    def test_frozen_tiny_selection_adds_only_its_timing_run(self) -> None:
        config = copy.deepcopy(self.config)
        config["frozen_allocation"] = {
            "execution_run_id": "planner_tiny",
            "materialized_new_run": False,
        }
        config["timing"]["selected_execution_run_id"] = "planner_tiny"

        timing_plan = build_plan(config, kind="timing", workers=1, seed=99)
        self.assertEqual(timing_plan["ordered_run_ids"].count("planner_tiny"), 1)
        self.assertEqual(set(timing_plan["ordered_run_ids"]) & TINY_RUNS, {"planner_tiny"})

        accuracy_plan = build_plan(config, kind="accuracy", workers=1, seed=99)
        self.assertEqual(set(accuracy_plan["ordered_run_ids"]), STATIC_RUNS)
        self.assertNotIn("ma_optimized_exploratory", accuracy_plan["ordered_run_ids"])

    def test_pilot_is_single_worker_and_only_compares_baseline_to_single(self) -> None:
        plan = build_plan(self.config, kind="pilot", workers=1, seed=99)
        self.assertEqual(plan["ordered_run_ids"], ["baseline", "single_fp16"])
        with self.assertRaises(ValueError):
            build_plan(self.config, kind="pilot", workers=2, seed=99)

    def test_restart_skips_only_finalized_hash_valid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["results_dir"] = directory
            retrieval = config["retrieval"]
            retrieval_identity = {
                "corpus_sha256": retrieval["expected_corpus_sha256"],
                "query_policy": retrieval["query_policy"],
                "k_per_step": retrieval["k"],
                "gold_sentence_coverage": 1.0,
                "anchor_k": retrieval["anchor_k"],
                "task_k": retrieval["task_k"],
            }
            payload = {
                "schema": "open_corpus_marag_v2",
                "architecture": config["architecture"],
                "pipeline_stages": prompts.PIPELINE_STAGES,
                "stage_role": prompts.STAGE_ROLE,
                "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
                "prompt_template_sha256": prompts.prompt_template_hashes(),
                "max_new_tokens": dict(prompts.MAX_NEW_TOKENS),
                "retrieval": retrieval_identity,
                "dataset_revision": config["dataset"]["revision"],
            }
            fingerprint = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest()
            manifest = json.loads(
                (ROOT / config["dataset"]["manifest_path"]).read_text(encoding="utf-8")
            )
            jsonl = Path(directory) / "baseline.jsonl"
            jsonl.write_text("".join(
                json.dumps({
                    "record_type": "answer",
                    "question_id": question_id,
                    "question_manifest_sha256": config["dataset"]["manifest_sha256"],
                    "experiment_fingerprint": fingerprint,
                }) + "\n"
                for question_id in manifest["question_ids"]
            ), encoding="utf-8")
            digest = hashlib.sha256(jsonl.read_bytes()).hexdigest()
            meta = {
                "run_id": "baseline",
                "n": config["dataset"]["n"],
                "seed": config["dataset"]["eval_seed"],
                "timing_mode": False,
                "pilot_mode": False,
                "metadata_complete": True,
                "jsonl_sha256": digest,
                "question_ids_sha256": config["dataset"]["manifest_sha256"],
                "manifest_file_sha256": config["dataset"]["manifest_file_sha256"],
                "environment_lock_sha256": "lock",
                "git_commit": "abc123",
                "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
                "prompt_template_sha256": prompts.prompt_template_hashes(),
                "stage_config_fingerprints": {
                    stage: treatment["config_fingerprint"]
                    for stage, treatment in resolve_treatments(config, "baseline").items()
                },
                "experiment_fingerprint": fingerprint,
                "experiment_fingerprint_payload": payload,
                "retrieval": retrieval_identity,
            }
            (Path(directory) / "baseline.meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            self.assertEqual(completed_run_ids(config, kind="accuracy"), {"baseline"})
            with mock.patch.dict(
                prompts.MAX_NEW_TOKENS, {"extractor": 999}, clear=False
            ):
                self.assertEqual(completed_run_ids(config, kind="accuracy"), set())
            self.assertEqual(
                completed_run_ids(
                    config,
                    kind="accuracy",
                    expected_environment_lock_sha256="different-lock",
                ),
                set(),
            )
            jsonl.write_text("tampered\n", encoding="utf-8")
            self.assertEqual(completed_run_ids(config, kind="accuracy"), set())

    def test_restart_rejects_old_fixed_hop_metadata_even_with_valid_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["results_dir"] = directory
            jsonl = Path(directory) / "baseline.jsonl"
            jsonl.write_text('{"record_type":"answer"}\n', encoding="utf-8")
            meta = {
                "run_id": "baseline",
                "n": config["dataset"]["n"],
                "seed": config["dataset"]["eval_seed"],
                "timing_mode": False,
                "pilot_mode": False,
                "metadata_complete": True,
                "jsonl_sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
            }
            (Path(directory) / "baseline.meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            self.assertEqual(completed_run_ids(config, kind="accuracy"), set())


if __name__ == "__main__":
    unittest.main()
