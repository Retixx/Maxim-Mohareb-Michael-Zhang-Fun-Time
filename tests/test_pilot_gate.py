from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import check_pilot as gate
from src import prompts
from src.metrics import exact_match, f1_score
from src.runner import resolve_treatments


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _question_id_hash(question_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{question_id}\n" for question_id in question_ids).encode("utf-8")
    ).hexdigest()


class PilotGateFixture:
    """Build two tiny but production-shaped pilot artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = copy.deepcopy(yaml.safe_load(
            (PROJECT_ROOT / "config" / "experiment.yaml").read_text(
                encoding="utf-8"
            )
        ))
        self.config["results_dir"] = "results"
        self.config["environment_lock_path"] = "config/environment.lock.json"
        self.config["pilot"] = {
            "required_before_final_campaign": True,
            "manifest_path": "config/pilot.json",
            "n": 2,
            "seed": 20260806,
            "retrieval_strata_counts": {
                "hidden_bridge": 1,
                "fully_named": 1,
            },
            "run_ids": ["baseline", "single_fp16"],
            "gate_artifact": "analysis/pilot_gate.json",
            "decision": {
                "primary_metric": "f1",
                "pass_if_multiagent_minus_single_overall_at_least": 0.0,
                "pass_if_multiagent_minus_single_hidden_bridge_at_least": 0.0,
                "failure_action": "stop_before_final_accuracy_or_timing",
            },
        }
        for directory in ("config", "results", "analysis"):
            (root / directory).mkdir(parents=True, exist_ok=True)

        self.question_ids = ["q-hidden", "q-fully"]
        self.manifest_hash = _question_id_hash(self.question_ids)
        manifest_path = root / "config" / "pilot.json"
        manifest_path.write_text(json.dumps({
            "question_ids": self.question_ids,
            "question_ids_sha256": self.manifest_hash,
        }, indent=2) + "\n", encoding="utf-8")
        self.config["pilot"]["manifest_sha256"] = self.manifest_hash
        self.config["pilot"]["manifest_file_sha256"] = gate.sha256_file(
            manifest_path
        )

        self.lock_path = root / "config" / "environment.lock.json"
        self.lock_path.write_text(
            json.dumps({"schema_version": 1}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.config_path = root / "config" / "experiment.yaml"
        self.write_config()
        self.gate_path = root / "analysis" / "pilot_gate.json"

    def write_config(self) -> None:
        self.config_path.write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )

    def _retrieval_identity(self) -> dict:
        retrieval = self.config["retrieval"]
        return {
            "corpus_passages": retrieval["expected_corpus_passages"],
            "corpus_sha256": retrieval["expected_corpus_sha256"],
            "algorithm": retrieval["algorithm"],
            "query_policy": retrieval["query_policy"],
            "k_per_step": retrieval["k"],
            "gold_sentence_coverage": 1.0,
            "initial_query_source": retrieval["initial_query_source"],
            "grounded_followup_k": retrieval["grounded_followup_k"],
            "grounded_followup_requires_evidence": (
                retrieval["grounded_followup_requires_evidence"]
            ),
            "gold_sentence_text_nfkc_whitespace_equivalent": True,
        }

    def _fingerprint_payload(
        self, *, stale: bool, schema: str = "open_corpus_marag_v3"
    ) -> dict:
        architecture = copy.deepcopy(self.config["architecture"])
        if stale:
            architecture["framework"] = "obsolete-fixed-hop-pipeline"
        return {
            "schema": schema,
            "architecture": architecture,
            "pipeline_stages": prompts.PIPELINE_STAGES,
            "stage_role": prompts.STAGE_ROLE,
            "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
            "prompt_template_sha256": prompts.prompt_template_hashes(),
            "max_new_tokens": dict(prompts.MAX_NEW_TOKENS),
            "retrieval": self._retrieval_identity(),
            "dataset_revision": self.config["dataset"]["revision"],
        }

    def _answer(
        self,
        *,
        run_id: str,
        question_id: str,
        stratum: str,
        correct: bool,
        fingerprint: str,
    ) -> dict:
        gold = "correct answer"
        prediction = gold if correct else "wrong"
        answer = {
            "record_type": "answer",
            "run_id": run_id,
            "question_id": question_id,
            "question_manifest_sha256": self.manifest_hash,
            "experiment_fingerprint": fingerprint,
            "retrieval_stratum": stratum,
            "predicted_answer": prediction,
            "gold_answer": gold,
            "f1": f1_score(prediction, gold),
            "em": exact_match(prediction, gold),
            "answer_stage": "plan_summary" if run_id == "baseline" else "solo",
            "final_answer_source": (
                "summary_parsed" if run_id == "baseline" else "solo_parsed"
            ),
            "retrieval_gold_title_recall": 1.0,
            "retrieval_all_gold": 1.0,
            "retrieval_step_count": 2.0 if run_id == "baseline" else 1.0,
            "retrieval_anchor_gold_title_recall": 1.0,
            "retrieval_query_count": 2.0 if run_id == "baseline" else 1.0,
            "retrieval_grounded_followup_firing_rate": (
                0.5 if run_id == "baseline" else 0.0
            ),
            "retrieval_incremental_task_gold_title_recall": (
                0.5 if run_id == "baseline" else 0.0
            ),
            "retrieval_zero_result_query_count": 0.0,
            "retrieval_aggregate_step_count": 0.0,
            "retrieval_passage_exposures": 20.0 if run_id == "baseline" else 10.0,
            "retrieval_unique_titles": 12.0 if run_id == "baseline" else 8.0,
        }
        if run_id == "baseline":
            answer.update({
                "executed_steps": 2.0,
                "planner_emitted_depth": 2.0,
                "plan_was_clamped": 0.0,
                "stop_reason": "plan_complete",
            })
        return answer

    def write_runs(
        self,
        *,
        baseline_correct: bool,
        single_correct: bool,
        stale_run_id: str | None = None,
        stale_schema_run_id: str | None = None,
    ) -> None:
        strata = ["hidden_bridge", "fully_named"]
        for run_id, correct in (
            ("baseline", baseline_correct),
            ("single_fp16", single_correct),
        ):
            payload = self._fingerprint_payload(
                stale=run_id == stale_run_id,
                schema=(
                    "open_corpus_marag_v1"
                    if run_id == stale_schema_run_id
                    else "open_corpus_marag_v3"
                ),
            )
            fingerprint = gate.content_hash(payload)
            role = "planner" if run_id == "baseline" else "solo"
            records = [{
                "record_type": "agent_call",
                "run_id": run_id,
                "question_id": self.question_ids[0],
                "experiment_fingerprint": fingerprint,
                "question_manifest_sha256": self.manifest_hash,
                "conceptual_role": role,
                "parse_status": "ok",
                "strict_format_ok": True,
                "protocol_ok": True,
            }]
            records.extend(
                self._answer(
                    run_id=run_id,
                    question_id=question_id,
                    stratum=stratum,
                    correct=correct,
                    fingerprint=fingerprint,
                )
                for question_id, stratum in zip(self.question_ids, strata)
            )
            jsonl_path = self.root / "results" / f"{run_id}.jsonl"
            jsonl_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            treatments = resolve_treatments(self.config, run_id)
            meta = {
                "run_id": run_id,
                "pilot_mode": True,
                "metadata_complete": True,
                "n": 2,
                "seed": self.config["pilot"]["seed"],
                "question_ids_sha256": self.manifest_hash,
                "pilot_manifest_file_sha256": self.config["pilot"][
                    "manifest_file_sha256"
                ],
                "jsonl_sha256": gate.sha256_file(jsonl_path),
                "environment_lock_sha256": gate.sha256_file(self.lock_path),
                "git_commit": "synthetic-pilot-commit",
                "stage_config_fingerprints": {
                    stage: treatment["config_fingerprint"]
                    for stage, treatment in treatments.items()
                },
                "experiment_fingerprint": fingerprint,
                "experiment_fingerprint_payload": payload,
                "retrieval": self._retrieval_identity(),
            }
            (self.root / "results" / f"{run_id}.meta.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )

    def write_gate(self) -> dict:
        artifact = gate.check_pilot(self.config_path)
        gate.write_atomic(self.gate_path, artifact)
        return artifact


class PilotGateTests(unittest.TestCase):
    def setUp(self) -> None:
        (PROJECT_ROOT / "results").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="pilot-gate-test-", dir=PROJECT_ROOT / "results"
        )
        self.root = Path(self.temporary.name)
        self.root_patch = mock.patch.object(gate, "ROOT", self.root)
        self.root_patch.start()
        self.fixture = PilotGateFixture(self.root)

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temporary.cleanup()

    def test_go_certificate_verifies_against_local_artifacts(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        artifact = self.fixture.write_gate()

        self.assertEqual(artifact["status"], "GO")
        self.assertTrue(artifact["passed"])
        self.assertEqual(
            gate.verify_gate(
                self.fixture.config_path,
                self.fixture.gate_path,
                require_tracked=False,
            ),
            artifact,
        )

    def test_logged_qa_fallback_is_valid_baseline_output(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        jsonl_path = self.root / "results" / "baseline.jsonl"
        records = [
            json.loads(line) for line in jsonl_path.read_text().splitlines()
        ]
        for record in records:
            if record.get("record_type") == "answer":
                record["final_answer_source"] = "qa_fallback"
        jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        meta_path = self.root / "results" / "baseline.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["jsonl_sha256"] = gate.sha256_file(jsonl_path)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        artifact = self.fixture.write_gate()
        self.assertEqual(artifact["status"], "GO")
        self.assertTrue(artifact["passed"])

    def test_unknown_final_answer_source_is_rejected(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        jsonl_path = self.root / "results" / "baseline.jsonl"
        records = [
            json.loads(line) for line in jsonl_path.read_text().splitlines()
        ]
        for record in records:
            if record.get("record_type") == "answer":
                record["final_answer_source"] = "mystery"
        jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        meta_path = self.root / "results" / "baseline.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["jsonl_sha256"] = gate.sha256_file(jsonl_path)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with self.assertRaisesRegex(
            RuntimeError, "invalid final-answer provenance"
        ):
            self.fixture.write_gate()

    def test_stop_certificate_is_rejected(self) -> None:
        self.fixture.write_runs(baseline_correct=False, single_correct=True)
        artifact = self.fixture.write_gate()

        self.assertEqual(artifact["status"], "STOP")
        self.assertFalse(artifact["passed"])
        with self.assertRaisesRegex(RuntimeError, "STOP"):
            gate.verify_gate(self.fixture.config_path, self.fixture.gate_path)

    def test_grounded_followup_guard_requires_a_literal_boolean(self) -> None:
        self.fixture.config["retrieval"][
            "grounded_followup_requires_evidence"
        ] = "true"
        self.fixture.write_config()
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        with self.assertRaisesRegex(RuntimeError, "fingerprint is stale"):
            self.fixture.write_gate()

    def test_gate_is_stale_after_decision_rule_changes(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        self.fixture.write_gate()
        self.fixture.config["pilot"]["decision"][
            "pass_if_multiagent_minus_single_overall_at_least"
        ] = 0.5
        self.fixture.write_config()

        with self.assertRaisesRegex(RuntimeError, "stale"):
            gate.verify_gate(self.fixture.config_path, self.fixture.gate_path)

    def test_missing_gate_is_rejected_with_actionable_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mandatory pilot gate is missing"):
            gate.verify_gate(self.fixture.config_path, self.fixture.gate_path)

    def test_internally_rehashed_modified_gate_is_rejected(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        artifact = self.fixture.write_gate()
        artifact["metrics"][
            "multiagent_minus_single_f1_points_overall"
        ] = 999.0
        semantic = {
            key: value for key, value in artifact.items()
            if key != "artifact_sha256"
        }
        artifact["artifact_sha256"] = gate.content_hash(semantic)
        gate.write_atomic(self.fixture.gate_path, artifact)

        with self.assertRaisesRegex(RuntimeError, "no longer matches.*metrics"):
            gate.verify_gate(self.fixture.config_path, self.fixture.gate_path)

    def test_self_consistent_but_inactive_experiment_fingerprint_is_rejected(self) -> None:
        self.fixture.write_runs(
            baseline_correct=True,
            single_correct=False,
            stale_run_id="baseline",
        )

        with self.assertRaisesRegex(RuntimeError, "experiment fingerprint is stale"):
            gate.check_pilot(self.fixture.config_path)

    def test_self_consistent_v1_pilot_artifact_is_rejected(self) -> None:
        self.fixture.write_runs(
            baseline_correct=True,
            single_correct=False,
            stale_schema_run_id="baseline",
        )

        with self.assertRaisesRegex(RuntimeError, "experiment fingerprint is stale"):
            gate.check_pilot(self.fixture.config_path)

    def test_active_generation_budget_change_rejects_old_pilot(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        with mock.patch.dict(
            prompts.MAX_NEW_TOKENS, {"extractor": 999}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "experiment fingerprint is stale"):
                gate.check_pilot(self.fixture.config_path)

    def test_scored_call_from_another_cohort_is_rejected(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        jsonl_path = self.root / "results" / "baseline.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        records[0]["question_manifest_sha256"] = "wrong-cohort"
        jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        meta_path = self.root / "results" / "baseline.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["jsonl_sha256"] = gate.sha256_file(jsonl_path)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "scored-record cohort mismatch"):
            gate.check_pilot(self.fixture.config_path)

    def test_staged_only_gate_fails_but_identical_head_gate_passes(self) -> None:
        self.fixture.write_runs(baseline_correct=True, single_correct=False)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "pilot-gate@example.invalid"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Pilot Gate Test"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "synthetic pilot inputs"],
            cwd=self.root,
            check=True,
        )

        self.fixture.write_gate()
        subprocess.run(
            ["git", "add", "analysis/pilot_gate.json"],
            cwd=self.root,
            check=True,
        )
        with self.assertRaisesRegex(RuntimeError, "committed unchanged"):
            gate.verify_gate(
                self.fixture.config_path,
                self.fixture.gate_path,
                require_tracked=True,
            )

        subprocess.run(
            ["git", "commit", "-q", "-m", "freeze pilot GO certificate"],
            cwd=self.root,
            check=True,
        )
        verified = gate.verify_gate(
            self.fixture.config_path,
            self.fixture.gate_path,
            require_tracked=True,
        )
        self.assertEqual(verified["status"], "GO")


if __name__ == "__main__":
    unittest.main()
