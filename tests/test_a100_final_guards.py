from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import a100_final


class A100FinalGuardsTests(unittest.TestCase):
    def test_public_runner_command_uses_v2_certified_shim(self) -> None:
        command = a100_final._runner_command(Path("config/experiment.yaml"), "baseline")
        self.assertIn("scripts.certified_runner_v2", command)
        self.assertEqual(command[-2:], ["--run", "baseline"])

    def test_automatic_stale_claim_recovery_is_disabled(self) -> None:
        plan = {"plan_sha256": "a" * 64, "assignments": {"0": ["baseline"]}}
        with tempfile.TemporaryDirectory() as directory:
            claim = a100_final.NonRecoveringClaim(
                Path(directory), phase="accuracy", worker_index=0, plan=plan,
                git_commit="commit", environment_lock_sha256="lock",
                recover_stale=True,
            )
            with self.assertRaisesRegex(RuntimeError, "automatic stale-claim"):
                claim.__enter__()

    def test_global_timing_lock_rejects_overlapping_plans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            with a100_final._global_timing_lock(state_dir):
                with self.assertRaisesRegex(RuntimeError, "another timing plan"):
                    with a100_final._global_timing_lock(state_dir):
                        pass

    def test_timing_artifact_gpu_must_match_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            state = results / ".fleet_state"
            state.mkdir()
            (state / "timing_device.json").write_text(json.dumps({
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "environment_lock_sha256": "lock",
            }), encoding="utf-8")
            jsonl = results / "baseline_timing.jsonl"
            jsonl.write_text('{"record_type":"batch"}\n', encoding="utf-8")
            (results / "baseline_timing.meta.json").write_text(json.dumps({
                "run_id": "baseline",
                "metadata_complete": True,
                "timing_mode": True,
                "pilot_mode": False,
                "question_ids_sha256": "manifest",
                "environment_lock_sha256": "lock",
                "jsonl_sha256": a100_final.base._sha256(jsonl),
                "execution_sessions": [{
                    "gpu_uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                }],
                "git_commit": "commit",
            }), encoding="utf-8")
            config = {
                "results_dir": str(results),
                "timing": {"manifest_sha256": "manifest"},
            }
            with (
                patch.object(a100_final.base, "_completed", return_value={"baseline"}),
                self.assertRaisesRegex(RuntimeError, "GPU drift"),
            ):
                a100_final._validate_timing_artifacts(
                    config,
                    plan={"assignments": {"0": ["baseline"]}},
                    environment_lock_sha256="lock",
                    require_all=True,
                    expected_git_commit="commit",
                )

    def test_timing_claim_is_not_enough_without_revalidated_artifacts(self) -> None:
        plan = {"plan_sha256": "b" * 64, "assignments": {"0": ["baseline"]}}
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            (state_dir / "timing_bbbbbbbbbbbbbbbb_worker0.json").write_text(
                json.dumps({
                    "status": "complete",
                    "phase": "timing",
                    "worker_index": 0,
                    "plan_sha256": plan["plan_sha256"],
                    "environment_lock_sha256": "lock",
                    "completed_run_ids": ["baseline"],
                }),
                encoding="utf-8",
            )
            config = {"frozen_allocation": {"execution_run_id": "baseline"}}
            with (
                patch.object(a100_final.base, "build_phase_plan", return_value=plan),
                patch.object(
                    a100_final,
                    "_validate_timing_artifacts",
                    side_effect=RuntimeError("timing artifacts are incomplete"),
                ) as validate,
                self.assertRaisesRegex(RuntimeError, "timing artifacts are incomplete"),
            ):
                a100_final._require_timing_complete(
                    config, state_dir, environment_lock_sha256="lock"
                )
            validate.assert_called_once()

    def test_static_campaign_ledger_rejects_commit_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            gate = Path(directory) / "gate.json"
            gate.write_text("{}\n", encoding="utf-8")
            config = {"pilot": {"gate_artifact": str(gate)}}
            a100_final._static_campaign_ledger(
                config, state_dir, git_commit="commit-a",
                environment_lock_sha256="lock",
            )
            with self.assertRaisesRegex(RuntimeError, "static campaign ledger mismatch"):
                a100_final._static_campaign_ledger(
                    config, state_dir, git_commit="commit-b",
                    environment_lock_sha256="lock",
                )


if __name__ == "__main__":
    unittest.main()
