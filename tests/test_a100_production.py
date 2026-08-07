from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.a100_production import (
    FROZEN_PLAN_SHA256,
    PersistentPhaseClaim,
    _claim_is_complete,
    build_phase_plan,
    canonical_full_gpu_uuid,
    certify_one_full_a100,
    ensure_timing_device_ledger,
    load_frozen_accuracy_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class _CudaProperties:
    name = "NVIDIA A100-SXM4-80GB"
    uuid = "12345678-1234-1234-1234-123456789abc"
    total_memory = 80 * 1024**3


class A100ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8")
        )

    def test_uuid_parser_is_strict_and_canonical(self) -> None:
        canonical = "GPU-12345678-1234-1234-1234-123456789abc"
        self.assertEqual(canonical_full_gpu_uuid(canonical), canonical)
        self.assertEqual(
            canonical_full_gpu_uuid("12345678123412341234123456789abc"),
            canonical,
        )
        self.assertEqual(
            canonical_full_gpu_uuid(uuid.UUID(canonical[4:]).bytes),
            canonical,
        )
        for malformed in (None, "", "garbage", "GPU-a-b-a-e", "MIG-1234"):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(RuntimeError, "GPU UUID|MIG UUID|malformed"):
                    canonical_full_gpu_uuid(malformed)

    @patch("torch.cuda.get_device_properties", return_value=_CudaProperties())
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    @patch(
        "subprocess.check_output",
        return_value=(
            "GPU-12345678-1234-1234-1234-123456789abc, "
            "NVIDIA A100-SXM4-80GB, 81920, 570.00, 00000000:01:00.0, Disabled\n"
        ),
    )
    def test_gpu_certificate_rewrites_child_visibility_to_canonical_uuid(
        self, _check_output, _available, _count, _current, _properties
    ) -> None:
        environment = {"CUDA_VISIBLE_DEVICES": "3"}
        with patch.dict(os.environ, environment, clear=False):
            certificate = certify_one_full_a100()
            self.assertEqual(
                os.environ["CUDA_VISIBLE_DEVICES"], certificate["gpu_uuid"]
            )
            self.assertEqual(
                os.environ["MARAG_CERTIFIED_GPU_UUID"], certificate["gpu_uuid"]
            )

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_gpu_certificate_rejects_missing_pytorch_uuid(
        self, _available, _count, _current, properties
    ) -> None:
        properties.return_value = type("Props", (), {
            "name": "NVIDIA A100-SXM4-80GB",
            "total_memory": 80 * 1024**3,
        })()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "certifiable physical GPU UUID"):
                certify_one_full_a100()

    def test_frozen_plan_is_exact_six_shard_22_arm_contract(self) -> None:
        plan = load_frozen_accuracy_plan(self.config)
        flattened = [
            run_id
            for index in range(6)
            for run_id in plan["assignments"][str(index)]
        ]
        self.assertEqual(len(flattened), 22)
        self.assertEqual(len(set(flattened)), 22)
        self.assertEqual(len(FROZEN_PLAN_SHA256), 64)
        built = build_phase_plan(self.config, "accuracy", 5)
        self.assertEqual(built["logical_workers"], 6)
        self.assertEqual(
            built["assignments"]["5"], plan["assignments"]["5"]
        )
        with self.assertRaisesRegex(ValueError, "\[0, 6\)"):
            build_phase_plan(self.config, "accuracy", 6)

    def test_selected_accuracy_is_single_authorized_run(self) -> None:
        config = copy.deepcopy(self.config)
        config["frozen_allocation"] = {"execution_run_id": "baseline"}
        plan = build_phase_plan(config, "selected-accuracy", 0)
        self.assertEqual(plan["assignments"], {"0": ["baseline"]})

    def test_shared_claim_rejects_duplicate_and_persists_completion(self) -> None:
        plan = {
            "plan_sha256": "a" * 64,
            "assignments": {"0": ["baseline"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            first = PersistentPhaseClaim(
                state_dir,
                phase="accuracy",
                worker_index=0,
                plan=plan,
                git_commit="commit",
                environment_lock_sha256="lock",
                recover_stale=False,
            )
            duplicate = PersistentPhaseClaim(
                state_dir,
                phase="accuracy",
                worker_index=0,
                plan=plan,
                git_commit="commit",
                environment_lock_sha256="lock",
                recover_stale=False,
            )
            with first:
                with self.assertRaisesRegex(RuntimeError, "active or stale"):
                    duplicate.__enter__()
                first.mark_complete({"baseline"})
            self.assertTrue(_claim_is_complete(
                state_dir,
                phase="accuracy",
                worker_index=0,
                plan=plan,
                environment_lock_sha256="lock",
            ))
            self.assertFalse(first.lock_dir.exists())

    def test_stale_claim_requires_explicit_recovery(self) -> None:
        plan = {"plan_sha256": "b" * 64, "assignments": {"0": ["baseline"]}}
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            blocked = PersistentPhaseClaim(
                state_dir,
                phase="timing",
                worker_index=0,
                plan=plan,
                git_commit="commit",
                environment_lock_sha256="lock",
                recover_stale=False,
            )
            blocked.lock_dir.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "recover-stale-claim"):
                blocked.__enter__()
            recovered = PersistentPhaseClaim(
                state_dir,
                phase="timing",
                worker_index=0,
                plan=plan,
                git_commit="commit",
                environment_lock_sha256="lock",
                recover_stale=True,
            )
            with recovered:
                recovered.mark_complete({"baseline"})
            self.assertTrue(list(state_dir.glob("*.active.stale.*")))

    def test_timing_device_ledger_pins_uuid_host_and_environment(self) -> None:
        gpu = {"gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc"}
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            ensure_timing_device_ledger(
                state_dir, gpu=gpu, environment_lock_sha256="lock"
            )
            ensure_timing_device_ledger(
                state_dir, gpu=gpu, environment_lock_sha256="lock"
            )
            different = {"gpu_uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}
            with self.assertRaisesRegex(RuntimeError, "timing device ledger mismatch"):
                ensure_timing_device_ledger(
                    state_dir, gpu=different, environment_lock_sha256="lock"
                )
            ledger = json.loads(
                (state_dir / "timing_device.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ledger["environment_lock_sha256"], "lock")


if __name__ == "__main__":
    unittest.main()
