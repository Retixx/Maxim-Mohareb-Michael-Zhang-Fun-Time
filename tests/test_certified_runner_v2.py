from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import runner
from scripts.certified_runner_v2 import install_certified_runtime


class CertifiedRunnerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_store = runner.JsonlStore
        self.original_gpu = runner._gpu_metadata
        self.original_atomic = runner._write_json_atomic

    def tearDown(self) -> None:
        runner.JsonlStore = self.original_store
        runner._gpu_metadata = self.original_gpu
        runner._write_json_atomic = self.original_atomic
        for name in (
            "_MARAG_TRUE_GPU_METADATA", "_MARAG_CERTIFIED_STORE_CLASS",
        ):
            if hasattr(runner, name):
                delattr(runner, name)

    def test_install_is_idempotent_and_retains_lock_until_explicit_release(self) -> None:
        first = install_certified_runtime(runner)
        second = install_certified_runtime(runner)
        self.assertIs(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.jsonl"
            store = runner.JsonlStore(path).open("session")
            store.write([{"record_type": "answer"}])
            store.durable_flush()
            store.close()
            self.assertTrue(store.lock_path.exists())
            store.release_certified_lock()
            self.assertFalse(store.lock_path.exists())

    def test_child_independently_rejects_mig_even_with_forged_environment(self) -> None:
        runner._gpu_metadata = lambda: {
            "gpu_available": True,
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        }
        install_certified_runtime(runner)
        props = type("Props", (), {
            "name": "NVIDIA A100-SXM4-80GB",
            "uuid": "12345678-1234-1234-1234-123456789abc",
        })()
        environment = {
            "CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc",
            "MARAG_CERTIFIED_GPU_UUID": "GPU-12345678-1234-1234-1234-123456789abc",
            "MARAG_CERTIFIED_GPU_NAME": "NVIDIA A100-SXM4-80GB",
            "MARAG_CERTIFIED_GPU_MEMORY_MIB": "81920.0",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=props),
            patch(
                "subprocess.check_output",
                return_value=(
                    "GPU-12345678-1234-1234-1234-123456789abc, "
                    "NVIDIA A100-SXM4-80GB, Enabled\n"
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "MIG-disabled"),
        ):
            runner._gpu_metadata()


if __name__ == "__main__":
    unittest.main()
