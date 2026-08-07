"""Independently certified runner shim with finalization lock hardening."""

from __future__ import annotations

import os
import subprocess

from scripts.a100_production import canonical_full_gpu_uuid


OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _independently_certified_gpu_metadata(runner) -> dict:
    certified_uuid = canonical_full_gpu_uuid(
        os.environ.get("MARAG_CERTIFIED_GPU_UUID")
    )
    certified_name = os.environ.get("MARAG_CERTIFIED_GPU_NAME")
    certified_memory = os.environ.get("MARAG_CERTIFIED_GPU_MEMORY_MIB")
    if not certified_name or "A100" not in certified_name:
        raise RuntimeError("certified GPU name is missing or is not an A100")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != certified_uuid:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must equal the certified physical GPU UUID"
        )
    if not runner.torch.cuda.is_available() or runner.torch.cuda.device_count() != 1:
        raise RuntimeError("certified runner requires exactly one visible CUDA device")
    props = runner.torch.cuda.get_device_properties(0)
    torch_uuid_value = getattr(props, "uuid", None)
    if torch_uuid_value is None:
        raise RuntimeError("PyTorch GPU UUID is unavailable in certified runner")
    torch_uuid = canonical_full_gpu_uuid(torch_uuid_value)
    if torch_uuid != certified_uuid or props.name != certified_name:
        raise RuntimeError("child PyTorch GPU identity differs from parent certificate")
    rows = subprocess.check_output(
        [
            "nvidia-smi", "-i", certified_uuid,
            "--query-gpu=uuid,name,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("child nvidia-smi did not return exactly one GPU")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 3:
        raise RuntimeError(f"unexpected child nvidia-smi row: {rows[0]!r}")
    smi_uuid = canonical_full_gpu_uuid(values[0])
    if smi_uuid != certified_uuid or values[1] != certified_name:
        raise RuntimeError("child nvidia-smi identity differs from parent certificate")
    if values[2].casefold() != "disabled" or "MIG" in values[1].upper():
        raise RuntimeError("child GPU is not a full MIG-disabled A100")

    metadata = runner._MARAG_TRUE_GPU_METADATA()
    if metadata.get("gpu_available") is not True:
        raise RuntimeError("runner lost the certified CUDA device")
    raw_uuid = metadata.get("gpu_uuid")
    if raw_uuid is None or canonical_full_gpu_uuid(raw_uuid) != certified_uuid:
        raise RuntimeError("runner metadata UUID differs from independent certificate")
    if metadata.get("gpu_name") != certified_name:
        raise RuntimeError("runner metadata name differs from independent certificate")
    metadata.update({
        "gpu_uuid": certified_uuid,
        "gpu_name": certified_name,
        "gpu_total_memory_mib": (
            float(certified_memory) if certified_memory is not None
            else metadata.get("gpu_total_memory_mib")
        ),
        "gpu_visible_count": 1,
        "gpu_identity_source": "independent_pytorch_plus_nvidia_smi_certificate",
        "gpu_identity_consistent": True,
        "gpu_nvidia_smi_verified": True,
        "gpu_mig_mode_current": "Disabled",
        "gpu_is_mig": False,
    })
    return metadata


def install_certified_runtime(runner):
    """Install once per process and return the hardened store class."""
    existing = getattr(runner, "_MARAG_CERTIFIED_STORE_CLASS", None)
    if existing is not None:
        return existing
    original_store = runner.JsonlStore

    class CertifiedJsonlStore(original_store):
        instances: list["CertifiedJsonlStore"] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.instances.append(self)

        def close(self) -> None:
            if self._fh:
                self.flush()
                self._fh.close()
                self._fh = None

        def release_certified_lock(self) -> None:
            try:
                if self._fh:
                    self.flush()
                    self._fh.close()
                    self._fh = None
            finally:
                if self._owns_lock:
                    self.lock_path.unlink(missing_ok=True)
                    self._owns_lock = False

    runner._MARAG_TRUE_GPU_METADATA = runner._gpu_metadata
    runner._gpu_metadata = lambda: _independently_certified_gpu_metadata(runner)
    runner.JsonlStore = CertifiedJsonlStore
    runner._MARAG_CERTIFIED_STORE_CLASS = CertifiedJsonlStore

    original_atomic_write = runner._write_json_atomic

    def durable_atomic_write(path, payload):
        original_atomic_write(path, payload)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    runner._write_json_atomic = durable_atomic_write
    return CertifiedJsonlStore


def main() -> int:
    os.environ.update(OFFLINE_ENVIRONMENT)
    from src import runner

    certified_store = install_certified_runtime(runner)
    try:
        runner.main()
    finally:
        first_error = None
        for store in reversed(certified_store.instances):
            try:
                store.release_certified_lock()
            except Exception as exc:  # preserve cleanup evidence after all attempts
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
