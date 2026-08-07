"""Fail-closed production orchestration for a two-to-six A100 fleet.

The accuracy matrix always uses six frozen logical shards. Physical A100 count
controls concurrency only. All authorized output goes through the certified
runner shim and a shared-results claim ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FROZEN_BASE_COMMIT = "d57b7c9"
FROZEN_SEED = 20260805
FROZEN_WORKERS = 6
FROZEN_PLAN_PATH = (
    ROOT / "config" / "manifests" / "accuracy_static22_w6_seed20260805.json"
)
FROZEN_PLAN_SHA256 = (
    "511e2b9f974400daffe58f29f8905e4dc4b692cb86516903ab982e4eab53b456"
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def validate_checkout() -> str:
    """Require a clean local no-bs checkout at its fetched remote revision."""
    head = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    if branch != "no-bs":
        raise RuntimeError(f"production branch must be no-bs; current={branch!r}")
    try:
        remote = _git_output("rev-parse", "--verify", "refs/remotes/origin/no-bs")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "origin/no-bs is unavailable; fetch and fast-forward it before launch"
        ) from exc
    if remote != head:
        raise RuntimeError(f"HEAD {head} differs from fetched origin/no-bs {remote}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_BASE_COMMIT, head],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            f"HEAD {head} is not descended from required base {FROZEN_BASE_COMMIT}"
        )
    if _git_output("status", "--porcelain"):
        raise RuntimeError("production worktree is dirty; commit exact source first")
    return head


def canonical_full_gpu_uuid(value: object) -> str:
    """Return strict NVIDIA ``GPU-<RFC4122>`` text and reject MIG/malformed IDs."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 16:
            raise RuntimeError("GPU UUID bytes must contain exactly 16 bytes")
        parsed = uuid.UUID(bytes=bytes(value))
        return f"GPU-{parsed}"
    text = str(value or "").strip()
    if text.casefold().startswith("mig-"):
        raise RuntimeError(f"MIG UUID is not a full physical GPU: {text!r}")
    if text.casefold().startswith("gpu-"):
        text = text[4:]
    canonical_pattern = (
        len(text) == 36
        and all(text[index] == "-" for index in (8, 13, 18, 23))
    )
    compact_pattern = len(text) == 32 and "-" not in text
    if not (canonical_pattern or compact_pattern):
        raise RuntimeError(f"malformed full GPU UUID: {value!r}")
    try:
        parsed = uuid.UUID(text)
    except ValueError as exc:
        raise RuntimeError(f"malformed full GPU UUID: {value!r}") from exc
    return f"GPU-{parsed}"


def certify_one_full_a100() -> dict:
    """Cross-check exactly one full A100 through PyTorch and nvidia-smi."""
    import torch

    selectors = [
        item.strip()
        for item in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        if item.strip()
    ]
    if len(selectors) != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if selectors[0].casefold().startswith("mig-"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES selects a MIG instance")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PyTorch must see exactly one CUDA device")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch_uuid_value = getattr(props, "uuid", None)
    if torch_uuid_value is None:
        raise RuntimeError("PyTorch did not expose a certifiable physical GPU UUID")
    torch_uuid = canonical_full_gpu_uuid(torch_uuid_value)
    rows = subprocess.check_output(
        [
            "nvidia-smi", "-i", selectors[0],
            "--query-gpu=uuid,name,memory.total,driver_version,pci.bus_id,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"nvidia-smi returned {len(rows)} GPUs; expected one")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 6:
        raise RuntimeError(f"unexpected nvidia-smi identity row: {rows[0]!r}")
    smi_uuid_value, name, memory_text, driver, pci_bus_id, mig_mode = values
    smi_uuid = canonical_full_gpu_uuid(smi_uuid_value)
    if smi_uuid != torch_uuid:
        raise RuntimeError(
            f"PyTorch/nvidia-smi UUID mismatch: {torch_uuid!r} != {smi_uuid!r}"
        )
    if props.name != name:
        raise RuntimeError(
            f"PyTorch/nvidia-smi name mismatch: {props.name!r} != {name!r}"
        )
    if "A100" not in name or "MIG" in name.upper():
        raise RuntimeError(f"full NVIDIA A100 required; detected {name!r}")
    if mig_mode.casefold() != "disabled":
        raise RuntimeError(f"MIG must be Disabled; detected {mig_mode!r}")
    memory_mib = round(props.total_memory / 1024**2, 1)
    certificate = {
        "gpu_uuid": smi_uuid,
        "gpu_name": name,
        "gpu_total_memory_mib": memory_mib,
        "gpu_nvidia_smi_memory_mib": memory_text,
        "gpu_driver_version": driver,
        "gpu_pci_bus_id": pci_bus_id,
        "gpu_mig_mode_current": mig_mode,
        "cuda_visible_devices_requested": selectors[0],
    }
    # All child processes address the already certified physical device by its
    # canonical UUID, eliminating CUDA-ordinal versus NVML-index ambiguity.
    os.environ["CUDA_VISIBLE_DEVICES"] = smi_uuid
    os.environ["MARAG_CERTIFIED_GPU_UUID"] = smi_uuid
    os.environ["MARAG_CERTIFIED_GPU_NAME"] = name
    os.environ["MARAG_CERTIFIED_GPU_MEMORY_MIB"] = str(memory_mib)
    return certificate


def _load_campaign_tools():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.check_pilot import verify_gate
    from scripts.run_campaign import (
        STATIC_RUNS,
        build_plan,
        completed_run_ids,
        validate_matrix,
    )
    return STATIC_RUNS, build_plan, completed_run_ids, validate_matrix, verify_gate


def load_frozen_accuracy_plan(config: dict) -> dict:
    static_runs, _build, _complete, validate_matrix, _gate = _load_campaign_tools()
    validate_matrix(config)
    if _sha256(FROZEN_PLAN_PATH) != FROZEN_PLAN_SHA256:
        raise RuntimeError("frozen accuracy plan file hash mismatch")
    plan = json.loads(FROZEN_PLAN_PATH.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "accuracy"
        or plan.get("seed") != FROZEN_SEED
        or plan.get("logical_workers") != FROZEN_WORKERS
    ):
        raise RuntimeError("frozen accuracy plan header mismatch")
    expected_order = sorted(static_runs)
    random.Random(FROZEN_SEED).shuffle(expected_order)
    expected_assignments = {
        str(index): expected_order[index::FROZEN_WORKERS]
        for index in range(FROZEN_WORKERS)
    }
    if plan.get("ordered_run_ids") != expected_order:
        raise RuntimeError("frozen accuracy order differs from executable contract")
    if plan.get("assignments") != expected_assignments:
        raise RuntimeError("frozen accuracy assignments differ from executable contract")
    flattened = [run_id for values in expected_assignments.values() for run_id in values]
    if len(flattened) != 22 or set(flattened) != static_runs or len(set(flattened)) != 22:
        raise RuntimeError("accuracy plan must assign every static arm exactly once")
    return plan


def build_phase_plan(config: dict, phase: str, worker_index: int) -> dict:
    _static, build_plan, _complete, validate_matrix, _gate = _load_campaign_tools()
    validate_matrix(config)
    if phase == "accuracy":
        if not 0 <= worker_index < FROZEN_WORKERS:
            raise ValueError("accuracy worker index must be in [0, 6)")
        frozen = load_frozen_accuracy_plan(config)
        ordered = list(frozen["ordered_run_ids"])
        assignments = frozen["assignments"]
        workers = FROZEN_WORKERS
    elif phase == "selected-accuracy":
        if worker_index != 0:
            raise ValueError("selected-accuracy uses worker index 0")
        selected = (config.get("frozen_allocation") or {}).get("execution_run_id")
        if not selected or selected not in (config.get("runs") or {}):
            raise RuntimeError("derived config has no authorized selected execution run")
        ordered = [selected]
        assignments = {"0": ordered}
        workers = 1
    else:
        if worker_index != 0:
            raise ValueError(f"{phase} uses worker index 0")
        plan = build_plan(
            config,
            kind=phase,
            workers=1,
            seed=FROZEN_SEED,
        )
        ordered = list(plan["ordered_run_ids"])
        assignments = plan["assignments"]
        workers = 1
    identity = {
        "schema_version": 1,
        "phase": phase,
        "seed": FROZEN_SEED,
        "logical_workers": workers,
        "ordered_run_ids": ordered,
        "assignments": assignments,
        "frozen_accuracy_plan_sha256": (
            FROZEN_PLAN_SHA256 if phase == "accuracy" else None
        ),
    }
    identity["plan_sha256"] = _content_hash(identity)
    return identity


class PersistentPhaseClaim:
    """Shared-results claim that survives success and detects hard-kill staleness."""

    def __init__(
        self,
        state_dir: Path,
        *,
        phase: str,
        worker_index: int,
        plan: dict,
        git_commit: str,
        environment_lock_sha256: str,
        recover_stale: bool,
    ):
        key = f"{phase}_{plan['plan_sha256'][:16]}_worker{worker_index}"
        self.lock_dir = state_dir / f"{key}.active"
        self.state_path = state_dir / f"{key}.json"
        self.phase = phase
        self.worker_index = worker_index
        self.plan = plan
        self.git_commit = git_commit
        self.environment_lock_sha256 = environment_lock_sha256
        self.recover_stale = recover_stale
        self.attempt_id = uuid.uuid4().hex
        self.completed = False

    def __enter__(self):
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_dir.mkdir()
        except FileExistsError as exc:
            if not self.recover_stale:
                raise RuntimeError(
                    f"phase claim is active or stale: {self.lock_dir}; verify the "
                    "owner is dead before using --recover-stale-claim"
                ) from exc
            stale = self.lock_dir.with_name(
                self.lock_dir.name + ".stale." + datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
            )
            os.replace(self.lock_dir, stale)
            self.lock_dir.mkdir()
        self._write("running")
        return self

    def _write(self, status: str, **extra) -> None:
        _write_json_atomic(self.state_path, {
            "schema_version": 1,
            "status": status,
            "phase": self.phase,
            "worker_index": self.worker_index,
            "attempt_id": self.attempt_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "git_commit": self.git_commit,
            "environment_lock_sha256": self.environment_lock_sha256,
            "plan_sha256": self.plan["plan_sha256"],
            "assigned_run_ids": self.plan["assignments"][str(self.worker_index)],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        })

    def mark_complete(self, completed_run_ids: set[str]) -> None:
        self._write("complete", completed_run_ids=sorted(completed_run_ids))
        self.completed = True

    def __exit__(self, exc_type, exc, _traceback):
        if not self.completed:
            self._write(
                "failed",
                error=(f"{exc_type.__name__}: {exc}" if exc_type else "incomplete"),
            )
        self.lock_dir.rmdir()


def _claim_is_complete(
    state_dir: Path,
    *,
    phase: str,
    worker_index: int,
    plan: dict,
    environment_lock_sha256: str,
) -> bool:
    key = f"{phase}_{plan['plan_sha256'][:16]}_worker{worker_index}"
    path = state_dir / f"{key}.json"
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        state.get("status") == "complete"
        and state.get("phase") == phase
        and state.get("worker_index") == worker_index
        and state.get("plan_sha256") == plan["plan_sha256"]
        and state.get("environment_lock_sha256") == environment_lock_sha256
        and set(state.get("completed_run_ids") or ())
        >= set(plan["assignments"][str(worker_index)])
    )


def ensure_timing_device_ledger(
    state_dir: Path,
    *,
    gpu: dict,
    environment_lock_sha256: str,
) -> None:
    path = state_dir / "timing_device.json"
    expected = {
        "gpu_uuid": gpu["gpu_uuid"],
        "host": socket.gethostname(),
        "environment_lock_sha256": environment_lock_sha256,
    }
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if ledger.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "timing device ledger mismatch: " + ", ".join(mismatches)
            )
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(path.name + f".{os.getpid()}.candidate")
    _write_json_atomic(temporary, payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        ledger = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if ledger.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "concurrent timing device ledger mismatch: " + ", ".join(mismatches)
            )
    finally:
        temporary.unlink(missing_ok=True)


def _runner_command(config_path: Path, run_id: str | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.certified_runner",
        "--config",
        str(config_path),
    ]
    if run_id is not None:
        command.extend(("--run", run_id))
    return command


def _validate_environment(config: dict, config_path: Path) -> tuple[Path, str]:
    lock_path = _repo_path(
        config.get("environment_lock_path", "config/environment.lock.json")
    )
    _run(_runner_command(config_path) + ["--validate-environment-lock"])
    return lock_path, _sha256(lock_path)


def _verify_gate(config: dict, config_path: Path) -> None:
    _static, _build, _complete, _matrix, verify_gate = _load_campaign_tools()
    verify_gate(
        config_path,
        _repo_path(config["pilot"]["gate_artifact"]),
        require_tracked=True,
    )


def _completed(
    config: dict,
    *,
    kind: str,
    environment_lock_sha256: str,
) -> set[str]:
    _static, _build, completed_run_ids, _matrix, _gate = _load_campaign_tools()
    return completed_run_ids(
        config,
        kind=kind,
        expected_environment_lock_sha256=environment_lock_sha256,
    )


def _require_timing_complete(
    config: dict,
    state_dir: Path,
    *,
    environment_lock_sha256: str,
) -> None:
    timing_plan = build_phase_plan(config, "timing", 0)
    if not _claim_is_complete(
        state_dir,
        phase="timing",
        worker_index=0,
        plan=timing_plan,
        environment_lock_sha256=environment_lock_sha256,
    ):
        raise RuntimeError(
            "accuracy is blocked until this config's complete timing plan is "
            "validated in the shared results ledger"
        )


def prepare(
    config_path: Path,
    *,
    container_ref: str,
    container_digest: str,
) -> None:
    _run([
        sys.executable,
        "scripts/prefetch_assets.py",
        "--config",
        str(config_path),
        "--offline-verify-only",
    ])
    _run([sys.executable, "-X", "utf8", "-m", "pytest", "-q"])
    _run(_runner_command(config_path) + [
        "--write-environment-lock",
        "config/environment.lock.json",
        "--container-ref",
        container_ref,
        "--container-digest",
        container_digest,
    ])


def execute_phase(
    config: dict,
    config_path: Path,
    *,
    phase: str,
    worker_index: int,
    git_commit: str,
    gpu: dict,
    recover_stale: bool,
) -> None:
    _lock_path, lock_hash = _validate_environment(config, config_path)
    if phase != "pilot":
        _verify_gate(config, config_path)
    results_dir = _repo_path(config.get("results_dir", "results"))
    state_dir = results_dir / ".fleet_state"
    plan = build_phase_plan(config, phase, worker_index)
    assigned = set(plan["assignments"][str(worker_index)])
    artifact_kind = (
        "accuracy" if phase in {"accuracy", "selected-accuracy"} else phase
    )
    if phase == "timing":
        ensure_timing_device_ledger(
            state_dir,
            gpu=gpu,
            environment_lock_sha256=lock_hash,
        )
    elif phase in {"accuracy", "selected-accuracy"}:
        _require_timing_complete(
            config,
            state_dir,
            environment_lock_sha256=lock_hash,
        )

    status_path = state_dir / (
        f"{phase}_{plan['plan_sha256'][:16]}_worker{worker_index}.plan.json"
    )
    _write_json_atomic(status_path, {
        **plan,
        "worker_index": worker_index,
        "assigned_run_ids": sorted(assigned),
        "git_commit": git_commit,
        "environment_lock_sha256": lock_hash,
    })
    completed_before = _completed(
        config,
        kind=artifact_kind,
        environment_lock_sha256=lock_hash,
    )
    if _claim_is_complete(
        state_dir,
        phase=phase,
        worker_index=worker_index,
        plan=plan,
        environment_lock_sha256=lock_hash,
    ):
        if not assigned <= completed_before:
            raise RuntimeError(
                "shared fleet ledger says complete but finalized artifacts are missing"
            )
        print(f"{phase} worker {worker_index} already complete")
        return

    with PersistentPhaseClaim(
        state_dir,
        phase=phase,
        worker_index=worker_index,
        plan=plan,
        git_commit=git_commit,
        environment_lock_sha256=lock_hash,
        recover_stale=recover_stale,
    ) as claim:
        for run_id in plan["assignments"][str(worker_index)]:
            if run_id in completed_before:
                print(f"skip finalized {artifact_kind} artifact: {run_id}")
                continue
            command = _runner_command(config_path, run_id)
            if phase == "pilot":
                command.append("--pilot-mode")
            elif phase == "timing":
                command.append("--timing-mode")
            _run(command)
        if phase == "pilot":
            _run([
                sys.executable,
                "scripts/check_pilot.py",
                "--config",
                str(config_path),
            ])
        completed_after = _completed(
            config,
            kind=artifact_kind,
            environment_lock_sha256=lock_hash,
        )
        if not assigned <= completed_after:
            raise RuntimeError(
                f"phase returned without finalized artifacts: {sorted(assigned-completed_after)}"
            )
        claim.mark_complete(completed_after & assigned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "pilot", "timing", "accuracy", "selected-accuracy"),
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--container-ref", required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument(
        "--recover-stale-claim",
        action="store_true",
        help="take over a stale shared claim only after its old process is verified dead",
    )
    args = parser.parse_args()

    os.environ.update(OFFLINE_ENVIRONMENT)
    os.environ["EXPERIMENT_CONTAINER_REF"] = args.container_ref
    os.environ["EXPERIMENT_CONTAINER_DIGEST"] = args.container_digest
    git_commit = validate_checkout()
    gpu = certify_one_full_a100()
    print(json.dumps({
        "phase": args.phase,
        "git_commit": git_commit,
        "gpu": gpu,
    }, indent=2))
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.phase == "prepare":
        if args.worker_index != 0 or args.recover_stale_claim:
            parser.error("prepare uses worker index 0 and no stale-claim recovery")
        prepare(
            config_path,
            container_ref=args.container_ref,
            container_digest=args.container_digest,
        )
        print("commit and push config/environment.lock.json to no-bs")
        return 0
    execute_phase(
        config,
        config_path,
        phase=args.phase,
        worker_index=args.worker_index,
        git_commit=git_commit,
        gpu=gpu,
        recover_stale=args.recover_stale_claim,
    )
    if args.phase == "pilot":
        print("commit and push the unchanged analysis/pilot_gate.json to no-bs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
