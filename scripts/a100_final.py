"""Final public A100 fleet command with adversarial infrastructure guards."""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from scripts import a100_production as base
from scripts.a100_production import canonical_full_gpu_uuid


def _runner_command(config_path: Path, run_id: str | None = None) -> list[str]:
    command = [
        base.sys.executable,
        "-m",
        "scripts.certified_runner_v2",
        "--config",
        str(config_path),
    ]
    if run_id is not None:
        command.extend(("--run", run_id))
    return command


def _prepare(
    config_path: Path,
    *,
    container_ref: str,
    container_digest: str,
) -> None:
    config = base.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock_path = base._repo_path(
        config.get("environment_lock_path", "config/environment.lock.json")
    )
    base._run([
        base.sys.executable,
        "scripts/prefetch_assets.py",
        "--config",
        str(config_path),
        "--offline-verify-only",
    ])
    base._run([base.sys.executable, "-X", "utf8", "-m", "pytest", "-q"])
    base._run(_runner_command(config_path) + [
        "--write-environment-lock",
        str(lock_path),
        "--container-ref",
        container_ref,
        "--container-digest",
        container_digest,
    ])


def _atomic_identity_file(path: Path, expected: dict, label: str) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatches:
            raise RuntimeError(f"{label} mismatch: " + ", ".join(mismatches))
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    candidate = path.with_name(path.name + f".{os.getpid()}.candidate")
    base._write_json_atomic(candidate, payload)
    try:
        os.link(candidate, path)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if existing.get(key) != value]
        if mismatches:
            raise RuntimeError(
                f"concurrent {label} mismatch: " + ", ".join(mismatches)
            )
        payload = existing
    finally:
        candidate.unlink(missing_ok=True)
    return payload


def _timing_ledger(
    state_dir: Path,
    *,
    gpu: dict,
    environment_lock_sha256: str,
) -> dict:
    return _atomic_identity_file(
        state_dir / "timing_device.json",
        {
            "gpu_uuid": canonical_full_gpu_uuid(gpu["gpu_uuid"]),
            "environment_lock_sha256": environment_lock_sha256,
        },
        "timing device ledger",
    )


def _static_campaign_ledger(
    config: dict,
    state_dir: Path,
    *,
    git_commit: str,
    environment_lock_sha256: str,
) -> dict:
    gate_path = base._repo_path(config["pilot"]["gate_artifact"])
    return _atomic_identity_file(
        state_dir / "static_campaign.json",
        {
            "git_commit": git_commit,
            "environment_lock_sha256": environment_lock_sha256,
            "pilot_gate_sha256": base._sha256(gate_path),
            "frozen_accuracy_plan_sha256": base.FROZEN_PLAN_SHA256,
        },
        "static campaign ledger",
    )


def _valid_meta_candidates(
    config: dict,
    *,
    run_id: str,
    kind: str,
    environment_lock_sha256: str,
) -> list[tuple[Path, dict]]:
    results_dir = base._repo_path(config.get("results_dir", "results"))
    cohort = config["timing"] if kind == "timing" else config["dataset"]
    out = []
    for meta_path in results_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            meta.get("run_id") != run_id
            or meta.get("metadata_complete") is not True
            or bool(meta.get("timing_mode")) != (kind == "timing")
            or bool(meta.get("pilot_mode"))
            or meta.get("question_ids_sha256") != cohort.get("manifest_sha256")
            or meta.get("environment_lock_sha256") != environment_lock_sha256
        ):
            continue
        jsonl_path = meta_path.with_name(
            meta_path.name.removesuffix(".meta.json") + ".jsonl"
        )
        if (
            not jsonl_path.exists()
            or not meta.get("jsonl_sha256")
            or base._sha256(jsonl_path) != meta["jsonl_sha256"]
        ):
            continue
        out.append((meta_path, meta))
    return out


def _validate_timing_artifacts(
    config: dict,
    *,
    plan: dict,
    environment_lock_sha256: str,
    require_all: bool,
    expected_git_commit: str | None,
) -> set[str]:
    completed = base._completed(
        config,
        kind="timing",
        environment_lock_sha256=environment_lock_sha256,
    )
    expected = set(plan["assignments"]["0"])
    if require_all and not expected <= completed:
        raise RuntimeError(
            f"timing artifacts are incomplete: {sorted(expected-completed)}"
        )
    state_dir = base._repo_path(config.get("results_dir", "results")) / ".fleet_state"
    ledger_path = state_dir / "timing_device.json"
    if not ledger_path.exists():
        if completed & expected:
            raise RuntimeError("timing artifacts exist without a timing device ledger")
        return completed
    ledger_uuid = canonical_full_gpu_uuid(
        json.loads(ledger_path.read_text(encoding="utf-8"))["gpu_uuid"]
    )
    for run_id in sorted(completed & expected):
        candidates = _valid_meta_candidates(
            config,
            run_id=run_id,
            kind="timing",
            environment_lock_sha256=environment_lock_sha256,
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"timing run {run_id} has {len(candidates)} finalized artifact candidates"
            )
        meta_path, meta = candidates[0]
        sessions = meta.get("execution_sessions") or []
        if not sessions or any(not session.get("gpu_uuid") for session in sessions):
            raise RuntimeError(f"timing artifact lacks certified GPU UUID: {meta_path}")
        session_uuids = {
            canonical_full_gpu_uuid(session["gpu_uuid"]) for session in sessions
        }
        if session_uuids != {ledger_uuid}:
            raise RuntimeError(
                f"timing artifact GPU drift for {run_id}: {sorted(session_uuids)}"
            )
        if expected_git_commit is not None and meta.get("git_commit") != expected_git_commit:
            raise RuntimeError(f"static timing Git commit drift for {run_id}")
    return completed


def _validate_static_accuracy_artifacts(
    config: dict,
    *,
    assigned: set[str],
    environment_lock_sha256: str,
    expected_git_commit: str,
) -> None:
    completed = base._completed(
        config,
        kind="accuracy",
        environment_lock_sha256=environment_lock_sha256,
    )
    for run_id in sorted(completed & assigned):
        candidates = _valid_meta_candidates(
            config,
            run_id=run_id,
            kind="accuracy",
            environment_lock_sha256=environment_lock_sha256,
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"accuracy run {run_id} has {len(candidates)} finalized candidates"
            )
        if candidates[0][1].get("git_commit") != expected_git_commit:
            raise RuntimeError(f"static accuracy Git commit drift for {run_id}")


def _require_timing_complete(
    config: dict,
    state_dir: Path,
    *,
    environment_lock_sha256: str,
) -> None:
    plan = base.build_phase_plan(config, "timing", 0)
    if not base._claim_is_complete(
        state_dir,
        phase="timing",
        worker_index=0,
        plan=plan,
        environment_lock_sha256=environment_lock_sha256,
    ):
        raise RuntimeError("accuracy requires a complete shared timing claim")
    static = not bool(config.get("frozen_allocation"))
    static_ledger = state_dir / "static_campaign.json"
    expected_commit = None
    if static:
        if not static_ledger.exists():
            raise RuntimeError("static campaign Git-commit ledger is missing")
        expected_commit = json.loads(
            static_ledger.read_text(encoding="utf-8")
        )["git_commit"]
    _validate_timing_artifacts(
        config,
        plan=plan,
        environment_lock_sha256=environment_lock_sha256,
        require_all=True,
        expected_git_commit=expected_commit,
    )


class NonRecoveringClaim(base.PersistentPhaseClaim):
    def __enter__(self):
        if self.recover_stale:
            raise RuntimeError(
                "automatic stale-claim takeover is disabled; after proving the old "
                "process dead, quarantine the exact .active directory and rerun "
                "without --recover-stale-claim"
            )
        return super().__enter__()


@contextmanager
def _global_timing_lock(state_dir: Path):
    path = state_dir / "timing_global.active"
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"another timing plan is active or stale: {path}; never overlap timing"
        ) from exc
    base._write_json_atomic(path / "owner.json", {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    try:
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def install_final_guards() -> None:
    if getattr(base, "_MARAG_FINAL_GUARDS_INSTALLED", False):
        return
    original_execute = base.execute_phase

    def guarded_execute(
        config: dict,
        config_path: Path,
        *,
        phase: str,
        worker_index: int,
        git_commit: str,
        gpu: dict,
        recover_stale: bool,
    ) -> None:
        lock_path = base._repo_path(
            config.get("environment_lock_path", "config/environment.lock.json")
        )
        if not lock_path.exists():
            raise RuntimeError("environment lock is missing")
        lock_hash = base._sha256(lock_path)
        results_dir = base._repo_path(config.get("results_dir", "results"))
        state_dir = results_dir / ".fleet_state"
        is_static = not bool(config.get("frozen_allocation"))
        static_ledger = None
        if is_static and phase in {"timing", "accuracy"}:
            static_ledger = _static_campaign_ledger(
                config,
                state_dir,
                git_commit=git_commit,
                environment_lock_sha256=lock_hash,
            )
        if phase == "timing":
            _timing_ledger(
                state_dir,
                gpu=gpu,
                environment_lock_sha256=lock_hash,
            )
            plan = base.build_phase_plan(config, "timing", 0)
            _validate_timing_artifacts(
                config,
                plan=plan,
                environment_lock_sha256=lock_hash,
                require_all=False,
                expected_git_commit=(
                    static_ledger["git_commit"] if static_ledger else None
                ),
            )
            with _global_timing_lock(state_dir):
                original_execute(
                    config,
                    config_path,
                    phase=phase,
                    worker_index=worker_index,
                    git_commit=git_commit,
                    gpu=gpu,
                    recover_stale=recover_stale,
                )
            _validate_timing_artifacts(
                config,
                plan=plan,
                environment_lock_sha256=lock_hash,
                require_all=True,
                expected_git_commit=(
                    static_ledger["git_commit"] if static_ledger else None
                ),
            )
            return
        if phase == "accuracy":
            plan = base.build_phase_plan(config, "accuracy", worker_index)
            assigned = set(plan["assignments"][str(worker_index)])
            _validate_static_accuracy_artifacts(
                config,
                assigned=assigned,
                environment_lock_sha256=lock_hash,
                expected_git_commit=static_ledger["git_commit"],
            )
            original_execute(
                config,
                config_path,
                phase=phase,
                worker_index=worker_index,
                git_commit=git_commit,
                gpu=gpu,
                recover_stale=recover_stale,
            )
            _validate_static_accuracy_artifacts(
                config,
                assigned=assigned,
                environment_lock_sha256=lock_hash,
                expected_git_commit=static_ledger["git_commit"],
            )
            return
        original_execute(
            config,
            config_path,
            phase=phase,
            worker_index=worker_index,
            git_commit=git_commit,
            gpu=gpu,
            recover_stale=recover_stale,
        )

    base._runner_command = _runner_command
    base.prepare = _prepare
    base.ensure_timing_device_ledger = _timing_ledger
    base._require_timing_complete = _require_timing_complete
    base.PersistentPhaseClaim = NonRecoveringClaim
    base.execute_phase = guarded_execute
    base._MARAG_FINAL_GUARDS_INSTALLED = True


def main() -> int:
    install_final_guards()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
