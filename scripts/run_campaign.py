"""Create and optionally execute a deterministic multi-A100 campaign plan.

Examples::

    python scripts/run_campaign.py --kind accuracy --workers 4
    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py --kind accuracy \
        --workers 4 --worker-index 0 --execute
    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py --kind timing --execute

Every accuracy arm is assigned whole to one worker. Timing is restricted to one
worker and excludes the five 0.5B appendix-floor arms by contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATIC_RUNS = {
    "baseline",
    *(f"{role}_{tier}" for tier in ("8bit", "4bit", "small", "tiny")
      for role in ("planner", "stepdef", "extractor", "qa")),
    "ma_uniform_8bit", "ma_uniform_4bit", "ma_uniform_small", "ma_uniform_tiny",
    "single_fp16",
}
TINY_RUNS = {
    "planner_tiny", "stepdef_tiny", "extractor_tiny", "qa_tiny", "ma_uniform_tiny"
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_run_ids(config: dict, *, kind: str) -> set[str]:
    """Find finalized, hash-valid artifacts so campaign restarts skip them."""
    results_dir = Path(config.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    dataset = config.get("dataset") or {}
    timing = config.get("timing") or {}
    expected_n = int(timing["n"] if kind == "timing" else dataset["n"])
    expected_seed = int(
        timing.get("seed", dataset["eval_seed"])
        if kind == "timing" else dataset["eval_seed"]
    )
    completed: set[str] = set()
    for meta_path in results_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if bool(meta.get("timing_mode")) != (kind == "timing"):
            continue
        if meta.get("metadata_complete") is not True:
            continue
        try:
            identity_matches = (
                int(meta.get("n", -1)) == expected_n
                and int(meta.get("seed", -1)) == expected_seed
            )
        except (TypeError, ValueError):
            identity_matches = False
        if not identity_matches:
            continue
        jsonl_path = meta_path.with_name(
            meta_path.name.removesuffix(".meta.json") + ".jsonl"
        )
        if (
            not jsonl_path.exists()
            or not meta.get("jsonl_sha256")
            or _sha256(jsonl_path) != meta["jsonl_sha256"]
        ):
            continue
        if isinstance(meta.get("run_id"), str):
            completed.add(meta["run_id"])
    return completed


def validate_matrix(config: dict) -> None:
    runs = set(config.get("runs") or {})
    if runs != STATIC_RUNS:
        raise RuntimeError(
            f"static matrix must be the exact 22-run contract; missing={sorted(STATIC_RUNS-runs)}, "
            f"extra={sorted(runs-STATIC_RUNS)}"
        )
    selector = config.get("allocation_selector") or {}
    if selector.get("candidate_allocation_count") != 256:
        raise RuntimeError("allocation selector must declare exactly 256 candidates")
    if set(selector.get("candidates") or {}) != {
        "base_fp16", "base_8bit", "base_4bit", "small_fp16"
    }:
        raise RuntimeError("selector candidate set changed or includes the 0.5B floor")
    generation = config.get("generation") or {}
    if generation.get("batch_size") != 32 or generation.get("min_batch_size") != 32:
        raise RuntimeError("production batch size must be pinned fail-closed at 32")


def build_plan(config: dict, *, kind: str, workers: int, seed: int) -> dict:
    validate_matrix(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    if kind == "timing" and workers != 1:
        raise ValueError("timing must run on exactly one reserved A100 worker")
    if kind == "accuracy":
        run_ids = sorted(STATIC_RUNS)
    else:
        timing = config.get("timing") or {}
        run_ids = list(timing.get("run_ids") or ())
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise RuntimeError("timing.run_ids must be a non-empty unique list")
        if set(run_ids) & TINY_RUNS:
            raise RuntimeError("0.5B floor arms are prohibited from timing")
        if not set(run_ids) <= STATIC_RUNS:
            raise RuntimeError("timing.run_ids contains an unknown/static-external arm")
    random.Random(seed).shuffle(run_ids)
    assignments = {
        str(worker): run_ids[worker::workers] for worker in range(workers)
    }
    return {
        "schema_version": 1,
        "kind": kind,
        "seed": seed,
        "workers": workers,
        "ordered_run_ids": run_ids,
        "assignments": assignments,
        "tiny_runs_excluded_from_timing": kind != "timing" or not bool(set(run_ids) & TINY_RUNS),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--kind", choices=("accuracy", "timing"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--execute", action="store_true", help="run assigned arms; default is plan-only")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan = build_plan(config, kind=args.kind, workers=args.workers, seed=args.seed)
    if not 0 <= args.worker_index < args.workers:
        parser.error("--worker-index must be in [0, workers)")
    plan["config_path"] = str(config_path)
    plan["config_sha256"] = _sha256(config_path)
    default_plan = ROOT / "logs" / f"{args.kind}_seed{args.seed}_w{args.workers}.plan.json"
    plan_path = Path(args.plan).resolve() if args.plan else default_plan
    _write_json_atomic(plan_path, plan)

    assigned = plan["assignments"][str(args.worker_index)]
    already_complete = completed_run_ids(config, kind=args.kind)
    plan["completed_before_start"] = sorted(already_complete)
    _write_json_atomic(plan_path, plan)
    print(f"plan: {plan_path}")
    print(f"worker {args.worker_index}/{args.workers}: {', '.join(assigned)}")
    if not args.execute:
        return 0
    for run_id in assigned:
        if run_id in already_complete:
            print(f"skip finalized {args.kind} artifact: {run_id}")
            continue
        command = [
            sys.executable, "-m", "src.runner", "--config", str(config_path), "--run", run_id
        ]
        if args.kind == "timing":
            command.append("--timing-mode")
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
