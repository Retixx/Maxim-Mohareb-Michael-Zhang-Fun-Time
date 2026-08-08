"""Create and optionally execute a deterministic multi-A100 campaign plan.

Examples::

    python scripts/run_campaign.py --kind accuracy --workers 4
    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py --kind accuracy \
        --workers 4 --worker-index 0 --execute
    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py --kind timing --execute

Every accuracy arm is assigned whole to one worker. Timing is restricted to one
worker and the prespecified matrix excludes the five 0.5B appendix-floor arms;
a frozen derived config may add exactly one selected tiny system for separately
labelled post-selection timing.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_pilot import verify_gate  # noqa: E402
from src.contracts import CAMPAIGN_PLAN_SCHEMA_VERSION, EXPERIMENT_SCHEMA  # noqa: E402
from src import prompts  # noqa: E402
from src.pipeline import load_id_manifest  # noqa: E402
from src.runner import resolve_treatments, validate_environment_lock  # noqa: E402
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
OPTIMIZED_RUN = "ma_optimized_exploratory"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _cohort_question_ids(config: dict, kind: str) -> list[str]:
    dataset = config.get("dataset") or {}
    cohort = (
        config.get("timing") or {}
        if kind == "timing"
        else config.get("pilot") or {}
        if kind == "pilot"
        else dataset
    )
    ids, _ = load_id_manifest(_repo_path(cohort["manifest_path"]))
    return ids


def completed_run_ids(
    config: dict,
    *,
    kind: str,
    expected_environment_lock_sha256: str | None = None,
) -> set[str]:
    """Find finalized, hash-valid artifacts so campaign restarts skip them."""
    results_dir = Path(config.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    dataset = config.get("dataset") or {}
    timing = config.get("timing") or {}
    pilot = config.get("pilot") or {}
    cohort = timing if kind == "timing" else pilot if kind == "pilot" else dataset
    expected_n = int(cohort["n"])
    expected_seed = int(cohort.get("seed", dataset["eval_seed"]))
    expected_manifest = cohort.get("manifest_sha256")
    expected_manifest_file = cohort.get("manifest_file_sha256")
    try:
        expected_question_ids = _cohort_question_ids(config, kind)
    except (OSError, ValueError, KeyError):
        return set()
    if len(expected_question_ids) != expected_n:
        return set()
    completed: set[str] = set()
    for meta_path in results_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if bool(meta.get("timing_mode")) != (kind == "timing"):
            continue
        if bool(meta.get("pilot_mode")) != (kind == "pilot"):
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
        run_id = meta.get("run_id")
        if not isinstance(run_id, str) or run_id not in (config.get("runs") or {}):
            continue
        manifest_file_key = {
            "accuracy": "manifest_file_sha256",
            "timing": "timing_manifest_file_sha256",
            "pilot": "pilot_manifest_file_sha256",
        }[kind]
        if (
            meta.get("question_ids_sha256") != expected_manifest
            or meta.get(manifest_file_key) != expected_manifest_file
            or not meta.get("environment_lock_sha256")
            or (
                expected_environment_lock_sha256 is not None
                and meta.get("environment_lock_sha256")
                != expected_environment_lock_sha256
            )
            or not meta.get("git_commit")
            or meta.get("git_commit") == "unknown"
            or meta.get("prompt_bundle_version") != prompts.PROMPT_BUNDLE_VERSION
            or meta.get("prompt_template_sha256") != prompts.prompt_template_hashes()
        ):
            continue
        expected_stage_fingerprints = {
            stage: treatment["config_fingerprint"]
            for stage, treatment in resolve_treatments(config, run_id).items()
        }
        if meta.get("stage_config_fingerprints") != expected_stage_fingerprints:
            continue
        experiment_fingerprint = meta.get("experiment_fingerprint")
        payload = meta.get("experiment_fingerprint_payload")
        retrieval_meta = meta.get("retrieval") or {}
        retrieval_config = config.get("retrieval") or {}
        if (
            not experiment_fingerprint
            or not isinstance(payload, dict)
            or _content_hash(payload) != experiment_fingerprint
            or payload.get("schema") != EXPERIMENT_SCHEMA
            or payload.get("architecture") != config.get("architecture")
            or tuple(payload.get("pipeline_stages") or ())
            != tuple(prompts.PIPELINE_STAGES)
            or payload.get("stage_role") != prompts.STAGE_ROLE
            or payload.get("max_new_tokens") != prompts.MAX_NEW_TOKENS
            or retrieval_meta.get("corpus_sha256")
            != retrieval_config.get("expected_corpus_sha256")
            or retrieval_meta.get("query_policy") != retrieval_config.get("query_policy")
            or int(retrieval_meta.get("k_per_step", -1))
            != int(retrieval_config.get("k", -2))
            or retrieval_meta.get("initial_query_source")
            != retrieval_config.get("initial_query_source")
            or int(retrieval_meta.get("grounded_followup_k", -1))
            != int(retrieval_config.get("grounded_followup_k", -2))
            or int(retrieval_meta.get("grounded_followup_k", -1))
            != int(retrieval_meta.get("k_per_step", -2))
            or int(retrieval_meta.get("anchor_k", -1))
            != int(retrieval_config.get("anchor_k", -2))
            or int(retrieval_meta.get("task_k", -1))
            != int(retrieval_config.get("task_k", -2))
            or int(retrieval_meta.get("anchor_k", -1))
            + int(retrieval_meta.get("task_k", -1))
            != int(retrieval_meta.get("k_per_step", -2))
            or retrieval_meta.get("grounded_followup_requires_evidence") is not False
            or retrieval_config.get("grounded_followup_requires_evidence") is not False
            or float(retrieval_meta.get("gold_sentence_coverage", -1)) != 1.0
        ):
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
        answer_ids: list[str] = []
        scientific_count = 0
        invalid_record = False
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    scientific = (
                        record.get("record_type") in {"agent_call", "answer"}
                        or (
                            record.get("record_type") == "batch"
                            and record.get("phase")
                            in {"scored", "timing_benchmark"}
                        )
                    )
                    if not scientific:
                        continue
                    scientific_count += 1
                    if (
                        record.get("experiment_fingerprint")
                        != experiment_fingerprint
                        or record.get("question_manifest_sha256")
                        != expected_manifest
                    ):
                        invalid_record = True
                        break
                    if record.get("record_type") == "answer":
                        question_id = record.get("question_id")
                        if not isinstance(question_id, str):
                            invalid_record = True
                            break
                        answer_ids.append(question_id)
        except (OSError, ValueError, TypeError):
            continue
        if invalid_record or scientific_count == 0:
            continue
        if kind == "timing":
            if answer_ids:
                continue
        elif answer_ids != expected_question_ids:
            continue
        completed.add(run_id)
    return completed


def validate_matrix(config: dict) -> None:
    runs = set(config.get("runs") or {})
    frozen = config.get("frozen_allocation") or {}
    allowed = {frozenset(STATIC_RUNS)}
    if frozen.get("execution_run_id") == OPTIMIZED_RUN:
        allowed.add(frozenset(STATIC_RUNS | {OPTIMIZED_RUN}))
    if frozenset(runs) not in allowed:
        raise RuntimeError(
            f"static matrix must be the exact 22-run contract; missing={sorted(STATIC_RUNS-runs)}, "
            f"extra={sorted(runs-STATIC_RUNS)}"
        )
    selector = config.get("allocation_selector") or {}
    if selector.get("candidate_allocation_count") != 625:
        raise RuntimeError("allocation selector must declare exactly 625 candidates")
    if set(selector.get("candidates") or {}) != {
        "base_fp16", "base_8bit", "base_4bit", "small_fp16", "tiny_fp16"
    }:
        raise RuntimeError("selector candidate set must cover all five frozen tiers")
    generation = config.get("generation") or {}
    if generation.get("batch_size") != 32 or generation.get("min_batch_size") != 32:
        raise RuntimeError("production batch size must be pinned fail-closed at 32")


def build_plan(config: dict, *, kind: str, workers: int, seed: int) -> dict:
    validate_matrix(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    if kind in {"timing", "pilot"} and workers != 1:
        raise ValueError(f"{kind} must run on exactly one reserved A100 worker")
    if kind == "accuracy":
        run_ids = sorted(STATIC_RUNS)
    elif kind == "pilot":
        run_ids = list((config.get("pilot") or {}).get("run_ids") or ())
        if run_ids != ["baseline", "single_fp16"]:
            raise RuntimeError("pilot run IDs must be [baseline, single_fp16]")
    else:
        timing = config.get("timing") or {}
        run_ids = list(timing.get("run_ids") or ())
        selected_execution = (
            timing.get("selected_execution_run_id")
            or (config.get("frozen_allocation") or {}).get("execution_run_id")
        )
        if selected_execution and selected_execution not in run_ids:
            run_ids.append(selected_execution)
        if not run_ids or len(run_ids) != len(set(run_ids)):
            raise RuntimeError("timing.run_ids must be a non-empty unique list")
        unauthorized_tiny = (set(run_ids) & TINY_RUNS) - (
            {selected_execution} if selected_execution else set()
        )
        if unauthorized_tiny:
            raise RuntimeError("only the frozen selected tiny system may be timed")
        allowed_timing = STATIC_RUNS | (
            {OPTIMIZED_RUN} if OPTIMIZED_RUN in (config.get("runs") or {}) else set()
        )
        if not set(run_ids) <= allowed_timing:
            raise RuntimeError("timing.run_ids contains an unknown/static-external arm")
    if kind != "pilot":
        random.Random(seed).shuffle(run_ids)
    assignments = {
        str(worker): run_ids[worker::workers] for worker in range(workers)
    }
    return {
        "schema_version": CAMPAIGN_PLAN_SCHEMA_VERSION,
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
    parser.add_argument("--kind", choices=("pilot", "accuracy", "timing"), required=True)
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
    plan["experiment_contract_sha256"] = _content_hash({
        "architecture": config.get("architecture"),
        "pipeline_stages": prompts.PIPELINE_STAGES,
        "stage_role": prompts.STAGE_ROLE,
        "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
        "prompt_template_sha256": prompts.prompt_template_hashes(),
        "retrieval": config.get("retrieval"),
        "dataset": config.get("dataset"),
        "pilot": config.get("pilot"),
        "timing": config.get("timing"),
        "model_revisions": config.get("model_revisions"),
        "runs": config.get("runs"),
    })
    default_plan = ROOT / "logs" / f"{args.kind}_seed{args.seed}_w{args.workers}.plan.json"
    plan_path = Path(args.plan).resolve() if args.plan else default_plan
    _write_json_atomic(plan_path, plan)

    assigned = plan["assignments"][str(args.worker_index)]
    expected_lock_hash = None
    if args.execute:
        config["_config_path"] = str(config_path)
        config["_config_dir"] = str(config_path.parent)
        lock_path = _repo_path(
            config.get("environment_lock_path", "config/environment.lock.json")
        )
        validate_environment_lock(config, config_path, lock_path)
        expected_lock_hash = _sha256(lock_path)
    already_complete = completed_run_ids(
        config,
        kind=args.kind,
        expected_environment_lock_sha256=expected_lock_hash,
    )
    plan["completed_before_start"] = sorted(already_complete)
    _write_json_atomic(plan_path, plan)
    print(f"plan: {plan_path}")
    print(f"worker {args.worker_index}/{args.workers}: {', '.join(assigned)}")
    if not args.execute:
        return 0
    if args.kind in {"accuracy", "timing"} and config.get("pilot", {}).get(
        "required_before_final_campaign"
    ):
        verify_gate(
            config_path,
            ROOT / config["pilot"]["gate_artifact"],
            require_tracked=True,
        )
    for run_id in assigned:
        if run_id in already_complete:
            print(f"skip finalized {args.kind} artifact: {run_id}")
            continue
        command = [
            sys.executable, "-m", "src.runner", "--config", str(config_path), "--run", run_id
        ]
        if args.kind == "timing":
            command.append("--timing-mode")
        elif args.kind == "pilot":
            command.append("--pilot-mode")
        subprocess.run(command, cwd=ROOT, check=True)
    if args.kind == "pilot":
        subprocess.run(
            [sys.executable, "scripts/check_pilot.py", "--config", str(config_path)],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
