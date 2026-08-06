#!/usr/bin/env python3
"""Final, CPU-only analysis for the A100 experiment.

The script deliberately keeps design decisions executable:

* F1 is primary and EM is co-reported.
* Answer comparisons are paired on exactly the same question IDs.
* The four 3B-8bit versus 1.5B-FP16 role contrasts share one 10,000-draw
  question bootstrap and use Holm family-wise correction.
* 3B-4bit is a separate secondary family.
* 0.5B is an appendix floor only: it is never offered to the allocation
  selector and never enters the main Pareto claim.
* Call-level rates are clustered by question; timing uses batch records only.
* The exploratory allocation selector enumerates exactly 4^4 assignments and
  charges every distinct resident configuration once.

Typical use after the static runs finish::

    python analyze.py --config config/experiment.yaml --results results \
        --output analysis

If all selector inputs are present, this also writes an immutable executable
config containing ``ma_optimized_exploratory``.  That run is explicitly marked
as same-sample, post-selection, and exploratory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from src import evidence as evidence_metrics
from src.mechanism import SELECTION_FIELD, selection_changed_set
from src.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_ci,
    clustered_ratio_bootstrap,
    holm_adjust,
    joint_bootstrap_draws,
    joint_paired_bootstrap,
)


ROLES = ("planner", "step_definer", "extractor", "qa")
ROLE_PREFIX = {
    "planner": "planner",
    "step_definer": "stepdef",
    "extractor": "extractor",
    "qa": "qa",
}
BASELINE_RUN = "baseline"
SOLO_RUN = "single_fp16"
OPTIMIZED_RUN = "ma_optimized_exploratory"
STATIC_RUN_IDS = {
    BASELINE_RUN,
    SOLO_RUN,
    *(f"{prefix}_{tier}" for prefix in ("planner", "stepdef", "extractor", "qa")
      for tier in ("8bit", "4bit", "small", "tiny")),
    "ma_uniform_8bit",
    "ma_uniform_4bit",
    "ma_uniform_small",
    "ma_uniform_tiny",
}

# Deliberately closed.  In particular, tiny/0.5B is not accepted here even if a
# config contract accidentally lists it.
SELECTOR_CONFIGS = ("base_fp16", "base_8bit", "base_4bit", "small_fp16")
SELECTOR_RUN_SUFFIX = {
    "base_8bit": "8bit",
    "base_4bit": "4bit",
    "small_fp16": "small",
}
UNIFORM_RUN = {
    "base_fp16": BASELINE_RUN,
    "base_8bit": "ma_uniform_8bit",
    "base_4bit": "ma_uniform_4bit",
    "small_fp16": "ma_uniform_small",
}
SELECTOR_TREATMENT = {
    "base_fp16": "fp16",
    "base_8bit": "8bit",
    "base_4bit": "4bit",
    "small_fp16": {"model": "small", "precision": "fp16"},
}

EXPECTED_FINAL_MANIFEST_SHA256 = (
    "5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af"
)


class AnalysisError(RuntimeError):
    """A condition that would make a reported comparison invalid."""


@dataclass(frozen=True)
class RunData:
    run_id: str
    answers: dict[str, dict[str, Any]]
    calls: tuple[dict[str, Any], ...]
    batches: tuple[dict[str, Any], ...]
    meta: dict[str, Any]
    jsonl_path: Path
    meta_path: Path
    timing_calls: tuple[dict[str, Any], ...] = ()
    timing_meta: dict[str, Any] | None = None
    timing_jsonl_path: Path | None = None
    timing_meta_path: Path | None = None
    timing_n_questions: int | None = None
    timing_repetitions: int | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _batch_record(record: dict[str, Any]) -> bool:
    return record.get("record_type") in {"batch", "batch_timing"}


def load_run(meta_path: Path) -> RunData:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    run_id = meta.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise AnalysisError(f"{meta_path}: missing run_id")
    jsonl_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".jsonl")
    if not jsonl_path.exists():
        raise AnalysisError(f"{meta_path}: missing matching {jsonl_path.name}")

    answers: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    call_keys: set[tuple[str, str, int]] = set()
    batches: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{jsonl_path}:{line_number}: invalid JSON") from exc
            if record.get("run_id") not in (None, run_id):
                raise AnalysisError(
                    f"{jsonl_path}:{line_number}: record run_id differs from metadata"
                )
            if record.get("record_type") == "answer":
                question_id = record.get("question_id")
                if not question_id:
                    raise AnalysisError(f"{jsonl_path}:{line_number}: answer has no question_id")
                if question_id in answers:
                    raise AnalysisError(
                        f"{jsonl_path}: duplicate answer for question {question_id!r}"
                    )
                for metric in ("f1", "em"):
                    if metric not in record or not math.isfinite(float(record[metric])):
                        raise AnalysisError(
                            f"{jsonl_path}:{line_number}: invalid answer metric {metric}"
                        )
                answers[str(question_id)] = record
            elif _batch_record(record):
                batches.append(record)
            elif record.get("record_type") in (None, "call", "agent_call"):
                key = (
                    str(record.get("question_id")),
                    str(record.get("stage")),
                    int(record.get("call_index", -1)),
                )
                if key in call_keys:
                    raise AnalysisError(f"{jsonl_path}: duplicate agent-call key {key}")
                call_keys.add(key)
                calls.append(record)
            # model_load/session/preflight telemetry is represented in metadata
            # or dedicated batch records and must never be mistaken for an agent
            # call (doing so would count it as a parse failure).

    expected_n = meta.get("n")
    if expected_n is not None and answers and len(answers) != int(expected_n):
        raise AnalysisError(
            f"{run_id}: metadata says n={expected_n}, but JSONL has "
            f"{len(answers)} unique answers"
        )
    return RunData(
        run_id=run_id,
        answers=answers,
        calls=tuple(calls),
        batches=tuple(batches),
        meta=meta,
        jsonl_path=jsonl_path,
        meta_path=meta_path,
    )


def discover_runs(
    results_dir: Path,
    *,
    n: int | None = None,
    seed: int | None = None,
    model_id: str | None = None,
    timing_manifest_sha256: str | None = None,
    timing_n: int | None = None,
    timing_seed: int | None = None,
    timing_repetitions: int | None = None,
) -> dict[str, RunData]:
    """Load one unambiguous result per run for the requested final cohort."""
    candidates: dict[str, list[RunData]] = defaultdict(list)
    for meta_path in sorted(results_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        is_timing = bool(meta.get("timing_mode", False))
        if is_timing:
            meta_manifest = _meta_manifest_hash(meta)
            if not timing_manifest_sha256 or meta_manifest != timing_manifest_sha256:
                continue
            if timing_n is not None and meta.get("n") != timing_n:
                continue
            if timing_seed is not None and meta.get("seed") != timing_seed:
                continue
        else:
            if n is not None and meta.get("n") != n:
                continue
            if seed is not None and meta.get("seed") != seed:
                continue
        # Solo runs may use a distinct top-level model description but their
        # stage model must still be the reference; do not filter on absent data.
        if model_id is not None and meta.get("model_id") not in (None, model_id):
            continue
        run = load_run(meta_path)
        candidates[run.run_id].append(run)

    output: dict[str, RunData] = {}
    for run_id, values in candidates.items():
        accuracy = [run for run in values if not run.meta.get("timing_mode", False)]
        timing = [run for run in values if run.meta.get("timing_mode", False)]
        if len(accuracy) > 1 or len(timing) > 1:
            raise AnalysisError(
                f"multiple accuracy/timing artifacts match {run_id}: "
                f"{[run.meta_path.name for run in values]}"
            )
        if accuracy and timing:
            primary, timed = accuracy[0], timing[0]
            output[run_id] = RunData(
                run_id=run_id,
                answers=primary.answers,
                calls=primary.calls,
                batches=timed.batches,
                meta=primary.meta,
                jsonl_path=primary.jsonl_path,
                meta_path=primary.meta_path,
                timing_calls=timed.calls,
                timing_meta=timed.meta,
                timing_jsonl_path=timed.jsonl_path,
                timing_meta_path=timed.meta_path,
                timing_n_questions=(
                    int(timed.meta.get("n") or timing_n)
                    * int(timing_repetitions or timed.meta.get("timing_repetitions") or 1)
                ),
                timing_repetitions=int(
                    timing_repetitions or timed.meta.get("timing_repetitions") or 1
                ),
            )
        elif accuracy:
            output[run_id] = accuracy[0]
        # A timing-only artifact never supplies F1/EM. It is held until the
        # matching final-manifest accuracy artifact exists.
    return output


def _same_question_ids(run_a: RunData, run_b: RunData) -> list[str]:
    ids_a, ids_b = set(run_a.answers), set(run_b.answers)
    if ids_a != ids_b:
        raise AnalysisError(
            f"{run_a.run_id} and {run_b.run_id} are not paired on the same questions: "
            f"{run_a.run_id}-only={len(ids_a - ids_b)}, "
            f"{run_b.run_id}-only={len(ids_b - ids_a)}"
        )
    if not ids_a:
        raise AnalysisError(f"{run_a.run_id} and {run_b.run_id} have no answers")
    return sorted(ids_a)


def _meta_manifest_hash(meta: dict[str, Any]) -> str | None:
    return (
        meta.get("question_manifest_sha256")
        or meta.get("manifest_sha256")
        or meta.get("question_ids_sha256")
        or (meta.get("dataset") or {}).get("manifest_sha256")
        or (meta.get("dataset") or {}).get("question_ids_sha256")
    )


def _run_manifest_hash(run: RunData) -> str | None:
    return _meta_manifest_hash(run.meta)


def _configured_timing_manifest_hash(config: dict[str, Any]) -> str | None:
    timing = config.get("timing") or {}
    cohort = timing.get("cohort") or {}
    return (
        timing.get("manifest_sha256")
        or timing.get("question_manifest_sha256")
        or cohort.get("manifest_sha256")
        or cohort.get("question_manifest_sha256")
    )


def validate_final_cohort(config: dict[str, Any], runs: dict[str, RunData]) -> None:
    """Every scored arm must certify and contain the one frozen question cohort."""
    configured_hash = (config.get("dataset") or {}).get("manifest_sha256")
    if configured_hash != EXPECTED_FINAL_MANIFEST_SHA256:
        raise AnalysisError(
            "analysis config does not identify the frozen final question manifest"
        )
    complete = [run for run in runs.values() if run.answers]
    if not complete:
        return
    reference = runs.get(BASELINE_RUN, complete[0])
    expected_manifest_file = (config.get("dataset") or {}).get("manifest_file_sha256")
    timing_config = config.get("timing") or {}
    timing_cohort = timing_config.get("cohort") or {}
    expected_timing_manifest = _configured_timing_manifest_hash(config)
    expected_timing_file = (
        timing_config.get("manifest_file_sha256")
        or timing_cohort.get("manifest_file_sha256")
    )
    expected_timing_n = timing_config.get("n", timing_cohort.get("n"))
    expected_timing_repetitions = timing_config.get(
        "repetitions", timing_cohort.get("repetitions")
    )
    environment_hashes: set[str] = set()
    git_commits: set[str] = set()
    timing_gpu_uuids: set[str] = set()
    timing_gpu_names: set[str] = set()
    configured_timing_runs = set(timing_config.get("run_ids") or ())
    post_selection_timing_run = timing_config.get("post_selection_run_id")
    allowed_timing_runs = configured_timing_runs | (
        {str(post_selection_timing_run)} if post_selection_timing_run else set()
    )
    for run in complete:
        if _run_manifest_hash(run) != configured_hash:
            raise AnalysisError(
                f"{run.run_id}: missing or mismatched question-manifest hash in metadata"
            )
        _same_question_ids(reference, run)
        if run.meta.get("metadata_complete") is not True:
            raise AnalysisError(f"{run.run_id}: metadata_complete is not true")
        expected_jsonl_hash = run.meta.get("jsonl_sha256")
        if not expected_jsonl_hash or file_hash(run.jsonl_path) != expected_jsonl_hash:
            raise AnalysisError(f"{run.run_id}: missing or mismatched finalized JSONL hash")
        if run.meta.get("manifest_file_sha256") != expected_manifest_file:
            raise AnalysisError(f"{run.run_id}: final manifest file hash mismatch")
        environment_hash = run.meta.get("environment_lock_sha256")
        if not environment_hash:
            raise AnalysisError(f"{run.run_id}: missing environment lock hash")
        # A post-selection executable config may have a different config/source
        # hash and git commit, but it must use the exact same committed runtime
        # lock as every static arm.
        environment_hashes.add(str(environment_hash))
        git_commit = run.meta.get("git_commit")
        if not git_commit or git_commit == "unknown":
            raise AnalysisError(f"{run.run_id}: missing immutable git commit")
        if run.run_id != OPTIMIZED_RUN:
            git_commits.add(str(git_commit))
        stage_fingerprints = run.meta.get("stage_config_fingerprints") or {}
        if not stage_fingerprints or any(not value for value in stage_fingerprints.values()):
            raise AnalysisError(f"{run.run_id}: missing stage configuration fingerprints")
        if run.timing_meta is not None:
            timing_meta = run.timing_meta
            if allowed_timing_runs and run.run_id not in allowed_timing_runs:
                raise AnalysisError(
                    f"{run.run_id}: timing benchmark is outside timing.run_ids"
                )
            timing_manifest = _meta_manifest_hash(timing_meta)
            if not expected_timing_manifest:
                raise AnalysisError("timing artifact present without configured timing manifest")
            if expected_timing_manifest == configured_hash:
                raise AnalysisError("timing and final accuracy manifests must be disjoint artifacts")
            if timing_manifest != expected_timing_manifest:
                raise AnalysisError(f"{run.run_id}: timing replay manifest mismatch")
            actual_timing_file = (
                timing_meta.get("timing_manifest_file_sha256")
                or timing_meta.get("manifest_file_sha256")
            )
            if expected_timing_file and actual_timing_file != expected_timing_file:
                raise AnalysisError(f"{run.run_id}: timing replay manifest file mismatch")
            if expected_timing_n is not None and int(timing_meta.get("n", -1)) != int(
                expected_timing_n
            ):
                raise AnalysisError(f"{run.run_id}: timing cohort size mismatch")
            if expected_timing_repetitions is not None and run.timing_repetitions != int(
                expected_timing_repetitions
            ):
                raise AnalysisError(f"{run.run_id}: timing repetition count mismatch")
            if timing_meta.get("metadata_complete") is not True:
                raise AnalysisError(f"{run.run_id}: timing replay metadata is incomplete")
            if (
                run.timing_jsonl_path is None
                or not timing_meta.get("jsonl_sha256")
                or file_hash(run.timing_jsonl_path) != timing_meta["jsonl_sha256"]
            ):
                raise AnalysisError(f"{run.run_id}: timing JSONL hash mismatch")
            if timing_meta.get("environment_lock_sha256") != environment_hash:
                raise AnalysisError(f"{run.run_id}: timing replay environment changed")
            if timing_meta.get("git_commit") != git_commit:
                raise AnalysisError(f"{run.run_id}: timing replay code revision changed")
            timing_fingerprints = timing_meta.get("stage_config_fingerprints") or {}
            if not timing_fingerprints or any(
                stage_fingerprints.get(stage) != fingerprint
                for stage, fingerprint in timing_fingerprints.items()
            ):
                raise AnalysisError(f"{run.run_id}: timing benchmark treatment changed")
            timing_ids = {str(record["question_id"]) for record in run.timing_calls}
            overlap = timing_ids & set(run.answers)
            if overlap:
                raise AnalysisError(
                    f"{run.run_id}: excluded timing cohort overlaps final accuracy "
                    f"cohort on {len(overlap)} questions"
                )
            # The configured benchmark policy is one reserved, uncontended A100.
            # UUID is stronger than a SKU string and prevents silently pooling
            # timings from separate workers that happen to have the same model.
            records = [*run.timing_calls, *run.batches]
            record_uuids = {
                str(record["gpu_uuid"])
                for record in records
                if record.get("gpu_uuid")
            }
            record_names = {
                str(record["gpu_name"])
                for record in records
                if record.get("gpu_name")
            }
            sessions = timing_meta.get("execution_sessions") or []
            record_uuids.update(
                str(session["gpu_uuid"])
                for session in sessions
                if session.get("gpu_uuid")
            )
            record_names.update(
                str(session["gpu_name"])
                for session in sessions
                if session.get("gpu_name")
            )
            if len(record_uuids) != 1:
                raise AnalysisError(
                    f"{run.run_id}: timing benchmark must certify exactly one GPU UUID"
                )
            if len(record_names) != 1 or "A100" not in next(iter(record_names)):
                raise AnalysisError(
                    f"{run.run_id}: timing benchmark is not certified on one A100"
                )
            timing_gpu_uuids.update(record_uuids)
            timing_gpu_names.update(record_names)
    if len(environment_hashes) > 1:
        raise AnalysisError("accuracy runs used different environment locks")
    if len(git_commits) > 1:
        raise AnalysisError("accuracy runs were generated from different git commits")
    if len(timing_gpu_uuids) > 1 or len(timing_gpu_names) > 1:
        raise AnalysisError(
            "timing benchmarks were not run on the same reserved physical A100"
        )


def _existing_selection_artifact(config: dict[str, Any]) -> dict[str, Any] | None:
    outputs = ((config.get("allocation_selector") or {}).get("output_artifacts") or {})
    trace_value = outputs.get("selection_trace")
    if not trace_value:
        return None
    trace_path = _repo_path(trace_value)
    if not trace_path.exists():
        return None
    artifact = json.loads(trace_path.read_text(encoding="utf-8"))
    config_hash = content_hash(config)
    is_source = artifact.get("source_config_sha256") == config_hash
    is_executable = artifact.get("executable_config_sha256") == config_hash
    if not (is_source or is_executable):
        raise AnalysisError(
            f"stale selection artifact does not belong to this config: {trace_path}"
        )
    if not verify_selection_artifact(
        artifact, config if is_executable and not is_source else None
    ):
        raise AnalysisError(f"invalid frozen selection artifact: {trace_path}")
    return artifact


def validate_campaign_completeness(
    config: dict[str, Any], runs: dict[str, RunData]
) -> None:
    """Fail closed unless the entire publishable campaign is present."""
    artifact = _existing_selection_artifact(config)
    materialized = bool(artifact and artifact.get("materialized_new_run") is True)
    configured_runs = set(config.get("runs") or {})
    allowed_configured_matrices = {frozenset(STATIC_RUN_IDS)}
    if materialized:
        allowed_configured_matrices.add(frozenset(STATIC_RUN_IDS | {OPTIMIZED_RUN}))
    if frozenset(configured_runs) not in allowed_configured_matrices:
        expected_display = STATIC_RUN_IDS | ({OPTIMIZED_RUN} if materialized else set())
        raise AnalysisError(
            "configured accuracy matrix is not the exact publishable matrix: "
            f"missing={sorted(expected_display - configured_runs)}, "
            f"extra={sorted(configured_runs - expected_display)}"
        )

    expected_accuracy = set(STATIC_RUN_IDS)
    if materialized:
        expected_accuracy.add(OPTIMIZED_RUN)
    present_accuracy = {run_id for run_id, run in runs.items() if run.answers}
    missing_accuracy = expected_accuracy - present_accuracy
    extra_accuracy = present_accuracy - expected_accuracy
    if missing_accuracy or extra_accuracy:
        raise AnalysisError(
            "final accuracy campaign is incomplete or contains an uncontracted arm: "
            f"missing={sorted(missing_accuracy)}, extra={sorted(extra_accuracy)}"
        )

    timing = config.get("timing") or {}
    expected_timing = set(timing.get("run_ids") or ())
    if materialized:
        expected_timing.add(OPTIMIZED_RUN)
    present_timing = {
        run_id for run_id, run in runs.items() if run.timing_meta is not None
    }
    missing_timing = expected_timing - present_timing
    extra_timing = present_timing - expected_timing
    if missing_timing or extra_timing:
        raise AnalysisError(
            "final excluded-cohort timing campaign is incomplete or uncontracted: "
            f"missing={sorted(missing_timing)}, extra={sorted(extra_timing)}"
        )
    if materialized:
        optimized = runs[OPTIMIZED_RUN]
        expected_trace_hash = artifact.get("artifact_sha256")
        expected_run_hash = artifact.get("run_config_sha256")
        for label, meta in (
            ("accuracy", optimized.meta),
            ("timing", optimized.timing_meta or {}),
        ):
            if meta.get("allocation_selection_artifact_sha256") != expected_trace_hash:
                raise AnalysisError(
                    f"optimized {label} result is not linked to the frozen selector trace"
                )
            if meta.get("allocation_run_config_sha256") != expected_run_hash:
                raise AnalysisError(
                    f"optimized {label} result used a different frozen run allocation"
                )
            if meta.get("allocation_claim_status") != artifact.get("claim_status"):
                raise AnalysisError(
                    f"optimized {label} result changed the exploratory claim status"
                )


def answer_mean(run: RunData, metric: str) -> float:
    if metric not in {"f1", "em"}:
        raise ValueError(f"unsupported answer metric {metric!r}")
    if not run.answers:
        raise AnalysisError(f"{run.run_id} has no answer records")
    return sum(float(record[metric]) for record in run.answers.values()) / len(run.answers)


def paired_differences(
    run_a: RunData,
    run_b: RunData,
    metric: str,
    *,
    scale: float = 100.0,
) -> dict[str, float]:
    """Return ``A - B`` per shared question; never compare separate CIs."""
    ids = _same_question_ids(run_a, run_b)
    return {
        question_id: scale * (
            float(run_a.answers[question_id][metric])
            - float(run_b.answers[question_id][metric])
        )
        for question_id in ids
    }


def role_run(role: str, suffix: str) -> str:
    return f"{ROLE_PREFIX[role]}_{suffix}"


def _require_runs(runs: dict[str, RunData], run_ids: list[str], section: str) -> None:
    missing = [run_id for run_id in run_ids if run_id not in runs]
    if missing:
        raise AnalysisError(f"{section} requires missing runs: {', '.join(missing)}")


def _family_analysis(
    runs: dict[str, RunData],
    *,
    left_suffix: str,
    right_suffix: str,
    family_name: str,
    n_resamples: int,
    seed: int,
    adjust_f1: bool,
) -> dict[str, Any]:
    required = [
        role_run(role, suffix)
        for role in ROLES
        for suffix in (left_suffix, right_suffix)
    ]
    _require_runs(runs, required, family_name)
    output: dict[str, Any] = {
        "family": family_name,
        "contrast_direction": f"{left_suffix}_minus_{right_suffix}",
        "resampling_unit": "question",
        "n_resamples": n_resamples,
        "metrics": {},
    }
    for metric_index, metric in enumerate(("f1", "em")):
        contrasts = {
            role: paired_differences(
                runs[role_run(role, left_suffix)],
                runs[role_run(role, right_suffix)],
                metric,
            )
            for role in ROLES
        }
        estimates = joint_paired_bootstrap(
            contrasts,
            n_resamples=n_resamples,
            seed=seed + metric_index,
        )
        if metric == "f1" and adjust_f1:
            adjusted = holm_adjust(
                {role: float(result["p_value"]) for role, result in estimates.items()}
            )
            for role, result in estimates.items():
                result["holm_p_value"] = adjusted[role]
                result["holm_significant_0.05"] = adjusted[role] < 0.05
        else:
            # EM is co-reported descriptively, not promoted to a second primary
            # hypothesis family.
            for result in estimates.values():
                result.pop("p_value", None)
        output["metrics"][metric] = estimates
    return output


def analyze_role_families(
    runs: dict[str, RunData],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Primary Q-vs-S and aggressive-4bit secondary role families."""
    primary = _family_analysis(
        runs,
        left_suffix="8bit",
        right_suffix="small",
        family_name="primary_3b8_vs_1p5b_fp16",
        n_resamples=n_resamples,
        seed=seed,
        adjust_f1=True,
    )
    secondary = _family_analysis(
        runs,
        left_suffix="4bit",
        right_suffix="small",
        family_name="secondary_3b4_vs_1p5b_fp16",
        n_resamples=n_resamples,
        seed=seed + 100,
        adjust_f1=True,
    )
    primary["interpretation"] = (
        "Positive means the 3B 8-bit treatment retained more accuracy than the "
        "near-memory-matched 1.5B FP16 treatment. Inference is on direct paired "
        "differences; overlap of separate arm CIs is not used."
    )
    secondary["interpretation"] = (
        "Aggressive 3B 4-bit is a separately corrected secondary family, not a "
        "clean bit-width dose-response continuation of LLM.int8."
    )
    return {"primary": primary, "secondary_4bit": secondary}


def analyze_accuracy_runs(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """F1 and co-reported EM for every complete run, with question CIs."""
    main: dict[str, Any] = {}
    appendix: dict[str, Any] = {}
    for run_index, (run_id, run) in enumerate(sorted(runs.items())):
        if not run.answers:
            continue
        summary: dict[str, Any] = {"n_questions": len(run.answers)}
        for metric_index, metric in enumerate(("f1", "em")):
            values = [100 * float(record[metric]) for record in run.answers.values()]
            estimate, lower, upper = bootstrap_ci(
                values,
                n_resamples=n_resamples,
                seed=seed + 2 * run_index + metric_index,
            )
            summary[metric] = {
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        summary["primary_metric"] = "f1"
        summary["em_inferential_status"] = "descriptive_co_reported"
        if "tiny" in run_id:
            summary["scope"] = "lower_limit_appendix_only"
            appendix[run_id] = summary
        else:
            summary["scope"] = (
                "exploratory_same_sample_post_selection"
                if run_id == OPTIMIZED_RUN else "main"
            )
            main[run_id] = summary
    return {"main": main, "tiny_floor_appendix": appendix}


def _rank_summary(
    names: list[str],
    draws: np.ndarray,
) -> dict[str, Any]:
    """Descriptive bootstrap ranks with fractional ties and pairwise probabilities."""
    n_draws, n_roles = draws.shape
    ranks = np.empty_like(draws, dtype=float)
    for column in range(n_roles):
        value = draws[:, column][:, None]
        ranks[:, column] = (
            1
            + (draws > value).sum(axis=1)
            + 0.5 * ((draws == value).sum(axis=1) - 1)
        )
    output: dict[str, Any] = {"roles": {}, "pairwise_probability_more_costly": {}}
    for column, name in enumerate(names):
        role_ranks = ranks[:, column]
        rank_probabilities = {
            str(rank): float(np.mean(role_ranks == rank))
            for rank in sorted(set(role_ranks.tolist()))
        }
        output["roles"][name] = {
            "mean_rank": float(role_ranks.mean()),
            "rank_95pct_interval": [
                float(np.quantile(role_ranks, 0.025, method="nearest")),
                float(np.quantile(role_ranks, 0.975, method="nearest")),
            ],
            "rank_probabilities": rank_probabilities,
            "probability_most_costly": float(
                np.mean(
                    (draws[:, column] == draws.max(axis=1))
                    / (draws == draws.max(axis=1)[:, None]).sum(axis=1)
                )
            ),
            "probability_cheapest": float(
                np.mean(
                    (draws[:, column] == draws.min(axis=1))
                    / (draws == draws.min(axis=1)[:, None]).sum(axis=1)
                )
            ),
        }
        output["pairwise_probability_more_costly"][name] = {
            other: float(np.mean(draws[:, column] > draws[:, other_column]))
            for other_column, other in enumerate(names)
            if other != name
        }
    output["interpretation"] = (
        "Ranks are descriptive with only four roles. Overlapping rank confidence "
        "sets are retained; no definitive total ordering is forced."
    )
    output["n_bootstrap_draws"] = n_draws
    return output


def analyze_role_costs(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Direct paired baseline-minus-treatment cost for each role and axis."""
    if BASELINE_RUN not in runs:
        return {}
    output: dict[str, Any] = {}
    for family_index, (label, suffix) in enumerate(
        (("3b_8bit", "8bit"), ("3b_4bit", "4bit"), ("1p5b_fp16", "small"))
    ):
        run_ids = [role_run(role, suffix) for role in ROLES]
        if not all(run_id in runs for run_id in run_ids):
            continue
        family: dict[str, Any] = {
            "contrast_direction": "baseline_minus_treatment",
            "positive_means_accuracy_cost": True,
            "metrics": {},
        }
        f1_draws = None
        f1_names = None
        for metric_index, metric in enumerate(("f1", "em")):
            contrasts = {
                role: paired_differences(
                    runs[BASELINE_RUN], runs[role_run(role, suffix)], metric
                )
                for role in ROLES
            }
            names, estimates, draws, question_ids = joint_bootstrap_draws(
                contrasts,
                n_resamples=n_resamples,
                seed=seed + 10 * family_index + metric_index,
            )
            estimates_report = {}
            lo_index = int(0.025 * n_resamples)
            hi_index = min(int(0.975 * n_resamples), n_resamples - 1)
            for column, name in enumerate(names):
                ordered = np.sort(draws[:, column])
                estimates_report[name] = {
                    "estimate": float(estimates[column]),
                    "ci_lower": float(ordered[lo_index]),
                    "ci_upper": float(ordered[hi_index]),
                    "n_questions": len(question_ids),
                    "n_resamples": n_resamples,
                }
            family["metrics"][metric] = estimates_report
            family["n_questions"] = len(question_ids)
            if metric == "f1":
                f1_draws, f1_names = draws, names
        assert f1_draws is not None and f1_names is not None
        family["f1_role_rank_uncertainty"] = _rank_summary(f1_names, f1_draws)
        output[label] = family
    return output


def analyze_solo(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    if SOLO_RUN not in runs or BASELINE_RUN not in runs:
        return None
    output: dict[str, Any] = {
        "contrast_direction": "single_3b_fp16_minus_uniform_multiagent_3b_fp16",
        "prespecified": True,
        "resampling_unit": "question",
    }
    for index, metric in enumerate(("f1", "em")):
        estimates = joint_paired_bootstrap(
            {"single_minus_multiagent": paired_differences(
                runs[SOLO_RUN], runs[BASELINE_RUN], metric
            )},
            n_resamples=n_resamples,
            seed=seed + index,
        )["single_minus_multiagent"]
        estimates.pop("p_value", None)
        output[metric] = estimates
    return output


def analyze_optimized_system(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
    selected_run_id: str = OPTIMIZED_RUN,
) -> dict[str, Any] | None:
    """Actual selected system versus baseline; direct but explicitly exploratory."""
    if selected_run_id not in runs or BASELINE_RUN not in runs:
        return None
    output: dict[str, Any] = {
        "contrast_direction": "optimized_minus_uniform_multiagent_3b_fp16",
        "claim_status": "exploratory_same_sample_post_selection",
        "resampling_unit": "question",
        "warning": (
            "This direct paired CI describes the realized same-sample run. It is "
            "not a confirmatory role-aware superiority test because the allocation "
            "was selected from these questions."
        ),
    }
    for index, metric in enumerate(("f1", "em")):
        estimate = joint_paired_bootstrap(
            {"optimized_minus_baseline": paired_differences(
                runs[selected_run_id], runs[BASELINE_RUN], metric
            )},
            n_resamples=n_resamples,
            seed=seed + index,
        )["optimized_minus_baseline"]
        estimate.pop("p_value", None)
        output[metric] = estimate
    output["evaluated_run_id"] = selected_run_id
    return output


def _memory_mib(run: RunData) -> float:
    for key in (
        "deduplicated_concurrent_model_footprint_mib",
        "deduplicated_distinct_weight_mib",
        "deduplicated_footprint_mib",
        "deduped_footprint_mib",
        "deduped_footprint_mb",  # legacy name; values were binary MiB
    ):
        value = run.meta.get(key)
        if value is not None:
            return float(value)
    raise AnalysisError(f"{run.run_id}: no deduplicated resident-weight footprint")


def treated_model_memory(run: RunData, role: str) -> float:
    stage = run.meta.get("stages", {}).get(role, {})
    for key in (
        "model_footprint_mib",
        "full_model_footprint_mib",
        "weight_footprint_mib",
        "weight_footprint_mb",
    ):
        if stage.get(key) is not None:
            return float(stage[key])
    raise AnalysisError(f"{run.run_id}: no treated-model footprint for {role}")


def near_memory_match(runs: dict[str, RunData]) -> dict[str, Any]:
    """Report, rather than erase, the discrete 3B8/1.5B-FP16 budget gap."""
    per_role: dict[str, Any] = {}
    for role in ROLES:
        q_run, s_run = role_run(role, "8bit"), role_run(role, "small")
        _require_runs(runs, [q_run, s_run], "near-memory-match")
        q_memory = treated_model_memory(runs[q_run], role)
        s_memory = treated_model_memory(runs[s_run], role)
        gap = q_memory - s_memory
        q_system_memory = _memory_mib(runs[q_run])
        s_system_memory = _memory_mib(runs[s_run])
        system_gap = q_system_memory - s_system_memory
        per_role[role] = {
            "treated_model": {
                "3b_8bit_mib": q_memory,
                "1p5b_fp16_mib": s_memory,
                "gap_mib_3b8_minus_1p5b": gap,
                "gap_percent_of_1p5b": 100 * gap / s_memory,
                "lower_memory_arm": (
                    "1p5b_fp16" if gap > 0 else "3b_8bit" if gap < 0 else "equal"
                ),
            },
            "one_role_full_system_deduplicated_concurrent": {
                "3b_8bit_mib": q_system_memory,
                "1p5b_fp16_mib": s_system_memory,
                "gap_mib_3b8_minus_1p5b": system_gap,
                "gap_percent_of_1p5b": 100 * system_gap / s_system_memory,
                "lower_memory_arm": (
                    "1p5b_fp16"
                    if system_gap > 0
                    else "3b_8bit" if system_gap < 0 else "equal"
                ),
            },
        }
    return {
        "label": "near-memory-matched",
        "primary_memory_topology": "deduplicated_concurrent_model_footprint_mib",
        "per_role": per_role,
        "interpretation": (
            "Both the treated-model gap and the one-role full-system gap are "
            "reported; the latter is the primary experiment memory topology. "
            "A lower-memory arm that holds accuracy is a favorable optimization. "
            "Do not add inert padding or describe these discrete models as exactly matched."
        ),
    }


def pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measured nondominance in (lower memory, higher F1); no interpolation."""
    frontier = []
    for point in points:
        dominated = any(
            other is not point
            and float(other["memory_mib"]) <= float(point["memory_mib"])
            and float(other["f1"]) >= float(point["f1"])
            and (
                float(other["memory_mib"]) < float(point["memory_mib"])
                or float(other["f1"]) > float(point["f1"])
            )
            for other in points
        )
        if not dominated:
            frontier.append({**point, "nondominated": True})
    return sorted(frontier, key=lambda point: (point["memory_mib"], -point["f1"], point["run_id"]))


def analyze_uniform_pareto(runs: dict[str, RunData]) -> dict[str, Any]:
    main_ids = [UNIFORM_RUN[name] for name in SELECTOR_CONFIGS]
    points = []
    for run_id in main_ids:
        if run_id not in runs:
            continue
        points.append({
            "run_id": run_id,
            "memory_mib": _memory_mib(runs[run_id]),
            "f1": 100 * answer_mean(runs[run_id], "f1"),
            "em": 100 * answer_mean(runs[run_id], "em"),
        })
    output: dict[str, Any] = {
        "comparison": "measured_uniform_points_only",
        "interpolation": False,
        "points": sorted(points, key=lambda point: point["memory_mib"]),
        "nondominated": pareto_frontier(points),
        "excluded_from_main": ["ma_uniform_tiny"],
    }
    if OPTIMIZED_RUN in runs and points:
        selected = {
            "run_id": OPTIMIZED_RUN,
            "memory_mib": _memory_mib(runs[OPTIMIZED_RUN]),
            "f1": 100 * answer_mean(runs[OPTIMIZED_RUN], "f1"),
            "em": 100 * answer_mean(runs[OPTIMIZED_RUN], "em"),
        }
        eligible = [p for p in points if p["memory_mib"] <= selected["memory_mib"]]
        best = max(eligible, key=lambda point: (point["f1"], -point["memory_mib"])) \
            if eligible else None
        output["exploratory_selected_system"] = selected
        output["best_measured_uniform_same_or_lower_memory"] = best
        output["f1_advantage_over_uniform_points"] = (
            selected["f1"] - best["f1"] if best else None
        )
        output["claim_status"] = "exploratory_same_sample_post_selection"
    return output


def analyze_tiny_appendix(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    run_ids = [role_run(role, "tiny") for role in ROLES] + ["ma_uniform_tiny"]
    output: dict[str, Any] = {
        "scope": "lower_limit_appendix_only",
        "excluded_from_selector": True,
        "excluded_from_primary_and_secondary_families": True,
        "runs": {},
    }
    for index, run_id in enumerate(run_ids):
        if run_id not in runs:
            continue
        run_result: dict[str, Any] = {}
        for metric in ("f1", "em"):
            values = [100 * float(record[metric]) for record in runs[run_id].answers.values()]
            estimate, lower, upper = bootstrap_ci(
                values, n_resamples=n_resamples, seed=seed + index
            )
            run_result[metric] = {
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_questions": len(values),
            }
        try:
            run_result["memory_mib"] = _memory_mib(runs[run_id])
        except AnalysisError:
            run_result["memory_mib"] = None
        output["runs"][run_id] = run_result
    return output


def clustered_call_rate(
    records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """A per-call rate whose uncertainty resamples whole questions."""
    clusters: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for record in records:
        question_id = record.get("question_id")
        if question_id is None:
            raise AnalysisError("call metric record has no question_id")
        clusters[str(question_id)][0] += float(predicate(record))
        clusters[str(question_id)][1] += 1.0
    result = clustered_ratio_bootstrap(
        {question_id: (value[0], value[1]) for question_id, value in clusters.items()},
        n_resamples=n_resamples,
        seed=seed,
    )
    result["n_calls"] = int(sum(value[1] for value in clusters.values()))
    result["resampling_unit"] = "question"
    return result


def analyze_parse_failures(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Mechanism telemetry; fan-out roles are question-clustered."""
    output: dict[str, Any] = {}
    for run_index, (run_id, run) in enumerate(sorted(runs.items())):
        stages: dict[str, Any] = {}
        present_stages = sorted(
            {str(record.get("stage")) for record in run.calls if record.get("stage")}
        )
        for role_index, role in enumerate(present_stages):
            records = [record for record in run.calls if record.get("stage") == role]
            if records:
                failure = clustered_call_rate(
                    records,
                    lambda record: record.get("parse_status") != "ok",
                    n_resamples=n_resamples,
                    seed=seed + 10 * run_index + role_index,
                )
                any_failure_clusters: dict[str, tuple[float, float]] = {}
                by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for record in records:
                    by_question[str(record["question_id"])].append(record)
                for question_id, question_records in by_question.items():
                    any_failure_clusters[question_id] = (
                        float(any(r.get("parse_status") != "ok" for r in question_records)),
                        1.0,
                    )
                any_failure = clustered_ratio_bootstrap(
                    any_failure_clusters,
                    n_resamples=n_resamples,
                    seed=seed + 10 * run_index + role_index + 1_000,
                )
                failure_calls = [r for r in records if r.get("parse_status") != "ok"]
                salvage_available = None
                if failure_calls:
                    salvage_available = clustered_call_rate(
                        failure_calls,
                        lambda record: record.get("salvaged") is not None,
                        n_resamples=n_resamples,
                        seed=seed + 10 * run_index + role_index + 2_000,
                    )
                consumed_source_records = [
                    r for r in records if r.get("consumer_payload_source") is not None
                ]
                consumed_salvage = None
                if consumed_source_records:
                    consumed_salvage = clustered_call_rate(
                        consumed_source_records,
                        lambda record: record.get("consumer_payload_source") == "salvaged",
                        n_resamples=n_resamples,
                        seed=seed + 10 * run_index + role_index + 3_000,
                    )
                stages[role] = {
                    "per_call_failure_rate": failure,
                    "per_call_success_rate": {
                        "estimate": 1 - float(failure["estimate"]),
                        "ci_lower": 1 - float(failure["ci_upper"]),
                        "ci_upper": 1 - float(failure["ci_lower"]),
                        "n_clusters": failure["n_clusters"],
                        "n_calls": failure["n_calls"],
                        "resampling_unit": "question",
                    },
                    "per_question_any_failure_rate": any_failure,
                    "salvage_available_among_parse_failures": salvage_available,
                    "downstream_input_used_salvage_rate": consumed_salvage,
                    "salvage_interpretation": (
                        "Salvage availability/consumption is reported separately. A "
                        "salvaged payload never changes parse_status into success."
                    ),
                }
        if stages:
            output[run_id] = stages
    return {
        "metric": "parse_failure_rate",
        "estimand": "failed_calls_over_all_calls",
        "runs": output,
    }


def _question_mean_summary(
    values: dict[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    estimate, lower, upper = bootstrap_ci(
        list(values.values()), n_resamples=n_resamples, seed=seed
    )
    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_questions": len(values),
        "resampling_unit": "question",
    }


def analyze_call_counts(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Calls and token volume per question, including zero-call questions."""
    main: dict[str, Any] = {}
    appendix: dict[str, Any] = {}
    for run_index, (run_id, run) in enumerate(sorted(runs.items())):
        if not run.answers:
            continue
        by_metric: dict[str, dict[str, float]] = {
            "calls": {question_id: 0.0 for question_id in run.answers},
            "prompt_tokens": {question_id: 0.0 for question_id in run.answers},
            "output_tokens": {question_id: 0.0 for question_id in run.answers},
        }
        per_stage: dict[str, dict[str, float]] = defaultdict(
            lambda: {question_id: 0.0 for question_id in run.answers}
        )
        for record in run.calls:
            question_id = str(record["question_id"])
            if question_id not in run.answers:
                raise AnalysisError(f"{run_id}: call references a non-cohort question")
            by_metric["calls"][question_id] += 1
            by_metric["prompt_tokens"][question_id] += float(record.get("prompt_tokens") or 0)
            by_metric["output_tokens"][question_id] += float(record.get("output_tokens") or 0)
            per_stage[str(record["stage"])][question_id] += 1
        summary = {
            metric: _question_mean_summary(
                values,
                n_resamples=n_resamples,
                seed=seed + 100 * run_index + metric_index,
            )
            for metric_index, (metric, values) in enumerate(by_metric.items())
        }
        summary["calls_by_stage"] = {
            stage: _question_mean_summary(
                values,
                n_resamples=n_resamples,
                seed=seed + 100 * run_index + 10 + stage_index,
            )
            for stage_index, (stage, values) in enumerate(sorted(per_stage.items()))
        }
        if "tiny" in run_id:
            appendix[run_id] = summary
        else:
            main[run_id] = summary
    return {"main": main, "tiny_floor_appendix": appendix}


def _call_index(run: RunData, stage: str) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for record in run.calls:
        if record.get("stage") != stage:
            continue
        key = (str(record["question_id"]), int(record["call_index"]))
        if key in output:
            raise AnalysisError(f"{run.run_id}/{stage}: duplicate call key {key}")
        output[key] = record
    return output


def _effective_record(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "parsed": record.get("parsed") or record.get("salvaged")}


def analyze_selection_churn(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Same-call selection churn, clustered by question with key intersections."""
    if BASELINE_RUN not in runs:
        return {}
    main: dict[str, Any] = {}
    appendix: dict[str, Any] = {}
    suffixes = ("8bit", "4bit", "small", "tiny")
    comparison_index = 0
    for suffix in suffixes:
        target_scope = appendix if suffix == "tiny" else main
        for role in ROLES:
            run_id = role_run(role, suffix)
            if run_id not in runs:
                continue
            baseline_index = _call_index(runs[BASELINE_RUN], role)
            treatment_index = _call_index(runs[run_id], role)
            shared = sorted(set(baseline_index) & set(treatment_index))
            if not shared:
                continue
            strict_records = []
            effective_records = []
            field = SELECTION_FIELD[role]
            for question_id, call_index in shared:
                baseline_record = baseline_index[(question_id, call_index)]
                treatment_record = treatment_index[(question_id, call_index)]
                strict_records.append({
                    "question_id": question_id,
                    "changed": selection_changed_set(baseline_record, treatment_record, field),
                })
                effective_records.append({
                    "question_id": question_id,
                    "changed": selection_changed_set(
                        _effective_record(baseline_record),
                        _effective_record(treatment_record),
                        field,
                    ),
                })
            strict = clustered_call_rate(
                strict_records,
                lambda record: bool(record["changed"]),
                n_resamples=n_resamples,
                seed=seed + 2 * comparison_index,
            )
            effective = clustered_call_rate(
                effective_records,
                lambda record: bool(record["changed"]),
                n_resamples=n_resamples,
                seed=seed + 2 * comparison_index + 1,
            )
            target_scope[run_id] = {
                "stage": role,
                "selection_field": field,
                "parser_success_payload_churn": strict,
                "downstream_effective_payload_churn": effective,
                "shared_calls": len(shared),
                "baseline_only_calls": len(set(baseline_index) - set(treatment_index)),
                "treatment_only_calls": len(set(treatment_index) - set(baseline_index)),
                "interpretation": (
                    "Calls are intersected by (question_id, call_index). Effective "
                    "churn uses parsed-or-salvaged payloads but does not relabel parse status."
                ),
            }
            comparison_index += 1
    return {"main": main, "tiny_floor_appendix": appendix}


def _gold_evidence_from_answer(answer: dict[str, Any]) -> set[tuple[str, int]] | None:
    labels = answer.get("gold_evidence_labels", answer.get("evidence_gold_labels"))
    if labels is not None:
        return {(str(title), int(index)) for title, index in labels}
    supporting = answer.get("supporting_facts")
    if isinstance(supporting, dict):
        return evidence_metrics.gold_evidence(supporting)
    return None


def analyze_evidence(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """HotpotQA supporting-fact set F1 from consumed parsed-or-salvaged spans."""
    main: dict[str, Any] = {}
    appendix: dict[str, Any] = {}
    missing_fields = False
    for run_index, (run_id, run) in enumerate(sorted(runs.items())):
        precomputed = {
            question_id: answer
            for question_id, answer in run.answers.items()
            if answer.get("evidence_status") == "applicable"
        }
        if precomputed and len(precomputed) == len(run.answers):
            per_question_f1 = {
                question_id: float(answer["evidence_f1"])
                for question_id, answer in precomputed.items()
            }
            per_question_em = {
                question_id: float(answer["evidence_em"])
                for question_id, answer in precomputed.items()
            }
            summary = {
                "evidence_f1": _question_mean_summary(
                    per_question_f1,
                    n_resamples=n_resamples,
                    seed=seed + 2 * run_index,
                ),
                "evidence_em": _question_mean_summary(
                    per_question_em,
                    n_resamples=n_resamples,
                    seed=seed + 2 * run_index + 1,
                ),
                "payload_used": "answer_record_precomputed_from_parsed_or_salvaged_spans",
            }
            (appendix if "tiny" in run_id else main)[run_id] = summary
            continue
        if any(answer.get("evidence_status") == "not_applicable" for answer in run.answers.values()):
            # Solo has no extractor and must not be assigned a fake zero score.
            continue
        extractor_calls = [r for r in run.calls if r.get("stage") == "extractor"]
        if not extractor_calls:
            continue
        spans_by_question: dict[str, list[str]] = defaultdict(list)
        for record in extractor_calls:
            payload = record.get("parsed") or record.get("salvaged") or {}
            spans_by_question[str(record["question_id"])].extend(payload.get("spans") or [])
        per_question_f1: dict[str, float] = {}
        per_question_em: dict[str, float] = {}
        for question_id, answer in run.answers.items():
            gold = _gold_evidence_from_answer(answer)
            sentence_index = answer.get("sentence_index")
            if gold is None or not isinstance(sentence_index, list):
                missing_fields = True
                per_question_f1 = {}
                break
            predicted = evidence_metrics.predicted_evidence(
                spans_by_question.get(question_id, []), sentence_index
            )
            score = evidence_metrics.prf(predicted, gold)
            per_question_f1[question_id] = float(score["f1"])
            per_question_em[question_id] = float(score["em"])
        if not per_question_f1:
            continue
        summary = {
            "evidence_f1": _question_mean_summary(
                per_question_f1, n_resamples=n_resamples, seed=seed + 2 * run_index
            ),
            "evidence_em": _question_mean_summary(
                per_question_em, n_resamples=n_resamples, seed=seed + 2 * run_index + 1
            ),
            "payload_used": "parsed_or_salvaged_as_consumed_downstream",
        }
        (appendix if "tiny" in run_id else main)[run_id] = summary
    if not main and not appendix and missing_fields:
        return {
            "status": "unavailable_missing_analysis_labels",
            "required_answer_fields": [
                "evidence_f1/evidence_em",
                "or evidence_gold_labels/sentence_index",
            ],
            "warning": (
                "Evidence scores cannot be reconstructed from hashes or raw extractor "
                "spans alone; analysis refuses to invent gold sentence labels."
            ),
        }
    return {"status": "available", "main": main, "tiny_floor_appendix": appendix}


def _batch_size(record: dict[str, Any]) -> int:
    for key in ("n_calls", "batch_size_actual", "batch_size", "items"):
        value = record.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    for key in ("members", "call_keys", "question_ids"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return len(value)
    raise AnalysisError("batch timing record has no positive item count")


def _batch_wall(record: dict[str, Any]) -> float:
    for key in ("batch_wall_s", "wall_s", "elapsed_s"):
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            return float(value)
    raise AnalysisError("batch timing record has no finite batch wall time")


def _timing_batches(run: RunData) -> list[dict[str, Any]]:
    """Return eligible scored batches or fail on contaminated timing state."""
    if not run.batches:
        return []
    selected = []
    for record in run.batches:
        if record.get("oom") is True:
            raise AnalysisError(
                f"{run.run_id}: timing invalid because an OOM/retry batch was recorded"
            )
        if "timing_eligible" in record:
            eligible = record["timing_eligible"]
        elif "latency_eligible" in record:  # current on-disk schema name
            eligible = record["latency_eligible"]
        else:
            raise AnalysisError(
                f"{run.run_id}: batch {record.get('batch_id')!r} lacks explicit "
                "timing/latency eligibility metadata"
            )
        if not isinstance(eligible, bool):
            raise AnalysisError(f"{run.run_id}: timing eligibility must be boolean")
        if not eligible:
            continue
        if record.get("phase") not in {
            "timing_benchmark", "scored", "measurement", "steady_state"
        }:
            raise AnalysisError(
                f"{run.run_id}: eligible batch {record.get('batch_id')!r} has "
                f"non-measurement phase {record.get('phase')!r}"
            )
        if not record.get("execution_session_id"):
            raise AnalysisError(
                f"{run.run_id}: eligible batch lacks execution_session_id"
            )
        selected.append(record)
    sessions = {record["execution_session_id"] for record in selected}
    if len(sessions) > 1:
        raise AnalysisError(
            f"{run.run_id}: eligible timing batches span {len(sessions)} execution "
            "sessions (resumed timing cannot be pooled)"
        )
    if not selected:
        return []

    timing_calls = run.timing_calls or run.calls
    call_keys = [
        (str(record.get("question_id")), str(record.get("stage")), int(record["call_index"]))
        for record in timing_calls
    ]
    if len(call_keys) != len(set(call_keys)):
        raise AnalysisError(f"{run.run_id}: duplicate agent calls in timing replay")
    if not call_keys:
        raise AnalysisError(f"{run.run_id}: timed batches have no matching agent calls")
    expected_questions = int((run.timing_meta or run.meta).get("n") or 0)
    observed_questions = {key[0] for key in call_keys}
    if expected_questions <= 0 or len(observed_questions) != expected_questions:
        raise AnalysisError(
            f"{run.run_id}: timing calls cover {len(observed_questions)} unique "
            f"questions, expected {expected_questions}"
        )
    for record in timing_calls:
        eligible = record.get("timing_eligible", record.get("latency_eligible"))
        if eligible is not True:
            raise AnalysisError(
                f"{run.run_id}: timing replay mixes eligible and noneligible agent calls"
            )
        if record.get("execution_session_id") not in sessions:
            raise AnalysisError(f"{run.run_id}: timing call/session mismatch")
        timing_manifest_hash = _meta_manifest_hash(run.timing_meta or run.meta)
        if record.get("question_manifest_sha256") != timing_manifest_hash:
            raise AnalysisError(f"{run.run_id}: timing call manifest mismatch")

    timing_manifest_hash = _meta_manifest_hash(run.timing_meta or run.meta)
    if not timing_manifest_hash or timing_manifest_hash == EXPECTED_FINAL_MANIFEST_SHA256:
        raise AnalysisError(f"{run.run_id}: timing cohort is not explicitly excluded")
    expected_repetitions = int(
        run.timing_repetitions
        or (run.timing_meta or {}).get("timing_repetitions")
        or 1
    )
    members_by_repeat: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    for record in selected:
        members = record.get("members")
        if not isinstance(members, list) or not members:
            raise AnalysisError(f"{run.run_id}: timed batch lacks canonical members")
        if record.get("question_manifest_sha256") != timing_manifest_hash:
            raise AnalysisError(f"{run.run_id}: timed batch manifest mismatch")
        repeat = record.get("timing_repeat")
        if not isinstance(repeat, int) or not 0 <= repeat < expected_repetitions:
            raise AnalysisError(f"{run.run_id}: invalid or missing timing_repeat")
        stage = str(record.get("stage"))
        members_by_repeat[repeat].extend(
            (str(member["question_id"]), stage, int(member["call_index"]))
            for member in members
        )
    if set(members_by_repeat) != set(range(expected_repetitions)):
        raise AnalysisError(f"{run.run_id}: incomplete timing repetitions")
    for repeat, member_keys in members_by_repeat.items():
        if len(member_keys) != len(set(member_keys)):
            raise AnalysisError(
                f"{run.run_id}: timing repeat {repeat} duplicates call membership"
            )
        if set(member_keys) != set(call_keys):
            raise AnalysisError(
                f"{run.run_id}: partial timing coverage in repeat {repeat}; batches "
                f"cover {len(set(member_keys))} of {len(set(call_keys))} replay calls"
            )

    timing_meta = run.timing_meta or run.meta
    expected_fingerprints = timing_meta.get("stage_config_fingerprints") or {}
    for stage in {key[1] for key in call_keys}:
        call_fingerprints = {
            record.get("config_fingerprint")
            for record in timing_calls if record.get("stage") == stage
        }
        batch_fingerprints = {
            record.get("config_fingerprint")
            for record in selected if record.get("stage") == stage
        }
        expected = expected_fingerprints.get(stage)
        if not expected or call_fingerprints != {expected} or batch_fingerprints != {expected}:
            raise AnalysisError(f"{run.run_id}/{stage}: timing configuration fingerprint mismatch")
    return selected


def _system_time_draws(
    records: list[dict[str, Any]],
    *,
    n_questions: int,
    n_resamples: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    if n_questions <= 0:
        raise AnalysisError("system timing needs a positive scored question count")
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        group = f"r{record.get('timing_repeat')}:{record.get('stage', 'system')}"
        grouped[group].append(_batch_wall(record))
    estimate = sum(sum(values) for values in grouped.values()) / n_questions
    rng = np.random.default_rng(seed)
    draws = np.zeros(n_resamples, dtype=float)
    # Stratify by stage so a pipeline with many extractor batches does not
    # bootstrap away an entire role. Each recorded batch remains the timing unit.
    for values in grouped.values():
        array = np.asarray(values, dtype=float)
        indices = rng.integers(0, len(array), size=(n_resamples, len(array)))
        draws += array[indices].sum(axis=1) / n_questions
    return estimate, draws


def analyze_latency_batches(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze explicit batch timings and never duplicated call ``latency_s``."""
    output: dict[str, Any] = {
        "primary_metric": "whole_system_steady_state_inverse_throughput",
        "primary_unit": "seconds_per_excluded_timing_question_on_A100",
        "resampling_unit": "recorded_batch",
        "call_latency_fields_used": False,
        "baseline_run": BASELINE_RUN,
        "runs": {},
    }
    eligible_by_run = {
        run_id: _timing_batches(run) for run_id, run in sorted(runs.items())
    }
    baseline_rates: dict[str, float] = {}
    if BASELINE_RUN in runs:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in eligible_by_run[BASELINE_RUN]:
            grouped[str(record.get("stage", "system"))].append(record)
        for stage, records in grouped.items():
            numerator = sum(_batch_wall(record) for record in records)
            denominator = sum(_batch_size(record) for record in records)
            baseline_rates[stage] = numerator / denominator

    system_draws: dict[str, np.ndarray] = {}
    system_estimates: dict[str, float] = {}
    for run_index, (run_id, run) in enumerate(sorted(runs.items())):
        records_for_run = eligible_by_run[run_id]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records_for_run:
            grouped[str(record.get("stage", "system"))].append(record)
        if not grouped:
            continue
        stages: dict[str, Any] = {}
        for stage_index, (stage, records) in enumerate(sorted(grouped.items())):
            clusters = {
                str(record.get("batch_id", index)): (_batch_wall(record), _batch_size(record))
                for index, record in enumerate(records)
            }
            if len(clusters) != len(records):
                raise AnalysisError(f"{run_id}/{stage}: duplicate batch_id")
            result = clustered_ratio_bootstrap(
                clusters,
                n_resamples=n_resamples,
                seed=seed + run_index * 10 + stage_index,
            )
            result["n_batches"] = len(records)
            total_wall = sum(_batch_wall(record) for record in records)
            prompt_tokens = sum(float(record.get("prompt_tokens_total") or 0) for record in records)
            output_tokens = sum(float(record.get("output_tokens_total") or 0) for record in records)
            result["prompt_tokens_per_second"] = (
                prompt_tokens / total_wall if total_wall else None
            )
            result["output_tokens_per_second"] = (
                output_tokens / total_wall if total_wall else None
            )
            baseline = baseline_rates.get(stage)
            result["ratio_to_uniform_ma_3b_fp16"] = (
                float(result["estimate"]) / baseline if baseline else None
            )
            stages[stage] = result
        system_estimate, draws = _system_time_draws(
            records_for_run,
            n_questions=int(run.timing_n_questions or len(run.answers)),
            n_resamples=n_resamples,
            seed=seed + 10_000 + run_index,
        )
        system_estimates[run_id] = system_estimate
        system_draws[run_id] = draws
        ordered = np.sort(draws)
        lo = float(ordered[int(0.025 * n_resamples)])
        hi = float(ordered[min(int(0.975 * n_resamples), n_resamples - 1)])
        total_calls = sum(_batch_size(record) for record in records_for_run)
        total_wall = sum(_batch_wall(record) for record in records_for_run)
        prompt_tokens = sum(
            float(record.get("prompt_tokens_total") or 0) for record in records_for_run
        )
        output_tokens = sum(
            float(record.get("output_tokens_total") or 0) for record in records_for_run
        )
        output["runs"][run_id] = {
            "system": {
                "estimate_seconds_per_question": system_estimate,
                "ci_lower": lo,
                "ci_upper": hi,
                "calls_per_question": total_calls / int(
                    run.timing_n_questions or len(run.answers)
                ),
                "prompt_tokens_per_second": prompt_tokens / total_wall if total_wall else None,
                "output_tokens_per_second": output_tokens / total_wall if total_wall else None,
                "prompt_tokens_per_question": prompt_tokens / int(
                    run.timing_n_questions or len(run.answers)
                ),
                "output_tokens_per_question": output_tokens / int(
                    run.timing_n_questions or len(run.answers)
                ),
                "execution_session_id": records_for_run[0]["execution_session_id"],
                "n_batches": len(records_for_run),
                "timing_cohort_questions_per_repeat": int(
                    (run.timing_meta or {}).get("n")
                    or (run.timing_n_questions or len(run.answers))
                ),
                "timing_repetitions": int(run.timing_repetitions or 1),
                "timing_manifest_sha256": _meta_manifest_hash(
                    run.timing_meta or run.meta
                ),
                "gpu_uuid": records_for_run[0].get("gpu_uuid"),
                "gpu_name": records_for_run[0].get("gpu_name"),
                "accuracy_answers_used": False,
                "cold_model_load_wall_s": sum(
                    float(stage.get("model_load_wall_s") or 0)
                    for stage in (
                        ((run.timing_meta or run.meta).get("stages") or {}).values()
                    )
                    if not stage.get("model_load_reused", False)
                ),
                "model_load_included_in_primary_timing": False,
            },
            "stages": stages,
        }
    baseline_system = system_estimates.get(BASELINE_RUN)
    if baseline_system:
        baseline_draws = system_draws[BASELINE_RUN]
        for run_id, estimate in system_estimates.items():
            system = output["runs"][run_id]["system"]
            if run_id == BASELINE_RUN:
                system["ratio_to_uniform_ma_3b_fp16"] = 1.0
                system["ratio_ci_lower"] = 1.0
                system["ratio_ci_upper"] = 1.0
            else:
                ratio_draws = np.divide(
                    system_draws[run_id],
                    baseline_draws,
                    out=np.full(n_resamples, np.nan),
                    where=baseline_draws > 0,
                )
                finite = np.sort(ratio_draws[np.isfinite(ratio_draws)])
                system["ratio_to_uniform_ma_3b_fp16"] = estimate / baseline_system
                system["ratio_ci_lower"] = float(finite[int(0.025 * len(finite))])
                system["ratio_ci_upper"] = float(
                    finite[min(int(0.975 * len(finite)), len(finite) - 1)]
                )
    if not output["runs"]:
        output["status"] = "unavailable_no_batch_records"
        output["warning"] = (
            "Per-call latency_s is deliberately ignored because it duplicates one "
            "batch wall time across every call and would pseudo-replicate timings."
        )
    else:
        output["status"] = "available"
        output["interpretation"] = (
            "Ratios are relative inverse throughput on the recorded A100, not "
            "edge-device latency estimates. Model loading is excluded here and "
            "must be co-reported from run metadata."
        )
    return output


def selector_effects(
    runs: dict[str, RunData],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[
    float,
    dict[str, dict[str, float]],
    dict[str, dict[str, np.ndarray]],
    int,
]:
    required = [BASELINE_RUN] + [
        role_run(role, SELECTOR_RUN_SUFFIX[configuration])
        for role in ROLES
        for configuration in SELECTOR_CONFIGS
        if configuration != "base_fp16"
    ]
    _require_runs(runs, required, "allocation selector")
    baseline = runs[BASELINE_RUN]
    baseline_f1 = 100 * answer_mean(baseline, "f1")
    paired_effects: dict[str, dict[str, float]] = {}
    for role in ROLES:
        for configuration in SELECTOR_CONFIGS:
            if configuration == "base_fp16":
                continue
            run = runs[role_run(role, SELECTOR_RUN_SUFFIX[configuration])]
            paired_effects[f"{role}|{configuration}"] = paired_differences(
                run, baseline, "f1"
            )
    names, estimates, draws, question_ids = joint_bootstrap_draws(
        paired_effects,
        n_resamples=n_resamples,
        seed=seed,
    )
    by_name = {name: column for column, name in enumerate(names)}
    effects: dict[str, dict[str, float]] = {}
    effect_draws: dict[str, dict[str, np.ndarray]] = {}
    zeros = np.zeros(n_resamples, dtype=float)
    for role in ROLES:
        effects[role] = {"base_fp16": 0.0}
        effect_draws[role] = {"base_fp16": zeros}
        for configuration in SELECTOR_CONFIGS:
            if configuration == "base_fp16":
                continue
            column = by_name[f"{role}|{configuration}"]
            effects[role][configuration] = float(estimates[column])
            effect_draws[role][configuration] = draws[:, column]
    return baseline_f1, effects, effect_draws, len(question_ids)


def selector_footprints(runs: dict[str, RunData]) -> dict[str, float]:
    _require_runs(
        runs,
        [UNIFORM_RUN[configuration] for configuration in SELECTOR_CONFIGS],
        "allocation selector memory",
    )
    return {
        configuration: _memory_mib(runs[UNIFORM_RUN[configuration]])
        for configuration in SELECTOR_CONFIGS
    }


def select_allocation(
    *,
    baseline_f1_points: float,
    role_effects_points: dict[str, dict[str, float]],
    role_effect_draws_points: dict[str, dict[str, np.ndarray]],
    footprints_mib: dict[str, float],
    margin_points: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Enumerate exactly 4^4 allocations with a fixed charge per config.

    Candidate F1 is the prespecified additive approximation from single-role
    effects.  Memory is the sum of *distinct* configuration footprints, so using
    the same model for four roles costs once while introducing a mixed precision
    or size pays the whole additional resident copy.
    """
    if set(footprints_mib) != set(SELECTOR_CONFIGS):
        raise AnalysisError(
            "selector footprints must contain exactly "
            f"{SELECTOR_CONFIGS}; got {tuple(footprints_mib)}"
        )
    if margin_points < 0:
        raise ValueError("noninferiority margin must be non-negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    draw_count: int | None = None
    for role in ROLES:
        if set(role_effects_points.get(role, {})) != set(SELECTOR_CONFIGS):
            raise AnalysisError(
                f"selector effects for {role} must contain exactly {SELECTOR_CONFIGS}"
            )
        if set(role_effect_draws_points.get(role, {})) != set(SELECTOR_CONFIGS):
            raise AnalysisError(
                f"selector bootstrap draws for {role} must contain exactly "
                f"{SELECTOR_CONFIGS}"
            )
        for configuration in SELECTOR_CONFIGS:
            values = np.asarray(role_effect_draws_points[role][configuration], dtype=float)
            if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
                raise AnalysisError(
                    f"invalid selector bootstrap draws for {role}/{configuration}"
                )
            if draw_count is None:
                draw_count = len(values)
            elif len(values) != draw_count:
                raise AnalysisError("selector bootstrap columns have unequal draw counts")

    assert draw_count is not None
    threshold = baseline_f1_points - margin_points
    lower_index = int((alpha / 2) * draw_count)
    upper_index = min(int((1 - alpha / 2) * draw_count), draw_count - 1)
    feasible: list[dict[str, Any]] = []
    for choices in itertools.product(SELECTOR_CONFIGS, repeat=len(ROLES)):
        allocation = dict(zip(ROLES, choices))
        predicted_effect = sum(
            role_effects_points[role][allocation[role]] for role in ROLES
        )
        predicted_f1 = baseline_f1_points + predicted_effect
        assignment_draws = sum(
            (
                np.asarray(role_effect_draws_points[role][allocation[role]], dtype=float)
                for role in ROLES
            ),
            start=np.zeros(draw_count, dtype=float),
        )
        ordered_draws = np.sort(assignment_draws)
        effect_lower = float(ordered_draws[lower_index])
        effect_upper = float(ordered_draws[upper_index])
        distinct = sorted(set(choices))
        memory = sum(float(footprints_mib[configuration]) for configuration in distinct)
        # Conservative NI feasibility: the two-sided 95% lower confidence bound
        # on assignment-minus-baseline must clear the approved -1.0 pp margin.
        if effect_lower + 1e-12 >= -margin_points:
            feasible.append({
                "allocation": allocation,
                "predicted_f1_points": predicted_f1,
                "predicted_effect_vs_baseline_points": predicted_effect,
                "predicted_effect_ci_lower": effect_lower,
                "predicted_effect_ci_upper": effect_upper,
                "deduplicated_memory_mib": memory,
                "distinct_configurations": distinct,
            })
    if not feasible:
        # Uniform FP16 is mathematically feasible for a non-negative margin; this
        # guard catches corrupt effects/contracts rather than inventing a result.
        raise AnalysisError("allocation selector found no feasible assignment")

    selected = min(
        feasible,
        key=lambda candidate: (
            candidate["deduplicated_memory_mib"],
            -candidate["predicted_f1_points"],
            len(candidate["distinct_configurations"]),
            tuple(candidate["allocation"][role] for role in ROLES),
        ),
    )
    return {
        "candidate_count": len(SELECTOR_CONFIGS) ** len(ROLES),
        "feasible_count": len(feasible),
        "criterion": {
            "primary_metric": "f1",
            "noninferiority_margin_points": margin_points,
            "threshold_f1_points": threshold,
            "memory_objective": (
                "minimum_deduplicated_concurrent_model_footprint_mib"
            ),
            "accuracy_model": "additive_single_role_f1_point_effects",
            "feasibility_rule": (
                "two_sided_95pct_bootstrap_lower_bound_assignment_minus_baseline_"
                "at_least_negative_margin"
            ),
            "bootstrap_draws": draw_count,
            "bootstrap_shared_across_all_12_nonzero_role_effects": True,
        },
        "tie_break": [
            "minimum_memory",
            "maximum_predicted_f1",
            "fewest_distinct_configurations",
            "lexical_role_order",
        ],
        "selected": selected,
    }


def _selector_margin(config: dict[str, Any]) -> float:
    contract = config.get("allocation_selector") or {}
    for key in ("noninferiority_margin_f1_points", "margin_f1_points", "margin_points"):
        if key in contract:
            return float(contract[key])
    criterion = contract.get("criterion") or {}
    for key in ("noninferiority_margin_f1_points", "margin_f1_points", "margin_points"):
        if key in criterion:
            return float(criterion[key])
    return 1.0


def validate_selector_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("allocation_selector")
    if not isinstance(contract, dict) or contract.get("enabled") is not True:
        raise AnalysisError("allocation_selector must be present and enabled")
    if set((contract.get("candidates") or {}).keys()) != set(SELECTOR_CONFIGS):
        raise AnalysisError(
            "allocation_selector.candidates must be the closed set "
            f"{SELECTOR_CONFIGS}; 0.5B is appendix-only"
        )
    if contract.get("candidate_allocation_count") != len(SELECTOR_CONFIGS) ** len(ROLES):
        raise AnalysisError("allocation_selector candidate count must be exactly 256")
    if contract.get("output_run_id") != OPTIMIZED_RUN:
        raise AnalysisError(f"allocation_selector output_run_id must be {OPTIMIZED_RUN}")
    if "tiny_fp16" not in (contract.get("excluded_candidates") or {}):
        raise AnalysisError("allocation_selector must explicitly exclude tiny_fp16")
    outputs = contract.get("output_artifacts") or {}
    if set(outputs) != {"selection_trace", "executable_config"}:
        raise AnalysisError(
            "allocation_selector.output_artifacts must define selection_trace and "
            "executable_config"
        )
    return contract


def materialize_selection(
    config: dict[str, Any],
    runs: dict[str, RunData],
    output_dir: Path,
    *,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    contract = validate_selector_contract(config)
    required_resamples = int(
        (config.get("analysis") or {}).get(
            "paired_bootstrap_replicates", DEFAULT_BOOTSTRAP_RESAMPLES
        )
    )
    if n_resamples != required_resamples:
        raise AnalysisError(
            f"refusing to freeze selection with {n_resamples} bootstrap draws; "
            f"the analysis contract requires {required_resamples}"
        )
    baseline_f1, effects, effect_draws, n_questions = selector_effects(
        runs,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
    )
    footprints = selector_footprints(runs)
    margin = _selector_margin(config)
    result = select_allocation(
        baseline_f1_points=baseline_f1,
        role_effects_points=effects,
        role_effect_draws_points=effect_draws,
        footprints_mib=footprints,
        margin_points=margin,
    )
    allocation = result["selected"]["allocation"]
    run_definition = {
        role: copy.deepcopy(SELECTOR_TREATMENT[allocation[role]]) for role in ROLES
    }
    run_config_hash = content_hash(run_definition)
    distinct = set(allocation.values())
    reuse_existing = bool(contract.get("reuse_existing_run_when_identical", True))
    existing_run_id = UNIFORM_RUN[next(iter(distinct))] if len(distinct) == 1 else None
    execution_run_id = existing_run_id if reuse_existing and existing_run_id else OPTIMIZED_RUN
    materialized_new_run = execution_run_id == OPTIMIZED_RUN
    input_ids = [BASELINE_RUN] + sorted(
        role_run(role, SELECTOR_RUN_SUFFIX[configuration])
        for role in ROLES
        for configuration in SELECTOR_CONFIGS
        if configuration != "base_fp16"
    ) + sorted(UNIFORM_RUN.values())
    configured_manifest_hash = (config.get("dataset") or {}).get("manifest_sha256")
    if configured_manifest_hash != EXPECTED_FINAL_MANIFEST_SHA256:
        raise AnalysisError(
            "selector requires the frozen final question manifest hash "
            f"{EXPECTED_FINAL_MANIFEST_SHA256}; config has {configured_manifest_hash!r}"
        )
    for run_id in input_ids:
        run_manifest_hash = _run_manifest_hash(runs[run_id])
        if run_manifest_hash != configured_manifest_hash:
            raise AnalysisError(
                f"{run_id}: metadata question-manifest hash {run_manifest_hash!r} "
                f"does not match frozen {configured_manifest_hash}"
            )
    source_files = {
        run_id: {
            "jsonl": file_hash(runs[run_id].jsonl_path),
            "meta": file_hash(runs[run_id].meta_path),
        }
        for run_id in input_ids
    }
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "run_id": OPTIMIZED_RUN,
        "execution_run_id": execution_run_id,
        "materialized_new_run": materialized_new_run,
        "status": "frozen",
        "claim_status": "exploratory_same_sample_post_selection",
        "zero_point_five_b_excluded": True,
        "question_manifest_sha256": configured_manifest_hash,
        "n_questions": n_questions,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": n_resamples,
        "source_config_sha256": content_hash(config),
        "source_file_sha256": source_files,
        "baseline_f1_points": baseline_f1,
        "role_effects_f1_points": effects,
        "configuration_footprints_mib": footprints,
        "selection": result,
        "run_definition": run_definition,
        "run_config_sha256": run_config_hash,
    }

    frozen_config = copy.deepcopy(config)
    if materialized_new_run:
        frozen_config.setdefault("runs", {})[OPTIMIZED_RUN] = run_definition
    frozen_config["allocation_selector"] = copy.deepcopy(contract)
    frozen_config["allocation_selector"].update({
        "status": "frozen",
        "selected_execution_run_id": execution_run_id,
        "selected_allocation": allocation,
        "selected_run_config_sha256": run_config_hash,
    })
    # Hash the semantic selection payload first. The executable config can then
    # reference that immutable decision hash without creating a circular hash.
    artifact["artifact_sha256"] = content_hash(artifact)
    frozen_config["frozen_allocation"] = {
        "run_id": OPTIMIZED_RUN,
        "execution_run_id": execution_run_id,
        "materialized_new_run": materialized_new_run,
        "run_config_sha256": run_config_hash,
        "selection_artifact_sha256": artifact["artifact_sha256"],
        "claim_status": artifact["claim_status"],
        "question_manifest_sha256": configured_manifest_hash,
    }
    artifact["executable_config_sha256"] = content_hash(frozen_config)

    outputs = contract["output_artifacts"]
    artifact_path = _repo_path(outputs["selection_trace"])
    config_path = _repo_path(outputs["executable_config"])
    _write_text_atomic(artifact_path, json.dumps(artifact, indent=2) + "\n")
    _write_text_atomic(
        config_path,
        yaml.safe_dump(frozen_config, sort_keys=False, allow_unicode=True),
    )
    # Read back both files and verify their mutual linkage before advertising a
    # runnable config. A partial/crashed write therefore cannot become a run.
    written_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    written_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not verify_selection_artifact(written_artifact, written_config):
        raise AnalysisError("written selection trace/executable config failed hash verification")
    return {
        **artifact,
        "artifact_path": str(artifact_path),
        "executable_config_path": str(config_path),
    }


def verify_selection_artifact(
    artifact: dict[str, Any],
    executable_config: dict[str, Any] | None = None,
) -> bool:
    """Verify the non-circular trace hash and, optionally, its executable config."""
    expected_artifact_hash = artifact.get("artifact_sha256")
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {
            "artifact_sha256",
            "executable_config_sha256",
            "artifact_path",
            "executable_config_path",
            "contract_artifact_path",
        }
    }
    if not isinstance(expected_artifact_hash, str):
        return False
    if content_hash(payload) != expected_artifact_hash:
        return False
    if content_hash(artifact.get("run_definition")) != artifact.get("run_config_sha256"):
        return False
    if artifact.get("question_manifest_sha256") != EXPECTED_FINAL_MANIFEST_SHA256:
        return False
    if executable_config is not None:
        if content_hash(executable_config) != artifact.get("executable_config_sha256"):
            return False
        frozen = executable_config.get("frozen_allocation") or {}
        if frozen.get("selection_artifact_sha256") != expected_artifact_hash:
            return False
        if frozen.get("run_config_sha256") != artifact.get("run_config_sha256"):
            return False
        execution_run_id = artifact.get("execution_run_id")
        actual_run = (executable_config.get("runs") or {}).get(execution_run_id)
        if actual_run != artifact.get("run_definition"):
            return False
        if content_hash(actual_run) != artifact.get("run_config_sha256"):
            return False
    return True


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final experiment analysis",
        "",
        "F1 is primary; EM is co-reported. All answer contrasts below are direct, "
        "question-paired differences in percentage points.",
        "",
    ]
    accuracy = report.get("per_run_accuracy", {})
    for scope, title in (
        ("main", "Per-run accuracy"),
        ("tiny_floor_appendix", "Appendix: 0.5B lower-limit accuracy"),
    ):
        summaries = accuracy.get(scope, {})
        if not summaries:
            continue
        lines.extend([
            f"## {title}",
            "",
            "| Run | F1 (95% CI) | EM (95% CI) | n |",
            "|---|---:|---:|---:|",
        ])
        for run_id, summary in sorted(summaries.items()):
            f1, em = summary["f1"], summary["em"]
            lines.append(
                f"| {run_id} | {f1['estimate']:.2f} "
                f"[{f1['ci_lower']:.2f}, {f1['ci_upper']:.2f}] | "
                f"{em['estimate']:.2f} [{em['ci_lower']:.2f}, "
                f"{em['ci_upper']:.2f}] | {summary['n_questions']} |"
            )
        lines.append("")
    role_analysis = report.get("role_analysis")
    if role_analysis:
        for key, title in (
            ("primary", "Primary: 3B 8-bit minus 1.5B FP16"),
            ("secondary_4bit", "Secondary: 3B 4-bit minus 1.5B FP16"),
        ):
            family = role_analysis[key]
            lines.extend([
                f"## {title}",
                "",
                "| Role | F1 difference (95% CI) | Holm p | EM difference (95% CI) |",
                "|---|---:|---:|---:|",
            ])
            for role in ROLES:
                f1 = family["metrics"]["f1"][role]
                em = family["metrics"]["em"][role]
                lines.append(
                    f"| {role} | {f1['estimate']:.2f} "
                    f"[{f1['ci_lower']:.2f}, {f1['ci_upper']:.2f}] | "
                    f"{f1['holm_p_value']:.4g} | {em['estimate']:.2f} "
                    f"[{em['ci_lower']:.2f}, {em['ci_upper']:.2f}] |"
                )
            lines.append("")
    role_costs = report.get("role_treatment_costs", {})
    for family_name, family in role_costs.items():
        lines.extend([
            f"## Role cost versus baseline: {family_name}",
            "",
            "Positive values mean accuracy loss. Ranks are descriptive.",
            "",
            "| Role | F1 cost (95% CI) | EM cost (95% CI) |",
            "|---|---:|---:|",
        ])
        for role in ROLES:
            f1 = family["metrics"]["f1"][role]
            em = family["metrics"]["em"][role]
            lines.append(
                f"| {role} | {f1['estimate']:.2f} [{f1['ci_lower']:.2f}, "
                f"{f1['ci_upper']:.2f}] | {em['estimate']:.2f} "
                f"[{em['ci_lower']:.2f}, {em['ci_upper']:.2f}] |"
            )
        lines.append("")
    if report.get("solo_vs_multiagent"):
        comparison = report["solo_vs_multiagent"]
        lines.extend(["## Single-agent versus uniform multi-agent 3B FP16", ""])
        for metric in ("f1", "em"):
            estimate = comparison[metric]
            lines.append(
                f"- {metric.upper()}: {estimate['estimate']:.2f} pp "
                f"[{estimate['ci_lower']:.2f}, {estimate['ci_upper']:.2f}]"
            )
        lines.append("")
    selection = report.get("allocation_selection")
    if selection:
        chosen = selection["selection"]["selected"]
        lines.extend([
            "## Exploratory frozen allocation",
            "",
            f"- Allocation: `{json.dumps(chosen['allocation'], sort_keys=True)}`",
            f"- Predicted F1: {chosen['predicted_f1_points']:.2f}",
            f"- Deduplicated resident weights: {chosen['deduplicated_memory_mib']:.1f} MiB",
            "- Status: exploratory, because selection and evaluation use the same sample.",
            "",
        ])
    mechanism_runs = (report.get("mechanism") or {}).get("runs", {})
    if mechanism_runs:
        lines.extend([
            "## Parse reliability",
            "",
            "All intervals resample questions; salvage does not change parse status.",
            "",
            "| Run | Stage | Failed calls (95% CI) | Questions with any failure (95% CI) |",
            "|---|---|---:|---:|",
        ])
        for run_id, stages in sorted(mechanism_runs.items()):
            for stage, summary in sorted(stages.items()):
                call = summary["per_call_failure_rate"]
                question = summary["per_question_any_failure_rate"]
                lines.append(
                    f"| {run_id} | {stage} | {100*call['estimate']:.2f}% "
                    f"[{100*call['ci_lower']:.2f}, {100*call['ci_upper']:.2f}] | "
                    f"{100*question['estimate']:.2f}% "
                    f"[{100*question['ci_lower']:.2f}, {100*question['ci_upper']:.2f}] |"
                )
        lines.append("")
    evidence = report.get("evidence") or {}
    if evidence.get("status") == "available":
        for scope, title in (
            ("main", "Evidence extraction"),
            ("tiny_floor_appendix", "Appendix: 0.5B evidence extraction"),
        ):
            summaries = evidence.get(scope, {})
            if not summaries:
                continue
            lines.extend([
                f"## {title}",
                "",
                "| Run | Evidence F1 (95% CI) | Evidence EM (95% CI) |",
                "|---|---:|---:|",
            ])
            for run_id, summary in sorted(summaries.items()):
                f1, em = summary["evidence_f1"], summary["evidence_em"]
                lines.append(
                    f"| {run_id} | {100*f1['estimate']:.2f}% "
                    f"[{100*f1['ci_lower']:.2f}, {100*f1['ci_upper']:.2f}] | "
                    f"{100*em['estimate']:.2f}% [{100*em['ci_lower']:.2f}, "
                    f"{100*em['ci_upper']:.2f}] |"
                )
            lines.append("")
    elif evidence.get("status"):
        lines.extend([
            "## Evidence extraction",
            "",
            f"Status: **{evidence['status']}**. {evidence.get('warning', '')}",
            "",
        ])
    churn = report.get("selection_churn") or {}
    for scope, title in (
        ("main", "Selection churn"),
        ("tiny_floor_appendix", "Appendix: 0.5B selection churn"),
    ):
        summaries = churn.get(scope, {})
        if not summaries:
            continue
        lines.extend([
            f"## {title}",
            "",
            "| Run | Stage | Effective churn (95% CI) | Shared calls |",
            "|---|---|---:|---:|",
        ])
        for run_id, summary in sorted(summaries.items()):
            estimate = summary["downstream_effective_payload_churn"]
            lines.append(
                f"| {run_id} | {summary['stage']} | {100*estimate['estimate']:.2f}% "
                f"[{100*estimate['ci_lower']:.2f}, {100*estimate['ci_upper']:.2f}] | "
                f"{summary['shared_calls']} |"
            )
        lines.append("")
    counts = report.get("call_and_token_counts") or {}
    for scope, title in (
        ("main", "Call and token volume"),
        ("tiny_floor_appendix", "Appendix: 0.5B call and token volume"),
    ):
        summaries = counts.get(scope, {})
        if not summaries:
            continue
        lines.extend([
            f"## {title}",
            "",
            "| Run | Calls/question | Prompt tokens/question | Output tokens/question |",
            "|---|---:|---:|---:|",
        ])
        for run_id, summary in sorted(summaries.items()):
            lines.append(
                f"| {run_id} | {summary['calls']['estimate']:.2f} | "
                f"{summary['prompt_tokens']['estimate']:.1f} | "
                f"{summary['output_tokens']['estimate']:.1f} |"
            )
        lines.append("")
    latency_runs = (report.get("latency") or {}).get("runs", {})
    if latency_runs:
        lines.extend([
            "## A100 whole-system inverse throughput",
            "",
            "| Run | seconds/question (95% CI) | ratio to uniform MA 3B FP16 | calls/question |",
            "|---|---:|---:|---:|",
        ])
        for run_id, timing in sorted(latency_runs.items()):
            system = timing["system"]
            ratio = system.get("ratio_to_uniform_ma_3b_fp16")
            ratio_text = f"{ratio:.3f}" if ratio is not None else "n/a"
            lines.append(
                f"| {run_id} | {system['estimate_seconds_per_question']:.4f} "
                f"[{system['ci_lower']:.4f}, {system['ci_upper']:.4f}] | "
                f"{ratio_text} | "
                f"{system['calls_per_question']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_report(
    config: dict[str, Any],
    runs: dict[str, RunData],
    output_dir: Path,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = 0,
    freeze_selection: bool = True,
    require_complete: bool = True,
) -> dict[str, Any]:
    if require_complete:
        validate_campaign_completeness(config, runs)
    validate_final_cohort(config, runs)
    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_contract": {
            "primary_metric": "f1",
            "secondary_coreported_metric": "em",
            "answer_resampling_unit": "question",
            "call_metric_resampling_unit": "question_cluster",
            "latency_resampling_unit": "batch",
            "n_resamples": n_resamples,
            "bootstrap_seed": bootstrap_seed,
            "ci_overlap_used_for_inference": False,
            "zero_point_five_b_scope": "lower_limit_appendix_only",
        },
        "runs_loaded": sorted(runs),
    }
    report["per_run_accuracy"] = analyze_accuracy_runs(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 50
    )
    family_ids = [
        role_run(role, suffix)
        for role in ROLES
        for suffix in ("8bit", "4bit", "small")
    ]
    if all(run_id in runs for run_id in family_ids):
        report["role_analysis"] = analyze_role_families(
            runs, n_resamples=n_resamples, seed=bootstrap_seed
        )
        report["near_memory_match"] = near_memory_match(runs)
    else:
        report["role_analysis_status"] = {
            "status": "incomplete",
            "missing_runs": [run_id for run_id in family_ids if run_id not in runs],
        }

    report["role_treatment_costs"] = analyze_role_costs(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 150
    )

    report["solo_vs_multiagent"] = analyze_solo(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 200
    )
    report["optimized_vs_baseline"] = analyze_optimized_system(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 225
    )
    report["uniform_pareto"] = analyze_uniform_pareto(runs)
    report["tiny_appendix"] = analyze_tiny_appendix(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 300
    )
    report["mechanism"] = analyze_parse_failures(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 400
    )
    report["selection_churn"] = analyze_selection_churn(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 425
    )
    report["evidence"] = analyze_evidence(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 450
    )
    report["call_and_token_counts"] = analyze_call_counts(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 475
    )
    report["latency"] = analyze_latency_batches(
        runs, n_resamples=n_resamples, seed=bootstrap_seed + 500
    )

    selector_ids = [BASELINE_RUN] + [
        role_run(role, SELECTOR_RUN_SUFFIX[configuration])
        for role in ROLES
        for configuration in SELECTOR_CONFIGS
        if configuration != "base_fp16"
    ] + list(UNIFORM_RUN.values())
    configured_accuracy_complete = all(
        run_id in runs and bool(runs[run_id].answers)
        for run_id in (config.get("runs") or {})
        if run_id != OPTIMIZED_RUN
    )
    if (
        freeze_selection
        and configured_accuracy_complete
        and all(run_id in runs for run_id in selector_ids)
    ):
        # A derived post-selection config is already cryptographically linked to
        # the selector artifact that produced its optimized run.  Re-freezing
        # here would overwrite that artifact with a new hash after completeness
        # validation, severing the optimized result's provenance.
        if config.get("frozen_allocation"):
            frozen_artifact = _existing_selection_artifact(config)
            if frozen_artifact is None:
                raise AnalysisError(
                    "frozen allocation config is missing its selector artifact"
                )
            outputs = (config.get("allocation_selector") or {}).get(
                "output_artifacts"
            ) or {}
            report["allocation_selection"] = {
                **frozen_artifact,
                "artifact_path": str(_repo_path(outputs["selection_trace"])),
                "executable_config_path": str(
                    _repo_path(outputs["executable_config"])
                ),
            }
        else:
            report["allocation_selection"] = materialize_selection(
                config,
                runs,
                output_dir,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed + 600,
            )
        execution_run_id = report["allocation_selection"]["execution_run_id"]
        report["selected_system_actual_vs_baseline"] = analyze_optimized_system(
            runs,
            n_resamples=n_resamples,
            seed=bootstrap_seed + 625,
            selected_run_id=execution_run_id,
        )
        if (
            require_complete
            and report["allocation_selection"].get("materialized_new_run") is True
            and (
                OPTIMIZED_RUN not in runs
                or not runs[OPTIMIZED_RUN].answers
                or runs[OPTIMIZED_RUN].timing_meta is None
            )
        ):
            raise AnalysisError(
                "selector materialized a distinct optimized allocation; run both its "
                "accuracy and excluded-cohort timing artifacts, then rerun analysis"
            )
    else:
        report["allocation_selection_status"] = {
            "status": "disabled" if not freeze_selection else "incomplete",
            "missing_runs": [run_id for run_id in selector_ids if run_id not in runs],
            "eligible_configurations": list(SELECTOR_CONFIGS),
            "candidate_count": len(SELECTOR_CONFIGS) ** len(ROLES),
            "zero_point_five_b_excluded": True,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--results", default=None, help="defaults to config results_dir")
    parser.add_argument("--output", default="analysis")
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--no-freeze-selection",
        action="store_true",
        help="analyze completed role arms without materializing the exploratory run",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "permit a non-final dry/intermediate report; final analysis fails closed "
            "unless all 22 static accuracy and configured timing arms are present"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    timing_config = config.get("timing") or {}
    timing_cohort = timing_config.get("cohort") or {}
    results_dir = _repo_path(args.results or config.get("results_dir", "results"))
    output_dir = _repo_path(args.output)
    runs = discover_runs(
        results_dir,
        n=int(config["dataset"]["n"]),
        seed=int(config["dataset"]["eval_seed"]),
        model_id=config.get("model_id"),
        timing_manifest_sha256=_configured_timing_manifest_hash(config),
        timing_n=timing_config.get("n", timing_cohort.get("n")),
        timing_seed=timing_config.get("seed", timing_cohort.get("seed")),
        timing_repetitions=timing_config.get(
            "repetitions", timing_cohort.get("repetitions")
        ),
    )
    report = build_report(
        config,
        runs,
        output_dir,
        n_resamples=args.n_resamples,
        bootstrap_seed=args.bootstrap_seed,
        freeze_selection=not args.no_freeze_selection,
        require_complete=not args.allow_incomplete,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Loaded {len(runs)} runs for n={config['dataset']['n']} "
          f"seed={config['dataset']['eval_seed']}")
    print(f"Wrote {report_path}")
    print(f"Wrote {markdown_path}")
    if report.get("allocation_selection"):
        allocation = report["allocation_selection"]
        print(f"Frozen exploratory allocation: {allocation['run_definition']}")
        print(f"Executable config: {allocation['executable_config_path']}")


if __name__ == "__main__":
    main()
