"""Validate the frozen n=200 pilot and emit the mandatory stop/go artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import prompts  # noqa: E402
from src.metrics import (  # noqa: E402
    exact_match,
    exact_mcnemar,
    f1_score,
    joint_paired_bootstrap,
)
from src.contracts import EXPERIMENT_SCHEMA, PILOT_GATE_SCHEMA_VERSION  # noqa: E402
from src.runner import resolve_treatments  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_pilot_run(
    config: dict, run_id: str
) -> tuple[dict, list[dict], Path, list[dict]]:
    results_dir = repo_path(config.get("results_dir", "results"))
    local_smoke = bool(config.get("local_smoke"))
    matches = []
    for meta_path in results_dir.glob("*.meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("run_id") == run_id
            and meta.get("pilot_mode") is True
            and bool(meta.get("local_smoke_mode")) == local_smoke
        ):
            matches.append((meta_path, meta))
    if len(matches) != 1:
        raise RuntimeError(
            f"pilot requires exactly one {run_id} artifact; found {len(matches)}"
        )
    meta_path, meta = matches[0]
    jsonl_path = meta_path.with_name(
        meta_path.name.removesuffix(".meta.json") + ".jsonl"
    )
    pilot = config["pilot"]
    if (
        meta.get("metadata_complete") is not True
        or int(meta.get("n", -1)) != int(pilot["n"])
        or int(meta.get("seed", -1)) != int(pilot["seed"])
        or meta.get("question_ids_sha256") != pilot["manifest_sha256"]
        or meta.get("pilot_manifest_file_sha256") != pilot["manifest_file_sha256"]
        or not jsonl_path.exists()
        or meta.get("jsonl_sha256") != sha256_file(jsonl_path)
        or bool(meta.get("local_smoke_mode")) != local_smoke
        or (
            local_smoke
            and meta.get("local_smoke") != config.get("local_smoke")
        )
    ):
        raise RuntimeError(f"{run_id}: incomplete or stale pilot artifact")
    expected_treatments = resolve_treatments(config, run_id)
    expected_fingerprints = {
        stage: treatment["config_fingerprint"]
        for stage, treatment in expected_treatments.items()
    }
    if meta.get("stage_config_fingerprints") != expected_fingerprints:
        raise RuntimeError(f"{run_id}: pilot treatment differs from config")
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fingerprint = meta.get("experiment_fingerprint")
    payload = meta.get("experiment_fingerprint_payload")
    retrieval_meta = meta.get("retrieval") or {}
    retrieval_config = config.get("retrieval") or {}
    if (
        not fingerprint
        or not isinstance(payload, dict)
        or content_hash(payload) != fingerprint
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or payload.get("architecture") != config.get("architecture")
        or tuple(payload.get("pipeline_stages") or ()) != tuple(prompts.PIPELINE_STAGES)
        or payload.get("stage_role") != prompts.STAGE_ROLE
        or payload.get("prompt_bundle_version") != prompts.PROMPT_BUNDLE_VERSION
        or payload.get("prompt_template_sha256") != prompts.prompt_template_hashes()
        or payload.get("max_new_tokens") != prompts.MAX_NEW_TOKENS
        or payload.get("retrieval") != retrieval_meta
        or retrieval_meta.get("corpus_sha256")
        != retrieval_config.get("expected_corpus_sha256")
        or int(retrieval_meta.get("corpus_passages", -1))
        != int(retrieval_config.get("expected_corpus_passages", -2))
        or retrieval_meta.get("algorithm") != retrieval_config.get("algorithm")
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
        raise RuntimeError(f"{run_id}: pilot experiment fingerprint is stale")
    if not fingerprint or any(
        record.get("experiment_fingerprint") != fingerprint
        for record in records
        if record.get("record_type") in {"agent_call", "batch", "answer"}
    ):
        raise RuntimeError(f"{run_id}: pilot record fingerprint mismatch")
    scored_records = [
        record for record in records
        if record.get("record_type") in {"agent_call", "answer"}
        or (
            record.get("record_type") == "batch"
            and record.get("phase") == "scored"
        )
    ]
    if not scored_records or any(
        record.get("question_manifest_sha256") != pilot["manifest_sha256"]
        for record in scored_records
    ):
        raise RuntimeError(f"{run_id}: pilot scored-record cohort mismatch")
    answers = [record for record in records if record.get("record_type") == "answer"]
    required_retrieval_fields = {
        "retrieval_gold_title_recall",
        "retrieval_query_count",
        "retrieval_step_count",
        "retrieval_anchor_gold_title_recall",
        "retrieval_followup_eligible_step_count",
        "retrieval_followup_fired_step_count",
        "retrieval_grounded_followup_firing_rate",
        "retrieval_incremental_task_gold_title_recall",
    }
    if any(not required_retrieval_fields <= answer.keys() for answer in answers):
        raise RuntimeError(f"{run_id}: missing repaired retrieval telemetry")
    return meta, answers, jsonl_path, records


def check_pilot(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    local_smoke = bool(config.get("local_smoke"))
    if local_smoke:
        from scripts.run_retrieval_smoke import validate_smoke_config

        validate_smoke_config(config)
        smoke = config["local_smoke"]
        source_path = repo_path(smoke["source_config_path"])
        if not source_path.exists() or sha256_file(source_path) != smoke.get(
            "source_config_sha256"
        ):
            raise RuntimeError("local smoke source config is missing or stale")
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        pilot_keys = (
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "n",
            "seed",
            "retrieval_strata_counts",
        )
        if (
            config.get("dataset") != source.get("dataset")
            or config.get("retrieval") != source.get("retrieval")
            or any(
                (config.get("pilot") or {}).get(key)
                != (source.get("pilot") or {}).get(key)
                for key in pilot_keys
            )
        ):
            raise RuntimeError("local smoke changed a frozen source input")
    pilot = config.get("pilot") or {}
    run_ids = list(pilot.get("run_ids") or ())
    if run_ids != ["baseline", "single_fp16"]:
        raise RuntimeError("pilot.run_ids must be [baseline, single_fp16]")
    manifest_path = repo_path(pilot["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        sha256_file(manifest_path) != pilot.get("manifest_file_sha256")
        or manifest.get("question_ids_sha256") != pilot.get("manifest_sha256")
        or len(manifest.get("question_ids") or ()) != int(pilot.get("n", -1))
    ):
        raise RuntimeError("pilot manifest bytes/IDs do not match config")

    loaded = {run_id: _load_pilot_run(config, run_id) for run_id in run_ids}
    fingerprints = {loaded[run_id][0].get("experiment_fingerprint") for run_id in run_ids}
    env_locks = {loaded[run_id][0].get("environment_lock_sha256") for run_id in run_ids}
    git_commits = {loaded[run_id][0].get("git_commit") for run_id in run_ids}
    if len(fingerprints) != 1 or None in fingerprints:
        raise RuntimeError("pilot arms use different experiment fingerprints")
    if len(git_commits) != 1 or None in git_commits or "unknown" in git_commits:
        raise RuntimeError("pilot arms use different environment/code revisions")
    if local_smoke:
        if env_locks != {None}:
            raise RuntimeError("local smoke must not claim a production environment lock")
        metas = [loaded[run_id][0] for run_id in run_ids]
        gpu_names = {meta.get("gpu_name") for meta in metas}
        if len(gpu_names) != 1 or None in gpu_names:
            raise RuntimeError("local smoke arms use different or missing GPUs")
        fingerprint_sets = [
            set((meta.get("stage_config_fingerprints") or {}).values())
            for meta in metas
        ]
        if (
            any(len(values) != 1 for values in fingerprint_sets)
            or fingerprint_sets[0] != fingerprint_sets[1]
            or any(meta.get("batch_size") != config["generation"]["batch_size"] for meta in metas)
        ):
            raise RuntimeError("local smoke arms are not memory matched")
        resolved_revisions = []
        for run_id in run_ids:
            loads = [
                record
                for record in loaded[run_id][3]
                if record.get("record_type") == "model_load"
            ]
            resolved = {
                (
                    record.get("resolved_model_revision"),
                    record.get("resolved_tokenizer_revision"),
                )
                for record in loads
            }
            if not loads or len(resolved) != 1 or any(
                value in {None, "TBD"}
                for pair in resolved
                for value in pair
            ):
                raise RuntimeError("local smoke runtime revision is missing or inconsistent")
            resolved_revisions.append(resolved)
        if resolved_revisions[0] != resolved_revisions[1]:
            raise RuntimeError("local smoke arms resolved different model revisions")
        footprints = {
            meta.get("deduplicated_concurrent_model_footprint_mib") for meta in metas
        }
        if len(footprints) != 1 or None in footprints:
            raise RuntimeError("local smoke arms are not memory matched")
    elif len(env_locks) != 1 or None in env_locks:
        raise RuntimeError("pilot arms use different environment/code revisions")

    answers: dict[str, dict[str, dict]] = {}
    expected_ids = manifest["question_ids"]
    for run_id in run_ids:
        rows = loaded[run_id][1]
        if [row.get("question_id") for row in rows] != expected_ids:
            raise RuntimeError(f"{run_id}: pilot answer IDs/order differ from manifest")
        indexed = {}
        for row in rows:
            if (
                row.get("question_manifest_sha256") != pilot["manifest_sha256"]
                or row.get("retrieval_stratum") not in {"hidden_bridge", "fully_named"}
                or "predicted_answer" not in row
                or "gold_answer" not in row
            ):
                raise RuntimeError(f"{run_id}: malformed pilot answer")
            expected_f1 = f1_score(row["predicted_answer"], row["gold_answer"])
            expected_em = exact_match(row["predicted_answer"], row["gold_answer"])
            if not (
                math.isclose(float(row.get("f1", -1)), expected_f1, abs_tol=1e-12)
                and math.isclose(float(row.get("em", -1)), expected_em, abs_tol=1e-12)
            ):
                raise RuntimeError(f"{run_id}/{row['question_id']}: metric mismatch")
            indexed[row["question_id"]] = row
        answers[run_id] = indexed
    if any(row.get("answer_stage") != "plan_summary" for row in answers["baseline"].values()):
        raise RuntimeError("baseline pilot did not score the MA-RAG plan summary")
    baseline_sources = {"summary_parsed", "summary_salvaged", "qa_fallback", "none"}
    if any(
        row.get("final_answer_source") not in baseline_sources
        for row in answers["baseline"].values()
    ):
        raise RuntimeError("baseline pilot has invalid final-answer provenance")
    solo_sources = {"solo_parsed", "solo_salvaged", "none"}
    if any(
        row.get("final_answer_source") not in solo_sources
        for row in answers["single_fp16"].values()
    ):
        raise RuntimeError("single pilot has invalid final-answer provenance")
    if any(row.get("answer_stage") != "solo" for row in answers["single_fp16"].values()):
        raise RuntimeError("single pilot did not score the one-call solo control")
    observed_strata = {
        label: sum(
            row["retrieval_stratum"] == label
            for row in answers["baseline"].values()
        )
        for label in ("hidden_bridge", "fully_named")
    }
    if observed_strata != pilot.get("retrieval_strata_counts"):
        raise RuntimeError(f"pilot retrieval strata changed: {observed_strata}")
    for question_id in expected_ids:
        if (
            answers["baseline"][question_id]["retrieval_stratum"]
            != answers["single_fp16"][question_id]["retrieval_stratum"]
        ):
            raise RuntimeError("pilot arms disagree on retrieval stratum")

    def mean(run_id: str, metric: str, stratum: str | None = None) -> float:
        values = [
            float(row[metric]) for row in answers[run_id].values()
            if stratum is None or row["retrieval_stratum"] == stratum
        ]
        if not values:
            raise RuntimeError(f"pilot has no {stratum or 'overall'} rows")
        return sum(values) / len(values)

    overall_delta = 100 * (mean("baseline", "f1") - mean("single_fp16", "f1"))
    hidden_delta = 100 * (
        mean("baseline", "f1", "hidden_bridge")
        - mean("single_fp16", "f1", "hidden_bridge")
    )
    fully_named_delta = 100 * (
        mean("baseline", "f1", "fully_named")
        - mean("single_fp16", "f1", "fully_named")
    )
    decision = pilot["decision"]
    paired_f1_differences = {
        question_id: 100 * (
            float(answers["baseline"][question_id]["f1"])
            - float(answers["single_fp16"][question_id]["f1"])
        )
        for question_id in expected_ids
    }
    bootstrap = joint_paired_bootstrap(
        {"overall_f1_points": paired_f1_differences},
        n_resamples=int(decision["paired_bootstrap_resamples"]),
        seed=int(decision["paired_bootstrap_seed"]),
    )["overall_f1_points"]
    mcnemar = exact_mcnemar(
        [int(float(answers["baseline"][question_id]["em"])) for question_id in expected_ids],
        [
            int(float(answers["single_fp16"][question_id]["em"]))
            for question_id in expected_ids
        ],
    )

    hidden_baseline = [
        row for row in answers["baseline"].values()
        if row["retrieval_stratum"] == "hidden_bridge"
    ]
    followup_counts = []
    for row in hidden_baseline:
        values = []
        for field in (
            "retrieval_followup_fired_step_count",
            "retrieval_followup_eligible_step_count",
        ):
            raw = row[field]
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("pilot follow-up firing counts are malformed") from exc
            if (
                isinstance(raw, bool)
                or not math.isfinite(numeric)
                or numeric < 0
                or not numeric.is_integer()
            ):
                raise RuntimeError("pilot follow-up firing counts are malformed")
            values.append(int(numeric))
        fired, eligible = values
        if fired > eligible:
            raise RuntimeError("pilot follow-up firing counts are malformed")
        followup_counts.append((fired, eligible))
    followup_fired = sum(fired for fired, _eligible in followup_counts)
    followup_eligible = sum(eligible for _fired, eligible in followup_counts)
    hidden_followup_firing_rate = (
        followup_fired / followup_eligible if followup_eligible else 0.0
    )

    overall_threshold = float(
        decision["pass_if_multiagent_minus_single_overall_at_least"]
    )
    hidden_threshold = float(
        decision["pass_if_multiagent_minus_single_hidden_bridge_at_least"]
    )
    bootstrap_lower_threshold = float(
        decision["pass_if_paired_bootstrap_95pct_lower_bound_above"]
    )
    mcnemar_threshold = float(
        decision["pass_if_mcnemar_exact_two_sided_p_below"]
    )
    fully_named_threshold = float(
        decision["pass_if_absolute_multiagent_minus_single_fully_named_at_most"]
    )
    firing_threshold = float(
        decision["pass_if_hidden_bridge_followup_firing_rate_at_least"]
    )
    decision_checks = {
        "overall_f1_delta": overall_delta >= overall_threshold,
        "paired_bootstrap_lower_bound": (
            float(bootstrap["ci_lower"]) > bootstrap_lower_threshold
        ),
        "mcnemar_exact_two_sided": (
            float(mcnemar["p_value"]) < mcnemar_threshold
        ),
        "hidden_bridge_f1_delta": hidden_delta >= hidden_threshold,
        "fully_named_f1_delta_sanity": (
            abs(fully_named_delta) <= fully_named_threshold
        ),
        "hidden_bridge_followup_firing": (
            hidden_followup_firing_rate >= firing_threshold
        ),
    }
    passed = all(decision_checks.values())

    strata = (None, "hidden_bridge", "fully_named")
    accuracy = {
        run_id: {
            stratum or "all": {
                metric: 100 * mean(run_id, metric, stratum)
                for metric in ("f1", "em")
            }
            for stratum in strata
        }
        for run_id in run_ids
    }
    retrieval_fields = (
        "retrieval_gold_title_recall",
        "retrieval_all_gold",
        "retrieval_anchor_gold_title_recall",
        "retrieval_task_gold_title_recall",
        "retrieval_step_count",
        "retrieval_query_count",
        "retrieval_task_query_count",
        "retrieval_grounded_followup_firing_rate",
        "retrieval_incremental_task_gold_title_recall",
        "retrieval_zero_result_query_count",
        "retrieval_aggregate_step_count",
        "retrieval_passage_exposures",
        "retrieval_unique_titles",
        "qa_grounded_answer_rate",
        "extractor_normalization_rejected_span_count",
        "qa_evidence_document_count",
        "qa_evidence_prompt_block_count",
        "qa_evidence_filtered_document_count",
        "executed_steps",
        "planner_emitted_depth",
        "plan_was_clamped",
    )
    retrieval_diagnostics = {
        run_id: {
            stratum or "all": {
                field: mean(run_id, field, stratum)
                for field in retrieval_fields
                if all(
                    row.get(field) is not None
                    for row in answers[run_id].values()
                    if stratum is None or row["retrieval_stratum"] == stratum
                )
            }
            for stratum in strata
        }
        for run_id in run_ids
    }
    mechanism = {}
    for run_id in run_ids:
        calls = [
            record for record in loaded[run_id][3]
            if record.get("record_type") == "agent_call"
        ]
        per_role = {}
        for role in sorted({str(record.get("conceptual_role")) for record in calls}):
            selected = [record for record in calls if record.get("conceptual_role") == role]
            per_role[role] = {
                "n_calls": len(selected),
                "parse_success_rate": sum(
                    record.get("parse_status") == "ok" for record in selected
                ) / len(selected),
                "strict_format_success_rate": sum(
                    record.get("strict_format_ok") is True for record in selected
                ) / len(selected),
                "strict_protocol_success_rate": sum(
                    record.get("protocol_ok") is True for record in selected
                ) / len(selected),
            }
        mechanism[run_id] = per_role
    stop_reasons = {
        reason: sum(row.get("stop_reason") == reason for row in answers["baseline"].values())
        for reason in ("plan_complete", "semantic_inability")
    }

    gate_profile = "local_smoke" if local_smoke else "production_a100"
    artifact = {
        "schema_version": PILOT_GATE_SCHEMA_VERSION,
        "gate_profile": gate_profile,
        "status": (
            "PASS_LOCAL_SMOKE" if passed else "FAIL_LOCAL_SMOKE"
        ) if local_smoke else ("GO" if passed else "STOP"),
        "passed": passed,
        "production_eligible": not local_smoke,
        "local_smoke_profile_sha256": (
            content_hash(config["local_smoke"]) if local_smoke else None
        ),
        "source_config_file_sha256": sha256_file(config_path),
        "pilot_manifest_file_sha256": sha256_file(manifest_path),
        "pilot_manifest_sha256": pilot["manifest_sha256"],
        "experiment_fingerprint": next(iter(fingerprints)),
        "environment_lock_sha256": next(iter(env_locks)),
        "git_commit": next(iter(git_commits)),
        "n": pilot["n"],
        "retrieval_strata_counts": observed_strata,
        "final_answer_sources": {
            run_id: {
                source: sum(
                    row.get("final_answer_source") == source
                    for row in answers[run_id].values()
                )
                for source in sorted({
                    row["final_answer_source"]
                    for row in answers[run_id].values()
                })
            }
            for run_id in run_ids
        },
        "run_jsonl_sha256": {
            run_id: sha256_file(loaded[run_id][2]) for run_id in run_ids
        },
        "decision_rule": decision,
        "decision_checks": decision_checks,
        "metrics": {
            "multiagent_minus_single_f1_points_overall": overall_delta,
            "multiagent_minus_single_f1_points_hidden_bridge": hidden_delta,
            "multiagent_minus_single_f1_points_fully_named": fully_named_delta,
            "paired_bootstrap_f1_points_overall": bootstrap,
            "mcnemar_exact_two_sided_em": mcnemar,
            "hidden_bridge_followup_firing": {
                "eligible_steps": followup_eligible,
                "fired_steps": followup_fired,
                "rate": hidden_followup_firing_rate,
            },
            "multiagent_minus_single_em_points_overall": (
                accuracy["baseline"]["all"]["em"]
                - accuracy["single_fp16"]["all"]["em"]
            ),
            "multiagent_minus_single_em_points_hidden_bridge": (
                accuracy["baseline"]["hidden_bridge"]["em"]
                - accuracy["single_fp16"]["hidden_bridge"]["em"]
            ),
        },
        "accuracy_points": accuracy,
        "retrieval_diagnostics": retrieval_diagnostics,
        "mechanism_diagnostics": mechanism,
        "baseline_stop_reasons": stop_reasons,
    }
    artifact["artifact_sha256"] = content_hash(artifact)
    return artifact


def verify_gate(
    config_path: Path, artifact_path: Path, *, require_tracked: bool = False
) -> dict:
    if not artifact_path.exists():
        raise RuntimeError(f"mandatory pilot gate is missing: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = artifact.get("artifact_sha256")
    semantic = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pilot = config.get("pilot") or {}
    lock_path = repo_path(config.get("environment_lock_path", "config/environment.lock.json"))
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    current_config_hash = sha256_file(config_path)
    source_config_ok = artifact.get("source_config_file_sha256") in {
        current_config_hash,
        lock.get("experiment_config_sha256")
        if config.get("frozen_allocation") else None,
    }
    if (
        expected != content_hash(semantic)
        or artifact.get("passed") is not True
        or artifact.get("status") != "GO"
        or artifact.get("gate_profile") != "production_a100"
        or artifact.get("production_eligible") is not True
        or not source_config_ok
        or artifact.get("pilot_manifest_sha256") != pilot.get("manifest_sha256")
        or artifact.get("pilot_manifest_file_sha256")
        != pilot.get("manifest_file_sha256")
        or artifact.get("decision_rule") != pilot.get("decision")
        or artifact.get("n") != pilot.get("n")
        or artifact.get("retrieval_strata_counts")
        != pilot.get("retrieval_strata_counts")
        or not lock_path.exists()
        or artifact.get("environment_lock_sha256") != sha256_file(lock_path)
    ):
        raise RuntimeError("pilot gate is missing, stale, corrupt, or STOP")
    results_dir = repo_path(config.get("results_dir", "results"))
    local_pilot_meta = [
        path for path in results_dir.glob("*.meta.json")
        if json.loads(path.read_text(encoding="utf-8")).get("pilot_mode") is True
    ]
    if local_pilot_meta:
        fresh = check_pilot(config_path)
        for key in (
            "status", "passed", "pilot_manifest_file_sha256",
            "pilot_manifest_sha256", "experiment_fingerprint",
            "environment_lock_sha256", "git_commit", "n",
            "retrieval_strata_counts", "run_jsonl_sha256", "decision_rule",
            "decision_checks", "metrics", "accuracy_points", "retrieval_diagnostics",
            "mechanism_diagnostics", "baseline_stop_reasons",
        ):
            if artifact.get(key) != fresh.get(key):
                raise RuntimeError(f"pilot gate no longer matches pilot artifacts: {key}")
    if require_tracked:
        try:
            relative = artifact_path.resolve().relative_to(ROOT).as_posix()
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                # Compare against HEAD, not merely the index.  A freshly
                # generated certificate that has only been `git add`ed is
                # present in `ls-files` and has no worktree-vs-index diff, but
                # it has not been distributed to any other worker.  Production
                # may proceed only from the exact committed certificate.
                ["git", "diff", "--quiet", "HEAD", "--", relative],
                cwd=ROOT, check=True,
            )
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "pilot GO certificate must be committed unchanged before final workers run"
            ) from exc
    return artifact


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve() if args.output else repo_path(
        config["pilot"]["gate_artifact"]
    )
    if args.verify:
        verify_gate(config_path, output, require_tracked=True)
        print(f"pilot gate verified: {output}")
        return 0
    artifact = check_pilot(config_path)
    write_atomic(output, artifact)
    print(f"pilot gate: {artifact['status']} -> {output}")
    return 0 if artifact["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
