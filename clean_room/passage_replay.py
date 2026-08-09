"""Frozen-upstream passage replay CLI and certified GPU runtime.

The CPU audit reads only committed records.  GPU helpers are imported lazily so
artifact validation and manifest construction never load model weights or
initialize retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_room import passage_replay_core as core
from src import prompts


MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_ID = core.MODEL_ID
PRECISION = core.PRECISION
BATCH_SIZE = core.BATCH_SIZE
EXPECTED_GPU_NAME = "Tesla T4"
EXPECTED_GPU_COMPUTE_CAPABILITY = "7.5"
EXPECTED_CENSUS = {
    "nominal_params": 1_720_574_976,
    "quantized_params": 1_409_286_144,
    "quantized_4bit_params": 1_409_286_144,
    "quantized_8bit_params": 0,
    "quantized_fraction": 0.8190786008502311,
    "requested_precision_fraction": 0.8190786008502311,
    "precision_validation_passed": True,
}
SENTINEL_CONDITION = "reproduction_sentinel"
SENTINEL_PHASE = "reproduction_sentinel"
SENTINEL_FIELDS = (
    "raw_output",
    "parsed",
    "salvaged",
    "prompt_tokens",
    "output_tokens",
    "generated_sequence_tokens",
)
SENTINEL_STAGES = (*core.QA_STAGES, "plan_summary")
TOKENIZER_SNAPSHOT_SCHEMA = "saved-tokenizer-snapshot-v1"
OUTPUT_CEILINGS = {"qa": 96, "plan_summary": 128}
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


@dataclass(frozen=True)
class TokenizerIdentity:
    chat_template_type: str
    chat_template_sha256: str
    files: dict[str, dict]
    snapshot_sha256: str


@dataclass(frozen=True)
class PromptRuntimeAudit:
    message_object_sha256: str
    rendered_chat_sha256: str
    prompt_tokens: int
    output_ceiling_tokens: int
    prompt_plus_ceiling_tokens: int
    recorded_context_window_tokens: int


def version_warnings(source_versions: dict, replay_versions: dict) -> list[dict]:
    """Record package drift without turning version strings into a hard gate."""
    warnings = []
    for package in sorted(set(source_versions) | set(replay_versions)):
        source_value = source_versions.get(package)
        replay_value = replay_versions.get(package)
        if source_value != replay_value:
            warnings.append(
                {
                    "package": package,
                    "source": source_value,
                    "replay": replay_value,
                    "severity": "warning",
                    "gate": "empirical_21_call_sentinel",
                }
            )
    return warnings


def sentinel_batches(source: core.ReplaySource) -> tuple[core.FrozenBatch, ...]:
    batches = tuple(core.source_scored_batches(source, stage)[0] for stage in SENTINEL_STAGES)
    sizes = [len(batch.members) for batch in batches]
    if sizes != [4, 4, 4, 4, 1, 4]:
        raise core.IntegrityError(f"sentinel source batch sizes changed: {sizes!r}")
    return batches


def sentinel_source_calls(source: core.ReplaySource) -> list[dict]:
    records: list[dict] = []
    for batch in sentinel_batches(source):
        for member in batch.members:
            record = source.baseline_index.get(
                (member.question_id, member.stage, member.call_index)
            )
            if record is None:
                raise core.IntegrityError(f"sentinel source call is missing: {member!r}")
            records.append(record)
    if len(records) != 21:
        raise core.IntegrityError(f"sentinel selected {len(records)} calls, expected 21")
    return records


def sentinel_source_batch_records(source: core.ReplaySource) -> list[dict]:
    """Return the six immutable source certificates paired with the sentinel."""
    records = []
    for batch in sentinel_batches(source):
        matches = [
            record
            for record in source.baseline_records
            if record.get("record_type") == "batch"
            and record.get("phase") == "scored"
            and record.get("stage") == batch.stage
            and record.get("batch_ordinal") == batch.ordinal
        ]
        if len(matches) != 1:
            raise core.IntegrityError(
                f"sentinel source telemetry for {batch.stage}/{batch.ordinal} "
                f"has {len(matches)} records"
            )
        records.append(dict(matches[0]))
    return records


def reject_thinking(records: Sequence[dict]) -> None:
    for index, record in enumerate(records):
        raw = record.get("raw_output")
        if isinstance(raw, str) and _THINK_TAG.search(raw):
            raise core.IntegrityError(
                f"thinking tag in generated output at sentinel/treatment member {index}"
            )


def compare_sentinel(source_records: Sequence[dict], replay_records: Sequence[dict]) -> dict:
    source_values = tuple(source_records)
    replay_values = tuple(replay_records)
    if len(source_values) != 21 or len(replay_values) != 21:
        raise core.IntegrityError(
            f"sentinel requires 21 paired calls, got {len(source_values)}/{len(replay_values)}"
        )
    reject_thinking(source_values)
    reject_thinking(replay_values)
    parse_status_matches = []
    mismatches: list[dict] = []
    for index, (source, replay) in enumerate(zip(source_values, replay_values)):
        for identity_field in (
            "question_id",
            "stage",
            "call_index",
            "batch_member_index",
        ):
            if source.get(identity_field) != replay.get(identity_field):
                raise core.IntegrityError(
                    f"sentinel member {index} {identity_field} mismatch: "
                    f"{source.get(identity_field)!r} != {replay.get(identity_field)!r}"
                )
        for field in SENTINEL_FIELDS:
            if source.get(field) != replay.get(field):
                mismatches.append(
                    {
                        "member_index": index,
                        "question_id": source.get("question_id"),
                        "stage": source.get("stage"),
                        "field": field,
                        "source": source.get(field),
                        "replay": replay.get(field),
                    }
                )
        parse_status_matches.append(source.get("parse_status") == replay.get("parse_status"))
    if mismatches:
        first = mismatches[0]
        raise core.IntegrityError(
            f"sentinel mismatch at {first['stage']}/{first['question_id']}: "
            f"{first['field']}"
        )
    return {
        "passed": True,
        "n": 21,
        "hard_fields": list(SENTINEL_FIELDS),
        "parse_status_all_match": all(parse_status_matches),
        "parse_status_matches": parse_status_matches,
        "think_tag_checks": {
            "source": {"checked_calls": 21, "passed": True},
            "replay": {"checked_calls": 21, "passed": True},
        },
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def capture_tokenizer_identity(
    tok,
    *,
    repo_root: Path = ROOT,
    scratch_parent: Path | None = None,
) -> TokenizerIdentity:
    repo_root = Path(repo_root).resolve()
    parent = Path(scratch_parent or (repo_root / ".passage-replay-tmp")).resolve()
    if not _path_is_within(parent, repo_root):
        raise core.IntegrityError("tokenizer snapshot scratch directory is outside repository")
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tokenizer-", dir=parent) as raw:
        snapshot = Path(raw)
        tok.save_pretrained(snapshot)
        file_paths: list[tuple[str, Path]] = []
        for path in snapshot.rglob("*"):
            if path.is_symlink():
                raise core.IntegrityError(f"tokenizer snapshot contains symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise core.IntegrityError(f"tokenizer snapshot contains special file: {path}")
            file_paths.append((path.relative_to(snapshot).as_posix(), path))
        file_paths.sort(key=lambda item: item[0])
        if not file_paths:
            raise core.IntegrityError("tokenizer snapshot contains no regular files")
        files = {
            relative: {
                "sha256": core.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for relative, path in file_paths
        }
    snapshot_payload = {
        "schema": TOKENIZER_SNAPSHOT_SCHEMA,
        "files": [
            {"path": relative, **metadata}
            for relative, metadata in files.items()
        ],
    }
    chat_template = getattr(tok, "chat_template", None)
    if chat_template is None or chat_template == "":
        raise core.IntegrityError("loaded tokenizer has no chat template")
    if isinstance(chat_template, str):
        chat_bytes = chat_template.encode("utf-8")
    else:
        chat_bytes = json.dumps(
            chat_template,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    return TokenizerIdentity(
        chat_template_type=type(chat_template).__name__,
        chat_template_sha256=hashlib.sha256(chat_bytes).hexdigest(),
        files=files,
        snapshot_sha256=core.canonical_json_sha256(snapshot_payload),
    )


def audit_prompt_runtime(
    tok,
    messages: list[dict],
    *,
    prompt_role: str,
    recorded_context_window_tokens: int,
) -> PromptRuntimeAudit:
    if prompt_role not in OUTPUT_CEILINGS:
        raise ValueError(f"unsupported replay prompt role {prompt_role!r}")
    ceiling = OUTPUT_CEILINGS[prompt_role]
    if prompts.MAX_NEW_TOKENS.get(prompt_role) != ceiling:
        raise core.IntegrityError(
            f"{prompt_role} output ceiling drift: "
            f"{prompts.MAX_NEW_TOKENS.get(prompt_role)!r} != {ceiling}"
        )
    if (
        isinstance(recorded_context_window_tokens, bool)
        or not isinstance(recorded_context_window_tokens, int)
        or recorded_context_window_tokens <= 0
    ):
        raise core.IntegrityError("recorded context window is not a positive integer")
    from src import models

    rendered = models.render_chat(tok, messages)
    encoded = tok(
        rendered,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    token_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], list)
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in token_ids
    ):
        raise core.IntegrityError("tokenizer returned an invalid untruncated token sequence")
    prompt_tokens = len(token_ids)
    total = prompt_tokens + ceiling
    if total > recorded_context_window_tokens:
        raise core.IntegrityError(
            f"{prompt_role} prompt plus ceiling exceeds recorded context window: "
            f"{prompt_tokens}+{ceiling}>{recorded_context_window_tokens}"
        )
    return PromptRuntimeAudit(
        message_object_sha256=core.rendered_prompt_sha256(messages),
        rendered_chat_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        prompt_tokens=prompt_tokens,
        output_ceiling_tokens=ceiling,
        prompt_plus_ceiling_tokens=total,
        recorded_context_window_tokens=recorded_context_window_tokens,
    )


def validate_generated_prompt(record: dict, audit: PromptRuntimeAudit) -> None:
    if record.get("prompt_tokens") != audit.prompt_tokens:
        raise core.IntegrityError("generated prompt token count differs from pre-generation audit")
    if record.get("context_window_tokens") != audit.recorded_context_window_tokens:
        raise core.IntegrityError("generated context window differs from source authority")
    if record.get("forced_full_generation") is not False:
        raise core.IntegrityError("replay generation unexpectedly forced the token ceiling")


def _batch_payload(batch: core.FrozenBatch) -> dict:
    return {
        "stage": batch.stage,
        "ordinal": batch.ordinal,
        "batch_id": batch.batch_id,
        "canonical_sha256": batch.canonical_sha256,
        "source_batch_ids": list(batch.source_batch_ids),
        "members": [asdict(member) for member in batch.members],
    }


def condition_fingerprint(
    execution_sha256: str,
    condition: str,
    ordered_ids: Sequence[str],
    batches: Sequence[core.FrozenBatch],
) -> tuple[str, dict]:
    if condition not in core.CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    ids = tuple(ordered_ids)
    payload = {
        "schema": "passage-replay-condition-v1",
        "execution_fingerprint_sha256": execution_sha256,
        "condition": condition,
        "ordered_ids": list(ids),
        "ordered_ids_sha256": core.ordered_ids_sha256(ids),
        "treatment_schema": {
            "passage_header": core.PASSAGE_HEADER,
            "spans_policy": (
                "retain_then_append_passages"
                if condition == core.SPANS_PLUS_PASSAGES
                else "replace_with_passages"
            ),
        },
        "batches": [_batch_payload(batch) for batch in batches],
    }
    return core.canonical_json_sha256(payload), payload


def _require_exact_identity_value(actual, expected, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise core.IntegrityError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise core.IntegrityError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_git_commit(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise core.IntegrityError(f"{label} is not a full lowercase Git commit")
    return value


def _validated_versions(value) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise core.IntegrityError(
            "execution identity library_versions is missing or empty"
        )
    versions: dict[str, str] = {}
    for package, version in value.items():
        if not isinstance(package, str) or not package.strip():
            raise core.IntegrityError(
                "execution identity library_versions has an empty package name"
            )
        if not isinstance(version, str) or not version.strip():
            raise core.IntegrityError(
                f"execution identity library version for {package!r} is empty"
            )
        versions[package] = version
    return versions


def _validated_tokenizer_identity(value) -> dict:
    if not isinstance(value, dict):
        raise core.IntegrityError("execution identity has no tokenizer identity")
    chat_template_type = value.get("chat_template_type")
    if not isinstance(chat_template_type, str) or not chat_template_type:
        raise core.IntegrityError("tokenizer identity chat_template_type is missing")
    chat_template_sha256 = _require_sha256(
        value.get("chat_template_sha256"),
        "tokenizer identity chat_template_sha256",
    )
    snapshot_sha256 = _require_sha256(
        value.get("snapshot_sha256"), "tokenizer identity snapshot_sha256"
    )
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise core.IntegrityError("tokenizer identity files are missing or empty")
    validated_files: dict[str, dict] = {}
    for relative_path, metadata in files.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise core.IntegrityError("tokenizer identity contains an empty file path")
        if not isinstance(metadata, dict):
            raise core.IntegrityError(
                f"tokenizer identity metadata for {relative_path!r} is invalid"
            )
        sha256 = _require_sha256(
            metadata.get("sha256"),
            f"tokenizer identity file {relative_path!r} sha256",
        )
        size_bytes = metadata.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise core.IntegrityError(
                f"tokenizer identity file {relative_path!r} size_bytes is invalid"
            )
        validated_files[relative_path] = {
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return {
        "chat_template_type": chat_template_type,
        "chat_template_sha256": chat_template_sha256,
        "files": validated_files,
        "snapshot_sha256": snapshot_sha256,
    }


def validate_execution_identity(identity: dict) -> dict:
    if not isinstance(identity, dict):
        raise core.IntegrityError("execution identity is not an object")
    required = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": PRECISION,
        "batch_size": BATCH_SIZE,
        "enable_thinking": False,
    }
    for key, expected in required.items():
        _require_exact_identity_value(
            identity.get(key), expected, f"execution identity {key}"
        )
    census = identity.get("census")
    if not isinstance(census, dict):
        raise core.IntegrityError("execution identity has no quantization census")
    if set(census) != set(EXPECTED_CENSUS):
        raise core.IntegrityError(
            "quantization census fields mismatch: expected "
            f"{sorted(EXPECTED_CENSUS)!r}, got {sorted(census)!r}"
        )
    for key, expected in EXPECTED_CENSUS.items():
        _require_exact_identity_value(
            census.get(key), expected, f"quantization census {key}"
        )
    expected_fraction = (
        EXPECTED_CENSUS["quantized_params"] / EXPECTED_CENSUS["nominal_params"]
    )
    _require_exact_identity_value(
        census["quantized_fraction"],
        expected_fraction,
        "quantization census derived quantized_fraction",
    )
    _require_exact_identity_value(
        census["requested_precision_fraction"],
        expected_fraction,
        "quantization census derived requested_precision_fraction",
    )
    gpu = identity.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("gpu_available") is not True:
        raise core.IntegrityError("Tesla T4 GPU is unavailable")
    _require_exact_identity_value(
        gpu.get("gpu_name"), EXPECTED_GPU_NAME, "GPU class"
    )
    _require_exact_identity_value(
        gpu.get("gpu_compute_capability"),
        EXPECTED_GPU_COMPUTE_CAPABILITY,
        "GPU compute capability",
    )
    driver_version = gpu.get("gpu_driver_version")
    if not isinstance(driver_version, str) or not driver_version.strip():
        raise core.IntegrityError("GPU driver version is missing from replay provenance")
    tokenizer_identity = _validated_tokenizer_identity(
        identity.get("tokenizer_identity")
    )
    library_versions = _validated_versions(identity.get("library_versions"))
    source_rendered_chat_manifest_sha256 = _require_sha256(
        identity.get("source_rendered_chat_manifest_sha256"),
        "source rendered-chat manifest sha256",
    )
    _require_exact_identity_value(
        identity.get("source_rendered_chat_manifest_count"),
        627,
        "source rendered-chat manifest count",
    )
    replay_git_commit = _require_git_commit(
        identity.get("replay_git_commit"), "replay git commit"
    )
    return {
        **required,
        "census": dict(EXPECTED_CENSUS),
        "gpu": dict(gpu),
        "tokenizer_identity": tokenizer_identity,
        "library_versions": library_versions,
        "source_rendered_chat_manifest_sha256": (
            source_rendered_chat_manifest_sha256
        ),
        "source_rendered_chat_manifest_count": 627,
        "replay_git_commit": replay_git_commit,
    }


def _frozen_trace_sha256(source: core.ReplaySource) -> str:
    questions = [
        {
            "question_id": qid,
            "question": source.questions[qid].question,
            "plan": list(source.questions[qid].plan),
            "stop_reason": source.questions[qid].stop_reason,
        }
        for qid in source.question_ids
    ]
    calls = []
    for record in source.baseline_records:
        if record.get("record_type") != "agent_call":
            continue
        prompt_role = record.get("prompt_role")
        if prompt_role not in {"qa", "extractor", "plan_summary"}:
            continue
        item = {
            "question_id": record.get("question_id"),
            "stage": record.get("stage"),
            "call_index": record.get("call_index"),
            "prompt_role": prompt_role,
            "record_sha256": core.canonical_json_sha256(record),
            "consumer_input": record.get("consumer_input"),
        }
        if prompt_role == "extractor":
            item["normalized_consumer_payload"] = record.get("consumer_payload")
        calls.append(item)
    return core.canonical_json_sha256({"questions": questions, "calls": calls})


def build_execution_fingerprint_payload(
    source: core.ReplaySource,
    runtime_identity: dict,
) -> tuple[str, dict]:
    validated_runtime_identity = validate_execution_identity(runtime_identity)
    formatter_source = inspect.getsource(prompts.format_paragraphs)
    prompt_module_path = Path(inspect.getsourcefile(prompts.format_paragraphs) or "")
    if not prompt_module_path.is_file():
        raise core.IntegrityError("cannot resolve passage formatter source file")
    replay_files = {
        "clean_room/passage_replay_core.py": core.sha256_file(
            ROOT / "clean_room" / "passage_replay_core.py"
        ),
        "clean_room/passage_replay.py": core.sha256_file(Path(__file__).resolve()),
    }
    source_prompt_hashes = [
        {
            "question_id": record["question_id"],
            "stage": record["stage"],
            "call_index": record["call_index"],
            "message_object_sha256": record["rendered_prompt_sha256"],
        }
        for record in (
            *core.source_qa_records(source),
            *core.source_summary_records(source),
        )
    ]
    condition_manifests = {
        condition: [
            _batch_payload(batch)
            for batch in core.condition_batches(source, condition)
        ]
        for condition in core.CONDITIONS
    }
    source_model_loads = [
        record
        for record in source.baseline_records
        if record.get("record_type") == "model_load"
    ]
    if not source_model_loads:
        raise core.IntegrityError("source model-load provenance is missing")
    source_gpu_fields = (
        "gpu_name",
        "gpu_compute_capability",
        "gpu_driver_version",
        "gpu_total_memory_mib",
    )
    source_environments = {
        (
            record.get("model_id"),
            record.get("precision"),
            *(record.get(key) for key in source_gpu_fields),
        )
        for record in source_model_loads
    }
    if len(source_environments) != 1:
        raise core.IntegrityError(
            "source model-load rows disagree on model, precision, or GPU provenance"
        )
    source_environment = next(iter(source_environments))
    if source_environment[:2] != (MODEL_ID, PRECISION):
        raise core.IntegrityError("source model-load identity differs from replay contract")
    source_gpu = dict(zip(source_gpu_fields, source_environment[2:]))
    payload = {
        "schema": "frozen-upstream-passage-replay-execution-v1",
        "source": {
            "artifact_sha256": dict(core.SOURCE_SHA256),
            "git_commit": core.SOURCE_COMMIT,
            "experiment_fingerprint": core.SOURCE_EXPERIMENT_FINGERPRINT,
            "full_ids_sha256": core.FULL_IDS_SHA256,
            "both_gold_ids_sha256": core.BOTH_GOLD_IDS_SHA256,
            "library_versions": source.baseline_meta.get("library_versions"),
            "python_version": None,
            "gpu": source_gpu,
        },
        "frozen_trace_sha256": _frozen_trace_sha256(source),
        "treatment_schemas": {
            core.SPANS_PLUS_PASSAGES: {
                "policy": "retain_normalized_spans_then_append_recorded_passages",
                "duplication": "selected spans intentionally repeat when present in passage",
            },
            core.PASSAGES_ONLY: {
                "policy": "replace_normalized_spans_with_recorded_passages",
                "cohort": "baseline_defined_both_gold_128",
            },
            "passage_header": core.PASSAGE_HEADER,
        },
        "passage_formatter": {
            "qualified_name": "src.prompts.format_paragraphs",
            "source_sha256": hashlib.sha256(formatter_source.encode("utf-8")).hexdigest(),
            "module_sha256": core.sha256_file(prompt_module_path),
        },
        "source_message_object_manifest": source_prompt_hashes,
        "source_message_object_manifest_sha256": core.canonical_json_sha256(
            source_prompt_hashes
        ),
        "prompt_contract": {
            "bundle_version": prompts.PROMPT_BUNDLE_VERSION,
            "role_versions": {
                role: prompts.ROLE_PROMPT_VERSIONS[role]
                for role in ("qa", "plan_summary")
            },
            "template_sha256": {
                role: prompts.prompt_template_sha256(role)
                for role in ("qa", "plan_summary")
            },
        },
        "state_policies": {
            "history_policy": "frozen_topology_replace_qa_derived_fields_v1",
            "grounding_policy": "contiguous_nfkc_casefold_token_phrase_v1",
            "aggregate_policy": "treated_grounded_prior_answers_only_v1",
            "stop_policy": "source_stop_reason_frozen_v1",
            "finalizer_policy": "summary_parsed_salvaged_reverse_qa_v1",
        },
        "runtime_identity": validated_runtime_identity,
        "source_rendered_chat_manifest": {
            "sha256": validated_runtime_identity[
                "source_rendered_chat_manifest_sha256"
            ],
            "count": validated_runtime_identity[
                "source_rendered_chat_manifest_count"
            ],
        },
        "replay_git_commit": validated_runtime_identity["replay_git_commit"],
        "decoding": {
            "method": "greedy",
            "sampling": False,
            "constraints": False,
            "enable_thinking": False,
            "batch_size": BATCH_SIZE,
            "output_ceilings": dict(OUTPUT_CEILINGS),
        },
        "sentinel_batch_manifest": [
            _batch_payload(batch) for batch in sentinel_batches(source)
        ],
        "condition_batch_manifests": condition_manifests,
        "condition_batch_manifests_sha256": core.canonical_json_sha256(
            condition_manifests
        ),
        "replay_code_sha256": replay_files,
        "scorer": {
            "answer_normalization": "HotpotQA",
            "f1": "token_overlap",
            "em": "normalized_exact_match",
            "bootstrap": {"draws": 10_000, "seed": 20260807},
            "mcnemar": "exact_two_sided",
            "practical_equivalence_f1_points": 2.0,
        },
    }
    return core.canonical_json_sha256(payload), payload


JSONL_FLUSH_EVERY = 50
PUBLISHED_DATA_ARTIFACTS = (
    "manifest.json",
    "calls.jsonl",
    "answers.jsonl",
    "summary.json",
)
REQUIRED_GENERATED_OUTPUT_FIELDS = (
    "raw_output",
    "parse_status",
    "parsed",
    "salvaged",
    "consumer_payload",
    "prompt_tokens",
    "output_ceiling_tokens",
    "prompt_plus_ceiling_tokens",
    "output_tokens",
    "generated_sequence_tokens",
    "context_window_tokens",
    "forced_full_generation",
    "strict_format_ok",
    "protocol_ok",
)
REGENERATED_OUTPUT_FIELDS = (
    *REQUIRED_GENERATED_OUTPUT_FIELDS,
    "consumer_payload_source",
    "extractor_normalization",
    "verbatim_copy_rate",
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
)
VOLATILE_REGENERATED_CALL_FIELDS = frozenset(
    {
        "execution_session_id",
        "latency_s",
        "timestamp",
    }
)
REPLAY_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "precision",
)
EXPECTED_COMPLETE_COUNTS = {
    "model_load": 1,
    "sentinel_calls": 21,
    "scored_calls": 1_038,
    "agent_calls": 1_059,
    "sentinel_certificates": 6,
    "scored_certificates": 262,
    "batch_certificates": 268,
    "answers": 328,
}
EXPECTED_STREAM_ALLOCATIONS = {
    (SENTINEL_PHASE, SENTINEL_CONDITION): {"calls": 21, "certificates": 6},
    ("scored", core.SPANS_PLUS_PASSAGES): {"calls": 627, "certificates": 158},
    ("scored", core.PASSAGES_ONLY): {"calls": 411, "certificates": 104},
}
REQUIRED_ANSWER_FIELDS = (
    "record_type",
    "condition",
    "condition_fingerprint_sha256",
    "question_id",
    "question",
    "gold_answer",
    "retrieval_stratum",
    "source_retrieval_all_gold",
    "predicted_answer",
    "final_answer_source",
    "final_answer_grounded",
    "final_answer_qa_step",
    "f1",
    "em",
    "qa_reverse_usable",
    "qa_last_executed",
    "qa_best_intermediate_oracle",
    "executed_steps",
    "stop_reason",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _default_replay_identity() -> dict[str, str]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": PRECISION,
    }


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise core.IntegrityError(f"{label} is missing")
    return value


def _validate_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise core.IntegrityError(f"{label} is not a non-negative integer: {value!r}")
    return value


def _json_key(key: tuple[str, str, str, int]) -> list[Any]:
    return [key[0], key[1], key[2], key[3]]


@dataclass(frozen=True)
class MemberSpec:
    """Immutable lineage and prompt identity for one generated replay call."""

    condition: str
    question_id: str
    stage: str
    call_index: int
    batch_member_index: int
    parent_key: tuple[str, str, int]
    parent_record_sha256: str
    source_message_sha256: str
    treatment_message_sha256: str
    rendered_chat_sha256: str

    def __post_init__(self) -> None:
        parent_key = tuple(self.parent_key)
        if len(parent_key) != 3:
            raise ValueError(f"invalid parent key: {self.parent_key!r}")
        object.__setattr__(self, "parent_key", parent_key)
        for label, value in (
            ("condition", self.condition),
            ("question_id", self.question_id),
            ("stage", self.stage),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid {label}: {value!r}")
        _validate_nonnegative_int(self.call_index, "member call_index")
        _validate_nonnegative_int(
            self.batch_member_index, "member batch_member_index"
        )
        _validate_nonnegative_int(parent_key[2], "member parent call_index")
        if not all(isinstance(value, str) and value for value in parent_key[:2]):
            raise ValueError(f"invalid parent key: {parent_key!r}")
        for label, value in (
            ("parent_record_sha256", self.parent_record_sha256),
            ("source_message_sha256", self.source_message_sha256),
            ("treatment_message_sha256", self.treatment_message_sha256),
            ("rendered_chat_sha256", self.rendered_chat_sha256),
        ):
            _validate_hash(value, f"member {label}")

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.condition, self.question_id, self.stage, self.call_index


def _member_payload(member: MemberSpec) -> dict:
    return {
        "condition": member.condition,
        "question_id": member.question_id,
        "stage": member.stage,
        "call_index": member.call_index,
        "batch_member_index": member.batch_member_index,
        "parent_key": list(member.parent_key),
        "parent_record_sha256": member.parent_record_sha256,
        "source_message_sha256": member.source_message_sha256,
        "treatment_message_sha256": member.treatment_message_sha256,
        "message_object_sha256": member.treatment_message_sha256,
        "rendered_prompt_sha256": member.treatment_message_sha256,
        "rendered_chat_sha256": member.rendered_chat_sha256,
    }


@dataclass(frozen=True)
class BatchSpec:
    """Canonical, condition-aware unit of fixed-batch replay execution."""

    phase: str
    condition: str
    condition_fingerprint_sha256: str
    stage: str
    batch_id: str
    batch_ordinal: int
    members: tuple[MemberSpec, ...]
    canonical_treatment_batch_sha256: str = ""

    def __post_init__(self) -> None:
        members = tuple(self.members)
        object.__setattr__(self, "members", members)
        if self.phase not in {SENTINEL_PHASE, "scored"}:
            raise ValueError(f"invalid replay phase {self.phase!r}")
        for label, value in (
            ("condition", self.condition),
            ("condition_fingerprint_sha256", self.condition_fingerprint_sha256),
            ("stage", self.stage),
            ("batch_id", self.batch_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid batch {label}: {value!r}")
        _validate_nonnegative_int(self.batch_ordinal, "batch ordinal")
        if not members or len(members) > BATCH_SIZE:
            raise ValueError(
                f"batch {self.batch_id!r} has {len(members)} members; "
                f"expected 1..{BATCH_SIZE}"
            )
        for index, member in enumerate(members):
            if member.condition != self.condition or member.stage != self.stage:
                raise ValueError(
                    f"batch {self.batch_id!r} member treatment identity changed"
                )
            if member.batch_member_index != index:
                raise ValueError(
                    f"batch {self.batch_id!r} member indices are not contiguous"
                )
        expected = canonical_treatment_batch_sha256(self)
        if self.canonical_treatment_batch_sha256:
            if self.canonical_treatment_batch_sha256 != expected:
                raise core.IntegrityError(
                    f"batch {self.batch_id!r} canonical treatment hash mismatch"
                )
        else:
            object.__setattr__(self, "canonical_treatment_batch_sha256", expected)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.phase, self.condition, self.batch_id


@dataclass(frozen=True)
class ResumeAudit:
    """Only certificate-backed calls are eligible to reconstruct treated state."""

    model_load: dict | None
    certified_calls: dict[tuple[str, str, str, int], dict]
    certified_batches: frozenset[tuple[str, str, str]]
    orphan_calls_by_batch: dict[tuple[str, str, str], tuple[dict, ...]]
    next_batch_index: int
    expected_batch_count: int

    @property
    def complete(self) -> bool:
        return (
            self.model_load is not None
            and not self.orphan_calls_by_batch
            and self.next_batch_index == self.expected_batch_count
        )


def canonical_treatment_batch_payload(batch: BatchSpec) -> dict:
    return {
        "schema": "passage-replay-certified-batch-v1",
        "phase": batch.phase,
        "condition": batch.condition,
        "condition_fingerprint_sha256": batch.condition_fingerprint_sha256,
        "stage": batch.stage,
        "batch_id": batch.batch_id,
        "batch_ordinal": batch.batch_ordinal,
        "members": [_member_payload(member) for member in batch.members],
    }


def canonical_treatment_batch_sha256(batch: BatchSpec) -> str:
    return core.canonical_json_sha256(canonical_treatment_batch_payload(batch))


def make_batch_spec(
    *,
    phase: str,
    condition: str,
    condition_fingerprint_sha256: str,
    stage: str,
    batch_ordinal: int,
    members: Sequence[MemberSpec],
    batch_id: str | None = None,
) -> BatchSpec:
    normalized = tuple(
        member
        if member.batch_member_index == index
        else replace(member, batch_member_index=index)
        for index, member in enumerate(members)
    )
    return BatchSpec(
        phase=phase,
        condition=condition,
        condition_fingerprint_sha256=condition_fingerprint_sha256,
        stage=stage,
        batch_id=batch_id or f"{condition}:{stage}:{batch_ordinal:06d}",
        batch_ordinal=batch_ordinal,
        members=normalized,
    )


def _lookup_member_spec(
    members: Mapping[Any, MemberSpec],
    condition: str,
    member: core.CallKey,
) -> MemberSpec:
    candidates: tuple[Any, ...] = (
        (condition, member.question_id, member.stage, member.call_index),
        core.ConditionCallKey(
            condition,
            member.question_id,
            member.stage,
            member.call_index,
        ),
        (member.question_id, member.stage, member.call_index),
        member,
    )
    for key in candidates:
        if key in members:
            value = members[key]
            if not isinstance(value, MemberSpec):
                raise TypeError(f"member map value for {key!r} is not a MemberSpec")
            return value
    raise core.IntegrityError(
        "missing member spec for "
        f"{(condition, member.question_id, member.stage, member.call_index)!r}"
    )


def condition_batch_specs(
    *,
    phase: str,
    condition: str,
    condition_fingerprint_sha256: str,
    frozen_batches: Sequence[core.FrozenBatch],
    members: Mapping[Any, MemberSpec],
) -> tuple[BatchSpec, ...]:
    """Project frozen source batches into condition-aware certified batches."""
    specs: list[BatchSpec] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    for batch in frozen_batches:
        projected: list[MemberSpec] = []
        for index, source_member in enumerate(batch.members):
            member = _lookup_member_spec(members, condition, source_member)
            member = replace(
                member,
                condition=condition,
                stage=batch.stage,
                batch_member_index=index,
            )
            if member.key in seen_keys:
                raise core.IntegrityError(f"duplicate condition call key {member.key!r}")
            seen_keys.add(member.key)
            projected.append(member)
        specs.append(
            make_batch_spec(
                phase=phase,
                condition=condition,
                condition_fingerprint_sha256=condition_fingerprint_sha256,
                stage=batch.stage,
                batch_ordinal=batch.ordinal,
                members=projected,
            )
        )
    return tuple(specs)


def _leaf_present(path: Path) -> bool:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


def _require_regular_leaf(
    path: Path,
    label: str,
    *,
    allow_absent: bool = False,
) -> bool:
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError:
        if allow_absent:
            return False
        raise core.IntegrityError(f"{label} is missing: {path}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise core.IntegrityError(f"{label} is not a regular file: {path}")
    return True


def _open_regular_fd(path: Path, flags: int, mode: int = 0o600) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(Path(path), flags, mode)
    except OSError as exc:
        raise core.IntegrityError(f"cannot safely open regular file {path}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise core.IntegrityError(f"opened artifact is not a regular file: {path}")
    return descriptor


class JsonlStore:
    """Replay-local append-only JSONL store with durable batch checkpoints."""

    def __init__(self, path: Path, *, lock_path: Path | None = None):
        self.path = Path(path)
        self.lock_path = (
            Path(lock_path)
            if lock_path is not None
            else self.path.with_suffix(self.path.suffix + ".lock")
        )
        self._fh = None
        self._lock_fd: int | None = None
        self._since_flush = 0

    def acquire_lock(self, execution_session_id: str | None = None) -> JsonlStore:
        if self._fh is not None or self._lock_fd is not None:
            raise RuntimeError(f"JSONL store is already open: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "execution_session_id": execution_session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        import fcntl

        fd: int | None = None
        try:
            _require_regular_leaf(
                self.lock_path,
                "replay output lock",
                allow_absent=True,
            )
            fd = _open_regular_fd(
                self.lock_path,
                os.O_CREAT | os.O_RDWR,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.lseek(fd, 0, os.SEEK_SET)
                owner = os.read(fd, 16 * 1024).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"exclusive output lock is held: {self.lock_path}; "
                    f"owner={owner.strip()!r}"
                ) from exc
            self._lock_fd = fd
            fd = None
            encoded_payload = (
                json.dumps(payload, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            os.ftruncate(self._lock_fd, 0)
            os.lseek(self._lock_fd, 0, os.SEEK_SET)
            remaining = memoryview(encoded_payload)
            while remaining:
                written = os.write(self._lock_fd, remaining)
                if written <= 0:
                    raise OSError("short write while recording JSONL lock owner")
                remaining = remaining[written:]
            os.fsync(self._lock_fd)
            return self
        except BaseException:
            if fd is not None:
                os.close(fd)
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None
            self._release_lock()
            raise

    def open_data(self, execution_session_id: str | None = None) -> JsonlStore:
        if self._lock_fd is None:
            raise RuntimeError(f"JSONL store has no output lock: {self.path}")
        if self._fh is not None:
            raise RuntimeError(f"JSONL data file is already open: {self.path}")
        descriptor: int | None = None
        try:
            _require_regular_leaf(
                self.path,
                "replay calls artifact",
                allow_absent=True,
            )
            descriptor = _open_regular_fd(
                self.path,
                os.O_CREAT | os.O_RDWR | os.O_APPEND,
            )
            repair = self._repair_torn_tail(descriptor, execution_session_id)
            self._fh = os.fdopen(descriptor, "a", encoding="utf-8")
            descriptor = None
            if repair is not None:
                self.write([repair])
                self.durable_flush()
            return self
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise

    def open(self, execution_session_id: str | None = None) -> JsonlStore:
        try:
            return self.acquire_lock(execution_session_id).open_data(
                execution_session_id
            )
        except BaseException:
            self.close()
            raise

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        import fcntl

        descriptor = self._lock_fd
        self._lock_fd = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _repair_torn_tail(
        self,
        descriptor: int,
        execution_session_id: str | None,
    ) -> dict | None:
        size = os.fstat(descriptor).st_size
        if size == 0:
            return None
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != size:
            raise core.IntegrityError(f"short read while repairing {self.path}")
        if raw.endswith(b"\n"):
            return None
        cut = raw.rfind(b"\n") + 1
        removed = len(raw) - cut
        os.ftruncate(descriptor, cut)
        os.fsync(descriptor)
        return {
            "record_type": "store_repair",
            "repair": "truncated_non_newline_tail",
            "removed_bytes": removed,
            "execution_session_id": execution_session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def read_existing(self) -> list[dict]:
        if not _require_regular_leaf(
            self.path,
            "replay calls artifact",
            allow_absent=True,
        ):
            return []
        records: list[dict] = []
        descriptor = _open_regular_fd(self.path, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise core.IntegrityError(
                        f"JSONL corruption at complete line {line_number} of "
                        f"{self.path}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise core.IntegrityError(
                        f"JSONL line {line_number} of {self.path} is not an object"
                    )
                records.append(record)
        return records

    def write(self, records: Sequence[dict]) -> None:
        if self._fh is None:
            raise RuntimeError(f"JSONL store is not open: {self.path}")
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("JSONL records must be objects")
            self._fh.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            self._since_flush += 1
            if self._since_flush >= JSONL_FLUSH_EVERY:
                self.flush()

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._since_flush = 0

    def durable_flush(self) -> None:
        if self._fh is not None:
            self.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            if self._fh is not None:
                try:
                    self.flush()
                finally:
                    self._fh.close()
                    self._fh = None
        finally:
            self._release_lock()

    def __enter__(self) -> JsonlStore:
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _expected_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    expected = dict(identity or _default_replay_identity())
    missing = [field for field in REPLAY_IDENTITY_FIELDS if field not in expected]
    if missing:
        raise ValueError(f"expected replay identity is missing {missing!r}")
    return {field: expected[field] for field in REPLAY_IDENTITY_FIELDS}


def _validate_record_identity(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for field in REPLAY_IDENTITY_FIELDS:
        if record.get(field) != expected[field]:
            raise core.IntegrityError(
                f"{label} {field} mismatch: "
                f"{record.get(field)!r} != {expected[field]!r}"
            )


def _call_key(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    condition = record.get("condition")
    question_id = record.get("question_id")
    stage = record.get("stage")
    call_index = record.get("call_index")
    if not all(isinstance(value, str) and value for value in (condition, question_id, stage)):
        raise core.IntegrityError("generated call has an invalid condition-aware key")
    _validate_nonnegative_int(call_index, "generated call_index")
    return condition, question_id, stage, call_index


def _batch_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    phase = record.get("phase")
    condition = record.get("condition")
    batch_id = record.get("batch_id")
    if not all(isinstance(value, str) and value for value in (phase, condition, batch_id)):
        raise core.IntegrityError("batch certificate has an invalid key")
    return phase, condition, batch_id


def _reject_generated_solo(record: Mapping[str, Any]) -> None:
    for field in ("stage", "prompt_role", "conceptual_role", "condition"):
        value = record.get(field)
        if isinstance(value, str) and value.casefold() == "solo":
            raise core.IntegrityError(f"generated solo call is forbidden ({field})")


def _validate_generated_call(
    record: dict,
    member: MemberSpec,
    batch: BatchSpec,
    identity: Mapping[str, Any],
) -> None:
    if record.get("record_type") != "agent_call":
        raise core.IntegrityError(f"{member.key!r} is not an agent_call")
    expected_values = {
        "phase": batch.phase,
        "condition": batch.condition,
        "condition_fingerprint_sha256": batch.condition_fingerprint_sha256,
        "canonical_treatment_batch_sha256": (
            batch.canonical_treatment_batch_sha256
        ),
        "stage": member.stage,
        "question_id": member.question_id,
        "call_index": member.call_index,
        "batch_id": batch.batch_id,
        "batch_ordinal": batch.batch_ordinal,
        "batch_member_index": member.batch_member_index,
        "parent_key": list(member.parent_key),
        "parent_record_sha256": member.parent_record_sha256,
        "source_message_sha256": member.source_message_sha256,
        "treatment_message_sha256": member.treatment_message_sha256,
        "rendered_chat_sha256": member.rendered_chat_sha256,
    }
    for field, expected in expected_values.items():
        actual = record.get(field)
        if field == "parent_key" and isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise core.IntegrityError(
                f"call {member.key!r} {field} mismatch: {actual!r} != {expected!r}"
            )
    _validate_record_identity(record, identity, f"call {member.key!r}")
    for field in REQUIRED_GENERATED_OUTPUT_FIELDS:
        if field not in record:
            raise core.IntegrityError(f"call {member.key!r} is missing {field}")
    if not isinstance(record["raw_output"], str):
        raise core.IntegrityError(f"call {member.key!r} raw_output is not text")
    if not isinstance(record["parse_status"], str):
        raise core.IntegrityError(f"call {member.key!r} parse_status is not text")
    for field in (
        "prompt_tokens",
        "output_ceiling_tokens",
        "prompt_plus_ceiling_tokens",
        "output_tokens",
        "generated_sequence_tokens",
        "context_window_tokens",
    ):
        _validate_nonnegative_int(record[field], f"call {member.key!r} {field}")
    prompt_role = "plan_summary" if member.stage == "plan_summary" else "qa"
    if member.stage not in (*core.QA_STAGES, "plan_summary"):
        raise core.IntegrityError(
            f"call {member.key!r} has unsupported generated stage {member.stage!r}"
        )
    expected_ceiling = OUTPUT_CEILINGS[prompt_role]
    if record["output_ceiling_tokens"] != expected_ceiling:
        raise core.IntegrityError(
            f"call {member.key!r} output ceiling mismatch: "
            f"{record['output_ceiling_tokens']!r} != {expected_ceiling}"
        )
    expected_total = record["prompt_tokens"] + expected_ceiling
    if record["prompt_plus_ceiling_tokens"] != expected_total:
        raise core.IntegrityError(
            f"call {member.key!r} prompt-plus-ceiling total mismatch"
        )
    if expected_total > record["context_window_tokens"]:
        raise core.IntegrityError(
            f"call {member.key!r} prompt plus ceiling exceeds context window"
        )
    if record["forced_full_generation"] is not False:
        raise core.IntegrityError(f"call {member.key!r} forced full generation")
    for field in ("strict_format_ok", "protocol_ok"):
        if not isinstance(record[field], bool):
            raise core.IntegrityError(f"call {member.key!r} {field} is not boolean")
    reject_thinking([record])
    _reject_generated_solo(record)


def augment_generated_call(
    generated: Mapping[str, Any],
    member: MemberSpec,
    batch: BatchSpec,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict:
    """Attach immutable replay lineage before a generated row can be persisted."""
    record = dict(generated)
    for field, expected in (
        ("question_id", member.question_id),
        ("stage", member.stage),
        ("call_index", member.call_index),
    ):
        if field in record and record[field] != expected:
            raise core.IntegrityError(
                f"generated call {field} changed: {record[field]!r} != {expected!r}"
            )
    expected_identity = _expected_identity(identity)
    for field, value in expected_identity.items():
        if field in record and record[field] != value:
            raise core.IntegrityError(
                f"generated call {field} changed: {record[field]!r} != {value!r}"
            )
        record[field] = value
    record.update(
        {
            "record_type": "agent_call",
            "phase": batch.phase,
            "condition": batch.condition,
            "condition_fingerprint_sha256": batch.condition_fingerprint_sha256,
            "canonical_treatment_batch_sha256": (
                batch.canonical_treatment_batch_sha256
            ),
            "question_id": member.question_id,
            "stage": member.stage,
            "call_index": member.call_index,
            "batch_id": batch.batch_id,
            "batch_ordinal": batch.batch_ordinal,
            "batch_member_index": member.batch_member_index,
            "parent_key": list(member.parent_key),
            "parent_record_sha256": member.parent_record_sha256,
            "source_message_sha256": member.source_message_sha256,
            "treatment_message_sha256": member.treatment_message_sha256,
            "rendered_chat_sha256": member.rendered_chat_sha256,
        }
    )
    _validate_generated_call(record, member, batch, expected_identity)
    return record


augment_replay_record = augment_generated_call


def compare_regenerated_output(existing: Mapping[str, Any], regenerated: Mapping[str, Any]) -> None:
    """Fail if any non-volatile call field differs on orphan regeneration."""
    fields = sorted(
        (set(existing) | set(regenerated)) - VOLATILE_REGENERATED_CALL_FIELDS
    )
    for field in fields:
        if field not in existing or field not in regenerated:
            raise core.IntegrityError(
                "orphan regeneration changed output field or deterministic "
                f"metadata presence: {field}"
            )
        if _canonical_json_bytes(existing[field]) != _canonical_json_bytes(
            regenerated[field]
        ):
            raise core.IntegrityError(
                "orphan regeneration changed output field or deterministic "
                f"metadata {field!r} for "
                f"{_call_key(existing)!r}"
            )


def build_batch_certificate(
    batch: BatchSpec,
    member_records: Sequence[dict],
    *,
    written_members: Sequence[tuple[str, str, str, int]],
    identity: Mapping[str, Any] | None = None,
) -> dict:
    expected_identity = _expected_identity(identity)
    records = tuple(member_records)
    if len(records) != len(batch.members):
        raise core.IntegrityError(
            f"batch {batch.batch_id!r} has {len(records)} certified records, "
            f"expected {len(batch.members)}"
        )
    for member, record in zip(batch.members, records):
        _validate_generated_call(record, member, batch, expected_identity)
    expected_keys = [member.key for member in batch.members]
    written = tuple(written_members)
    if len(set(written)) != len(written) or any(key not in expected_keys for key in written):
        raise core.IntegrityError(
            f"batch {batch.batch_id!r} has invalid written-member keys"
        )
    written_set = set(written)
    written_in_batch_order = [key for key in expected_keys if key in written_set]
    if list(written) != written_in_batch_order:
        raise core.IntegrityError(
            f"batch {batch.batch_id!r} written-member order changed"
        )
    certificate = {
        "record_type": "batch",
        "phase": batch.phase,
        "condition": batch.condition,
        "condition_fingerprint_sha256": batch.condition_fingerprint_sha256,
        "canonical_treatment_batch_sha256": (
            batch.canonical_treatment_batch_sha256
        ),
        "stage": batch.stage,
        "batch_id": batch.batch_id,
        "batch_ordinal": batch.batch_ordinal,
        "members": [_member_payload(member) for member in batch.members],
        "member_record_sha256": [
            {"key": _json_key(member.key), "sha256": core.canonical_json_sha256(record)}
            for member, record in zip(batch.members, records)
        ],
        "written_members": [_json_key(key) for key in written],
        "resume_regenerated_members": len(records) - len(written),
        "batch_size_actual": len(records),
        "batch_size_requested": BATCH_SIZE,
    }
    certificate.update(expected_identity)
    return certificate


def reconcile_orphan_batch(
    batch: BatchSpec,
    orphan_records: Sequence[dict],
    regenerated_records: Sequence[dict],
    *,
    identity: Mapping[str, Any] | None = None,
) -> tuple[tuple[dict, ...], dict]:
    """Validate a full-batch replay and return only missing rows plus its cert."""
    expected_identity = _expected_identity(identity)
    existing = tuple(orphan_records)
    regenerated = tuple(regenerated_records)
    if len(regenerated) != len(batch.members):
        raise core.IntegrityError(
            f"resume must regenerate full batch {batch.batch_id!r}: "
            f"got {len(regenerated)}/{len(batch.members)} members"
        )
    if len(existing) > len(batch.members):
        raise core.IntegrityError(f"too many orphan rows for {batch.batch_id!r}")
    for index, (member, record) in enumerate(zip(batch.members, regenerated)):
        _validate_generated_call(record, member, batch, expected_identity)
        if index < len(existing):
            _validate_generated_call(existing[index], member, batch, expected_identity)
            compare_regenerated_output(existing[index], record)
    resolved = (*existing, *regenerated[len(existing) :])
    missing = tuple(regenerated[len(existing) :])
    certificate = build_batch_certificate(
        batch,
        resolved,
        written_members=[record_key for record_key in map(_call_key, missing)],
        identity=expected_identity,
    )
    return missing, certificate


def persist_certified_batch(
    store: JsonlStore,
    batch: BatchSpec,
    orphan_records: Sequence[dict],
    regenerated_records: Sequence[dict],
    *,
    identity: Mapping[str, Any] | None = None,
) -> tuple[tuple[dict, ...], dict]:
    """Write calls first and a trailing certificate, then make both durable."""
    missing, certificate = reconcile_orphan_batch(
        batch,
        orphan_records,
        regenerated_records,
        identity=identity,
    )
    store.write([*missing, certificate])
    store.durable_flush()
    return missing, certificate


def _validate_certificate(
    certificate: dict,
    batch: BatchSpec,
    call_records: Mapping[tuple[str, str, str, int], dict],
    identity: Mapping[str, Any],
) -> None:
    expected = {
        "record_type": "batch",
        "phase": batch.phase,
        "condition": batch.condition,
        "condition_fingerprint_sha256": batch.condition_fingerprint_sha256,
        "canonical_treatment_batch_sha256": (
            batch.canonical_treatment_batch_sha256
        ),
        "stage": batch.stage,
        "batch_id": batch.batch_id,
        "batch_ordinal": batch.batch_ordinal,
        "members": [_member_payload(member) for member in batch.members],
        "batch_size_actual": len(batch.members),
        "batch_size_requested": BATCH_SIZE,
    }
    for field, value in expected.items():
        if certificate.get(field) != value:
            raise core.IntegrityError(
                f"certificate {batch.key!r} {field} mismatch"
            )
    _validate_record_identity(certificate, identity, f"certificate {batch.key!r}")
    expected_hashes = [
        {
            "key": _json_key(member.key),
            "sha256": core.canonical_json_sha256(call_records[member.key]),
        }
        for member in batch.members
    ]
    if certificate.get("member_record_sha256") != expected_hashes:
        raise core.IntegrityError(
            f"certificate {batch.key!r} member record hashes mismatch"
        )
    written = certificate.get("written_members")
    if not isinstance(written, list):
        raise core.IntegrityError(f"certificate {batch.key!r} has no written members")
    expected_keys = [_json_key(member.key) for member in batch.members]
    if len({_canonical_json_bytes(key) for key in written}) != len(written):
        raise core.IntegrityError(
            f"certificate {batch.key!r} repeats written members"
        )
    if any(key not in expected_keys for key in written):
        raise core.IntegrityError(
            f"certificate {batch.key!r} has unexpected written members"
        )
    if written != [key for key in expected_keys if key in written]:
        raise core.IntegrityError(
            f"certificate {batch.key!r} written-member order mismatch"
        )
    if certificate.get("resume_regenerated_members") != len(batch.members) - len(
        written
    ):
        raise core.IntegrityError(
            f"certificate {batch.key!r} resume count mismatch"
        )


def audit_resume(
    records: Sequence[dict],
    batch_specs: Sequence[BatchSpec],
    *,
    identity: Mapping[str, Any] | None = None,
) -> ResumeAudit:
    """Audit a partial stream; only valid trailing certificates admit state."""
    expected_identity = _expected_identity(identity)
    specs = tuple(batch_specs)
    spec_by_key: dict[tuple[str, str, str], BatchSpec] = {}
    member_to_batch: dict[tuple[str, str, str, int], BatchSpec] = {}
    for spec in specs:
        if spec.key in spec_by_key:
            raise core.IntegrityError(f"duplicate expected batch key {spec.key!r}")
        spec_by_key[spec.key] = spec
        if canonical_treatment_batch_sha256(spec) != (
            spec.canonical_treatment_batch_sha256
        ):
            raise core.IntegrityError(f"expected batch {spec.key!r} hash changed")
        for member in spec.members:
            if member.key in member_to_batch:
                raise core.IntegrityError(
                    f"duplicate expected condition call key {member.key!r}"
                )
            member_to_batch[member.key] = spec

    model_load: dict | None = None
    all_calls: dict[tuple[str, str, str, int], dict] = {}
    call_order: list[tuple[str, str, str, int]] = []
    certificate_order: list[tuple[str, str, str]] = []
    certified_batches: set[tuple[str, str, str]] = set()
    certified_calls: dict[tuple[str, str, str, int], dict] = {}
    open_segment: list[tuple[str, str, str, int]] = []
    saw_generated_record = False

    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise core.IntegrityError(f"resume row {position} is not an object")
        record_type = record.get("record_type")
        if record_type == "store_repair":
            continue
        if record_type == "model_load":
            if model_load is not None or saw_generated_record:
                raise core.IntegrityError("model_load must occur exactly once before calls")
            _validate_record_identity(record, expected_identity, "model_load")
            model_load = record
            continue
        saw_generated_record = True
        if model_load is None:
            raise core.IntegrityError("generated rows precede successful model_load")
        if record_type == "agent_call":
            key = _call_key(record)
            if key in all_calls:
                raise core.IntegrityError(f"duplicate generated call key {key!r}")
            batch = member_to_batch.get(key)
            if batch is None:
                raise core.IntegrityError(f"unexpected generated call key {key!r}")
            member = batch.members[record.get("batch_member_index", -1)] if (
                isinstance(record.get("batch_member_index"), int)
                and not isinstance(record.get("batch_member_index"), bool)
                and 0 <= record["batch_member_index"] < len(batch.members)
            ) else None
            if member is None or member.key != key:
                raise core.IntegrityError(f"call {key!r} has invalid member index")
            _validate_generated_call(record, member, batch, expected_identity)
            all_calls[key] = record
            call_order.append(key)
            open_segment.append(key)
            continue
        if record_type != "batch":
            raise core.IntegrityError(
                f"unexpected resume record_type at row {position}: {record_type!r}"
            )
        batch_key = _batch_key(record)
        batch = spec_by_key.get(batch_key)
        if batch is None:
            raise core.IntegrityError(f"unexpected batch certificate {batch_key!r}")
        if batch_key in certified_batches:
            raise core.IntegrityError(f"duplicate batch certificate {batch_key!r}")
        expected_segment = [member.key for member in batch.members]
        if open_segment != expected_segment:
            raise core.IntegrityError(
                f"certificate {batch_key!r} does not trail its exact ordered members"
            )
        _validate_certificate(record, batch, all_calls, expected_identity)
        certified_batches.add(batch_key)
        certificate_order.append(batch_key)
        for key in expected_segment:
            certified_calls[key] = all_calls[key]
        open_segment = []

    expected_batch_order = [spec.key for spec in specs]
    if certificate_order != expected_batch_order[: len(certificate_order)]:
        raise core.IntegrityError("certified batches are not a canonical prefix")
    next_index = len(certificate_order)
    orphan_calls_by_batch: dict[tuple[str, str, str], tuple[dict, ...]] = {}
    if open_segment:
        if next_index >= len(specs):
            raise core.IntegrityError("orphan calls follow a complete batch plan")
        orphan_batch = specs[next_index]
        expected_prefix = [member.key for member in orphan_batch.members][
            : len(open_segment)
        ]
        if open_segment != expected_prefix:
            raise core.IntegrityError(
                f"orphan rows are not a canonical prefix of {orphan_batch.key!r}"
            )
        orphan_calls_by_batch[orphan_batch.key] = tuple(
            all_calls[key] for key in open_segment
        )
    if len(call_order) != len(certified_calls) + len(open_segment):
        raise core.IntegrityError("resume stream contains unaccounted generated calls")
    return ResumeAudit(
        model_load=model_load,
        certified_calls=certified_calls,
        certified_batches=frozenset(certified_batches),
        orphan_calls_by_batch=orphan_calls_by_batch,
        next_batch_index=next_index,
        expected_batch_count=len(specs),
    )


@dataclass(frozen=True)
class PublicationValidationPayload:
    """Independent authorities required to certify publication data."""

    source: core.ReplaySource
    execution_fingerprint_sha256: str
    replay_fingerprint_sha256: str
    condition_fingerprints: Mapping[str, str]
    batch_specs: tuple[BatchSpec, ...]
    identity: Mapping[str, Any]
    expected_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_sha256(
            self.execution_fingerprint_sha256,
            "publication execution fingerprint",
        )
        _require_sha256(
            self.replay_fingerprint_sha256,
            "publication replay fingerprint",
        )
        fingerprints = dict(self.condition_fingerprints)
        if set(fingerprints) != set(core.CONDITIONS):
            raise ValueError("publication condition fingerprints are incomplete")
        for condition, fingerprint in fingerprints.items():
            _require_sha256(fingerprint, f"{condition} condition fingerprint")
        if not isinstance(self.source, core.ReplaySource):
            raise TypeError("publication source authority is not a ReplaySource")
        if not isinstance(self.expected_manifest, Mapping):
            raise TypeError("publication expected manifest is not a mapping")
        manifest = dict(self.expected_manifest)
        if core.canonical_json_sha256(manifest) != self.replay_fingerprint_sha256:
            raise core.IntegrityError(
                "expected manifest does not produce the replay fingerprint"
            )
        object.__setattr__(self, "condition_fingerprints", fingerprints)
        object.__setattr__(self, "batch_specs", tuple(self.batch_specs))
        object.__setattr__(self, "identity", dict(self.identity))
        object.__setattr__(self, "expected_manifest", manifest)

    @property
    def contract_sha256(self) -> str:
        return core.canonical_json_sha256(
            {
                "schema": "passage-replay-publication-validation-v1",
                "execution_fingerprint_sha256": self.execution_fingerprint_sha256,
                "replay_fingerprint_sha256": self.replay_fingerprint_sha256,
                "condition_fingerprints": dict(self.condition_fingerprints),
                "source_experiment_fingerprint": core.SOURCE_EXPERIMENT_FINGERPRINT,
                "source_question_ids_sha256": self.source.question_ids_sha256,
                "source_both_gold_ids_sha256": core.ordered_ids_sha256(
                    self.source.both_gold_ids
                ),
                "identity": _expected_identity(self.identity),
                "batch_plan": [
                    {
                        "key": list(spec.key),
                        "sha256": spec.canonical_treatment_batch_sha256,
                    }
                    for spec in self.batch_specs
                ],
                "manifest_sha256": core.canonical_json_sha256(
                    dict(self.expected_manifest)
                ),
            }
        )


def _validate_model_load_provenance(
    record: Mapping[str, Any],
    *,
    execution_fingerprint_sha256: str | None,
    condition_fingerprints: Mapping[str, str] | None,
    source: core.ReplaySource | None,
) -> None:
    if record.get("status") != "success":
        raise core.IntegrityError("model_load does not record successful completion")
    execution = _require_sha256(
        record.get("execution_fingerprint_sha256"),
        "model_load execution fingerprint",
    )
    if execution_fingerprint_sha256 is not None and execution != execution_fingerprint_sha256:
        raise core.IntegrityError("model_load execution fingerprint mismatch")
    if record.get("source_experiment_fingerprint") != core.SOURCE_EXPERIMENT_FINGERPRINT:
        raise core.IntegrityError("model_load source experiment fingerprint mismatch")
    if record.get("source_artifact_sha256") != dict(core.SOURCE_SHA256):
        raise core.IntegrityError("model_load source artifact provenance mismatch")
    if record.get("source_question_ids_sha256") != core.FULL_IDS_SHA256:
        raise core.IntegrityError("model_load full-cohort provenance mismatch")
    if record.get("source_both_gold_ids_sha256") != core.BOTH_GOLD_IDS_SHA256:
        raise core.IntegrityError("model_load both-gold provenance mismatch")
    recorded_fingerprints = record.get("condition_fingerprints")
    if not isinstance(recorded_fingerprints, dict) or set(recorded_fingerprints) != set(
        core.CONDITIONS
    ):
        raise core.IntegrityError("model_load condition fingerprint map is incomplete")
    for condition, fingerprint in recorded_fingerprints.items():
        _require_sha256(fingerprint, f"model_load {condition} fingerprint")
    if condition_fingerprints is not None and recorded_fingerprints != dict(
        condition_fingerprints
    ):
        raise core.IntegrityError("model_load condition fingerprints mismatch")
    if source is not None:
        if source.question_ids_sha256 != core.FULL_IDS_SHA256:
            raise core.IntegrityError("source authority full-cohort fingerprint mismatch")
        if core.ordered_ids_sha256(source.both_gold_ids) != core.BOTH_GOLD_IDS_SHA256:
            raise core.IntegrityError("source authority both-gold fingerprint mismatch")


def _validate_batch_specs_against_source(
    source: core.ReplaySource,
    specs: Sequence[BatchSpec],
    *,
    execution_fingerprint_sha256: str,
    condition_fingerprints: Mapping[str, str],
) -> None:
    expected: list[tuple[str, str, str, core.FrozenBatch]] = []
    expected.extend(
        (SENTINEL_PHASE, SENTINEL_CONDITION, execution_fingerprint_sha256, batch)
        for batch in sentinel_batches(source)
    )
    for condition in core.CONDITIONS:
        expected.extend(
            ("scored", condition, condition_fingerprints[condition], batch)
            for batch in core.condition_batches(source, condition)
        )
    if len(specs) != len(expected):
        raise core.IntegrityError("batch plan length differs from source manifests")
    for index, (spec, authority) in enumerate(zip(specs, expected)):
        phase, condition, fingerprint, frozen = authority
        expected_batch_id = f"{condition}:{frozen.stage}:{frozen.ordinal:06d}"
        expected_members = [
            (member.question_id, member.stage, member.call_index)
            for member in frozen.members
        ]
        actual_members = [
            (member.question_id, member.stage, member.call_index)
            for member in spec.members
        ]
        observed = (
            spec.phase,
            spec.condition,
            spec.condition_fingerprint_sha256,
            spec.stage,
            spec.batch_id,
            spec.batch_ordinal,
            actual_members,
        )
        required = (
            phase,
            condition,
            fingerprint,
            frozen.stage,
            expected_batch_id,
            frozen.ordinal,
            expected_members,
        )
        if observed != required:
            raise core.IntegrityError(
                f"batch plan entry {index} differs from sentinel/condition source manifest"
            )


def _validate_answer_shape(
    record: Mapping[str, Any],
    condition_fingerprints: Mapping[str, str] | None,
) -> tuple[str, str]:
    if record.get("record_type") != "answer":
        raise core.IntegrityError("answers stream contains a non-answer row")
    missing = [field for field in REQUIRED_ANSWER_FIELDS if field not in record]
    if missing:
        raise core.IntegrityError(f"answer is missing required fields: {missing!r}")
    condition = record.get("condition")
    question_id = record.get("question_id")
    if condition not in core.CONDITIONS or not isinstance(question_id, str) or not question_id:
        raise core.IntegrityError("answer has an invalid condition-aware key")
    fingerprint = _require_sha256(
        record.get("condition_fingerprint_sha256"),
        f"answer {condition}/{question_id} condition fingerprint",
    )
    if condition_fingerprints is not None and fingerprint != condition_fingerprints[condition]:
        raise core.IntegrityError(
            f"answer {condition}/{question_id} condition fingerprint mismatch"
        )
    for field in ("question", "gold_answer", "predicted_answer", "final_answer_source", "stop_reason"):
        if not isinstance(record.get(field), str):
            raise core.IntegrityError(f"answer {condition}/{question_id} {field} is not text")
    if record.get("retrieval_stratum") not in {"hidden_bridge", "fully_named"}:
        raise core.IntegrityError(f"answer {condition}/{question_id} has invalid stratum")
    if not isinstance(record.get("source_retrieval_all_gold"), bool):
        raise core.IntegrityError(
            f"answer {condition}/{question_id} has invalid retrieval provenance"
        )
    for field in ("f1", "em"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise core.IntegrityError(f"answer {condition}/{question_id} {field} is not numeric")
    _validate_nonnegative_int(
        record.get("executed_steps"), f"answer {condition}/{question_id} executed_steps"
    )
    return condition, question_id


def validate_complete_streams(
    calls: Sequence[dict],
    answers: Sequence[dict],
    batch_specs: Sequence[BatchSpec],
    *,
    identity: Mapping[str, Any] | None = None,
    source: core.ReplaySource | None = None,
    execution_fingerprint_sha256: str | None = None,
    condition_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Enforce cardinality, provenance, source manifests, and answer membership."""
    strict_values = (source, execution_fingerprint_sha256, condition_fingerprints)
    if any(value is not None for value in strict_values) and not all(
        value is not None for value in strict_values
    ):
        raise ValueError(
            "source-aware validation requires source, execution fingerprint, and "
            "condition fingerprints together"
        )
    if execution_fingerprint_sha256 is not None:
        _require_sha256(execution_fingerprint_sha256, "execution fingerprint")
    if condition_fingerprints is not None:
        condition_fingerprints = dict(condition_fingerprints)
        if set(condition_fingerprints) != set(core.CONDITIONS):
            raise ValueError("condition fingerprints are incomplete")
        for condition, fingerprint in condition_fingerprints.items():
            _require_sha256(fingerprint, f"{condition} condition fingerprint")

    specs = tuple(batch_specs)
    if source is not None:
        _validate_batch_specs_against_source(
            source,
            specs,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
            condition_fingerprints=condition_fingerprints,
        )
    audit = audit_resume(calls, specs, identity=identity)
    if not audit.complete:
        raise core.IntegrityError("complete stream has missing or uncertified replay data")
    _validate_model_load_provenance(
        audit.model_load,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        condition_fingerprints=condition_fingerprints,
        source=source,
    )

    model_loads = [record for record in calls if record.get("record_type") == "model_load"]
    agent_calls = [record for record in calls if record.get("record_type") == "agent_call"]
    certificates = [record for record in calls if record.get("record_type") == "batch"]
    sentinel_calls = [record for record in agent_calls if record.get("phase") == SENTINEL_PHASE]
    scored_calls = [record for record in agent_calls if record.get("phase") == "scored"]
    sentinel_certificates = [record for record in certificates if record.get("phase") == SENTINEL_PHASE]
    scored_certificates = [record for record in certificates if record.get("phase") == "scored"]
    counts = {
        "model_load": len(model_loads),
        "sentinel_calls": len(sentinel_calls),
        "scored_calls": len(scored_calls),
        "agent_calls": len(agent_calls),
        "sentinel_certificates": len(sentinel_certificates),
        "scored_certificates": len(scored_certificates),
        "batch_certificates": len(certificates),
        "answers": len(answers),
    }
    if counts != EXPECTED_COMPLETE_COUNTS:
        raise core.IntegrityError(
            f"complete replay cardinality mismatch: {counts!r} != {EXPECTED_COMPLETE_COUNTS!r}"
        )
    allocations: dict[tuple[str, str], dict[str, int]] = {}
    for record_type, records, count_key in (
        ("call", agent_calls, "calls"),
        ("certificate", certificates, "certificates"),
    ):
        for record in records:
            key = (record.get("phase"), record.get("condition"))
            if key not in EXPECTED_STREAM_ALLOCATIONS:
                raise core.IntegrityError(f"unexpected {record_type} phase/condition allocation {key!r}")
            allocation = allocations.setdefault(key, {"calls": 0, "certificates": 0})
            allocation[count_key] += 1
    if allocations != EXPECTED_STREAM_ALLOCATIONS:
        raise core.IntegrityError(
            f"phase/condition stream allocation mismatch: {allocations!r}"
        )
    if len(specs) != EXPECTED_COMPLETE_COUNTS["batch_certificates"]:
        raise core.IntegrityError("expected batch plan does not contain 268 certificates")
    if sum(len(spec.members) for spec in specs) != EXPECTED_COMPLETE_COUNTS["agent_calls"]:
        raise core.IntegrityError("expected batch plan does not contain 1,059 calls")

    answer_keys: set[tuple[str, str]] = set()
    condition_ids = {condition: set() for condition in core.CONDITIONS}
    for record in answers:
        if not isinstance(record, Mapping):
            raise core.IntegrityError("answers stream contains a non-object row")
        key = _validate_answer_shape(record, condition_fingerprints)
        if key in answer_keys:
            raise core.IntegrityError(f"duplicate answer key {key!r}")
        answer_keys.add(key)
        condition_ids[key[0]].add(key[1])
    expected_answer_counts = {
        core.SPANS_PLUS_PASSAGES: 200,
        core.PASSAGES_ONLY: 128,
    }
    if {condition: len(ids) for condition, ids in condition_ids.items()} != expected_answer_counts:
        raise core.IntegrityError("condition answer counts changed")
    if source is not None:
        expected_ids = {
            core.SPANS_PLUS_PASSAGES: set(source.question_ids),
            core.PASSAGES_ONLY: set(source.both_gold_ids),
        }
        if condition_ids != expected_ids:
            raise core.IntegrityError("answer ID sets differ from frozen source cohorts")
    return counts


def fsync_directory(path: Path) -> None:
    """Make preceding directory-entry renames durable on POSIX filesystems."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_object(path: Path, label: str) -> dict:
    try:
        _require_regular_leaf(path, label)
        descriptor = _open_regular_fd(path, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise core.IntegrityError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise core.IntegrityError(f"{label} {path} is not a JSON object")
    return value


def _write_json_fsync(path: Path, payload: dict) -> None:
    if _leaf_present(path):
        _require_regular_leaf(path, "JSON publication target")
        raise core.IntegrityError(f"refusing to overwrite JSON artifact: {path}")
    descriptor = _open_regular_fd(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_objects(path: Path, label: str) -> list[dict]:
    records: list[dict] = []
    try:
        _require_regular_leaf(path, label)
        descriptor = _open_regular_fd(path, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise core.IntegrityError(
                        f"{label} {path} line {line_number} is not an object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise core.IntegrityError(f"cannot read {label} {path}: {exc}") from exc
    return records


def _published_or_partial_path(directory: Path, name: str) -> Path:
    final_path = directory / name
    partial_path = directory / f"{name}.partial"
    if _require_regular_leaf(final_path, name, allow_absent=True):
        return final_path
    if _require_regular_leaf(partial_path, f"partial {name}", allow_absent=True):
        return partial_path
    raise core.IntegrityError(f"publication artifact is missing: {name}")


def _validate_publication_data(
    directory: Path,
    validation_payload: PublicationValidationPayload,
) -> dict[str, int]:
    manifest = _read_json_object(
        _published_or_partial_path(directory, "manifest.json"),
        "replay manifest",
    )
    if _canonical_json_bytes(manifest) != _canonical_json_bytes(
        dict(validation_payload.expected_manifest)
    ):
        raise core.IntegrityError("published manifest differs from static authority")
    if core.canonical_json_sha256(manifest) != (
        validation_payload.replay_fingerprint_sha256
    ):
        raise core.IntegrityError("published manifest replay fingerprint mismatch")

    calls = _read_jsonl_objects(
        _published_or_partial_path(directory, "calls.jsonl"), "replay calls"
    )
    answers = _read_jsonl_objects(
        _published_or_partial_path(directory, "answers.jsonl"), "replay answers"
    )
    counts = validate_complete_streams(
        calls,
        answers,
        validation_payload.batch_specs,
        identity=validation_payload.identity,
        source=validation_payload.source,
        execution_fingerprint_sha256=(
            validation_payload.execution_fingerprint_sha256
        ),
        condition_fingerprints=validation_payload.condition_fingerprints,
    )

    scored_index = {
        _call_key(record): record
        for record in calls
        if record.get("record_type") == "agent_call"
        and record.get("phase") == "scored"
    }
    results: dict[str, core.ConditionResult] = {}
    expected_answers: dict[tuple[str, str], dict] = {}
    expected_order: list[tuple[str, str]] = []
    for condition in core.CONDITIONS:
        fingerprint = validation_payload.condition_fingerprints[condition]
        result = core.build_condition_result(
            validation_payload.source,
            condition,
            scored_index,
            scored_index,
            fingerprint,
        )
        results[condition] = result
        for question_id in result.question_ids:
            key = (condition, question_id)
            expected_order.append(key)
            expected_answers[key] = {
                **result.answer_records[question_id],
                "condition_fingerprint_sha256": fingerprint,
            }
    observed_order = [
        (record.get("condition"), record.get("question_id")) for record in answers
    ]
    if observed_order != expected_order:
        raise core.IntegrityError("published answers are not in canonical cohort order")
    for record in answers:
        key = (record["condition"], record["question_id"])
        if _canonical_json_bytes(record) != _canonical_json_bytes(expected_answers[key]):
            raise core.IntegrityError(
                f"published answer differs from reconstructed result: {key!r}"
            )

    summary = _read_json_object(
        _published_or_partial_path(directory, "summary.json"), "replay summary"
    )
    expected_summary = core.score_replay(validation_payload.source, results)
    if _canonical_json_bytes(summary) != _canonical_json_bytes(expected_summary):
        raise core.IntegrityError(
            "published summary differs from recomputed calls/answers report"
        )
    return counts


def validate_published_result(
    output_dir: Path,
    *,
    validation_payload: PublicationValidationPayload,
) -> dict:
    directory = Path(output_dir)
    _validate_publication_data(directory, validation_payload)
    meta_path = directory / "meta.json"
    if not _require_regular_leaf(
        meta_path,
        "final completion certificate",
        allow_absent=True,
    ):
        raise core.IntegrityError(f"final completion certificate is missing: {meta_path}")
    meta = _read_json_object(meta_path, "completion certificate")
    if meta.get("status") != "complete":
        raise core.IntegrityError("meta.json is not a complete certificate")
    hashes = meta.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(PUBLISHED_DATA_ARTIFACTS):
        raise core.IntegrityError("meta.json has an invalid artifact hash manifest")
    for name in PUBLISHED_DATA_ARTIFACTS:
        path = directory / name
        if not _require_regular_leaf(
            path,
            f"published artifact {name}",
            allow_absent=True,
        ) or core.sha256_file(path) != hashes[name]:
            raise core.IntegrityError(f"published artifact hash mismatch: {name}")
    base_hash = meta.get("meta_payload_sha256")
    base_payload = {
        key: value
        for key, value in meta.items()
        if key
        not in {
            "status",
            "artifact_sha256",
            "meta_payload_sha256",
            "validation_contract_sha256",
        }
    }
    if base_hash != core.canonical_json_sha256(base_payload):
        raise core.IntegrityError("meta.json payload hash mismatch")
    if meta.get("validation_contract_sha256") != validation_payload.contract_sha256:
        raise core.IntegrityError("meta.json validation contract mismatch")
    return meta


def _fsync_regular_file(path: Path) -> None:
    _require_regular_leaf(path, "publication artifact")
    descriptor = _open_regular_fd(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_artifact_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in PUBLISHED_DATA_ARTIFACTS:
        final_path = directory / name
        partial_path = directory / f"{name}.partial"
        final_exists = _require_regular_leaf(
            final_path,
            f"final publication artifact {name}",
            allow_absent=True,
        )
        partial_exists = _require_regular_leaf(
            partial_path,
            f"partial publication artifact {name}",
            allow_absent=True,
        )
        if not final_exists and not partial_exists:
            raise core.IntegrityError(
                f"publication artifact is missing in final and partial form: {name}"
            )
        if partial_exists:
            _fsync_regular_file(partial_path)
        if final_exists:
            _fsync_regular_file(final_path)
        final_hash = core.sha256_file(final_path) if final_exists else None
        partial_hash = core.sha256_file(partial_path) if partial_exists else None
        if final_hash is not None and partial_hash is not None and final_hash != partial_hash:
            raise core.IntegrityError(
                f"final and partial publication artifacts differ: {name}"
            )
        hashes[name] = final_hash or partial_hash or ""
    return hashes


def publish_meta_last(
    output_dir: Path,
    meta_payload: Mapping[str, Any],
    *,
    validation_payload: PublicationValidationPayload,
    replace_fn=os.replace,
) -> dict:
    """Recover data renames safely and publish ``meta.json`` strictly last."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final_meta = directory / "meta.json"
    partial_meta = directory / "meta.json.partial"
    if _require_regular_leaf(
        final_meta,
        "final completion certificate",
        allow_absent=True,
    ):
        validate_published_result(
            directory, validation_payload=validation_payload
        )
        raise core.IntegrityError("refusing to overwrite complete replay result")
    protected = {
        "status",
        "artifact_sha256",
        "meta_payload_sha256",
        "validation_contract_sha256",
    }
    overlap = protected.intersection(meta_payload)
    if overlap:
        raise ValueError(f"meta payload contains publisher-owned keys: {sorted(overlap)!r}")
    base_payload = dict(meta_payload)
    base_sha256 = core.canonical_json_sha256(base_payload)
    _validate_publication_data(directory, validation_payload)
    observed_hashes = _publication_artifact_hashes(directory)

    if _require_regular_leaf(
        partial_meta,
        "partial completion certificate",
        allow_absent=True,
    ):
        certificate = _read_json_object(partial_meta, "partial completion certificate")
        if certificate.get("status") != "complete":
            raise core.IntegrityError("partial meta is not a complete certificate")
        certificate_base = {
            key: value for key, value in certificate.items() if key not in protected
        }
        certificate_base_sha256 = core.canonical_json_sha256(certificate_base)
        if (
            _canonical_json_bytes(certificate_base)
            != _canonical_json_bytes(base_payload)
            or certificate_base_sha256 != base_sha256
            or certificate.get("meta_payload_sha256") != certificate_base_sha256
        ):
            raise core.IntegrityError("partial meta belongs to a different replay payload")
        if certificate.get("artifact_sha256") != observed_hashes:
            raise core.IntegrityError("partial meta artifact hashes changed during recovery")
        if certificate.get("validation_contract_sha256") != (
            validation_payload.contract_sha256
        ):
            raise core.IntegrityError("partial meta validation contract changed")
    else:
        certificate = {
            **base_payload,
            "status": "complete",
            "artifact_sha256": observed_hashes,
            "meta_payload_sha256": base_sha256,
            "validation_contract_sha256": validation_payload.contract_sha256,
        }
        temp_meta = directory / f"meta.json.partial.{os.getpid()}.tmp"
        if _leaf_present(temp_meta):
            _require_regular_leaf(temp_meta, "publisher temporary certificate")
            raise core.IntegrityError(f"stale publisher temporary file: {temp_meta}")
        try:
            _write_json_fsync(temp_meta, certificate)
            replace_fn(temp_meta, partial_meta)
            fsync_directory(directory)
        except BaseException:
            if _leaf_present(temp_meta):
                temp_meta.unlink()
                fsync_directory(directory)
            raise

    expected_hashes = certificate["artifact_sha256"]
    for name in PUBLISHED_DATA_ARTIFACTS:
        final_path = directory / name
        partial_path = directory / f"{name}.partial"
        final_exists = _require_regular_leaf(
            final_path,
            f"recovery final artifact {name}",
            allow_absent=True,
        )
        partial_exists = _require_regular_leaf(
            partial_path,
            f"recovery partial artifact {name}",
            allow_absent=True,
        )
        if final_exists:
            _fsync_regular_file(final_path)
            if core.sha256_file(final_path) != expected_hashes[name]:
                raise core.IntegrityError(f"recovery final hash changed: {name}")
            if partial_exists:
                _fsync_regular_file(partial_path)
                if core.sha256_file(partial_path) != expected_hashes[name]:
                    raise core.IntegrityError(
                        f"recovery partial hash changed: {name}"
                    )
                partial_path.unlink()
                fsync_directory(directory)
            continue
        if not partial_exists:
            raise core.IntegrityError(f"recovery partial artifact is missing: {name}")
        _fsync_regular_file(partial_path)
        if core.sha256_file(partial_path) != expected_hashes[name]:
            raise core.IntegrityError(f"recovery partial hash changed: {name}")
        replace_fn(partial_path, final_path)
        fsync_directory(directory)
    for name in PUBLISHED_DATA_ARTIFACTS:
        final_path = directory / name
        _fsync_regular_file(final_path)
        if core.sha256_file(final_path) != expected_hashes[name]:
            raise core.IntegrityError(f"post-rename artifact hash changed: {name}")
    _validate_publication_data(directory, validation_payload)
    _fsync_regular_file(partial_meta)
    replace_fn(partial_meta, final_meta)
    fsync_directory(directory)
    return validate_published_result(
        directory, validation_payload=validation_payload
    )


def build_cpu_audit(source: core.ReplaySource) -> dict:
    """Run every source/manfiest check that does not require model weights."""
    audit = core.audit_source(source)
    source_strata = {
        stratum: sum(
            question.stratum == stratum for question in source.scoring.values()
        )
        for stratum in ("hidden_bridge", "fully_named")
    }
    both_gold_strata = {
        stratum: sum(
            source.scoring[qid].stratum == stratum for qid in source.both_gold_ids
        )
        for stratum in ("hidden_bridge", "fully_named")
    }
    retrieval_depths = {qid: 0 for qid in source.question_ids}
    for record in core.source_qa_records(source):
        if (record.get("consumer_input") or {}).get("task_type") == "question-answering":
            retrieval_depths[record["question_id"]] += 1
    retrieval_depth_distribution = {
        depth: sum(value == depth for value in retrieval_depths.values())
        for depth in sorted(set(retrieval_depths.values()))
    }
    conditions = {}
    for condition in core.CONDITIONS:
        calls = core.condition_call_keys(source, condition)
        batches = core.condition_batches(source, condition)
        conditions[condition] = {"calls": len(calls), "batches": len(batches)}
    matched_prompt_hashes = audit.qa_prompt_hash_matches + audit.summary_prompt_hash_matches
    result = {
        "source_questions": len(source.question_ids),
        "source_strata": source_strata,
        "both_gold_questions": len(source.both_gold_ids),
        "both_gold_strata": both_gold_strata,
        "source_qa_stage_counts": dict(audit.qa_stage_counts),
        "retrieval_depth_distribution": retrieval_depth_distribution,
        "source_calls": {
            "qa": sum(audit.qa_stage_counts.values()),
            "plan_summary": audit.summary_prompt_hash_matches,
            "extractor": audit.extractor_joins,
        },
        "source_prompt_hashes": {"matched": matched_prompt_hashes, "expected": 627},
        "conditions": conditions,
        "scored_total": {
            "calls": sum(item["calls"] for item in conditions.values()),
            "batches": sum(item["batches"] for item in conditions.values()),
        },
        "answer_survival": core.extractor_survival_headline(source),
        "normalizer_near_match": core.extractor_near_match_diagnostic(source),
    }
    expected = {
        "source_questions": 200,
        "source_strata": {"hidden_bridge": 160, "fully_named": 40},
        "both_gold_questions": 128,
        "both_gold_strata": {"hidden_bridge": 95, "fully_named": 33},
        "source_prompt_hashes": {"matched": 627, "expected": 627},
        "conditions": {
            core.SPANS_PLUS_PASSAGES: {"calls": 627, "batches": 158},
            core.PASSAGES_ONLY: {"calls": 411, "batches": 104},
        },
        "scored_total": {"calls": 1038, "batches": 262},
        "retrieval_depth_distribution": {1: 13, 2: 156, 3: 24, 4: 6, 5: 1},
    }
    for key, value in expected.items():
        if result[key] != value:
            raise core.IntegrityError(
                f"CPU replay audit {key} mismatch: {result[key]!r} != {value!r}"
            )
    return result


def print_cpu_audit(report: Mapping[str, Any]) -> None:
    source_strata = report["source_strata"]
    subset_strata = report["both_gold_strata"]
    source_calls = report["source_calls"]
    prompt_hashes = report["source_prompt_hashes"]
    plus = report["conditions"][core.SPANS_PLUS_PASSAGES]
    only = report["conditions"][core.PASSAGES_ONLY]
    total = report["scored_total"]
    survival = report["answer_survival"]
    near_match = report["normalizer_near_match"]["all_questions"]
    print(
        f"source questions: {report['source_questions']} "
        f"(hidden_bridge={source_strata['hidden_bridge']}, "
        f"fully_named={source_strata['fully_named']})"
    )
    print(
        f"both-gold subset: {report['both_gold_questions']} "
        f"(hidden_bridge={subset_strata['hidden_bridge']}, "
        f"fully_named={subset_strata['fully_named']})"
    )
    print(
        f"source calls: qa={source_calls['qa']} "
        f"plan_summary={source_calls['plan_summary']} extractor={source_calls['extractor']}"
    )
    print(
        f"source prompt hashes: {prompt_hashes['matched']}/{prompt_hashes['expected']}"
    )
    print(f"retrieval depth distribution: {report['retrieval_depth_distribution']}")
    print(
        f"{core.SPANS_PLUS_PASSAGES}: {plus['calls']} calls, {plus['batches']} batches"
    )
    print(f"{core.PASSAGES_ONLY}: {only['calls']} calls, {only['batches']} batches")
    print(f"scored total: {total['calls']} calls, {total['batches']} batches")
    print(
        "answer survival: "
        f"{survival['raw_producer_present']}/{survival['n']} -> "
        f"{survival['normalized_qa_prompt_present']}/{survival['n']}"
    )
    print(
        "not_in_source near matches: "
        f"{near_match['at_or_above']['0.80']}/"
        f"{near_match['rejected_not_in_source_spans']} at token-F1>=0.80"
    )
    print("AUDIT_PASS")


def _resolve_repo_path(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not _path_is_within(resolved, ROOT):
        raise core.IntegrityError(f"{label} is outside the repository: {resolved}")
    return resolved


def _replay_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return _require_git_commit(completed.stdout.strip(), "replay git commit")


def _gpu_metadata() -> dict:
    from src import runner

    return dict(runner._gpu_metadata())


def _require_t4_preload(gpu: Mapping[str, Any]) -> None:
    if gpu.get("gpu_available") is not True:
        raise core.IntegrityError(
            "Tesla T4 GPU is required for --execute; no CUDA GPU is available"
        )
    if gpu.get("gpu_name") != EXPECTED_GPU_NAME:
        raise core.IntegrityError(
            f"Tesla T4 GPU is required for --execute; got {gpu.get('gpu_name')!r}"
        )
    if gpu.get("gpu_compute_capability") != EXPECTED_GPU_COMPUTE_CAPABILITY:
        raise core.IntegrityError(
            "Tesla T4 compute capability 7.5 is required for --execute; got "
            f"{gpu.get('gpu_compute_capability')!r}"
        )


def _runtime_library_versions() -> dict[str, str]:
    import torch

    versions = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    for distribution, label in (
        ("transformers", "transformers"),
        ("bitsandbytes", "bitsandbytes"),
        ("datasets", "datasets"),
    ):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise core.IntegrityError(
                f"required replay package is not installed: {distribution}"
            ) from exc
    return versions


def audit_source_prompt_runtime(
    tok,
    source: core.ReplaySource,
) -> tuple[list[dict], str]:
    manifest: list[dict] = []
    records = (*core.source_qa_records(source), *core.source_summary_records(source))
    for record in records:
        prompt_role = record.get("prompt_role")
        if prompt_role == "qa":
            fields = core.reconstruct_source_qa_fields(source, record)
        elif prompt_role == "plan_summary":
            fields = core.reconstruct_source_summary_fields(source, record)
        else:
            raise core.IntegrityError(
                f"unexpected source replay prompt role: {prompt_role!r}"
            )
        messages = prompts.build_messages(prompt_role, **fields)
        context_window = record.get("context_window_tokens")
        audit = audit_prompt_runtime(
            tok,
            messages,
            prompt_role=prompt_role,
            recorded_context_window_tokens=context_window,
        )
        if audit.message_object_sha256 != record.get("rendered_prompt_sha256"):
            raise core.IntegrityError(
                f"source runtime message hash changed for "
                f"{record.get('question_id')}/{record.get('stage')}"
            )
        manifest.append(
            {
                "question_id": record["question_id"],
                "stage": record["stage"],
                "call_index": record["call_index"],
                "prompt_role": prompt_role,
                "source_parent_record_sha256": core.canonical_json_sha256(record),
                **asdict(audit),
            }
        )
    if len(manifest) != 627:
        raise core.IntegrityError(
            f"source runtime prompt manifest has {len(manifest)} entries, expected 627"
        )
    return manifest, core.canonical_json_sha256(manifest)


@dataclass(frozen=True)
class PreparedReplayCall:
    request: dict
    audit: PromptRuntimeAudit
    member: MemberSpec

    @property
    def prompt_manifest_record(self) -> dict:
        return {
            "condition": self.member.condition,
            "question_id": self.member.question_id,
            "stage": self.member.stage,
            "call_index": self.member.call_index,
            "parent_key": list(self.member.parent_key),
            "parent_record_sha256": self.member.parent_record_sha256,
            "source_message_sha256": self.member.source_message_sha256,
            "treatment_message_sha256": self.member.treatment_message_sha256,
            **asdict(self.audit),
        }


def _source_replay_request(source: core.ReplaySource, record: dict) -> dict:
    prompt_role = record.get("prompt_role")
    if prompt_role == "qa":
        fields = core.reconstruct_source_qa_fields(source, record)
    elif prompt_role == "plan_summary":
        fields = core.reconstruct_source_summary_fields(source, record)
    else:
        raise core.IntegrityError(
            f"unsupported source sentinel prompt role: {prompt_role!r}"
        )
    parent_key = [record["question_id"], record["stage"], record["call_index"]]
    return {
        "question_id": record["question_id"],
        "stage": record["stage"],
        "call_index": record["call_index"],
        "fields": fields,
        "consumer_payload_source": record.get("consumer_payload_source"),
        "consumer_input": record.get("consumer_input"),
        "source_parent_key": parent_key,
        "source_parent_record_sha256": core.canonical_json_sha256(record),
        "source_message_sha256": record["rendered_prompt_sha256"],
        "treatment_message_sha256": record["rendered_prompt_sha256"],
    }


def _prepare_replay_call(
    tok,
    request: dict,
    parent: dict,
    condition: str,
    batch_member_index: int,
) -> PreparedReplayCall:
    stage = request["stage"]
    prompt_role = "plan_summary" if stage == "plan_summary" else "qa"
    messages = prompts.build_messages(prompt_role, **request["fields"])
    audit = audit_prompt_runtime(
        tok,
        messages,
        prompt_role=prompt_role,
        recorded_context_window_tokens=parent.get("context_window_tokens"),
    )
    if audit.message_object_sha256 != request.get("treatment_message_sha256"):
        raise core.IntegrityError(
            f"treated message hash changed for "
            f"{condition}/{request['question_id']}/{stage}"
        )
    parent_key = tuple(request.get("source_parent_key") or ())
    member = MemberSpec(
        condition=condition,
        question_id=request["question_id"],
        stage=stage,
        call_index=request["call_index"],
        batch_member_index=batch_member_index,
        parent_key=parent_key,
        parent_record_sha256=request["source_parent_record_sha256"],
        source_message_sha256=request["source_message_sha256"],
        treatment_message_sha256=request["treatment_message_sha256"],
        rendered_chat_sha256=audit.rendered_chat_sha256,
    )
    return PreparedReplayCall(request=dict(request), audit=audit, member=member)


def _prepare_replay_batch(
    tok,
    source: core.ReplaySource,
    frozen_batch: core.FrozenBatch,
    requests: Sequence[dict],
    *,
    phase: str,
    condition: str,
    condition_fingerprint_sha256: str,
) -> tuple[BatchSpec, tuple[PreparedReplayCall, ...]]:
    if len(requests) != len(frozen_batch.members):
        raise core.IntegrityError(
            f"prepared batch {condition}/{frozen_batch.stage}/{frozen_batch.ordinal} "
            f"has {len(requests)}/{len(frozen_batch.members)} requests"
        )
    prepared: list[PreparedReplayCall] = []
    for member_index, (authority, request) in enumerate(
        zip(frozen_batch.members, requests)
    ):
        observed = (
            request.get("question_id"),
            request.get("stage"),
            request.get("call_index"),
        )
        expected = (authority.question_id, authority.stage, authority.call_index)
        if observed != expected:
            raise core.IntegrityError(
                f"prepared request order differs from frozen batch: {observed!r} != "
                f"{expected!r}"
            )
        parent = source.baseline_index.get(expected)
        if not isinstance(parent, dict):
            raise core.IntegrityError(f"missing source parent for {expected!r}")
        prepared.append(
            _prepare_replay_call(
                tok,
                request,
                parent,
                condition,
                member_index,
            )
        )
    spec = make_batch_spec(
        phase=phase,
        condition=condition,
        condition_fingerprint_sha256=condition_fingerprint_sha256,
        stage=frozen_batch.stage,
        batch_ordinal=frozen_batch.ordinal,
        members=[item.member for item in prepared],
    )
    return spec, tuple(prepared)


def _split_existing_stream(records: Sequence[dict]) -> tuple[dict | None, list[dict]]:
    meaningful = [record for record in records if record.get("record_type") != "store_repair"]
    model_loads = [
        (index, record)
        for index, record in enumerate(meaningful)
        if record.get("record_type") == "model_load"
    ]
    if not model_loads:
        if meaningful:
            raise core.IntegrityError("partial replay rows exist without model_load")
        return None, []
    if len(model_loads) != 1 or model_loads[0][0] != 0:
        raise core.IntegrityError("model_load must occur exactly once at stream start")
    return model_loads[0][1], meaningful[1:]


class ExistingStreamCursor:
    """Consume a previously persisted stream in newly reconstructed batch order."""

    def __init__(
        self,
        records: Sequence[dict],
        model_load: dict,
        identity: Mapping[str, Any],
    ):
        self._records = [
            record for record in records if record.get("record_type") != "store_repair"
        ]
        self._model_load = model_load
        self._identity = dict(identity)
        self._position = 0

    def consume(self, batch: BatchSpec) -> tuple[str, tuple[dict, ...]]:
        if self._position >= len(self._records):
            return "missing", ()
        calls: list[dict] = []
        while (
            self._position < len(self._records)
            and self._records[self._position].get("record_type") == "agent_call"
            and len(calls) < len(batch.members)
        ):
            calls.append(self._records[self._position])
            self._position += 1
        if not calls:
            record_type = self._records[self._position].get("record_type")
            raise core.IntegrityError(
                f"expected calls for {batch.key!r}, found {record_type!r}"
            )
        if self._position < len(self._records):
            next_record = self._records[self._position]
            if len(calls) != len(batch.members) or next_record.get("record_type") != "batch":
                raise core.IntegrityError(
                    f"uncertified batch {batch.key!r} is not the final stream segment"
                )
            self._position += 1
            segment = [self._model_load, *calls, next_record]
            audit = audit_resume(segment, (batch,), identity=self._identity)
            if not audit.complete:
                raise core.IntegrityError(
                    f"persisted batch {batch.key!r} did not validate as complete"
                )
            return "certified", tuple(calls)
        audit = audit_resume(
            [self._model_load, *calls],
            (batch,),
            identity=self._identity,
        )
        orphan = audit.orphan_calls_by_batch.get(batch.key)
        if orphan is None:
            raise core.IntegrityError(
                f"final persisted segment is not a valid orphan for {batch.key!r}"
            )
        return "orphan", tuple(orphan)

    def assert_exhausted(self) -> None:
        if self._position != len(self._records):
            raise core.IntegrityError(
                f"resume stream has {len(self._records) - self._position} extra rows"
            )


def _run_prepared_batch(
    model,
    tok,
    prepared: Sequence[PreparedReplayCall],
    batch: BatchSpec,
    *,
    run_calls_fn,
    run_id: str,
    execution_session_id: str,
    gpu: Mapping[str, Any],
    model_config_fingerprint: str,
    execution_fingerprint_sha256: str,
    question_manifest_sha256: str,
) -> tuple[tuple[dict, ...], dict]:
    requests = [
        {
            "question_id": item.request["question_id"],
            "call_index": item.request["call_index"],
            "fields": item.request["fields"],
            "consumer_payload_source": item.request.get("consumer_payload_source"),
            "consumer_input": item.request.get("consumer_input"),
        }
        for item in prepared
    ]
    generated, telemetry = run_calls_fn(
        model,
        tok,
        batch.stage,
        requests,
        PRECISION,
        run_id,
        batch_size=BATCH_SIZE,
        log_confidence=False,
        model_id=MODEL_ID,
        batch_id=batch.batch_id,
        execution_session_id=execution_session_id,
        gpu_metadata=dict(gpu),
        timing_eligible=False,
        phase=batch.phase,
        config_fingerprint=model_config_fingerprint,
        model_revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        question_manifest_sha256=question_manifest_sha256,
        batch_ordinal=batch.batch_ordinal,
        experiment_fingerprint=execution_fingerprint_sha256,
        force_full_generation=False,
        return_batch_record=True,
    )
    if len(generated) != len(prepared):
        raise core.IntegrityError(
            f"generated batch {batch.key!r} returned "
            f"{len(generated)}/{len(prepared)} records"
        )
    identity = _default_replay_identity()
    augmented: list[dict] = []
    for item, record in zip(prepared, generated):
        if record.get("rendered_prompt_sha256") != item.audit.message_object_sha256:
            raise core.IntegrityError(
                f"agent renderer message hash differs for {item.member.key!r}"
            )
        validate_generated_prompt(record, item.audit)
        enriched = {
            **record,
            "message_object_sha256": item.audit.message_object_sha256,
            "rendered_chat_sha256": item.audit.rendered_chat_sha256,
            "output_ceiling_tokens": item.audit.output_ceiling_tokens,
            "prompt_plus_ceiling_tokens": item.audit.prompt_plus_ceiling_tokens,
        }
        augmented.append(
            augment_generated_call(
                enriched,
                item.member,
                batch,
                identity=identity,
            )
        )
    reject_thinking(augmented)
    if not isinstance(telemetry, dict):
        raise core.IntegrityError(f"batch telemetry is invalid for {batch.key!r}")
    return tuple(augmented), telemetry


def _persist_reconciled_batch(
    store: JsonlStore,
    batch: BatchSpec,
    orphan_records: Sequence[dict],
    regenerated_records: Sequence[dict],
    telemetry: Mapping[str, Any],
) -> tuple[dict, ...]:
    missing, certificate = reconcile_orphan_batch(
        batch,
        orphan_records,
        regenerated_records,
        identity=_default_replay_identity(),
    )
    certificate["generation_batch_telemetry"] = dict(telemetry)
    store.write([*missing, certificate])
    store.durable_flush()
    return (*tuple(orphan_records), *missing)


def _resolve_scored_batch(
    cursor: ExistingStreamCursor,
    store: JsonlStore,
    model,
    tok,
    prepared: Sequence[PreparedReplayCall],
    batch: BatchSpec,
    *,
    run_calls_fn,
    run_id: str,
    execution_session_id: str,
    gpu: Mapping[str, Any],
    model_config_fingerprint: str,
    execution_fingerprint_sha256: str,
    question_manifest_sha256: str,
    output_is_already_final: bool,
) -> tuple[dict, ...]:
    status, persisted = cursor.consume(batch)
    if status == "certified":
        return persisted
    if output_is_already_final:
        raise core.IntegrityError(
            "final calls artifact is incomplete; refusing to append to published data"
        )
    regenerated, telemetry = _run_prepared_batch(
        model,
        tok,
        prepared,
        batch,
        run_calls_fn=run_calls_fn,
        run_id=run_id,
        execution_session_id=execution_session_id,
        gpu=gpu,
        model_config_fingerprint=model_config_fingerprint,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        question_manifest_sha256=question_manifest_sha256,
    )
    return _persist_reconciled_batch(
        store,
        batch,
        persisted if status == "orphan" else (),
        regenerated,
        telemetry,
    )


def _load_replay_model(
    source: core.ReplaySource,
    gpu: Mapping[str, Any],
) -> tuple[Any, Any, dict, list[dict], str, dict, float]:
    from src import models

    model = tok = None
    try:
        started = time.perf_counter()
        model, tok = models.load_model(
            MODEL_ID,
            PRECISION,
            device="cuda:0",
            revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
        )
        load_seconds = time.perf_counter() - started
        resolved = models.resolved_revision_metadata(
            model,
            tok,
            MODEL_REVISION,
            MODEL_REVISION,
        )
        if resolved.get("resolved_model_revision") != MODEL_REVISION:
            raise core.IntegrityError(
                "loaded model revision differs from the pinned commit"
            )
        if resolved.get("resolved_tokenizer_revision") != MODEL_REVISION:
            raise core.IntegrityError(
                "loaded tokenizer revision differs from the pinned commit"
            )
        census = models.validate_loaded_precision(model, PRECISION, MODEL_ID)
        tokenizer_identity = capture_tokenizer_identity(tok)
        source_prompt_manifest, source_prompt_manifest_sha256 = (
            audit_source_prompt_runtime(tok, source)
        )
        runtime_identity = validate_execution_identity(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "precision": PRECISION,
                "batch_size": BATCH_SIZE,
                "enable_thinking": False,
                "census": census,
                "gpu": dict(gpu),
                "tokenizer_identity": asdict(tokenizer_identity),
                "library_versions": _runtime_library_versions(),
                "source_rendered_chat_manifest_sha256": (
                    source_prompt_manifest_sha256
                ),
                "source_rendered_chat_manifest_count": len(source_prompt_manifest),
                "replay_git_commit": _replay_git_commit(),
            }
        )
        model_config_fingerprint, model_config_payload = models.config_fingerprint(
            MODEL_ID,
            PRECISION,
            MODEL_REVISION,
            MODEL_REVISION,
        )
        return (
            model,
            tok,
            runtime_identity,
            source_prompt_manifest,
            model_config_fingerprint,
            model_config_payload,
            load_seconds,
        )
    except BaseException:
        model = tok = None
        try:
            models.unload()
        except Exception:
            pass
        raise


def _prepare_sentinel_plan(
    tok,
    source: core.ReplaySource,
    execution_fingerprint_sha256: str,
) -> tuple[tuple[BatchSpec, tuple[PreparedReplayCall, ...]], ...]:
    output = []
    for frozen_batch in sentinel_batches(source):
        requests = []
        for member in frozen_batch.members:
            parent = source.baseline_index.get(
                (member.question_id, member.stage, member.call_index)
            )
            if not isinstance(parent, dict):
                raise core.IntegrityError(f"missing sentinel parent {member!r}")
            requests.append(_source_replay_request(source, parent))
        output.append(
            _prepare_replay_batch(
                tok,
                source,
                frozen_batch,
                requests,
                phase=SENTINEL_PHASE,
                condition=SENTINEL_CONDITION,
                condition_fingerprint_sha256=execution_fingerprint_sha256,
            )
        )
    return tuple(output)


def _run_sentinel_buffer(
    model,
    tok,
    source: core.ReplaySource,
    sentinel_plan: Sequence[tuple[BatchSpec, tuple[PreparedReplayCall, ...]]],
    *,
    run_calls_fn,
    run_id: str,
    execution_session_id: str,
    gpu: Mapping[str, Any],
    model_config_fingerprint: str,
    execution_fingerprint_sha256: str,
) -> tuple[list[tuple[BatchSpec, tuple[dict, ...], dict]], dict]:
    buffered: list[tuple[BatchSpec, tuple[dict, ...], dict]] = []
    replay_records: list[dict] = []
    source_batch_records = sentinel_source_batch_records(source)
    report_batches = []
    for (batch, prepared), source_batch_record in zip(
        sentinel_plan, source_batch_records
    ):
        generated, telemetry = _run_prepared_batch(
            model,
            tok,
            prepared,
            batch,
            run_calls_fn=run_calls_fn,
            run_id=run_id,
            execution_session_id=execution_session_id,
            gpu=gpu,
            model_config_fingerprint=model_config_fingerprint,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
            question_manifest_sha256=core.FULL_IDS_SHA256,
        )
        paired_telemetry = {
            "source_batch": source_batch_record,
            "replay_batch": dict(telemetry),
            "think_tag_checks": {"source": True, "replay": True},
        }
        buffered.append((batch, generated, paired_telemetry))
        report_batches.append(
            {
                "stage": batch.stage,
                "source": source_batch_record,
                "replay": dict(telemetry),
            }
        )
        replay_records.extend(generated)
    report = compare_sentinel(sentinel_source_calls(source), replay_records)
    report["batch_telemetry"] = report_batches
    return buffered, report


def _persist_or_verify_sentinel(
    cursor: ExistingStreamCursor,
    store: JsonlStore,
    buffered: Sequence[tuple[BatchSpec, tuple[dict, ...], dict]],
    *,
    output_is_already_final: bool,
) -> dict[tuple[str, str, str, int], dict]:
    accepted: dict[tuple[str, str, str, int], dict] = {}
    for batch, regenerated, telemetry in buffered:
        status, persisted = cursor.consume(batch)
        if status == "certified":
            for existing, replay in zip(persisted, regenerated):
                compare_regenerated_output(existing, replay)
            resolved = persisted
        else:
            if output_is_already_final:
                raise core.IntegrityError(
                    "final calls artifact has an incomplete reproduction sentinel"
                )
            resolved = _persist_reconciled_batch(
                store,
                batch,
                persisted if status == "orphan" else (),
                regenerated,
                telemetry,
            )
        for record in resolved:
            key = _call_key(record)
            if key in accepted:
                raise core.IntegrityError(f"duplicate accepted sentinel call {key!r}")
            accepted[key] = record
    if len(accepted) != 21:
        raise core.IntegrityError(
            f"accepted sentinel contains {len(accepted)} calls, expected 21"
        )
    return accepted


def _execute_condition(
    condition: str,
    condition_fingerprint_sha256: str,
    source: core.ReplaySource,
    cursor: ExistingStreamCursor,
    store: JsonlStore,
    model,
    tok,
    *,
    run_calls_fn,
    run_id: str,
    execution_session_id: str,
    gpu: Mapping[str, Any],
    model_config_fingerprint: str,
    execution_fingerprint_sha256: str,
    output_is_already_final: bool,
    batch_specs: list[BatchSpec],
    prompt_manifest: list[dict],
) -> core.ConditionResult:
    treated_index: dict[tuple[str, str, str, int], dict] = {}
    for stage in core.QA_STAGES:
        for frozen_batch in core.condition_batches(source, condition, stage):
            requests = []
            for member in frozen_batch.members:
                parent = source.baseline_index.get(
                    (member.question_id, member.stage, member.call_index)
                )
                if not isinstance(parent, dict):
                    raise core.IntegrityError(
                        f"missing treated QA parent for {member!r}"
                    )
                requests.append(
                    core.build_treated_qa_call(
                        source,
                        parent,
                        condition,
                        treated_index,
                        condition_fingerprint_sha256,
                    )
                )
            batch, prepared = _prepare_replay_batch(
                tok,
                source,
                frozen_batch,
                requests,
                phase="scored",
                condition=condition,
                condition_fingerprint_sha256=condition_fingerprint_sha256,
            )
            batch_specs.append(batch)
            prompt_manifest.extend(item.prompt_manifest_record for item in prepared)
            resolved = _resolve_scored_batch(
                cursor,
                store,
                model,
                tok,
                prepared,
                batch,
                run_calls_fn=run_calls_fn,
                run_id=run_id,
                execution_session_id=execution_session_id,
                gpu=gpu,
                model_config_fingerprint=model_config_fingerprint,
                execution_fingerprint_sha256=execution_fingerprint_sha256,
                question_manifest_sha256=(
                    core.FULL_IDS_SHA256
                    if condition == core.SPANS_PLUS_PASSAGES
                    else core.BOTH_GOLD_IDS_SHA256
                ),
                output_is_already_final=output_is_already_final,
            )
            for record in resolved:
                key = _call_key(record)
                if key in treated_index:
                    raise core.IntegrityError(f"duplicate treated QA row {key!r}")
                treated_index[key] = record

    summary_records: dict[str, dict] = {}
    for frozen_batch in core.condition_batches(source, condition, "plan_summary"):
        requests = []
        for member in frozen_batch.members:
            history = core.rebuild_treated_history(
                source,
                member.question_id,
                treated_index,
                condition,
                condition_fingerprint_sha256,
            )
            requests.append(
                core.build_treated_summary_call(
                    source,
                    member.question_id,
                    history,
                    condition,
                )
            )
        batch, prepared = _prepare_replay_batch(
            tok,
            source,
            frozen_batch,
            requests,
            phase="scored",
            condition=condition,
            condition_fingerprint_sha256=condition_fingerprint_sha256,
        )
        batch_specs.append(batch)
        prompt_manifest.extend(item.prompt_manifest_record for item in prepared)
        resolved = _resolve_scored_batch(
            cursor,
            store,
            model,
            tok,
            prepared,
            batch,
            run_calls_fn=run_calls_fn,
            run_id=run_id,
            execution_session_id=execution_session_id,
            gpu=gpu,
            model_config_fingerprint=model_config_fingerprint,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
            question_manifest_sha256=(
                core.FULL_IDS_SHA256
                if condition == core.SPANS_PLUS_PASSAGES
                else core.BOTH_GOLD_IDS_SHA256
            ),
            output_is_already_final=output_is_already_final,
        )
        for record in resolved:
            qid = record["question_id"]
            if qid in summary_records:
                raise core.IntegrityError(f"duplicate treated summary row {condition}/{qid}")
            summary_records[qid] = record
            treated_index[_call_key(record)] = record

    result = core.build_condition_result(
        source,
        condition,
        treated_index,
        summary_records,
        condition_fingerprint_sha256,
    )
    expected_calls = 627 if condition == core.SPANS_PLUS_PASSAGES else 411
    if len(result.qa_records) + len(result.summary_records) != expected_calls:
        raise core.IntegrityError(
            f"{condition} result call count changed from {expected_calls}"
        )
    return result


def _write_json_artifact(output_dir: Path, name: str, payload: Mapping[str, Any]) -> None:
    final_path = output_dir / name
    partial_path = output_dir / f"{name}.partial"
    final_exists = _require_regular_leaf(
        final_path,
        f"final {name}",
        allow_absent=True,
    )
    partial_exists = _require_regular_leaf(
        partial_path,
        f"partial {name}",
        allow_absent=True,
    )
    if final_exists or partial_exists:
        existing = _read_json_object(
            final_path if final_exists else partial_path,
            name,
        )
        if _canonical_json_bytes(existing) != _canonical_json_bytes(dict(payload)):
            raise core.IntegrityError(f"existing {name} differs from reconstructed output")
        return
    _write_json_fsync(partial_path, dict(payload))


def _write_jsonl_artifact(output_dir: Path, name: str, records: Sequence[dict]) -> None:
    final_path = output_dir / name
    partial_path = output_dir / f"{name}.partial"
    final_exists = _require_regular_leaf(
        final_path,
        f"final {name}",
        allow_absent=True,
    )
    partial_exists = _require_regular_leaf(
        partial_path,
        f"partial {name}",
        allow_absent=True,
    )
    if final_exists or partial_exists:
        existing = _read_jsonl_objects(
            final_path if final_exists else partial_path,
            name,
        )
        if _canonical_json_bytes(existing) != _canonical_json_bytes(list(records)):
            raise core.IntegrityError(f"existing {name} differs from reconstructed output")
        return
    descriptor = _open_regular_fd(
        partial_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _final_replay_manifest(
    source: core.ReplaySource,
    execution_fingerprint_sha256: str,
    execution_fingerprint_payload: Mapping[str, Any],
    condition_fingerprints: Mapping[str, str],
    condition_fingerprint_payloads: Mapping[str, Mapping[str, Any]],
    source_prompt_manifest: Sequence[dict],
    replay_prompt_manifest: Sequence[dict],
    batch_specs: Sequence[BatchSpec],
) -> dict:
    if len(source_prompt_manifest) != 627 or len(replay_prompt_manifest) != 1059:
        raise core.IntegrityError(
            "final prompt manifests do not contain 627 source and 1,059 replay calls"
        )
    return {
        "schema": "frozen-upstream-passage-replay-manifest-v1",
        "source": {
            "artifact_sha256": dict(core.SOURCE_SHA256),
            "source_commit": core.SOURCE_COMMIT,
            "source_experiment_fingerprint": core.SOURCE_EXPERIMENT_FINGERPRINT,
            "question_ids": list(source.question_ids),
            "question_ids_sha256": core.FULL_IDS_SHA256,
            "both_gold_ids": list(source.both_gold_ids),
            "both_gold_ids_sha256": core.BOTH_GOLD_IDS_SHA256,
        },
        "execution_fingerprint_sha256": execution_fingerprint_sha256,
        "execution_fingerprint_payload": dict(execution_fingerprint_payload),
        "condition_fingerprints": dict(condition_fingerprints),
        "condition_fingerprint_payloads": {
            condition: dict(condition_fingerprint_payloads[condition])
            for condition in core.CONDITIONS
        },
        "source_runtime_prompt_manifest": list(source_prompt_manifest),
        "source_runtime_prompt_manifest_sha256": core.canonical_json_sha256(
            list(source_prompt_manifest)
        ),
        "replay_runtime_prompt_manifest": list(replay_prompt_manifest),
        "replay_runtime_prompt_manifest_sha256": core.canonical_json_sha256(
            list(replay_prompt_manifest)
        ),
        "batch_plan": [canonical_treatment_batch_payload(spec) for spec in batch_specs],
        "batch_plan_sha256": core.canonical_json_sha256(
            [canonical_treatment_batch_payload(spec) for spec in batch_specs]
        ),
        "counts": {
            "source_prompt_calls": 627,
            "sentinel_calls": 21,
            "scored_calls": 1038,
            "batch_certificates": 268,
            "answers": 328,
        },
    }


def execute_replay(
    source: core.ReplaySource,
    output_dir: Path,
) -> dict:
    """Execute the exact pinned T4 replay; no retrieval or solo path is imported."""
    output_dir = _resolve_repo_path(output_dir, "replay output directory")
    meta_path = output_dir / "meta.json"
    if _leaf_present(meta_path):
        _require_regular_leaf(meta_path, "completed replay certificate")
        raise core.IntegrityError(
            f"refusing to overwrite completed replay: {meta_path}"
        )
    gpu = _gpu_metadata()
    _require_t4_preload(gpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    calls_final = output_dir / "calls.jsonl"
    calls_partial = output_dir / "calls.jsonl.partial"
    execution_session_id = uuid.uuid4().hex
    run_id = "frozen_upstream_passage_replay"
    store = JsonlStore(
        calls_partial,
        lock_path=output_dir / ".passage_replay.lock",
    ).acquire_lock(execution_session_id)
    model = tok = None
    try:
        if _leaf_present(meta_path):
            _require_regular_leaf(meta_path, "completed replay certificate")
            raise core.IntegrityError(
                f"refusing to overwrite completed replay: {meta_path}"
            )
        final_exists = _require_regular_leaf(
            calls_final,
            "final calls artifact",
            allow_absent=True,
        )
        partial_exists = _require_regular_leaf(
            calls_partial,
            "partial calls artifact",
            allow_absent=True,
        )
        if final_exists and partial_exists:
            if core.sha256_file(calls_final) != core.sha256_file(calls_partial):
                raise core.IntegrityError("final and partial calls artifacts differ")
            calls_path = calls_final
        elif final_exists:
            calls_path = calls_final
        else:
            calls_path = calls_partial
        output_is_already_final = calls_path == calls_final
        store.path = calls_path
        from src import agents, models

        (
            model,
            tok,
            runtime_identity,
            source_prompt_manifest,
            model_config_fingerprint,
            model_config_payload,
            load_seconds,
        ) = _load_replay_model(source, gpu)
        execution_fingerprint_sha256, execution_fingerprint_payload = (
            build_execution_fingerprint_payload(source, runtime_identity)
        )
        condition_fingerprints: dict[str, str] = {}
        condition_fingerprint_payloads: dict[str, dict] = {}
        for condition in core.CONDITIONS:
            ids = (
                source.question_ids
                if condition == core.SPANS_PLUS_PASSAGES
                else source.both_gold_ids
            )
            fingerprint, payload = condition_fingerprint(
                execution_fingerprint_sha256,
                condition,
                ids,
                core.condition_batches(source, condition),
            )
            condition_fingerprints[condition] = fingerprint
            condition_fingerprint_payloads[condition] = payload
        warnings = version_warnings(
            source.baseline_meta.get("library_versions") or {},
            runtime_identity["library_versions"],
        )
        for warning in warnings:
            print(
                "VERSION_WARNING "
                f"{warning['package']}: source={warning['source']!r} "
                f"replay={warning['replay']!r}; sentinel remains the hard gate"
            )

        sentinel_plan = _prepare_sentinel_plan(
            tok,
            source,
            execution_fingerprint_sha256,
        )
        buffered_sentinel, sentinel_report = _run_sentinel_buffer(
            model,
            tok,
            source,
            sentinel_plan,
            run_calls_fn=agents.run_calls,
            run_id=run_id,
            execution_session_id=execution_session_id,
            gpu=gpu,
            model_config_fingerprint=model_config_fingerprint,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
        )
        store.open_data(execution_session_id)
        existing_records = store.read_existing()
        existing_model_load, existing_tail = _split_existing_stream(existing_records)
        model_load_record = {
            "record_type": "model_load",
            "status": "success",
            "run_id": run_id,
            "execution_session_id": execution_session_id,
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "condition_fingerprints": dict(condition_fingerprints),
            "source_experiment_fingerprint": core.SOURCE_EXPERIMENT_FINGERPRINT,
            "source_artifact_sha256": dict(core.SOURCE_SHA256),
            "source_question_ids_sha256": core.FULL_IDS_SHA256,
            "source_both_gold_ids_sha256": core.BOTH_GOLD_IDS_SHA256,
            **_default_replay_identity(),
            "runtime_identity": runtime_identity,
            "model_config_fingerprint": model_config_fingerprint,
            "model_config_payload": model_config_payload,
            "load_seconds": round(load_seconds, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if existing_model_load is None:
            if output_is_already_final:
                raise core.IntegrityError("final calls artifact has no model_load row")
            store.write([model_load_record])
            store.durable_flush()
            accepted_model_load = model_load_record
        else:
            _validate_record_identity(
                existing_model_load,
                _default_replay_identity(),
                "existing model_load",
            )
            _validate_model_load_provenance(
                existing_model_load,
                execution_fingerprint_sha256=execution_fingerprint_sha256,
                condition_fingerprints=condition_fingerprints,
                source=source,
            )
            if existing_model_load.get("runtime_identity") != runtime_identity:
                raise core.IntegrityError("existing model_load runtime identity changed")
            accepted_model_load = existing_model_load
        cursor = ExistingStreamCursor(
            existing_tail,
            accepted_model_load,
            _default_replay_identity(),
        )
        _persist_or_verify_sentinel(
            cursor,
            store,
            buffered_sentinel,
            output_is_already_final=output_is_already_final,
        )

        batch_specs = [batch for batch, _prepared in sentinel_plan]
        replay_prompt_manifest = [
            item.prompt_manifest_record
            for _batch, prepared in sentinel_plan
            for item in prepared
        ]
        results: dict[str, core.ConditionResult] = {}
        for condition in core.CONDITIONS:
            print(f"executing {condition}", flush=True)
            results[condition] = _execute_condition(
                condition,
                condition_fingerprints[condition],
                source,
                cursor,
                store,
                model,
                tok,
                run_calls_fn=agents.run_calls,
                run_id=run_id,
                execution_session_id=execution_session_id,
                gpu=gpu,
                model_config_fingerprint=model_config_fingerprint,
                execution_fingerprint_sha256=execution_fingerprint_sha256,
                output_is_already_final=output_is_already_final,
                batch_specs=batch_specs,
                prompt_manifest=replay_prompt_manifest,
            )
        cursor.assert_exhausted()
        if len(batch_specs) != 268 or sum(len(spec.members) for spec in batch_specs) != 1059:
            raise core.IntegrityError("completed replay batch plan changed cardinality")
        summary = core.score_replay(source, results)
        answers = []
        for condition in core.CONDITIONS:
            result = results[condition]
            fingerprint = condition_fingerprints[condition]
            answers.extend(
                {
                    **result.answer_records[qid],
                    "condition_fingerprint_sha256": fingerprint,
                }
                for qid in result.question_ids
            )
        store.durable_flush()
        calls = store.read_existing()
        counts = validate_complete_streams(
            calls,
            answers,
            batch_specs,
            identity=_default_replay_identity(),
            source=source,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
            condition_fingerprints=condition_fingerprints,
        )
        manifest = _final_replay_manifest(
            source,
            execution_fingerprint_sha256,
            execution_fingerprint_payload,
            condition_fingerprints,
            condition_fingerprint_payloads,
            source_prompt_manifest,
            replay_prompt_manifest,
            batch_specs,
        )
        replay_fingerprint_sha256 = core.canonical_json_sha256(manifest)
        _write_json_artifact(output_dir, "manifest.json", manifest)
        _write_jsonl_artifact(output_dir, "answers.jsonl", answers)
        _write_json_artifact(output_dir, "summary.json", summary)
        store.durable_flush()
        store.close()
        validation_payload = PublicationValidationPayload(
            source=source,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
            replay_fingerprint_sha256=replay_fingerprint_sha256,
            condition_fingerprints=condition_fingerprints,
            batch_specs=tuple(batch_specs),
            identity=_default_replay_identity(),
            expected_manifest=manifest,
        )
        meta_payload = {
            "schema": "frozen-upstream-passage-replay-meta-v1",
            "report_label": core.REPORT_LABEL,
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "replay_fingerprint_sha256": replay_fingerprint_sha256,
            "condition_fingerprints": condition_fingerprints,
            "source_artifact_sha256": dict(core.SOURCE_SHA256),
            "source_experiment_fingerprint": core.SOURCE_EXPERIMENT_FINGERPRINT,
            "runtime_identity": runtime_identity,
            "version_warnings": warnings,
            "sentinel": sentinel_report,
            "counts": counts,
            "extractor_answer_survival_both_gold": (
                summary["extractor_answer_survival_both_gold"]
            ),
            "extractor_normalizer_near_match_diagnostic": (
                summary["extractor_normalizer_near_match_diagnostic"]
            ),
        }
        meta = publish_meta_last(
            output_dir,
            meta_payload,
            validation_payload=validation_payload,
        )
        print(
            f"REPLAY_COMPLETE replay_fingerprint_sha256={replay_fingerprint_sha256}",
            flush=True,
        )
        return meta
    finally:
        store.close()
        if model is not None or tok is not None:
            model = tok = None
            try:
                from src import models

                models.unload()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-upstream passage access replay for the Gate C trace"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("evidence/gate_c_1.7b"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/passage_plus_spans_replay"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_dir = _resolve_repo_path(args.source_dir, "source directory")
    source = core.load_source_bundle(source_dir)
    audit = build_cpu_audit(source)
    print_cpu_audit(audit)
    if args.audit_only:
        return 0
    execute_replay(source, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
