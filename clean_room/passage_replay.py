"""Frozen-upstream passage replay CLI and certified GPU runtime.

The CPU audit reads only committed records.  GPU helpers are imported lazily so
artifact validation and manifest construction never load model weights or
initialize retrieval.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


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
    "quantized_4bit_params": 1_409_286_144,
    "quantized_8bit_params": 0,
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


def validate_execution_identity(identity: dict) -> dict:
    required = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "precision": PRECISION,
        "batch_size": BATCH_SIZE,
        "enable_thinking": False,
    }
    for key, expected in required.items():
        if identity.get(key) != expected:
            raise core.IntegrityError(
                f"execution identity {key} mismatch: "
                f"expected {expected!r}, got {identity.get(key)!r}"
            )
    census = identity.get("census")
    if not isinstance(census, dict):
        raise core.IntegrityError("execution identity has no quantization census")
    for key, expected in EXPECTED_CENSUS.items():
        if census.get(key) != expected:
            raise core.IntegrityError(
                f"quantization census {key} mismatch: "
                f"expected {expected}, got {census.get(key)!r}"
            )
    gpu = identity.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("gpu_available") is not True:
        raise core.IntegrityError("Tesla T4 GPU is unavailable")
    if gpu.get("gpu_name") != EXPECTED_GPU_NAME:
        raise core.IntegrityError(
            f"GPU class mismatch: expected {EXPECTED_GPU_NAME!r}, "
            f"got {gpu.get('gpu_name')!r}"
        )
    if gpu.get("gpu_compute_capability") != EXPECTED_GPU_COMPUTE_CAPABILITY:
        raise core.IntegrityError(
            "GPU compute capability mismatch: expected "
            f"{EXPECTED_GPU_COMPUTE_CAPABILITY}, "
            f"got {gpu.get('gpu_compute_capability')!r}"
        )
    return identity


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
    validate_execution_identity(runtime_identity)
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
        "runtime_identity": runtime_identity,
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
