"""CPU-safe core for the frozen-upstream passage replay experiment.

The replay is intentionally downstream of retrieval.  This module projects the
committed Gate-C artifacts into a gold-free prompt trace and a separate scoring
view, then fails closed if any source identity changes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src import prompts


BASELINE_JSONL = "baseline_qwen3-1.7b_n200_seed20260806_pilot.jsonl"
BASELINE_META = "baseline_qwen3-1.7b_n200_seed20260806_pilot.meta.json"
SINGLE_JSONL = "single_fp16_qwen3-1.7b_n200_seed20260806_pilot.jsonl"
SINGLE_META = "single_fp16_qwen3-1.7b_n200_seed20260806_pilot.meta.json"

SOURCE_SHA256 = {
    BASELINE_JSONL: "203795707234dcf94eb50dcd861483e3364accd3034d13806498f6ba36bfa163",
    BASELINE_META: "3dc51ea687cc6c4973701b61b1dfd5636e3f68c3e27eff258d263bada8cc4863",
    SINGLE_JSONL: "79d5a814f7e3f54ae5d39efc1f2d065ac14039e2616de26695a55df27fa150d5",
    SINGLE_META: "49543b660d02f2c472c541b147f0d227b4c7faa11a37904cb8e7622637918a85",
}
SOURCE_EXPERIMENT_FINGERPRINT = (
    "68a765d4bdaf49bd760bdb7866c09e25e65bae9ddbb5f2dc30b3a856968baa2a"
)
FULL_IDS_SHA256 = "f8c3f16458340cb0bc74aa827e3b51528ba351963a46dba456ac4e68ad20f7d7"
BOTH_GOLD_IDS_SHA256 = (
    "cf7a9dbf8bfc48dfc42c40e5caaae68ad47e9ba26ef5a6536a27f83fada3508a"
)
SOURCE_COMMIT = "56cd63f789b45eb4cc983f82684837a42d168bee"
MODEL_ID = "Qwen/Qwen3-1.7B"
PRECISION = "4bit"
BATCH_SIZE = 4


class IntegrityError(RuntimeError):
    """The committed source or frozen replay contract changed."""


@dataclass(frozen=True)
class FrozenQuestion:
    question_id: str
    question: str
    plan: tuple[str, ...]
    stop_reason: str


@dataclass(frozen=True)
class ScoringQuestion:
    gold_answer: str
    stratum: str
    both_gold: bool


@dataclass(frozen=True)
class ReplaySource:
    baseline_records: tuple[dict, ...]
    baseline_meta: dict
    single_records: tuple[dict, ...]
    single_meta: dict
    baseline_index: dict
    single_index: dict
    question_ids: tuple[str, ...]
    question_ids_sha256: str
    both_gold_ids: tuple[str, ...]
    baseline_answers: dict[str, dict]
    single_answers: dict[str, dict]
    questions: dict[str, FrozenQuestion]
    scoring: dict[str, ScoringQuestion]


@dataclass(frozen=True)
class AuditReport:
    qa_stage_counts: dict[str, int]
    qa_prompt_hash_matches: int
    summary_prompt_hash_matches: int
    question_answering_calls: int
    aggregate_calls: int
    extractor_joins: int


@dataclass(frozen=True, order=True)
class CallKey:
    question_id: str
    stage: str
    call_index: int


@dataclass(frozen=True, order=True)
class ConditionCallKey:
    condition: str
    question_id: str
    stage: str
    call_index: int


@dataclass(frozen=True)
class PassageJoin:
    titles: tuple[str, ...]
    sentence_lists: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FrozenBatch:
    stage: str
    ordinal: int
    members: tuple[CallKey, ...]
    source_batch_id: str | None
    batch_id: str
    canonical_sha256: str
    source_batch_ids: tuple[str, ...]


QA_STAGES = ("qa", "qa_step2", "qa_step3", "qa_step4", "qa_step5")
PASSAGE_HEADER = "Retrieved passages for the current step:"
SPANS_PLUS_PASSAGES = "spans_plus_passages"
PASSAGES_ONLY = "passages_only"
CONDITIONS = (SPANS_PLUS_PASSAGES, PASSAGES_ONLY)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_NO_ANSWER_SENTINELS = frozenset(
    {
        "unknown",
        "no answer",
        "no answer found",
        "no relevant information",
        "no relevant information found",
        "not enough information",
        "insufficient information",
        "cannot determine",
        "unable to determine",
        "i don't know",
        "n/a",
        "none",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(question_ids: tuple[str, ...] | list[str]) -> str:
    payload = "".join(f"{qid}\n" for qid in question_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rendered_prompt_sha256(messages: list[dict]) -> str:
    """Match ``src.agents.rendered_prompt_sha256`` without importing models."""
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_str(mapping: dict, key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{label} has invalid {key}: {value!r}")
    return value


def _require_int(mapping: dict, key: str, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{label} has invalid {key}: {value!r}")
    return value


def _index_agent_calls(records: tuple[dict, ...], label: str) -> dict:
    index: dict[tuple[str, str, int], dict] = {}
    for record in records:
        if record.get("record_type") != "agent_call":
            continue
        qid = _require_str(record, "question_id", label)
        stage = _require_str(record, "stage", label)
        call_index = _require_int(record, "call_index", label)
        key = (qid, stage, call_index)
        if key in index:
            raise IntegrityError(f"duplicate {label} source call {key!r}")
        index[key] = record
    return index


def _render_prior_steps(prior_steps: list[dict] | None) -> str:
    if not prior_steps:
        return "(none; this is the first plan step)"
    lines: list[str] = []
    for item in prior_steps:
        grounded = item.get("answer_grounded")
        answer = item.get("answer") or "(no answer)"
        lines.append(f"Step {item['step_number']} goal: {item['sub_question']}")
        lines.append(f"Step {item['step_number']} answer: {answer}")
        if grounded is not None:
            lines.append(
                f"Step {item['step_number']} grounding: "
                f"{'evidence-grounded' if grounded else 'unsupported guess'}"
            )
        if item.get("success") is not None:
            lines.append(
                f"Step {item['step_number']} success/rating: "
                f"{item.get('success')} / {item.get('rating')}"
            )
    return "\n".join(lines)


def _build_qa_fields(
    question: str,
    evidence_blocks: list[tuple[str, list[str]]],
    *,
    sub_question: str,
    step_number: int,
    plan_steps: int,
) -> dict:
    lines: list[str] = []
    displayed = 0
    for label, spans in evidence_blocks:
        if not spans:
            continue
        displayed += 1
        lines.append(f"{displayed}. {label}")
        lines.extend(f"   - {span}" for span in spans)
    return {
        "question": question,
        "sub_question": sub_question,
        "step_number": step_number,
        "plan_steps": plan_steps,
        "evidence": "\n".join(lines) if lines else "(no evidence collected)",
    }


def _build_plan_summary_fields(
    question: str,
    plan: tuple[str, ...] | list[str],
    prior_steps: list[dict],
    stop_reason: str,
) -> dict:
    return {
        "question": question,
        "full_plan": "\n".join(
            f"{index}. {item}" for index, item in enumerate(plan, start=1)
        ),
        "prior_state": _render_prior_steps(prior_steps),
        "stop_reason": stop_reason,
    }


def source_qa_records(source: ReplaySource) -> tuple[dict, ...]:
    return tuple(
        record
        for record in source.baseline_records
        if record.get("record_type") == "agent_call" and record.get("prompt_role") == "qa"
    )


def source_summary_records(source: ReplaySource) -> tuple[dict, ...]:
    return tuple(
        record
        for record in source.baseline_records
        if record.get("record_type") == "agent_call"
        and record.get("prompt_role") == "plan_summary"
    )


def reconstruct_source_qa_fields(source: ReplaySource, qa: dict) -> dict:
    qid = _require_str(qa, "question_id", "QA call")
    if qid not in source.questions:
        raise IntegrityError(f"QA call has unexpected question ID {qid}")
    call_index = _require_int(qa, "call_index", f"QA call {qid}")
    expected_stage = prompts.stage_for("qa", call_index)
    _require_equal(qa.get("stage"), expected_stage, f"QA stage for {qid}")
    _require_equal(qa.get("prompt_role"), "qa", f"QA prompt role for {qid}")
    consumer_input = qa.get("consumer_input")
    if not isinstance(consumer_input, dict):
        raise IntegrityError(f"QA call {qid}/{expected_stage} has invalid consumer input")
    step_definition = consumer_input.get("step_definition")
    if not isinstance(step_definition, dict):
        raise IntegrityError(f"QA call {qid}/{expected_stage} has invalid step definition")
    task = _require_str(step_definition, "task", f"QA call {qid}/{expected_stage}")
    evidence_blocks = consumer_input.get("evidence_blocks")
    if not isinstance(evidence_blocks, list):
        raise IntegrityError(f"QA call {qid}/{expected_stage} has invalid evidence blocks")
    task_type = consumer_input.get("task_type")
    if task_type not in {"question-answering", "aggregate"}:
        raise IntegrityError(f"QA call {qid}/{expected_stage} has invalid task type {task_type!r}")

    blocks: list[tuple[str, list[str]]] = []
    for block_number, block in enumerate(evidence_blocks):
        if not isinstance(block, dict):
            raise IntegrityError(
                f"QA call {qid}/{expected_stage} block {block_number} is not an object"
            )
        spans = block.get("prompt_spans")
        if not isinstance(spans, list) or not all(isinstance(span, str) for span in spans):
            raise IntegrityError(
                f"QA call {qid}/{expected_stage} block {block_number} has invalid prompt spans"
            )
        if not spans:
            continue
        if task_type == "aggregate":
            label = _require_str(
                block,
                "sub_question",
                f"QA call {qid}/{expected_stage} aggregate block {block_number}",
            )
        else:
            rank = _require_int(
                block,
                "document_rank",
                f"QA call {qid}/{expected_stage} block {block_number}",
            )
            title = _require_str(
                block,
                "document_title",
                f"QA call {qid}/{expected_stage} block {block_number}",
            )
            label = f"Document {rank + 1}: {title}"
        blocks.append((label, list(spans)))

    frozen = source.questions[qid]
    fields = _build_qa_fields(
        frozen.question,
        blocks,
        sub_question=task,
        step_number=call_index + 1,
        plan_steps=len(frozen.plan),
    )
    messages = prompts.build_messages("qa", **fields)
    if rendered_prompt_sha256(messages) != qa.get("rendered_prompt_sha256"):
        raise IntegrityError(f"source QA message hash mismatch for {qid}/{expected_stage}")
    return fields


def reconstruct_source_summary_fields(source: ReplaySource, summary: dict) -> dict:
    qid = _require_str(summary, "question_id", "summary call")
    if qid not in source.questions:
        raise IntegrityError(f"summary call has unexpected question ID {qid}")
    _require_equal(summary.get("stage"), "plan_summary", f"summary stage for {qid}")
    _require_equal(summary.get("call_index"), 0, f"summary call index for {qid}")
    consumer_input = summary.get("consumer_input")
    if not isinstance(consumer_input, dict):
        raise IntegrityError(f"summary call {qid} has invalid consumer input")
    frozen = source.questions[qid]
    _require_equal(tuple(consumer_input.get("plan") or ()), frozen.plan, f"summary plan for {qid}")
    _require_equal(consumer_input.get("stop_reason"), frozen.stop_reason, f"summary stop for {qid}")
    history = consumer_input.get("completed_steps")
    if not isinstance(history, list):
        raise IntegrityError(f"summary call {qid} has invalid completed history")
    fields = _build_plan_summary_fields(
        frozen.question,
        frozen.plan,
        history,
        frozen.stop_reason,
    )
    messages = prompts.build_messages("plan_summary", **fields)
    if rendered_prompt_sha256(messages) != summary.get("rendered_prompt_sha256"):
        raise IntegrityError(f"source summary message hash mismatch for {qid}")
    return fields


def join_recorded_passages(source: ReplaySource, qa: dict) -> PassageJoin:
    qid = _require_str(qa, "question_id", "QA call")
    call_index = _require_int(qa, "call_index", f"QA call {qid}")
    expected_stage = prompts.stage_for("qa", call_index)
    _require_equal(qa.get("stage"), expected_stage, f"QA stage for {qid}")
    consumer_input = qa.get("consumer_input")
    if not isinstance(consumer_input, dict):
        raise IntegrityError(f"QA call {qid}/{expected_stage} has invalid consumer input")
    _require_equal(
        consumer_input.get("task_type"),
        "question-answering",
        f"passage route for {qid}/{expected_stage}",
    )
    step_definition = consumer_input.get("step_definition")
    retrieval = consumer_input.get("retrieval")
    blocks = consumer_input.get("evidence_blocks")
    if not isinstance(step_definition, dict) or step_definition.get("type") != "question-answering":
        raise IntegrityError(f"invalid QA step definition for {qid}/{expected_stage}")
    if not isinstance(retrieval, dict):
        raise IntegrityError(f"invalid QA retrieval event for {qid}/{expected_stage}")
    titles = retrieval.get("titles")
    if not isinstance(titles, list) or len(titles) != 10 or not all(
        isinstance(title, str) and title for title in titles
    ):
        raise IntegrityError(f"QA retrieval titles are not an ordered top-10 for {qid}/{expected_stage}")
    if not isinstance(blocks, list) or len(blocks) != 10:
        raise IntegrityError(f"QA evidence blocks are not an ordered top-10 for {qid}/{expected_stage}")

    ext_stage = prompts.stage_for("extractor", call_index)
    expected_keys = {(qid, ext_stage, rank) for rank in range(10)}
    actual_keys = {
        key
        for key in source.baseline_index
        if key[0] == qid and key[1] == ext_stage
    }
    if actual_keys != expected_keys:
        raise IntegrityError(
            f"Extractor ranks for {qid}/{ext_stage} are not exactly 0..9: "
            f"{sorted(key[2] for key in actual_keys)!r}"
        )

    sentence_lists: list[tuple[str, ...]] = []
    for rank in range(10):
        block = blocks[rank]
        extractor = source.baseline_index[(qid, ext_stage, rank)]
        ext_input = extractor.get("consumer_input")
        if not isinstance(block, dict) or not isinstance(ext_input, dict):
            raise IntegrityError(f"invalid passage join object for {qid}/{ext_stage}/{rank}")
        _require_equal(extractor.get("call_index"), rank, f"Extractor call index {qid}/{ext_stage}")
        _require_equal(ext_input.get("document_rank"), rank, f"Extractor document rank {qid}/{ext_stage}")
        _require_equal(block.get("document_rank"), rank, f"QA document rank {qid}/{expected_stage}")
        _require_equal(ext_input.get("document_title"), titles[rank], f"Extractor title {qid}/{ext_stage}/{rank}")
        _require_equal(block.get("document_title"), titles[rank], f"QA title {qid}/{expected_stage}/{rank}")
        _require_equal(ext_input.get("step_definition"), step_definition, f"Extractor task {qid}/{ext_stage}/{rank}")
        _require_equal(ext_input.get("retrieval"), retrieval, f"Extractor retrieval {qid}/{ext_stage}/{rank}")
        for span_key in ("spans", "prompt_spans"):
            spans = block.get(span_key)
            if not isinstance(spans, list) or not all(isinstance(span, str) for span in spans):
                raise IntegrityError(f"invalid QA {span_key} for {qid}/{expected_stage}/{rank}")
        _require_equal(
            block.get("included_in_prompt"),
            bool(block.get("prompt_spans")),
            f"QA prompt inclusion for {qid}/{expected_stage}/{rank}",
        )
        sentences = ext_input.get("document_sentences")
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise IntegrityError(f"invalid source sentences for {qid}/{ext_stage}/{rank}")
        sentence_lists.append(tuple(sentences))
    return PassageJoin(tuple(titles), tuple(sentence_lists))


def render_treated_evidence(original: str, join: PassageJoin, condition: str) -> str:
    passages = prompts.format_paragraphs(
        list(join.titles),
        [list(sentences) for sentences in join.sentence_lists],
    )
    block = f"{PASSAGE_HEADER}\n{passages}"
    if condition == SPANS_PLUS_PASSAGES:
        return f"{original}\n\n{block}"
    if condition == PASSAGES_ONLY:
        return block
    raise ValueError(f"unknown replay condition {condition!r}")


def _source_batch_sha256(members: tuple[CallKey, ...]) -> str:
    payload = "".join(
        f"{member.question_id}\t{member.call_index}\n" for member in members
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _condition_batch_sha256(
    condition: str,
    stage: str,
    ordinal: int,
    members: tuple[CallKey, ...],
) -> str:
    return canonical_json_sha256(
        {
            "condition": condition,
            "stage": stage,
            "ordinal": ordinal,
            "members": [
                {
                    "question_id": member.question_id,
                    "stage": member.stage,
                    "call_index": member.call_index,
                }
                for member in members
            ],
        }
    )


def source_scored_batches(source: ReplaySource, stage: str) -> tuple[FrozenBatch, ...]:
    if stage not in (*QA_STAGES, "plan_summary"):
        raise ValueError(f"unsupported replay stage {stage!r}")
    records = [
        record
        for record in source.baseline_records
        if record.get("record_type") == "batch"
        and record.get("phase") == "scored"
        and record.get("stage") == stage
    ]
    ordinals = [_require_int(record, "batch_ordinal", f"source {stage} batch") for record in records]
    if len(ordinals) != len(set(ordinals)) or sorted(ordinals) != list(range(len(records))):
        raise IntegrityError(f"source {stage} batch ordinals are not unique and contiguous")
    records.sort(key=lambda record: record["batch_ordinal"])

    expected_keys = {
        CallKey(record["question_id"], stage, int(record["call_index"]))
        for record in source.baseline_index.values()
        if record.get("stage") == stage
        and record.get("record_type") == "agent_call"
        and record.get("prompt_role") in {"qa", "plan_summary"}
    }
    seen: set[CallKey] = set()
    batches: list[FrozenBatch] = []
    for record in records:
        ordinal = int(record["batch_ordinal"])
        batch_id = f"{stage}:{ordinal:06d}"
        _require_equal(record.get("batch_id"), batch_id, f"source {stage} batch ID")
        _require_equal(record.get("batch_size_requested"), BATCH_SIZE, f"source {stage} batch size")
        _require_equal(record.get("oom"), False, f"source {stage} batch OOM")
        _require_equal(record.get("model_id"), MODEL_ID, f"source {stage} batch model")
        _require_equal(record.get("precision"), PRECISION, f"source {stage} batch precision")
        _require_equal(record.get("model_revision"), "TBD", f"source {stage} model revision")
        _require_equal(record.get("tokenizer_revision"), "TBD", f"source {stage} tokenizer revision")
        _require_equal(
            record.get("experiment_fingerprint"),
            SOURCE_EXPERIMENT_FINGERPRINT,
            f"source {stage} batch experiment",
        )
        _require_equal(
            record.get("question_manifest_sha256"),
            FULL_IDS_SHA256,
            f"source {stage} batch manifest",
        )
        raw_members = record.get("members")
        if not isinstance(raw_members, list) or not 1 <= len(raw_members) <= BATCH_SIZE:
            raise IntegrityError(f"source {stage} batch {ordinal} has invalid members")
        members = tuple(
            CallKey(
                _require_str(member, "question_id", f"source {stage} batch {ordinal}"),
                stage,
                _require_int(member, "call_index", f"source {stage} batch {ordinal}"),
            )
            for member in raw_members
            if isinstance(member, dict)
        )
        if len(members) != len(raw_members):
            raise IntegrityError(f"source {stage} batch {ordinal} has non-object member")
        _require_equal(
            record.get("batch_size_actual"),
            len(members),
            f"source {stage} batch actual size",
        )
        _require_equal(
            record.get("canonical_batch_sha256"),
            _source_batch_sha256(members),
            f"source {stage} canonical batch SHA-256",
        )
        if seen.intersection(members):
            raise IntegrityError(f"source {stage} batch membership is duplicated")
        for member_index, member in enumerate(members):
            call = source.baseline_index.get(
                (member.question_id, member.stage, member.call_index)
            )
            if call is None:
                raise IntegrityError(f"source {stage} batch references missing call {member!r}")
            _require_equal(call.get("batch_id"), batch_id, f"source call batch for {member!r}")
            _require_equal(
                call.get("batch_member_index"),
                member_index,
                f"source call member index for {member!r}",
            )
            _require_equal(call.get("model_id"), MODEL_ID, f"source call model for {member!r}")
            _require_equal(call.get("precision"), PRECISION, f"source call precision for {member!r}")
            _require_equal(
                call.get("experiment_fingerprint"),
                SOURCE_EXPERIMENT_FINGERPRINT,
                f"source call experiment for {member!r}",
            )
        seen.update(members)
        batches.append(
            FrozenBatch(
                stage=stage,
                ordinal=ordinal,
                members=members,
                source_batch_id=batch_id,
                batch_id=batch_id,
                canonical_sha256=_source_batch_sha256(members),
                source_batch_ids=(batch_id,),
            )
        )
    if seen != expected_keys:
        raise IntegrityError(
            f"source {stage} certificates do not exactly cover calls: "
            f"missing={len(expected_keys - seen)}, extra={len(seen - expected_keys)}"
        )
    return tuple(batches)


def condition_batches(
    source: ReplaySource,
    condition: str,
    stage: str | None = None,
) -> tuple[FrozenBatch, ...]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    stages = (stage,) if stage is not None else (*QA_STAGES, "plan_summary")
    subset = set(source.both_gold_ids)
    output: list[FrozenBatch] = []
    for concrete_stage in stages:
        source_batches = source_scored_batches(source, concrete_stage)
        if condition == SPANS_PLUS_PASSAGES:
            groups = [
                (batch.members, batch.source_batch_ids)
                for batch in source_batches
            ]
        else:
            retained: list[tuple[CallKey, str]] = []
            for batch in source_batches:
                for member in batch.members:
                    if member.question_id not in subset:
                        continue
                    if concrete_stage in QA_STAGES:
                        qa = source.baseline_index[
                            (member.question_id, member.stage, member.call_index)
                        ]
                        if (qa.get("consumer_input") or {}).get("task_type") != "question-answering":
                            raise IntegrityError(
                                "passages_only cohort unexpectedly contains aggregate QA"
                            )
                    retained.append((member, batch.batch_id))
            groups = []
            for start in range(0, len(retained), BATCH_SIZE):
                chunk = retained[start : start + BATCH_SIZE]
                source_ids = tuple(dict.fromkeys(source_id for _, source_id in chunk))
                groups.append((tuple(member for member, _ in chunk), source_ids))

        for ordinal, (members, source_ids) in enumerate(groups):
            batch_id = f"{condition}:{concrete_stage}:{ordinal:06d}"
            output.append(
                FrozenBatch(
                    stage=concrete_stage,
                    ordinal=ordinal,
                    members=tuple(members),
                    source_batch_id=(source_ids[0] if len(source_ids) == 1 else None),
                    batch_id=batch_id,
                    canonical_sha256=_condition_batch_sha256(
                        condition,
                        concrete_stage,
                        ordinal,
                        tuple(members),
                    ),
                    source_batch_ids=tuple(source_ids),
                )
            )
    return tuple(output)


def condition_call_keys(
    source: ReplaySource,
    condition: str,
) -> tuple[ConditionCallKey, ...]:
    keys = tuple(
        ConditionCallKey(
            condition,
            member.question_id,
            member.stage,
            member.call_index,
        )
        for batch in condition_batches(source, condition)
        for member in batch.members
    )
    if len(keys) != len(set(keys)):
        raise IntegrityError(f"{condition} call manifest contains duplicate keys")
    expected = 627 if condition == SPANS_PLUS_PASSAGES else 411
    if len(keys) != expected:
        raise IntegrityError(
            f"{condition} call manifest has {len(keys)} calls, expected {expected}"
        )
    return keys


def effective_payload(record: dict | None) -> tuple[dict, str]:
    if record and isinstance(record.get("parsed"), dict):
        return dict(record["parsed"]), "parsed"
    if record and isinstance(record.get("salvaged"), dict):
        return dict(record["salvaged"]), "salvaged"
    return {}, "fallback"


def _normalized_token_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(_TOKEN.findall(normalized))


def answer_is_grounded(answer: str, evidence_texts: list[str]) -> bool:
    answer_norm = _normalized_token_phrase(answer)
    if not answer_norm:
        return False
    needle = f" {answer_norm} "
    return any(
        needle in f" {_normalized_token_phrase(text)} "
        for text in evidence_texts
        if isinstance(text, str) and text.strip()
    )


def _source_summary(source: ReplaySource, qid: str) -> dict:
    summary = source.baseline_index.get((qid, "plan_summary", 0))
    if summary is None:
        raise IntegrityError(f"missing source summary for {qid}")
    return summary


def _treated_record_for(
    treated_qa_index: Mapping,
    condition: str,
    qid: str,
    stage: str,
    call_index: int,
) -> dict:
    record = treated_qa_index.get((condition, qid, stage, call_index))
    if record is None:
        record = treated_qa_index.get(ConditionCallKey(condition, qid, stage, call_index))
    if not isinstance(record, dict):
        raise IntegrityError(
            f"missing treated QA record for {(condition, qid, stage, call_index)!r}"
        )
    _require_equal(record.get("question_id"), qid, "treated QA question ID")
    _require_equal(record.get("stage"), stage, "treated QA stage")
    _require_equal(record.get("call_index"), call_index, "treated QA call index")
    return record


def _visible_texts_for_step(
    source: ReplaySource,
    source_qa: dict,
    condition: str,
    prior_history: list[dict],
) -> list[str]:
    consumer_input = source_qa.get("consumer_input") or {}
    if consumer_input.get("task_type") == "aggregate":
        return [
            f"Prior step answer: {item['answer']}"
            for item in prior_history
            if item.get("answer_grounded") is True and item.get("answer")
        ]
    join = join_recorded_passages(source, source_qa)
    visible = [sentence for sentences in join.sentence_lists for sentence in sentences]
    if condition == SPANS_PLUS_PASSAGES:
        visible.extend(
            span
            for block in consumer_input.get("evidence_blocks") or []
            for span in block.get("prompt_spans") or []
        )
    elif condition != PASSAGES_ONLY:
        raise ValueError(f"unknown replay condition {condition!r}")
    return visible


def rebuild_treated_history(
    source: ReplaySource,
    qid: str,
    treated_qa_index: Mapping,
    condition: str,
    *,
    before_step: int | None = None,
) -> list[dict]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    if qid not in source.questions:
        raise IntegrityError(f"unknown treated-history question ID {qid}")
    source_summary = _source_summary(source, qid)
    source_history = (source_summary.get("consumer_input") or {}).get("completed_steps")
    if not isinstance(source_history, list):
        raise IntegrityError(f"invalid source history for {qid}")
    limit = len(source_history) if before_step is None else min(before_step, len(source_history))
    if limit < 0:
        raise ValueError("before_step must be nonnegative")

    treated_history: list[dict] = []
    for step_index, source_item in enumerate(source_history[:limit]):
        stage = prompts.stage_for("qa", step_index)
        source_qa = source.baseline_index.get((qid, stage, step_index))
        if source_qa is None:
            raise IntegrityError(f"missing source QA call for {qid}/{stage}/{step_index}")
        treated_record = _treated_record_for(
            treated_qa_index,
            condition,
            qid,
            stage,
            step_index,
        )
        payload, payload_source = effective_payload(treated_record)
        answer_value = payload.get("answer", "")
        answer = answer_value if isinstance(answer_value, str) else ""
        visible_texts = _visible_texts_for_step(
            source,
            source_qa,
            condition,
            treated_history,
        )
        grounded = answer_is_grounded(answer, visible_texts)
        treated_item = copy.deepcopy(source_item)
        treated_item.update(
            {
                "answer": answer,
                "answer_grounded": grounded,
                "answer_grounding": "consumed_evidence" if grounded else "unsupported",
                "success": payload.get("success"),
                "rating": payload.get("rating"),
                "qa_source": payload_source,
            }
        )
        treated_history.append(treated_item)
    return treated_history


def _source_message_hash(source_record: dict, prompt_role: str, fields: dict) -> str:
    messages = prompts.build_messages(prompt_role, **fields)
    message_hash = rendered_prompt_sha256(messages)
    _require_equal(
        message_hash,
        source_record.get("rendered_prompt_sha256"),
        f"source {prompt_role} parent message hash",
    )
    return message_hash


def build_treated_qa_call(
    source: ReplaySource,
    qa: dict,
    condition: str,
    treated_qa_index: Mapping,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    qid = _require_str(qa, "question_id", "source QA parent")
    call_index = _require_int(qa, "call_index", f"source QA parent {qid}")
    stage = prompts.stage_for("qa", call_index)
    _require_equal(qa.get("stage"), stage, f"source QA parent stage for {qid}")
    source_fields = reconstruct_source_qa_fields(source, qa)
    fields = copy.deepcopy(source_fields)
    consumer_input = copy.deepcopy(qa.get("consumer_input") or {})
    if consumer_input.get("task_type") == "question-answering":
        join = join_recorded_passages(source, qa)
        fields["evidence"] = render_treated_evidence(fields["evidence"], join, condition)
        payload_source = qa.get("consumer_payload_source")
    elif consumer_input.get("task_type") == "aggregate":
        history = rebuild_treated_history(
            source,
            qid,
            treated_qa_index,
            condition,
            before_step=call_index,
        )
        blocks: list[tuple[str, list[str]]] = []
        consumed: list[dict] = []
        for item in history:
            prompt_spans = (
                [f"Prior step answer: {item['answer']}"]
                if item.get("answer_grounded") is True and item.get("answer")
                else []
            )
            if prompt_spans:
                blocks.append((item["sub_question"], prompt_spans))
            consumed.append(
                {
                    "sub_question": item["sub_question"],
                    "answer": item["answer"],
                    "answer_grounded": item["answer_grounded"],
                    "spans": [],
                    "prompt_spans": prompt_spans,
                    "included_in_prompt": bool(prompt_spans),
                    "document_title": None,
                    "document_rank": None,
                    "consumer_payload_source": item["qa_source"],
                }
            )
        frozen = source.questions[qid]
        fields = _build_qa_fields(
            frozen.question,
            blocks,
            sub_question=consumer_input["step_definition"]["task"],
            step_number=call_index + 1,
            plan_steps=len(frozen.plan),
        )
        consumer_input["evidence_document_count"] = len(consumed)
        consumer_input["evidence_prompt_block_count"] = len(blocks)
        consumer_input["evidence_blocks"] = consumed
        payload_source = "treated_history"
    else:
        raise IntegrityError(f"unknown frozen task type for {qid}/{stage}")

    source_message_hash = _source_message_hash(qa, "qa", source_fields)
    treatment_message_hash = rendered_prompt_sha256(prompts.build_messages("qa", **fields))
    parent_key = [qid, stage, call_index]
    return {
        "question_id": qid,
        "stage": stage,
        "call_index": call_index,
        "fields": fields,
        "condition": condition,
        "consumer_payload_source": payload_source,
        "consumer_input": consumer_input,
        "source_parent_key": parent_key,
        "source_parent_record_sha256": canonical_json_sha256(qa),
        "source_message_sha256": source_message_hash,
        "treatment_message_sha256": treatment_message_hash,
    }


def build_treated_summary_call(
    source: ReplaySource,
    qid: str,
    history: list[dict],
    condition: str,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    frozen = source.questions.get(qid)
    if frozen is None:
        raise IntegrityError(f"unknown summary question ID {qid}")
    source_summary = _source_summary(source, qid)
    source_fields = reconstruct_source_summary_fields(source, source_summary)
    fields = _build_plan_summary_fields(
        frozen.question,
        frozen.plan,
        history,
        frozen.stop_reason,
    )
    source_message_hash = _source_message_hash(source_summary, "plan_summary", source_fields)
    treatment_message_hash = rendered_prompt_sha256(
        prompts.build_messages("plan_summary", **fields)
    )
    return {
        "question_id": qid,
        "stage": "plan_summary",
        "call_index": 0,
        "fields": fields,
        "condition": condition,
        "consumer_payload_source": "treated_history",
        "consumer_input": {
            "plan": list(frozen.plan),
            "completed_steps": copy.deepcopy(history),
            "stop_reason": frozen.stop_reason,
        },
        "source_parent_key": [qid, "plan_summary", 0],
        "source_parent_record_sha256": canonical_json_sha256(source_summary),
        "source_message_sha256": source_message_hash,
        "treatment_message_sha256": treatment_message_hash,
    }


def usable_short_answer(value: object) -> bool:
    if not isinstance(value, str):
        return False
    answer = " ".join(value.split())
    if not answer or len(answer.split()) > 12:
        return False
    normalized = answer.casefold().strip(" \t\r\n.,;:!?\"'`()[]{}")
    return normalized not in _NO_ANSWER_SENTINELS


def resolve_treated_answer(summary_record: dict | None, history: list[dict]) -> dict:
    if summary_record is not None:
        for key, source in (("parsed", "summary_parsed"), ("salvaged", "summary_salvaged")):
            payload = summary_record.get(key) or {}
            answer = payload.get("answer")
            if usable_short_answer(answer):
                return {
                    "answer": " ".join(answer.split()),
                    "source": source,
                    "grounded": None,
                    "qa_step": None,
                }
    for item in reversed(history):
        answer = item.get("answer")
        if usable_short_answer(answer):
            return {
                "answer": " ".join(answer.split()),
                "source": "qa_fallback",
                "grounded": item.get("answer_grounded") is True,
                "qa_step": item.get("step_number"),
            }
    return {"answer": "", "source": "none", "grounded": None, "qa_step": None}


def audit_source(source: ReplaySource) -> AuditReport:
    _require_equal(source.baseline_meta.get("git_commit"), SOURCE_COMMIT, "source commit")
    _require_equal(len(source.question_ids), 200, "source question count")
    _require_equal(source.question_ids_sha256, FULL_IDS_SHA256, "source question hash")
    _require_equal(len(source.both_gold_ids), 128, "both-gold count")
    _require_equal(
        Counter(item.stratum for item in source.scoring.values()),
        Counter({"hidden_bridge": 160, "fully_named": 40}),
        "source strata",
    )
    _require_equal(
        Counter(source.scoring[qid].stratum for qid in source.both_gold_ids),
        Counter({"hidden_bridge": 95, "fully_named": 33}),
        "both-gold strata",
    )

    qa_records = source_qa_records(source)
    stage_counts = Counter(record.get("stage") for record in qa_records)
    expected_counts = Counter(
        {"qa": 200, "qa_step2": 187, "qa_step3": 32, "qa_step4": 7, "qa_step5": 1}
    )
    _require_equal(stage_counts, expected_counts, "source QA stage counts")
    qa_hash_matches = 0
    question_answering = 0
    aggregate = 0
    joins = 0
    qa_by_question: Counter[str] = Counter()
    for qa in qa_records:
        qid = _require_str(qa, "question_id", "source QA")
        call_index = _require_int(qa, "call_index", f"source QA {qid}")
        qa_by_question[qid] += 1
        consumer_input = qa.get("consumer_input") or {}
        step_definition = consumer_input.get("step_definition")
        step_stage = prompts.stage_for("step_definer", call_index)
        step_record = source.baseline_index.get((qid, step_stage, call_index))
        payload, _ = effective_payload(step_record)
        _require_equal(payload, step_definition, f"Step Definer/QA task for {qid}/{call_index}")
        reconstruct_source_qa_fields(source, qa)
        qa_hash_matches += 1
        if consumer_input.get("task_type") == "question-answering":
            join_recorded_passages(source, qa)
            question_answering += 1
            joins += 10
        elif consumer_input.get("task_type") == "aggregate":
            aggregate += 1
            _require_equal(
                bool((consumer_input.get("retrieval") or {}).get("attempted")),
                False,
                f"aggregate retrieval for {qid}/{call_index}",
            )
            if qid in set(source.both_gold_ids):
                raise IntegrityError("aggregate QA unexpectedly lies in both-gold cohort")
        else:
            raise IntegrityError(f"unknown QA task type for {qid}/{call_index}")

    summaries = source_summary_records(source)
    _require_equal(len(summaries), 200, "source summary count")
    if {record.get("question_id") for record in summaries} != set(source.question_ids):
        raise IntegrityError("source summaries do not cover the question cohort exactly")
    summary_hash_matches = 0
    for summary in summaries:
        qid = _require_str(summary, "question_id", "source summary")
        history = (summary.get("consumer_input") or {}).get("completed_steps")
        if not isinstance(history, list):
            raise IntegrityError(f"source summary has invalid history for {qid}")
        _require_equal(len(history), qa_by_question[qid], f"summary/QA history length for {qid}")
        answer = source.baseline_answers[qid]
        _require_equal(tuple(answer.get("plan_steps") or ()), source.questions[qid].plan, f"answer plan for {qid}")
        _require_equal(answer.get("stop_reason"), source.questions[qid].stop_reason, f"answer stop for {qid}")
        _require_equal(answer.get("executed_steps"), len(history), f"answer executed steps for {qid}")
        reconstruct_source_summary_fields(source, summary)
        summary_hash_matches += 1

    source_batches = tuple(
        batch
        for stage in (*QA_STAGES, "plan_summary")
        for batch in source_scored_batches(source, stage)
    )
    _require_equal(len(source_batches), 158, "source replay batch count")
    _require_equal(sum(len(batch.members) for batch in source_batches), 627, "source replay calls")
    plus_batches = condition_batches(source, SPANS_PLUS_PASSAGES)
    only_batches = condition_batches(source, PASSAGES_ONLY)
    _require_equal((len(plus_batches), len(only_batches)), (158, 104), "condition batch counts")
    _require_equal(
        (
            sum(len(batch.members) for batch in plus_batches),
            sum(len(batch.members) for batch in only_batches),
        ),
        (627, 411),
        "condition call counts",
    )
    combined = condition_call_keys(source, SPANS_PLUS_PASSAGES) + condition_call_keys(
        source, PASSAGES_ONLY
    )
    if len(combined) != len(set(combined)):
        raise IntegrityError("condition-aware call keys collide")

    return AuditReport(
        qa_stage_counts=dict(stage_counts),
        qa_prompt_hash_matches=qa_hash_matches,
        summary_prompt_hash_matches=summary_hash_matches,
        question_answering_calls=question_answering,
        aggregate_calls=aggregate,
        extractor_joins=joins,
    )


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot parse source JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"source JSON is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict, ...]:
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise IntegrityError(
                        f"source JSONL record is not an object at {path}:{line_number}"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot parse source JSONL {path}: {exc}") from exc
    if not records:
        raise IntegrityError(f"source JSONL is empty: {path}")
    return tuple(records)


def _answer_records(records: tuple[dict, ...], label: str) -> tuple[list[str], dict[str, dict]]:
    answers = [record for record in records if record.get("record_type") == "answer"]
    ids = [str(record.get("question_id") or "") for record in answers]
    if not ids or any(not qid for qid in ids):
        raise IntegrityError(f"{label} has missing answer question IDs")
    if len(ids) != len(set(ids)):
        raise IntegrityError(f"{label} has duplicate answer question IDs")
    return ids, dict(zip(ids, answers))


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise IntegrityError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _validate_metadata(meta: dict, *, jsonl_name: str) -> None:
    _require_equal(meta.get("jsonl_sha256"), SOURCE_SHA256[jsonl_name], "internal JSONL SHA-256")
    _require_equal(
        meta.get("experiment_fingerprint"),
        SOURCE_EXPERIMENT_FINGERPRINT,
        "source experiment fingerprint",
    )
    _require_equal(meta.get("question_ids_sha256"), FULL_IDS_SHA256, "question ID SHA-256")
    _require_equal(meta.get("git_commit"), SOURCE_COMMIT, "source git commit")
    _require_equal(meta.get("model_id"), MODEL_ID, "source model")
    _require_equal(meta.get("batch_size"), BATCH_SIZE, "source batch size")
    _require_equal(meta.get("thinking_mode"), False, "source thinking mode")
    _require_equal(meta.get("n"), 200, "source cohort size")
    stage_models = set((meta.get("stage_models") or {}).values())
    stage_precisions = set((meta.get("stage_precision") or {}).values())
    _require_equal(stage_models, {MODEL_ID}, "stage model identity")
    _require_equal(stage_precisions, {PRECISION}, "stage precision")


def load_source_bundle(source_dir: Path) -> ReplaySource:
    """Load and validate the immutable committed Gate-C source artifacts."""
    source_dir = Path(source_dir)
    for name, expected in SOURCE_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise IntegrityError(f"missing source artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise IntegrityError(
                f"source SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )

    baseline_meta = _read_json(source_dir / BASELINE_META)
    single_meta = _read_json(source_dir / SINGLE_META)
    _validate_metadata(baseline_meta, jsonl_name=BASELINE_JSONL)
    _validate_metadata(single_meta, jsonl_name=SINGLE_JSONL)

    baseline_records = _read_jsonl(source_dir / BASELINE_JSONL)
    single_records = _read_jsonl(source_dir / SINGLE_JSONL)
    baseline_ids, baseline_answers = _answer_records(baseline_records, "baseline")
    single_ids, single_answers = _answer_records(single_records, "single")

    _require_equal(len(baseline_ids), 200, "baseline answer count")
    _require_equal(single_ids, baseline_ids, "cross-arm ordered answer IDs")
    question_ids = tuple(baseline_ids)
    question_ids_sha256 = ordered_ids_sha256(question_ids)
    _require_equal(question_ids_sha256, FULL_IDS_SHA256, "ordered question ID SHA-256")

    for qid in question_ids:
        baseline = baseline_answers[qid]
        single = single_answers[qid]
        for key in ("question", "gold_answer"):
            _require_equal(single.get(key), baseline.get(key), f"cross-arm {key} for {qid}")
        for label, record in (("baseline", baseline), ("single", single)):
            _require_equal(
                record.get("experiment_fingerprint"),
                SOURCE_EXPERIMENT_FINGERPRINT,
                f"{label} answer experiment fingerprint for {qid}",
            )
            _require_equal(
                record.get("question_manifest_sha256"),
                FULL_IDS_SHA256,
                f"{label} answer question manifest for {qid}",
            )

    baseline_index = _index_agent_calls(baseline_records, "baseline")
    single_index = _index_agent_calls(single_records, "single")
    questions: dict[str, FrozenQuestion] = {}
    scoring: dict[str, ScoringQuestion] = {}
    for qid in question_ids:
        answer = baseline_answers[qid]
        summary = baseline_index.get((qid, "plan_summary", 0))
        if summary is None:
            raise IntegrityError(f"missing frozen plan summary call for {qid}")
        consumer_input = summary.get("consumer_input") or {}
        plan = consumer_input.get("plan")
        stop_reason = consumer_input.get("stop_reason")
        if not isinstance(plan, list) or not plan or not all(
            isinstance(item, str) and item for item in plan
        ):
            raise IntegrityError(f"invalid frozen plan for {qid}")
        if not isinstance(stop_reason, str) or not stop_reason:
            raise IntegrityError(f"invalid frozen stop reason for {qid}")
        question = answer.get("question")
        gold_answer = answer.get("gold_answer")
        stratum = answer.get("retrieval_stratum")
        if not isinstance(question, str) or not question:
            raise IntegrityError(f"invalid question text for {qid}")
        if not isinstance(gold_answer, str):
            raise IntegrityError(f"invalid gold answer for {qid}")
        if stratum not in {"hidden_bridge", "fully_named"}:
            raise IntegrityError(f"invalid retrieval stratum for {qid}: {stratum!r}")
        both_gold = answer.get("retrieval_all_gold") is True
        questions[qid] = FrozenQuestion(qid, question, tuple(plan), stop_reason)
        scoring[qid] = ScoringQuestion(gold_answer, stratum, both_gold)

    both_gold_ids = tuple(qid for qid in question_ids if scoring[qid].both_gold)
    _require_equal(len(both_gold_ids), 128, "baseline-defined both-gold count")
    _require_equal(
        ordered_ids_sha256(both_gold_ids),
        BOTH_GOLD_IDS_SHA256,
        "baseline-defined both-gold ordered ID SHA-256",
    )
    _require_equal(
        Counter(scoring[qid].stratum for qid in both_gold_ids),
        Counter({"hidden_bridge": 95, "fully_named": 33}),
        "baseline-defined both-gold strata",
    )

    return ReplaySource(
        baseline_records=baseline_records,
        baseline_meta=baseline_meta,
        single_records=single_records,
        single_meta=single_meta,
        baseline_index=baseline_index,
        single_index=single_index,
        question_ids=question_ids,
        question_ids_sha256=question_ids_sha256,
        both_gold_ids=both_gold_ids,
        baseline_answers=baseline_answers,
        single_answers=single_answers,
        questions=questions,
        scoring=scoring,
    )
