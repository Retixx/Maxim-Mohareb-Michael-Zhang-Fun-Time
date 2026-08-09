"""CPU-safe core for the frozen-upstream passage replay experiment.

The replay is intentionally downstream of retrieval.  This module projects the
committed Gate-C artifacts into a gold-free prompt trace and a separate scoring
view, then fails closed if any source identity changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src import pipeline


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(question_ids: tuple[str, ...] | list[str]) -> str:
    payload = "".join(f"{qid}\n" for qid in question_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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

    baseline_index = pipeline.index_records(list(baseline_records))
    single_index = pipeline.index_records(list(single_records))
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
