"""Dataset loading and stage-major orchestration.

SPEC §6: the pipeline runs stage-by-stage across ALL questions, not
question-by-question through all stages. Only one model is resident at a time,
which is what makes a 4 GB card work and what enables large batches.

This module is pure: it turns (questions, records-so-far) into the list of calls
a stage still needs to make. It owns no model and touches no disk. The driver
that loads models, writes JSONL and resumes is src/runner.py.
"""

import random

from . import agents, prompts
from .metrics import exact_match, f1_score

_HOTPOT_SOURCES = ("hotpotqa/hotpot_qa", "hotpot_qa")


def load_questions(n: int, seed: int = 0, split: str = "validation") -> list[dict]:
    """Load `n` HotpotQA distractor-setting questions.

    SPEC §3: distractor setting, dev split, no retriever — the 10 provided
    paragraphs (2 gold + 8 distractors) are used directly.
    """
    from datasets import load_dataset

    ds = None
    errors = []
    for src in _HOTPOT_SOURCES:
        try:
            ds = load_dataset(src, "distractor", split=split)
            break
        except Exception as e:  # noqa: BLE001 - report all attempts if none work
            errors.append(f"{src}: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError("could not load HotpotQA distractor:\n  " + "\n  ".join(errors))

    # Seeded sample rather than head(n), so a subset is not biased by the
    # dataset's own ordering. Reproducible for a given (n, seed).
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))

    out = []
    for i in idx:
        row = ds[i]
        ctx = row["context"]
        out.append(
            {
                "question_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "level": row.get("level"),
                "type": row.get("type"),
                "paragraphs": prompts.format_paragraphs(ctx["title"], ctx["sentences"]),
            }
        )
    return out


def index_records(records: list[dict]) -> dict:
    """Key agent-call records by (question_id, stage, call_index)."""
    return {
        (r["question_id"], r["stage"], r["call_index"]): r
        for r in records
        if r.get("record_type") != "answer"
    }


def sub_questions_for(q: dict, idx: dict) -> list[str]:
    """The sub-questions a question's downstream stages fan out over.

    Derived from the Planner's record. Applies the degraded-propagation rule
    (agents.clamp_sub_questions) when the Planner failed to parse, so the shape
    of every downstream stage is fully determined by what is already on disk.
    """
    rec = idx.get((q["question_id"], "planner", 0))
    parsed = rec["parsed"] if rec else None
    return agents.clamp_sub_questions(
        (parsed or {}).get("sub_questions", []), fallback_question=q["question"]
    )


def build_stage_calls(stage: str, questions: list[dict], idx: dict) -> list[dict]:
    """Every call `stage` needs to make, given the records already on disk.

    Upstream stages must be complete before this is called for a downstream one;
    the runner enforces that by walking stages in order. Returns items shaped
    {"question_id", "call_index", "fields"} — the input to agents.run_calls.
    """
    calls = []

    if stage == "planner":
        for q in questions:
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": agents.build_planner_fields(q["question"]),
            })
        return calls

    if stage == "step_definer":
        for q in questions:
            for i, sq in enumerate(sub_questions_for(q, idx)):
                calls.append({
                    "question_id": q["question_id"],
                    "call_index": i,
                    "fields": agents.build_step_definer_fields(q["question"], sq),
                })
        return calls

    if stage == "extractor":
        for q in questions:
            for i, sq in enumerate(sub_questions_for(q, idx)):
                spec_rec = idx.get((q["question_id"], "step_definer", i))
                spec = spec_rec["parsed"] if spec_rec else None
                calls.append({
                    "question_id": q["question_id"],
                    "call_index": i,
                    "fields": agents.build_extractor_fields(q["paragraphs"], sq, spec),
                })
        return calls

    if stage == "qa":
        for q in questions:
            evidence = []
            for i, sq in enumerate(sub_questions_for(q, idx)):
                rec = idx.get((q["question_id"], "extractor", i))
                spans = (rec["parsed"] or {}).get("spans", []) if rec else []
                evidence.append((sq, spans))
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": agents.build_qa_fields(q["question"], evidence),
            })
        return calls

    raise KeyError(f"unknown stage {stage!r}; expected one of {prompts.ROLES}")


def build_answer_records(questions: list[dict], idx: dict, run_id: str) -> list[dict]:
    """One scored record per question, from the QA stage's outputs (SPEC §7)."""
    out = []
    for q in questions:
        rec = idx.get((q["question_id"], "qa", 0))
        parsed = rec["parsed"] if rec else None
        pred = (parsed or {}).get("answer", "")
        out.append({
            "run_id": run_id,
            "question_id": q["question_id"],
            "record_type": "answer",
            "question": q["question"],
            "gold_answer": q["answer"],
            "predicted_answer": pred,
            "em": exact_match(pred, q["answer"]),
            "f1": f1_score(pred, q["answer"]),
            "n_sub_questions": len(sub_questions_for(q, idx)),
            "level": q.get("level"),
            "type": q.get("type"),
        })
    return out
