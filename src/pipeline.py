"""Dataset loading and stage-major orchestration.

SPEC §6: the pipeline runs stage-by-stage across ALL questions, not
question-by-question through all stages. Only one model is resident at a time,
which is what makes a 4 GB card work and what enables large batches.

This module is pure: it turns (questions, records-so-far) into the list of calls
a stage still needs to make. It owns no model and touches no disk. The driver
that loads models, writes JSONL and resumes is src/runner.py.
"""

import random

from . import agents, evidence, prompts
from .metrics import exact_match, f1_score

_HOTPOT_SOURCES = ("hotpotqa/hotpot_qa", "hotpot_qa")


def load_questions(
    n: int,
    seed: int = 0,
    split: str = "validation",
    config: str = "distractor",
    name: str | None = None,
    exclude: set | None = None,
) -> list[dict]:
    """Load `n` HotpotQA distractor-setting questions.

    SPEC §3: distractor setting, dev split, no retriever — the 10 provided
    paragraphs (2 gold + 8 distractors) are used directly.

    `exclude` is a set of question_ids that must NOT appear in the sample, used
    to draw the confirmation set disjointly from the selection set (SPEC §5e).
    Disjointness cannot be got by changing the seed: `random.sample` is nested in
    k for a fixed seed but says nothing across seeds, and two independent
    750-question samples from 7405 overlap by ~76 in expectation. The exclusion
    is applied to the index space BEFORE sampling, so the returned sample is
    exactly n and provably disjoint — see the assertion at the end.
    """
    from datasets import load_dataset

    sources = (name,) + _HOTPOT_SOURCES if name else _HOTPOT_SOURCES
    ds = None
    errors = []
    for src in sources:
        try:
            ds = load_dataset(src, config, split=split)
            break
        except Exception as e:  # noqa: BLE001 - report all attempts if none work
            errors.append(f"{src}: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(
            f"could not load {config}/{split}:\n  " + "\n  ".join(errors)
        )

    # Seeded sample rather than head(n), so a subset is not biased by the
    # dataset's own ordering. Reproducible for a given (n, seed).
    pool = range(len(ds))
    if exclude:
        pool = [i for i in pool if ds[i]["id"] not in exclude]
        if len(pool) < n:
            raise ValueError(
                f"only {len(pool)} questions remain after excluding "
                f"{len(exclude)}; cannot draw n={n}"
            )
    idx = sorted(random.Random(seed).sample(pool, n))

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
                # Gold evidence labels, for extraction accuracy (src/evidence.py).
                # Analysis-only: nothing here reaches a prompt, so carrying them
                # cannot change what any agent generates.
                "supporting_facts": row.get("supporting_facts"),
                "sentence_index": evidence.build_sentence_index(
                    ctx["title"], ctx["sentences"]
                ),
            }
        )
    if exclude:
        # SPEC §5e requires this to be enforced, not assumed.
        overlap = {q["question_id"] for q in out} & set(exclude)
        assert not overlap, f"confirmation set overlaps selection set on {len(overlap)} ids"
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

    if stage == "solo":
        # SPEC §4a: one call per question, whole paragraph set in, short answer
        # out. No planner, no sub-questions, no extraction — that absence IS the
        # experiment.
        for q in questions:
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": {"paragraphs": q["paragraphs"], "question": q["question"]},
            })
        return calls

    if stage == "qa":
        for q in questions:
            # NOT named `evidence`: that would bind a local of the same name as
            # the imported `evidence` module for the whole function, turning any
            # future module use in build_stage_calls into an UnboundLocalError.
            blocks = []
            for i, sq in enumerate(sub_questions_for(q, idx)):
                rec = idx.get((q["question_id"], "extractor", i))
                spans = (rec["parsed"] or {}).get("spans", []) if rec else []
                blocks.append((sq, spans))
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": agents.build_qa_fields(q["question"], blocks),
            })
        return calls

    raise KeyError(f"unknown stage {stage!r}; expected one of {prompts.ROLES}")


def build_answer_records(questions: list[dict], idx: dict, run_id: str) -> list[dict]:
    """One scored record per question, from the QA stage's outputs (SPEC §7)."""
    out = []
    for q in questions:
        # A single-call run has no qa stage; its answer comes from `solo`.
        rec = idx.get((q["question_id"], "qa", 0)) or idx.get((q["question_id"], "solo", 0))
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
            "n_sub_questions": len(sub_questions_for(q, idx))
                                if idx.get((q["question_id"], "planner", 0)) else 0,
            "level": q.get("level"),
            "type": q.get("type"),
        })
    return out
