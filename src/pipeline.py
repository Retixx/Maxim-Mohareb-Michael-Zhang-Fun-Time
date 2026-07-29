"""Dataset loading and pipeline orchestration.

BUILD STAGE 1/2 (SPEC §11): this is the naive per-question driver used for the
Gate 1 smoke test. The stage-major refactor described in SPEC §6 is build step
3 and is deliberately NOT here yet.
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


def run_question_naive(model, tok, q: dict, run_id: str, precision: str, verbose: bool = True):
    """Run one question through all four agents. Returns (records, answer_record).

    Build-stage-1 shape: one question at a time, one model resident, FP16.
    """
    records = []
    qid, question = q["question_id"], q["question"]

    def _log(role, recs):
        records.extend(recs)
        if verbose:
            for r in recs:
                print(f"\n--- [{role}] qid={qid} call_index={r['call_index']} "
                      f"status={r['parse_status']} "
                      f"tok_in={r['prompt_tokens']} tok_out={r['output_tokens']} "
                      f"{r['latency_s']}s ---")
                print(r["raw_output"])

    # 1. Planner ------------------------------------------------------------
    recs = agents.run_calls(
        model, tok, "planner",
        [{"question_id": qid, "call_index": 0,
          "fields": agents.build_planner_fields(question)}],
        precision, run_id,
    )
    _log("planner", recs)
    plan = recs[0]["parsed"]
    sub_questions = agents.clamp_sub_questions(
        (plan or {}).get("sub_questions", []), fallback_question=question
    )

    # 2. Step Definer -------------------------------------------------------
    recs = agents.run_calls(
        model, tok, "step_definer",
        [{"question_id": qid, "call_index": i,
          "fields": agents.build_step_definer_fields(question, sq)}
         for i, sq in enumerate(sub_questions)],
        precision, run_id,
    )
    _log("step_definer", recs)
    specs = [r["parsed"] for r in recs]

    # 3. Extractor ----------------------------------------------------------
    recs = agents.run_calls(
        model, tok, "extractor",
        [{"question_id": qid, "call_index": i,
          "fields": agents.build_extractor_fields(q["paragraphs"], sq, spec)}
         for i, (sq, spec) in enumerate(zip(sub_questions, specs))],
        precision, run_id,
    )
    _log("extractor", recs)
    evidence = [
        (sq, (r["parsed"] or {}).get("spans", []))
        for sq, r in zip(sub_questions, recs)
    ]

    # 4. QA -----------------------------------------------------------------
    recs = agents.run_calls(
        model, tok, "qa",
        [{"question_id": qid, "call_index": 0,
          "fields": agents.build_qa_fields(question, evidence)}],
        precision, run_id,
    )
    _log("qa", recs)
    final_answer = (recs[0]["parsed"] or {}).get("answer", "")

    answer_record = {
        "run_id": run_id,
        "question_id": qid,
        "record_type": "answer",
        "question": question,
        "gold_answer": q["answer"],
        "predicted_answer": final_answer,
        "em": exact_match(final_answer, q["answer"]),
        "f1": f1_score(final_answer, q["answer"]),
        "n_sub_questions": len(sub_questions),
        "level": q.get("level"),
        "type": q.get("type"),
    }
    return records, answer_record
