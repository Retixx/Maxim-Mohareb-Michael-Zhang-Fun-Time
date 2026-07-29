"""Role call wrappers: prompt -> generate -> parse -> log record.

One function per role builds that role's template fields from upstream state;
`run_calls` executes a homogeneous batch of calls for a single role.

DEGRADED PROPAGATION, NOT RETRY (SPEC §5)
-----------------------------------------
When a role's output fails to parse there is no second generation, ever. The
pipeline instead continues with a fixed, deterministic degraded input and the
failure is recorded in the JSONL:

    planner      fails -> sub_questions = [the original question]
    step_definer fails -> target_entity = "", search_terms = []
    extractor    fails -> spans = [] for that sub-question
    qa           fails -> final answer = "" (scores EM 0, F1 0)

This keeps the accuracy signal graded instead of collapsing every parse failure
to a dead question, while still costing exactly one generation per call. If you
are tempted to "just re-ask once when it fails" — don't. Read SPEC §5.
"""

import time
from datetime import datetime, timezone

from . import prompts
from .models import generate_batch
from .parsing import parse_output

MAX_SUB_QUESTIONS = 3


def build_planner_fields(question: str) -> dict:
    return {"question": question}


def build_step_definer_fields(question: str, sub_question: str) -> dict:
    return {"question": question, "sub_question": sub_question}


def build_extractor_fields(paragraphs: str, sub_question: str, spec: dict | None) -> dict:
    spec = spec or {}
    terms = spec.get("search_terms") or []
    return {
        "paragraphs": paragraphs,
        "sub_question": sub_question,
        "target_entity": spec.get("target_entity") or "(unspecified)",
        "search_terms": ", ".join(terms) if terms else "(none)",
    }


def build_qa_fields(question: str, evidence_blocks: list[tuple[str, list[str]]]) -> dict:
    lines = []
    for i, (sub_q, spans) in enumerate(evidence_blocks, start=1):
        lines.append(f"{i}. {sub_q}")
        if spans:
            for s in spans:
                lines.append(f"   - {s}")
        else:
            lines.append("   - (no supporting text found)")
    evidence = "\n".join(lines) if lines else "(no evidence collected)"
    return {"question": question, "evidence": evidence}


def clamp_sub_questions(sub_questions: list[str], fallback_question: str) -> list[str]:
    """Apply the degraded-propagation rule for the Planner, plus the §4 cap.

    Emitting more than 3 sub-questions is a prompt-following lapse, not a format
    error, so it is clamped here rather than failed in the parser.
    """
    if not sub_questions:
        return [fallback_question]
    return sub_questions[:MAX_SUB_QUESTIONS]


def run_calls(
    model,
    tok,
    role: str,
    calls: list[dict],
    precision: str,
    run_id: str,
    batch_size: int = 1,
    log_confidence: bool = False,
) -> list[dict]:
    """Execute a batch of same-role calls and return SPEC §7 log records.

    `calls` items: {"question_id": str, "call_index": int, "fields": dict}
    """
    if not calls:
        return []

    messages_list = [prompts.build_messages(role, **c["fields"]) for c in calls]
    gens = generate_batch(
        model, tok, messages_list, prompts.MAX_NEW_TOKENS[role], batch_size=batch_size,
        log_confidence=log_confidence,
    )

    records = []
    for call, gen in zip(calls, gens):
        status, parsed = parse_output(role, gen["raw_output"], gen["hit_token_cap"])
        conf = {k: gen[k] for k in ("mean_logprob", "min_logprob", "mean_entropy")
                if k in gen}
        records.append(
            {
                **conf,
                "run_id": run_id,
                "question_id": call["question_id"],
                "stage": role,
                "precision": precision,
                "call_index": call["call_index"],
                "prompt_tokens": gen["prompt_tokens"],
                "output_tokens": gen["output_tokens"],
                "latency_s": gen["latency_s"],
                "raw_output": gen["raw_output"],
                "parse_status": status,
                "parsed": parsed,
                "prompt_version": prompts.PROMPT_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return records
