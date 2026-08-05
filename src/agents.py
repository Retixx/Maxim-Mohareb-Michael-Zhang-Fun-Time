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

import hashlib
import json
import time
from datetime import datetime, timezone

from . import prompts
from .models import generate_batch
from .parsing import parse_output, salvage

MAX_SUB_QUESTIONS = 3


def rendered_prompt_sha256(messages: list[dict]) -> str:
    """Hash the exact rendered message objects supplied to the chat template."""
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def build_solo_fields(question: str, paragraphs: str) -> dict:
    """The single-agent control sees all ten provided paragraphs directly."""
    return {"question": question, "paragraphs": paragraphs}


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
    model_id: str | None = None,
    batch_id: str | None = None,
    execution_session_id: str | None = None,
    gpu_metadata: dict | None = None,
    timing_eligible: bool = True,
    phase: str = "scored",
    config_fingerprint: str | None = None,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    question_manifest_sha256: str | None = None,
    batch_ordinal: int | None = None,
    timing_repeat: int | None = None,
    execution_ordinal: int | None = None,
    return_batch_record: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Execute a batch of same-role calls and return SPEC §7 log records.

    `calls` items: {"question_id": str, "call_index": int, "fields": dict}

    `model_id` is recorded per call, not just per run. SPEC §7: with the size
    ablation (Phase S) the base model varies *within* a run, so a record carrying
    only `precision` cannot say which model produced it.
    """
    if not calls:
        empty = []
        return (empty, {}) if return_batch_record else empty

    messages_list = [prompts.build_messages(role, **c["fields"]) for c in calls]
    rendered_hashes = [rendered_prompt_sha256(m) for m in messages_list]
    gens = generate_batch(
        model, tok, messages_list, prompts.MAX_NEW_TOKENS[role], batch_size=batch_size,
        log_confidence=log_confidence,
    )

    records = []
    for member_index, (call, gen) in enumerate(zip(calls, gens)):
        status, parsed = parse_output(role, gen["raw_output"], gen["hit_token_cap"])
        # SPEC §13b.1: a failed call may still carry usable fields. Recorded
        # separately so parse_status stays exactly what it was.
        salvaged = None if status == "ok" else salvage(role, gen["raw_output"])
        conf = {k: gen[k] for k in ("mean_logprob", "min_logprob", "mean_entropy")
                if k in gen}
        records.append(
            {
                **conf,
                "run_id": run_id,
                "record_type": "agent_call",
                "question_id": call["question_id"],
                "stage": role,
                "model_id": model_id,
                "precision": precision,
                "call_index": call["call_index"],
                "batch_id": batch_id,
                "batch_member_index": member_index,
                "timing_repeat": timing_repeat,
                "execution_session_id": execution_session_id,
                "timing_eligible": timing_eligible,
                "question_manifest_sha256": question_manifest_sha256,
                "config_fingerprint": config_fingerprint,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "prompt_tokens": gen["prompt_tokens"],
                "output_tokens": gen["output_tokens"],
                "latency_s": gen["latency_s"],
                "raw_output": gen["raw_output"],
                "parse_status": status,
                "parsed": parsed,
                "salvaged": salvaged,
                "consumer_payload_source": call.get("consumer_payload_source"),
                "consumer_input": call.get("consumer_input"),
                "prompt_version": prompts.ROLE_PROMPT_VERSIONS[role],
                "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
                "prompt_template_sha256": prompts.prompt_template_sha256(role),
                "rendered_prompt_sha256": rendered_hashes[member_index],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(gpu_metadata or {}),
            }
        )
    if not return_batch_record:
        return records

    first = gens[0]
    batch_record = {
        "record_type": "batch",
        "run_id": run_id,
        "stage": role,
        "model_id": model_id,
        "precision": precision,
        "batch_id": batch_id,
        "phase": phase,
        "timing_eligible": timing_eligible,
        "execution_session_id": execution_session_id,
        "members": [
            {"question_id": c["question_id"], "call_index": c["call_index"]}
            for c in calls
        ],
        "batch_size_actual": len(calls),
        "batch_size_requested": batch_size,
        "batch_ordinal": batch_ordinal,
        "timing_repeat": timing_repeat,
        "execution_ordinal": execution_ordinal,
        "batch_wall_s": first.get("batch_wall_s"),
        "prompt_tokens_total": sum(g["prompt_tokens"] for g in gens),
        "prompt_tokens_max": max(g["prompt_tokens"] for g in gens),
        "output_tokens_total": sum(g["output_tokens"] for g in gens),
        "output_tokens_max": max(g["output_tokens"] for g in gens),
        "padded_input_tokens": first.get("padded_input_tokens"),
        "oom": False,
        "retry_count": 0,
        "question_manifest_sha256": question_manifest_sha256,
        "config_fingerprint": config_fingerprint,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
        "prompt_template_sha256": prompts.prompt_template_sha256(role),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(gpu_metadata or {}),
    }
    return records, batch_record
