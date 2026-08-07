"""Role call wrappers: prompt -> generate -> parse -> log record.

One function per role builds that role's template fields from upstream state;
`run_calls` executes a homogeneous batch of calls for a single role.

DEGRADED PROPAGATION, NOT RETRY (SPEC §5)
-----------------------------------------
When a role's output fails to parse there is no second generation, ever. The
pipeline instead continues with a fixed, deterministic degraded input and the
failure is recorded in the JSONL:

    planner      fails -> sub_questions = [the original question]
    step_definer fails -> salvage its task, else goal + prior answers + question
    extractor    fails -> spans = [] for that document
    qa           fails -> answer = "" and the bounded plan continues
    plan summary fails -> final answer = "" (scores EM 0, F1 0)

This keeps the accuracy signal graded instead of collapsing every parse failure
to a dead question, while still costing exactly one generation per call. If you
are tempted to "just re-ask once when it fails" — don't. Read SPEC §5.
"""

import hashlib
import json
import time
from datetime import datetime, timezone

from . import extraction, mechanism, prompts
from .models import generate_batch
from .parsing import parse_output, salvage

MAX_SUB_QUESTIONS = prompts.MAX_PLAN_STEPS


def rendered_prompt_sha256(messages: list[dict]) -> str:
    """Hash the exact rendered message objects supplied to the chat template."""
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_planner_fields(question: str) -> dict:
    return {"question": question}


def _render_prior_steps(
    prior_steps: list[dict] | None, *, grounded_only: bool = False
) -> str:
    if not prior_steps:
        return "(none; this is the first plan step)"
    lines = []
    for item in prior_steps:
        grounded = item.get("answer_grounded")
        answer = item.get("answer") or "(no answer)"
        if grounded_only and grounded is not True:
            answer = "(withheld: not evidence-grounded)"
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


def build_step_definer_fields(
    question: str,
    sub_question: str,
    *,
    step_number: int = 1,
    plan_steps: int = 1,
    full_plan: list[str] | None = None,
    prior_steps: list[dict] | None = None,
) -> dict:
    return {
        "question": question,
        "sub_question": sub_question,
        "full_plan": "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(full_plan or [sub_question], start=1)
        ),
        "step_number": step_number,
        "plan_steps": plan_steps,
        "prior_state": _render_prior_steps(prior_steps, grounded_only=True),
    }


def build_plan_summary_fields(
    question: str,
    plan: list[str],
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


def build_extractor_fields(document: str, sub_question: str) -> dict:
    return {
        "document": document,
        "sub_question": sub_question,
    }


def build_qa_fields(
    question: str,
    evidence_blocks: list[tuple[str, list[str]]],
    *,
    sub_question: str | None = None,
    step_number: int = 1,
    plan_steps: int = 1,
) -> dict:
    lines = []
    displayed = 0
    for sub_q, spans in evidence_blocks:
        if not spans:
            continue
        displayed += 1
        lines.append(f"{displayed}. {sub_q}")
        for s in spans:
            lines.append(f"   - {s}")
    evidence = "\n".join(lines) if lines else "(no evidence collected)"
    return {
        "question": question,
        "sub_question": sub_question or question,
        "step_number": step_number,
        "plan_steps": plan_steps,
        "evidence": evidence,
    }


def build_solo_fields(question: str, paragraphs: str) -> dict:
    """The single-agent control sees its one-query top-10 retrieval result."""
    return {"question": question, "paragraphs": paragraphs}


def clamp_sub_questions(sub_questions: list[str], fallback_question: str) -> list[str]:
    """Apply Planner degraded propagation and the logged five-step edge cap."""
    if not sub_questions:
        return [fallback_question]
    return sub_questions[:MAX_SUB_QUESTIONS]


def run_calls(
    model,
    tok,
    stage: str,
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
    experiment_fingerprint: str | None = None,
    force_full_generation: bool = False,
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

    service_started = time.perf_counter()
    role = prompts.role_for(stage)
    prompt_role = prompts.prompt_for(stage)
    messages_list = [prompts.build_messages(prompt_role, **c["fields"]) for c in calls]
    rendered_hashes = [rendered_prompt_sha256(m) for m in messages_list]
    gens = generate_batch(
        model, tok, messages_list, prompts.MAX_NEW_TOKENS[prompt_role], batch_size=batch_size,
        log_confidence=log_confidence,
        force_full_generation=force_full_generation,
    )

    records = []
    for member_index, (call, gen) in enumerate(zip(calls, gens)):
        status, parsed = parse_output(prompt_role, gen["raw_output"], gen["hit_token_cap"])
        # SPEC §13b.1: a failed call may still carry usable fields. Recorded
        # separately so parse_status stays exactly what it was.
        salvaged = None if status == "ok" else salvage(prompt_role, gen["raw_output"])
        conf = {k: gen[k] for k in ("mean_logprob", "min_logprob", "mean_entropy")
                if k in gen}
        paragraphs = (
            call.get("fields", {}).get("paragraphs")
            or call.get("fields", {}).get("document")
        )
        consumer_payload = None
        extractor_normalization = None
        source_sentences = (
            (call.get("consumer_input") or {}).get("document_sentences")
            if prompt_role == "extractor"
            else None
        )
        if prompt_role == "extractor":
            producer_payload = parsed if parsed is not None else salvaged
            if source_sentences is not None:
                normalized, extractor_normalization = extraction.normalize_spans(
                    (producer_payload or {}).get("spans", []), source_sentences
                )
                consumer_payload = {"spans": normalized}
            else:
                consumer_payload = producer_payload
                extractor_normalization = {
                    "status": "source_sentences_unavailable",
                    "input_span_count": len(
                        (producer_payload or {}).get("spans", [])
                    ),
                }
        copy_rate = (
            mechanism.verbatim_rate(
                ((parsed or salvaged) or {}).get("spans", []), paragraphs or ""
            )
            if prompt_role == "extractor" else None
        )
        records.append(
            {
                **conf,
                "run_id": run_id,
                "record_type": "agent_call",
                "question_id": call["question_id"],
                "stage": stage,
                "conceptual_role": role,
                "prompt_role": prompt_role,
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
                "experiment_fingerprint": experiment_fingerprint,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "prompt_tokens": gen["prompt_tokens"],
                "output_tokens": gen["output_tokens"],
                "generated_sequence_tokens": gen.get("generated_sequence_tokens"),
                "context_window_tokens": gen.get("context_window_tokens"),
                "forced_full_generation": gen.get("forced_full_generation", False),
                "latency_s": gen["latency_s"],
                "raw_output": gen["raw_output"],
                "parse_status": status,
                "parsed": parsed,
                "salvaged": salvaged,
                "consumer_payload": consumer_payload,
                "extractor_normalization": extractor_normalization,
                "strict_format_ok": mechanism.strict_format_ok(
                    prompt_role, gen["raw_output"]
                ),
                "protocol_ok": mechanism.protocol_ok(
                    prompt_role, gen["raw_output"], parsed,
                    paragraphs=paragraphs,
                    source_sentences=source_sentences,
                    max_plan_steps=MAX_SUB_QUESTIONS,
                ),
                "verbatim_copy_rate": copy_rate,
                "consumer_payload_source": call.get("consumer_payload_source"),
                "consumer_input": call.get("consumer_input"),
                "prompt_version": prompts.ROLE_PROMPT_VERSIONS[prompt_role],
                "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
                "prompt_template_sha256": prompts.prompt_template_sha256(prompt_role),
                "rendered_prompt_sha256": rendered_hashes[member_index],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(gpu_metadata or {}),
            }
        )
    if not return_batch_record:
        return records

    first = gens[0]
    service_wall_s = time.perf_counter() - service_started
    batch_record = {
        "record_type": "batch",
        "run_id": run_id,
        "stage": stage,
        "conceptual_role": role,
        "prompt_role": prompt_role,
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
        # Prompt construction/rendering, tokenization/H2D, generation, decode,
        # parse/salvage, and protocol metrics. Durable JSONL I/O is excluded.
        "service_wall_s": service_wall_s,
        "prompt_tokens_total": sum(g["prompt_tokens"] for g in gens),
        "prompt_tokens_max": max(g["prompt_tokens"] for g in gens),
        "output_tokens_total": sum(g["output_tokens"] for g in gens),
        "generated_sequence_tokens_total": sum(
            int(g.get("generated_sequence_tokens") or 0) for g in gens
        ),
        "output_tokens_max": max(g["output_tokens"] for g in gens),
        "padded_input_tokens": first.get("padded_input_tokens"),
        "context_window_tokens": first.get("context_window_tokens"),
        "forced_full_generation": first.get("forced_full_generation", False),
        "oom": False,
        "retry_count": 0,
        "question_manifest_sha256": question_manifest_sha256,
        "config_fingerprint": config_fingerprint,
        "experiment_fingerprint": experiment_fingerprint,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
        "prompt_template_sha256": prompts.prompt_template_sha256(prompt_role),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(gpu_metadata or {}),
    }
    return records, batch_record
