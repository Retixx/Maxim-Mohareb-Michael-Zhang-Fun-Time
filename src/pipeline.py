"""Dataset loading and stage-major orchestration.

SPEC §6: the pipeline runs stage-by-stage across ALL questions, not
question-by-question through all stages. Only one model is resident at a time,
which bounds execution VRAM and enables fixed large batches.

This module is pure: it turns (questions, records-so-far) into the list of calls
a stage still needs to make. It owns no model and touches no disk. The driver
that loads models, writes JSONL and resumes is src/runner.py.
"""

import hashlib
import json
import random
from pathlib import Path

from . import agents, evidence, prompts, retrieval
from .metrics import exact_match, f1_score

_HOTPOT_SOURCES = ("hotpotqa/hotpot_qa", "hotpot_qa")


def _require_retriever(retriever, stage: str) -> None:
    if retriever is None:
        raise ValueError(
            f"stage {stage!r} reads passages and needs a retriever; SPEC §3 no "
            "longer hands the dataset's 10 paragraphs to any stage"
        )


def question_ids_sha256(question_ids: list[str]) -> str:
    """Hash ordered IDs; order is part of the canonical batch definition."""
    payload = "".join(f"{qid}\n" for qid in question_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_id_manifest(path: str | Path) -> tuple[list[str], dict]:
    """Read a JSON ID manifest and verify uniqueness and its embedded hash."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON question manifest {path}: {exc}") from exc

    if isinstance(payload, list):
        ids, meta = payload, {}
    elif isinstance(payload, dict):
        ids = payload.get("question_ids", payload.get("ids"))
        meta = dict(payload)
    else:
        ids, meta = None, {}
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        raise ValueError(f"{path}: expected a non-empty list of string question IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: contains {len(ids) - len(set(ids))} duplicate IDs")

    actual = question_ids_sha256(ids)
    expected = (
        meta.get("question_ids_sha256")
        or meta.get("ordered_ids_sha256")
        or meta.get("ids_sha256")
    )
    if expected and expected != actual:
        raise ValueError(
            f"{path}: ordered question-ID SHA-256 mismatch; expected {expected}, got {actual}"
        )
    return ids, {
        **meta,
        "question_ids_sha256": actual,
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_path": str(path),
    }


def load_questions(
    n: int,
    seed: int = 0,
    split: str = "validation",
    config: str = "distractor",
    name: str | None = None,
    exclude: set | None = None,
    manifest_path: str | Path | None = None,
    manifest_sha256: str | None = None,
    revision: str | None = None,
    metadata: dict | None = None,
    warmup_question_ids: list[str] | None = None,
    warmup_out: list[dict] | None = None,
) -> list[dict]:
    """Load HotpotQA questions/labels without exposing row context to prompts.

    The distractor config is the canonical source for answers and supporting
    sentence IDs. Its context is retained only for analysis/integrity checks;
    every answering passage comes from the separate pooled retrieval corpus.
    Final execution supplies a frozen ordered manifest plus exact exclusions.
    """
    from datasets import load_dataset

    sources = (name,) + _HOTPOT_SOURCES if name else _HOTPOT_SOURCES
    ds = None
    errors = []
    for src in sources:
        try:
            ds = load_dataset(src, config, split=split, revision=revision)
            break
        except Exception as e:  # noqa: BLE001 - report all attempts if none work
            errors.append(f"{src}: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError(
            f"could not load {config}/{split}:\n  " + "\n  ".join(errors)
        )

    all_ids = [ds[i]["id"] for i in range(len(ds))]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"dataset {config}/{split} contains duplicate question IDs")
    id_to_index = {qid: i for i, qid in enumerate(all_ids)}
    exclude = set(exclude or ())

    manifest_meta = None
    if manifest_path is not None:
        selected_ids, manifest_meta = load_id_manifest(manifest_path)
        exclusions_meta = manifest_meta.get("exclusions") or {}
        if not isinstance(exclusions_meta, dict):
            raise ValueError(f"{manifest_path}: exclusions must be an object")
        embedded_exclusions = (
            exclusions_meta.get("question_ids")
            or manifest_meta.get("excluded_question_ids")
            or []
        )
        if not isinstance(embedded_exclusions, list) or not all(
            isinstance(x, str) and x for x in embedded_exclusions
        ):
            raise ValueError(f"{manifest_path}: excluded_question_ids must be a string list")
        if len(embedded_exclusions) != len(set(embedded_exclusions)):
            raise ValueError(f"{manifest_path}: exclusion list contains duplicate IDs")
        exclusion_hash = question_ids_sha256(embedded_exclusions) if embedded_exclusions else None
        expected_exclusion_hash = (
            exclusions_meta.get("question_ids_sha256")
            or exclusions_meta.get("ordered_ids_sha256")
        )
        if expected_exclusion_hash and expected_exclusion_hash != exclusion_hash:
            raise ValueError(
                f"{manifest_path}: exclusion SHA-256 mismatch; expected "
                f"{expected_exclusion_hash}, got {exclusion_hash}"
            )
        expected_exclusion_count = exclusions_meta.get(
            "count", exclusions_meta.get("unique_count")
        )
        if expected_exclusion_count is not None and expected_exclusion_count != len(embedded_exclusions):
            raise ValueError(
                f"{manifest_path}: exclusion count says {expected_exclusion_count}, "
                f"contains {len(embedded_exclusions)}"
            )
        exclude.update(embedded_exclusions)
        if len(selected_ids) != n:
            raise ValueError(
                f"{manifest_path}: contains {len(selected_ids)} IDs but requested n={n}"
            )
        sampling_meta = manifest_meta.get("sampling") or {}
        if sampling_meta.get("n", n) != n or sampling_meta.get("seed", seed) != seed:
            raise ValueError(
                f"{manifest_path}: sampling provenance does not match requested n={n}, seed={seed}"
            )
        dataset_meta = manifest_meta.get("dataset") or {}
        if dataset_meta.get("name", name) != name:
            raise ValueError(f"{manifest_path}: dataset name does not match runtime")
        if dataset_meta.get("config", config) != config or dataset_meta.get("split", split) != split:
            raise ValueError(f"{manifest_path}: dataset config/split does not match runtime")
        if dataset_meta.get("revision", revision) != revision:
            raise ValueError(f"{manifest_path}: dataset revision does not match runtime")
        if dataset_meta.get("row_count", len(all_ids)) != len(all_ids):
            raise ValueError(f"{manifest_path}: pinned dataset row count changed")
        actual_dataset_hash = question_ids_sha256(all_ids)
        if dataset_meta.get("ordered_ids_sha256", actual_dataset_hash) != actual_dataset_hash:
            raise ValueError(f"{manifest_path}: pinned dataset ID order/hash changed")
        if manifest_sha256 and manifest_meta["question_ids_sha256"] != manifest_sha256:
            raise ValueError(
                "configured question-manifest SHA-256 mismatch: expected "
                f"{manifest_sha256}, got {manifest_meta['question_ids_sha256']}"
            )
        overlap = set(selected_ids) & exclude
        if overlap:
            raise ValueError(
                f"final manifest overlaps the exclusion set on {len(overlap)} IDs; "
                f"examples: {sorted(overlap)[:5]}"
            )
        missing = [qid for qid in selected_ids if qid not in id_to_index]
        if missing:
            raise ValueError(
                f"final manifest has {len(missing)} IDs absent from pinned dataset; "
                f"examples: {missing[:5]}"
            )
        # Preserve manifest order exactly. Sorting would change batch neighbors.
        idx = [id_to_index[qid] for qid in selected_ids]
    else:
        pool = [i for i, qid in enumerate(all_ids) if qid not in exclude]
        if len(pool) < n:
            raise ValueError(
                f"only {len(pool)} questions remain after excluding "
                f"{len(exclude)}; cannot draw n={n}"
            )
        idx = sorted(random.Random(seed).sample(pool, n))

    def convert_row(i: int) -> dict:
        row = ds[i]
        ctx = row["context"]
        sf = row.get("supporting_facts") or {"title": [], "sent_id": []}
        # SPEC §3 (revised): the dataset's ten paragraphs are NO LONGER handed to
        # any stage. They now serve only as corpus material and as the gold
        # labels for scoring. Every stage that reads passages retrieves them.
        gold_titles = list(dict.fromkeys(sf["title"]))
        hidden = [
            title for title in gold_titles
            if not retrieval.title_is_mentioned(title, row["question"])
        ]
        return {
            "question_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "level": row.get("level"),
            "type": row.get("type"),
            # Gold evidence labels are analysis-only and never enter a prompt.
            "supporting_facts": row.get("supporting_facts"),
            # Prespecified analysis-only stratifier. `fully_named` questions name
            # every gold page in the question; `hidden_bridge` questions withhold
            # at least one title and therefore expose whether stateful later
            # retrieval resolves information learned during execution. This does
            # not assume a fixed number of hops or guarantee a particular effect.
            # It is computed before any model runs and never enters a prompt.
            "retrieval_stratum": "hidden_bridge" if hidden else "fully_named",
            "hidden_gold_titles": hidden,
            "sentence_index": evidence.build_sentence_index(
                ctx["title"], ctx["sentences"]
            ),
        }

    out = [convert_row(i) for i in idx]
    if warmup_question_ids is not None:
        if warmup_out is None:
            raise ValueError("warmup_out is required with warmup_question_ids")
        if len(warmup_question_ids) != len(set(warmup_question_ids)):
            raise ValueError("warm-up question IDs contain duplicates")
        if not set(warmup_question_ids) <= exclude:
            raise ValueError("every warm-up question must be in the frozen exclusion set")
        missing_warmup = [qid for qid in warmup_question_ids if qid not in id_to_index]
        if missing_warmup:
            raise ValueError(f"warm-up IDs absent from pinned dataset: {missing_warmup[:5]}")
        warmup_out.extend(convert_row(id_to_index[qid]) for qid in warmup_question_ids)
    overlap = {q["question_id"] for q in out} & exclude
    if overlap:
        raise AssertionError(f"evaluation set overlaps exclusions on {len(overlap)} IDs")
    if metadata is not None:
        metadata.update({
            "config": config,
            "split": split,
            "revision": revision,
            "dataset_fingerprint": getattr(ds, "_fingerprint", None),
            "dataset_ordered_ids_sha256": question_ids_sha256(all_ids),
            "question_ids_sha256": question_ids_sha256(
                [q["question_id"] for q in out]
            ),
            "manifest": manifest_meta,
            "excluded_ids_count": len(exclude),
            "excluded_ids_sha256": question_ids_sha256(sorted(exclude)) if exclude else None,
        })
    return out


def index_records(records: list[dict]) -> dict:
    """Key agent-call records by (question_id, stage, call_index)."""
    return {
        (r["question_id"], r["stage"], r["call_index"]): r
        for r in records
        if r.get("record_type") in (None, "agent_call")
        and all(k in r for k in ("question_id", "stage", "call_index"))
    }


def sub_questions_for(q: dict, idx: dict) -> list[str]:
    """The sub-questions a question's downstream stages fan out over.

    Derived from the Planner's record. Applies the degraded-propagation rule
    (agents.clamp_sub_questions) when the Planner failed to parse, so the shape
    of every downstream stage is fully determined by what is already on disk.
    """
    rec = idx.get((q["question_id"], "planner", 0))
    parsed = (rec.get("parsed") or rec.get("salvaged")) if rec else None
    return agents.clamp_sub_questions(
        (parsed or {}).get("sub_questions", []), fallback_question=q["question"]
    )


def planner_depth_for(q: dict, idx: dict) -> dict:
    rec = idx.get((q["question_id"], "planner", 0))
    payload, source = _effective_payload(rec)
    emitted = (payload or {}).get("sub_questions") or []
    emitted_depth = len(emitted) if emitted else 0
    executed_depth = len(sub_questions_for(q, idx))
    return {
        "planner_payload_source": source,
        "planner_emitted_depth": emitted_depth,
        "executed_plan_depth": executed_depth,
        "plan_depth_cap": agents.MAX_SUB_QUESTIONS,
        "plan_was_clamped": emitted_depth > agents.MAX_SUB_QUESTIONS,
    }


def spans_of(rec: dict | None) -> tuple[list[str], str]:
    """The spans a consumer should read from an Extractor record, and their source.

    SPEC §13b.1: prefer the validated payload, else whatever was salvageable from
    a failed call. The call still counts as a parse failure — this only stops
    usable spans being discarded along with a malformed container.

    Repaired runs additionally preserve the producer payload while exposing a
    sentence-normalized ``consumer_payload``. Prefer that deterministic payload
    and retain whether its source was parsed or salvaged in the telemetry label.
    """
    if rec:
        if rec.get("consumer_payload") is not None:
            source = "parsed" if rec.get("parsed") is not None else "salvaged"
            return (
                (rec["consumer_payload"] or {}).get("spans", []),
                f"normalized_{source}",
            )
        if rec.get("parsed") is not None:
            return (rec["parsed"] or {}).get("spans", []), "parsed"
        if rec.get("salvaged") is not None:
            return (rec["salvaged"] or {}).get("spans", []), "salvaged"
    return [], "fallback"


def _effective_payload(rec: dict | None) -> tuple[dict | None, str]:
    if rec:
        if rec.get("parsed") is not None:
            return rec["parsed"], "parsed"
        if rec.get("salvaged") is not None:
            return rec["salvaged"], "salvaged"
    return None, "fallback"


def _extractor_records(qid: str, step_index: int, idx: dict) -> list[dict]:
    """Return one step's contiguous document records without scanning the run.

    Extractor call indices are retrieval ranks starting at zero.  Downstream
    stages only run after the runner has completed the upstream stage, so the
    durable records are contiguous; a missing rank is therefore the bounded
    terminator (and rank zero missing represents a zero-result retrieval).
    """
    stage = prompts.stage_for("extractor", step_index)
    records = []
    document_rank = 0
    while True:
        rec = idx.get((qid, stage, document_rank))
        if rec is None:
            return records
        records.append(rec)
        document_rank += 1


def step_history_for(q: dict, idx: dict, before_step: int | None = None) -> list[dict]:
    """Compact append-only MA-RAG state reconstructed from durable call records."""
    plan = sub_questions_for(q, idx)
    limit = len(plan) if before_step is None else min(before_step, len(plan))
    history = []
    for step_index in range(limit):
        qa_stage = prompts.stage_for("qa", step_index)
        qa_rec = idx.get((q["question_id"], qa_stage, step_index))
        if qa_rec is None:
            break
        qa_payload, qa_source = _effective_payload(qa_rec)
        spans = []
        span_sources = []
        for ext_rec in _extractor_records(q["question_id"], step_index, idx):
            got, source = spans_of(ext_rec)
            spans.extend(got)
            span_sources.append(source)
        sd_stage = prompts.stage_for("step_definer", step_index)
        task, task_source = _effective_payload(
            idx.get((q["question_id"], sd_stage, step_index))
        )
        history.append({
            "step_number": step_index + 1,
            "sub_question": plan[step_index],
            "task": task or {},
            "task_source": task_source,
            "answer": (qa_payload or {}).get("answer", ""),
            "success": (qa_payload or {}).get("success"),
            "rating": (qa_payload or {}).get("rating"),
            "qa_source": qa_source,
            "spans": list(dict.fromkeys(spans)),
            "span_sources": span_sources,
        })
    return history


def _step_is_active(q: dict, idx: dict, step_index: int) -> bool:
    plan = sub_questions_for(q, idx)
    if step_index >= len(plan):
        return False
    if step_index == 0:
        return True
    history = step_history_for(q, idx, before_step=step_index)
    if len(history) != step_index:
        return False
    return not history or history[-1].get("success") != "no"


def _step_task(q: dict, idx: dict, step_index: int) -> tuple[dict, str]:
    stage = prompts.stage_for("step_definer", step_index)
    payload, source = _effective_payload(
        idx.get((q["question_id"], stage, step_index))
    )
    payload = payload or {}
    task_type = payload.get("type")
    task = payload.get("task")
    if isinstance(task, str) and task.strip():
        # A malformed/missing type must not discard a usable salvaged task.  It
        # remains a parse failure in the producer record; the downstream source
        # records that the query came from salvage rather than inventing a clean
        # parse in telemetry.
        if task_type not in {"aggregate", "question-answering"}:
            task_type = "question-answering"
        return {"type": task_type, "task": task.strip()}, source

    plan = sub_questions_for(q, idx)
    current_goal = plan[step_index]
    prior_answers = []
    for prior in step_history_for(q, idx, before_step=step_index):
        answer = " ".join(str(prior.get("answer") or "").split())
        if answer and answer not in prior_answers:
            prior_answers.append(answer)
    context = "; ".join(prior_answers) if prior_answers else "(none yet)"
    return {
        "type": "question-answering",
        "task": (
            f"Current goal: {current_goal} | Prior answers: {context} | "
            f"Original question: {q['question']}"
        ),
    }, "fallback"


def _retrieval_event_for_qa(
    q: dict, idx: dict, step_index: int, task: dict
) -> dict:
    """Persist retrieval intent on QA, including no-hit and no-query steps."""
    if task["type"] == "aggregate":
        return {
            "step": step_index + 1,
            "task_type": "aggregate",
            "attempted": False,
            "query": None,
            "titles": [],
        }

    records = _extractor_records(q["question_id"], step_index, idx)
    event = None
    if records:
        event = ((records[0].get("consumer_input") or {}).get("retrieval"))
    got = dict(event or {})
    got.setdefault("step", step_index + 1)
    got.setdefault("task_type", "question-answering")
    got.setdefault("attempted", True)
    got.setdefault("query", task["task"])
    if "titles" not in got:
        got["titles"] = list(dict.fromkeys(
            title
            for rec in records
            for title in [
                (rec.get("consumer_input") or {}).get("document_title")
            ]
            if title
        ))
    else:
        got["titles"] = list(got.get("titles") or [])
    return got


def build_stage_calls(stage: str, questions: list[dict], idx: dict,
                      retriever=None) -> list[dict]:
    """Every call `stage` needs to make, given the records already on disk.

    Upstream stages must be complete before this is called for a downstream one;
    the runner enforces that by walking stages in order. Returns items shaped
    {"question_id", "call_index", "fields"} — the input to agents.run_calls.

    `retriever` is required for Extractor stages and solo. It is a
    RetrievalContext from src/runner.py.
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

    if prompts.role_for(stage) == "step_definer" and stage != prompts.PLAN_SUMMARY_STAGE:
        step_index = prompts.step_index_for(stage)
        assert step_index is not None
        for q in questions:
            if not _step_is_active(q, idx, step_index):
                continue
            plan = sub_questions_for(q, idx)
            history = step_history_for(q, idx, before_step=step_index)
            calls.append({
                "question_id": q["question_id"],
                "call_index": step_index,
                "fields": agents.build_step_definer_fields(
                    q["question"], plan[step_index],
                    step_number=step_index + 1, plan_steps=len(plan),
                    full_plan=plan, prior_steps=history,
                ),
                "consumer_payload_source": "parsed" if history else "initial",
                "consumer_input": {
                    "plan": plan,
                    "current_step": step_index + 1,
                    "prior_steps": history,
                },
            })
        return calls

    if prompts.role_for(stage) == "extractor":
        _require_retriever(retriever, stage)
        step_index = prompts.step_index_for(stage)
        assert step_index is not None
        for q in questions:
            if not _step_is_active(q, idx, step_index):
                continue
            task, source = _step_task(q, idx, step_index)
            if task["type"] == "aggregate":
                continue
            titles = retriever.index.search_titles(task["task"], retriever.k)
            passages = retriever.passages(titles)
            event = {
                "step": step_index + 1,
                "task_type": task["type"],
                "attempted": True,
                "query": task["task"],
                "titles": titles,
            }
            # MA-RAG extracts one note per retrieved document. We retain that
            # provenance while accelerator batching amortizes the calls.
            for document_rank, passage in enumerate(passages):
                calls.append({
                    "question_id": q["question_id"],
                    "call_index": document_rank,
                    "fields": agents.build_extractor_fields(
                        retrieval.format_passages([passage]),
                        task["task"],
                        {"target_entity": task["task"], "search_terms": [task["task"]]},
                    ),
                    "consumer_payload_source": source,
                    "consumer_input": {
                        "step_definition": task,
                        "retrieval": event,
                        "document_rank": document_rank,
                        "document_title": passage.title,
                        "document_sentences": list(passage.sentences),
                    },
                })
        return calls

    if prompts.role_for(stage) == "qa":
        step_index = prompts.step_index_for(stage)
        assert step_index is not None
        for q in questions:
            if not _step_is_active(q, idx, step_index):
                continue
            plan = sub_questions_for(q, idx)
            history = step_history_for(q, idx, before_step=step_index)
            task, task_source = _step_task(q, idx, step_index)
            retrieval_event = _retrieval_event_for_qa(
                q, idx, step_index, task
            )
            blocks = []
            consumed = []
            sources = []
            if task["type"] == "aggregate":
                for prior in history:
                    blocks.append((
                        prior["sub_question"],
                        [f"Prior step answer: {prior['answer'] or '(no answer)'}"],
                    ))
                    consumed.append({
                        "sub_question": prior["sub_question"],
                        "answer": prior["answer"],
                        "spans": [],
                        "document_title": None,
                        "document_rank": None,
                        "consumer_payload_source": prior["qa_source"],
                    })
            else:
                extractor_records = _extractor_records(
                    q["question_id"], step_index, idx
                )
                for ext_rec in extractor_records:
                    spans, source = spans_of(ext_rec)
                    spans = list(dict.fromkeys(spans))
                    sources.append(source)
                    metadata = ext_rec.get("consumer_input") or {}
                    document_title = metadata.get("document_title")
                    document_rank = metadata.get("document_rank")
                    if document_title:
                        displayed_rank = (
                            document_rank + 1
                            if isinstance(document_rank, int) else "?"
                        )
                        prompt_label = (
                            f"Document {displayed_rank}: {document_title}"
                        )
                    else:
                        prompt_label = task["task"]
                    blocks.append((prompt_label, spans))
                    consumed.append({
                        "sub_question": task["task"],
                        "answer": None,
                        "spans": spans,
                        "document_title": document_title,
                        "document_rank": document_rank,
                        "consumer_payload_source": source,
                    })
                if not extractor_records:
                    # A QA call still runs after a valid zero-result query.  Its
                    # empty evidence block and retrieval event make that state
                    # different from an aggregate step, which did not query.
                    blocks.append((task["task"], []))
                    consumed.append({
                        "sub_question": task["task"],
                        "answer": None,
                        "spans": [],
                        "document_title": None,
                        "document_rank": None,
                        "consumer_payload_source": "retrieval_empty",
                    })
            calls.append({
                "question_id": q["question_id"],
                "call_index": step_index,
                "fields": agents.build_qa_fields(
                    q["question"], blocks, sub_question=task["task"],
                    step_number=step_index + 1, plan_steps=len(plan),
                ),
                "consumer_payload_source": (
                    "+".join(dict.fromkeys(sources)) if sources else task_source
                ),
                "consumer_input": {
                    "task_type": task["type"],
                    "step_definition": task,
                    "evidence_blocks": consumed,
                    "retrieval": retrieval_event,
                },
            })
        return calls

    if stage == prompts.PLAN_SUMMARY_STAGE:
        for q in questions:
            plan = sub_questions_for(q, idx)
            history = step_history_for(q, idx)
            if not history:
                continue
            stopped = history[-1].get("success") == "no"
            if not stopped and len(history) < len(plan):
                continue
            stop_reason = "semantic_inability" if stopped else "plan_complete"
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": agents.build_plan_summary_fields(
                    q["question"], plan, history, stop_reason
                ),
                "consumer_payload_source": "parsed_history",
                "consumer_input": {
                    "plan": plan,
                    "completed_steps": history,
                    "stop_reason": stop_reason,
                },
            })
        return calls

    if stage == "solo":
        _require_retriever(retriever, stage)
        for q in questions:
            # The architecture control retrieves ONCE with the whole question and
            # reads top-k for that one call. The MA-RAG arm instead receives
            # top-k for every question-answering step; query count and passage
            # exposure are therefore reported explicitly rather than described
            # as context-budget matched. Neither arm is handed gold.
            titles = retriever.index.search_titles(q["question"], retriever.k)
            passages = retriever.passages(titles)
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": agents.build_solo_fields(
                    q["question"], retrieval.format_passages(passages)
                ),
                "consumer_payload_source": "retrieved",
                "consumer_input": {
                    "retrieval": {
                        "step": 1,
                        "task_type": "question-answering",
                        "attempted": True,
                        "query": q["question"],
                        "titles": titles,
                    },
                },
            })
        return calls

    raise KeyError(f"unknown stage {stage!r}; expected one of {prompts.PIPELINE_STAGES}")


def retrieved_titles_for(q: dict, idx: dict) -> list[str]:
    """Every passage title this question's answering stages actually saw.

    Read back out of the records rather than recomputed, so it is exactly what
    the model was shown and stays correct across a resume even if retrieval
    constants were to change between sessions.
    """
    seen: dict[str, None] = {}
    events = retrieval_events_for(q, idx)
    solo_rec = idx.get((q["question_id"], "solo", 0))
    if solo_rec is not None:
        event = ((solo_rec.get("consumer_input") or {}).get("retrieval") or {})
        if event:
            events.append(event)
    for event in events:
        for title in event.get("titles") or ():
            seen.setdefault(title, None)
    return list(seen)


def retrieval_events_for(q: dict, idx: dict) -> list[dict]:
    """One QA-persisted retrieval decision per executed reasoning step."""
    events = []
    for step_index, stage in enumerate(prompts.stages_for_role("qa")):
        rec = idx.get((q["question_id"], stage, step_index))
        if rec is None:
            continue
        event = ((rec.get("consumer_input") or {}).get("retrieval") or {})
        if event:
            events.append(dict(event))
    return events


def build_answer_records(
    questions: list[dict], idx: dict, run_id: str,
    question_manifest_sha256: str | None = None,
    retriever=None,
) -> list[dict]:
    """One scored record per question from solo or Step Definer plan summary."""
    out = []
    for q in questions:
        solo_rec = idx.get((q["question_id"], "solo", 0))
        summary_rec = idx.get((q["question_id"], prompts.PLAN_SUMMARY_STAGE, 0))
        rec = solo_rec or summary_rec
        parsed = (rec.get("parsed") or rec.get("salvaged")) if rec else None
        pred = (parsed or {}).get("answer", "")
        answer_stage = rec.get("stage") if rec else None
        plan = None if solo_rec else sub_questions_for(q, idx)
        depth_fields = {} if solo_rec else planner_depth_for(q, idx)
        history = [] if solo_rec else step_history_for(q, idx)
        retrieval_events = []
        if solo_rec:
            got = ((solo_rec.get("consumer_input") or {}).get("retrieval") or {})
            if got:
                retrieval_events = [dict(got)]
        else:
            retrieval_events = retrieval_events_for(q, idx)
        retrieved_titles = list(dict.fromkeys(
            title for event in retrieval_events for title in event.get("titles", [])
        ))
        attempted_events = [
            event for event in retrieval_events
            if bool(event.get("attempted", event.get("query") is not None))
        ]
        zero_result_events = [
            event for event in attempted_events if not event.get("titles")
        ]
        aggregate_events = [
            event for event in retrieval_events
            if event.get("task_type") == "aggregate"
        ]
        gold_titles = list(dict.fromkeys(
            (q.get("supporting_facts") or {}).get("title", [])
        ))
        retrieved_gold = [title for title in gold_titles if title in set(retrieved_titles)]
        title_recall = (
            len(retrieved_gold) / len(gold_titles) if gold_titles else 1.0
        )
        evidence_fields = {
            "evidence_status": "not_applicable",
            "evidence_predicted_labels": None,
            "evidence_gold_labels": None,
            "evidence_precision": None,
            "evidence_recall": None,
            "evidence_f1": None,
            "evidence_em": None,
            "evidence_attribution": None,
            "evidence_gold_retrieved": None,
        }
        if answer_stage != "solo":
            spans = [span for step in history for span in step.get("spans", [])]
            # The index must cover every passage the Extractor SAW, not the
            # dataset's original ten. A span copied from a retrieved passage
            # outside that ten would otherwise be unattributable and silently
            # excluded from the predicted set, flattering precision.
            sentence_index = (
                retriever.sentence_index(retrieved_titles_for(q, idx))
                if retriever is not None else q.get("sentence_index", [])
            )
            predicted_labels = evidence.predicted_evidence(spans, sentence_index)
            gold_labels = evidence.gold_evidence(q["supporting_facts"])
            scores = evidence.prf(predicted_labels, gold_labels)
            evidence_fields = {
                "evidence_status": "applicable",
                "evidence_predicted_labels": sorted(
                    ([title, sent_id] for title, sent_id in predicted_labels),
                    key=lambda x: (x[0], x[1]),
                ),
                "evidence_gold_labels": sorted(
                    ([title, sent_id] for title, sent_id in gold_labels),
                    key=lambda x: (x[0], x[1]),
                ),
                **{f"evidence_{key}": value for key, value in scores.items()},
                "evidence_attribution": evidence.attribution_stats(
                    spans, sentence_index
                ),
                # Gold that retrieval never surfaced is unreachable by any
                # Extractor, so extraction quality and retrieval coverage must be
                # separable in analysis.
                "evidence_gold_retrieved": sorted(
                    t for t in {g[0] for g in gold_labels}
                    if t in {s["title"] for s in sentence_index}
                ),
            }
        out.append({
            "run_id": run_id,
            "question_id": q["question_id"],
            "record_type": "answer",
            "question": q["question"],
            "gold_answer": q["answer"],
            "predicted_answer": pred,
            "answer_stage": answer_stage,
            "question_manifest_sha256": question_manifest_sha256,
            "consumer_payload_source": (
                "parsed" if rec and rec.get("parsed") is not None
                else "salvaged" if rec and rec.get("salvaged") is not None
                else "fallback"
            ),
            "em": exact_match(pred, q["answer"]),
            "f1": f1_score(pred, q["answer"]),
            "n_sub_questions": (
                None if solo_rec else len(plan or [])
            ),
            "plan_depth": None if solo_rec else len(plan or []),
            "executed_steps": None if solo_rec else len(history),
            "plan_steps": plan,
            **depth_fields,
            "stop_reason": (
                None if solo_rec else (
                    ((summary_rec or {}).get("consumer_input") or {}).get("stop_reason")
                )
            ),
            "summary_output": (parsed or {}).get("output") if summary_rec else None,
            "summary_score": (parsed or {}).get("score") if summary_rec else None,
            "level": q.get("level"),
            "type": q.get("type"),
            "retrieval_stratum": q.get("retrieval_stratum"),
            "hidden_gold_titles": q.get("hidden_gold_titles"),
            "gold_titles": gold_titles,
            "retrieved_titles": retrieved_titles,
            "retrieved_gold_titles": retrieved_gold,
            "retrieval_gold_title_recall": title_recall,
            "retrieval_all_gold": len(retrieved_gold) == len(gold_titles),
            "retrieval_query_count": len(attempted_events),
            "retrieval_zero_result_query_count": len(zero_result_events),
            "retrieval_aggregate_step_count": len(aggregate_events),
            "retrieval_passage_exposures": sum(
                len(event.get("titles", [])) for event in retrieval_events
            ),
            "retrieval_unique_titles": len(retrieved_titles),
            "retrieval_events": retrieval_events,
            **evidence_fields,
        })
    return out
