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
    """Load `n` HotpotQA distractor-setting questions.

    SPEC §3: distractor setting, dev split, no retriever — the 10 provided
    paragraphs (2 gold + 8 distractors) are used directly.

    Final execution supplies a frozen ordered manifest plus the exact pilot/dev
    exclusions. Development smoke tests may omit the manifest and sample only
    from explicitly approved development seeds.
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
        qn = retrieval.norm(row["question"])
        hidden = [t for t in gold_titles if retrieval.norm(t) not in qn]
        return {
            "question_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "level": row.get("level"),
            "type": row.get("type"),
            # Gold evidence labels are analysis-only and never enter a prompt.
            "supporting_facts": row.get("supporting_facts"),
            # Prespecified stratifier. `fully_named` questions name every page
            # they need, so a second retrieval hop has nothing to resolve and
            # gains exactly 0.000 — they are the built-in control that the gain
            # on `hidden_bridge` comes from resolving a withheld name rather than
            # from reading more text. Computed from the question string and gold
            # titles only, so it is fixed before any model runs.
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


def spans_of(rec: dict | None) -> tuple[list[str], str]:
    """The spans a consumer should read from an Extractor record, and their source.

    SPEC §13b.1: prefer the validated payload, else whatever was salvageable from
    a failed call. The call still counts as a parse failure — this only stops
    usable spans being discarded along with a malformed container.
    """
    if rec:
        if rec.get("parsed") is not None:
            return (rec["parsed"] or {}).get("spans", []), "parsed"
        if rec.get("salvaged") is not None:
            return (rec["salvaged"] or {}).get("spans", []), "salvaged"
    return [], "fallback"


def hop1_titles_for(q: dict, retriever, sub_question: str) -> list[str]:
    """Hop-1 passages for one sub-question: the ORIGINAL question's top-k.

    Deliberately the original question, not the sub-question. Sub-question
    queries were measured retrieving WORSE than the whole question (recall@10
    0.743 vs 0.794): HotpotQA questions carry their lexical signal in the entity
    names, and splitting them discards signal without adding any. The second hop
    is where decomposition earns its keep, not the first.
    """
    return retriever.index.search_titles(q["question"], retriever.k)


def build_stage_calls(stage: str, questions: list[dict], idx: dict,
                      retriever=None) -> list[dict]:
    """Every call `stage` needs to make, given the records already on disk.

    Upstream stages must be complete before this is called for a downstream one;
    the runner enforces that by walking stages in order. Returns items shaped
    {"question_id", "call_index", "fields"} — the input to agents.run_calls.

    `retriever` is required for every stage that reads passages (extractor,
    extractor_hop2, solo). It is a RetrievalContext from src/runner.py.
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
        _require_retriever(retriever, stage)
        for q in questions:
            titles = hop1_titles_for(q, retriever, q["question"])
            # Only the first `hop1` reach this pass; the remainder of the budget
            # is held back for the follow-up query. If no follow-up fires,
            # build_answer_records still sees the held-back passages because
            # extractor_hop2 falls back to them (see below).
            passages = retriever.passages(titles[: retriever.hop1])
            rendered = retrieval.format_passages(passages)
            for i, sq in enumerate(sub_questions_for(q, idx)):
                spec_rec = idx.get((q["question_id"], "step_definer", i))
                # SPEC §13b.1: prefer the validated payload; fall back to whatever
                # was salvageable from a failed call rather than discarding it.
                # The call still counts as a parse failure — this only stops a
                # good search_terms list being thrown away with an empty entity.
                spec = None
                source = "fallback"
                if spec_rec:
                    if spec_rec.get("parsed") is not None:
                        spec, source = spec_rec["parsed"], "parsed"
                    elif spec_rec.get("salvaged") is not None:
                        spec, source = spec_rec["salvaged"], "salvaged"
                calls.append({
                    "question_id": q["question_id"],
                    "call_index": i,
                    "fields": agents.build_extractor_fields(rendered, sq, spec),
                    "consumer_payload_source": source,
                    "consumer_input": {
                        "step_definition": spec,
                        "retrieval": {
                            "hop": 1,
                            "query": q["question"],
                            "titles": [p.title for p in passages],
                        },
                    },
                })
        return calls

    if stage == prompts.EXTRACTOR_HOP2:
        _require_retriever(retriever, stage)
        for q in questions:
            titles = hop1_titles_for(q, retriever, q["question"])
            hop1 = titles[: retriever.hop1]
            for i, sq in enumerate(sub_questions_for(q, idx)):
                spec_rec = idx.get((q["question_id"], "step_definer", i))
                # Same step definition as hop 1. The two passes must differ ONLY
                # in which passages they see, or the comparison between them
                # confounds retrieval with prompt content.
                spec = None
                if spec_rec:
                    spec = spec_rec.get("parsed") or spec_rec.get("salvaged")
                spans, source = spans_of(idx.get((q["question_id"], "extractor", i)))
                hop = retrieval.retrieve(
                    retriever.index, q["question"], k=retriever.k,
                    hop1=retriever.hop1, spans=spans, hop1_titles=hop1,
                )
                if not hop["fired"]:
                    # Nothing worth a second query — every name the Extractor
                    # found is either in the question or already retrieved. Spend
                    # the held-back budget on more depth from hop 1 instead, so a
                    # fully-named question is never punished for the machinery.
                    passages = retriever.passages(titles[retriever.hop1: retriever.k])
                else:
                    passages = retriever.passages(hop["titles"])
                calls.append({
                    "question_id": q["question_id"],
                    "call_index": i,
                    "fields": agents.build_extractor_fields(
                        retrieval.format_passages(passages), sq, spec
                    ),
                    "consumer_payload_source": source,
                    "consumer_input": {
                        "retrieval": {
                            "hop": 2,
                            "fired": hop["fired"],
                            "query": hop["query"],
                            "titles": [p.title for p in passages],
                        },
                    },
                })
        return calls

    if stage == "qa":
        for q in questions:
            # NOT named `evidence`: that would bind a local of the same name as
            # the imported `evidence` module for the whole function, turning any
            # future module use in build_stage_calls into an UnboundLocalError.
            blocks = []
            consumed = []
            for i, sq in enumerate(sub_questions_for(q, idx)):
                # Both Extractor passes feed QA. Order is hop-1 then hop-2 and
                # duplicates are dropped: hop-2 often re-selects a sentence hop-1
                # already found, and repeating it in the evidence block would
                # make QA read the same fact twice.
                per_hop = []
                for st in ("extractor", prompts.EXTRACTOR_HOP2):
                    rec = idx.get((q["question_id"], st, i))
                    # A stage with no record at all did not run — that is not a
                    # "fallback" payload and must not be reported as one, or a
                    # single-hop run would look like it degraded.
                    if rec is not None:
                        per_hop.append(spans_of(rec))
                merged = list(dict.fromkeys(s for spans, _ in per_hop for s in spans))
                sources = list(dict.fromkeys(src for _, src in per_hop)) or ["fallback"]
                blocks.append((sq, merged))
                consumed.append({
                    "sub_question": sq,
                    "spans": merged,
                    "hop_spans": [len(spans) for spans, _ in per_hop],
                    "consumer_payload_source": "+".join(sources),
                })
            consumed_sources = {
                item["consumer_payload_source"] for item in consumed
            }
            calls.append({
                "question_id": q["question_id"],
                "call_index": 0,
                "fields": agents.build_qa_fields(q["question"], blocks),
                "consumer_payload_source": (
                    next(iter(consumed_sources))
                    if len(consumed_sources) == 1 else "mixed"
                ),
                "consumer_input": {"evidence_blocks": consumed},
            })
        return calls

    if stage == "solo":
        _require_retriever(retriever, stage)
        for q in questions:
            # The architecture control retrieves ONCE with the whole question and
            # reads the same k passages the pipeline gets in total. It is not
            # handed gold. Giving it the dataset's 10 paragraphs, as SPEC §4a
            # originally did, made it an oracle rather than a baseline — which is
            # why it beat the four-agent pipeline by 9.2 EM.
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
                        "hop": 1, "query": q["question"], "titles": titles,
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
    for stage in ("extractor", prompts.EXTRACTOR_HOP2, "solo"):
        for i in range(agents.MAX_SUB_QUESTIONS):
            rec = idx.get((q["question_id"], stage, i))
            if not rec:
                continue
            got = ((rec.get("consumer_input") or {}).get("retrieval") or {})
            for t in got.get("titles") or ():
                seen.setdefault(t, None)
    return list(seen)


def build_answer_records(
    questions: list[dict], idx: dict, run_id: str,
    question_manifest_sha256: str | None = None,
    retriever=None,
) -> list[dict]:
    """One scored record per question, from the QA stage's outputs (SPEC §7)."""
    out = []
    for q in questions:
        rec = (
            idx.get((q["question_id"], "qa", 0))
            or idx.get((q["question_id"], "solo", 0))
        )
        parsed = (rec.get("parsed") or rec.get("salvaged")) if rec else None
        pred = (parsed or {}).get("answer", "")
        answer_stage = rec.get("stage") if rec else None
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
            spans = []
            for i, _ in enumerate(sub_questions_for(q, idx)):
                for stage in ("extractor", prompts.EXTRACTOR_HOP2):
                    got, _src = spans_of(idx.get((q["question_id"], stage, i)))
                    spans.extend(got)
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
                None if rec and rec.get("stage") == "solo"
                else len(sub_questions_for(q, idx))
            ),
            "level": q.get("level"),
            "type": q.get("type"),
            "retrieval_stratum": q.get("retrieval_stratum"),
            "hidden_gold_titles": q.get("hidden_gold_titles"),
            **evidence_fields,
        })
    return out
