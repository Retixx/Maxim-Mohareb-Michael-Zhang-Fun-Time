# Frozen-Upstream Passage-Plus-Spans Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed Kaggle T4 replay that regenerates frozen-trace QA and plan summaries under `spans_plus_passages` and `passages_only`, then emits the preregistered mechanism and final-answer comparisons without rerunning retrieval or single-hop.

**Architecture:** Keep scientific reconstruction, validation, treatment assembly, state propagation, and scoring in a CPU-testable core module. Put model loading, the 21-call reproduction sentinel, condition-aware batching, durable output publication, and CLI handling in a thin runtime module. Both modules read committed evidence directly; neither invokes live pipeline orchestration or retrieval.

**Tech Stack:** Python 3.11, repository `src` prompt/model/parsing/metrics helpers, PyTorch/Transformers/bitsandbytes for GPU execution, `unittest`/pytest for CPU verification, JSON/JSONL with SHA-256 provenance.

---

## File Structure

- Create `clean_room/passage_replay_core.py`: immutable artifact loader, trace projections, prompt reconstruction, passage joins, both treatment builders, treated state and summary rebuilding, survival headline, paired metrics, fingerprints, and interpretation.
- Create `clean_room/passage_replay.py`: `--audit-only`/`--execute` CLI, GPU/model checks, version warnings, exact sentinel, condition-aware batch execution, no-thinking guard, resumable partial calls file, and completion-certificate-last publication.
- Create `tests/test_passage_replay.py`: synthetic unit fixtures plus one cached integration audit over `evidence/gate_c_1.7b/`.
- Do not modify `src/`, `SPEC.md`, config, manifests, prompts, retrieval, or committed evidence.

## Binding Independent-Review Amendments

These requirements restore details already approved in the design and supersede any looser wording later in this plan:

- [ ] Persist one successful `model_load` record, 21 `phase=reproduction_sentinel` calls with six sentinel batch certificates, and 1,038 `phase=scored` calls with 262 scored batch certificates. Final validation therefore expects 1,059 agent calls, 268 batch certificates, 328 answers, and one model-load record. No sentinel row may be counted as a scored treatment call.
- [ ] Resume is certificate-gated. A call row without its valid trailing batch certificate is an orphan and cannot enter treated state. Reject duplicate/unexpected keys. Validate condition, fingerprint, canonical treatment batch hash, exact membership/order, batch ID/ordinal/member index, parent key/hash, source/treatment message hashes, rendered-chat hash, model/revision/precision, and required output fields. For an orphan/partial batch, regenerate the complete original batch, compare every preexisting output field byte-for-byte, append only missing members, then append the certificate and `fsync`; any mismatch aborts.
- [ ] Hash `tok.chat_template` and a deterministic tokenizer snapshot: save the loaded tokenizer into a repository-local temporary snapshot directory, SHA-256 every regular file in sorted relative-path order, and record the per-file map plus a canonical aggregate hash. Record a SHA-256 of `models.render_chat()` for every source/sentinel/treatment call.
- [ ] Before generation or persistence, render and tokenize every dynamic prompt with no truncation and require `prompt_tokens + ceiling <= recorded context_window_tokens`, using ceiling 96 for QA and 128 for summaries. Persist measured prompt tokens, ceiling, total, and context window; cover retrieval-backed QA, aggregate QA, and summaries.
- [ ] The root replay fingerprint binds both treatment schemas, passage header, formatter source/hash, frozen trace, old/new message hashes, rendered-chat hashes, QA/summary template versions and hashes, history/grounding/stop/finalizer policy identifiers, model commit, tokenizer snapshot hashes, exact quantization census, package/GPU identity, ceilings, batch manifests, `enable_thinking=false`, replay code hash/commit, and scorer/bootstrap settings. Distinct condition fingerprints bind their own IDs and batches. Output hashes remain outside these fingerprints.
- [ ] Prove prompt isolation with poison mutations to gold answer/title/supporting facts/stratum and single-arm all-gold flags: messages and hashes must not change. Add a static/runtime guard forbidding `pipeline.build_stage_calls`, retriever construction, title search, solo call construction/generation, and any retrieval import in both replay modules.
- [ ] QA reporting compares the reverse usable treated candidate against the reverse usable source-MA QA candidate for overall n=200, both-gold n=128, and both strata. Also emit last-executed and best-intermediate oracle diagnostics, per-step parse/salvage, success disagreement directions and IDs, grounding, literal answer survival, prompt tokens, earliest-new-no sensitivity, and explicit original-stop/new-success right-censoring. New success values never change the primary frozen call graph.
- [ ] Every paired comparison reports `a_f1`, `b_f1`, `delta_f1_points`, `a_em`, `b_em`, `delta_em_points`, bootstrap, McNemar, and wins/losses/ties. Never extrapolate `passages_only` outside the frozen 128 subset; its strata are the subset's 95 hidden-bridge and 33 fully-named questions.
- [ ] Test every preregistered interpretation branch and the inclusive two-point boundary with numeric tolerance. Show the SPEC §15.5 thresholds beside diagnostics, while forbidding `PASS_GATE_C`, `GO`, and “§4.3 is repairable.”
- [ ] Strengthen immutable/output audits: exact source commit, subset strata 95/33, baseline-only membership, no generated solo calls, complete manifest/calls/answers/summary/meta schema, lineage on sentinel/treatment calls, exact certificate coverage, 200/128 condition answers, final artifact hashes, and `meta.json` published last as the sole completion certificate.
- [ ] Sentinel hard equality is exactly raw output, parsed payload, salvaged payload, prompt tokens, output tokens, and generated-sequence tokens. Record `parse_status`, source/replay batch telemetry, and think-tag checks, but do not make package strings or `parse_status` additional equivalence gates.
- [ ] Replay recorded sentence lists faithfully, including an empty list if the pinned source ever contains one; source hashes and relational joins—not an added nonempty-sentence rule—define validity.

### Task 1: Immutable Source Bundle and Gold-Free Trace Projection

**Files:**
- Create: `clean_room/passage_replay_core.py`
- Create: `tests/test_passage_replay.py`

- [ ] **Step 1: Write failing source-integrity tests**

Add tests that load the real bundle once, assert every committed identity, and prove a mutated hash fails before call construction:

```python
class PassageReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = core.load_source_bundle(ROOT / "evidence/gate_c_1.7b")

    def test_committed_sources_match_all_pins_and_cohorts(self):
        source = self.source
        self.assertEqual(source.question_ids_sha256, core.FULL_IDS_SHA256)
        self.assertEqual(len(source.question_ids), 200)
        self.assertEqual(Counter(q.stratum for q in source.scoring.values()), {
            "hidden_bridge": 160, "fully_named": 40,
        })
        self.assertEqual(len(source.both_gold_ids), 128)
        self.assertEqual(core.ordered_ids_sha256(source.both_gold_ids),
                         core.BOTH_GOLD_IDS_SHA256)
        self.assertEqual(sum(source.single_answers[q].get("retrieval_all_gold") is True
                             for q in source.question_ids), 108)

    def test_hash_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            target = Path(raw)
            for name in core.SOURCE_SHA256:
                shutil.copy2(ROOT / "evidence/gate_c_1.7b" / name, target / name)
            path = target / core.BASELINE_META
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(core.IntegrityError, "SHA-256"):
                core.load_source_bundle(target)
```

- [ ] **Step 2: Run the focused test and verify it fails**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
```

Expected: import failure for `clean_room.passage_replay_core`.

- [ ] **Step 3: Implement immutable constants, projections, and loading**

Define exact source constants and projections that physically separate prompt-safe fields from scoring fields:

```python
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
SOURCE_EXPERIMENT_FINGERPRINT = "68a765d4bdaf49bd760bdb7866c09e25e65bae9ddbb5f2dc30b3a856968baa2a"
FULL_IDS_SHA256 = "f8c3f16458340cb0bc74aa827e3b51528ba351963a46dba456ac4e68ad20f7d7"
BOTH_GOLD_IDS_SHA256 = "cf7a9dbf8bfc48dfc42c40e5caaae68ad47e9ba26ef5a6536a27f83fada3508a"

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

class IntegrityError(RuntimeError):
    pass
```

`load_source_bundle()` populates the exact `ReplaySource` projection above. It must hash all four files before parsing, check both internal `jsonl_sha256` fields, enforce identical ordered IDs/questions/golds/model/4-bit precision/experiment fingerprint across arms, derive `both_gold_ids` only from baseline `retrieval_all_gold is True`, and never put gold or stratum fields in `FrozenQuestion`.

- [ ] **Step 4: Run tests and commit the source contract**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
git add clean_room/passage_replay_core.py tests/test_passage_replay.py
git commit -m "test: pin frozen passage replay sources"
```

Expected: the two source tests pass.

### Task 2: Exact Trace Audit, Prompt Reconstruction, and Context Conditions

**Files:**
- Modify: `clean_room/passage_replay_core.py`
- Modify: `tests/test_passage_replay.py`

- [ ] **Step 1: Write failing trace and context tests**

```python
def test_committed_trace_reconstructs_all_source_prompts(self):
    audit = core.audit_source(self.source)
    self.assertEqual(audit.qa_stage_counts, {
        "qa": 200, "qa_step2": 187, "qa_step3": 32,
        "qa_step4": 7, "qa_step5": 1,
    })
    self.assertEqual(audit.qa_prompt_hash_matches, 427)
    self.assertEqual(audit.summary_prompt_hash_matches, 200)
    self.assertEqual(audit.question_answering_calls, 426)
    self.assertEqual(audit.aggregate_calls, 1)
    self.assertEqual(audit.extractor_joins, 4260)

def test_context_conditions_have_exact_text_and_duplication(self):
    original = "1. Document 1: Alpha\n   - Selected sentence."
    join = core.PassageJoin(("Alpha", "Beta"),
                            (("Selected sentence.", "Remaining sentence."),
                             ("Other sentence.",)))
    plus = core.render_treated_evidence(original, join, core.SPANS_PLUS_PASSAGES)
    only = core.render_treated_evidence(original, join, core.PASSAGES_ONLY)
    passage = "[1] Alpha: Selected sentence. Remaining sentence.\n[2] Beta: Other sentence."
    self.assertEqual(plus, original + "\n\nRetrieved passages for the current step:\n" + passage)
    self.assertEqual(only, "Retrieved passages for the current step:\n" + passage)
    self.assertEqual(plus.count("Selected sentence."), 2)
    self.assertEqual(only.count("Selected sentence."), 1)
```

Add mutations for missing/duplicate rank, swapped title, changed task, and changed retrieval title; each must raise `IntegrityError`.

- [ ] **Step 2: Verify the new tests fail**

Expected: missing `audit_source`, `PassageJoin`, and treatment builders.

- [ ] **Step 3: Implement exact source reconstruction and passage rendering**

```python
PASSAGE_HEADER = "Retrieved passages for the current step:"
SPANS_PLUS_PASSAGES = "spans_plus_passages"
PASSAGES_ONLY = "passages_only"

@dataclass(frozen=True)
class PassageJoin:
    titles: tuple[str, ...]
    sentence_lists: tuple[tuple[str, ...], ...]

def reconstruct_source_qa_fields(source: ReplaySource, qa: dict) -> dict:
    frozen = source.questions[qa["question_id"]]
    ci = qa["consumer_input"]
    blocks = []
    for block in ci["evidence_blocks"]:
        spans = list(block.get("prompt_spans") or ())
        if not spans:
            continue
        label = (block["sub_question"] if ci["task_type"] == "aggregate" else
                 f"Document {int(block['document_rank']) + 1}: {block['document_title']}")
        blocks.append((label, spans))
    fields = agents.build_qa_fields(
        frozen.question, blocks, sub_question=ci["step_definition"]["task"],
        step_number=int(qa["call_index"]) + 1, plan_steps=len(frozen.plan),
    )
    if agents.rendered_prompt_sha256(prompts.build_messages("qa", **fields)) != qa["rendered_prompt_sha256"]:
        raise IntegrityError("source QA message hash mismatch")
    return fields

def render_treated_evidence(original: str, join: PassageJoin, condition: str) -> str:
    passages = prompts.format_paragraphs(list(join.titles),
                                         [list(x) for x in join.sentence_lists])
    block = f"{PASSAGE_HEADER}\n{passages}"
    if condition == SPANS_PLUS_PASSAGES:
        return f"{original}\n\n{block}"
    if condition == PASSAGES_ONLY:
        return block
    raise ValueError(f"unknown replay condition {condition!r}")
```

Implement `join_recorded_passages()` with ranks exactly `0..9` and title/task/retrieval equality, replaying each pinned sentence list faithfully. Implement `source_scored_batches()` from `record_type=batch`, `phase=scored` certificates. Implement `condition_batches()` so plus preserves source certificates exactly and only filters each stage's source order to the frozen 128 IDs then chunks by four.

- [ ] **Step 4: Add exact population and batch assertions**

Assert plus is 427 QA/200 summary/108+50 batches; only is 283 QA/128 summary/72+32 batches; combined is 1,038 calls/262 certificates; every condition key occurs exactly once.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
git add clean_room/passage_replay_core.py tests/test_passage_replay.py
git commit -m "feat: reconstruct frozen replay treatments"
```

### Task 3: Treated QA State, Aggregate Propagation, and Coherent Summaries

**Files:**
- Modify: `clean_room/passage_replay_core.py`
- Modify: `tests/test_passage_replay.py`

- [ ] **Step 1: Write failing downstream-coherence tests**

Use synthetic treated records to prove later routing stays frozen while new QA state reaches the aggregate and summary:

```python
def test_aggregate_uses_treated_grounded_answers_without_retrieval(self):
    treated = self.synthetic_trace_with_answer("NewBridge", grounded=True)
    call = core.build_treated_qa_call(treated.source, treated.aggregate_source,
                                      core.SPANS_PLUS_PASSAGES, treated.qa_index)
    self.assertEqual(call["consumer_input"]["step_definition"],
                     treated.aggregate_source["consumer_input"]["step_definition"])
    self.assertIn("Prior step answer: NewBridge", call["fields"]["evidence"])
    self.assertNotIn("OldBridge", call["fields"]["evidence"])
    self.assertFalse(call["consumer_input"]["retrieval"]["attempted"])

def test_treated_history_and_summary_are_coherent(self):
    history = core.rebuild_treated_history(self.source, "qid", self.treated_index,
                                           core.SPANS_PLUS_PASSAGES)
    self.assertEqual([x["answer"] for x in history], ["NewBridge", "Final"])
    self.assertEqual(history[0]["task"], self.frozen_task)
    call = core.build_treated_summary_call(self.source, "qid", history,
                                           core.SPANS_PLUS_PASSAGES)
    rendered = prompts.build_messages("plan_summary", **call["fields"])[1]["content"]
    self.assertIn("NewBridge", rendered)
    self.assertNotIn("OldBridge", rendered)
    self.assertEqual(call["consumer_input"]["stop_reason"], self.source_stop_reason)
```

Add a local `synthetic_trace_with_answer(answer, grounded)` fixture builder that returns a `SimpleNamespace` containing a minimal `ReplaySource`, one frozen question-answering source call, one aggregate source call, a treated QA index, the frozen task, and source stop reason. The fixture's old answer is `OldBridge`; the treated record's answer is the supplied value. Cover parsed, salvaged, and empty QA payloads; grounded and ungrounded aggregate answers; new `success=no` without call-graph truncation; and earliest-new-no sensitivity.

- [ ] **Step 2: Run tests and verify they fail**

Expected: missing treated-state helpers.

- [ ] **Step 3: Implement treated calls and history construction**

```python
def effective_payload(record: dict | None) -> tuple[dict, str]:
    if record and record.get("parsed") is not None:
        return dict(record["parsed"]), "parsed"
    if record and record.get("salvaged") is not None:
        return dict(record["salvaged"]), "salvaged"
    return {}, "fallback"

def build_treated_qa_call(source: ReplaySource, qa: dict, condition: str,
                          treated_qa_index: Mapping[tuple[str, str, int], dict]) -> dict:
    fields = reconstruct_source_qa_fields(source, qa)
    ci = copy.deepcopy(qa["consumer_input"])
    if ci["task_type"] == "question-answering":
        join = join_recorded_passages(source, qa)
        fields["evidence"] = render_treated_evidence(fields["evidence"], join, condition)
    else:
        history = rebuild_treated_history(source, qa["question_id"], treated_qa_index,
                                          condition, before_step=qa["call_index"])
        blocks = [(x["sub_question"], [f"Prior step answer: {x['answer']}"])
                  for x in history if x["answer_grounded"] is True]
        frozen = source.questions[qa["question_id"]]
        fields = agents.build_qa_fields(
            frozen.question, blocks, sub_question=ci["step_definition"]["task"],
            step_number=qa["call_index"] + 1, plan_steps=len(frozen.plan),
        )
    source_key = (qa["question_id"], qa["stage"], int(qa["call_index"]))
    return {
        "question_id": qa["question_id"],
        "stage": qa["stage"],
        "call_index": int(qa["call_index"]),
        "condition": condition,
        "fields": fields,
        "consumer_payload_source": "frozen_trace_treatment",
        "consumer_input": {
            **ci,
            "replay_parent": {
                "key": list(source_key),
                "record_sha256": canonical_json_sha256(qa),
            },
        },
    }
```

`rebuild_treated_history()` starts from frozen summary `completed_steps`, replaces only QA-derived answer/success/rating/source/grounding, and calls `pipeline._answer_is_grounded()` against actual visible texts: prompt spans plus passages for plus, passages for only, and treated grounded prior answers for aggregate. It never feeds new grounding into frozen tasks or retrieval.

- [ ] **Step 4: Implement summary rebuilding and final precedence**

```python
def build_treated_summary_call(source: ReplaySource, qid: str, history: list[dict],
                               condition: str) -> dict:
    frozen = source.questions[qid]
    fields = agents.build_plan_summary_fields(
        frozen.question, list(frozen.plan), history, frozen.stop_reason,
    )
    return {
        "question_id": qid, "call_index": 0, "fields": fields,
        "condition": condition, "consumer_payload_source": "treated_history",
        "consumer_input": {
            "plan": list(frozen.plan), "completed_steps": history,
            "stop_reason": frozen.stop_reason,
        },
    }

def resolve_treated_answer(summary_record: dict | None, history: list[dict]) -> dict:
    return pipeline.final_answer_for(None, summary_record, history)
```

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
git add clean_room/passage_replay_core.py tests/test_passage_replay.py
git commit -m "feat: propagate treated QA state to summaries"
```

### Task 4: Survival Headline, Paired Metrics, and Preregistered Interpretation

**Files:**
- Modify: `clean_room/passage_replay_core.py`
- Modify: `tests/test_passage_replay.py`

- [ ] **Step 1: Write failing survival and scoring tests**

```python
def test_survival_headline_reproduces_artifact(self):
    self.assertEqual(core.extractor_survival_headline(self.source), {
        "n": 128,
        "raw_producer_present": 84,
        "normalized_qa_prompt_present": 52,
        "raw_and_prompt_present": 50,
        "absent_from_raw_and_prompt": 42,
        "present_raw_removed_by_normalizer": 34,
        "absent_raw_recovered_by_normalizer": 2,
        "method": "contiguous normalize_answer token phrase",
    })

def test_near_single_interpretation_is_predeclared(self):
    text = core.interpret_scores(single_f1=.4213, plus_f1=.4090,
                                 only_f1=.4110)
    self.assertIn("extraction is not contributing a net advantage over raw retrieval", text)
    self.assertNotIn("§4.3 is repairable", text)
    self.assertTrue(core.practically_equivalent(.4213, .4013))
    self.assertFalse(core.practically_equivalent(.4213, .4012))
```

Add deterministic paired fixtures covering A-minus-B direction, wins/losses/ties, exact McNemar discordance, reverse-usable QA candidate, incomplete-ID rejection, all three passage-condition contrasts, both strata, and the 128 subset.

- [ ] **Step 2: Run tests and verify they fail**

Expected: missing survival/report functions.

- [ ] **Step 3: Implement literal-answer survival**

Use `metrics.normalize_answer` and contiguous normalized token phrases. For each baseline-defined both-gold question, search raw producer spans and exact `prompt_spans` delivered to QA, count all four flows, and assert the result equals the predeclared object before returning it.

- [ ] **Step 4: Implement paired metrics and report schema**

```python
def paired_comparison(a: Mapping[str, dict], b: Mapping[str, dict],
                      ids: Sequence[str]) -> dict:
    if set(ids) - a.keys() or set(ids) - b.keys():
        raise IntegrityError("paired comparison has incomplete condition IDs")
    diffs = {qid: 100.0 * (float(a[qid]["f1"]) - float(b[qid]["f1"]))
             for qid in ids}
    bootstrap = metrics.joint_paired_bootstrap(
        {"f1_points": diffs}, n_resamples=10_000, seed=20260807,
    )["f1_points"]
    mcnemar = metrics.exact_mcnemar(
        [int(a[qid]["em"]) for qid in ids],
        [int(b[qid]["em"]) for qid in ids],
    )
    values = list(diffs.values())
    return {
        "n": len(ids), "a_f1": mean(float(a[q]["f1"]) for q in ids),
        "b_f1": mean(float(b[q]["f1"]) for q in ids),
        "delta_f1_points": mean(values),
        "a_em": mean(float(a[q]["em"]) for q in ids),
        "b_em": mean(float(b[q]["em"]) for q in ids),
        "delta_em_points": 100.0 * mean(
            float(a[q]["em"]) - float(b[q]["em"]) for q in ids
        ),
        "bootstrap": bootstrap,
        "mcnemar": mcnemar, "wins": sum(x > 0 for x in values),
        "losses": sum(x < 0 for x in values), "ties": sum(x == 0 for x in values),
    }
```

`score_replay()` emits the survival object at top level, QA reverse-usable and truncation-sensitivity endpoints, final comparisons versus source MA and single overall/subset/strata, the full three-way subset matrix, gap recovery, final source distribution, fixed-trace drift, call/token cost, and label `Gate-C-comparable fixed-trace diagnostic`. It never emits `PASS_GATE_C` or `GO`.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
git add clean_room/passage_replay_core.py tests/test_passage_replay.py
git commit -m "feat: score frozen passage replay"
```

### Task 5: Treatment Fingerprints, Version Warnings, and Reproduction Sentinel

**Files:**
- Create: `clean_room/passage_replay.py`
- Modify: `tests/test_passage_replay.py`

- [ ] **Step 1: Write failing runtime-contract tests**

```python
def test_package_mismatch_warns_but_sentinel_is_hard_gate(self):
    warnings = runtime.version_warnings(
        {"torch": "2.10.0+cu128", "transformers": "5.14.1"},
        {"torch": "2.11.0+cu128", "transformers": "5.15.0"},
    )
    self.assertEqual(len(warnings), 2)
    matching = self.fake_sentinel_records()
    runtime.compare_sentinel(matching.source, matching.replay)
    matching.replay[0]["output_tokens"] += 1
    with self.assertRaisesRegex(core.IntegrityError, "sentinel"):
        runtime.compare_sentinel(matching.source, matching.replay)

def test_sentinel_selects_exactly_21_source_members(self):
    batches = runtime.sentinel_batches(self.source)
    self.assertEqual([len(x.members) for x in batches], [4, 4, 4, 4, 1, 4])
    self.assertEqual(sum(len(x.members) for x in batches), 21)
```

Add `fake_sentinel_records()` as a local fixture returning two deep-copied lists of 21 minimal records populated with every field in `SENTINEL_FIELDS`; mutate only the replay copy in negative tests. Test both think tags case-insensitively, wrong commit/census/GPU/batch, and fingerprint changes for every semantic axis listed in the design.

- [ ] **Step 2: Run runtime tests and verify they fail**

Expected: import failure for `clean_room.passage_replay`.

- [ ] **Step 3: Implement runtime identity and fingerprint helpers**

Use `models.load_model()` with model and tokenizer revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, `models.validate_loaded_precision()`, `models.resolved_revision_metadata()`, `runner._library_versions()`, and `runner._gpu_metadata()`. Require Tesla T4/7.5 and exact source census; record source Python as `None` because it was not captured. Package version differences only populate `environment_warnings`.

The canonical fingerprint payload includes source hashes/commit/fingerprint, condition, IDs, frozen trace hash, renderer/template hashes, model/tokenizer/quant identity, decoding, condition-specific batches, `enable_thinking=False`, code commit/hash, and scorer seed. Output hashes are excluded.

- [ ] **Step 4: Implement the exact 21-call sentinel**

Select the first scored certificate for `qa`, `qa_step2`, `qa_step3`, `qa_step4`, `qa_step5`, and `plan_summary`. Run each original batch separately through `agents.run_calls(..., batch_size=4, return_batch_record=True)`. Buffer all 21 outputs in memory and require equality for:

```python
SENTINEL_FIELDS = (
    "raw_output", "parsed", "salvaged",
    "prompt_tokens", "output_tokens", "generated_sequence_tokens",
)
```

Record `parse_status` but do not add it to hard equality. Reject `<think>` or `</think>` before persistence. Only a complete match authorizes scored conditions, regardless of package warnings. After all 21 match in memory, persist the successful model-load record, sentinel calls, and their six certificates before starting scored conditions.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
git add clean_room/passage_replay.py tests/test_passage_replay.py
git commit -m "feat: gate replay on empirical sentinel"
```

### Task 6: Condition-Aware GPU Execution and Completion-Certificate-Last Outputs

**Files:**
- Modify: `clean_room/passage_replay.py`
- Modify: `tests/test_passage_replay.py`

- [ ] **Step 1: Write failing batch, no-thinking, and atomic-output tests**

Inject a fake `agents.run_calls` callback and assert 1,038 calls/262 certificates, exact condition keys, state-dependent summary scheduling, and abort-before-write on think output. Inject failures during audit, sentinel, generation, context validation, and rename; no finalized `meta.json` may remain.

- [ ] **Step 2: Run tests and verify they fail**

Expected: missing executor and output publisher.

- [ ] **Step 3: Implement condition-aware certified batches**

Implement `execute_batch_plan()` rather than calling `runner._run_stage`, whose resume key omits condition. Use keys `(condition, question_id, stage, call_index)` and batch IDs like `spans_plus_passages:qa:000000`. For each batch:

1. build calls directly from source records;
2. render and store canonical message-object SHA plus `models.render_chat()` SHA;
3. call `agents.run_calls()` with the original concrete stage and batch size 4;
4. reject any think tag before writing or state propagation;
5. augment records with condition, treatment fingerprint, parent key/hash, source/treatment prompt hashes, and phase;
6. write missing call records, then the batch certificate, then `fsync`.

Use `runner.JsonlStore` on `calls.jsonl.partial`. Apply the binding certificate-gated resume rules above; fingerprint-and-key equality alone is never sufficient.

- [ ] **Step 4: Implement CLI and atomic publication**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=Path("evidence/gate_c_1.7b"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/passage_plus_spans_replay"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser
```

`--audit-only` performs every CPU integrity check, builds all call/batch manifests, computes the survival headline, and prints 200/128 cohorts, 427/200 source calls, 4,260 joins, 627/627 hashes, 627/411 condition calls, and 262 batches without loading model weights.

`--execute` runs audit, loads one model, records version warnings, passes and persists the sentinel, runs plus QA stages then plus summaries, runs only QA stages then only summaries, creates 328 answer records, scores, and validates the full counts in the binding amendments. Write all data as `.partial`; close/fsync/hash/rename data artifacts; write `meta.json` last with hashes as the sole completion certificate. Never overwrite a complete result or publish final meta after failure.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
git add clean_room/passage_replay.py tests/test_passage_replay.py
git commit -m "feat: execute and certify passage replay"
```

### Task 7: Full Artifact Audit and Repository Verification

**Files:**
- Modify only if verification exposes a scoped replay defect: `clean_room/passage_replay_core.py`, `clean_room/passage_replay.py`, `tests/test_passage_replay.py`

- [ ] **Step 1: Run the real audit-only path**

```bash
.venv/bin/python -X utf8 clean_room/passage_replay.py --audit-only
```

Expected output includes:

```text
source questions: 200 (hidden_bridge=160, fully_named=40)
both-gold subset: 128 (hidden_bridge=95, fully_named=33)
source calls: qa=427 plan_summary=200 extractor=4260
source prompt hashes: 627/627
spans_plus_passages: 627 calls, 158 batches
passages_only: 411 calls, 104 batches
scored total: 1038 calls, 262 batches
answer survival: 84/128 -> 52/128
AUDIT_PASS
```

- [ ] **Step 2: Run focused and full verification**

```bash
.venv/bin/python -X utf8 -m pytest tests/test_passage_replay.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check clean_room/passage_replay.py clean_room/passage_replay_core.py tests/test_passage_replay.py
.venv/bin/python -m compileall -q clean_room/passage_replay.py clean_room/passage_replay_core.py tests/test_passage_replay.py
git diff --check
git diff --name-only b6b1a01..HEAD
```

Expected: all tests and lint pass; the implementation diff contains only the two clean-room modules, one test file, and this plan.

- [ ] **Step 3: Prove the local host cannot be mistaken for the scored environment**

```bash
.venv/bin/python -X utf8 clean_room/passage_replay.py --execute
```

Expected here: fail before model load with a clear Tesla T4/CUDA requirement and no finalized `meta.json`.

- [ ] **Step 4: Commit final verification fixes, if any**

```bash
git add clean_room/passage_replay.py clean_room/passage_replay_core.py tests/test_passage_replay.py
git commit -m "test: verify frozen passage replay harness"
```

Skip this commit when verification requires no changes.

### Task 8: Kaggle Execution Handoff

**Files:**
- No source changes
- Runtime output: `analysis/passage_plus_spans_replay/`

- [ ] **Step 1: Run on Kaggle T4 from the final harness commit**

From the existing branch—never a parallel branch—fetch the final harness commit, reset the index/worktree only as part of the user-required clean Kaggle checkout line-ending normalization, install compatible dependencies without exact package-string aborts, and run:

```bash
git fetch origin multihop-vs-single-hop-rag-bug-fix
git checkout multihop-vs-single-hop-rag-bug-fix
git pull --ff-only origin multihop-vs-single-hop-rag-bug-fix
git rm --cached -r .
git reset --hard
# No --upgrade and no exact version gate: retain Kaggle's PyTorch/CUDA image,
# install only missing direct packages, record all versions, then trust only
# the byte-exact 21-call sentinel.
python -m pip install transformers bitsandbytes datasets accelerate PyYAML numpy scipy huggingface-hub
python -u clean_room/passage_replay.py \
  --source-dir evidence/gate_c_1.7b \
  --output-dir analysis/passage_plus_spans_replay \
  --execute
```

Expected: package drift is printed as warnings; the 21-call sentinel must match exactly before any of 1,038 scored generations begins.

- [ ] **Step 2: Validate the completion certificate and report**

Require final `meta.json`, matching calls/answers hashes, 1,038 scored calls, 262 scored batch certificates, 328 answer rows, no think tags, the top-level survival object, and all preregistered comparisons. If the sentinel fails, report the exact member/field mismatch and stop; do not weaken or bypass it.

- [ ] **Step 3: Package results without changing the source experiment**

```bash
tar -czf passage_plus_spans_replay.tar.gz -C analysis passage_plus_spans_replay
sha256sum passage_plus_spans_replay.tar.gz
ssh codex-guest@144.217.94.114 download passage_plus_spans_replay.tar.gz > passage_plus_spans_replay.tar.gz
```

Keep the archive in the repository for SSH handoff. Do not edit SPEC §4.3 or claim `PASS_GATE_C`; report the result as `Gate-C-comparable fixed-trace diagnostic` using the preregistered interpretation table.
