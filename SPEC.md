# Experiment contract: role-aware capacity allocation in multi-agent RAG

This is the authoritative scientific contract for the final campaign.
config/experiment.yaml is its machine-readable counterpart. If code, analysis,
or another active document conflicts with either source, stop and resolve the
conflict before running an experiment.

## 1. MA-RAG reference and local scope

The architecture follows the plan-driven workflow in Nguyen, Chin, and Tai,
“MA-RAG: Multi-Agent Retrieval-Augmented Generation” (arXiv:2505.20096) and its
public implementation. Reference MA-RAG is not a double-hop-only design. Its
Planner emits a question-dependent plan, the Step Definer routes each plan step,
and execution continues for as many steps as that plan requires. The paper and
public examples show different plan depths; they do not define two as the
architecture ceiling.

This experiment preserves that control flow:

1. Planner emits an ordered plan.
2. For every active plan step, Step Definer chooses question-answering or
   aggregate and emits a self-contained task.
3. A question-answering step retrieves top-k passages, invokes Extractor once
   for every retrieved document, and asks QA for step feedback.
4. An aggregate step skips retrieval and Extractor, then asks QA to combine
   prior step answers.
5. The QA result is appended to state. Execution stops when the plan completes
   or QA returns success=no.
6. The Step Definer treatment is reused for a required plan-summary call. A
   usable parsed or salvaged summary supplies the scored answer; if that call
   degrades, the last usable intermediate QA answer is retained with explicit
   fallback provenance instead of being destroyed.

The four conceptual agents are Planner, Step Definer, Extractor, and QA.
Repeated reasoning steps are repeated invocations of those agents, not new
independently configured agents.

The local implementation intentionally differs from reference MA-RAG in three
ways:

- Reference MA-RAG leaves plan depth contextual. Within this campaign, Planner
  still emits a question-dependent plan of one through five steps; five is an
  explicit edge-resource ceiling. The emitted depth and any clamp are logged.
  Five is an experiment limit, not a claim about MA-RAG.
- Reference MA-RAG evaluates dense inner-product retrieval with a FAISS index.
  This campaign uses deterministic sparse BM25 so retrieval can be reproduced
  cheaply on a controlled corpus. Results compare agent capacity allocations
  under this retriever and do not establish parity with dense retrieval.
- Reference MA-RAG searches a much larger knowledge base. This campaign searches
  72,094 HotpotQA validation passages. It is a controlled retrieval experiment,
  not a Wikipedia-scale benchmark.

These adaptations are the edge-systems variables being studied. They do not
change the multi-agent, stateful, variable-depth reasoning topology.

## 2. Research questions

Across a 23× parameter range (0.6B to 14B, Qwen3 family), can role-specific
model capacity and quantization reduce resident memory and accelerator work
while retaining answer quality in a fully executed multi-agent RAG workflow?

The prespecified questions are:

1. Which conceptual roles are most sensitive to moving the 8B model to 8-bit?
2. Which roles are most sensitive to swapping 8B FP16 for the 4B FP16 sibling
   (near-memory-matched to 8B 8-bit)?
3. At a near memory match, do quantization and a smaller sibling impose different
   role-level costs?
4. Which roles benefit most from scaling up to 14B?
5. How do uniform multi-agent 8B FP16 and a competitive one-call 8B FP16
   retrieval baseline compare overall and on hidden-bridge questions?
6. Which guarded role allocation minimizes deduplicated resident model memory
   while satisfying the frozen F1 constraint?

The selected mixed allocation is in-sample and exploratory. It is a deployment
demonstration and interaction check, not confirmatory evidence of superiority.

## 3. Frozen questions and corpus

### 3.1 Accuracy cohort

All accuracy arms use the same ordered sample once:

- dataset: hotpotqa/hotpot_qa
- source configuration: distractor
- split: validation
- dataset revision: 1908d6afbbead072334abe2965f91bd2709910ab
- n: 1,500
- seed: 20,260,805
- manifest: config/manifests/final_n1500_seed20260805.json
- ordered-ID SHA-256:
  5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af
- exclusion-ID SHA-256:
  a5cfacb84fa9a48217f3206a095706a6d48802bd244151c72f2eef08372c00a8
- frozen retrieval strata: 1,097 hidden_bridge and 403 fully_named

The exclusion set contains the earlier 3,000 design-pilot IDs and all
prompt-development IDs. Every run preserves question order and canonical batch
membership. Failed questions remain in the denominator; no replacement or
resampling is allowed.

Before model loading, the runner verifies dataset identity and revision,
manifest count and hashes, uniqueness, exclusion disjointness, ID existence,
and exact loaded order.

### 3.2 Retrieval corpus

The retriever indexes the deterministic first-occurrence union of validation
passages from the HotpotQA distractor and fullwiki configurations, in that order:

- passages: 72,094
- corpus SHA-256:
  931425de7b123e3081ed63387c8d591a8aba4cf872d2cf47144e924260d92b73
- required selected/auxiliary gold-title coverage: 1.0
- required selected/auxiliary supporting-sentence coverage: 1.0

The configuration name fullwiki does not mean that this repository indexes the
full Wikipedia dump. It contributes passages exposed by the HotpotQA
configuration. The corpus is global across the validation split and contains
the target pages, but the question-to-context assignment, supporting-fact
labels, and gold answers never enter a model prompt.

Corpus order is load-bearing. Some shared titles have different sentence
splits, and distractor must win so supporting-fact sentence indices stay valid.
The runner checks passage count, content fingerprint, algorithm fingerprint,
query-policy fingerprint, unique titles, and gold-title and supporting-sentence
reachability before loading a model.

### 3.3 Exposure caveat

Questions, corpus passages, and gold labels all originate from the same pinned
validation split. The model is forced to discover passages through retrieval,
but this is still a controlled, target-reachable corpus with evaluation-set
content in the index. Report it as controlled open-corpus retrieval. Do not
generalize its recall, accuracy, or cost to an unseen corpus or full Wikipedia.

## 4. Executed architecture

### 4.1 Planner

Planner receives only the original question and emits:

    {"analysis": "...", "sub_questions": ["...", "..."]}

The plan must contain the shortest sufficient sequence from one through five
steps. Later steps may depend on earlier answers. If Planner output cannot be
parsed or salvaged, execution degrades deterministically to a one-step plan
containing the original question. Outputs longer than five are clamped and
logged.

### 4.2 Repeated Step Definer routing

For plan step i, Step Definer receives:

- the original question;
- the full ordered plan;
- the current sub-question and step position; and
- the append-only history of earlier tasks, answers, success flags, ratings, and
  evidence-grounding status.

Only evidence-grounded prior answers are rendered as facts for later Step
Definer calls. Unsupported fallback guesses remain logged but are withheld.

It emits exactly one route:

    {"type": "question-answering", "task": "..."}

or:

    {"type": "aggregate", "task": "..."}

The task is self-contained. A malformed route degrades to
question-answering on the current plan sub-question; there is no new generation.

### 4.3 Question-answering route

Every question-answering step issues exactly one BM25 query and exposes at most
10 documents. Step 1 uses the original-question top 10. A later step with at
least one evidence-grounded prior answer uses the resolved Step Definer task
plus grounded prior answers as one self-contained query and gives that query the
full top 10. A later step without grounded state falls back to the original
question top 10. Unsupported guesses never enter a query. Gold labels, answers,
supporting facts, and retrieval strata never enter query construction.

Extractor is invoked separately for every retrieved document, preserving document
rank and title in the call record. Its raw parsed output is retained, but only
spans that map unambiguously to exact source sentences are eligible for QA.
Unique fragments expand to their exact sentence; whole-passage/multi-sentence
echoes, ambiguous fragments, non-source text, duplicates, and spans beyond the
three-sentence cap are rejected and logged. This is deterministic post-hoc
normalization, not another generation.

QA receives the current task and only non-empty normalized document blocks;
empty documents remain in telemetry but do not pad the prompt. Cross-document
duplicate sentences are shown once. If no evidence survives, QA sees one
`(no evidence collected)` marker. QA emits:

    {"analysis": "...", "answer": "...", "success": "yes", "rating": 8}

QA uses retrieved evidence first. When that evidence is insufficient, it gives
its best short answer from general knowledge and emits `success=no` only when it
cannot produce a usable short answer. This matches the reference fallback
behavior. Such a candidate is marked evidence-grounded only when its normalized
answer occurs in evidence actually supplied to QA or in a grounded prior answer.
Unsupported candidates may be summarized but never become retrieval facts.

Each active question-answering step therefore has one retrieval event, exactly
one query, up to ten Extractor generations, and one QA generation. Retrieval depth is determined by plan depth and routing, not
by a global hop count.

### 4.4 Aggregate route

Aggregate is used when the current answer can be computed from prior step state.
It performs no retrieval and invokes no Extractor. QA receives the earlier
evidence-grounded answers only and emits the same feedback schema. Unsupported
guesses remain logged but are not presented as facts. Aggregate steps remain
agentic reasoning steps while avoiding unnecessary corpus and generation work.

### 4.5 Semantic stop and finalization

Every QA payload is appended to question state. A later step is active only when
all earlier steps exist and the latest success field is not no. The loop stops
for either:

- plan_complete: every planned step produced QA feedback; or
- semantic_inability: the latest QA feedback has success=no.

After either stop, the Step Definer model/prompt treatment performs a distinct
plan-summary call over the original question, full plan, completed history, and
stop reason. It emits:

    {"output": "Successful", "answer": "...", "score": 8}

A usable parsed summary answer takes precedence, followed by a usable salvaged
summary answer. If neither exists, the scored prediction deterministically
falls back to the most recent usable non-sentinel QA answer. The summary call is
still always attempted, `answer_stage` remains `plan_summary`, and
`final_answer_source` records `summary_parsed`, `summary_salvaged`,
`qa_fallback`, or `none`. A fallback is not claimed correct merely because it is
non-empty; when no usable candidate exists the empty prediction remains in the
denominator.

## 5. Retrieval and baseline contracts

The active retriever is a sparse, vectorized BM25 index with fixed tokenization,
Lucene-style IDF, stable corpus-order tie-breaking, and unique passage titles.
The configured multi-agent policy is
`original_question_first_then_full_grounded_task_v1`: the original question owns
the first top 10; a later evidence-grounded Step Definer task plus grounded
answers owns that step's full top 10; and an ungrounded later step falls back to
the original question. There is exactly one query per question-answering step
and no learned bridge-query helper. This policy was frozen before the pilot and
was not chosen from pilot/final answer F1.

### 5.1 Measured retrieval headroom (2026-08-07)

The policy above is only worth its cost if a second query can reach evidence the
first cannot. That is a property of the corpus and is measurable without any
model. Measured on the pooled 72,094-passage corpus, k=10, n=600, by
`clean_room/retrieval_headroom.py`:

| Stratum | n | SINGLE both-gold | ORACLE two-pass | Headroom |
|---|---:|---:|---:|---:|
| hidden_bridge | 475 | 0.4716 | 0.8863 | **+0.4147** |
| fully_named | 125 | 0.8320 | 0.8880 | +0.0560 |

Headroom is large and falls where the design predicts: on hidden_bridge, which
is 1,097 of the frozen 1,500. `single_any_recall` on hidden_bridge is 0.9684 —
one gold passage is nearly always retrieved by the first query; it is the
second, hidden one that is missed. That is exactly what the second hop exists to
reach. On fully_named there is effectively no headroom and decomposition can
only lose.

ORACLE is an upper bound: it is handed the hidden gold title for free, which a
real pipeline must resolve by reading. The gap between a live pipeline and
0.8863 is the cost of bridge-entity resolution, and it is a reportable result of
this experiment rather than an error term.

### 5.2 Known deviation of the implementation from this contract

The implementation does not currently satisfy §5. `_retrieval_decision`
(`src/pipeline.py`) builds the grounded follow-up query as
`[task["task"], *grounded_answers]` and **omits the original question**. The
contract text above ("a later evidence-grounded Step Definer task plus grounded
answers owns that step's full top 10") describes intended behaviour that the
code does not implement in the form that preserves anchor signal.

Measured consequence, using the real planner sub-questions from
`baseline_qwen2.5-1.5b_n750_seed7` (n=730, unstratified, k=10):

| Arm | both-gold recall@10 |
|---|---:|
| single query (question only) | 0.495 |
| decomposed, equal read budget | **0.296** |
| decomposed, double read budget | 0.410 |

Decomposition moves retrieval backwards. Combined with the follow-up firing gate
(§14, BUG-2), this is the mechanism by which the multi-agent system loses to the
one-call control on multi-hop questions, and why that loss did not shrink across
a 0.5B→7B sweep: the deficit is created upstream of the reader, so reader
capacity cannot recover it.

No accuracy claim comparing multi-agent to `single_fp16` may be published from
any artifact produced before BUG-1 and BUG-2 are fixed and the query-policy
fingerprint is re-frozen. All existing `results/` artifacts predate the fix and
originate from commit `fa661f5`, which is not an ancestor of the current
lineage; they are diagnostic only.

The one-call control, single_fp16:

- uses one 8B FP16 generation;
- queries the same BM25 corpus once with the original question;
- reads the top 10 returned passages;
- receives no decomposition, state loop, Extractor, or extra query; and
- uses the same answer normalization and immutable model revision.

The control and multi-agent system each expose at most 10 passages per
question-answering step, but they do not have the same total passage, query, or
model-call budget. Multi-agent exposure and query count vary with plan depth and
routing. The multi-vs-single result is therefore a system-level cost-benefit
contrast, not a total-context-matched retrieval comparison. Report query count,
passage exposures, unique titles, model calls, prompt/output tokens, F1, EM,
memory, and timing whenever that contrast appears.

Answer records include retrieval events, retrieved and gold titles, gold-title
recall, initial-query and grounded-follow-up recall, grounded-follow-up firing
rate, incremental task gold recall, grounding rate, Extractor normalization, QA
evidence filtering, plan depth, executed steps, stop reason, summary status, and
final-answer provenance. Gold fields are attached only after generation.

Repaired artifacts use experiment schema `open_corpus_marag_v3`. Resume,
pilot-gate, and analysis paths reject v1/v2 artifacts, stale prompt hashes, a
stale query-policy fingerprint, a non-original initial query, a grounded
follow-up budget other than k=10, or removal of the evidence-grounding guard.

Questions are prespecified as hidden_bridge or fully_named from question text
and gold titles. The frozen final cohort contains exactly 1,097 hidden_bridge
and 403 fully_named questions. The runner and analyzer reject any other counts;
report accuracy and retrieval diagnostics overall and by both strata.

## 6. Models, treatments, and run matrix

All models are from the Qwen3 family (April 2025). Qwen3 unifies base and
instruct capabilities in a single set of weights with switchable thinking mode.
All generation uses non-thinking mode (enable_thinking=False) to produce
deterministic structured output without chain-of-thought token overhead.

Immutable model revisions (to be pinned before campaign launch):

| Alias | Model | Parameters | Revision |
|---|---|---|---|
| large | Qwen3-14B | 14B | TBD |
| base | Qwen3-8B | 8B | TBD |
| mid | Qwen3-4B | 4B | TBD |
| small | Qwen3-1.7B | 1.7B | TBD |
| tiny | Qwen3-0.6B | 0.6B | TBD |

The five sizes span a 23× parameter range within one architecture and training
recipe, isolating capacity from training-data or architecture confounds.

The static matrix contains 32 run IDs:

| Family | Count |
|---|---:|
| Uniform four-agent 8B FP16 reference | 1 |
| One-role 8B 8-bit ablations | 4 |
| One-role 8B 4-bit ablations | 4 |
| One-role 4B FP16 ablations (near-memory-matched to 8-bit) | 4 |
| One-role 1.7B FP16 ablations | 4 |
| One-role 0.6B FP16 floor ablations | 4 |
| One-role 14B FP16 upward ablations | 4 |
| Uniform controls (8-bit, 4-bit, 4B, 1.7B, 0.6B, 14B) | 6 |
| One-call 8B FP16 architecture control | 1 |

All repeated Step Definer, Extractor, and QA stage labels mirror their
conceptual role treatment. The plan-summary stage mirrors Step Definer.
Repeated stages cannot be configured independently.

The primary treatment family compares one-role 8B 8-bit ablations with the
corresponding 4B FP16 ablations. These are near-memory-matched: 8B 8-bit
(~8.5 GiB) vs 4B FP16 (~8 GiB). The 4-bit family is mandatory secondary
evidence. The 1.7B and 0.6B runs measure the lower capacity/compliance
boundary and feed a guarded exploratory selector. The 14B runs measure the
marginal value of additional capacity per role.

## 7. Exploratory role allocation

The selector declares seven candidates for each of four conceptual roles:

    {14B FP16, 8B FP16, 8B 8-bit, 8B 4-bit, 4B FP16, 1.7B FP16, 0.6B FP16}

The full universe is 7^4 = 2,401 allocations. A role may use 0.6B only when its
corresponding one-role tiny ablation has a question-clustered 95% lower bound of
at least 0.90 for strict protocol success. Failure removes tiny only for that
role.

Eligible candidates must also satisfy the paired-bootstrap lower-bound
noninferiority constraint: predicted F1 may be no more than 1.0 point below the
uniform 8B FP16 reference. The selector minimizes deduplicated concurrent model
footprint, then prefers higher predicted F1, fewer distinct configurations, and
lexical allocation ID.

The selector writes a trace and derived executable config. If the selected
allocation matches a static run, reuse it. Otherwise execute
ma_optimized_exploratory once. Because selection and evaluation use the same
1,500 questions, report the result only as in-sample exploratory evidence.

Accuracy reuse and timing reuse are separate decisions. If selection matches a
non-tiny static arm, its validated static accuracy and timing artifacts may be
reused. If it matches a tiny static arm, reuse the accuracy artifact but do not
pretend a timing artifact exists: tiny arms are absent from the shared timing
matrix. Selection materialization writes timing.selected_execution_run_id and
timing.selected_system_timing_required=true into the derived config. Campaign
planning and the runner admit only that frozen selected execution beyond the
static timing IDs, and strict analysis verifies the selection-artifact and run-
config hashes recorded in frozen_allocation. This narrowly authorizes one
post-selection timing run on the reserved A100 without broadening the
prespecified static timing matrix.

## 8. Outcomes and statistics

The primary outcome is HotpotQA token F1 under the repository normalization.
Exact Match is co-reported.

For each role r:

    Q_loss_r = F1_baseline - F1_role_8bit
    M_loss_r = F1_baseline - F1_role_mid
    axis_contrast_r = F1_role_8bit - F1_role_mid
    L_gain_r = F1_role_large - F1_baseline

Use 10,000 paired question-level bootstrap replicates. Holm-adjust the four
primary role contrasts. Report point estimates, paired 95% intervals, adjusted
p-values, and direction. Keep 4-bit, 1.7B, 0.6B floor, 14B upward, evidence,
parsing, retrieval, plan-depth, and semantic-stop analyses secondary or
exploratory. Never stratify
a primary treatment comparison by that arm's own emitted or executed plan depth:
depth is post-treatment. A baseline-defined depth label may be applied unchanged
to every arm only as a prespecified secondary diagnostic.

Repeated calls are not independent observations. Cluster role diagnostics by
question. Resample complete batches for timing intervals. Sort question IDs
before seeded resampling so write order cannot affect results.

## 8a. Multiplicity — one primary test, everything else descriptive

Ported from SPEC v2.1 §5f (branch `results-n3000`), which the `no-bs` merge
overwrote. The prior experiment's terms are translated to this one: F1 primary
rather than EM, 8-bit rather than 4-bit. **This is the most likely statistical
objection to the paper.**

Counting honestly: 4 roles on the quantization axis, 4 on the mid-size axis,
4 on each of 3 additional size axes (small, tiny, large), 4 axis contrasts,
across F1 / EM / ev-F1 — this is a large family of comparisons. A referee who
checks will call uncorrected per-role findings a multiple-comparisons artifact,
and on that framing they would be right.

The fix is to designate one pre-registered primary test and demote the rest,
rather than correcting dozens of tests into oblivion:

- **PRIMARY — confirmatory, one test, no correction needed.** The pooled contrast
  between format-heavy roles (Step Definer, Extractor) and knowledge-heavy roles
  (Planner, QA). It was pre-registered before the confirmatory data existed, it is
  a *single* number, and it is better powered than any per-role test because it
  pools two roles per side. Report on the §8 primary outcome (F1), with EM
  co-reported.
- **SECONDARY — pre-registered, Holm-corrected.** The four role contrasts of §8
  (8B 8-bit vs 4B FP16). Holm–Bonferroni is uniformly more powerful than
  Bonferroni and assumes no independence — these tests share a baseline and are
  positively correlated, which Holm tolerates and Šidák does not.
- **DESCRIPTIVE — no significance claims at all.** Every per-role number across
  all size/quantization tiers, the upward 14B gains, the full ranking, and the
  Spearman correlation between axes. Report point estimates with intervals and
  describe them as estimates. **Do not write "significant" next to a per-role
  result.** "The Extractor is the only role whose interval excludes zero
  uncorrected" is true, informative, and not a significance claim.

**Consequence for how the paper is written.** Lead with the format-heavy vs
knowledge-heavy contrast, because that is the one claim the design licenses. The
four-way ranking becomes a figure and a paragraph of description, not the
headline. This is a presentation change, not a re-analysis — every number stays.

**Axis-contrast variance.** `size cost − quantization cost` is a difference of two
differences, but both are paired against the same baseline on the same questions,
so they are positively correlated and Var(A−B) = Var(A) + Var(B) − 2Cov(A,B) is
materially less than 2·Var. **Compute the paired bootstrap on the per-question
contrast directly; never estimate it by adding the two arms' variances.** The
latter overstates the interval and would bury a real effect.

## 9. Memory and edge-efficiency contract

Estimated resident model footprints (must be re-measured on A100 before
campaign launch):

| Configuration | MiB (est.) |
|---|---:|
| 14B FP16 | ~28,000 |
| 8B FP16 | ~16,000 |
| 8B 8-bit | ~8,500 |
| 4B FP16 | ~8,000 |
| 8B 4-bit | ~5,000 |
| 1.7B FP16 | ~3,400 |
| 0.6B FP16 | ~1,200 |

The 8B 8-bit and 4B FP16 treatments are near-memory-matched, not equal.

The primary memory quantity is deduplicated concurrent model-footprint MiB:
charge parameters plus buffers once for each distinct model/revision/precision
configuration used by any conceptual role. Also report isolated role-service
footprint, sequential stage-major peak VRAM, allocated/reserved peaks,
activation-inclusive peaks, and cold load/swap time.

The edge claim is limited to resource behavior measured by this experiment:
role allocation, lower resident model footprint, fewer calls for shorter or
aggregate plans, and relative accelerator throughput. The five-step ceiling is
part of this resource envelope. Actual phone, embedded GPU, CPU, NPU, power,
thermal, and wall-time performance require measurements on those devices.

## 10. Generation and orchestration controls

- Greedy generation only; no sampling.
- Prompt templates, parsers, and token caps are frozen across treatments.
- Frozen generation ceilings are Planner 160, Step Definer 160, Extractor 320,
  QA 96, plan summary 128, and solo 48 new tokens. The complete mapping is part
  of the experiment fingerprint.
- Before any generation, loaded parameter tensors must match requested precision:
  FP16 has zero recognized quantized parameters; Q8 and Q4 each require at least
  50% of nominal parameters in the matching bitsandbytes tensor type and reject
  parameters from the other quantized type. The validated census is logged.
- No size-, precision-, or role-ablation-specific prompt tuning.
- No grammar-constrained decoding.
- No generation retry or regeneration after parse/protocol failure.
- Salvage may expose usable fields without changing the original failure status.
- Deterministic degraded state is propagated when no usable payload exists.
- A production OOM is fatal; batch size is never reduced for one arm.

Logical execution is question-stateful, but physical execution is stage-major.
The runner walks Planner, repeated Step Definer/Extractor/QA rounds, plan summary,
or the solo control in canonical dependency order. At each stage it batches all
currently active homogeneous calls across questions and, for Extractor, across
documents. Inactive plan steps and Extractor work for aggregate routes produce
no calls and do not load a model.

Only one model configuration is resident at a time; consecutive stages with the
same fingerprint may reuse it. This scheduling amortizes SLM inference and bounds
sequential VRAM without changing per-question MA-RAG state dependencies.

Production batch_size and min_batch_size are both 32. Canonical batch IDs,
ordered membership, model/prompt/config/manifest/retrieval fingerprints, and
execution identity participate in resume validation.

## 11. Pilot and launch gate

Final-question timing and accuracy execution are prohibited until a frozen
excluded-data pilot produces a persisted GO certificate.

The pilot:

1. uses the 200 unique IDs in
   config/manifests/pilot_excluded200_seed20260806.json (seed 20,260,806),
   drawn from the committed exclusion set and disjoint from final, timing, and
   warm-up cohorts, with exactly 160 hidden_bridge and 40 fully_named questions;
2. runs baseline followed by single_fp16 on one locked worker;
3. uses the same corpus, prompts, architecture, models, batch policy, and
   experiment fingerprint as the final campaign; and
4. reports paired F1/EM, parse/protocol success, retrieval recall, query/exposure
   counts, plan depth, stop reasons, and both retrieval strata.

GO requires both:

    F1_baseline - F1_single_fp16 >= 0 overall
    F1_baseline - F1_single_fp16 >= 0 on hidden_bridge

The committed pilot manifest has file SHA-256
975210805c382788bb39c800266ae22a88cc526e0626bd8a0106c35d316a8bb1 and
ordered-ID SHA-256
f8c3f16458340cb0bc74aa827e3b51528ba351963a46dba456ac4e68ad20f7d7. Its
deterministic sampler replaced malformed ID 5ae61bfd5542992663a4f261 with the
next ordered eligible nonsampled exclusion, 5ae622495542995703ce8b20, because
the source annotation points to supporting sentence 902 on a five-sentence gold
page. The replacement and reason are recorded in data_quality_replacements; it
prevents an annotation defect from making gold-sentence reachability impossible
for every system.

The content-addressed GO certificate is analysis/pilot_gate.json. Missing,
incomplete, stale, or hash-mismatched pilot artifacts mean STOP.
Failure of either quality condition means STOP and architecture diagnosis before
any final IDs are consumed. The threshold must not be relaxed after seeing the
pilot.

The pilot path is implemented: scripts/run_campaign.py --kind pilot enforces
one-worker baseline-then-single order and invokes scripts/check_pilot.py after
execution. The checker recomputes metrics, validates artifact, cohort,
environment, retrieval, prompt, stratum, and content hashes, and writes the
content-addressed decision. Timing and accuracy execute modes require the GO
artifact to be committed and unchanged. No target-A100 pilot or GO result is
asserted yet; pilot execution and all downstream GPU runs remain pending.

Before any A100 phase, scripts/prefetch_assets.py downloads the pinned dataset
configurations and model snapshots, validates all manifest bytes plus the corpus
count and fingerprints, checks free space, and writes logs/prefetch_report.json.
Its --offline-verify-only mode forbids network access and proves the cache is
complete. scripts/a100_entrypoint.py is the fail-closed production wrapper for
prepare, pilot, accuracy, and timing. It requires an actual NVIDIA A100;
prepare additionally requires clean committed source and immutable container
identity, performs offline asset verification and the full test suite, and
writes the environment lock for commit before pilot execution.

After GO:

1. validate the immutable environment lock on every worker;
2. preflight every active stage shape at batch 32 on excluded data;
3. rerun the complete timing matrix under this architecture;
4. execute all 32 static accuracy arms once;
5. freeze and commit selector artifacts;
6. execute and time a distinct selected allocation if required; and
7. run strict final analysis.

## 12. A100 systems proxy

Timing uses two complete repetitions of the frozen 128-question excluded cohort
on one reserved, uncontended A100 after a separate excluded warm-up. Timing
artifacts never enter F1, EM, selector fitting, or accuracy estimates. Tiny arms
are excluded from the shared timing matrix and must pass their own fail-closed
preflight before scoring. The sole exception is a selected tiny allocation:
after the selector and derived config are committed, it requires one separately
labelled post-selection timing artifact before an exploratory throughput number
is reported.

The primary systems metric is steady-state end-to-end service inverse
throughput: seconds per excluded A100 timing question, relative to uniform
multi-agent 8B FP16 at 1.00x. It includes deterministic retrieval, routing, and
state construction plus prompt construction/rendering, tokenization and H2D,
generation, decoding, parsing/salvage, and protocol accounting. It excludes
model loading and durable JSONL logging, which are recorded separately.

Each timing summary also nests a generation-only inverse-throughput estimate
from raw generation batch_wall_s. Report service_wall_s, orchestration time,
the non-generation service component, questions/second, token-normalized
throughput, model load time, GPU identity, driver/runtime, and the
container/software lock. Generation-only timing is diagnostic and must not be
substituted for the end-to-end service primary.

These measurements are an A100 execution proxy for comparative scheduling and
model-allocation cost. They are not literal edge-device timing, energy, or
thermal results. Do not infer a device-specific speedup without running that
device.

## 13. Required records and claim boundaries

Every call record must retain run/question/stage/call/batch IDs, conceptual and
prompt roles, model and revision, precision/quantization, prompt and experiment
fingerprints, rendered-prompt hash, raw output, parsed and salvaged payloads,
format/protocol status, tokens, timing, retrieval provenance where applicable,
and retry_count=0.

Every answer record must retain the plan and emitted/clamped depth, executed
steps, stop reason, summary result, prediction/gold/F1/EM, retrieval stratum and
events, title recall, passage exposures, and evidence attribution. Run metadata
must retain corpus/config/environment hashes, canonical batch topology, stage
activity and calls, memory, load time, timing mode, completion, and artifact
hashes.

Supported claims are limited to paired role sensitivity, answer quality,
retrieval behavior on the controlled corpus, deduplicated memory, executed call
and token cost, and relative A100 throughput.

Do not claim:

- exact equivalence to reference MA-RAG retrieval;
- that reference MA-RAG has a five-step limit;
- Wikipedia-scale or unseen-corpus retrieval quality;
- pure parameter-count or pure bit-width causality (five sizes share one
  architecture and training recipe, but they are not controlled for compute);
- exact memory equality between 8-bit and 4B;
- confirmatory superiority of the selected mixed allocation;
- literal edge-device timing, energy, or thermals from A100 results;
- probability calibration; or
- independent final evidence from the pilot or historical runs.

RUNBOOK.md is the normative execution order. Any silent change in sample,
corpus, plan ceiling, routing, retrieval k, model/revision, precision, prompt,
parser, batch membership, or artifact path invalidates the affected comparison.

## 14. Open defect register

Status as of 2026-08-07 at commit `e1d8072`. Every entry below is either
reproduced from a measurement or read directly from source. Launch is blocked
until BUG-1 through BUG-5 are closed.

### BUG-1 — grounded follow-up query discards the original question (BLOCKER)

`src/pipeline.py`, `_retrieval_decision`: the follow-up query is built as
`[task["task"], *grounded_answers]`. The original question is omitted. The
ORACLE arm in §5.1 reaches 0.8863 precisely by retaining it.

Measured: both-gold recall@10 falls 0.495 → 0.296 at equal read budget on real
planner sub-questions (n=730). Even at double budget it only reaches 0.410.

Fix: include `q["question"]` in `query_parts`. Bump `QUERY_POLICY` and re-freeze
the retrieval fingerprint.

### BUG-2 — second hop is gated behind a verbatim-substring test (BLOCKER)

`grounded_followup_fired = bool(step_index > 0 and grounded_answers)`, where
`_grounded_prior_answers` admits only answers for which `_answer_is_grounded`
finds the answer as a literal token phrase inside Extractor spans. When that
test fails the code falls back to the anchor branch and **re-issues the same
query as step 1** — identical top-10, zero new evidence.

Consequence: on an unmeasured but likely large fraction of questions the
pipeline silently degenerates to single-hop retrieval plus roughly four times
the model calls. This is the primary suspect for the multi-agent-loses-to-
single-hop result, and it explains why that gap did not close across a 0.5B→7B
sweep: the deficit is created upstream of the reader.

Fix: when no grounded answer exists, still issue `anchor + task` rather than
bare `anchor`, so the second hop always contributes a distinct query. Retain
`answer_grounded` as telemetry, not as a fire/no-fire switch. Log the firing
rate per stratum.

### BUG-3 — retrieval replaces instead of unioning (BLOCKER)

Each branch performs exactly one `search_titles(...)`. The `components` list
already declares both `original_question_anchor` and `grounded_step_task`, but
exactly one is ever `attempted` — the structure anticipates a union that was
never implemented. ORACLE reaches 0.8863 by unioning anchor top-k/2 with
follow-up top-k/2.

Fix: retrieve both components, dedupe, cap at k. Populate both entries.

### BUG-4 — follow-up query is built from the digest, not the passages

The bridge entity must survive retrieval → Extractor compression → QA answer →
verbatim grounding check → query string. Four lossy stages. Meanwhile
`single_any_recall` on hidden_bridge is 0.9684: the passage naming the bridge
entity is already retrieved at hop 1 in ~97% of cases, and
`retriever.passages(titles)` is available at the call site but never consulted
for query construction.

Fix (phase 2, after BUG-1..3): derive follow-up candidates from hop-1 retrieved
titles and passage text. This is what closes the remaining distance to 0.8863.

### BUG-5 — every model revision is `TBD` (BLOCKER)

`config/experiment.yaml` pins `TBD` for all five Qwen3 sizes.
`scripts/prefetch_assets.py` and `a100_entrypoint.py prepare` fail closed on
unpinned revisions. No campaign can start.

Fix: resolve and commit immutable Hugging Face commit SHAs for Qwen3-14B, -8B,
-4B, -1.7B, -0.6B.

### BUG-6 — hash pins are not portable across line-ending platforms

There is no `.gitattributes` and `core.autocrlf=true` is common on Windows. The
frozen manifests are pinned by raw file SHA-256, so a Windows checkout expands
LF→CRLF (`final_n1500_seed20260805.json`: 153,098 → 157,681 bytes) and three
integrity tests fail against correct data. The committed blobs are intact; only
the working tree diverges.

Fix: add `.gitattributes` with `*.json text eol=lf` (or `* -text`).

### BUG-7 — all existing `results/` artifacts are off-lineage

Every artifact in `results/` was produced at commit `fa661f5`, which is **not**
an ancestor of the current lineage, and records `prompt_version: v5` against the
current `PROMPT_VERSION = "marag-v3"`. They also predate BUG-1..3. They are
diagnostic only and must not appear in any published comparison.

### BUG-9 — the Qwen3 migration did not propagate to the test suite (BLOCKER)

Commit `e1d8072` updated `config/experiment.yaml` and SPEC §6/§7/§9 to the
five-size Qwen3 axis (22 → 32 arms, 5 → 7 selector treatments, 5^4 → 7^4) but
left the tests asserting the old Qwen2.5 matrix. On `e1d8072`: **13 failed, 123
passed**. Eleven are the un-propagated matrix (e.g.
`test_frozen_plan_is_exact_six_shard_22_arm_contract` — `AssertionError: 32 != 22`;
`test_selector_enumerates_five_configs_and_fixed_charges`); two are BUG-6.

This is a launch blocker because `a100_entrypoint.py prepare` runs the full
suite and fails closed.

Fix: update `tests/test_a100_production.py`, `tests/test_analyze.py`,
`tests/test_campaign.py`, `tests/test_contract.py` to the 32-arm / 7-treatment
contract, and rename the arm-count assertions so they read from config rather
than hard-coding 22.

### BUG-8 — §9 memory table is estimated, not measured

The footprints in §9 are estimates. An independent shard-planner check against
the real model config gives Qwen3-14B FP16 = 27.51 GiB weights (≈28,170 MiB)
versus the tabulated ~28,000 MiB, so the estimates are close — but the primary
memory metric must be measured on target hardware before publication.

### Resolved — recorded so they are not re-raised

- **14B FP16 memory envelope.** Qwen3-14B FP16 is 27.51 GiB of weights plus
  7.81 GiB of KV cache at batch 32 / seq 1600 (40 layers, 8 GQA kv-heads,
  head_dim 128) = 35.32 GiB. It fits one A100-40GB with ~2.7 GiB spare. No
  sharding required. Verified by `clean_room/fleet_fit.py`.
- **`device_count() != 1` in `a100_production.py`.** Not a defect. The accuracy
  matrix uses six frozen *logical* shards executed as parallel single-GPU
  workers; one GPU per worker is the intended topology.
- **Qwen3 vs Qwen2.5 regression.** Not reproducible in a clean harness. A
  paired n=300 run gives Qwen3-1.7B-4bit F1 0.5300 vs Qwen2.5-1.5B-4bit 0.4815
  (ΔF1 +0.0485, 95% CI [−0.0034, 0.0992], McNemar p=0.3135 — not significant).
  Qwen3 strict-JSON adherence was 300/300 versus 297/300. Earlier reports of a
  large Qwen3 deficit came from a harness with a divergent generation path, not
  from the model. Batch-size invariance under greedy decoding is 1.000 for both
  families in the clean harness.

### Claim boundary inherited from MA-RAG

MA-RAG (arXiv:2505.20096) reports HotpotQA EM against other published systems
(Atlas 11B, RECOMP 20B, RA-DIT 65B, Self-RAG 8B, ChatQA-1.5 8B/70B, RankRAG
8B/70B, ReAct 70B, Adaptive-RAG GPT-3.5); MA-RAG (Llama3-8B) scores 40.3. It
contains **no single-hop same-backbone baseline** and tests no model below 8B,
and it retrieves densely (gte-multilingual + FAISS) over the full
Karpukhin/DPR Wikipedia corpus. This experiment therefore cannot inherit a
"multi-agent beats one-call RAG" result from MA-RAG; it must establish or refute
it directly on this corpus.

## 15. Additive work — sequential deployment frontier

Additive to §1–§13. Does not alter the frozen cohort, corpus, plan ceiling,
retrieval k, or any manifest hash.

### 15.1 Sequential is the declared deployment topology

Execution is already stage-major with one model resident at a time (§10). This
promotes that to the declared deployment topology. Concurrent residency is not a
deployment target and will not be run as a campaign.

The two costs aggregate differently over the same allocation:

    peak memory   =  max over roles of resident footprint (+ KV cache)
    added latency =  sum over stage transitions of model load time

Memory is set by the single largest role. Latency is paid by every role,
weighted by firing frequency. Shrinking three roles saves zero memory if the
fourth still holds the largest model.

Accuracy is invariant to residency topology — identical weights, greedy
decoding, and prompts give identical outputs whether models are co-resident or
swapped. No accuracy arm is re-run for topology. Any concurrent probe is
timing-only on the 128-question cohort, used to validate
`concurrent = sequential − swap time`.

### 15.2 Two primary findings

**Finding 1 — per-role quantization ranking (headline).** Size fixed at 8B, only
precision varies, one role at a time, against the uniform 8B FP16 reference. The
direct analogue of MA-RAG Table 2 ("Ablation study on LLMs' size") on the
precision axis, which that paper does not examine at all.
`analysis.primary_family` moves to `role_8bit_vs_reference`, Holm-corrected
across the four roles.

**Finding 2 — memory-budget deployment frontier.** At a fixed peak-memory
budget, is capacity better spent on a quantized large model or a native smaller
one? Evaluated per role and uniformly. `role_8bit_vs_mid_fp16` is demoted to
secondary but retains full paired statistics.

### 15.3 Memory-matched treatment tiers

Footprints from model configs (params × bytes/param; 4-bit assumes NF4 with
double quantization, ~0.55 B/param):

| Tier | Treatments | MiB | Spread |
|---|---|---|---|
| ~8 GiB | 14B 4-bit / 8B 8-bit / 4B FP16 | 7,746 / 7,811 / 8,414 | 8% |
| ~15 GiB | 14B 8-bit / 8B FP16 | 14,084 / 15,623 | 10% |

The ~8 GiB tier is a three-point capacity-versus-precision frontier at constant
memory, not a pairwise contrast.

### 15.4 Run matrix extension: 32 → 44 arms

| # | Family | Arms | New |
|---|---|---:|:-:|
| 1 | Uniform 8B FP16 reference (`baseline`) | 1 | |
| 2 | One-role 8B 8-bit | 4 | |
| 3 | One-role 8B 4-bit | 4 | |
| 4 | One-role 4B FP16 (mid) | 4 | |
| 5 | One-role 1.7B FP16 (small) | 4 | |
| 6 | One-role 0.6B FP16 (tiny) | 4 | |
| 7 | One-role 14B FP16 (large) | 4 | |
| 8 | One-role 14B 4-bit | 4 | ✅ |
| 9 | Uniform MA controls (8bit, 4bit, mid, small, tiny, large) | 6 | |
| 10 | Uniform MA 14B 4-bit | 1 | ✅ |
| 11 | Single-hop control `single_fp16` | 1 | |
| 12 | Single-hop, memory-matched | 7 | ✅ |
| | **Total** | **44** | |

Row 12 is `single_8bit`, `single_4bit`, `single_mid`, `single_small`,
`single_tiny`, `single_large`, `single_14b_4bit`. Every uniform MA arm gains a
one-call comparator at the same peak footprint:

| Peak MiB | Uniform MA | Single-hop |
|---:|---|---|
| 1,200 | `ma_uniform_tiny` | `single_tiny` |
| 3,400 | `ma_uniform_small` | `single_small` |
| 4,296 | `ma_uniform_4bit` | `single_4bit` |
| 7,746 | `ma_uniform_14b_4bit` | `single_14b_4bit` |
| 7,811 | `ma_uniform_8bit` | `single_8bit` |
| 8,414 | `ma_uniform_mid` | `single_mid` |
| 15,623 | `baseline` | `single_fp16` |
| 28,168 | `ma_uniform_large` | `single_large` |

Compute impact is ~+20%, not +38%: a single-hop arm issues one call per question
against ~6.3 for a multi-agent arm, measured at 16% of a multi-agent arm's wall
time (497 s vs 3,084 s at n=3000). Selector universe becomes 8⁴ = 4,096.

### 15.5 Latency: two denominators

§12 defines one ratio, against `baseline` = 1.00×. This adds a second. Both are
required.

| Ratio | Denominator | Purpose |
|---|---|---|
| `inverse_throughput_ratio_vs_baseline` | uniform MA 8B FP16 | Scientific — architecture constant, isolates one-role change |
| `inverse_throughput_ratio_vs_matched_single` | that arm's memory-matched single-hop arm | Deployment — "at this budget, what does multi-agent cost over one call?" |

The second requires all seven new single-hop arms in `timing.run_ids`. Without
them there are memory-matched accuracy pairs with no memory-matched latency
pairs, and the deployment claim cannot be made.

Log per-transition model load time so batch-1 on-device latency is re-derivable:

    device_latency_per_question = service_time + (stage_transitions × model_load_s)

Stage-major batching amortizes swap cost across 1,500 questions to near zero; at
batch 1 it is paid per question. The campaign figure systematically understates
on-device swap cost. Both must be reported.

Reportable unit per arm: peak MiB, F1 vs matched single-hop, latency ÷ baseline,
latency ÷ matched single-hop.

### 15.6 Pareto frontier under sequential residency

Because peak memory is `max` over roles, the frontier is a step function with
one step per treatment footprint. For budget B, the best allocation assigns
every role the highest-F1 treatment costing ≤ B.

| Budget B (MiB) | Treatments affordable |
|---:|---|
| 1,200 | 0.6B FP16 |
| 3,400 | + 1.7B FP16 |
| 4,296 | + 8B 4-bit |
| 7,746 | + 14B 4-bit |
| 7,811 | + 8B 8-bit |
| 8,414 | + 4B FP16 |
| 15,623 | + 8B FP16 |
| 28,168 | + 14B FP16 |

Computed post hoc from the one-role ablations. Costs no additional GPU time.

### 15.7 Additivity test — three allocations, not one

The frontier composes one-role effects additively, because one-role ablations
are the only available data. That assumption is untested if one allocation is
executed: a single measurement cannot distinguish a correct prediction from a
lucky one.

Execute three allocations at **best, median, and worst** predicted F1 and report
predicted versus measured. Spread is required; three near-optimal allocations
would score similarly and could not detect miscalibration.

| Outcome | Interpretation | Frontier status |
|---|---|---|
| Ranking preserved, small errors | Effects compose additively | All 4,096 predicted points usable |
| Ranking preserved, large consistent errors | Additive with bias | Usable for ranking, not absolute F1 |
| Ranking inverted | Roles interact | Measured points only: 8 uniform + 3 allocations = 11 |

Non-additivity is a reportable finding, not a failure — MA-RAG's per-agent
ablation assumes role independence, and showing it fails at SLM scale is a
result. Report direction: measured consistently worse than predicted means
effects compound and allocation must be conservative.

Cost: two arms beyond the already-budgeted `ma_optimized_exploratory`.

### 15.8 Claim boundary

The frontier is in-sample and exploratory. The publishable claim is "a
Pareto-optimal allocation method with a tested additivity assumption", not "the
optimal allocation". Parse and protocol status remain a reported covariate — the
confound control separating "quantization degraded reasoning" from "quantization
degraded JSON emission". At 4-bit the Extractor's parse-OK rate was 0.947, worst
of the four roles and the same role that is most accuracy-sensitive.

## 16. Bug report — multi-agent loses to single-hop (Codex handoff)

Authoritative, self-contained description of the defect blocking every result in
this repository. Read with §5, §5.1, §5.2, and §14.

### 16.1 Symptom

The four-agent pipeline loses to the one-call control on multi-hop questions,
and the deficit does not shrink with model scale (observed across 0.5B → 7B).

Paired on identical question IDs, `baseline` vs `single_fp16`:

| Cohort | n | MA F1 | single F1 | Δ |
|---|---:|---:|---:|---:|
| n750 | 731 | 0.412 | 0.542 | −0.129 |
| n300 | 299 | 0.442 | 0.550 | −0.108 |
| n200 | 199 | 0.445 | 0.547 | −0.102 |

Win/loss/tie on n750: MA wins 89, loses 202, ties 440. A 60% tie rate with a
2.3:1 loss ratio on divergence is the signature of information destruction, not
reasoning failure.

Ruled out: answer verbosity (predicted lengths 2.15–2.32 words vs 2.43 gold
across all arms); schema collapse (parse-OK 0.969/0.995/0.947/0.997 for
planner/step_definer/extractor/qa); model family (Qwen3 vs Qwen2.5 ΔF1 +0.0485,
n.s.).

### 16.2 Working hypothesis — unverified, probably incomplete

**This section is a lead, not a diagnosis.** It was produced by static reading
plus retrieval-level probes, and no fix derived from it has been executed
end-to-end. Treat it as one plausible account among several. It may be partly
wrong, it is very likely incomplete, and the true cause may lie somewhere not
named here — in the Extractor contract, the QA evidence filter, prompt/schema
handling, plan-depth routing, the stop condition, or the aggregate route.

Diagnose independently before acting on it. If the evidence contradicts this
section, the evidence wins and this section should be rewritten. What is *not*
hypothesis is §16.1 (the measured symptom) and §16.3 (the measured headroom):
those are reproducible from the artifacts and scripts named.

The four candidate defects below are all in `_retrieval_decision`,
`src/pipeline.py`.

**BUG-1 — the follow-up query discards the original question.**

```python
query_parts = [task["task"]]
```

The grounded follow-up query is `task | grounded_answers`; `q["question"]` is
omitted. Measured on real planner sub-questions from
`baseline_qwen2.5-1.5b_n750_seed7` (n=730, k=10):

| Arm | both-gold recall@10 |
|---|---:|
| single query | 0.495 |
| decomposed, equal read budget | 0.296 |
| decomposed, double read budget | 0.410 |

Decomposition moves retrieval backwards by 19.9 points at equal budget. The
ORACLE arm reaching 0.8863 (§5.1) does so by *retaining* the question.

**BUG-2 — the second hop is gated behind a verbatim substring test.**

```python
grounded_followup_fired = bool(step_index > 0 and grounded_answers)
```

`_grounded_prior_answers` admits only answers where `answer_grounded is True`,
which requires `_answer_is_grounded` to find the QA answer as a literal token
phrase inside Extractor spans. The Extractor emits a median of 22 words, so this
check fails often. When it fails, control reaches the anchor branch and
**re-issues step 1's identical query** — same top-10, zero new evidence.

The pipeline then degenerates to single-hop retrieval plus ~4× the model calls.
This is the primary suspect for the scale-invariant gap: the deficit is created
upstream of the reader, so reader capacity cannot recover it. The firing rate has
never been logged and must be instrumented as part of the fix.

**BUG-3 — retrieval replaces instead of unioning.**

Each branch performs exactly one `search_titles(...)`. The `components` list
declares both `original_question_anchor` and `grounded_step_task`, but exactly
one is ever `attempted` — the structure anticipates a union never implemented.
ORACLE reaches 0.8863 by unioning anchor top-k/2 with follow-up top-k/2.

**BUG-4 — the follow-up query is built from the digest, not the passages.**

The bridge entity must survive retrieval → Extractor compression → QA answer →
verbatim grounding check → query string. Four lossy stages. Measured
compression: Extractor sees 1,588 prompt tokens and hands QA 376 (median 22
words retained, p10 = 3). Gold answer present in Extractor output on 30.7% of MA
losses versus 68.5% of wins.

Meanwhile `single_any_recall` on hidden_bridge is 0.9684 — the passage naming
the bridge entity is already retrieved at hop 1 in 97% of cases, and
`retriever.passages(titles)` is available at the call site but never consulted
for query construction.

### 16.3 Proof the defect is fixable

`clean_room/retrieval_headroom.py`, n=600, k=10, pooled 72,094-passage corpus:

| Stratum | n | SINGLE both-gold | ORACLE two-pass | Headroom |
|---|---:|---:|---:|---:|
| hidden_bridge | 475 | 0.4716 | 0.8863 | **+0.4147** |
| fully_named | 125 | 0.8320 | 0.8880 | +0.0560 |

Verdict GO. 41.5 points of both-gold recall are reachable by a second query and
unreachable by the first, on the stratum that is 1,097 of the frozen 1,500.
fully_named has effectively no headroom — decomposition can only lose there,
which is a required sanity check on any fix.

### 16.4 Candidate fix, in order (verify before applying)

1. **BUG-1** — include `q["question"]` in `query_parts`.
2. **BUG-2** — when no grounded answer exists, still issue `anchor + task`
   rather than bare `anchor`, so the second hop always contributes a distinct
   query. Demote `answer_grounded` to telemetry. Log follow-up firing rate per
   stratum.
3. **BUG-3** — retrieve both components, dedupe, cap at k; populate both
   `components` entries.
4. **BUG-4** (phase 2) — derive follow-up candidates from hop-1 retrieved titles
   and passage text.
5. Bump `retrieval.QUERY_POLICY` and re-freeze the retrieval fingerprint.

### 16.5 Acceptance criteria

A fix is accepted only if all four gates pass. Gates A and D are CPU-only and
deterministic; run them first.

**Gate A — retrieval (CPU, no GPU, no model).** On hidden_bridge, n ≥ 1000,
k = 10: both-gold recall@10 ≥ **0.75** (from 0.4716, toward the 0.8863 oracle).
This gate is deterministic — no sampling, no model — so passing it is strong
evidence the mechanism is repaired.

**Gate B — follow-up firing.** Follow-up fires on ≥ **0.80** of hidden_bridge
question-answering steps beyond step 1. Directly tests BUG-2.

**Gate C — accuracy (GPU, minimal).** On ≥ 200 excluded questions, paired,
multi-agent versus its memory-matched single-hop control:

- overall ΔF1 ≥ **+5.0** points;
- paired bootstrap 95% CI lower bound > **+2.0**;
- McNemar p < **0.01**;
- hidden_bridge ΔF1 ≥ **+8.0** points.

**Gate D — stratum sanity.** fully_named ΔF1 within **±2.0** points. There is
only +0.056 headroom there; a large multi-agent win on fully_named indicates
leakage or a bug, not a fix, and fails the gate.

Threshold justification: closing hidden_bridge both-gold recall from 0.4716 to
~0.85 makes the answer reachable on ~38 additional points of that stratum. At a
conservative 60% conversion from reachable evidence to correct answer, that is
~+23 F1 on hidden_bridge, which at 73% cohort weight is ~+17 F1 versus the
current multi-agent arm (0.412), landing near 0.58 against single-hop's 0.542.
The +5.0 overall threshold is deliberately below that projection and is
consistent with published multi-agent gains at comparable scale — MA-RAG
(Llama3-8B) reports 40.3 EM on HotpotQA against 35.3 for the best same-scale
comparator (RankRAG 8B), a ~5-point margin.

Artifacts predating the fix must not be used for any multi-agent versus
single-hop claim (§14 BUG-7).
