# Experiment contract: role-aware SLM allocation in multi-agent RAG

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
  cheaply on a controlled corpus. Results compare SLM agent allocations under
  this retriever and do not establish parity with dense retrieval.
- Reference MA-RAG searches a much larger knowledge base. This campaign searches
  72,094 HotpotQA validation passages. It is a controlled retrieval experiment,
  not a Wikipedia-scale benchmark.

These adaptations are the edge-systems variables being studied. They do not
change the multi-agent, stateful, variable-depth reasoning topology.

## 2. Research questions

At small-language-model scale, can role-specific model size and quantization
reduce resident memory and accelerator work while retaining answer quality in a
fully executed multi-agent RAG workflow?

The prespecified questions are:

1. Which conceptual roles are most sensitive to moving the 3B model to 8-bit?
2. Which roles are most sensitive to swapping 3B FP16 for the 1.5B FP16 sibling?
3. At a near memory match, do quantization and a smaller sibling impose different
   role-level costs?
4. How do uniform multi-agent 3B FP16 and a competitive one-call 3B FP16
   retrieval baseline compare overall and on hidden-bridge questions?
5. Which guarded role allocation minimizes deduplicated resident model memory
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

The one-call control, single_fp16:

- uses one 3B FP16 generation;
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

Immutable model revisions are:

| Alias | Model | Revision |
|---|---|---|
| base | Qwen2.5-3B-Instruct | aa8e72537993ba99e69dfaafa59ed015b17504d1 |
| small | Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| tiny | Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 |

The static matrix contains 22 run IDs:

| Family | Count |
|---|---:|
| Uniform four-agent 3B FP16 reference | 1 |
| One-role 3B 8-bit ablations | 4 |
| One-role 3B 4-bit ablations | 4 |
| One-role 1.5B FP16 ablations | 4 |
| Uniform 8-bit, 4-bit, and 1.5B controls | 3 |
| Four one-role 0.5B ablations and one uniform 0.5B run | 5 |
| One-call 3B FP16 architecture control | 1 |

All repeated Step Definer, Extractor, and QA stage labels mirror their
conceptual role treatment. The plan-summary stage mirrors Step Definer.
Repeated stages cannot be configured independently.

The primary treatment family compares one-role 3B 8-bit ablations with the
corresponding 1.5B FP16 ablations. The 4-bit family is mandatory secondary
evidence. The 0.5B runs measure the lower capacity/compliance boundary and also
feed a guarded exploratory selector.

## 7. Exploratory role allocation

The selector declares five candidates for each of four conceptual roles:

    {3B FP16, 3B 8-bit, 3B 4-bit, 1.5B FP16, 0.5B FP16}

The full universe is 5^4 = 625 allocations. A role may use 0.5B only when its
corresponding one-role tiny ablation has a question-clustered 95% lower bound of
at least 0.90 for strict protocol success. Failure removes tiny only for that
role.

Eligible candidates must also satisfy the paired-bootstrap lower-bound
noninferiority constraint: predicted F1 may be no more than 1.0 point below the
uniform 3B FP16 reference. The selector minimizes deduplicated concurrent model
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
    S_loss_r = F1_baseline - F1_role_small
    axis_contrast_r = F1_role_8bit - F1_role_small

Use 10,000 paired question-level bootstrap replicates. Holm-adjust the four
primary role contrasts. Report point estimates, paired 95% intervals, adjusted
p-values, and direction. Keep 4-bit, evidence, parsing, retrieval, plan-depth,
semantic-stop, and tiny-floor analyses secondary or exploratory. Never stratify
a primary treatment comparison by that arm's own emitted or executed plan depth:
depth is post-treatment. A baseline-defined depth label may be applied unchanged
to every arm only as a prespecified secondary diagnostic.

Repeated calls are not independent observations. Cluster role diagnostics by
question. Resample complete batches for timing intervals. Sort question IDs
before seeded resampling so write order cannot affect results.

## 9. Memory and edge-efficiency contract

Measured resident model footprints are:

| Configuration | MiB |
|---|---:|
| 3B FP16 | 5,886.0 |
| 3B 8-bit | 3,240.0 |
| 1.5B FP16 | 2,944.4 |
| 3B 4-bit | 1,917.0 |
| 0.5B FP16 | 942.3 |

The 3B 8-bit and 1.5B FP16 treatments are near-memory-matched, not equal.

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
4. execute all 22 static accuracy arms once;
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
multi-agent 3B FP16 at 1.00x. It includes deterministic retrieval, routing, and
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
- pure parameter-count or pure bit-width causality;
- exact memory equality between 8-bit and 1.5B;
- confirmatory superiority of the selected mixed allocation;
- literal edge-device timing, energy, or thermals from A100 results;
- probability calibration; or
- independent final evidence from the pilot or historical runs.

RUNBOOK.md is the normative execution order. Any silent change in sample,
corpus, plan ceiling, routing, retrieval k, model/revision, precision, prompt,
parser, batch membership, or artifact path invalidates the affected comparison.
