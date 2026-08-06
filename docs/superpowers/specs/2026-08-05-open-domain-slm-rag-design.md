# Plan-Driven SLM Multi-Agent RAG Design

## Goal

Build a proper multi-agent RAG experiment for small language models that retains
the control-flow principles of MA-RAG while making resource limits, retrieval
substitutions, and evidence boundaries explicit.

The study asks whether role-specific size and quantization choices can preserve
HotpotQA answer quality with lower resident memory and less accelerator work. It
does not claim to reproduce reference MA-RAG retrieval results or actual
edge-device performance.

## Reference invariants

Nguyen, Chin, and Tai's MA-RAG design is plan-driven:

- Planner decomposes the question into a context-dependent ordered plan.
- Step Definer chooses the next task using the plan and accumulated state.
- A task can request new knowledge or aggregate existing state.
- Knowledge-seeking tasks retrieve top-k documents.
- Extractor processes retrieved documents independently.
- QA returns an answer plus success feedback for the current step.
- State grows across steps.
- Execution stops on plan completion or semantic inability.
- Step Definer summarizes the completed plan to produce the final answer.

The reference is not limited to two reasoning steps. Its planner examples have
different depths, and contextual plan length is part of the framework.

The local system must preserve every invariant above. Any optimization may
change scheduling or resource use, but not the logical dependencies.

## Local architecture

### Conceptual roles

There are exactly four configurable agents:

1. Planner
2. Step Definer
3. Extractor
4. QA

A plan-summary call reuses the Step Definer treatment. Repeated stage labels are
invocations of their conceptual role and cannot receive separate model or
precision settings.

### State and routing

Planner emits one through five ordered sub-questions. Five is a fail-closed
edge-resource cap for this experiment; it is not a MA-RAG restriction.

For step i, Step Definer sees:

- original question;
- full plan;
- current sub-question and position;
- every completed task, answer, success flag, and rating.

It returns either:

    {"type": "question-answering", "task": "..."}

or:

    {"type": "aggregate", "task": "..."}

For question-answering:

1. use task verbatim as the BM25 query;
2. retrieve top 10 passages;
3. create one Extractor call for each returned document;
4. combine validated or salvaged spans for the current task; and
5. ask QA for answer, success, and rating.

For aggregate:

1. issue no retrieval query;
2. create no Extractor call; and
3. ask QA to combine prior answers.

Append the QA payload to state. Continue only if another planned step exists and
the latest success field is not no. Then run the Step Definer plan-summary
contract for both normal completion and semantic inability. The summary answer
is the scored prediction.

### Failure behavior

There is exactly one generation for each scheduled call. Parsing may salvage
usable fields while retaining the original failure status. If salvage cannot
recover a usable payload:

- Planner falls back to the original question as a one-step plan;
- Step Definer falls back to question-answering on the current sub-question;
- Extractor contributes no spans;
- QA contributes empty degraded state; and
- plan summary contributes an empty prediction if no answer is recoverable.

There is no regeneration, model re-ask, constrained decoding, or treatment-
specific prompt repair. Production OOM is fatal at batch 32.

## Physical execution design

Per-question execution is logically iterative, but a question-at-a-time executor
would repeatedly swap agent models and waste SLM throughput. The runner therefore
uses stage-major scheduling:

1. execute Planner for the cohort;
2. for each possible plan index, execute active Step Definer calls;
3. build retrieval and all per-document Extractor calls for active
   question-answering routes;
4. execute active QA calls after the Extractor stage is durable;
5. skip inactive steps and empty Extractor stages;
6. execute plan summaries for questions that reached a stop; and
7. execute the solo control through its separate path.

Each stage batches homogeneous calls in canonical order across questions and,
for Extractor, documents. Only one model configuration is resident at a time;
consecutive stages may reuse an identical configuration. This changes physical
scheduling only. A downstream call is constructed from durable upstream state,
so no question can jump a dependency.

## Corpus and retrieval design

The corpus is the ordered first-occurrence union of validation passages from the
HotpotQA distractor and fullwiki configurations:

    passages             72094
    gold-title coverage  1.0 for selected and auxiliary cohorts
    gold-sentence cover  1.0 for selected and auxiliary cohorts
    algorithm            sparse BM25, Lucene-style IDF
    query policy         Step Definer task per question-answering step
    top k                10

Distractor must precede fullwiki because shared titles can have different
sentence splits and supporting-fact indices use the distractor split. Retrieval
uses deterministic tokenization, stable corpus-order tie-breaking, and unique
titles. Count, corpus content hash, algorithm, query policy, and coverage are
checked before model allocation and included in experiment/resume identity.

The fullwiki label is not a claim that the corpus is the full Wikipedia dump.
Reference MA-RAG uses dense inner-product FAISS retrieval over a much larger
knowledge base. BM25 on 72,094 passages is a deliberate low-cost experimental
substitution.

Question-to-context mappings, supporting-fact labels, and answers never enter
agent prompts. Nevertheless, corpus passages and evaluation questions come from
the same validation split, and target pages are kept reachable. This design
supports controlled retrieval comparisons, not unseen-corpus generalization.

The frozen final cohort has 1,097 hidden_bridge and 403 fully_named questions;
the excluded pilot has 160 and 40, respectively. Both distributions are
machine-checked. The deterministic pilot sampler replaces malformed ID
5ae61bfd5542992663a4f261 with the next ordered eligible nonsampled exclusion,
5ae622495542995703ce8b20, because supporting sent_id 902 cannot resolve on its
five-sentence gold page. This explicit, content-hashed replacement removes an
annotation defect without outcome-dependent resampling.

## Architecture control

single_fp16 makes one 3B FP16 direct-answer generation. It queries the same BM25
index once with the original question and reads its top 10 passages.

The baseline shares the corpus, algorithm, per-query k, model revision, answer
normalization, and frozen cohort. It does not share the multi-agent system's
variable total exposure: the latter may retrieve once per question-answering
step and generate multiple Extractor calls. Query count, passages, calls, tokens,
and throughput are therefore required alongside F1 and EM.

## Experimental allocation design

The static matrix has 22 runs:

- one uniform 3B FP16 multi-agent reference;
- four one-role 3B 8-bit ablations;
- four one-role 3B 4-bit ablations;
- four one-role 1.5B FP16 ablations;
- four one-role 0.5B FP16 ablations;
- four uniform frontier/floor controls; and
- one 3B FP16 direct-answer architecture control.

The primary near-memory comparison is one-role 3B 8-bit versus the corresponding
1.5B FP16 sibling. The 4-bit tier is secondary. Tiny arms characterize the lower
capacity/compliance boundary.

The exploratory selector declares five configurations for each of four roles,
giving 625 allocations. Tiny is available to a role only if its one-role tiny
ablation's question-clustered 95% lower bound for strict protocol success is at
least 0.90. Eligible allocations must then pass the paired-F1 lower-bound
noninferiority constraint. The objective charges each distinct resident
configuration once and minimizes that deduplicated footprint.

Selection uses the same final cohort, so any materialized mixed result is
in-sample and exploratory.

Static tiny arms are excluded from the shared timing matrix. If selection
matches one of them, its accuracy artifact may be reused but its timing cannot.
Selection materialization writes selected_execution_run_id and
selected_system_timing_required into the derived config. Campaign planning and
the runner authorize exactly that frozen post-selection execution beyond the
static timing IDs, while strict analysis checks the selection-artifact and run-
config hashes in frozen_allocation. The resulting artifact remains exploratory,
retains a verifiable link from the immutable base environment through the
committed selector trace, and does not edit the shared timing matrix
retroactively.

## Measurement design

Accuracy:

- HotpotQA token F1 primary;
- Exact Match co-reported;
- paired uncertainty and Holm adjustment for the four primary role contrasts;
- overall, hidden_bridge, and fully_named strata.

Mechanism:

- emitted and clamped plan depth;
- executed steps and stop reason;
- task route counts;
- retrieval queries, passage exposures, unique titles, and gold-title recall;
- Extractor spans and evidence attribution;
- per-stage and conceptual-role parse/protocol success;
- calls and prompt/output tokens.

Memory:

- deduplicated concurrent model footprint primary;
- isolated role-service footprint;
- stage-major peak allocated/reserved VRAM;
- parameters, buffers, activations, and cold loading separately.

Systems:

- two repeats on a frozen excluded 128-question cohort;
- one reserved uncontended A100;
- steady-state end-to-end service inverse throughput relative to uniform 3B
  FP16 as the primary metric;
- deterministic retrieval, routing, state construction, prompt rendering,
  tokenization/H2D, generation, decode, parse/salvage, and protocol accounting
  inside that primary service time;
- durable logging and cold model loading outside the primary ratio and reported
  separately; and
- a nested generation-only estimate from raw generation batch wall time, plus
  token-normalized throughput and service-component diagnostics.

The A100 benchmark is a comparative proxy for model allocation and scheduling.
It is not a measurement of an edge device. Device-specific speed, energy,
thermal, and memory behavior require deployment on that hardware.

## Pilot and production gates

Before final timing or accuracy:

1. prefetch every pinned dataset/model asset, verify manifest bytes and corpus
   fingerprints, and prove the cache complete in offline mode;
2. validate the complete implementation and corpus on CPU;
3. lock the actual A100 container/software/GPU environment;
4. preflight every active batch shape at 32;
5. run baseline then single_fp16 on the 200 frozen IDs in
   config/manifests/pilot_excluded200_seed20260806.json; and
6. persist GO only when baseline F1 is not lower overall or on hidden_bridge.

Missing or stale artifacts and either negative pilot contrast mean STOP. The
pilot cannot enter final estimates, selector fitting, or claims.

The repository implements these paths. scripts/prefetch_assets.py performs
networked prefetch or network-forbidden cache verification.
scripts/a100_entrypoint.py requires an actual A100 and provides prepare, pilot,
timing, and accuracy phases. Pilot campaign execution is fixed to one worker in
baseline-then-single order and automatically runs scripts/check_pilot.py. The
checker recomputes metrics and writes a content-addressed decision; timing and
accuracy execute modes require that GO file to be tracked and unchanged.

After GO, replay the complete timing matrix, execute each static accuracy arm
once, freeze selector artifacts, optionally execute a distinct mixed allocation,
and run strict final analysis.

The implementation and CPU contract checks are available, but no target-A100
pilot/GO, timing matrix, final accuracy matrix, or selector result is claimed
yet. Those hardware runs remain the experiment's outstanding work.

## Rejected shortcuts

The design explicitly rejects:

- a global hop-count script in place of plan execution;
- splitting one retrieval allowance between hard-coded phases;
- one Extractor call over a concatenated multi-document block;
- using the final intermediate QA answer without plan summary;
- continuing after semantic inability;
- treating repeated calls as separately configurable agents;
- retrying failed generations;
- handing question-specific dataset contexts to answer agents;
- describing the controlled corpus as full Wikipedia; and
- translating A100 ratios into literal edge-device performance.

These shortcuts either remove MA-RAG's stateful mechanism or create an
unsupported claim.
