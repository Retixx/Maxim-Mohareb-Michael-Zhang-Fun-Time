# Current MA-RAG SLM handoff

This file records the active architecture and launch status. Historical designs
and outputs remain in Git history but are not evidence for the final campaign.

## 2026-08-06 — architecture audit and documentation correction

The runtime was audited against the workflow in MA-RAG
(arXiv:2505.20096) and its public implementation.

### Discrepancies found and corrected

The previous active prose described a retrieval shortcut whose depth and budget
were globally fixed. It also described the last intermediate QA call as final,
counted a constant set of generation stages, and sometimes implied that
dataset-assigned contexts were model inputs. Those claims did not match either
MA-RAG or the current runtime.

The active contract now documents:

- a question-dependent one-through-five-step Planner plan;
- repeated Step Definer routing for every active plan step;
- question-answering and aggregate routes;
- one top-10 BM25 retrieval per question-answering step;
- one Extractor invocation per retrieved document;
- one QA feedback call per active step;
- append-only answer/success/rating state;
- semantic stop on QA success=no;
- a Step Definer plan summary as the scored final answer;
- stage-major batching across questions and retrieved documents;
- no generation retries; and
- a five-step edge-resource ceiling that is local to this experiment.

Reference MA-RAG is compatible with variable plan depth and does not define a
two-step architectural limit. Repeated stages in this repository are invocations
of four conceptual agents, not extra independently configured agents.

### Intentional reference differences

Three differences remain by design and must be disclosed with every result:

1. Reference MA-RAG uses dense FAISS retrieval; this experiment uses
   deterministic sparse BM25.
2. Reference MA-RAG searches a much larger knowledge base; this controlled
   corpus has exactly 72,094 HotpotQA validation passages.
3. Reference MA-RAG leaves plan depth contextual without this campaign's local
   ceiling; the experiment caps plans at five to bound edge-resource cost.

These are experiment choices, not claims that the systems are identical.

### Data and exposure boundary

All final arms share the frozen ordered 1,500-question cohort. The retriever
indexes the first-occurrence union of the HotpotQA distractor and fullwiki
validation configurations. The fullwiki name is only a dataset configuration;
the index is not the full Wikipedia dump.

The final cohort is frozen at 1,097 hidden_bridge and 403 fully_named.
The excluded pilot is frozen at 160 hidden_bridge and 40 fully_named. Its
deterministic manifest replaces malformed ID
5ae61bfd5542992663a4f261 with the next ordered eligible nonsampled exclusion,
5ae622495542995703ce8b20, because the source annotation requests supporting
sentence 902 from a five-sentence page. The manifest records this data-quality
exception and its reason; its file and ordered-ID hashes are pinned in
config/experiment.yaml.

Question-to-context mappings, supporting-fact labels, and gold answers do not
enter model prompts. However, evaluation questions and corpus passages originate
from the same pinned validation split and the corpus is constructed to keep
gold titles reachable. Results therefore measure controlled, target-reachable
retrieval rather than unseen-corpus generalization.

## Active experiment contract

### Architecture

    Planner
      -> repeated Step Definer
      -> [per-step retrieval -> per-document Extractor] or [aggregate]
      -> per-step QA feedback
      -> plan completion or semantic stop
      -> Step Definer plan summary

Logical state is per question. Physical scheduling is stage-major so homogeneous
SLM calls are batched and one model configuration is resident at a time.
Inactive plan steps and Extractor work for aggregate routes generate no calls.

### Retrieval

    algorithm         sparse BM25 with Lucene-style IDF
    corpus passages   72094
    query policy      Step Definer task, once per question-answering step
    top k             10
    extractor unit    one retrieved document

### Matrix and selector

- 22 static runs;
- four conceptual roles;
- 3B FP16, 3B 8-bit, 3B 4-bit, 1.5B FP16, and 0.5B FP16 treatments;
- one 3B FP16 retrieval/direct-answer architecture control;
- guarded 5^4 = 625 exploratory allocation universe; and
- at most one distinct ma_optimized_exploratory execution.

Tiny use is role-specific: the corresponding one-role tiny ablation must clear
the frozen question-clustered strict-protocol lower-bound gate. Selection also
must satisfy the paired-F1 noninferiority constraint. Any selected system is
in-sample and exploratory.

### Outcomes and efficiency

- HotpotQA token F1 is primary; Exact Match is co-reported.
- Report overall, hidden_bridge, and fully_named results.
- Report plan depth, executed steps, stop reason, retrieval recall, query and
  passage exposure, calls, tokens, parse/protocol status, and evidence.
- Primary memory is deduplicated concurrent model-footprint MiB.
- Sequential peak VRAM, isolated role-service memory, and cold loading are
  separate diagnostics.
- Reserved-A100 steady-state end-to-end service inverse throughput is the
  primary systems proxy. It includes retrieval, routing, state/prompt building,
  tokenization/H2D, generation, decode, and parse/protocol work while excluding
  model loading and durable logging.
- Generation-only inverse throughput from raw batch generation time is nested
  as a diagnostic; it is not the primary timing result.
- Neither timing measure is literal edge-device timing, energy, or thermals.

## Current launch gates

The fail-closed launch machinery is implemented:

- scripts/prefetch_assets.py verifies all manifest bytes, free space, pinned
  dataset/model caches, and the corpus count and fingerprints; its offline mode
  proves that an A100 job needs no network fetch.
- scripts/a100_entrypoint.py requires an NVIDIA A100 and provides prepare,
  pilot, timing, and accuracy phases. Prepare requires clean committed source
  plus an immutable container identity, runs offline cache verification and the
  full tests, and writes the environment lock.
- pilot campaign mode enforces one worker and the ordered baseline then
  single_fp16 pair. It automatically invokes scripts/check_pilot.py, which
  recomputes the prespecified metrics and validates cohort, environment,
  experiment, retrieval, prompt, artifact, stratum, and content hashes.
- timing and accuracy execute modes reject a missing, STOP, stale, uncommitted,
  or locally changed analysis/pilot_gate.json.
- selection materialization writes the selected timing run and requirement into
  the derived config. Campaign planning, the runner, and strict analysis admit
  only that selected tiny run and verify frozen selection/run-config provenance.
- primary service timing and its nested generation-only diagnostic are both
  implemented, including deterministic orchestration timing records.

The remaining gates require target hardware execution, not architecture work:

1. populate and offline-verify the pinned cache on the A100 host;
2. create, commit, and distribute the environment lock from the exact clean
   source/container/A100 environment;
3. pass excluded batch-32 preflight for every active stage shape;
4. run the 200-question pilot and commit a valid GO certificate;
5. run the two-repeat 128-question timing matrix on one reserved A100;
6. execute all 22 static accuracy arms once with valid hashes and canonical
   batches;
7. freeze the selector trace and derived config, then execute/time a distinct
   selected allocation if required; and
8. run strict final analysis.

No target-A100 pilot, GO certificate, timing result, final accuracy result, or
selector result is asserted by this documentation. Do not consume final IDs
until the committed pilot gate is GO, and do not diagnose a STOP on final data.

## Active files

| File | Role |
|---|---|
| SPEC.md | Authoritative scientific and architecture contract |
| README.md | User-facing system overview |
| RUNBOOK.md | Normative production order and hard stops |
| config/experiment.yaml | Machine-readable frozen experiment |
| src/prompts.py | Agent schemas, stage mapping, and resource ceiling |
| src/pipeline.py | Stateful routing, retrieval, extraction, QA, and summary |
| src/runner.py | Corpus validation, stage-major execution, resume, memory, timing |
| src/retrieval.py | Deterministic controlled-corpus BM25 |
| scripts/prefetch_assets.py | Pinned-asset prefetch and offline verification |
| scripts/a100_entrypoint.py | Fail-closed target-A100 production phases |
| scripts/check_pilot.py | Pilot validation and content-addressed decision |
| analyze.py | Paired statistics, diagnostics, and exploratory selector |

If an active file disagrees with SPEC.md or config/experiment.yaml, stop and
resolve the discrepancy on excluded data before any production run.
