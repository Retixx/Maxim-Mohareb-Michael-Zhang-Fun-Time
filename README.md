# Role-aware SLM multi-agent RAG

This repository evaluates whether a MA-RAG-style system can assign smaller or
quantized language models to different agent roles while retaining answer
quality and reducing memory and inference work.

It is a real multi-agent, multi-hop RAG pipeline. Within this experiment the
number of reasoning steps comes from each question's one-through-five-step plan;
it is not hard-coded to a double hop.

## Architecture

The four conceptual agents are Planner, Step Definer, Extractor, and QA:

    question
       |
       v
    Planner -> ordered plan
       |
       v
    Step Definer (repeated for each active plan step)
       |-- question-answering -> anchored BM25 fusion -> Extractor/document -> QA
       |-- aggregate -----------------------------------------------> QA
       |
       +-- append QA answer/success/rating to state
       +-- continue, plan complete, or stop on success=no
       |
       v
    Step Definer plan summary -> scored short answer

The first question-answering step retrieves the original-question top 10. Later
steps with grounded state deterministically fuse seven original-question anchor
results with three unique results from the resolved task plus grounded answers.
Each returned document gets its own Extractor call; only normalized exact source
sentences enter QA, and empty document blocks are omitted. Unsupported guesses
remain logged but cannot poison later retrieval or aggregate evidence. Aggregate
steps reuse grounded answers without retrieval or extraction. Plan summary is
always attempted; if it cannot yield a usable answer, the last usable QA answer
is retained with explicit fallback provenance.

The runner preserves those state dependencies while executing stage-major:
homogeneous calls are batched across questions and retrieved documents, inactive
steps are skipped, and one model configuration is resident at a time. Repeated
stage labels still map to the same four conceptual role treatments.

## Relationship to MA-RAG

Reference MA-RAG is plan-driven and supports question-dependent plan depth; it
is not limited to two steps. This implementation follows that principle and
adds a five-step ceiling solely as an explicit edge-resource cap for this frozen
experiment. The cap must not be described as a MA-RAG limit.

Two other adaptations are deliberate:

| Reference MA-RAG | This experiment |
|---|---|
| Dense inner-product retrieval with FAISS | Deterministic sparse BM25 |
| Much larger knowledge base | 72,094 controlled HotpotQA passages |
| Context-dependent plan length | Context-dependent plan length, capped at five |
| Per-step routing, retrieval, extraction, QA, and summary | Same logical control flow |

These choices make the role-allocation experiment reproducible and affordable
for SLM/edge research. They do not establish equivalence to dense retrieval or
Wikipedia-scale performance.

## Data and exposure boundary

Accuracy uses a frozen ordered sample of 1,500 HotpotQA validation questions.
The search corpus is the first-occurrence union of validation passages from the
distractor and fullwiki configurations:

    passages: 72,094
    retrieval k: 10 per question-answering plan step
    final retrieval strata: hidden_bridge 1097, fully_named 403
    pilot retrieval strata: hidden_bridge 160, fully_named 40
    final ordered-ID SHA-256:
    5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af

The fullwiki label is a HotpotQA configuration name; the corpus is not the full
Wikipedia dump. The corpus contains evaluation-set target pages, but no
question-to-context mapping, supporting-fact label, or gold answer enters a
model prompt. Treat results as controlled, target-reachable retrieval with
validation-split exposure, not unseen-corpus evidence.

The pilot manifest deterministically replaces malformed source ID
5ae61bfd5542992663a4f261 with 5ae622495542995703ce8b20. The original annotation
names sentence 902 on a five-sentence gold page; the committed manifest records
the replacement and reason so every system is not forced to fail an impossible
gold-sentence reachability check. Its file and ordered-ID SHA-256 values are,
respectively, 975210805c382788bb39c800266ae22a88cc526e0626bd8a0106c35d316a8bb1
and f8c3f16458340cb0bc74aa827e3b51528ba351963a46dba456ac4e68ad20f7d7.

## Experiment

The static matrix contains 22 runs:

- uniform four-agent 3B FP16 reference;
- one-role 3B 8-bit, 3B 4-bit, 1.5B FP16, and 0.5B FP16 ablations;
- uniform 8-bit, 4-bit, 1.5B, and 0.5B controls; and
- a one-call 3B FP16 architecture control.

The one-call control issues one original-question BM25 query and reads its top
10 passages. The multi-agent system exposes at most 10 passages per
question-answering step and may issue a second grounded task query after step 1;
its frozen fusion quota is 7 anchor/3 task. The comparison reports passage
exposure, component-query count, model calls, tokens, F1, EM, memory, and timing
rather than pretending the two architectures have identical work budgets.

F1 is primary and Exact Match is co-reported. Results are also split into
hidden_bridge and fully_named strata. The primary role comparison is 3B 8-bit
versus the near-memory-matched 1.5B FP16 sibling; 4-bit and 0.5B findings are
secondary.

After the static runs, a guarded selector evaluates 5^4 = 625 role allocations.
A role may use the 0.5B model only if its one-role ablation clears the frozen
question-clustered protocol-success gate. Any selected mixed run is explicitly
in-sample and exploratory.

Tiny arms are not in the prespecified timing matrix. If selection reuses a tiny
static accuracy arm, accuracy can be reused but timing cannot. Selection
materialization now writes the selected execution and timing requirement into
the derived config; campaign planning and the runner authorize exactly that
separately labelled post-selection run, while strict analysis checks its frozen
selection and run-config hashes before reporting exploratory throughput.

## Safety gates

No final timing or accuracy run may start before all of these hold:

1. corpus count, corpus fingerprint, and gold-title and supporting-sentence
   reachability validate;
2. the immutable A100 environment lock validates on every worker;
3. every active stage shape passes excluded batch-32 preflight;
4. the excluded 200-question pilot manifest
   config/manifests/pilot_excluded200_seed20260806.json runs baseline then
   single_fp16; and
5. the persisted pilot certificate says GO.

Pilot GO requires multi-agent F1 to be at least the one-call F1 both overall and
on hidden_bridge. A missing, stale, incomplete, or failed pilot is STOP. Do not
touch final IDs while diagnosing it.

See [RUNBOOK.md](RUNBOOK.md) for the normative command order and
[SPEC.md](SPEC.md) for the complete scientific contract.

## Environment

Model, tokenizer, and dataset revisions are immutable in
config/experiment.yaml. Production also requires a committed environment lock
created from the selected A100/container. Historical package versions in the
repository are not a substitute for that resolved lock.

For local integrity checks:

    python -X utf8 -m pytest -q
    python -X utf8 smoke_test.py --n 10 --run baseline

The smoke path uses excluded development data. It is a plumbing check, not
accuracy evidence.

## Production outline

Use the full runbook; this is only an orientation:

    export EXPERIMENT_CONTAINER_REF=REGISTRY/IMAGE:TAG
    export EXPERIMENT_CONTAINER_DIGEST=sha256:IMMUTABLE_DIGEST

    python scripts/prefetch_assets.py

    python scripts/a100_entrypoint.py prepare \
      --container-ref "$EXPERIMENT_CONTAINER_REF" \
      --container-digest "$EXPERIMENT_CONTAINER_DIGEST"

Commit and distribute the generated environment lock, then run the pilot on its
single reserved A100:

    python scripts/a100_entrypoint.py pilot \
      --container-ref "$EXPERIMENT_CONTAINER_REF" \
      --container-digest "$EXPERIMENT_CONTAINER_DIGEST"

Pilot execution is fixed to baseline then single_fp16 and automatically invokes
scripts/check_pilot.py. Commit and distribute the resulting unchanged GO
certificate before any final worker proceeds. Then use the same entrypoint for
the frozen timing and accuracy assignments:

    python scripts/a100_entrypoint.py timing \
      --container-ref "$EXPERIMENT_CONTAINER_REF" \
      --container-digest "$EXPERIMENT_CONTAINER_DIGEST"

    python scripts/a100_entrypoint.py accuracy --workers 4 --worker-index 0 \
      --container-ref "$EXPERIMENT_CONTAINER_REF" \
      --container-digest "$EXPERIMENT_CONTAINER_DIGEST"

    python analyze.py --config config/experiment.yaml

Run the accuracy command once per worker index. Do not run timing or accuracy
unless the campaign entrypoint validates a committed, unchanged
analysis/pilot_gate.json. Prefetch, the A100 wrapper, pilot mode/checking, and
gate enforcement are implemented and covered by CPU tests; the target-A100
pilot/GO, timing matrix, final accuracy matrix, selector execution, and strict
final analysis have not yet been run.

## Timing, memory, and edge interpretation

The primary memory metric charges each distinct resident model configuration
once. Sequential peak VRAM, isolated role-service memory, parameters, buffers,
activation peaks, and cold loading are separate diagnostics.

One reserved A100 measures two repetitions over a frozen excluded 128-question
cohort. The primary metric is steady-state end-to-end service inverse
throughput relative to uniform four-agent 3B FP16 at 1.00x. It includes
retrieval, routing, state and prompt construction, tokenization/H2D, generation,
decode, parse/salvage, and protocol accounting; model loading and durable JSONL
logging are excluded and reported separately. Each run also reports a nested
generation-only estimate from raw generation batch time. This is a controlled
accelerator proxy for comparing model allocation and stage scheduling, not
phone, embedded-GPU, CPU, NPU, energy, thermal, or device wall-time evidence.

## Repository layout

| Path | Purpose |
|---|---|
| SPEC.md | Authoritative architecture, experiment, and claim contract |
| RUNBOOK.md | Ordered production procedure and hard stops |
| PROGRESS.md | Current implementation and launch status |
| config/experiment.yaml | Frozen models, architecture, retrieval, matrix, and metrics |
| config/manifests/ | Final, warm-up, timing, and pilot cohort identities |
| src/prompts.py | Four agent contracts, repeated-stage map, and plan summary |
| src/pipeline.py | Stateful call construction, routing, retrieval, and answer records |
| src/retrieval.py | Deterministic BM25 corpus and index |
| src/runner.py | Stage-major batching, resume integrity, memory, and timing |
| scripts/prefetch_assets.py | Pinned-asset download and offline cache verification |
| scripts/a100_entrypoint.py | Fail-closed prepare, pilot, accuracy, and timing wrapper |
| scripts/check_pilot.py | Pilot recomputation and content-addressed GO/STOP gate |
| analyze.py | Paired analysis, diagnostics, and exploratory selection |

## Controls that must not change

- no generation retries or regeneration;
- no constrained decoding;
- no treatment-specific prompt tuning;
- no question replacement or resampling;
- no reduction of the production batch size after OOM;
- no floating model, dataset, corpus, or prompt revision;
- no independent configuration of repeated role stages;
- no final execution without pilot GO; and
- no literal edge-device claim from the A100 proxy.
