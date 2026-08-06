# Plan-Driven SLM Multi-Agent RAG Completion Plan

## Goal

Finish and validate a one-shot, A100-ready HotpotQA experiment that executes the
MA-RAG control flow with small language models, measures role-specific
size/quantization choices, and makes no unsupported edge-device or
Wikipedia-scale claim.

Checkboxes describe the repository state at the 2026-08-06 documentation audit:

- [x] verified in the current implementation or completed by this documentation
  pass;
- [ ] still required before final timing or accuracy execution.

## Frozen architecture

The implementation target is:

    Planner
      -> for each planned step:
           Step Definer
             -> question-answering:
                  BM25 top-10
                  Extractor once per document
                  QA feedback
             -> aggregate:
                  QA feedback from prior state
           append answer/success/rating
           stop when complete or success=no
      -> Step Definer plan summary

Plan length is contextual. The local executor supports variable plans of one
through five steps, with five as an edge-resource cap. Reference MA-RAG itself
is not assigned that ceiling.

Physical execution is stage-major: batch all active homogeneous calls in
canonical order, preserve per-question dependencies, and keep one model
configuration resident at a time.

The frozen final retrieval strata are 1,097 hidden_bridge and 403 fully_named;
the excluded pilot strata are 160 and 40. Runner and analysis contracts reject
any other counts.

## Task 1: Lock the reference interpretation

Files:

- SPEC.md
- README.md
- docs/superpowers/specs/2026-08-05-open-domain-slm-rag-design.md

- [x] Confirm from the MA-RAG paper and public implementation that plan depth is
  question-dependent rather than double-hop-only.
- [x] Record the reference loop: Planner, repeated Step Definer routing,
  per-step retrieval, per-document extraction, per-step QA feedback, semantic
  stop, and Step Definer summary.
- [x] Record reference/local differences:
  - dense FAISS versus sparse BM25;
  - large reference knowledge base versus 72,094 controlled passages;
  - unconstrained contextual depth versus a local five-step resource ceiling.
- [x] Prohibit descriptions that turn local resource choices into MA-RAG
  framework limits.

Acceptance:

- Every active document describes the same logical state machine.
- No active document uses a global hop count as the architecture definition.

## Task 2: Implement variable-depth roles and schemas

Files:

- src/prompts.py
- src/parsing.py
- src/mechanism.py
- tests/test_contract.py
- tests/test_execution_integrity.py

- [x] Keep exactly four conceptual roles: planner, step_definer, extractor, qa.
- [x] Generate repeated Step Definer/Extractor/QA stage labels for plan indices
  zero through four.
- [x] Mirror every repeated label to its conceptual role treatment.
- [x] Mirror plan_summary to Step Definer.
- [x] Give Planner a one-through-five ordered-plan contract.
- [x] Give Step Definer question-answering and aggregate routes.
- [x] Give QA answer, success, and rating feedback.
- [x] Give plan summary output, answer, and score.
- [x] Enforce the short-answer protocol on finalization.

Acceptance:

    python -m pytest tests/test_contract.py \
      tests/test_execution_integrity.py -q

Tests must prove stage mapping, role mirroring, summary schema, cap consistency,
and that the scored answer comes from plan_summary.

## Task 3: Implement the stateful MA-RAG loop

Files:

- src/agents.py
- src/pipeline.py
- tests/test_execution_integrity.py
- tests/test_retrieval.py

- [x] Reconstruct append-only step history from durable records.
- [x] Pass original question, full plan, current step, and prior results to every
  active Step Definer call.
- [x] Use the exact Step Definer task as the retrieval query.
- [x] Skip retrieval and Extractor for aggregate.
- [x] Create one Extractor call per retrieved document and retain document rank,
  title, query, and step provenance.
- [x] Build QA after every active step, including aggregate.
- [x] Prevent a later step when the previous QA success field is no.
- [x] Create plan_summary for both plan completion and semantic inability.
- [x] Use summary answer for multi-agent scoring.

Acceptance tests must include:

- a one-step plan;
- plans with more than two steps;
- mixed question-answering and aggregate routes;
- early semantic stop;
- top-k per active retrieval step;
- ten independently attributable Extractor calls when ten passages return; and
- finalization after both stop reasons.

## Task 4: Make retrieval deterministic and disclose its limits

Files:

- src/retrieval.py
- src/runner.py
- config/experiment.yaml
- tests/test_retrieval.py

- [x] Build the first-occurrence corpus union in distractor, fullwiki order.
- [x] Require exactly 72,094 passages and the frozen corpus SHA-256.
- [x] Require unique titles and 100% selected/auxiliary gold-title and
  supporting-sentence reachability.
- [x] Use fixed tokenization, Lucene-style IDF, and stable corpus-order ties.
- [x] Set k=10 for each question-answering plan step.
- [x] Persist corpus, algorithm, and query-policy fingerprints.
- [x] Bind fingerprints into experiment identity, call records, metadata,
  completion, and resume checks.
- [x] Keep question-to-context mapping, labels, and gold answers outside prompts.

Acceptance:

    python -m pytest tests/test_retrieval.py -q

Documentation must say that fullwiki is a HotpotQA configuration name, the
corpus is not full Wikipedia, and validation questions/passages share a source
split.

## Task 5: Preserve MA-RAG dependencies under stage-major batching

Files:

- src/runner.py
- src/pipeline.py
- src/agents.py
- tests/test_execution_integrity.py

- [x] Walk canonical repeated stages only after upstream state is durable.
- [x] Batch active calls across questions.
- [x] Batch per-document Extractor calls without merging document provenance.
- [x] Avoid model loading for stages with zero calls.
- [x] Reuse consecutive identical model fingerprints without changing stage
  memory accounting.
- [x] Keep production batch_size and min_batch_size pinned at 32.
- [x] Record canonical batch order and membership.
- [x] Treat OOM as fatal and record retry_count=0.
- [x] Allow resume only when model, prompt, manifest, retrieval, experiment, GPU,
  and batch identities match.

Acceptance:

- Stage-major scheduling produces the same logical call graph as per-question
  execution.
- Empty stages do not allocate a model.
- Resume cannot repack pending calls.

## Task 6: Make records and analysis depth-aware

Files:

- src/pipeline.py
- analyze.py
- tests/test_analyze.py

- [x] Record emitted/clamped plan depth, plan text, executed steps, stop reason,
  summary output, and summary score.
- [x] Record every retrieval event, query count, passage exposure, unique titles,
  retrieved/gold titles, title recall, and all-gold status.
- [x] Attribute Extractor spans against passages actually retrieved.
- [x] Report every concrete repeated stage for calls, tokens, timing, parsing,
  and protocol behavior.
- [x] Roll repeated stages up to four conceptual roles only where a role-level
  statistic is required.
- [x] Cluster repeated-call statistics by question.
- [x] Report F1 and EM overall, hidden_bridge, and fully_named.
- [x] Compare baseline and single_fp16 with paired uncertainty and exposure/cost
  diagnostics.

Acceptance:

    python -m pytest tests/test_analyze.py -q

No analysis field may assume a constant number of calls per question.

## Task 7: Complete the guarded allocation experiment

Files:

- config/experiment.yaml
- analyze.py
- tests/test_analyze.py
- tests/test_contract.py

- [x] Keep the exact 22 static run IDs.
- [x] Declare five candidate configurations for each of four roles.
- [x] Declare the full 625-allocation universe.
- [x] Gate each role's tiny candidate on its corresponding one-role tiny
  ablation.
- [x] Use a question-clustered 95% strict-protocol lower bound threshold of
  0.90.
- [x] Retain the paired-F1 lower-bound noninferiority constraint.
- [x] Minimize deduplicated concurrent model footprint.
- [x] Emit deterministic eligibility, feasibility, and tie-break traces.
- [ ] Run the selector only after every required final static artifact validates.
- [ ] Commit the trace and derived config before any distinct selected run.
- [ ] Execute the selected system once if it is not an existing static arm.
- [x] Implement derived-config materialization that, when selection reuses a
  tiny static accuracy arm, reuses accuracy only and authorizes exactly one
  separately labelled timing run.
- [x] Implement selected-timing provenance validation without disabling the
  immutable base environment/source checks.

The selected result is in-sample and exploratory.

## Task 8: Add and enforce the excluded-data pilot

Files:

- config/manifests/pilot_excluded200_seed20260806.json
- config/experiment.yaml
- src/runner.py
- scripts/run_campaign.py
- scripts/check_pilot.py
- tests/test_campaign.py
- tests/test_pilot_gate.py
- RUNBOOK.md

- [x] Freeze exactly 200 unique excluded IDs, disjoint from final, timing, and
  warm-up cohorts, with committed hashes and source linkage.
- [x] Freeze the exact pilot strata at 160 hidden_bridge and 40 fully_named.
- [x] Record the deterministic replacement of malformed source ID
  5ae61bfd5542992663a4f261 by the next ordered eligible nonsampled exclusion,
  5ae622495542995703ce8b20, including the impossible sentence-902 reason.
- [x] Add pilot artifact mode with a distinct cohort identity and paths.
- [x] Make the one-worker assignment exactly baseline then single_fp16.
- [x] Validate environment, experiment, retrieval, prompt, cohort, JSONL, and
  metadata hashes.
- [x] Report paired F1/EM, parse/protocol success, retrieval recall and exposure,
  plan depth, stop reasons, and stratum counts.
- [x] Emit GO only when baseline-minus-single F1 is nonnegative both overall and
  on hidden_bridge.
- [x] Persist a content-addressed GO certificate.
- [x] Write the accepted certificate to analysis/pilot_gate.json.
- [x] Make timing and accuracy execute modes reject missing, STOP, stale, or
  mismatched certificates.
- [x] Require the accepted gate to be committed and unchanged before final
  execution.
- [x] Test GO, both failure directions, missing data, stale data, wrong order,
  wrong cohort, and plan-only behavior.

Acceptance:

    python -m pytest tests/test_campaign.py tests/test_pilot_gate.py -q
    python scripts/run_campaign.py --kind pilot --workers 1

No final ID may be read by the pilot or by diagnosis after STOP.
Actual target-A100 pilot execution and the resulting GO/STOP decision belong to
Task 9 and remain pending.

## Task 9: Rebuild production timing and verify edge claim boundaries

Files:

- RUNBOOK.md
- PROGRESS.md
- analyze.py
- config/experiment.yaml
- scripts/prefetch_assets.py
- scripts/a100_entrypoint.py

- [x] Implement pinned manifest/dataset/model prefetch, disk-space checking,
  corpus fingerprint verification, and a network-forbidden cache check.
- [x] Implement a fail-closed A100 entrypoint for prepare, pilot, accuracy, and
  timing phases.
- [ ] Validate and commit the immutable A100 environment lock.
- [ ] Pass excluded batch-32 preflight for every active stage shape.
- [ ] Run the 200-question pilot on the target A100 and commit a valid GO
  certificate.
- [ ] After pilot GO, rerun the complete timing matrix under the current
  experiment fingerprint.
- [ ] Use one reserved uncontended A100 for all timing arms.
- [ ] Record two full repetitions on the frozen excluded 128-question cohort.
- [x] Instrument steady-state end-to-end service inverse throughput as primary,
  including retrieval, routing, state/prompt construction, tokenization/H2D,
  generation, decode, parse/salvage, and protocol accounting while excluding
  durable logging and cold load.
- [x] Nest a generation-only estimate from raw batch generation wall time and
  report service orchestration/non-generation components separately.
- [x] Keep timing artifacts out of accuracy and selector estimates.
- [x] Implement the conditional selected-tiny timing path without adding tiny arms
  to the shared prespecified timing matrix.
- [x] State that A100 measurements are a comparative systems proxy.
- [x] Make no device-specific speed, energy, thermal, or memory claim without
  measurements on that device.
- [ ] Produce and report the target-A100 service, generation-only, token, and
  cold-load measurements.

## Task 10: Documentation and final verification

Documentation reconciled in this implementation-state audit:

- SPEC.md
- README.md
- PROGRESS.md
- docs/superpowers/specs/2026-08-05-open-domain-slm-rag-design.md
- docs/superpowers/plans/2026-08-05-open-domain-slm-rag.md

RUNBOOK.md remains the normative operator sequence and was intentionally not
edited in this five-document pass.

- [x] Align the five audited documents to repeated Step Definer routing.
- [x] Document top-k per question-answering step and per-document Extractor.
- [x] Document aggregate routing, semantic stop, and Step Definer summary.
- [x] Document the local five-step edge cap versus reference variable depth.
- [x] Document controlled BM25 versus reference dense FAISS.
- [x] Document corpus exposure and the fullwiki naming caveat.
- [x] Document stage-major batching and retry_count=0.
- [x] Document the pilot threshold and A100 proxy boundary.
- [x] Run fresh documentation terminology and scope checks.
- [x] Run the complete test suite against the implemented pilot/timing paths
  (81 passed, 1 hardware-dependent skip on 2026-08-06).
- [ ] Run an end-to-end mock walk covering deep plans, aggregate, early stop,
  per-document extraction, summary, resume, and answer records.
- [ ] Run pilot, timing, and accuracy plan-only commands.
- [ ] Inspect final diff, hashes, and clean production tree.

Final verification:

    python -m compileall -q src scripts analyze.py smoke_test.py
    python -m pytest -q
    git diff --check

The final handoff must report any remaining difference from reference MA-RAG as
an intentional, measured experiment boundary or an unresolved blocker. No
architectural shortcut may be hidden as an implementation detail.
