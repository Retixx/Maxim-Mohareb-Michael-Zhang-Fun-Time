# Multi-hop RAG retrieval repair design

**Date:** 2026-08-07
**Scope:** SPEC §15 only. SPEC §16 additive campaign work is explicitly excluded.

## Diagnosis

The paired n=731 symptom reproduces exactly: multi-agent F1 is 0.4125,
single-hop F1 is 0.5420, and the paired delta is -0.1295 (89 wins, 202
losses, 440 ties). The defect is upstream information destruction, not answer
verbosity, parser collapse, or reader scale.

An instrumented step-2 trace found three coupled production defects:

1. A grounded follow-up replaces the original-question ranking and its query
   omits the original question.
2. A follow-up is attempted only when a prior answer passes a literal token-
   phrase grounding test; otherwise step 2 repeats step 1's query and ranking.
3. The event schema declares anchor and task components, but production attempts
   only one of them. The existing deterministic fusion helper is unused.

Two adjacent defects amplify or conceal those failures:

- A parsed `aggregate` task is accepted without any grounded prior state. It
  skips retrieval and Extractor, then sends QA an empty evidence set.
- The follow-up firing denominator includes step 1. A perfect two-step execution
  therefore reports 0.5 instead of 1.0, contrary to Gate B's "beyond step 1"
  denominator.

The Extractor normalization and QA evidence filter behave consistently with
their contracts; neither can recover a gold passage that retrieval discarded.
Plan depth is not the primary cause: nearly all measured historical plans had
more than one step.

## Independent retrieval evidence

The production BM25 implementation, pinned 72,094-passage corpus, frozen 1,500
questions, and strict post-query gold scoring give:

| Policy | hidden_bridge both-gold | fully_named both-gold |
|---|---:|---:|
| original question, top 10 | 0.5077 | 0.8412 |
| oracle anchored union, 5/5 | 0.9380 | 0.8412 |
| oracle anchored union, 6/4 | 0.9362 | 0.8412 |
| oracle anchored union, 7/3 | 0.9298 | 0.8412 |

The 7/3 split is selected because it remains far above Gate A's 0.75 floor
while protecting seven original-question slots against live task-query drift.
The 5/5 oracle gains only 0.8 recall points.

The passage-derived hypothesis in the original §15.2 does not survive a no-gold
benchmark. Raw or guarded passage-name heuristics peak at 0.6044 hidden-bridge
recall and reduce fully-named recall by 3.47 points. Gold-supporting-sentence NER
probes are not live-valid. Passage-name expansion is therefore rejected in this
repair rather than promoted to production without evidence.

## Retrieval policy

The new policy preserves the ten-passage exposure ceiling per QA step:

- Step 1 searches the original question once and exposes its top 10.
- Every later `question-answering` step performs two rankings:
  - original question, top 10;
  - original question + current Step Definer task + non-duplicate grounded prior
    answers, top 10.
- The exposed ranking is a stable, deduplicated 7/3 union. Unused task quota is
  backfilled from the anchor ranking, and exposure never exceeds 10.
- The task component fires regardless of literal answer grounding. Grounding is
  retained as evidence-trust telemetry only.
- Query construction whitelists the original question, current task, and prior
  runtime state. Gold answers, supporting facts, strata, hidden titles, and
  dataset sentence indexes never enter it.

The component keeps the historical telemetry name `grounded_step_task` for
artifact continuity, although it is no longer gated on grounding. New policy
fields make that semantic change explicit.

## Routing and telemetry

A requested aggregate is valid only when at least one grounded prior answer is
available. Otherwise `_step_task` downgrades it to `question-answering`, keeping
the task text and forcing evidence acquisition.

Answer records add explicit later-step eligible and fired counts. The firing
rate is `sum(fired) / sum(eligible)` over QA steps with `step > 1`; step 1 and
aggregate steps are excluded. This supports an exact stratum-level Gate B rather
than an average of per-question ratios.

## Fingerprint and stale-artifact handling

`retrieval.QUERY_POLICY` is bumped. The retrieval fingerprint pins `anchor_k=7`,
`task_k=3`, and `grounded_followup_requires_evidence=false` in addition to the
existing corpus, algorithm, k, and initial-query fields. Resume, pilot-gate,
prefetch, runner, and analysis validators compare all of them and reject old
artifacts. No cohort, corpus, plan ceiling, retrieval k, or manifest changes.

## Acceptance enforcement

The pilot checker enforces every SPEC §15.5 gate:

- overall paired F1 delta at least +5.0 points;
- paired-bootstrap 95% lower bound above +2.0 points;
- exact two-sided McNemar p-value below 0.01 using EM discordance;
- hidden-bridge F1 delta at least +8.0 points;
- fully-named F1 delta within +/-2.0 points;
- hidden-bridge later-step follow-up firing rate at least 0.80.

The required retrieval regression tests are written red-first. A fixture-backed
headroom canary invokes the same fusion primitive as production. The full frozen
Gate A runner uses an oracle hidden-title follow-up only to test the deterministic
retrieval mechanism; it does not make a live query-quality claim. Gate C remains
the end-to-end model check.

## Integrity repair

The current config already defines the merged 32-arm matrix and seven selector
tiers, while campaign/analysis code and tests still enforce 22 arms and five
tiers. The repair propagates those existing definitions without editing frozen
manifests: the immutable 22-arm plan is verified first, then the ten already-
configured mid/large arms are assigned deterministically in code. This closes
the post-merge integrity failures but does not perform the additive 44-arm work
in SPEC §16.
