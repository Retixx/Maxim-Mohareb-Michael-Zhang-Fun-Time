# Pre-Pilot Integrity Fixes Design

## Goal

Remove three known, avoidable handicaps from the frozen MA-RAG campaign before
the scored pilot while preserving its questions, corpus, retrieval topology,
arm matrix, no-retry rule, and role-allocation estimands.

The implementation must be a direct descendant of GitHub `no-bs` commit
`8cefd9247497f4ed96b745f679d24cc31a2eab07`. The deliverable is a Git bundle
whose branch can be applied to `no-bs` with `--ff-only`; it must never require a
merge commit or conflict resolution.

## Scope

### 1. Restore the Extractor generation budget

Set `MAX_NEW_TOKENS["extractor"]` to 320 for every treatment. This restores the
budget present in both parents of merge `d57b7c9`; the merge's unvalidated
reduction to 128 is not retained.

Generation stops normally when the model emits EOS. The larger value is only a
ceiling and does not force every Extractor call to generate 320 tokens. The
one-document-per-Extractor-call architecture remains unchanged.

Add the complete `MAX_NEW_TOKENS` mapping to the experiment fingerprint and to
the pilot/campaign fingerprint verification. This prevents artifacts generated
with different role budgets from being combined even if their prompt text is
identical.

### 2. Fail closed when requested quantization is absent

After each model load and before any warm-up, preflight, timing, or scored
generation, validate the loaded parameters against the requested precision:

- FP16 must have zero bitsandbytes quantized parameters.
- Q8 and Q4 must have a positive, substantial quantized-parameter population,
  not merely quantization metadata in configuration.
- A mismatch raises a descriptive `RuntimeError` containing the model ID,
  requested precision, nominal parameter count, quantized parameter count, and
  quantized fraction.

The guard uses the existing `param_census` tensor inspection, not configured
precision labels or reported memory alone. Its validated census is recorded in
model-load/stage metadata so the paper artifact proves that quantization was
materially applied.

This guard changes no successful model output. It only stops an invalid arm
before it can create scored records.

### 3. Remove the MA-versus-single fallback asymmetry

Change only the MA QA prompt so it follows reference MA-RAG behavior:

- use retrieved evidence first;
- when evidence is insufficient, produce the best short answer available from
  the model's knowledge;
- emit `success=no` only when it cannot produce a usable short answer.

The single-agent prompt already requests a best short guess and remains
unchanged. The semantic stop mechanism remains unchanged: a genuine
`success=no` still stops later plan steps. The plan-summary finalizer remains
the scored output and its implementation remains unchanged.

Because QA prompt text changes, bump the QA prompt version and the prompt bundle
version. Planner, Step Definer, Extractor v5, plan-summary, and solo prompt text
and per-prompt versions remain unchanged.

## Explicit Non-Goals

This patch does not:

- add deterministic fallback from an empty plan summary to an intermediate QA
  answer;
- tune against the 100-question smoke results;
- modify Planner depth, semantic stopping, plan-summary routing, parsing,
  salvage, retries, constrained decoding, retrieval, corpus construction, or
  scoring;
- modify the final, pilot, preflight, or timing manifests;
- modify any model assignment, quantization recipe, arm, batch size, seed, or
  analysis estimand; or
- include smoke artifacts in the experiment branch.

Empty-answer policy and other smoke findings will be evaluated separately from
raw JSONL/meta artifacts before any additional proposal.

## Files and Responsibilities

- `src/prompts.py`: restore the Extractor ceiling, revise only the MA QA
  fallback instructions, and bump only the affected prompt/bundle versions.
- `src/models.py`: expose one fail-closed loaded-precision validator using the
  existing parameter census.
- `src/runner.py`: invoke validation immediately after load, persist its proof,
  and bind generation budgets into the experiment fingerprint.
- `scripts/check_pilot.py`: reject a pilot whose fingerprint generation budgets
  do not equal the current frozen mapping.
- `scripts/run_campaign.py`: reject completed-arm metadata whose fingerprint
  generation budgets are stale.
- `SPEC.md`: record the generation ceilings, loaded-precision validation, and
  evidence-first/reference fallback behavior.
- `tests/`: add regression coverage for all three changes and for stale-budget
  artifact rejection.

## Error Handling

Precision validation is fail-closed and occurs before generation. A stale
fingerprint is rejected rather than repaired. Existing interrupted-run resume
rules remain unchanged; old 128-token artifacts cannot be resumed into a
320-token run because their experiment fingerprint and environment lock differ.

## Verification

Verification must establish all of the following from fresh commands:

1. Regression tests fail against the old behavior and pass against the patch.
2. All repository tests pass in the pinned environment.
3. Static compilation succeeds.
4. The exact diff contains no manifest, retrieval, dataset, matrix, stopping,
   finalization, or scoring changes.
5. The resulting commit is a direct descendant of `8cefd924`.
6. `git bundle verify` succeeds and the bundle advertises the intended branch.
7. Applying the bundle to a disposable clone of `8cefd924` with `git merge
   --ff-only` succeeds, and the resulting tree equals the exported commit.
8. Live GitHub `no-bs` still points to the expected base immediately before
   handoff. If it moved, do not provide merge instructions; rebuild the patch on
   the new tip or stop for review.

## A100 Handoff

After the fast-forward push, the A100 operator must run the normal prepare phase
to generate a new environment lock bound to the changed source bundle, then run
the excluded pilot. No old pilot certificate or partial accuracy artifact is
valid for the new fingerprint.
