# SLM MA-RAG Accuracy Repair Design

## Goal

Repair the proven multi-agent accuracy failures introduced by the variable-depth
executor while preserving the frozen questions, corpus, 22-arm treatment matrix,
model revisions, quantization recipes, no-retry rule, metrics, and pilot gate.

The implementation is a linear descendant of GitHub `no-bs` commit
`298137d23d6c0b0df898c68b5718f985371a97dd`. The deliverable must fast-forward
that exact commit without merging, cherry-picking, or force-pushing.

## Proven Failure Boundaries

1. Multi-agent retrieval uses model-generated Step Definer text while the solo
   control uses the original question. A valid but unresolved later task such as
   `That city is on what bay?` is searched without the grounded prior answer.
2. QA guesses are appended to state without distinguishing answers supported by
   retrieved evidence from parametric guesses. An unsupported bridge can poison
   every later task and query.
3. Extractor v5 was validated on a different input shape. The current per-document
   caller repeats the same task as sub-question, target entity, and search terms.
   Whole multi-sentence passages can pass the current protocol check because it
   checks only verbatim copying and span count.
4. QA renders every empty per-document extraction as `(no supporting text found)`.
   Ten retrieved documents therefore become a prompt dominated by negative
   padding even when useful evidence exists.
5. A missing, malformed, or empty plan summary destroys a usable intermediate QA
   answer and forces an automatic empty prediction.
6. The excluded smoke driver on `smoke-handoff-20260807` has a syntax error. It is
   diagnostic tooling, not campaign code, and must be repaired separately without
   merging that branch into the experiment branch.

## Frozen Repair Policy

### Anchored retrieval

Each question-answering step exposes at most ten fused documents to Extractor and
QA.

- Step 1, or any later step with no evidence-grounded prior answer, uses the
  original-question top 10. This prevents a weak generated query from making the
  first retrieval worse than the solo query.
- A later step with evidence-grounded prior answers searches two components:
  the original question and the Step Definer task augmented with those grounded
  answers.
- Fusion keeps the first seven original-question results, then the first three
  unique task-query results. If the task component cannot supply three unique
  results, remaining original-question results fill the ten slots, followed by
  remaining task results only if necessary.
- Gold labels and the hidden/fully-named stratum never enter query construction.
- The 7/3 rule is frozen before the pilot. It is an architectural prior from the
  excluded retrieval work, not selected on answer F1.

Every retrieval event logs the raw task, grounded answers, component queries,
component rankings, fused ranking, query count, and quotas. Answer records report
fused, anchor, and task gold-title recall separately.

### Grounded state propagation

A QA answer is evidence-grounded only when its normalized text occurs in an
evidence span or prior grounded answer actually supplied to that QA call. The
classification is deterministic and logged.

- Step Definer sees grounded prior answers. Unsupported guesses are labelled and
  withheld as retrieval facts.
- The targeted retrieval component uses only grounded prior answers.
- Aggregate QA evidence contains only grounded prior answers.
- Plan summary may see unsupported QA candidates, but they are explicitly marked
  as guesses and the prompt instructs it to prefer evidence-grounded results.
- QA retains the reference MA-RAG general-knowledge fallback. The guard changes
  propagation, not whether QA may emit a short fallback answer.

### Extractor normalization

Per-document extraction remains intact. The prompt is updated to describe the
actual one-document input and removes obsolete duplicated target/search fields.

After parsing or salvage, each candidate span is deterministically matched to the
source document's exact sentence list. A candidate is accepted only when it maps
unambiguously to one source sentence; a whole multi-sentence passage maps to more
than one sentence and is rejected. Accepted output is the exact source sentence,
deduplicated and capped at three. Raw output, original parsed payload, normalized
consumer payload, rejection counts, and parse status are all retained.

This is post-hoc normalization, not constrained decoding. It performs no retry,
regeneration, grammar masking, logits processing, or gold-based repair.

### QA evidence rendering

QA receives only non-empty normalized evidence blocks. Empty document outcomes
remain in `consumer_input` telemetry, so extraction failures and retrieval
exposure are still measurable. Duplicate spans are removed across documents.
If every extraction is empty, QA sees one `(no evidence collected)` marker rather
than ten negative padding lines.

The QA worked example is revised to match the real current-step document-labelled
layout. The same prompt is used at every size and precision.

### Final-answer degradation

Plan summary remains the required finalizer and is always attempted. When its
parsed/salvaged answer is usable, it remains the scored answer. Only when that
answer is missing or a known no-answer sentinel does scoring fall back
deterministically to the last usable non-empty QA answer. `answer_stage` remains
`plan_summary` for pilot-contract compatibility, while `final_answer_source`
records whether the value came from parsed summary, salvaged summary, or QA
fallback. No fallback is claimed correct merely because it is non-empty.

## Scientific Integrity

- Bump the prompt bundle, affected role versions and hashes, retrieval policy,
  and experiment schema to `open_corpus_marag_v2`.
- Bind fusion quotas into retrieval metadata and validate them in the runner,
  campaign resume logic, pilot gate, and analysis loader.
- Artifacts from `298137d` cannot resume into or be analyzed as repaired runs.
- Do not alter final, pilot, preflight, timing, or exclusion manifests.
- Do not alter treatment assignment, quantization, model loading, batching,
  generation ceilings, scoring formulas, bootstrap procedures, or gate thresholds.
- Do not use final/pilot answer F1 to tune any repair.

## Verification

1. Each failure boundary receives a regression test that fails on `298137d` and
   passes after its isolated repair.
2. All repository tests pass in the existing `.venv`.
3. Python compilation succeeds for source, scripts, tests, and analysis.
4. Static checks prove no constrained decoding, retry, manifest, matrix, or gold
   query dependency was added.
5. A disposable clone of `298137d` accepts the final branch with `--ff-only` and
   produces the exact same tree.
6. The Git bundle verifies and advertises one repair ref descended directly from
   `298137d`.

## Expected Interpretation

These changes remove known implementation handicaps; they do not guarantee that
an untrained sub-4B model clears the pilot. A post-repair excluded smoke run is
still required to separate remaining model-capability limits from implementation
defects. The only acceptable completion claim is that the repaired pipeline now
measures the intended architecture and that the tested failure modes no longer
occur.
