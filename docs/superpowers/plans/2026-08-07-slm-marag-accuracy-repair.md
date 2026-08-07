# SLM MA-RAG Accuracy Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]` and `- [x]`) syntax for tracking.

**Goal:** Remove the proven retrieval, state-propagation, extraction, evidence-rendering, and finalization defects without changing the frozen experimental sample or treatment matrix.

**Architecture:** Preserve variable-depth per-document MA-RAG, but anchor each retrieval step on the original question, admit only evidence-grounded bridge state into targeted queries, normalize Extractor output against exact source sentences, remove empty evidence padding, and degrade a failed finalizer to a logged intermediate candidate. Version every behavioral contract so old artifacts cannot mix with repaired runs.

**Tech Stack:** Python 3.12, NumPy/SciPy BM25, PyTorch/transformers, YAML, unittest, Git bundles.

---

### Task 1: Add sentence-level extraction normalization

**Files:**
- Create: `src/extraction.py`
- Modify: `src/agents.py`
- Modify: `src/mechanism.py`
- Test: `tests/test_extraction.py`
- Test: `tests/test_execution_integrity.py`

- [x] **Step 1: Write failing tests** for exact-sentence acceptance, unique-fragment expansion, multi-sentence echo rejection, ambiguity rejection, three-span cap, original-payload preservation, and normalized consumer payload.
- [x] **Step 2: Run** `.venv/bin/python -m unittest -v tests.test_extraction` and confirm the missing-module/function failures.
- [x] **Step 3: Implement** `normalize_spans(spans, source_sentences, limit=3)` returning normalized spans plus accepted/rejected telemetry, and wire it into `agents.run_calls` without changing raw `parsed`, `salvaged`, or `parse_status`.
- [x] **Step 4: Make** Extractor `protocol_ok` require every raw span to map unambiguously to one exact source sentence.
- [x] **Step 5: Run** the focused extraction and execution-integrity tests.
- [x] **Step 6: Commit** with `fix: normalize extractor evidence to source sentences`.

### Task 2: Implement grounded state and anchored 7/3 retrieval

**Files:**
- Modify: `src/retrieval.py`
- Modify: `src/pipeline.py`
- Modify: `src/agents.py`
- Modify: `src/runner.py`
- Modify: `config/experiment.yaml`
- Test: `tests/test_retrieval.py`

- [x] **Step 1: Write failing tests** proving unresolved later tasks include grounded bridge answers, unsupported guesses never enter targeted queries, step 1 equals original-question top 10, later fusion preserves seven anchor results and adds three unique task results, and fusion always exposes at most ten passages.
- [x] **Step 2: Run** the focused retrieval tests and confirm failures on the shipped Step-Definer-only policy.
- [x] **Step 3: Implement** deterministic answer grounding from the evidence actually consumed by QA and grounded-only Step Definer state rendering.
- [x] **Step 4: Implement** `fuse_rankings` and a retrieval event with anchor/task queries, component rankings, quotas, fused titles, and actual query count.
- [x] **Step 5: Bind** `anchor_k: 7`, `task_k: 3`, and the new policy string in config, runtime fingerprint, and runner validation.
- [x] **Step 6: Run** focused retrieval, contract, and campaign tests.
- [x] **Step 7: Commit** with `fix: anchor multistep retrieval on grounded state`.

### Task 3: Repair Extractor and QA prompt/input alignment

**Files:**
- Modify: `src/prompts.py`
- Modify: `src/agents.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_execution_integrity.py`
- Test: `tests/test_retrieval.py`

- [x] **Step 1: Write failing tests** proving Extractor no longer receives duplicated target/search fields, QA prompt blocks omit empty documents and cross-document duplicate spans, empty retrieval renders a single no-evidence marker, and the QA worked example uses the real document-labelled current-step layout.
- [x] **Step 2: Run** the focused tests and confirm the shipped input-shape failures.
- [x] **Step 3: Revise** only the affected Extractor, QA, and plan-summary templates; bump their role versions and the bundle version uniformly across precisions.
- [x] **Step 4: Keep** every empty/filtered document in `consumer_input` telemetry while excluding it from the rendered QA evidence.
- [x] **Step 5: Recompute** and pin all prompt hashes, proving unaffected prompts remain byte-identical.
- [x] **Step 6: Run** prompt, retrieval, parsing, and contract tests.
- [x] **Step 7: Commit** with `fix: align small-model evidence prompts with runtime inputs`.

### Task 4: Add deterministic final-answer degradation

**Files:**
- Modify: `src/pipeline.py`
- Modify: `scripts/check_pilot.py`
- Test: `tests/test_execution_integrity.py`
- Test: `tests/test_pilot_gate.py`

- [x] **Step 1: Write failing tests** for parsed summary precedence, salvaged summary precedence, empty/malformed summary fallback to the last usable QA answer, no-answer-sentinel rejection, and pilot acceptance of a logged QA fallback whose finalizer stage was still plan summary.
- [x] **Step 2: Run** the focused tests and confirm the shipped empty-answer behavior.
- [x] **Step 3: Implement** `usable_short_answer` and `final_answer_for`, retaining `answer_stage=plan_summary` and adding `final_answer_source` plus grounding telemetry.
- [x] **Step 4: Run** execution-integrity and pilot-gate tests.
- [x] **Step 5: Commit** with `fix: preserve usable answer when final summary degrades`.

### Task 5: Version telemetry and analysis contracts

**Files:**
- Modify: `src/runner.py`
- Modify: `scripts/check_pilot.py`
- Modify: `scripts/run_campaign.py`
- Modify: `analyze.py`
- Modify: `tests/test_campaign.py`
- Modify: `tests/test_pilot_gate.py`
- Modify: `tests/test_analyze.py`

- [x] **Step 1: Write failing tests** that reject v1/stale fusion metadata and accept only `open_corpus_marag_v2` with exact quotas and prompt hashes.
- [x] **Step 2: Run** the focused campaign, gate, and analysis tests and confirm the stale schema remains accepted before the fix.
- [x] **Step 3: Bump** the experiment schema and validate the full retrieval identity in resume, gate, and analysis paths.
- [x] **Step 4: Add** anchor/task recall, step count, actual component-query count, grounding rate, normalization, and final-answer-source diagnostics without altering F1/EM.
- [x] **Step 5: Run** all analysis, campaign, and pilot tests.
- [x] **Step 6: Commit** with `fix: bind repaired pipeline into experiment identity`.

### Task 6: Update the scientific contract and verify the branch

**Files:**
- Modify: `SPEC.md`
- Modify: `RUNBOOK.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-07-slm-marag-accuracy-repair-design.md`
- Modify: `docs/superpowers/plans/2026-08-07-slm-marag-accuracy-repair.md`

- [x] **Step 1: Document** anchored fusion, grounded propagation, normalization, empty-block filtering, fallback semantics, and the explicit no-constrained/no-retry guard.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest discover -s tests -v` and require zero failures.
- [ ] **Step 3: Run** `.venv/bin/python -m compileall -q src scripts tests analyze.py` and require exit 0.
- [ ] **Step 4: Audit** the diff for manifest, run-matrix, model, quantization, metric, bootstrap, gate-threshold, retry, constrained-decoding, and gold-query changes.
- [ ] **Step 5: Verify** all commits descend from `298137d` and the worktree contains no tracked changes.
- [ ] **Step 6: Create and verify** a one-ref Git bundle; apply it to a disposable clone at `298137d` using `git merge --ff-only`, then compare tree hashes.
- [ ] **Step 7: Report** the exact commit, bundle SHA-256, verified base, changed behavior, unchanged experiment contracts, and the remaining capability-floor uncertainty.
