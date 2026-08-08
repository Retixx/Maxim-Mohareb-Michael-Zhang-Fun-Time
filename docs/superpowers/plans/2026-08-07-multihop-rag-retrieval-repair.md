# Multi-hop RAG retrieval repair implementation plan

> **For Codex:** Use executing-plans to implement this plan task by task, with
> red/green checks and verification-before-completion before any success claim.

**Goal:** Repair the multi-agent retrieval contract, enforce all SPEC §15 gates,
and restore post-merge integrity without starting SPEC §16 work.

**Architecture:** Keep step 1 as original-question top-10. On every later QA
step, search both the original question and an original-question-anchored Step
Definer task, then expose a stable 7/3 union. Reject stale fingerprints, downgrade
evidence-free aggregate routes, and measure follow-up firing over eligible later
steps only.

**Stack:** Python 3, unittest/pytest, NumPy/SciPy BM25, PyYAML, JSONL artifacts.

---

### Task 1: Pin the retrieval regressions red-first

**Files:**
- Modify: `tests/test_retrieval.py`
- Create: `tests/fixtures/retrieval_headroom_canary.json`

1. Add `test_followup_query_contains_anchor`.
2. Add `test_followup_fires_without_verbatim_grounding`.
3. Add `test_retrieval_unions_both_components`.
4. Add `test_retrieval_headroom_floor` with a small frozen corpus/question
   fixture and the production fusion path.
5. Add aggregate-without-grounded-state and later-step-denominator tests.
6. Run the focused tests and record the expected failures.

### Task 2: Implement the production retrieval and route repair

**Files:**
- Modify: `src/retrieval.py`
- Modify: `src/pipeline.py`

1. Promote 7/3 fusion from archived helper to the active policy.
2. Add `anchor_k`/`task_k` to `RetrievalContext` and its fingerprint.
3. Build later task queries from the exact original question, current task, and
   token-phrase-novel grounded answers.
4. Always attempt both components after step 1 and emit accurate query/component
   telemetry.
5. Downgrade aggregate tasks that lack grounded prior state.
6. Add explicit eligible/fired follow-up counts and correct the legacy rate.
7. Run the focused retrieval tests until green.

### Task 3: Re-freeze and enforce the query-policy fingerprint

**Files:**
- Modify: `config/experiment.yaml`
- Modify: `src/retrieval.py`
- Modify: `src/runner.py`
- Modify: `scripts/run_campaign.py`
- Modify: `scripts/check_pilot.py`
- Modify: `scripts/prefetch_assets.py`
- Modify: `analyze.py`
- Modify: fingerprint fixtures in `tests/`

1. Bump `QUERY_POLICY`; pin 7/3 and no verbatim-grounding requirement.
2. Make every resume/pilot/prefetch/analysis path compare the complete policy.
3. Add `test_query_policy_fingerprint_is_pinned` and stale-policy rejection
   coverage.
4. Run retrieval, campaign, pilot-gate, runner, and analysis tests.

### Task 4: Enforce the statistical acceptance gate

**Files:**
- Modify: `src/metrics.py`
- Modify: `scripts/check_pilot.py`
- Modify: `config/experiment.yaml`
- Modify: `src/contracts.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_pilot_gate.py`

1. Add and unit-test exact two-sided McNemar from paired binary outcomes.
2. Compute paired-bootstrap F1 CI with the frozen resample count and seed.
3. Aggregate hidden-bridge follow-up firing from fired/eligible counts.
4. Enforce +5 overall, CI lower >+2, McNemar p<0.01, +8 hidden, fully
   named within +/-2, and firing >=0.80.
5. Bump the pilot-gate schema and test each fail-closed condition.

### Task 5: Restore the merged 32-arm integrity contract

**Files:**
- Modify: `scripts/run_campaign.py`
- Modify: `scripts/a100_production.py`
- Modify: `analyze.py`
- Modify: affected tests only
- Do not modify: `config/manifests/**`

1. Define the 32 static arms and seven selector tiers from the committed config.
2. Verify the frozen 22-arm manifest unchanged, then append the ten configured
   mid/large arms deterministically across the same six workers.
3. Update selector enumeration from 5^4 to 7^4 and corresponding invariants.
4. Prove there are no duplicate or orphaned arms and no manifest diff.

### Task 6: Add deterministic Gate A and gate-report tooling

**Files:**
- Create: `scripts/check_retrieval_gate.py`
- Modify: focused tests/documentation as needed

1. Load the exact frozen cohort and pinned production corpus.
2. Whitelist query inputs and score gold only after ranking.
3. Exercise the production 7/3 union with an oracle follow-up term on all 1,097
   hidden-bridge questions; assert n>=1000, k=10, both-gold recall>=0.75.
4. Report single, repaired, and fully-named sanity metrics in JSON.

### Task 7: Verify and document the actual diagnosis

**Files:**
- Modify surgically: `SPEC.md` §5, §9, §12, §15.2 and append a dated
  diagnosis/gate record; leave §1-§4 and §16 untouched.

1. Run focused tests, then the full offline suite.
2. Run Gate A before GPU work; stop if it fails.
3. Run Gate B from pilot records.
4. If an exposed <=4 GB NVIDIA GPU is available, run the paired excluded-n>=200
   0.6B/1.7B 4-bit Gate C and post-process Gate D. Never substitute CPU results.
5. Audit every SHA pin, dataset-block identity, arm mapping, off-lineage result
   exclusion, and unchanged manifests. Flag model revisions as TBD without
   resolving them.
6. Rewrite §15.2 to distinguish confirmed causes from the rejected passage-name
   hypothesis and record exact gate results.

### Task 8: Rebase/merge proof and handoff

1. Rebase onto `origin/main` if needed; never merge main into the branch.
2. Confirm `git merge-base --is-ancestor origin/main HEAD`.
3. From a temporary main worktree, run `git merge --no-commit --no-ff` against
   the branch, inspect, and abort.
4. Confirm `git diff f92391b..HEAD -- config/manifests/` is empty.
5. Commit clear, scoped changes and create a Git bundle/patch inside the repo.
6. Print the exact SSH download command. Do not begin SPEC §16.
