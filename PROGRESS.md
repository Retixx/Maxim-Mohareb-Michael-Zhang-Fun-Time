# PROGRESS

Handoff log. Read SPEC.md first, then the newest entry here, then `git log`.

---

## 2026-07-29 (later) — build steps 3-7 done, prompts frozen at v5, at GATE 1 again

**Done.** Build steps 3-7: stage-major runner (`src/runner.py`, `src/pipeline.py`),
checkpoint/resume, precision switching via `config/experiment.yaml`, batch-size OOM
autotune, Kaggle notebook. Prompts settled at **v5 and FROZEN** — do not edit them.
Constrained/grammar decoding is now explicitly forbidden in SPEC §12 and guarded in
`models.py` and `parsing.py`; it would zero the parse-failure metric by construction.

**Prompt story, so nobody repeats it.** The one-shot example added to the Extractor
in v2 was not a fix, it WAS a 23-point regression. Measured head-to-head on the same
68 extractor calls: v1-original 95.6%, v4-with-example 72.1%, bare-array variant
89.7%. v1 wording restored. Two intervening hypotheses (v3, v4) were tested and both
wrong. The Extractor is deliberately the only role without a worked example.
Failure sets are prompt-dependent, not item-dependent — zero calls failed under all
four variants (Jaccard 0.00), so there is no unparseable subset of HotpotQA.

**Gate 1 re-run** (`baseline`, n=30, seed 1234, batch 16, FP16): 30/30 completed.
Parse success planner 100%, step_definer 98.5%, extractor 95.5%, qa 100% — all clear
the §5a 90% bar. EM 20.0% [95% CI 6.7-33.3], F1 33.8% [19.7-49.3].
Resume verified on real data: rerun found 192 calls, skipped all four stages, wall 0s.

**Next.** Waiting at Gate 1 for human approval to start build step 8 (n=300, five
runs, eval_seed 7). Do not start it unprompted.

**Known issues / open questions.**
- **Local VRAM readings are not real.** Extractor peak was 6516 MB on a 4096 MB card:
  Windows WDDM silently spills CUDA allocations into system RAM instead of raising
  OOM. Consequences: (a) the OOM autotune path has NEVER actually fired and is
  unvalidated, (b) batch 16 made the extractor *slower* locally (19.7 s/call vs
  11.6 s/call at batch 1), (c) the §13 "under 6 GB at FP16 batch 16" criterion reads
  as 6516 MB and cannot be honestly checked on this machine. All three need
  re-measuring on the Kaggle T4 (Linux, no such fallback) via the notebook's
  10-question cell before any full sweep.
- Local throughput extrapolates to 4.07 GPU-h per 300-question run, which would blow
  §13's 4 GPU-h budget for the whole 5-run tier. That number is inflated by the
  memory spill above; the real figure must come from the T4.
- EM 20.0% sits in §5a's "concerning" band (20-30%), below "healthy" (30-45%), though
  the n=30 CI spans both. 6 of 30 answers are granularity near-misses (EM 0 with
  F1>=0.5), e.g. "Gainesville" vs "Gainesville, Florida". Prompts are frozen, so this
  is logged as an observation, not something to tune away.

## 2026-07-29 — build steps 1-2 complete, at GATE 1 (awaiting human)

**Done.** Repo created at `C:/Users/maxim/Projects/marag-precision` (new git repo; the
session's original cwd was an unrelated VST plugin project). Local env is a Python
3.11 venv at `.venv/` — torch 2.6.0+cu124, transformers 5.14.1, bitsandbytes 0.50.0,
datasets 5.0.1, CUDA visible on an RTX 3050 Laptop (4 GB).
Build step 1 (FP16 pipeline, hardcoded, 10 questions, all raw outputs printed) and
build step 2 (parsers + six-label failure taxonomy, zero retry paths) are done:
`src/prompts.py`, `src/parsing.py`, `src/models.py`, `src/agents.py`,
`src/metrics.py`, `src/pipeline.py`, `smoke_test.py`.

**Gate 1 smoke test ran** — `python smoke_test.py --n 10`, FP16, seed 0. 10/10
questions completed end-to-end. Parse success: planner 100%, step_definer 100%,
extractor 95.5%, **qa 70%**. EM 20.0%, F1 29.5%. Peak VRAM 3905 MB. 2.19
extrapolated GPU-hours per 300-question run at batch_size=1.

**Next.** Blocked at Gate 1 — recommendation is FIX FIRST (QA parse success 70% is
below the §5a 90% threshold). Proposed fix is a one-shot example in the QA and
Extractor prompt templates, applied uniformly at all precisions (§5a sanctions
prompt work to get baseline parse failure under 10%; §12 forbids only *per-precision*
tuning). Do not start build step 3 until a human approves.

**Known issues / open questions.**
- Single dominant failure mode: the model emits unquoted JSON string values,
  e.g. `{"answer": Richard Strauss}`. All 3 QA failures and the 1 extractor failure
  are this. All 3 QA failures score EM 0 with an empty answer, so EM 20% understates
  the pipeline; one of the three would have been an exact match.
- QA sometimes answers in a full sentence despite the prompt forbidding it (4 of 10),
  costing EM but earning partial F1.
- Peak VRAM 3905 MB of 4096 MB at FP16 with batch_size=1. Batch 16 at FP16 will not
  fit locally; that is a Kaggle/T4 configuration only. Local stays batch 1.
- Unbatched throughput (2.19 GPU-h/run x 5 runs) exceeds the §13 budget of 4 GPU-h
  for the whole 4-bit tier. Build step 6 (batching) is required to meet it, as
  SPEC §6 anticipates.
- Degraded-propagation policy on parse failure is documented at the top of
  `src/agents.py`. It performs no re-generation. Worth a human sanity check.
