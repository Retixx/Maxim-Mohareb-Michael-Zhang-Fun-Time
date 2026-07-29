# PROGRESS

Handoff log. Read SPEC.md first, then the newest entry here, then `git log`.

---

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
