# PROGRESS

Handoff log. Read SPEC.md first, then the newest entry here, then `git log`.

---

## 2026-07-29 (latest) — EXTRACTION ACCURACY added; first significant per-role effects

**Read SPEC §5c.** New third metric (`src/evidence.py`): the Extractor's spans scored
against HotpotQA's gold `supporting_facts` labels. Human-approved addition. Pure
re-analysis of existing runs — nothing reaches a prompt, generation unaffected.

**Why it mattered.** Nothing measured extraction *correctness*. A model verbatim-copying
an irrelevant sentence scored perfectly on parse status, verbatim_rate and churn. So
"73.8% of Extractor calls select different evidence at 4-bit" could not be turned into
"and the new evidence is worse". Now it can.

**Result — the first per-role effects with CIs excluding zero:**

    Extractor      ev-F1 drop +4.66 [+1.17, +8.08]   ROBUST (significant at every threshold)
    Step Definer   ev-F1 drop +4.38 [+1.97, +6.88]   THRESHOLD-DEPENDENT (gone below 25 chars)
    Planner        ev-F1 drop +0.00 [-3.45, +3.43]   null
    QA             ev-F1 drop +0.00 [+0.00, +0.00]   zero by construction (negative control)

Both positive roles are the format-heavy ones, which is what SPEC §1 predicted — but
only the Extractor survives the sensitivity sweep. Report Step Definer as provisional.

**QA = exactly 0.00 is a built-in negative control that PASSES.** QA runs after
extraction so quantizing it cannot change extractor output. A non-zero value there means
the metric is broken — check that first. It also confirms generation is deterministic
under identical batch composition.

**Metric design constraint — do not violate.** Scoring is set-F1 over discrete
(title, sent_id) labels, NEVER token overlap. Token-F1 on sentence-length references
ranks an incomplete verbatim copy (0.59) above a half-copy-plus-fabrication (0.51) above
a complete faithful paraphrase (0.46), because precision taxes synonyms and lies
identically. Label comparison is immune. If someone "improves" this to text similarity,
they have reintroduced that bug.

**Limitation, stated:** 26% of spans too short to attribute, 17% unmatched, so absolute
ev-F1 (37.5%) is a floor. Deltas are valid — attribution quality is near-identical
across runs.

---

## 2026-07-29 (earlier) — mechanism reframed and PRE-REGISTERED; calibration instrumented

**Read SPEC §5b before touching any mechanism metric.** SPEC §1's format-damage claim was
refuted at 4-bit on model 1 by three independent instruments, all with the point estimate
in the WRONG direction: parse success -0.79 pp [-2.86, +1.28], strict format (no parser
tolerance) +0.16 [-1.11, +1.43], verbatim span fidelity -2.48 [-5.48, +0.53]. Baseline
failure rates are 0.3-4.3%, so there is no dynamic range left — this is an
instrumentation floor, not a power problem, and n=750 makes it SHARPER not softer (QA
projects to [-0.65, -0.01], i.e. significantly *better* under 4-bit). **Do not keep
re-measuring format hoping for a different answer.**

What the data does support: **selection perturbation without quality degradation.**
Selection churn under 4-bit is planner 88.7%, extractor 73.8%, step_definer 59.0%,
qa 26.0% — while format and fidelity hold and accuracy moves <=4 pp. Quantization changes
WHICH content is chosen, not how well-formed it is.

That reframe was found by testing three metrics on the same data, so it is exploratory.
SPEC §5b now records four falsifiable predictions committed before the confirmatory
analysis. Note model 2 at n=750 was already running when this was written, so it is
analyst-blind but NOT pre-data — the planned n=5000 rerun is the confirmatory test.

**New code.** `src/mechanism.py` (strict_format_ok, verbatim_rate, selection_changed —
all pure re-analysis of existing JSONL fields, no GPU). `metrics.auroc` and
`metrics.expected_calibration_error`. `models.sequence_confidence` logs per-call
mean/min token logprob and mean entropy so the CALIBRATION half of §1 — never measured
until now — can finally be tested. `gate2_report.py` grew Table 3 for all of this.

**generation.log_confidence is OFF by default and must stay off until the n=5000 rerun.**
It costs an extra prefill per batch; enabling it mid-project would make wall-time
comparisons across runs meaningless. Verified it cannot affect results: generations are
bit-identical with the flag off vs on (greedy, and the confidence pass runs strictly
after generation). It is a teacher-forced forward pass, deliberately NOT a
LogitsProcessor, so there is no ambiguity with SPEC §12's constrained-decoding ban.

**Safe to pull mid-sweep** — nothing in the generation path changed with the flag off.

---

## 2026-07-29 (earlier) — n raised to 750, awaiting re-run on Kaggle

**Human decision at Gate 2: n raised from 300 to 750.** The n=300 tier was a null with
every CI spanning zero; measured SE of the EM drop is 2.21 pp, so ~720 questions give
80% power against a true 4 pp effect. MA-RAG evaluated on 5600 HotpotQA dev questions
(18.7x our original n), which largely explains why they resolved a role ranking and we
did not. Recorded as an explicit amendment in SPEC §5 so a later session does not
"restore" 300; config/experiment.yaml is authoritative at n=750.

**Fits Kaggle comfortably**: ~4.0 GPU-h for all five runs (measured 1.61 h at n=300),
~48 min per run, longest ~51 min against a 12-h session cap, 13% of the weekly quota.
This makes SPEC §13's "inside 4 GPU-hours" marginal by construction — judge against ~4 h.

**The n=300 sample is a strict SUBSET of the n=750 sample.** random.sample with a fixed
seed is nested for increasing k (verified: overlap 300, nested True). So do NOT pool the
two — pooling would double-count all 300. n=750 supersedes n=300 outright. The n=300
records are also NOT reused to seed the n=750 files despite matching question ids,
because at n=750 a question sits in a different batch with different padding neighbours;
greedy decoding is deterministic for a fixed batch, not across re-batchings. Comparing
the shared 300 afterwards is a free check on exactly that.

**Next.** Human re-runs the Kaggle notebook: cell 2 to pull the new config, then cell 5.
New outputs land as *_n750_seed7.*. Then re-run gate2_report.py with --n 750 --seed 7.
Model 2 (build step 9) comes after, still gated on a human decision.

---

## 2026-07-29 (earlier) — BUILD STEP 8 COMPLETE, at GATE 2

**Done.** Full 4-bit tier at n=300, seed 7, on a Kaggle T4. All five runs complete:
300 answers each, identical question sets, zero duplicate call keys, `stage_precision`
verified correct in every metadata blob, same library versions throughout, prompts v5,
commit 45cd985. **Whole sweep took 1.61 GPU-hours** (SPEC §13 budgets 4). Peak VRAM
5898 MB at FP16 batch 16 (§13 wants <6 GB) — batch 16 held on every stage, autotune
never needed. Results are in `results/*_n300_seed7.*` (gitignored); Gate 2 analysis is
`gate2_report.py`.

**Headline: baseline EM 34.7% [29.3, 40.0], F1 44.4% — squarely in §5a's healthy band.**

**The result is a null, and it is not a bug.** No role's EM drop has a CI excluding
zero: Extractor +4.00 [-0.33, +8.33], Step Definer +1.00 [-2.00, +4.00], QA
+0.67 [-2.67, +4.00], Planner -1.33 [-5.67, +3.00] pp. Verified quantization really
applied: 83-91% of each quantized stage's raw outputs differ from baseline, and 28-44%
of final answers change. Churn is near-symmetric (e.g. planner 21 right->wrong vs 25
wrong->right), which is *why* net EM barely moves — 4-bit scrambles individual answers
without systematically degrading them.

**The hypothesised mechanism did not appear.** SPEC §1 predicts quantization damages
output format. Parse-failure rates were essentially unchanged, and if anything slightly
*lower* when quantized: planner -0.67 [-2.67, +1.33], step_definer -0.48
[-1.43, +0.48], extractor -0.79 [-2.86, +1.27], qa -0.33 [-1.00, 0.00] pp. This is the
metric §5 calls "as important as accuracy", and it shows no format damage at 4-bit.

**Pre-specified §10 Figure 3 re-cuts** (pooled, so n is effectively doubled) point the
hypothesised direction but still do not reach significance: format-heavy +2.50
[-0.83, +5.67] vs knowledge-heavy -0.33 [-3.50, +2.67], contrast +2.83 [-0.67, +6.33].
Downstream +2.33 vs upstream -0.17, contrast +2.50 [-0.67, +5.67].

**Power: n=300 is underpowered for effects this size.** SE of the EM drop is 2.21 pp,
so ~720 questions would be needed for 80% power to detect a true 4 pp effect (~960 for
90%). HotpotQA dev has 7405, and cost scales to ~4 GPU-h per tier at n=750. Changing n
is a design change (§2/§5 lock it at 300) so it is a human decision, NOT to be done
unilaterally.

**One bug-flag fired and was a false positive:** `qa_4bit` parse-failure rate is exactly
0.0%. Verified genuine — 300 statuses populated, all `ok`, 280 distinct raw outputs;
baseline had exactly 1 failure. The heuristic just distrusts round numbers.

**Next.** Waiting at Gate 2. Per §2 priority order the next step is build step 9, model 2
(`Llama-3.2-3B-Instruct`), 4-bit tier only — but see the power question above, which the
human may want to resolve first. Llama-3.2 is a gated HF repo: needs an accepted license
and an `HF_TOKEN` Kaggle Secret. Do not start step 9 unprompted.

---

## 2026-07-29 (earlier) — pushed to GitHub, 4-bit tier validated, sweep handed to Kaggle

**Done.** Gate 1 approved by human. Repo pushed to
https://github.com/Retixx/Maxim-Mohareb-Michael-Zhang-Fun-Time (rebased onto the
existing initial commit, not force-pushed). Kaggle notebook points at it and now runs
all five run IDs from one cell.

Fixed a real bug in `weight_footprint_mb`: a bitsandbytes `Params4bit` tensor is uint8
whose `.numel()` is the PACKED byte count, so the old `params * bytes_per_param`
shortcut applied the 4-bit compression twice and understated the footprint by 2.6x
(423.7 MB reported vs 1070.2 MB true). That number is what the paper reports (SPEC §7).
Now summed as `numel * element_size`, exact-matched against transformers'
`get_memory_footprint()` at both fp16 and 4-bit. Added `param_census()`, which
surfaces that bnb leaves embeddings/biases/norms at fp16 — a "4-bit" Qwen2.5-1.5B is
1.310B params at 4-bit plus 234M at fp16.

**All four 4-bit configs validated end-to-end at n=5, seed 1234.** Mixed precision
wires correctly; the quantized stage loads at 4-bit and the other three at fp16.
`coresident_footprint_mb = 9903` for all four, vs 11778 for baseline — the controlled
constant SPEC §7 requires, proven: every configuration saves the same 1875 MB.

**Next.** Build step 8 runs on Kaggle, not locally. A human must execute
`notebooks/kaggle_run.ipynb` (Internet ON, GPU T4 x2), run the 10-question cell first,
then the five sweeps, and return `results.zip`. Then Gate 2 analysis. Do not start a
300-question run locally — see the VRAM note below.

**Known issues / open questions.**
- Do NOT split one run across local and Kaggle. Resume keys on
  (question_id, stage, call_index) and is blind to which machine produced a record, so
  a run started locally and finished on Kaggle would interleave two execution stacks
  with no key collision to flag it. One venue per run.
- Local VRAM readings remain untrustworthy (Windows WDDM spills CUDA allocations into
  system RAM rather than raising OOM). FP16 stages report ~5900 MB peaks on a 4096 MB
  card even at batch 1. Consequences unchanged: the OOM autotune path has still never
  fired, and SPEC §13's "under 6 GB at FP16 batch 16" cannot be checked here. The T4
  gives the first trustworthy numbers.
- n=5 parse rates are a plumbing check only (~28 calls/run) and must not be read as a
  role ranking: planner_4bit 100% on its quantized stage, stepdef_4bit 100%,
  extractor_4bit 83.3%, qa_4bit 100%.

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
