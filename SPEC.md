# Project Spec: Role-Aware Precision Allocation in Multi-Agent RAG

You are building the experimental harness for a NeurIPS 2026 workshop paper. Deadline is **Aug 29, 2026**. The scientific design below is **locked** — do not redesign it, propose alternatives, or expand scope. Build exactly this.

---

## 0. Operating rules — read before anything else

**Follow these for the entire project, every session.**

1. **Work through §11 in order.** Do not skip ahead, do not build later stages "while you're in there."
2. **STOP at the two gates in §11a and wait for explicit human approval.** Do not continue on your own initiative. Do not start the next step while waiting. Do not treat silence as approval. Gate 1 comes early — after build step 2 — and requires you to *run* the smoke test and report on it rather than continuing to build.
3. **Before each gate, checkpoint your work:**
   - `git commit` with a message describing what was completed
   - Append 3–5 lines to `PROGRESS.md`: what is done, what is next, any known issues or open questions

   A different agent or a fresh session may pick this up. `PROGRESS.md` and the git log are the only handoff — nothing carries over in memory.
4. **The design is locked.** If you believe something in this spec is wrong or infeasible, say so and wait. Do not silently substitute a different approach.
5. **Never add retry logic to parse failures.** This will feel like an obvious improvement. It destroys the experiment's primary measurement. See §5.
6. **Ask before deviating.** Scope creep is the main failure mode for this project, not code quality.

Save this document as `SPEC.md` in the repo root so later sessions can re-read it.

---

## 1. The research question

**At SLM scale, does quantizing one agent in a multi-agent RAG pipeline hurt the same roles that shrinking one agent hurts?**

Prior work (MA-RAG, arXiv:2505.20096) built a four-agent RAG pipeline and ablated **model size** per role, finding: QA agent hurts most, Planner and Extractor clearly, Step Definer barely at all. They never varied **numerical precision**. We do.

**Hypothesis:** the cheapest agent to shrink is the most expensive agent to quantize. Rationale — quantization damages output format and calibration while leaving knowledge intact; parameter reduction does the reverse. The Step Definer's job is emitting structured specs (format-heavy, knowledge-light), so it should be the most quantization-sensitive role despite being the least size-sensitive.

Both outcomes are publishable. Do not optimize toward confirming the hypothesis.

---

## 2. The experiment (locked)

One base model, four agent roles, same weights, different prompts. Exactly one agent is quantized per run; the rest stay FP16.

| Run ID | Planner | Step Definer | Extractor | QA |
|---|---|---|---|---|
| `baseline` | FP16 | FP16 | FP16 | FP16 |
| `planner_4bit` | **4-bit** | FP16 | FP16 | FP16 |
| `stepdef_4bit` | FP16 | **4-bit** | FP16 | FP16 |
| `extractor_4bit` | FP16 | FP16 | **4-bit** | FP16 |
| `qa_4bit` | FP16 | FP16 | FP16 | **4-bit** |

This 5-run block is the **core result**. It is then repeated on a **second base model** (§3) to test whether the ordering is architecture-dependent. Further bit-widths (8-bit, then 3-bit) are added only after both models are done.

Priority order — do not reorder:

1. Model 1, 4-bit tier (5 runs) — the core result
2. Model 2, 4-bit tier (5 runs) — the generalization claim
3. Model 1, 8-bit tier (4 runs)
4. 3-bit, only if time remains

Generalization across architectures is worth more to this paper than dose-response on a single model.

---

## 3. Stack

- **Base model 1:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Base model 2:** `meta-llama/Llama-3.2-3B-Instruct` — different family and tokenizer, same capability tier. Kaggle only (6.4 GB at FP16, will not fit a 4 GB local card). Add at build stage 9, not before.
- **Quantization:** `bitsandbytes` via transformers — `load_in_4bit=True` (NF4) and `load_in_8bit=True`. Chosen for reliability, not speed. 3-bit requires GPTQ (`gptqmodel`), treat as a separate optional path.
- **Inference:** HuggingFace `transformers` with **batched generation**. Batching is mandatory, not an optimization — see §6.
- **Dataset:** HotpotQA, **distractor setting**, dev split. Each question ships with 10 paragraphs (2 gold, 8 distractors).
- **No retriever.** Use the provided paragraphs directly. Retrieval quality is not under study and must not become a confound.
- **No training, no fine-tuning, ever.** Inference only.

Record the exact quantization config (method, group size, compute dtype) in the results metadata. It matters for reproducibility and will be discussed in the paper.

**Hardware context:** local machine is an RTX 3050 Laptop with **4 GB VRAM**, 16 GB system RAM. Full sweeps run on Kaggle (T4, 16 GB, ~30 GPU-hours/week). Local is for development and the Gate 1 smoke test only.

---

## 4. Pipeline

Four agents, reimplemented from the MA-RAG paper description. Assume no reference code exists; this is a **MA-RAG-style** pipeline, not a reproduction.

1. **Planner** — reads the question, emits a plan: an ordered list of sub-questions. Output must be parseable structured text (JSON list preferred).
2. **Step Definer** — for each sub-question, emits a structured retrieval/extraction spec. Format-heavy role.
3. **Extractor** — given a sub-question and the paragraph set, returns supporting spans verbatim. Must not paraphrase or invent. Called once per sub-question.
4. **QA** — synthesizes the final short answer from accumulated evidence.

Keep the pipeline **non-iterative** (single forward pass, no replanning loops). Document this simplification. Iterative loops make stage-batching intractable and are not needed to answer the question.

Prompts live in one module, versioned, one per role. Do not tune prompts per precision level — that would confound the experiment.

---

## 5. Metrics

**Primary — accuracy:** Exact Match and F1 against HotpotQA gold answers, using the official normalization (lowercase, strip articles and punctuation).

**Secondary — parse-failure rate:** for every agent call, whether the output parsed into the expected structure. Track per role, per precision. This is the mechanism evidence and is **as important as accuracy** — do not treat it as optional telemetry.

Failure taxonomy, logged per call:

- `ok`
- `malformed_json`
- `schema_mismatch` (valid JSON, wrong fields)
- `empty_output`
- `truncated` (hit max_new_tokens)
- `refusal_or_offtopic`

**Never silently drop or retry a failed parse.** Log it, count it, and score that question as-is. Silent retries destroy the entire secondary measurement.

**Statistics:** n = **300 questions** per run. Report bootstrap 95% confidence intervals (10k resamples) on every reported number. A difference whose CIs overlap is not a result.

> **AMENDED AT GATE 2 (human decision, 2026-07-29): n = 750, not 300.**
> `config/experiment.yaml` is authoritative. The n=300 tier returned a null with
> every CI spanning zero; measured SE of the EM drop was 2.21 pp, so ~720
> questions are needed for 80% power against a true 4 pp effect. MA-RAG evaluated
> on 5600 HotpotQA dev questions, 18.7x the original n, which largely explains why
> they resolved a role ranking and we did not. Cost is ~4.0 GPU-h for the five-run
> tier. Note this makes §13's "inside 4 GPU-hours" criterion marginal by
> construction — judge it against ~4 h at n=750, not the original figure.
>
> **The n=300 results must NOT be pooled with the n=750 results.** Verified:
> `random.sample` with a fixed seed is *nested* for increasing k, so the 300-question
> sample is a strict SUBSET of the 750-question sample (overlap 300, nested = True).
> Pooling would therefore double-count all 300. The n=750 run supersedes the n=300
> run outright — report n=750 alone.
>
> The nesting does buy one free check: after the n=750 runs land, the shared 300
> questions can be compared against the n=300 outputs. Any disagreement is
> attributable to batch composition (a question sits in a different batch, with
> different padding neighbours, at n=750 than at n=300), which is also why the
> n=300 records are NOT reused to seed the n=750 files even though the question ids
> match. Greedy decoding is deterministic for a fixed batch, not across
> re-batchings.

---

## 5a. Expected values — what "healthy" looks like

**Read this before reporting any result as a problem.** Small models on multi-hop QA score far lower than people expect, and a correct pipeline will look broken if you calibrate to the wrong numbers.

| Metric | Healthy | Concerning | Broken — stop and fix |
|---|---|---|---|
| **Parse success rate** (per agent, FP16) | > 90% | 70–85% | < 70% |
| **Answer EM** (FP16 baseline) | 30–45% | 20–30% | < 15% |

Context for the EM range: fine-tuned SOTA on HotpotQA distractor is ~68–72 EM; GPT-4-class few-shot is ~50–60. A 1.5B–3B model in a four-agent pipeline landing at **35 EM is a healthy result, not a defect.** Do not attempt to "fix" the pipeline toward 70%.

Get baseline parse failure **under 10%** via prompt work before scaling to n=300. Absolute failure level matters less than dynamic range — the experiment measures deltas — but a high baseline compresses headroom and adds noise.

---

## 5b. PRE-REGISTERED MECHANISM PREDICTIONS (added 2026-07-29, after Gate 2)

**Why this section exists.** §1 asserts a mechanism: quantization damages output
format and calibration while leaving knowledge intact. The 4-bit tier on model 1
(n=300) **refuted the format half of that claim** on three independent instruments,
each with the point estimate in the *wrong* direction:

| instrument | baseline | 4-bit | paired delta |
|---|---|---|---|
| parse success (Extractor) | 95.7% | 96.5% | −0.79 pp [−2.86, +1.28] |
| strict format, no parser tolerance | 94.1% | 94.8% | +0.16 pp [−1.11, +1.43] |
| verbatim span fidelity | 79.7% | 81.3% | −2.48 pp [−5.48, +0.53] |

This is not a power problem and more questions will not change it — baseline failure
rates of 0.3–4.3% leave almost no dynamic range, and all three estimates favour 4-bit.
**Do not keep re-measuring format hoping for a different answer.**

What the data *does* show, with a large unambiguous effect: **73.8% of Extractor calls
selected different evidence under 4-bit** while format and fidelity held and accuracy
moved ≤4 pp. The mechanism the evidence supports is therefore *selection perturbation
without quality degradation* — quantization changes **which** content is chosen, not how
well-formed it is.

**That reframing was found by testing three metrics against the same dataset. Reporting
it as though it had been predicted would be HARKing** — the same error as asserting the
original mechanism. Hence the predictions below, committed before the confirmatory data
is analysed.

### Predictions

Confirmatory test is the planned **n=5000 rerun** (different hardware, all runs from
scratch). Model 2 at n=750 was already executing when this was written, so it is
**analyst-blind but not pre-data** — treat it as supporting evidence, not confirmation.

1. **Format is NOT damaged.** For every role, at 4-bit, the paired delta in parse
   success, strict-format compliance, and verbatim fidelity each have a 95% CI
   containing zero, or are negative (favouring the quantized run).
2. **Selection churn is HIGH.** >50% of Extractor calls select a different span set at
   4-bit than at FP16. >25% of final answers change text.
3. **Role type predicts sensitivity.** Pooled EM drop for format-heavy roles
   (Step Definer, Extractor) exceeds knowledge-heavy roles (Planner, QA), with the
   contrast's 95% CI excluding zero.
4. **Calibration is the open question.** §1's calibration claim has never been measured.
   Prediction is stated as directional only: if any part of §1's mechanism survives, it
   is calibration — AUROC of answer confidence against correctness degrades under 4-bit
   while accuracy does not.

Predictions 1 and 2 are well powered already. Prediction 3 is the one n=5000 is for.
Prediction 4 requires `generation.log_confidence: true` (see §7).

**A failed prediction here is a result, not a problem to engineer around.** If 3 fails
at n=5000, the honest finding is that role-aware precision allocation does not matter at
4-bit and uniform allocation is sufficient — which is publishable and useful.

### Contingencies

- **Floor effect** — if 4-bit collapses *all four* roles to near-zero, there is no ranking to measure. In order: (1) switch to the 8-bit tier where degradation is gentler, (2) move model 1 to `Qwen/Qwen2.5-3B-Instruct` for more parametric redundancy, (3) report uniform collapse as the finding. Do not silently tune around it.
- **Baseline EM below 15%** — the pipeline is broken, not the model. Inspect raw outputs before changing anything else.
- **Local smoke test OOM** — Qwen2.5-1.5B at FP16 is ~3.1 GB against a 4 GB card. If it OOMs, run the smoke test at 4-bit instead. You are checking plumbing, not measuring anything.

---

## 6. Execution model — stage-major with checkpointing

This is the part that makes it fit in Kaggle's quota. Read carefully.

**Do not run question-by-question through the whole pipeline.** Run **stage-by-stage across all questions**:

```
for stage in [planner, step_definer, extractor, qa]:
    load model at the precision this run assigns to this stage
    process ALL 300 questions through this stage in batches
    write outputs to disk
    unload model, free VRAM
```

Two reasons this is required:

1. Only one model is ever resident, so a 4 GB GPU works and a 16 GB T4 is comfortable.
2. It enables large batch sizes, which is the difference between ~43 GPU-hours and ~12 for the full sweep.

**Batch size:** start at 16, auto-reduce on OOM. Left-pad for decoder-only batched generation and confirm the tokenizer's pad token is set.

**Checkpointing is mandatory.** Kaggle GPU sessions cap around 12 hours and can terminate without warning.

- Append every agent call to a JSONL file as it completes.
- On startup, read existing output and skip any `(run_id, question_id, stage, call_index)` already present.
- Resuming must be the default behavior, not a flag.
- Flush to disk at least every 50 records.

---

## 7. Logging schema

One JSONL record per agent call:

```json
{
  "run_id": "stepdef_4bit",
  "question_id": "5a8b57f25542995d1e6f1371",
  "stage": "step_definer",
  "precision": "4bit",
  "call_index": 0,
  "prompt_tokens": 812,
  "output_tokens": 96,
  "latency_s": 1.84,
  "raw_output": "...",
  "parse_status": "ok",
  "parsed": {...},
  "timestamp": "..."
}
```

Plus one record per question with the final answer, EM, and F1.

Plus a run-level metadata blob: model ID, quantization config (method, group size, compute dtype), batch size, git commit, GPU name, total wall time, and **memory**:

- `peak_vram_mb` per stage, from `torch.cuda.max_memory_allocated()`
- `weight_footprint_mb` per stage, computed as params × bytes-per-param
- `coresident_footprint_mb` — total if all four agents were loaded simultaneously. This is the deployment-relevant number and the one the paper reports.

Note: memory is **identical across runs 1–4 by construction** — same model, same bit-width, one agent quantized. It is a controlled constant, not an outcome. The paper's claim depends on this: all four configurations save the same bytes and differ only in which agent pays the accuracy cost. Log it to prove that, not to compare it.

Keep `raw_output` always. You will need it to build the failure taxonomy and to show examples in the paper.

---

## 8. Repo structure

```
marag-precision/
  SPEC.md                  # this document
  PROGRESS.md              # updated before every gate
  config/
    experiment.yaml        # run definitions, n, seeds, batch size
  src/
    models.py              # load base model at a given precision
    prompts.py             # one prompt template per role, versioned
    agents.py              # role call wrappers
    pipeline.py            # stage-major orchestration
    parsing.py             # parsers + failure taxonomy
    metrics.py             # EM/F1, bootstrap CIs
    runner.py              # sweep driver, checkpoint/resume
  notebooks/
    kaggle_run.ipynb       # thin wrapper, see §9
  results/                 # JSONL outputs (gitignored)
  analyze.py               # produces the figures in §10
  README.md
```

Everything driven by `config/experiment.yaml`. No interactive prompts, no hardcoded paths. It must run as `python -m src.runner --config config/experiment.yaml --run stepdef_4bit`.

---

## 9. Kaggle notebook

Keep it to about five cells. The notebook is a launcher, not where logic lives.

1. `!pip install -q -U transformers accelerate bitsandbytes datasets`
2. `!git clone <repo>` (or pull latest)
3. Print `nvidia-smi`, confirm T4 and free VRAM
4. `!python -m src.runner --config config/experiment.yaml --run <run_id>`
5. Zip `results/` into `/kaggle/working/` for download

Notes to surface in the notebook markdown:

- **Internet must be enabled** in notebook settings (Settings → Internet On), otherwise HuggingFace downloads fail.
- Accelerator set to **GPU T4 x2** (use one; the second is idle but the quota is the same).
- `/kaggle/working` holds ~20 GB. Model cache goes to `/kaggle/tmp` or `~/.cache` to avoid filling it.
- Save results **incrementally**, not just at the end.

---

## 10. Analysis output

`analyze.py` produces three figures and one table.

**Figure 1 — the money plot.** Accuracy drop from baseline, per role, grouped by precision. Four roles on the x-axis, bars for 8-bit / 4-bit / 3-bit, error bars from bootstrap CIs. Annotate with MA-RAG's published size-sensitivity ordering for visual comparison.

**Figure 2 — the mechanism.** Parse-failure rate per role, per precision, same layout.

**Figure 3 — role type vs pipeline position.** Same accuracy-drop data, re-cut two ways: by **role type** (format-heavy: Step Definer, Extractor | knowledge-heavy: Planner, QA) and by **pipeline position** (upstream: Planner, Step Definer | downstream: Extractor, QA). This separates two competing explanations — that quantization damage tracks the *kind of work* a role does, versus that it tracks *how far downstream the damage propagates*. No new runs required; this is a re-cut of existing data.

**Table 1.** Per-run EM, F1, parse-failure rate, each with 95% CI, plus mean latency, peak VRAM, and co-resident footprint.

---

## 11. Build order

1. Pipeline at FP16 only, hardcoded, 10 questions. **Print every agent's raw output to the terminal for manual inspection.** This is the smoke test — the human reads these before anything else proceeds.
2. Parsers plus failure taxonomy. Verify failures are detected, counted, and never retried.
   → **STOP. Run the smoke test and report. See §11a Gate 1.**
3. Stage-major refactor with disk-cached intermediate state.
4. Checkpoint and resume. Test by killing the process mid-run and restarting.
5. Precision switching, baseline plus the four 4-bit runs.
6. Batching and batch-size autotuning.
7. Kaggle notebook, verified on 10 questions end to end before any full run.
8. Scale to n=300 on model 1, full 4-bit tier (baseline + 4 runs).
   → **STOP. Report results. See §11a Gate 2.**
9. **Add model 2** (`Llama-3.2-3B-Instruct`), 4-bit tier only. Model must be a config value — do not build a multi-model abstraction layer.
10. `analyze.py`.
11. 8-bit tier on model 1, if time allows.
12. 3-bit via GPTQ, only if everything above is done.

---

## 11a. MANDATORY STOP GATES

**You must halt and wait for explicit human approval at two points.** Do not continue past a gate on your own initiative, do not begin the next stage "while waiting," and do not treat silence as approval.

Before each gate: `git commit` and update `PROGRESS.md` (see §0).

### GATE 1 — after build step 2

Stop building. **Run the smoke test**: 10 questions, FP16 only, all four agents, locally.

Then produce a report of at most one screen:

- Did the pipeline complete end-to-end on all 10 questions? If not, where did it break?
- For each of the four agents: **2 verbatim raw outputs**, unedited and untruncated
- Parse success/failure counts per agent, broken down by failure type
- The 10 final answers beside gold answers, with EM
- Measured throughput: seconds per agent call, and the **extrapolated GPU-hours for one 300-question run**
- Your assessment of whether the pipeline is sound enough to scale, naming specific defects if not
- An explicit recommendation: **PROCEED** or **FIX FIRST**

Judge against the thresholds in **§5a**, not against intuition. EM in the 30–45% range is healthy. Recommend FIX FIRST if any agent's parse success is below 90%, or if baseline EM is under 15%.

**Do not compare anything to MA-RAG at this gate.** The smoke test is FP16-only and produces no role-sensitivity data; any such comparison would be meaningless.

Then **WAIT** for the human.

### GATE 2 — after build step 8 (first complete 4-bit tier on model 1)

Stop. Produce:

- Table: accuracy drop from baseline per role, with bootstrap 95% CIs
- Table: parse-failure rate per role, baseline vs 4-bit
- **The resulting ranking of the four roles by quantization sensitivity**
- Side by side with MA-RAG's published **size**-sensitivity ranking: *QA hurts most → Planner ≈ Extractor → Step Definer barely at all*
- State plainly whether the two orderings **match, differ, or are indistinguishable** given the confidence intervals
- Flag anything that looks like a bug rather than a finding — a role showing exactly zero change, CIs spanning the full range, or parse-failure rates of 0% or 100%
- Recommendation on whether to proceed to model 2

Then **WAIT** for the human.

---

## 12. Non-goals — do not build these

- Any retriever or vector store
- Training, fine-tuning, distillation, LoRA
- Iterative replanning or agent-to-agent negotiation loops
- A web UI, dashboard, or CLI beyond the single runner entrypoint
- A multi-model abstraction layer. Two models are supported (§3), but the model is a **config string**, not a plugin architecture.
- Prompt optimization or per-precision prompt tuning
- Multi-dataset support (HotpotQA only; 2WikiMQA is a later stretch, not now)
- Automatic retry on parse failure — this actively breaks the experiment
- **Constrained or grammar-based decoding — forbidden.** No `outlines`, no
  `guidance`, no `lm-format-enforcer`, no JSON-schema decoding, no logit masking,
  no `transformers` constrained-beam or prefix-allowed-tokens generation, no
  `response_format`-style structured-output modes.

  This is the single most tempting engineering fix in the whole project and it
  must never be applied. Constrained decoding drives the parse-failure rate to
  zero *by construction*. The parse-failure rate is the mechanism evidence for
  the paper's entire argument — that quantization damages output format while
  leaving knowledge intact. Forcing valid output does not fix the model; it
  deletes the measurement and makes every run look identical on the secondary
  metric. A low parse-failure baseline achieved this way is worthless.

  Generation stays plain greedy sampling (`do_sample=False`) with no logit
  processors. If baseline parse failure cannot be brought under 10% by prompt
  wording alone, the correct response is to report the higher baseline and note
  the compressed headroom — not to constrain the decoder. See §5a: dynamic range
  matters more than absolute level.

---

## 13. Acceptance criteria

- `python -m src.runner --run baseline` completes 300 questions and writes valid JSONL
- Killing the process mid-run and rerunning resumes without duplicating work
- Parse failures appear in output with correct taxonomy labels, zero retries
- Peak VRAM stays under 6 GB at FP16 with batch size 16
- All five 4-bit-tier runs complete inside 4 GPU-hours total
- `analyze.py` emits the figures and Table 1 from results alone, with CIs

---

**Begin with build step 1. Stop at Gate 1.**
