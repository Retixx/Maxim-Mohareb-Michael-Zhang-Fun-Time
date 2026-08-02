# Project Spec: Role-Aware Capacity Allocation in Multi-Agent RAG

You are building the experimental harness for a NeurIPS 2026 workshop paper. Deadline is **Aug 29, 2026**. The scientific design below is **locked** — do not redesign it, propose alternatives, or expand scope. Build exactly this.

> **v2 of this spec (2026-08-01).** v1 asked one question: *which role is most sensitive to quantization?* That question is answered for model 1 (§14). v2 adds the second axis — **size** — so the trade-off can be measured inside one pipeline instead of borrowed from another paper, and adds the deployment comparison that makes the finding actionable. §§1, 2, 4a, 5d, 5e, 10, 11, 12, 13, 14 are new or substantially rewritten. §§5, 5a, 5b, 5c are preserved as the historical record: they contain pre-registered commitments and must not be silently edited — §5b's predictions 5–7 were appended, not substituted.

---

## 0. Operating rules — read before anything else

**Follow these for the entire project, every session.**

1. **Work through §11 in order.** Do not skip ahead, do not build later stages "while you're in there."
2. **STOP at the gates in §11a and wait for explicit human approval.** Do not continue on your own initiative. Do not start the next step while waiting. Do not treat silence as approval.
3. **Before each gate, checkpoint your work:**
   - `git commit` with a message describing what was completed
   - Append 3–5 lines to `PROGRESS.md`: what is done, what is next, any known issues or open questions

   A different agent or a fresh session may pick this up. `PROGRESS.md` and the git log are the only handoff — nothing carries over in memory.
4. **The design is locked.** If you believe something in this spec is wrong or infeasible, say so and wait. Do not silently substitute a different approach.
5. **Never add retry logic to parse failures.** This will feel like an obvious improvement. It destroys the experiment's primary measurement. See §5.
6. **Never compare across arms at unmatched memory.** Every accuracy comparison in this paper is meaningless without the footprint beside it. See §5d.
7. **Ask before deviating.** Scope creep is the main failure mode for this project, not code quality.

This document lives at `SPEC.md` in the repo root so later sessions can re-read it.

---

## 1. The research question

**At SLM scale, is a multi-agent RAG pipeline better served by spending its memory budget unevenly across roles than evenly?**

That decomposes into three questions, answered in order:

- **Q1 (done, §14).** Which role is most sensitive to **quantization**?
- **Q2 (new).** Which role is most sensitive to **parameter reduction**, measured *in this pipeline* rather than borrowed from another paper?
- **Q3 (new).** Given Q1 and Q2, does a **role-aware** allocation beat a **uniform** one at the same memory footprint — and does either beat plain single-call RAG?

Prior work (MA-RAG, arXiv:2505.20096) built this four-agent pipeline and ablated **model size** per role, reporting that the planner and extractor are critical for multi-hop reasoning and that the QA agent is the one that most needs a high-capacity model. They never varied **numerical precision**.

**Why Q2 has to be run in-house.** v1 of this spec compared our quantization ranking directly against MA-RAG's published size ranking. That comparison is not admissible and must not appear in the paper as evidence. It differs in base model (LLaMA3-8B/70B and GPT-4o-mini vs our 1.5B), in pipeline (they retrieve; we use gold+distractor paragraphs directly, and we are non-iterative), in dataset scale (they evaluate on ~5600 HotpotQA dev questions; we use 750), and in prompts. Any of those alone would sink it. MA-RAG's ordering is **related work to cite, not a control arm.** The size ablation below is the control arm.

**Hypothesis (unchanged from v1, still under test):** the cheapest agent to shrink is the most expensive agent to quantize. Rationale — quantization damages output format and calibration while leaving knowledge intact; parameter reduction does the reverse. If true, the two rankings are close to inverted and role-aware allocation has real headroom. If the rankings instead coincide, the honest finding is that one capacity axis is a proxy for the other and uniform allocation is sufficient.

Both outcomes are publishable. Do not optimize toward confirming the hypothesis. §5b records what was already refuted; §14 records what is already known.

---

## 2. The experiment (locked)

Four agent roles. One agent is perturbed per run; the rest stay at the reference configuration (Qwen2.5-1.5B-Instruct at FP16). Three phases.

**Reference configuration and the `baseline` run are shared by every phase.** There is exactly one `baseline` per (model, n, seed). Do not re-run it per phase.

### Phase Q — quantization ablation (COMPLETE on model 1, §14)

| Run ID | Planner | Step Definer | Extractor | QA |
|---|---|---|---|---|
| `baseline` | 1.5B fp16 | 1.5B fp16 | 1.5B fp16 | 1.5B fp16 |
| `planner_4bit` | **1.5B 4-bit** | 1.5B fp16 | 1.5B fp16 | 1.5B fp16 |
| `stepdef_4bit` | 1.5B fp16 | **1.5B 4-bit** | 1.5B fp16 | 1.5B fp16 |
| `extractor_4bit` | 1.5B fp16 | 1.5B fp16 | **1.5B 4-bit** | 1.5B fp16 |
| `qa_4bit` | 1.5B fp16 | 1.5B fp16 | 1.5B fp16 | **1.5B 4-bit** |

### Phase S — size ablation (NEW). Four runs; reuses `baseline`.

| Run ID | Planner | Step Definer | Extractor | QA |
|---|---|---|---|---|
| `planner_small` | **0.5B fp16** | 1.5B fp16 | 1.5B fp16 | 1.5B fp16 |
| `stepdef_small` | 1.5B fp16 | **0.5B fp16** | 1.5B fp16 | 1.5B fp16 |
| `extractor_small` | 1.5B fp16 | 1.5B fp16 | **0.5B fp16** | 1.5B fp16 |
| `qa_small` | 1.5B fp16 | 1.5B fp16 | 1.5B fp16 | **0.5B fp16** |

The perturbed role's footprint is what makes Q and S comparable:

| treatment | weight footprint | measured? |
|---|---|---|
| 1.5B fp16 (reference) | 2944.4 MB | yes, §14 |
| **1.5B 4-bit** (Phase Q) | **1070.2 MB** | yes, §14 |
| **0.5B fp16** (Phase S) | **942.2 MB** | computed; must be confirmed at run time |
| 0.5B 4-bit | ~430 MB | for Phase D only |

**The two treatments are matched to within 13.6%, and the quantized arm is the one holding more bytes.** This is stated, not hidden: if the size arm wins, it wins despite a smaller budget and the conclusion is safe; if the quantized arm wins by less than the budget gap could explain, the honest verdict is "indistinguishable at matched budget." `analyze.py` must print the gap next to every Q-vs-S comparison. Do not silently treat 1070 and 942 as equal.

### Phase H — head-to-head (NEW). **No new runs.** Pure re-analysis of Q and S.

For each role, both arms score the same questions against the same `baseline`, so the comparison is paired twice over:

    quantization cost(role) = EM(baseline) − EM(role_4bit)
    size cost(role)         = EM(baseline) − EM(role_small)
    axis contrast(role)     = size cost − quantization cost

A positive contrast means that role would rather be quantized than shrunk. **This is the paper's central table.** It is computed per question and bootstrapped paired, so between-question variance cancels in both differences.

### Phase D — deployment comparison (NEW). Runs on a **disjoint confirmation set** (§5e).

Phases Q, S and H are *selection*: they measure per-role sensitivity and are used to choose an allocation. Scoring that allocation on the same questions it was selected on would be selecting on noise. Phase D therefore runs on questions none of the earlier phases touched.

| Arm | Run IDs | What it is |
|---|---|---|
| **Role-aware MA-RAG** | `ma_optimized_hi`, `ma_optimized_lo` | Per-role treatment chosen by the §5e rule from Phase H |
| **Uniform MA-RAG** | `ma_uniform_4bit`, `ma_uniform_small` | All four roles get the same treatment |
| **Single-call RAG** | `single_fp16`, `single_4bit`, `single_small` | No decomposition — one call, §4a |
| **Reference** | `baseline` (re-run on the confirmation set) | Accuracy ceiling, highest footprint |

The deliverable is **accuracy against deduplicated memory footprint** (§5d, §10 Figure 4). The claim "role-aware allocation is worth it" means exactly one thing: **the role-aware points sit above the frontier traced by the uniform points.** Not "the optimized run scores higher" — higher at a higher budget is not a result.

### Priority order — do not reorder

1. **Phase Q re-run at the new n on model 1** (5 runs) — the pilot at n=750 resolved one role; the tier has to be regenerated at the chosen n so Q and S share a question set and a batch size.
2. **Phase S on model 1** (4 runs) — makes Q2 answerable and Phase H free. Highest value per GPU-hour in the project.
3. **Phase H analysis** (0 runs) — the central table, and the primary test in §5f.
4. **Phase D on model 1** (8 runs, confirmation set) — the actionable claim.
5. **Phase Q + S on a second family** (9 runs) — generalization, and the answer to model currency (§3).
6. 8-bit tier on model 1 (4 runs), only if time remains.
7. 3-bit via GPTQ, only if everything above is done.

Steps 1–3 are the paper. Everything from 4 on is an extension; a 5-page workshop
submission does not need it.

### Compute budget — sizing n to the GPU you actually have

Measured: Qwen2.5-1.5B, n=750, five runs, **4.24 GPU-h on a Tesla T4**. Decode here is
memory-bandwidth-bound, where an A100 is roughly 5x a T4, plus better utilization at
larger batch — call it **6x, conservatively**. Phase S is slightly cheaper than Phase Q
because the 0.5B stage replaces a 1.5B one. So on one A100:

| n | Phase Q (5 runs) | Phase S (4 runs) | Q+S total | SE of the EM drop |
|---|---|---|---|---|
| 750 | 0.7 h | 0.6 h | **1.3 h** | 2.21 pp (measured) |
| 2000 | 1.9 h | 1.6 h | **3.5 h** | ~1.35 pp |
| **3000** | **2.8 h** | **2.4 h** | **~5.2 h** | **~1.10 pp** |
| 5000 | 4.7 h | 4.0 h | **8.7 h** | ~0.86 pp |

**Rule: pick the largest n whose Q+S total is under half your wall-clock budget.** The
other half absorbs model and dataset download, a smoke test, at least one OOM-autotune
retry, and the run you will have to redo. n=5000 inside a 10-hour window has no margin
and is the wrong trade — a completed n=3000 tier beats a truncated n=5000 one, and
resume does not help if the instance disappears.

At n=3000, SE ≈ 1.10 pp gives ~80% power against a 3 pp effect on a single test. The
observed Phase Q effects were +3.20 / +1.73 / +0.53 / −1.73 pp, so **expect the primary
contrast (§5f) to resolve and expect most per-role numbers not to.** That is the honest
prior, and it is why §5f makes the contrast primary rather than the ranking.

---

## 3. Stack

- **Base model 1:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5437B params, 2944.4 MB fp16 — measured)
- **Small model 1:** `Qwen/Qwen2.5-0.5B-Instruct` (0.49B params, 0.36B non-embedding, 24 layers). **Same family, same tokenizer, same instruction-tuning recipe as the base model.** This matters: it is the closest thing available to a pure parameter-count intervention. It is still not clean — 0.5B and 1.5B differ in depth, width and data mix, not just count — and §12's limitation note covers that.
- **Base model 2 (only if compute remains):** `meta-llama/Llama-3.2-3B-Instruct`, with `meta-llama/Llama-3.2-1B-Instruct` as its small model. Different family and tokenizer, same capability tiers. Gated repo: needs an accepted licence and an `HF_TOKEN`.

> **Why not Llama-3.1-8B.** Two reasons, the first fatal. (1) **The 3.1 family has no
> small sibling** — it is 8B / 70B / 405B. Phase S requires a same-family, same-tokenizer,
> same-instruct-recipe smaller model to swap into one role; pairing 3.1-8B with a
> Llama-3.2 small model crosses a generation and a training recipe, which is exactly the
> confound §12 says the in-house size ablation exists to remove. (2) 8B is 16 GB at FP16,
> which is not an on-device budget and not an SLM by the venue's own framing, and it costs
> ~5x model 1 per run. If a second family is added it must be one with a size ladder:
> Llama-3.2 (1B/3B) qualifies, Llama-3.1 does not.
>
> **Model currency is a known reviewer risk, not an eligibility one.** Qwen2.5 is two
> generations behind Qwen3.5. No target venue restricts base models. Mitigation, in
> priority order: (a) state the limitation, (b) add a current-generation second family if
> compute allows. Do NOT swap model 1 — prompts are frozen at v5 and validated on
> Qwen2.5, and §12 forbids retuning them per model, so a swap risks a floor effect that
> cannot be fixed.
- **Quantization:** `bitsandbytes` via transformers — `load_in_4bit=True` (NF4, double quant, fp16 compute) and `load_in_8bit=True`. Chosen for reliability, not speed. 3-bit requires GPTQ (`gptqmodel`), treat as a separate optional path.
- **Inference:** HuggingFace `transformers` with **batched generation**. Batching is mandatory, not an optimization — see §6.
- **Dataset:** HotpotQA, **distractor setting**, dev split (7405 questions). Each question ships with 10 paragraphs (2 gold, 8 distractors) and gold `supporting_facts` labels.
- **No retriever.** Use the provided paragraphs directly. Retrieval quality is not under study and must not become a confound.
- **No training, no fine-tuning, ever.** Inference only.

Record the exact model ID *and* quantization config (method, group size, compute dtype) per stage in the results metadata. With Phase S, the model ID now varies **within** a run — metadata that records only a run-level `model_id` is wrong. See §7.

**Hardware context:** local machine is an RTX 3050 Laptop with **4 GB VRAM**, 16 GB system RAM, Windows. Full sweeps run on Kaggle (T4, 16 GB, ~30 GPU-hours/week). Local is for development and smoke tests only, and its VRAM readings are not trustworthy (§14).

---

## 4. Pipeline

Four agents, reimplemented from the MA-RAG paper description. Assume no reference code exists; this is a **MA-RAG-style** pipeline, not a reproduction.

1. **Planner** — reads the question, emits a plan: an ordered list of sub-questions (2–3, capped at 3). Output must be parseable structured text.
2. **Step Definer** — for each sub-question, emits a structured retrieval/extraction spec (`search_terms`, `target_entity`, `answer_type`). Format-heavy role. Called once per sub-question.
3. **Extractor** — given a sub-question and the paragraph set, returns supporting spans verbatim. Must not paraphrase or invent. Called once per sub-question.
4. **QA** — synthesizes the final short answer from accumulated evidence. Called once per question.

Keep the pipeline **non-iterative** (single forward pass, no replanning loops). Document this simplification. Iterative loops make stage-batching intractable and are not needed to answer the question.

### 4a. Single-call RAG baseline (NEW — required by Phase D)

A fifth role, `solo`, used **only** by the `single_*` runs. One call per question: the same 10 paragraphs and the question in, one short answer out. No decomposition, no evidence selection, no intermediate structure.

- Same output contract as QA (`{"answer": "..."}`), same parser, same failure taxonomy.
- Same prompt discipline: versioned, frozen, never tuned per precision or per size.
- `max_new_tokens` matches QA's (48). Its prompt is long (all 10 paragraphs), which is exactly the point — it is what a single-agent system actually has to read.
- Runs as a one-stage pipeline. It produces no planner/step_definer/extractor records, so **any analysis that assumes four stages must skip `single_*` runs explicitly** rather than silently averaging over missing data.

This arm is not a strawman and must not be built as one. It is a genuinely competitive baseline at SLM scale — a 1.5B model reading 10 paragraphs and answering directly may well beat the same model routed through four lossy hops. If it does, that is the paper's most useful finding and it must be reported as prominently as anything else.

Prompts live in one module, versioned, one per role. **Do not tune prompts per precision level or per model size** — that would confound both axes. The same template string is used at every precision and for both the 1.5B and the 0.5B model.

---

## 5. Metrics

**Primary — accuracy:** Exact Match and F1 against HotpotQA gold answers, using the official normalization (lowercase, strip articles and punctuation).

**Secondary — parse-failure rate:** for every agent call, whether the output parsed into the expected structure. Track per role, per treatment. This is the mechanism evidence and is **as important as accuracy** — do not treat it as optional telemetry.

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
> tier.
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

> **AMENDED FOR v2 (2026-08-01): n is now sized to available compute, not fixed.**
> The two amendments above are the historical record of the 300 -> 750 decision and
> stand. `config/experiment.yaml` remains authoritative for the current n; §2's
> compute-budget table is how it gets chosen. **The n=750 tier is now PILOT data** —
> superseded by whatever tier is generated next, and never pooled with it, for exactly
> the reason the 300/750 amendment gives. Note the nesting argument used there does not
> generalise: see §13b item 4.
>
> **Paired power, not one-sample power.**
> Phase H's axis contrast is a *difference of two differences*. Both differences
> are paired on the same questions against the same baseline, which cancels most
> of the between-question variance, but the contrast still carries more noise than
> either arm alone. n=750 resolved exactly one role on the single-difference EM
> metric (§14). **Assume n=750 is underpowered for the contrast and say so** —
> report Phase H at n=750 as directional, and treat ev-F1 (§5c), which resolved
> two roles at n=750, as the better-powered instrument for it. Do not raise n
> unilaterally; it is a human decision (see the Gate 2 precedent above).

---

## 5a. Expected values — what "healthy" looks like

**Read this before reporting any result as a problem.** Small models on multi-hop QA score far lower than people expect, and a correct pipeline will look broken if you calibrate to the wrong numbers.

| Metric | Healthy | Concerning | Broken — stop and fix |
|---|---|---|---|
| **Parse success rate** (per agent, FP16 1.5B) | > 90% | 70–85% | < 70% |
| **Answer EM** (FP16 1.5B baseline) | 30–45% | 20–30% | < 15% |

Context for the EM range: fine-tuned SOTA on HotpotQA distractor is ~68–72 EM; GPT-4-class few-shot is ~50–60. A 1.5B–3B model in a four-agent pipeline landing at **35 EM is a healthy result, not a defect.** Do not attempt to "fix" the pipeline toward 70%.

**These thresholds are for the 1.5B reference configuration only.** The 0.5B model is a smaller model and is *expected* to parse worse and score lower — that is the intervention, not a defect. Do not "fix" a Phase S run because it falls below the table above, and do not apply the §11a Gate FIX-FIRST rule to it. The only thing that would invalidate Phase S is the floor case in §5b's contingencies: the 0.5B model failing so completely that no ranking is measurable.

Absolute failure level matters less than dynamic range — the experiment measures deltas — but a high baseline compresses headroom and adds noise.

---

## 5b. PRE-REGISTERED MECHANISM PREDICTIONS (added 2026-07-29, after Gate 2)

**Why this section exists.** §1 asserts a mechanism: quantization damages output
format and calibration while leaving knowledge intact. The 4-bit tier on model 1
**refuted the format half of that claim** on three independent instruments,
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

### Predictions (1–4 committed 2026-07-29; 5–7 committed 2026-08-01, before Phase S ran)

Confirmatory test for 1–4 is the planned **n=5000 rerun**. Model 2 at n=750 was already
executing when 1–4 were written, so it is **analyst-blind but not pre-data** — treat it
as supporting evidence, not confirmation. Predictions 5–7 are pre-data with respect to
Phase S, which had not been run in any form when they were written.

1. **Format is NOT damaged by quantization.** For every role, at 4-bit, the paired delta
   in parse success, strict-format compliance, and verbatim fidelity each have a 95% CI
   containing zero, or are negative (favouring the quantized run).
2. **Selection churn is HIGH.** >50% of Extractor calls select a different span set at
   4-bit than at FP16. >25% of final answers change text.
3. **Role type predicts quantization sensitivity.** Pooled EM drop for format-heavy roles
   (Step Definer, Extractor) exceeds knowledge-heavy roles (Planner, QA), with the
   contrast's 95% CI excluding zero.
4. **Calibration is the open question.** §1's calibration claim has never been measured.
   Directional only: if any part of §1's mechanism survives, it is calibration — AUROC of
   answer confidence against correctness degrades under 4-bit while accuracy does not.
5. **Format IS damaged by shrinking.** This is the discriminating prediction, and it is
   the mirror of 1. At 0.5B, parse success and strict-format compliance drop with a 95%
   CI **excluding** zero on at least two of the four roles. If both axes leave format
   intact, then format is simply not the mechanism for either and §1's rationale is dead
   on both halves — say so.
6. **The two rankings are not the same ranking.** Spearman correlation between the
   per-role quantization-cost ordering (Phase Q) and the per-role size-cost ordering
   (Phase S) is < +0.6. The strong form of §1's hypothesis — that they are inverted —
   predicts a *negative* correlation; it is recorded here as the strong form and is NOT
   what prediction 6 asserts.
7. **Role-aware beats uniform, but only just.** At matched deduplicated footprint (§5d),
   the best role-aware allocation exceeds the best uniform allocation on EM by a positive
   margin whose 95% CI excludes zero, and that margin is under 5 pp.

Predictions 1 and 2 are well powered already. Prediction 3 is the one n=5000 is for.
Prediction 4 requires `generation.log_confidence: true` (§7). Predictions 5–7 are scored
on Phases S and D and are the reason those phases exist.

**A failed prediction here is a result, not a problem to engineer around.** If 6 fails —
the rankings coincide — the honest finding is that quantization and size reduction are
interchangeable at this scale, role-aware allocation buys nothing, and practitioners
should just pick whichever is cheaper to deploy. That is a genuinely useful negative
result and it is what §2 Phase D is designed to be able to state cleanly.

### Contingencies

- **Floor effect** — if a treatment collapses *all four* roles to near-zero, there is no ranking to measure. In order: (1) for Phase Q, switch to the 8-bit tier where degradation is gentler; for Phase S, there is no gentler size step in the Qwen2.5 family below 1.5B, so instead (2) move the reference model to `Qwen/Qwen2.5-3B-Instruct` so that 1.5B becomes the small model and the ratio is 2x instead of 3x, then (3) report uniform collapse as the finding. Do not silently tune around it.
- **Baseline EM below 15%** — the pipeline is broken, not the model. Inspect raw outputs before changing anything else. This applies to the 1.5B reference only, never to a 0.5B arm.
- **Local smoke test OOM** — Qwen2.5-1.5B at FP16 is ~2.9 GB against a 4 GB card. If it OOMs, run the smoke test at 4-bit or on the 0.5B model instead. You are checking plumbing, not measuring anything.
- **0.5B parse collapse** — if the 0.5B model's parse-failure rate exceeds ~50% on any role, Phase S measures "the small model cannot emit JSON" rather than "the small model is worse at this role." That is still a finding, but it is a *different* finding and must be labelled as such, not folded into the size-cost ranking. §12 still forbids fixing it with constrained decoding or a per-size prompt.

---

## 5c. EXTRACTION ACCURACY (added 2026-07-29, human-approved after Gate 2)

**Third metric, alongside answer EM/F1 (§5) and parse-failure rate (§5).** Scores the
Extractor's spans against HotpotQA's gold `supporting_facts` — the `(title, sent_id)`
labels marking the true evidence sentences. Implemented in `src/evidence.py`. Pure
re-analysis: nothing here reaches a prompt, so it cannot change generation.

**Why it was needed.** Nothing measured extraction *correctness*. `parse_status` says
the JSON was well-formed; `verbatim_rate` says spans were copied not paraphrased;
`selection_changed` says spans differed between runs. A model that verbatim-copies a
completely irrelevant sentence scores perfectly on all three. So every claim about
"extraction under quantization" was unsupported — we could show 73.8% of Extractor
calls select different evidence at 4-bit, but not whether the new evidence was worse.

**Why set-F1 over labels, never token overlap.** Token-F1 against sentence-length
references ranks answers backwards. With a 15-token gold fact:

| candidate | F1 |
|---|---|
| incomplete but verbatim copy | 0.59 |
| half copy + fabricated clause | 0.51 |
| complete faithful paraphrase | 0.46 |

Best score to the least informative answer, worst to the only correct one — because
precision taxes a synonym exactly as hard as an invented fact. Comparing discrete
`(title, sent_id)` labels is immune: fabrication earns nothing, paraphrase is not
punished. This is also HotpotQA's own official supporting-facts metric, so numbers are
comparable to published work. **Do not re-implement this as text similarity.**

**Built-in negative control.** In Phase Q and Phase S alike, the `qa_*` run's ev-F1
delta must be exactly 0.00: QA runs after extraction, so perturbing it cannot alter
extractor output. **A non-zero value there means the metric is broken — check it
first.** It also confirms generation is deterministic under identical batch composition.

**Known limitation, stated not hidden.** 26% of spans are too short to attribute and
17% match no sentence, so absolute ev-F1 (37.5% baseline) is a FLOOR, not the true
value. Attribution quality is near-identical across runs, so the deltas remain valid.
`MIN_ATTRIBUTABLE_CHARS = 25` is a judgment call; `analyze.py` records the sensitivity
sweep. **Phase S must re-check that assumption**: a smaller model may emit
systematically shorter or more paraphrased spans, which would change attribution
quality between arms and break the "near-identical across runs" premise the deltas
rest on. Report the attribution stats per arm, not just for the baseline.

---

## 5d. MEMORY ACCOUNTING (NEW — read before quoting any footprint)

**Every capacity claim in this paper is a claim about bytes, and the byte number v1 reported was wrong.**

v1 computed `coresident_footprint_mb` as the sum of all four stages' weight footprints. Because all four roles run the *same weights at the same precision* in the `baseline` run, that summed four copies of one model:

    baseline (v1)      = 4 x 2944.4 = 11777.6 MB
    any *_4bit  (v1)   = 3 x 2944.4 + 1070.2 = 9903.4 MB
    "saving"           = 1874.2 MB, identical across all four runs

That last line was reported as a controlled constant proving all four configurations save the same bytes. **It is an artifact.** A deployment does not load four identical copies of one model; it loads one and swaps prompts. The honest numbers are:

    baseline (deduped)     = 2944.4 MB      one fp16 instance
    any *_4bit (deduped)   = 2944.4 + 1070.2 = 4014.6 MB     two instances

**Role-aware allocation with one role at 4-bit therefore does not save 1874 MB. It costs an extra 1070 MB** relative to uniform FP16, because mixing precisions is precisely what forces a second resident copy. The sign is inverted. This does not affect any accuracy result in §14 — those never used the footprint — but it invalidates the deployment framing v1 was heading toward, and it is why §2 Phase D compares against a *frontier* rather than against a single "saving."

**Required from now on.** Log both, and make the assumption explicit:

- `coresident_footprint_mb` — sum over stages, i.e. **one resident instance per stage**. Correct for a deployment that runs each agent as its own model server. Keep it; it is the right number under that topology, and it is what makes the four Phase Q runs comparable to each other.
- `deduped_footprint_mb` — sum over **distinct `(model_id, precision)` pairs** actually used by the run. Correct for a single-process deployment that loads each distinct configuration once. **This is the number the Pareto frontier in §10 Figure 4 is plotted against, and the number the paper leads with.**

Both are computed from the same per-stage metadata and neither is an estimate. Report the topology assumption in the caption of every figure that has memory on an axis. A reader who assumes the wrong topology reads the plot backwards.

Note the interaction with Phase D: `ma_uniform_4bit` (all four roles at 4-bit) has a *deduped* footprint of 1070.2 MB — lower than any single-role-quantized run, because it needs only one instance. Uniform allocation is memory-efficient precisely because it is uniform. That is the frontier role-aware allocation has to beat, and it is a harder target than v1's arithmetic suggested.

---

## 5e. SELECTION VS CONFIRMATION (NEW — the anti-HARKing device for Phase D)

Phase D's allocation is *chosen* using Phase H's results. Scoring it on the same questions would be selecting on noise and reporting the winner's curse as a finding. Effect sizes in this project already shrank by ~half between n=300 and n=750 (§14) — the curse is measurably present here, not hypothetical.

**Two disjoint question sets:**

| set | seed | n | used by |
|---|---|---|---|
| selection | 7 | 750 | Phases Q, S, H |
| confirmation | 8 | 750 | Phase D only |

**Disjointness must be enforced in code, not assumed.** `random.sample(range(N), k)` with a fixed seed is nested in `k` but says nothing across seeds — two independent 750-question samples from 7405 overlap by ~76 questions in expectation. `load_questions` therefore takes an `exclude` argument, the confirmation set is drawn from the complement of the selection set, and a runtime assertion fails the run if the intersection is non-empty. There must be a test for this.

**The allocation rule must be committed to `config/experiment.yaml` before Phase D runs**, as a literal per-role treatment table, with the Phase H numbers that produced it in a comment. The rule itself is fixed here and is deliberately crude — a complicated selection rule fitted to four noisy numbers is the same overfitting in a different coat:

> For each role, assign the treatment with the **lower measured cost** in Phase H. Where the axis contrast's 95% CI includes zero, assign **4-bit**, because it is the cheaper of the two to deploy at matched accuracy and defaulting to it makes the role-aware arm harder to distinguish from uniform, not easier.

Two allocations are built from that rule: `ma_optimized_hi` (roles whose cost CI excludes zero keep FP16) and `ma_optimized_lo` (every role takes its cheaper treatment). They bracket the budget range and give the frontier two points instead of one.

**If the rule produces an allocation identical to a uniform one, that is the finding.** Report it and do not hand-adjust the rule to manufacture a distinct arm.

---

## 5f. MULTIPLICITY — one primary test, everything else descriptive (NEW)

**This is the most likely statistical objection to the paper and it is currently
unanswered.** Phase Q tests 4 roles. Phase S tests 4 more. Phase H tests 4 contrasts.
On three metrics (EM, F1, ev-F1) that is up to 36 hypothesis tests, and §14 reports
"Extractor +3.20 [+0.53, +5.87], SIGNIFICANT" with no correction. Under Bonferroni at
even 4 tests (α = 0.0125) that interval no longer excludes zero. A referee who checks
will say the headline result is a multiple-comparisons artifact, and on the current
framing they would be right.

**Fix: designate one pre-registered primary test and demote the rest, rather than
correcting 36 tests into oblivion.**

- **PRIMARY (confirmatory, one test, no correction needed).** §5b prediction 3's
  contrast: pooled cost for format-heavy roles (Step Definer, Extractor) minus
  knowledge-heavy roles (Planner, QA). This was pre-registered on 2026-07-29, before the
  confirmatory data existed, and it is a *single* number. It is also better powered than
  any per-role test because it pools two roles per side. Report it on EM and on ev-F1,
  and say which was pre-specified as primary — **EM**, because §5 names accuracy primary.
- **SECONDARY (pre-registered, Holm-corrected).** §5b predictions 5, 6, 7. Four tests.
  Use Holm–Bonferroni, which is uniformly more powerful than Bonferroni and needs no
  independence assumption — the tests share a baseline and are positively correlated,
  which Holm tolerates and Šidák does not.
- **DESCRIPTIVE (no significance claims at all).** Every per-role number, both rankings,
  and the Spearman correlation. Report point estimates with CIs and describe them as
  estimates. **Do not write "significant" next to a per-role result.** Say "the Extractor
  is the only role whose interval excludes zero uncorrected" — which is true, informative,
  and not a significance claim.

**Consequence for §14 and for how the paper is written.** The current draft leads with a
four-way role ranking. It should lead with the format-heavy vs knowledge-heavy contrast,
because that is the one claim the design can actually license. The ranking becomes a
figure and a paragraph of description, not the headline. This is a *presentation* change,
not a re-analysis — every number stays the same.

**Note on Phase H's contrast variance.** `size cost − quantization cost` is a difference
of two differences, but both are paired against the *same* baseline on the *same*
questions, so they are positively correlated and Var(A−B) = Var(A) + Var(B) − 2Cov(A,B)
is materially less than 2·Var. **Compute the paired bootstrap on the per-question
contrast directly; do not estimate it by adding the two arms' variances.** Doing the
latter overstates the interval and would bury a real effect.

---

## 6. Execution model — stage-major with checkpointing

This is the part that makes it fit in Kaggle's quota. Read carefully.

**Do not run question-by-question through the whole pipeline.** Run **stage-by-stage across all questions**:

```
for stage in [planner, step_definer, extractor, qa]:
    load the model this run assigns to this stage, at the precision it assigns
    process ALL n questions through this stage in batches
    write outputs to disk
    unload model, free VRAM
```

Two reasons this is required:

1. Only one model is ever resident, so a 4 GB GPU works and a 16 GB T4 is comfortable.
2. It enables large batch sizes, which is the difference between ~43 GPU-hours and ~12 for the full sweep.

**Consecutive stages sharing the same (model, precision) may keep the model loaded** rather than unloading and reloading it. This is a pure wall-time optimization, it changes no output — but it must not change `peak_vram_mb` accounting, so reset the peak counter per stage regardless. `single_*` runs have one stage and skip all of this.

**Batch size:** start at 16, auto-reduce on OOM. Left-pad for decoder-only batched generation and confirm the tokenizer's pad token is set. **Batch size must be identical across any runs being compared**, because greedy decoding is deterministic for a fixed batch but not across re-batchings (§5), and because `latency_s` is derived from batch wall-time (§7). If autotune fires on one run and not another, the pair is no longer comparable — record it and say so.

**Checkpointing is mandatory.** Kaggle GPU sessions cap around 12 hours and can terminate without warning.

- Append every agent call to a JSONL file as it completes.
- On startup, read existing output and skip any `(question_id, stage, call_index)` already present.
- Resuming must be the default behavior, not a flag.
- Flush to disk at least every 50 records.
- **One venue per run.** Resume is blind to which machine produced a record, so a run started locally and finished on Kaggle would interleave two execution stacks with no key collision to flag it. Never split a run.

---

## 7. Logging schema

One JSONL record per agent call:

```json
{
  "run_id": "stepdef_small",
  "question_id": "5a8b57f25542995d1e6f1371",
  "stage": "step_definer",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "precision": "fp16",
  "call_index": 0,
  "prompt_tokens": 812,
  "output_tokens": 96,
  "latency_s": 1.84,
  "raw_output": "...",
  "parse_status": "ok",
  "parsed": {},
  "prompt_version": "v5",
  "timestamp": "..."
}
```

`model_id` per call is **new in v2 and required**: with Phase S the base model varies within a run, and a record carrying only `precision` cannot say which model produced it.

Plus one record per question with the final answer, EM, and F1.

Plus a run-level metadata blob: per-stage model ID and quantization config (method, group size, compute dtype), batch size, git commit, GPU name, total wall time, library versions, and **memory**:

- `peak_vram_mb` per stage, from `torch.cuda.max_memory_allocated()`
- `weight_footprint_mb` per stage, computed as `sum(numel * element_size)` over parameters
- `coresident_footprint_mb` — sum over stages (one instance per stage)
- `deduped_footprint_mb` — sum over distinct `(model_id, precision)` pairs

See §5d for what those two mean and which one the paper leads with. **Do not reintroduce the `params * bytes_per_param` shortcut for the footprint**: a `Params4bit` tensor's `.numel()` is the packed byte count, so multiplying by 0.5 applies the compression twice and understates 4-bit by 2.6x. That bug shipped once already.

`latency_s` is **batch wall-time divided by batch size** — inverse throughput, not user-facing latency. It is comparable only between runs sharing a batch size. Label it as such wherever it is printed; Phase S makes it tempting to read as a speed result, and a 0.5B stage genuinely is faster, so the axis is real but the units are not what a reader assumes.

Keep `raw_output` always. You will need it to build the failure taxonomy and to show examples in the paper.

---

## 8. Repo structure

```
marag-precision/
  SPEC.md                  # this document
  PROGRESS.md              # updated before every gate
  config/
    experiment.yaml        # run definitions, models, n, seeds, batch size
  src/
    models.py              # load a model at a given precision; footprints; generation
    prompts.py             # one prompt template per role, versioned
    agents.py              # role call wrappers
    pipeline.py            # stage-major orchestration, question sampling
    parsing.py             # parsers + failure taxonomy
    metrics.py             # EM/F1, bootstrap CIs, AUROC, ECE
    evidence.py            # extraction accuracy vs gold supporting_facts (§5c)
    mechanism.py           # strict format, verbatim rate, selection churn (§5b)
    runner.py              # sweep driver, checkpoint/resume
  notebooks/
    kaggle_run.ipynb       # thin wrapper, see §9
  results/                 # JSONL outputs
  analyze.py               # produces every figure and table in §10
  gate2_report.py          # Gate 2 report (superseded by analyze.py; kept for provenance)
  smoke_test.py            # gate smoke-test driver
  README.md
```

Everything driven by `config/experiment.yaml`. **No interactive prompts, no hardcoded paths, no absolute paths belonging to one developer's machine.** Every script must run unchanged on Windows local, macOS, and Kaggle Linux. It must run as:

    python -m src.runner --config config/experiment.yaml --run stepdef_small

### Config schema for two capacity axes

A run maps each stage to a treatment. Backward compatible: a bare string is a precision on the base model, so every Phase Q run definition already written stays valid.

```yaml
models:
  base:  Qwen/Qwen2.5-1.5B-Instruct
  small: Qwen/Qwen2.5-0.5B-Instruct

runs:
  baseline:                      # bare string == {model: base, precision: <string>}
    planner: fp16
    step_definer: fp16
    extractor: fp16
    qa: fp16
  stepdef_small:
    planner: fp16
    step_definer: {model: small, precision: fp16}
    extractor: fp16
    qa: fp16
  single_fp16:
    solo: fp16                   # one stage; §4a
```

**The model is a config value, not a plugin.** Two base models and two small models are supported; do not build an abstraction layer, a registry, or a strategy pattern.

---

## 9. Kaggle notebook

Keep it a launcher, not where logic lives.

1. `!pip install` the pinned versions
2. `!git clone <repo>` (or pull latest)
3. Print `nvidia-smi`, confirm T4 and free VRAM
4. Push credentials setup (results are pushed to GitHub after every run — a session that dies loses at most the run in flight)
5. `!python -m src.runner --config config/experiment.yaml --run <run_id>`
6. Zip `results/` into `/kaggle/working/` for download

Notes to surface in the notebook markdown:

- **Internet must be enabled** in notebook settings (Settings → Internet On), otherwise HuggingFace downloads fail.
- Accelerator set to **GPU T4 x2** (use one; the second is idle but the quota is the same).
- `/kaggle/working` holds ~20 GB. Model cache goes to `/kaggle/tmp` to avoid filling it.
- Save results **incrementally**, not just at the end.
- Phase S downloads a second model (~1 GB). Phase D on model 2 downloads four. Budget cache space.

---

## 10. Analysis output

`analyze.py` produces every figure and table below **from `results/` alone**, with CIs, and must not require a GPU or a network round-trip beyond loading the dataset for gold labels. It supersedes `gate2_report.py`.

**Figure 1 — quantization sensitivity.** EM drop from baseline, per role, at each bit-width. Four roles on the x-axis, bars for 8-bit / 4-bit / 3-bit, error bars from paired bootstrap CIs.

**Figure 2 — the mechanism.** Parse-failure rate per role, per treatment, same layout. Include both axes: 4-bit and 0.5B side by side. This is where prediction 5 (§5b) is scored.

**Figure 3 — role type vs pipeline position.** Accuracy-drop data re-cut two ways: by **role type** (format-heavy: Step Definer, Extractor | knowledge-heavy: Planner, QA) and by **pipeline position** (upstream: Planner, Step Definer | downstream: Extractor, QA). This separates two competing explanations — that damage tracks the *kind of work* a role does, versus *how far downstream it propagates*. No new runs; a re-cut of existing data.

> **Pool by resampling questions, never by stacking runs.** Earlier write-ups described
> these re-cuts as "pooled, so n is effectively doubled." It is not — stacking two runs'
> per-question deltas reuses the same 750 questions against the same baseline, so the
> two contributions are correlated and the CI comes out too narrow. This is
> pseudo-replication and at n=750 it **flips a significance call**: format-heavy is
> +1.87 [+0.13, +3.53] stacked naively (excludes zero) but +1.87 [−0.07, +3.87] under a
> question-clustered bootstrap (includes zero). **Resample question ids and carry both
> roles' deltas for the drawn question.** The *contrast* between groups was always
> computed correctly and is unaffected (+1.87 [−0.13, +4.00], logged as not confirmed).

**Figure 4 — the money plot (NEW).** Answer EM against `deduped_footprint_mb` (§5d), one point per Phase D arm, with CI whiskers on EM. Uniform MA-RAG points connected into a frontier; single-call RAG points connected into a second frontier; role-aware points plotted as distinct markers. **The paper's central claim is the vertical gap between the role-aware markers and the uniform frontier at the same x.** Caption must state the deployment topology assumption (§5d).

**Table 1.** Per-run EM, F1, parse-failure rate, each with 95% CI, plus mean inverse-throughput, peak VRAM, coresident and deduped footprint.

**Table 2 — the head-to-head (NEW, Phase H).** Per role: quantization cost, size cost, and the axis contrast, each with a paired bootstrap 95% CI, plus the footprint of each treatment and the 13.6% budget gap flagged explicitly. Report on EM, F1 and ev-F1. This is the paper's central table.

**Table 3 — ranking comparison (NEW).** Our quantization ranking, our size ranking, Spearman correlation between them with a CI, and MA-RAG's published size ranking **in a clearly separated block labelled as related work, not as a control arm** (§1). State plainly whether our two in-house rankings match, differ, or are indistinguishable given the CIs — that comparison is admissible because it is internal. The MA-RAG column is context for the reader and nothing is inferred from agreement or disagreement with it.

---

## 11. Build order

Steps 1–9 are complete (§14). Continue from step 10.

1. ~~Pipeline at FP16 only, hardcoded, 10 questions, all raw outputs printed.~~ done
2. ~~Parsers plus failure taxonomy.~~ done → Gate 1 passed
3. ~~Stage-major refactor with disk-cached intermediate state.~~ done
4. ~~Checkpoint and resume.~~ done
5. ~~Precision switching, baseline plus the four 4-bit runs.~~ done
6. ~~Batching and batch-size autotuning.~~ done
7. ~~Kaggle notebook.~~ done
8. ~~Scale to n=750 on model 1, full 4-bit tier.~~ done → Gate 2 passed
9. ~~Model 2 wiring.~~ done (sweep not yet run)
10. **Repo repairs (§13a).** Fix the deduped-footprint bug, the hardcoded path, the per-call `model_id`, and the stale docs. No new science. **Do this first — Phase S metadata is wrong without it.**
11. **Per-stage model support.** Config schema in §8, `{model, precision}` per stage, backward compatible with bare strings. Validate at n=5 that a mixed-model run really loads two different models.
12. **`analyze.py`.** Figures 1–3 and Tables 1–3 over the *existing* Phase Q results. Must reproduce §14's numbers before Phase S runs — if it cannot reproduce a result already reported, the analysis code is wrong and everything after it is untrustworthy.
    → **STOP. See §11a Gate 3.**
13. **Phase S**, 4 runs on model 1, selection set (seed 7, n per §2's compute budget).
14. **Phase H** analysis. Table 2, Table 3, Figure 2 with both axes. Score predictions 5 and 6.
    → **STOP. See §11a Gate 4.**
15. **Single-call RAG role** (§4a) plus the disjoint confirmation sampler (§5e), validated at n=5.
16. **Phase D**, 8 runs on the confirmation set (seed 8, drawn with `exclude=`). Figure 4. Score prediction 7.
17. Phase Q + S on **model 2**, selection set.
18. 8-bit tier on model 1, if time allows.
19. 3-bit via GPTQ, only if everything above is done.

---

## 11a. MANDATORY STOP GATES

**You must halt and wait for explicit human approval at each gate.** Do not continue past one on your own initiative, do not begin the next stage "while waiting," and do not treat silence as approval.

Before each gate: `git commit` and update `PROGRESS.md` (see §0).

### GATE 1 — after build step 2 — PASSED 2026-07-29

Smoke test: 10 questions, FP16, all four agents, locally. Report raw outputs, parse counts, answers vs gold, throughput, and a PROCEED / FIX FIRST recommendation judged against §5a.

### GATE 2 — after build step 8 — PASSED 2026-07-29

First complete 4-bit tier on model 1. Outcome recorded in §14.

> **AMENDED 2026-08-01: time-boxed GPU access may collect data before Gate 3.**
> The gate exists to stop new data being *generated* with analysis code that has never
> been checked against a known answer. That risk does not apply to the generation path:
> `runner.py` / `agents.py` / `models.py` / `parsing.py` are the same code that produced
> the n=750 tier, and the parser change was verified byte-identical on all 33,426
> existing records. Only `n` and `batch_size` differ.
>
> So when GPU access expires on a clock — a rented A100, a workshop deadline — **run
> Phases Q and S first and analyse afterwards.** The GPU is the scarce resource; the
> analysis is CPU-bound and has no deadline. Gate 3 still blocks *reporting* anything,
> and `analyze.py` must still reproduce §14 from the n=750 pilot before any n=3000 number
> is believed. What is forbidden is skipping the gate, not reordering it against a
> hardware constraint. Record in `PROGRESS.md` that this amendment was used.

### GATE 3 — after build step 12 (`analyze.py` reproduces Phase Q)

Stop. Produce:

- Every §10 figure and table that Phase Q data can support, from `analyze.py` alone
- **A line-by-line diff against §14's numbers.** Any discrepancy is a bug in the analysis code and blocks the gate — §14 was computed by `gate2_report.py`, and two independent implementations disagreeing means at least one is wrong.
- The corrected memory table (§5d), both topologies, with the v1 error stated plainly
- Confirmation that no absolute path, no developer-specific path, and no interactive prompt remains anywhere in the repo

Then **WAIT**.

### GATE 4 — after build step 14 (Phase H complete)

Stop. Produce:

- **Table 2**: per-role quantization cost, size cost, and axis contrast, with paired 95% CIs, on EM / F1 / ev-F1
- **Table 3**: both in-house rankings and their Spearman correlation, with MA-RAG shown as related work only
- Explicit scoring of predictions **5** (format damaged by shrinking) and **6** (rankings differ) — including "not resolved at this n" where that is the answer
- Parse-failure rates for the 0.5B arms, with the §5b "0.5B parse collapse" contingency checked
- Attribution stats per arm (§5c) — confirm the ev-F1 deltas are still admissible
- The proposed Phase D allocation, derived mechanically from the §5e rule, with the numbers that produced it
- Flag anything that looks like a bug rather than a finding — a role showing exactly zero change (outside the §5c negative control, where zero is required), CIs spanning the full range, parse-failure rates of 0% or 100%
- Recommendation on whether to proceed to Phase D

Then **WAIT**.

---

## 12. Non-goals — do not build these

- Any retriever or vector store
- Training, fine-tuning, distillation, LoRA, quantization-aware training
- Iterative replanning or agent-to-agent negotiation loops
- A web UI, dashboard, or CLI beyond the single runner entrypoint
- A multi-model abstraction layer. Four models are supported (§3), but a model is a **config string**, not a plugin architecture.
- Prompt optimization, per-precision prompt tuning, or **per-model-size prompt tuning**. The 0.5B model will parse worse than the 1.5B. That is the measurement.
- Multi-dataset support (HotpotQA only; 2WikiMQA is a later stretch, not now)
- Automatic retry on parse failure — this actively breaks the experiment
- Any allocation-search procedure beyond the fixed rule in §5e. No greedy search, no hill-climbing, no per-role budget optimizer. Four roles and two treatments is 16 allocations; fitting a search over 750 noisy questions would produce a winner's-curse artifact and nothing else.
- **Constrained or grammar-based decoding — forbidden.** No `outlines`, no
  `guidance`, no `lm-format-enforcer`, no JSON-schema decoding, no logit masking,
  no `transformers` constrained-beam or prefix-allowed-tokens generation, no
  `response_format`-style structured-output modes.

  This is the single most tempting engineering fix in the whole project and it
  must never be applied. Constrained decoding drives the parse-failure rate to
  zero *by construction*. The parse-failure rate is the mechanism evidence for
  the paper's argument. Forcing valid output does not fix the model; it deletes
  the measurement and makes every run look identical on the secondary metric.

  The temptation is now **worse**, not better: the 0.5B model parses worse than the
  1.5B, and prediction 5 (§5b) says it should. Constraining it would erase exactly
  the effect Phase S exists to measure. Generation stays plain greedy
  (`do_sample=False`) with no logit processors.

**Stated limitation, not a non-goal.** Qwen2.5-0.5B and Qwen2.5-1.5B differ in depth, width and training mix, not only in parameter count, so Phase S measures "swap in the smaller sibling," not "remove parameters." No public model family offers a clean parameter-count intervention without retraining, which §12 forbids. Same-family, same-tokenizer, same-instruct-recipe is the closest available approximation, and it is the same approximation MA-RAG's ablation makes. Say this in the paper's limitations; do not pretend the intervention is cleaner than it is.

---

## 13. Acceptance criteria

- `python -m src.runner --config config/experiment.yaml --run baseline` completes the configured n and writes valid JSONL
- A mixed-model run (`--run stepdef_small`) loads two *different* models and records the correct `model_id` on every call record
- A `single_*` run completes with one stage and is skipped, not silently averaged, by four-stage analyses
- Killing the process mid-run and rerunning resumes without duplicating work
- The confirmation set is provably disjoint from the selection set, enforced by assertion and covered by a test — **derived with `exclude=`, never inferred from `random.sample` nesting (§13b item 4)**
- Parse failures appear in output with correct taxonomy labels, zero retries
- No absolute or developer-specific path anywhere in the repo; every entrypoint runs on Windows, macOS and Kaggle Linux
- `deduped_footprint_mb` and `coresident_footprint_mb` are both recorded and differ for any mixed-precision run
- Phase Q + Phase S complete inside half the available wall-clock, per §2's compute-budget table; Phase S is cheaper than Phase Q (the 0.5B stage replaces a 1.5B one) and must not exceed it
- `analyze.py` emits every §10 figure and table from `results/` alone, with CIs, and reproduces §14

**Peak VRAM.** v1 required "under 6 GB at FP16 with batch size 16." That was written for n=300 and no longer holds: peak is a max over batches, so at n=750 the extractor stage reaches 7510–8162 MB with the same batch size. Restated: **peak VRAM must stay under 14 GB** (the usable T4 budget) at the configured batch size, and any OOM autotune firing must be recorded in metadata. The 6 GB figure was never a scientific constraint, only a fit-on-the-card one, and the card it referred to is not where the experiment runs.

### 13a. Known defects to repair at build step 10

Found by audit 2026-08-01. All are repo defects, not design changes.

1. **`coresident_footprint_mb` counts identical instances separately.** §5d. Add `deduped_footprint_mb`; keep both; fix the framing in `PROGRESS.md` and any draft text.
2. **`gate2_report.py` hardcodes an absolute Windows path** (`sys.path.insert(0, r"C:\Users\maxim\...")`). Breaks on every machine but one, including Kaggle. Resolve the repo root from `__file__`.
3. **Call records carry `precision` but not `model_id`.** Phase S is unanalysable without it (§7).
4. **`config/experiment.yaml`'s `dataset.name`, `dataset.config` and `dataset.split` are never read.** `load_questions` hardcodes them. §8 requires everything to be config-driven: either wire them through or delete them, but do not leave dead config that looks live.
5. **`src/pipeline.py` `build_stage_calls` binds a local named `evidence`,** shadowing the imported `evidence` module for the whole function. Harmless today, an `UnboundLocalError` the moment anyone uses the module in that function.
6. **`src/mechanism.py` `SELECTION_FIELD["qa"] = "answer"` is a scalar,** but `selection_changed` iterates its argument — so for QA it compares lists of *characters*. It gets the right answer for the wrong reason and breaks on whitespace-only differences. Handle scalar fields explicitly.
7. **`analyze.py` does not exist** though §8 and §13 require it. Build step 12.
8. **`README.md` is stale** — claims build steps 3–12 are unimplemented, omits half the modules, and describes `results/` as gitignored when results are force-added and committed.
9. **`.gitignore` ignores `results/`** while every sweep force-adds it. Carve out the committed results explicitly so `git add -A` stops being a trap.
10. **`src/metrics.py` `bootstrap_ci` is a pure-Python double loop** — 10k resamples x n draws per CI. Tolerable at n=750, roughly 7.5M operations per interval, and `analyze.py` computes dozens. At the planned n=5000 it is 50M per interval and will dominate analysis wall-time. Vectorize it before that rerun, not after.
11. **`bootstrap_ci` is order-sensitive at a fixed seed.** The same 750 F1 values give [38.21, 44.64] in file order and [38.16, 44.50] sorted by question id — the RNG draws indices, so input order changes which values are drawn. The spread is the same magnitude as seed-to-seed Monte Carlo noise (~0.15 pp) and no reported conclusion moves, but a number that goes in the paper must not depend on the order records happened to be written in. Sort inside the function.
12. **`gate2_report.py` Table 2b bootstraps per-call latency, which pseudo-replicates ~16x.** `latency_s` is batch wall-time divided by batch size, so all 16 calls in a batch carry the *same* value — only 43–101 distinct values exist per (run, stage). Bootstrapping over calls treats them as independent and yields CIs roughly 5x too narrow: planner reads +0.0735 s [+0.0704, +0.0765] where the honest batch-level comparison is 0.154 [0.147, 0.162] vs 0.228 [0.217, 0.240]. The printed caveat covers the *units* but not the CI width. Resample batches, not calls. Substantively: planner (+48%) and step_definer (+26%) slow down under 4-bit above session noise; the extractor's +9.1% does **not** clear the cross-run fp16 noise band for that stage (0.946–1.053) and must not be reported as resolved.
13. **`PROGRESS.md` claims QA's parse delta "projects to [−0.65, −0.01], significantly *better* under 4-bit".** The realised n=750 value is −0.267 [−0.667, **0.000**] — the upper bound is exactly zero. Do not claim significance. Fix the claim where it appears.

### 13b. OPEN — needs a human decision at Gate 3, do NOT change unilaterally

Each of these would move a number that has already been reported. §0 rule 4 applies:
raise them, do not silently substitute a fix.

1. **The Step Definer's parse-failure rate is mostly an emptiness check the parser
   says it does not perform.** `_validate_step_definer` rejects `target_entity` when
   `not entity.strip()`. Across the five n=750 runs, **55 of 63** Step Definer failures
   are `{"search_terms": [...good...], "target_entity": "", "answer_type": "..."}` —
   correct types, usable keywords, one empty string. Only 7 are genuinely malformed.
   `parsing.py`'s own module comment says semantic quality is deliberately not checked
   because "folding them into the taxonomy would make the parse-failure rate measure
   two different things at once" — which is exactly what is happening, on a role §1
   singles out. It also costs accuracy twice: the validator returns `None`, so the
   *whole* spec is discarded and the Extractor falls back to `(unspecified)`/`(none)`
   rather than keeping the good `search_terms`. Options: (a) accept empty
   `target_entity` as `ok` and let the Extractor use the terms, (b) keep the failure
   but stop discarding the parsed payload, (c) leave it and document. All three change
   published Step Definer numbers. **(b) is the recommendation** — it fixes the
   accuracy leak without redefining the metric mid-experiment.
2. **`evidence.attribute_span` applies `MIN_ATTRIBUTABLE_CHARS` to the span but never
   to the index sentence.** Matching is bidirectional, so any context sentence shorter
   than the span can be claimed by a correct span — including sentences in *distractor*
   paragraphs. Demonstrated: a correct verbatim gold span co-matches a 7-character
   distractor sentence ("in 1804"), taking precision from 1.00 to 0.50 on that question.
   This deflates ev-P/ev-F1 asymmetrically by which distractors a question happened to
   draw, and **the published MIN_ATTRIBUTABLE_CHARS sensitivity sweep cannot detect it**
   because sweeping that constant only moves the span-side guard. Fixing it changes
   every §5c number. Decide at Gate 3, then re-run the §5c analysis once.
3. **Selection churn compares ordered lists, but §5b prediction 2 is phrased as a
   "different span *set*".** A pure reordering counts as churn, so the 73.8% / 75.9%
   figures are upper bounds by an unmeasured amount. Either re-phrase the prediction to
   "sequence" or compare sets — but the prediction is pre-registered, so changing its
   wording after seeing data is not available. Measure the set-based number, report both.
4. **`random.sample` nesting is not a language guarantee and breaks above k≈1365.**
   §5's amendment and §5e both assert that a fixed seed makes samples nested in k. That
   is a CPython implementation detail holding only while `21 + 4**ceil(log(3k,4)) < N`;
   for HotpotQA's 7405-row dev split the boundary is **k ≈ 1365**. Verified:
   `sample(300) ⊆ sample(750)` is True, but `sample(750) ⊆ sample(5000)` is **False**
   (overlap 739/750). So the n=300 ⊂ n=750 claim is correct by luck, and **the planned
   n=5000 rerun will not contain 11 of the 750 selection-set questions** — the "free
   check on the shared questions" §5 banks on quietly stops existing. It can also differ
   between CPython versions, i.e. between local Windows and Kaggle Linux. Do not rely on
   nesting: derive every subset relationship explicitly with `exclude=` (§5e) and assert
   it, which `load_questions` now supports.

---

## 13c. ANTICIPATED CRITIQUES — the evaluation, and how it is defended (NEW)

**The evaluation method is sound and standard, but it has six attackable seams.** Five
have real answers; one is a genuine limitation to concede. Write the answers into the
paper rather than waiting for a referee to find them.

**What the metrics actually are** (so nobody mis-describes them in the writeup):

| metric | granularity | how it scores |
|---|---|---|
| Answer **EM** | whole string | official HotpotQA normalization (lowercase → strip punctuation → strip articles → collapse whitespace), then exact equality |
| Answer **F1** | token bag | same normalization, multiset overlap of whitespace tokens, harmonic mean of P and R; yes/no/noanswer short-circuit to 0 on mismatch |
| **ev-F1** (§5c) | discrete `(title, sent_id)` labels | Extractor spans resolved to sentence labels, then **set** P/R/F1 against gold `supporting_facts` |

There is **no claim-level fact atomization** (FActScore / SAFE style) and there should not
be: that is a long-form-generation metric, HotpotQA answers are 1–4 token spans, and
deviating would make the numbers incomparable to MA-RAG and to every published HotpotQA
result. ev-F1 is atomization at the *sentence* level, which is HotpotQA's own official
supporting-facts metric.

1. **"Your headline is a multiple-comparisons artifact."** The strongest objection.
   Answered by §5f: one pre-registered primary contrast, Holm-corrected secondaries,
   everything per-role explicitly descriptive. Requires the paper to lead with the
   format-heavy vs knowledge-heavy contrast, not the four-way ranking.
2. **"EM is too brittle for multi-hop QA — you need atomized fact scoring."** Answered
   empirically, measured on the n=750 baseline (2026-08-01):

   | | count | share |
   |---|---|---|
   | EM=1 | 231 | 30.8% |
   | EM=0 but F1 ≥ 0.5 — arguably correct | 98 | **13.1%** |
   | EM=0, 0 < F1 < 0.5 | 49 | 6.5% |
   | EM=0, F1=0 — genuinely wrong | 372 | 49.6% |

   **EM is brittle — 13.1% of questions are near-misses** (`Marvel Comics` vs `Marvel`;
   `Sir George Cayley` vs `George Cayley`), two thirds pure granularity: 32 where the
   prediction is a substring of gold, 32 where it contains gold. **But brittleness costs
   variance, not validity, because every comparison here is a paired delta on the same
   questions**, so it hits both arms identically and cancels. Demonstrated: EM and F1
   drops agree in sign on all four roles and to within 0.2 pp on three of four (Extractor
   +3.20 vs +3.16), and mean answer length is flat across arms (2.18–2.29 words) — so
   quantization does not change verbosity, which is the only route by which EM could
   penalise one arm more than another. F1's intervals are no tighter than EM's, so there
   is no power argument for switching primary metric either.

   **Atomization is not applicable.** Mean predicted answer is 2.24 words. FActScore and
   SAFE decompose paragraph-length generations into 10–50 claims; the atomic
   decomposition of a two-word span is the span. Adopting it would also break
   comparability with MA-RAG and every published HotpotQA number.

   **The legitimate multi-hop concern is different, and §5c already answers it:** EM
   cannot distinguish real multi-hop reasoning from a lucky guess. That is what
   supporting-facts ev-F1 is for — it scores the reasoning trace rather than the final
   string, and it resolved *two* roles at n=750 where EM resolved one. Frame the three
   metrics as answering three different questions: EM/F1 for comparability with prior
   work, ev-F1 for multi-hop faithfulness, parse-failure for mechanism.
3. **"EM 30.8% is near the floor; your deltas are noise on a broken pipeline."** Answered
   by §5a: fine-tuned SOTA on HotpotQA distractor is ~68–72 EM and GPT-4-class few-shot is
   ~50–60, so a 1.5B model in a four-agent pipeline at ~31–35 EM is the expected range,
   not a defect. Reinforced by the parse rates (94.7–99.7% per role) and by the
   determinism check. Report the §5a table in the paper.
4. **"Your evidence metric is unreliable."** Partly true and must be conceded up front:
   26% of spans are too short to attribute, 17% match nothing, so **absolute ev-F1 is a
   floor, not the true value**. The defence is that only *deltas* are claimed and
   attribution quality is near-identical across arms — which §5c now requires be
   demonstrated per arm, not assumed. §13b item 2 (the index-side length floor) must be
   resolved before ev-F1 goes in a paper.
5. **"Degraded propagation contaminates the accuracy signal."** On a parse failure the
   pipeline substitutes a fixed fallback rather than retrying, so a failure costs
   accuracy indirectly. This is deliberate (§5 forbids retries, which would destroy the
   parse-failure measurement) and it is *uniform across arms*, so it cannot favour one.
   State the policy explicitly and report the clean-vs-dirty EM split, which the harness
   already computes.
6. **Genuine limitations — concede, do not defend.** One dataset (HotpotQA). One base
   model family unless a second lands. Greedy decoding with a single question sample, so
   no generation-variance estimate. Qwen2.5-0.5B vs 1.5B differ in depth and width and
   data mix, not only parameter count, so Phase S measures "swap in the smaller sibling",
   not "remove parameters". The 13.6% footprint gap between the two treatments. Model
   currency (§3). None of these is fatal for a 5-page workshop paper; all of them look
   fatal if a referee finds them before you say them.

---

## 14. State of the experiment (as of 2026-08-01)

**Phases Q complete on model 1.** Qwen2.5-1.5B-Instruct, n=750, seed 7, Tesla T4, prompts v5, 4.24 GPU-h. Model 2 is wired but its sweep has not been run.

**Baseline at n=750: EM 30.80% [27.47, 34.13], F1 41.38% [38.2, 44.6].** Parse success per role 97.3 / 99.3 / 94.7 / 99.7%, all clearing §5a's 90% bar.

> **Do not quote 34.7% EM.** That is the *n=300* figure, and it is the only headline
> accuracy number PROGRESS.md ever wrote down. n=750 supersedes n=300 outright (§5),
> and the correct figure is **30.80%** — 3.87 pp lower, sitting on the bottom edge of
> §5a's healthy band rather than mid-band. The gap is sampling noise in the original
> 300, not a regression: the 300 shared questions score 34.33 and the 450 new ones
> 28.44 (permutation p = 0.093), and all five runs move the same direction. Verified
> by independent recomputation 2026-08-01.

Also do not repeat "baseline had 2 failures" without qualifying it: that is the **QA
stage only**. Baseline has **118** parse failures across all stages at n=750.

**Answer EM, paired drop from baseline** (positive = quantizing that role hurt):

    Extractor      +3.20 [+0.53, +5.87]   SIGNIFICANT
    QA             +1.73 [-0.13, +3.60]   marginal
    Step Definer   +0.53 [-1.60, +2.67]   null
    Planner        -1.73 [-4.53, +1.07]   null

> **Reproducibility note for Gate 3.** The Extractor's lower bound reads +0.53 with the
> current `bootstrap_ci` and read +0.67 before it was made order-invariant (§13a item
> 11). Point estimates are bit-identical; only that one bound moved, by less than the
> seed-to-seed Monte-Carlo spread — across bootstrap seeds 0/1/7/42 the bound ranges
> +0.53…+0.67 and never touches zero, so "only the Extractor excludes zero" is robust.
> **Gate 3's line-by-line diff should expect this one difference and no others.**
> All four point estimates and the other seven bounds reproduce exactly.

**Extraction accuracy, ev-F1 drop** (§5c) — resolves two roles, both format-heavy:

    Step Definer   +2.80 [+1.20, +4.42]
    Extractor      +2.18 [+0.04, +4.27]
    Planner        -0.73 [-2.72, +1.27]
    QA             +0.00 [+0.00, +0.00]   negative control PASSES

**Pre-registered predictions 1–4, scored:**

    1 format not damaged       HOLDS on all four roles
    2 selection churn high     HOLDS (extractor spans 75.9%, answer churn 29-44%)
    3 format-heavy > knowledge CONFIRMED on ev-F1 +2.85 [+1.34, +4.37]
                               NOT confirmed on answer EM +1.87 [-0.13, +4.00]
    4 calibration              UNTESTED - log_confidence was off

**Effect sizes halved between n=300 and n=750** (StepDef 4.38 → 2.80, Extractor 4.66 → 2.18) while CIs tightened enough to stay significant. Consistent with the n=300 estimates having been inflated by winner's curse. This is the direct evidence behind §5e: treat any allocation chosen on these numbers as provisional until confirmed out of sample.

**Quantization verifiably applied, but the effect is not uniform across roles.** Share
of the quantized stage's raw outputs differing from baseline, at n=750:

    planner        90.3%        extractor      91.9%
    step_definer   85.0%        qa             39.3%

Earlier write-ups compressed this to "83–91% of each quantized stage's outputs differ."
That is wrong: **QA is 39.3%**, less than half the stated floor. The sentence is
load-bearing for "we verified quantization really applied," so state it per role. QA's
low churn is not evidence that quantization failed to apply — QA emits short answers
from a 48-token budget, so there is far less text to differ.

**Determinism independently confirmed.** In `extractor_4bit` and `qa_4bit`, every
*unquantized upstream* stage is bit-identical to baseline (0.0% divergence over
750/1597 calls). This validates §5c's QA negative control from a second direction and
confirms greedy decoding is deterministic under fixed batch composition.

**Perturbing the Planner changes the shape of the run.** `planner_4bit` produced 1605
step_definer *and* 1605 extractor calls against baseline's 1597 — 165 of 750 questions
drew a different sub-question count. Downstream stages therefore pair with baseline on
**1516 shared call keys, not 1597**. Any per-call analysis must intersect keys, never
assume alignment; any analysis that reports a denominator must report the intersected
one. This is a property of the design, not a defect.

**What is NOT yet known** — and what v2 exists to establish:

- No size ablation has been run. Every "vs size" statement in the project so far is a cross-paper comparison against MA-RAG and is **not admissible** (§1).
- The memory framing was wrong (§5d). No accuracy result depends on it.
- Only one role (Extractor) is resolved on answer EM. A four-way ranking is not yet supported by the data, on either axis.

**Environment notes.**

- Local VRAM readings are untrustworthy: Windows WDDM spills CUDA allocations into system RAM rather than raising OOM, so FP16 stages report ~5900 MB peaks on a 4096 MB card even at batch 1. The OOM autotune path has therefore **never actually fired** and remains unvalidated. Only T4 numbers are real.
- Kaggle results are pushed to GitHub after every run; an earlier n=750 sweep was lost entirely to an idle timeout before that was in place.
- `qa_4bit`'s 0.0% parse-failure rate was flagged by the bug-sniffer and re-checked: genuine (750 statuses, all `ok`, 684 distinct outputs; baseline had 2 failures).

---

**Next action: build step 10 — the §13a repairs. Then step 11, step 12, and stop at Gate 3.**
