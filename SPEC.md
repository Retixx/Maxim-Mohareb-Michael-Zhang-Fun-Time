# Experiment contract: role-aware capacity allocation in SLM multi-agent QA

This file is the authoritative scientific contract for the final A100 campaign.
`config/experiment.yaml` is its machine-readable counterpart. If code, analysis,
or prose conflicts with either file, stop before running and resolve the conflict.

## 1. Research question and scope

At small-language-model scale, can a four-role, provided-context multi-agent QA
pipeline allocate model capacity across roles more efficiently than a uniform
pipeline while retaining answer quality?

The four roles are Planner, Step Definer, Extractor, and QA. HotpotQA supplies ten
paragraphs per question (two gold and eight distractors). There is no learned or
external retriever in this experiment. The paper must describe the system as
**provided-context, retrieval-free multi-agent QA** or **MA-RAG-style QA**, not as
a test of retrieval quality.

The final campaign answers four prespecified questions:

1. Which roles are most sensitive to 3B 8-bit quantization?
2. Which roles are most sensitive to swapping 3B FP16 for its 1.5B FP16 sibling?
3. At a near memory match, do the quantized and smaller-model interventions have
   different role costs?
4. How do uniform multi-agent 3B FP16 and a competitive single-call 3B FP16
   system compare?

An allocation derived from the same results is run once as an explicitly
**exploratory** deployment demonstration. It cannot support a confirmatory claim
that role-aware allocation is superior.

## 2. Frozen data and seed hygiene

There is one final evaluation sample, used once:

- dataset: `hotpotqa/hotpot_qa`, `distractor`, `validation`
- dataset revision: `1908d6afbbead072334abe2965f91bd2709910ab`
- n: 1,500
- sampling seed: 20,260,805
- manifest: `config/manifests/final_n1500_seed20260805.json`
- manifest-file SHA-256: `841dbca9ac7e76c0277a5696fba9f7e254b973afb0b670efbc5edfc006b4af46`
- ordered final-ID SHA-256: `5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af`
- sorted exclusion-ID SHA-256: `a5cfacb84fa9a48217f3206a095706a6d48802bd244151c72f2eef08372c00a8`

Every arm receives exactly the same 1,500 question IDs in exactly the same order
and canonical batch membership. This shared set is desirable: all arm contrasts
are paired at the question level. There is no second selection or confirmation
set in the final campaign.

A new seed alone would not make the sample independent of earlier work. The
manifest therefore excludes:

- the exact 3,000 unique IDs in the old seed-7 design-pilot result blob;
- the exact 30 seed-1234 prompt-development IDs in the committed result blob;
- the ten seed-0 prompt-development IDs reconstructed from the pinned dataset
  row order and the original sampling implementation because no unambiguously
  labelled seed-0 result blob survives.

The manifest commits all 3,031 unique excluded IDs, not only their hash, so a
shallow or offline checkout can prove disjointness. It records the source commit,
path, and blob hash for result-derived exclusions. `scripts/freeze_final_sample.py`
is the provenance generator; it is not called during a run.

Before loading a model, the runner must fail unless it verifies:

1. manifest dataset identity and revision match the config;
2. exactly 1,500 final IDs are present and unique;
3. final and exclusion hashes match, using UTF-8 `id + "\n"` records including
   the final trailing newline;
4. no final ID occurs in the exclusion set;
5. all IDs exist in the pinned dataset revision; and
6. loaded question order exactly matches manifest order.

Do not resample missing or failed questions. A failed question remains in the
denominator.

## 3a. Retrieval (amendment — supersedes "no retriever")

The original §3 used the distractor setting and handed every stage ten
paragraphs already containing both gold. That deleted the retriever, left the
Step Definer's `search_terms` field consumed by nothing, and gave decomposition
no mechanism through which to help. Measured consequence: `single_fp16` beat the
four-agent pipeline by **+9.23 EM [+7.47, +11.03]** at identical memory, n=3000
paired. That is an artifact of the harness, not a finding about multi-agent RAG.
MA-RAG retrieves over millions of passages; a system with nothing to search is
not the system being studied.

**Corpus.** Every unique paragraph from both HotpotQA configs, pooled:
`fullwiki` supplies a real IR system's top-10 (hard lexical distractors; gold
absent for 39% of questions), `distractor` guarantees gold stays reachable.
Union: **72,094 passages, 100% gold-in-corpus.** The runner asserts this before
loading any model.

`corpus_configs` order is load-bearing — `distractor` must be first. 402 titles
appear in both configs with different sentence splits, and `supporting_facts`
`sent_id` is defined against the distractor context. Reversing the order shifts
gold sentence indices and corrupts every extraction-accuracy number silently.

**Two hops.** Retrieval is `k=10`, split `hop1=7` / `hop2=3`:

1. hop 1 queries the **original question** (not the sub-questions — per-sub-question
   queries retrieve strictly worse, recall@10 0.743 vs 0.794, because HotpotQA
   carries its lexical signal in entity names and splitting discards it);
2. the Extractor reads those 7;
3. a follow-up query is built from capitalised name phrases in the Extractor's
   spans, dropping any name that is already in the question or already the title
   of a hop-1 passage;
4. the Extractor runs a second pass over the 3 passages that query returns.

If no candidate name survives step 3, no second query is issued and the held-back
budget is spent on hop-1 depth instead, so a question needing only one hop is not
penalised for the machinery.

Step 3 is a regex, deliberately **not** a model call: a quantized Extractor must
be able to damage retrieval only through which sentences it selects, never
through a second learned component that also degrades. Cost of that choice,
versus splicing the gold bridge title directly: 0.794 vs 0.812 hidden-title
recall.

**Prespecified stratifier.** Each question is labelled from its text and gold
titles alone, before any model runs:

- `hidden_bridge` (1204/1500, 80%) — at least one gold page is never named in the
  question, so no single query can reach it;
- `fully_named` (296/1500, 20%) — every gold page is named.

Every metric is reported by stratum. `fully_named` is the control: two hops
cannot help where nothing is hidden, and it must show no gain. Measured at k=10,
n=1500, equal read budget:

| stratum | all-gold-retrieved | hidden-title recall |
|---|---|---|
| hidden_bridge | 0.520 → 0.678 | 0.653 → 0.794 |
| fully_named | 0.892 → 0.797 | (nothing hidden) |

`k` and `hop1` were swept on CPU before any GPU time — k ∈ {5,8,10,12,16,20},
hop1 ∈ {5,6,7,8}. Gain plateaus above k=8; 7/3 keeps the full hidden-bridge gain
at roughly half the fully-named cost of a 5/5 split (−0.095 vs −0.172).

**Cost.** The Extractor now runs twice, so pipeline arms make 5 generation
stages instead of 4. `extractor_hop2` is the same agent at the same precision and
is **not independently configurable**; the runner rejects any run that tries.

## 3. Models and treatments

All model and tokenizer artifacts resolve at immutable Hugging Face revisions:

| Alias | Model | Revision |
|---|---|---|
| base | Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| small | Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| tiny | Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |

The reference system uses the 3B model at FP16 for all four roles.

The primary head-to-head compares these treated-role configurations:

- **Q:** 3B at bitsandbytes 8-bit;
- **S:** 1.5B at FP16.

The 3B 4-bit tier is mandatory but secondary. Bitsandbytes 8-bit uses LLM.int8
outlier routing, while 4-bit uses NF4 with double quantization. These are named
treatment configurations, not a clean numerical bit-width dose response.

Swapping 3B for 1.5B or 0.5B changes depth, width, and training mix as well as
parameter count. Call this a **smaller-sibling model swap**, not pure parameter
removal.

The five 0.5B FP16 runs measure the lower compliance/capacity floor. They are
appendix-only and excluded from the allocation selector, the primary hypothesis
family, and main-paper allocation claims. They do not measure probability
calibration.

## 4. Run matrix

The static config contains 22 runs:

| Family | Run IDs | Count | Inferential role |
|---|---|---:|---|
| Reference | `baseline` | 1 | Uniform four-role 3B FP16 baseline |
| 3B 8-bit role ablations | `planner_8bit`, `stepdef_8bit`, `extractor_8bit`, `qa_8bit` | 4 | Primary Q side |
| 3B 4-bit role ablations | `planner_4bit`, `stepdef_4bit`, `extractor_4bit`, `qa_4bit` | 4 | Secondary aggressive quantization |
| 1.5B FP16 role ablations | `planner_small`, `stepdef_small`, `extractor_small`, `qa_small` | 4 | Primary S side |
| Uniform frontier | `ma_uniform_8bit`, `ma_uniform_4bit`, `ma_uniform_small` | 3 | Prespecified deployment controls |
| 0.5B FP16 floor | four `*_tiny` role runs plus `ma_uniform_tiny` | 5 | Appendix/lower-limit evidence only |
| Architecture control | `single_fp16` | 1 | Single-call versus multi-agent 3B FP16 |

The post-ablation selector can materialize at most one additional run,
`ma_optimized_exploratory`. Therefore:

- 22 static run IDs are always configured;
- at most 23 distinct executions occur;
- if the selector returns `baseline` or an existing uniform configuration, that
  result is reused and the campaign remains at 22 executions.

No run may silently substitute a different model, precision, seed, sample,
prompt version, batch size, or model revision.

## 5. Single-call control

`single_fp16` is a competitive, frozen baseline rather than a strawman:

- one 3B FP16 model call per question;
- the same question, and the same `k` passages the multi-agent system reads in
  total, retrieved by a single BM25 query over the same corpus (§3a);
- a dedicated direct-answer prompt developed only on excluded development IDs;
- greedy decoding and the same 48-token answer budget as the QA role;
- the same model/tokenizer revision and answer parser/normalization;
- no hidden decomposition, retrieval, or extra context.

Report total prompt tokens, output tokens, model calls, F1, EM, memory, and timing
for both systems. Multi-agent context is read multiple times, so call and token
counts are required to interpret efficiency.

The `single_fp16` versus `baseline` comparison is prespecified and may use the
same final 1,500 questions without post-selection bias.

## 6. Primary and secondary statistics

**Primary outcome:** HotpotQA answer token F1 under the official normalization.

**Co-reported secondary outcome:** Exact Match. EM appears beside F1 in every
accuracy table but does not replace the primary endpoint.

For each role `r`, define question-level losses against the shared reference:

```text
Q_loss_r = F1_baseline - F1_role_8bit
S_loss_r = F1_baseline - F1_role_small
axis_contrast_r = S_loss_r - Q_loss_r
                = F1_role_8bit - F1_role_small
```

The four role-specific 8-bit-versus-1.5B axis contrasts form the primary
hypothesis family. Use 10,000 paired question-level bootstrap replicates and
Holm-adjust the four primary tests. Report unadjusted point estimates, 95%
paired CIs, adjusted p-values, and the direction of the contrast.

Do not decide significance by whether two separate confidence intervals overlap.
Compute the paired interval on the difference itself.

The 3B 4-bit family, extraction evidence, parse/mechanism diagnostics, and 0.5B
floor are declared secondary or exploratory. Keep those families visibly
separate; do not move a favorable secondary result into the primary family.

Role ranking with only four roles is unstable. Report bootstrap rank
probabilities and pairwise role-cost contrasts. Do not publish a definitive total
ordering when uncertainty overlaps.

Planner changes may alter the number of downstream calls. Any per-call metric for
Step Definer or Extractor must be aggregated or cluster-bootstrapped by question,
never treated as independent calls. Timing intervals resample complete batches.

All calculations start from per-question records. Sort by question ID before
seeded resampling so results cannot depend on file-write order.

## 7. Memory accounting

Memory is central, but it is not a hard equality constraint. A configuration that
maintains accuracy while reducing memory is a successful optimization. Increased
memory must be charged and justified.

### 7.1 Near-memory-matched Q versus S

Measured treated-model footprints are:

| Configuration | Resident weight footprint |
|---|---:|
| 3B FP16 | 5,886.0 MiB |
| 3B 8-bit | 3,240.0 MiB |
| 1.5B FP16 | 2,944.4 MiB |
| 3B 4-bit | 1,917.0 MiB |
| 0.5B FP16 | 942.3 MiB |

The primary treated-role match, 3B 8-bit versus 1.5B FP16, differs by 295.6 MiB:
the 8-bit treatment holds 10.04% more than the 1.5B treatment. In a one-role
ablation system with the shared 3B FP16 base also resident, totals are 9,126.0
versus 8,830.4 MiB, a 3.35% difference. Describe this as **near-memory-matched**.

The 4-bit and 0.5B arms remain informative efficiency/floor points even though
they are not memory-matched to the primary comparison. Never imply equality.

### 7.2 Primary topology

The primary memory quantity is **deduplicated concurrent model-footprint MiB**:
sum the measured parameters-plus-buffers footprint of each distinct model
configuration once, even if several roles share it. A configuration fingerprint includes model
ID, immutable revision, precision, quantization method/settings, and compute dtype.

This is a fixed-charge allocation problem. Introducing the first role that uses a
new configuration charges its entire resident footprint; additional roles sharing
that exact configuration do not charge it again. Never sum independent per-role
savings to estimate a mixed system.

Co-report, under explicit labels:

1. isolated role-service footprint, summing one copy per role;
2. sequential execution peak VRAM, matching the stage-major runner;
3. parameters, buffers/quantization state, and full model footprint separately;
4. CUDA allocated/reserved peak including activations; and
5. cold loading/swapping time.

Use MiB (`bytes / 1024^2`) consistently. Do not label these figures MB.

## 8. Exploratory allocation selector

The selector is separate from the near-memory-match test. After all non-tiny role
ablations finish, it enumerates every four-role allocation over:

```text
{3B FP16, 3B 8-bit, 3B 4-bit, 1.5B FP16}
```

There are `4^4 = 256` candidates. The 0.5B configuration is prohibited.

For each candidate, the selector estimates system F1 from the observed per-role
costs and charges every distinct configuration once using the primary memory
topology. A candidate is feasible only when the paired-bootstrap 95% lower bound
for its predicted F1 difference from uniform 3B FP16 is at least -1.0 point. It
selects the lowest-memory feasible allocation. Ties resolve by:

1. higher predicted F1;
2. fewer distinct configurations;
3. fixed lexical allocation ID.

The selector may return an existing uniform/reference run. Otherwise `analyze.py`
writes `analysis/ma_optimized_exploratory.selection.json` and
`analysis/ma_optimized_exploratory.experiment.yaml`, including input-result
hashes, the chosen role mapping, predicted F1 and its bootstrap interval, memory
accounting, selection rule/version, and tie-break trace. Commit both artifacts
before executing the selected system. Do not hand-edit them after observing
final answers.

Because selection and evaluation use the same 1,500 questions, the selected
system's measured result is explicitly in-sample and exploratory. Report it as a
deployment demonstration and interaction check, not confirmatory superiority or
a fresh validation. The actual mixed run is required because per-role effects may
interact non-additively.

## 9. Generation, parsing, and batching controls

- Greedy generation only; no sampling.
- Prompt templates and parser conventions are frozen before the final manifest is
  scored and shared across treatment configurations.
- No precision- or size-specific prompt tuning.
- No parse retries or regeneration.
- No grammar-constrained decoding or prefix-token constraints.
- Tolerant parsing/salvage may recover usable content for downstream propagation,
  but the original call keeps its original parse-failure status.
- QA must consume validated or salvaged Extractor spans consistently; recovered
  evidence must not be logged and then discarded.
- Maximum downstream fan-out remains frozen and identical across arms.

Batch size is pinned at 32 for every production arm. `batch_size` and
`min_batch_size` are both 32. Preflight every distinct model configuration and
stage shape at batch 32. A production OOM must fail loudly; it must never reduce
batch size for one arm.

Canonical batch IDs, ordered membership, and batch size are committed in run
metadata. Resume may skip completed canonical batches but may not repack remaining
questions, because greedy outputs can change with batch neighbors. Question IDs,
stage, call index, model fingerprint, prompt hash, manifest hash, and batch ID are
part of resume compatibility.

## 10. A100 execution and timing

Multiple A100s may execute accuracy arms concurrently, but one complete run stays
on one GPU. A run may never be assembled from records produced on different GPUs
or software environments.

Reserve one otherwise idle A100 for standardized timing. Every timing configuration
runs there under the same container and settings:

- batch size 32;
- fixed canonical batches;
- warm-up before measurement;
- CUDA synchronization around timed regions;
- randomized configuration order;
- repeated timing subset;
- no concurrent GPU workload.

Timing uses two complete repetitions of the frozen 128-question excluded cohort
in `config/manifests/timing_excluded128_seed20260805.json`. It is disjoint from
both the final 1,500 and the 32 warm-up questions. Timing outputs never enter F1,
EM, selection, or any other accuracy estimate. The five 0.5B floor arms are not
timed; the post-selected allocation is timed only if it is a distinct execution.

The primary timing measure is steady-state end-to-end inverse throughput
(seconds/question) relative to uniform four-role 3B FP16 `baseline = 1.00x`.
Report per-stage ratios against the same role at 3B FP16, raw batch wall times,
questions/second, and token-normalized throughput as secondary diagnostics.

Exclude model download/load from the primary steady-state ratio because the
primary deployment topology assumes resident weights. Record cold model loading,
cache reuse, and swapping separately. The overall campaign wall clock across
several GPUs is not system latency.

Record GPU UUID, exact A100 SKU/memory, driver, CUDA, clock/power state when
available, container digest, and software lock. If accuracy runs span different
A100 SKUs or stacks, run the same fixed calibration batch on each device and
compare raw outputs. Block or randomize by device if they differ.

Call timing results **relative inverse throughput on A100**. They are not estimates
of edge-device latency. Hardware transfer is a stated hypothesis unless a real
edge platform is evaluated.

## 11. Reproducibility and environment

The final environment must be immutable across arms. Known, already exercised
core versions are:

- PyTorch `2.13.0+cu130` / CUDA `13.0` on the prior A100 pilot;
- Transformers `5.14.1`;
- bitsandbytes `0.50.0`;
- datasets `5.0.1`.

These historical versions are not a complete container lock. Before production,
the chosen A100 environment must produce and commit an environment artifact with:

- container image name and immutable digest;
- Python version and platform;
- complete `pip freeze` with hashes or equivalent environment lock;
- PyTorch/CUDA/cuDNN/driver versions;
- Transformers, bitsandbytes, datasets, Accelerate, and PyYAML versions;
- model/tokenizer and dataset resolved revisions;
- GPU SKU, UUID, and memory;
- repository commit and dirty-worktree state.

The preflight artifact becomes the campaign lock. A mismatched worker must refuse
production work. Do not invent unobserved package or container versions in this
repository merely to make the lock look complete.

## 12. Required run records and analysis outputs

Every call record includes at least run/question/stage/call/batch IDs, model
fingerprint, precision/quantization config, frozen prompt-template version/hash,
the reconstructible input fields, exact rendered-prompt SHA-256, raw output,
parsed and salvaged payloads, parse status, input/output token counts, and timing.
The full rendered prompt need not be duplicated in every JSONL record.

Every answer record includes the final prediction, gold answer, F1, EM, question
metadata, and manifest hash. Run metadata includes all configuration and environment
fields described above, per-stage and system memory, batch topology, cold and
steady-state times, completion state, and input/output artifact hashes.

Required final outputs are:

1. per-run F1 and EM with question-paired uncertainty;
2. the four-role 8-bit-versus-1.5B near-match table with memory gaps;
3. the separate 4-bit table;
4. the appendix 0.5B compliance/capacity-floor table;
5. single-call versus uniform multi-agent 3B FP16;
6. accuracy versus deduplicated concurrent resident memory;
7. A100 relative inverse-throughput and cold-load tables;
8. parse, evidence, churn, and call-count diagnostics clustered by question;
9. the selector trace and exploratory actual mixed-system result, if distinct.

## 13. Claim boundaries

The paper may claim paired role sensitivity and system differences supported by
the prespecified F1 analysis. It may describe the 8-bit/1.5B comparison as a
near-memory-match and report whether accuracy holds as memory decreases.

The paper must not claim:

- a retrieval-quality result;
- exact Q-vs-S memory equality;
- pure parameter-count or pure bit-width causality;
- confirmatory superiority of the in-sample optimized allocation;
- edge-device latency from A100 ratios;
- probability-calibration findings (`log_confidence` is disabled); or
- independent evidence from the old pilot.

The old n=3,000 run is a disclosed design pilot that informed the final design.
Its exact questions are excluded and its outputs do not enter final estimates.

## 14. Launch gate

Do not start the 1,500-question campaign until all of the following pass:

- frozen manifest and exclusion validation;
- immutable model/tokenizer/dataset revision resolution;
- environment/container lock on every worker;
- batch-32 preflight for FP16, 8-bit, 4-bit, small, tiny, and solo paths;
- salvage-to-QA and answer-scoring tests;
- canonical-batch resume test;
- complete metadata/memory/timing smoke test;
- static matrix count of exactly 22 and selector candidate count of 256;
- single-call fairness check; and
- analysis dry-run on synthetic or excluded development data.

Any silent fallback in sample, batch size, model revision, precision, prompt,
parser, or output path invalidates the affected comparison. Fail closed.
