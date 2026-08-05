# Role-aware SLM multi-agent QA

Experimental harness for a workshop study of memory-aware capacity allocation
across four SLM agent roles: Planner, Step Definer, Extractor, and QA.

The system answers HotpotQA distractor questions from the ten paragraphs supplied
with each example. It has no retriever, so results concern provided-context,
retrieval-free multi-agent QA rather than retrieval quality.

Read [SPEC.md](SPEC.md) before running anything. It is the authoritative
scientific contract; [config/experiment.yaml](config/experiment.yaml) is its
machine-readable counterpart.

## Final design

- Qwen2.5-3B-Instruct FP16 is the uniform multi-agent reference.
- Primary role comparison: 3B 8-bit versus 1.5B FP16, described as
  **near-memory-matched**.
- 3B 4-bit is a mandatory secondary treatment.
- Five 0.5B FP16 arms establish an appendix-only compliance/capacity floor and
  never enter allocation selection.
- A competitive single-call 3B FP16 arm is compared with uniform multi-agent 3B
  FP16.
- A role-aware allocation selected from non-tiny ablations is run once and
  labelled in-sample/exploratory.
- F1 is primary; Exact Match is co-reported.

All arms use the same frozen 1,500 questions. The manifest excludes the exact
old n=3,000 design-pilot IDs and prompt-development questions:

```text
config/manifests/final_n1500_seed20260805.json
final ID SHA-256: 5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af
```

The static matrix has 22 runs: the previous 21 arms plus `single_fp16`. The
selector can add one distinct `ma_optimized_exploratory` execution, so the
campaign maximum is 23. If it selects an existing uniform/reference arm, no new
execution is added.

## Environment

Production runs target A100 GPUs. Known exercised core versions are Transformers
5.14.1, bitsandbytes 0.50.0, datasets 5.0.1, and the prior A100 pilot's PyTorch
2.13.0+cu130/CUDA 13.0 stack. These do not constitute a complete lock.

Before production, run the A100 preflight and commit its complete environment
artifact: immutable container digest, Python and full package lock, driver/CUDA,
GPU SKU/UUID, model/tokenizer revisions, dataset revision, and repository commit.
Every worker must match it. See SPEC section 11.

The model and dataset revisions are already pinned in `config/experiment.yaml`.
Do not replace them with floating `main` revisions.

## Running

The runner reads the frozen manifest and fails before model loading if its count,
hashes, exclusions, or dataset revision do not match.

```bash
# Smoke/preflight on excluded development data only
python -X utf8 smoke_test.py --n 10 --run baseline

# From a clean commit on the selected A100/container, generate and commit the lock
python -m src.runner --write-environment-lock \
  --container-ref REGISTRY/IMAGE:TAG --container-digest sha256:IMMUTABLE_DIGEST
git add config/environment.lock.json && git commit -m "Lock final A100 environment"
python -m src.runner --validate-environment-lock

# First, certify every non-tiny configuration on excluded data and collect the
# two-repeat reserved-A100 timing benchmark. No final question is touched.
CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py --kind timing --execute

# Deterministically inspect the exact 22-arm assignment for any GPU count
python scripts/run_campaign.py --kind accuracy --workers 4

# Launch one whole-arm worker per A100 (set a different index/device per process)
CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py \
  --kind accuracy --workers 4 --worker-index 0 --execute

# One production arm (after the launch gate passes)
python -m src.runner --config config/experiment.yaml --run baseline

# Examples of the primary near-match arms
python -m src.runner --config config/experiment.yaml --run extractor_8bit
python -m src.runner --config config/experiment.yaml --run extractor_small

# Direct single-call architecture control
python -m src.runner --config config/experiment.yaml --run single_fp16

# Static paired analysis freezes the selector artifacts. It is explicitly
# intermediate because a distinct selected run may still be pending.
python analyze.py --config config/experiment.yaml --allow-incomplete

# If the selector trace says materialized_new_run=true, commit both frozen
# artifacts, then execute the distinct exploratory system once:
python -m src.runner --config analysis/ma_optimized_exploratory.experiment.yaml \
  --run ma_optimized_exploratory

# If distinct, time the exploratory system on that same reserved A100 as well:
python -m src.runner --config analysis/ma_optimized_exploratory.experiment.yaml \
  --run ma_optimized_exploratory --timing-mode

# Re-run analysis with the frozen config to add the actual exploratory comparison.
python analyze.py --config analysis/ma_optimized_exploratory.experiment.yaml
```

Batch size is fixed at 32. An OOM must fail the run; no production arm may
autotune to a different batch size. Resume is allowed only with the same manifest,
model/prompt fingerprints, and canonical batch membership. Never merge one run
from multiple GPUs or environments.

## Timing and memory

The paper's primary memory number is deduplicated concurrent model-footprint MiB:
each exact configuration's measured parameters and buffers are charged once.
Sequential peak VRAM and one-server-per-role totals are co-reported separately.

One uncontended A100 benchmarks the non-tiny configurations twice on a frozen
128-question excluded cohort at batch 32. These calls never enter accuracy or
selection. Uniform four-role 3B FP16 is `1.00x`; other results are steady-state
inverse-throughput ratios. Cold model loading is reported separately. These are
A100 ratios, not edge-latency estimates.

## Repository layout

| Path | Purpose |
|---|---|
| `SPEC.md` | Locked experiment and claim contract |
| `PROGRESS.md` | Current handoff and launch blockers |
| `config/experiment.yaml` | Models, revisions, matrix, metrics, selector, timing/memory policy |
| `config/manifests/` | Frozen final question and exclusion IDs |
| `ENVIRONMENT.md` | Production container/package/GPU lock contract |
| `requirements-core.txt` | Recorded core inference versions (not a complete lock) |
| `scripts/freeze_final_sample.py` | Reproduce/audit the committed sample manifest |
| `src/` | Prompts, parsing, inference, pipeline, runner, and metrics |
| `smoke_test.py` | Preflight plumbing checks on excluded development data |
| `analyze.py` | Paired final analysis and allocation selection |
| `results/` | Generated outputs; empty in the source branch except `.gitkeep` |

Historical notebooks, interim results, and the obsolete Gate 2 report were
removed from the active tree. They remain recoverable from Git history.

## Controls that must not change

- no parse retries or regeneration;
- no constrained/grammar decoding;
- no model-, precision-, or size-specific prompt tuning;
- no question replacement or resampling;
- no variable production batch size;
- no floating model or dataset revisions; and
- no confirmatory claim from the post-selected optimized arm.
