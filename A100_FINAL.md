# Final 2–6 A100 launch guide

Use only `python -m scripts.a100_final` for production GPU phases. This
hardening changes no prompts, questions, models, treatments, batch size,
metrics, or pilot thresholds.

## Setup

The experiment always uses six logical accuracy shards. Physical GPU count
changes concurrency only. Use homogeneous full, non-MIG A100s with identical
memory capacity, driver/CUDA stack, package freeze, and container digest.

For multi-node execution, mount the same shared POSIX filesystem at `results/`
in every checkout. Preserve the tracked one-byte `results/.gitkeep` on that
mount so Git remains clean. If shared results are unavailable, use one
multi-GPU node; copy-after-run collection cannot prevent duplicate shards.

Before network isolation, on every node:

```bash
git fetch origin
git checkout no-bs
git merge --ff-only origin/no-bs
git merge-base --is-ancestor d57b7c9 HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/no-bs)"
test -z "$(git status --porcelain)"

export EXPERIMENT_CONTAINER_REF='REGISTRY/IMAGE:TAG'
export EXPERIMENT_CONTAINER_DIGEST='sha256:IMMUTABLE_DIGEST'

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python scripts/prefetch_assets.py
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/prefetch_assets.py --offline-verify-only
```

Check free space on `results/` and both Hugging Face cache mounts. Every launch
must expose exactly one physical GPU in `CUDA_VISIBLE_DEVICES`.

## Ordered run

Create the environment lock on one selected A100, then commit/push it to
`no-bs` and pull that commit everywhere:

```bash
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU python -m scripts.a100_final prepare \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST"
```

Run the excluded pilot:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU python -m scripts.a100_final pilot \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST" \
  2>&1 | tee logs/final_pilot.log
```

A nonzero exit alone does not distinguish outcomes. A freshly recomputed,
hash-valid `analysis/pilot_gate.json` with `status: STOP`, produced after both
pilot artifacts validate, is scientific STOP. Missing/stale decision output,
OOM, exception, or cache/GPU/artifact/lock failure is an infrastructure abort.

On GO, commit/push `analysis/pilot_gate.json` to `no-bs`, pull that exact commit
on all workers, and do not advance `no-bs` until all six static shards finish.

Complete timing first on one reserved, idle physical A100. Keep the same GPU
UUID and shared results for the whole timing campaign:

```bash
CUDA_VISIBLE_DEVICES=TIMING_PHYSICAL_GPU python -m scripts.a100_final timing \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST" \
  2>&1 | tee logs/final_timing.log
```

Timing artifacts are hash-revalidated against the device ledger before
accuracy. All timing plans share one global active lock, so base and selected
timing cannot overlap.

Then launch indices `0` through `5` exactly once:

```bash
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU python -m scripts.a100_final accuracy \
  --worker-index INDEX \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST" \
  2>&1 | tee logs/final_accuracy_worker_INDEX.log
```

| A100s | Index waves |
|---:|---|
| 2 | A: `0→2→4`; B: `1→3→5` |
| 3 | A: `0→3`; B: `1→4`; C: `2→5` |
| 4 | Start `0,1,2,3`; then `4,5` on the next free GPUs |
| 5 | Start `0,1,2,3,4`; then `5` on the first free GPU |
| 6 | Start `0,1,2,3,4,5`, one per GPU |

The frozen manifest hash is
`511e2b9f974400daffe58f29f8905e4dc4b692cb86516903ab982e4eab53b456`.
Shared persistent claims reject duplicate shards, all static artifacts are
pinned to one GO commit, and each arm lock remains held through final metadata.

## Recovery

There is deliberately no automatic stale-lock takeover. After a hard kill:

1. Prove the old PID/job is dead.
2. Validate the arm's JSONL hash and `metadata_complete` metadata. If finalized,
   preserve the pair and quarantine only its stale `.lock`; if incomplete,
   quarantine the pair and lock together.
3. Manually move the exact stale `.active` directory under
   `results/.fleet_state/` to a quarantine name.
4. Rerun the normal command without `--recover-stale-claim`.

For timing, also move a stale `timing_global.active` only after proving no
timing process exists; never remove `timing_device.json`. The same rules cover
a killed pilot. A partial accuracy arm may resume only on its original GPU UUID;
otherwise quarantine it and restart that arm on a homogeneous replacement.

## Post-selection

After static analysis creates the derived config, commit/push both selection
artifacts to `no-bs`. On the original timing GPU run:

```bash
CUDA_VISIBLE_DEVICES=TIMING_PHYSICAL_GPU python -m scripts.a100_final timing \
  --config analysis/ma_optimized_exploratory.experiment.yaml \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST"
```

Then run or validate selected accuracy:

```bash
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU python -m scripts.a100_final selected-accuracy \
  --config analysis/ma_optimized_exploratory.experiment.yaml \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST"
```

If selection reuses a finalized static arm it is verified and skipped. Keep all
pushes on `no-bs`. Never change the seed, six assignments, prompts, manifests,
models, batch 32, matrix, or gate to force GO.
