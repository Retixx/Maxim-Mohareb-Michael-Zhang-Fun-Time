# Final A100 campaign runbook

This is the operator handoff for `final-3b-reference`. Follow it in order. Do
not reconstruct the experiment from old notebooks, logs, or branches.

## 1. Recognize the correct experiment

Every production accuracy log must begin with all of the following:

```text
n=1500
seed=20260805
batch=32
```

The static campaign has exactly 22 accuracy run IDs. Standardized timing has 17
non-tiny run IDs on 128 excluded questions with two repetitions. A selector may
add one distinct `ma_optimized_exploratory` accuracy/timing pair after the static
campaign. Output showing n=3000, seed=7, batch=64, `uni_3b_*`, or another old run
name is historical and must not be used in the final analysis.

Never pass production overrides such as `--n`, `--seed`, `--batch-size`,
`--model-id`, `--small-model-id`, or `--dev-sample`.

## 2. Synchronize and test the branch

On every machine:

```bash
git fetch origin
git checkout final-3b-reference
git pull --ff-only origin final-3b-reference
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/final-3b-reference)"
test -z "$(git status --porcelain)"
python -X utf8 -m unittest discover -s tests -q
```

The test suite must pass. Do not run production from a dirty checkout. Keep old
results outside `results/`; do not delete them, and do not mix them with final
artifacts. Generated final results are intentionally not committed to Git, so
back them up separately.

## 3. Lock the real A100 environment

Choose the immutable container that every worker will use. On one A100, from a
clean checkout, set its real identity:

```bash
export EXPERIMENT_CONTAINER_REF='REGISTRY/IMAGE:TAG'
export EXPERIMENT_CONTAINER_DIGEST='sha256:IMMUTABLE_DIGEST'
python -m src.runner --write-environment-lock config/environment.lock.json \
  --container-ref "$EXPERIMENT_CONTAINER_REF" \
  --container-digest "$EXPERIMENT_CONTAINER_DIGEST"
git add config/environment.lock.json
git commit -m 'Lock final A100 environment'
git push origin final-3b-reference
```

Every worker must then pull that lock commit, export the same two variables, and
validate:

```bash
git pull --ff-only origin final-3b-reference
export EXPERIMENT_CONTAINER_REF='REGISTRY/IMAGE:TAG'
export EXPERIMENT_CONTAINER_DIGEST='sha256:IMMUTABLE_DIGEST'
python -m src.runner --validate-environment-lock
```

Do not invent a digest or copy a lock from another image. A stale or mismatched
lock must stop the run.

## 4. Run standardized timing first

Reserve one otherwise idle physical A100. Use that same GPU for every timing arm,
including a distinct optimized arm later. Do not run another workload on it.

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py \
  --kind timing --workers 1 --worker-index 0 --execute \
  2>&1 | tee logs/final_timing.log
```

This first performs excluded batch-32 warm-up/preflight and then two timing
repetitions. It never scores final questions. The five 0.5B appendix arms are
intentionally not timed; each one performs its own excluded batch-32 preflight
immediately before its scored calls.

Campaign restarts skip finalized artifacts whose JSONL hash matches metadata.
Timing itself never resumes a partial artifact. If one timing arm was interrupted,
move only that arm's partial `.jsonl` and `.meta.json` together to a backup
directory, then rerun the campaign. Never merge timing sessions.

## 5. Plan and run the 22 static accuracy arms

Choose worker count `W` once. Every worker must use the same `W`, a unique index
from `0` through `W-1`, the same lock, and one assigned A100. First inspect the
deterministic plan:

```bash
python scripts/run_campaign.py --kind accuracy --workers W
```

Then launch one process per worker, changing both the physical device and worker
index as appropriate:

```bash
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU python scripts/run_campaign.py \
  --kind accuracy --workers W --worker-index INDEX --execute \
  2>&1 | tee logs/final_accuracy_worker_INDEX.log
```

One complete run must stay on one physical GPU. If interrupted, rerun the same
worker assignment on the same GPU; canonical-batch resume is allowed. Never
combine one run across GPUs. An OOM is a hard failure: do not lower batch size.

Do not proceed until all workers finish. A `complete:` log line is useful but is
not the validity check; finalized JSONL/meta hashes and strict analysis are.

## 6. Freeze the exploratory allocation

After all 22 accuracy arms and all 17 timing arms exist:

```bash
python analyze.py --config config/experiment.yaml --allow-incomplete
```

This creates:

```text
analysis/ma_optimized_exploratory.selection.json
analysis/ma_optimized_exploratory.experiment.yaml
```

Do not edit either file. Commit and push them before executing the selection:

```bash
git add analysis/ma_optimized_exploratory.selection.json \
        analysis/ma_optimized_exploratory.experiment.yaml
git commit -m 'Freeze exploratory allocation'
git push origin final-3b-reference
```

Inspect `materialized_new_run` and `execution_run_id` in the selection JSON.

- If `materialized_new_run` is `false`, the selector chose an existing static
  run. Do not execute another accuracy or timing arm.
- If it is `true`, pull the selector commit on the chosen workers and run exactly:

```bash
python -m src.runner \
  --config analysis/ma_optimized_exploratory.experiment.yaml \
  --run ma_optimized_exploratory

CUDA_VISIBLE_DEVICES=0 python -m src.runner \
  --config analysis/ma_optimized_exploratory.experiment.yaml \
  --run ma_optimized_exploratory --timing-mode
```

The second command must use the same reserved physical A100 as section 4. The
optimized run is in-sample and exploratory because the same 1,500 questions
informed its selection.

## 7. Run strict final analysis

Always use the frozen derived config for the final report, even when the selector
reused an existing run:

```bash
python analyze.py \
  --config analysis/ma_optimized_exploratory.experiment.yaml
```

Do not use `--allow-incomplete` for this final command. The analyzer must fail if
an accuracy/timing arm, manifest hash, environment lock, git revision, selector
link, GPU identity, or finalized JSONL hash is wrong. It reuses the committed
selector artifact and must not replace the decision that produced the optimized
run.

Preserve and back up at minimum:

```text
config/environment.lock.json
results/*.jsonl
results/*.meta.json
analysis/ma_optimized_exploratory.selection.json
analysis/ma_optimized_exploratory.experiment.yaml
analysis/report.json
analysis/report.md
logs/*.plan.json
logs/*.log
```

`analysis/report.json` is the complete machine-readable result. `report.md` is a
convenience summary; use JSON/run metadata for detailed memory, Pareto, and cold
load tables.

## 8. Hard stop conditions

Stop rather than improvising if any of these occurs:

- the header is not n=1500, seed=20260805, batch=32;
- the working tree is dirty before a production invocation;
- environment-lock validation fails;
- any model/dataset revision cannot be resolved exactly;
- any arm OOMs at batch 32;
- a resume requests a different GPU UUID or canonical batch;
- a timing artifact is partial or spans sessions/GPUs;
- the selector artifacts are edited or uncommitted before the optimized run;
- strict final analysis raises an error; or
- someone proposes replacing a failed question, tuning a treatment-specific
  prompt, retrying parse failures, or using constrained decoding.

Do not solve a stop condition by changing the scientific contract. Preserve all
artifacts and diagnose the failure first.
