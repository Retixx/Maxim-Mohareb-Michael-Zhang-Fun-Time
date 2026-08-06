# Final MA-RAG SLM campaign runbook

This is the normative operator handoff. Follow it in order. SPEC.md defines the
scientific contract; config/experiment.yaml defines the executable contract.
Stop if either disagrees with the current code or generated metadata.

## 1. Recognize the experiment

Every final accuracy invocation must resolve to:

    n=1500
    seed=20260805
    batch=32
    corpus passages=72094
    retrieval k=10 per question-answering plan step
    max plan steps=5
    conceptual roles=planner, step_definer, extractor, qa
    finalizer=step_definer_plan_summary

The static matrix has exactly 22 accuracy run IDs. Logical plan depth is
question-dependent. The value five is an edge-resource ceiling, not an expected
depth and not a MA-RAG framework limit.

Reject output from old seeds, old sample sizes, old batch sizes, old run names,
or any artifact whose architecture/retrieval fingerprint predates the repeated
Step Definer routing loop. Do not pass production overrides for n, seed, batch
size, models, revisions, corpus, k, or plan ceiling.

## 2. Prepare a clean worker

On every worker:

    git fetch origin
    git checkout final-3b-reference
    git pull --ff-only origin final-3b-reference
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/final-3b-reference)"
    test -z "$(git status --porcelain)"

Install the pinned project requirements inside the selected immutable container,
then run:

    python -m pip install -r requirements-core.txt
    python -X utf8 -m pytest -q

The suite must pass. Production must start from a clean committed tree. Preserve
older result artifacts outside results/; never mix them with this campaign.

Before GPU work, make both pinned HotpotQA configurations and all pinned model
revisions available in the worker cache. On a network-enabled preparation node,
then in the immutable offline A100 container, run:

    python scripts/prefetch_assets.py
    python scripts/prefetch_assets.py --offline-verify-only

The equivalent all-in-one A100 preparation command is:

    python scripts/a100_entrypoint.py prepare \
      --container-ref 'REGISTRY/IMAGE:TAG' \
      --container-digest 'sha256:IMMUTABLE_DIGEST'

The runner rebuilds and validates the corpus fingerprint before its first model
load. A cache hit is not proof of correctness; the runner's passage count,
content hash, unique-title, query-policy, and gold-title reachability checks are
authoritative.

## 3. Create and validate the environment lock

Choose one immutable container image for all workers. On the selected A100, from
a clean checkout:

    export EXPERIMENT_CONTAINER_REF='REGISTRY/IMAGE:TAG'
    export EXPERIMENT_CONTAINER_DIGEST='sha256:IMMUTABLE_DIGEST'

    python -m src.runner \
      --write-environment-lock config/environment.lock.json \
      --container-ref "$EXPERIMENT_CONTAINER_REF" \
      --container-digest "$EXPERIMENT_CONTAINER_DIGEST"

    git add config/environment.lock.json
    git commit -m 'Lock final MA-RAG environment'
    git push origin final-3b-reference

Every worker then pulls that exact commit, exports the same container identity,
and validates:

    git pull --ff-only origin final-3b-reference
    export EXPERIMENT_CONTAINER_REF='REGISTRY/IMAGE:TAG'
    export EXPERIMENT_CONTAINER_DIGEST='sha256:IMMUTABLE_DIGEST'
    python -m src.runner --validate-environment-lock

Never invent a digest or copy a lock from another image. A mismatch is a hard
stop.

## 4. Inspect plans without executing

The implemented pilot gate and campaign planner can be inspected safely before
pilot GO with these plan-only commands:

    python scripts/run_campaign.py --kind pilot --workers 1
    python scripts/run_campaign.py --kind timing --workers 1
    python scripts/run_campaign.py --kind accuracy --workers 4

Verify:

- pilot assignment is baseline followed by single_fp16 on one worker;
- timing uses one worker and the configured non-tiny timing matrix;
- accuracy assigns every static run exactly once;
- no plan references final IDs for pilot or timing; and
- all plan files bind config and experiment fingerprints.

Do not use --execute yet.

If either command is unavailable, STOP. The pilot implementation is a launch
prerequisite, not an operator step to reconstruct manually.

## 5. Run the excluded pilot

The pilot is a quality gate, not final evidence. It uses
config/manifests/pilot_excluded200_seed20260806.json: a frozen 200-question,
seed-20260806 cohort drawn from exclusions and disjoint from final, timing, and
warm-up cohorts.

On one locked worker:

    mkdir -p logs
    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py \
      --kind pilot --workers 1 --worker-index 0 --execute \
      2>&1 | tee logs/final_pilot.log

Then generate and validate the certificate:

    python scripts/check_pilot.py

The checker must validate both finalized pilot JSONLs, their metadata and hashes,
the cohort, environment, prompt bundle, architecture, retrieval fingerprint,
and run order. It must report paired F1 and EM, overall and stratum-specific
counts, parse/protocol success, retrieval recall, query/passages exposure, plan
depth, and stop reasons.

GO requires:

    baseline F1 >= single_fp16 F1 overall
    baseline F1 >= single_fp16 F1 on hidden_bridge

If the checker exits non-zero or reports STOP, stop the campaign. Preserve all
pilot artifacts and diagnose the architecture on excluded data. Do not relax
the rule and do not consume final question IDs.

GO is written to analysis/pilot_gate.json. Timing and accuracy execute modes
must independently reject a missing, stale, or mismatched certificate.

The GO certificate is a committed campaign input, not a local handoff. On the
pilot worker, freeze and distribute the exact bytes produced by the checker:

    git add analysis/pilot_gate.json
    git commit -m 'Freeze excluded-pilot GO certificate'
    git push origin final-3b-reference

Staging the certificate is insufficient: production verification compares it
to HEAD, so an uncommitted or subsequently edited gate fails closed. Do not
commit a STOP certificate as authority to continue.

Before timing or accuracy starts, every worker must pull the certificate commit
and verify both the clean checkout and the certificate locally:

    git pull --ff-only origin final-3b-reference
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/final-3b-reference)"
    test -z "$(git status --porcelain)"
    python scripts/check_pilot.py --verify
    python -m src.runner --validate-environment-lock

The gate commit is the only intended post-environment-lock source-control step
before the static campaign. Its content-addressed certificate is excluded from
the environment source bundle, while the verifier binds it to the locked
environment, active config, pilot cohort, experiment fingerprint, and pilot
artifact hashes.

## 6. Run the A100 timing proxy

Only after pilot GO, reserve one otherwise idle physical A100. Use that same GPU
for every timing arm and for a distinct selected allocation later. Do not share
the device with another workload.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py \
      --kind timing --workers 1 --worker-index 0 --execute \
      2>&1 | tee logs/final_timing.log

Timing uses two complete repetitions of the frozen 128-question excluded cohort
after excluded warm-up/preflight. It does not create accuracy records. Every
artifact from an earlier architecture fingerprint is invalid and the complete
matrix must be rerun.

The timing benchmark is a relative A100 systems proxy. Report steady-state
inverse throughput against baseline=1.00x, raw batch wall time,
questions/second, token-normalized throughput, and cold model loading
separately. Never relabel these measurements as phone, embedded-GPU, CPU, NPU,
energy, thermal, or literal edge-device timing.

Timing must be a fresh uncontended session. If an arm is interrupted, move that
arm's partial JSONL and metadata together to a backup location, then replay the
whole arm. Do not merge timing sessions.

## 7. Run the 22 static accuracy arms

Choose worker count W once. Every worker uses the same W, a unique index in
[0, W), the same environment lock, and one assigned A100.

Inspect the deterministic plan:

    python scripts/run_campaign.py --kind accuracy --workers W

Launch one process per worker, changing both device and index:

    CUDA_VISIBLE_DEVICES=PHYSICAL_GPU python scripts/run_campaign.py \
      --kind accuracy --workers W --worker-index INDEX --execute \
      2>&1 | tee logs/final_accuracy_worker_INDEX.log

One complete run stays on one physical GPU. Accuracy resume is allowed only for
matching canonical batches on the same GPU and environment. The runner may skip
certified completed batches but may not repack remaining calls.

The physical runner is stage-major. At each canonical repeated stage it batches
only active calls; question-answering steps can fan out to one Extractor call per
retrieved document, while aggregate steps have no Extractor calls. An inactive
stage should report zero calls and avoid loading a model. Do not mistake varying
stage call counts for missing work.

An OOM at batch 32 is a hard failure. Do not lower the batch size or retry only
the failed generation.

## 8. Freeze the exploratory allocation

After all static accuracy and timing artifacts validate:

    python analyze.py --config config/experiment.yaml

The selector evaluates a declared 625-allocation universe. Tiny eligibility is
decided per conceptual role from the corresponding one-role ablation's
question-clustered strict-protocol lower bound. The F1 noninferiority constraint
and deduplicated resident-memory objective are then applied.

The command creates:

    analysis/ma_optimized_exploratory.selection.json
    analysis/ma_optimized_exploratory.experiment.yaml

Do not edit either file. Commit and push both before executing any new selection:

    git add analysis/ma_optimized_exploratory.selection.json \
            analysis/ma_optimized_exploratory.experiment.yaml
    git commit -m 'Freeze exploratory MA-RAG allocation'
    git push origin final-3b-reference

Inspect materialized_new_run and execution_run_id in the selection trace.

- If materialized_new_run is false and the selected static arm is non-tiny,
  reuse its validated accuracy and timing artifacts.
- If materialized_new_run is false and the selected static arm is tiny, reuse
  its accuracy artifact only. The shared timing matrix excludes tiny arms. The
  committed derived config explicitly authorizes exactly one separately labelled
  post-selection timing run on the reserved A100. Run the derived timing plan;
  its already-complete matrix arms will be skipped and only the selected tiny
  system will execute:

    CUDA_VISIBLE_DEVICES=0 python scripts/run_campaign.py \
      --config analysis/ma_optimized_exploratory.experiment.yaml \
      --kind timing --workers 1 --worker-index 0 --execute

- If materialized_new_run is true, run exactly the derived configuration once:

    python -m src.runner \
      --config analysis/ma_optimized_exploratory.experiment.yaml \
      --run ma_optimized_exploratory

    CUDA_VISIBLE_DEVICES=0 python -m src.runner \
      --config analysis/ma_optimized_exploratory.experiment.yaml \
      --run ma_optimized_exploratory --timing-mode

Use the reserved timing A100 for the second command. The selected result is
in-sample and exploratory.

Derived execution occurs after selector artifacts are committed, so its source
commit necessarily differs from the base environment-lock commit. The runner
must validate a narrow provenance link from the immutable base experiment and
environment lock to the committed selector trace and derived config. Do not
disable clean-tree, source-hash, or environment validation to make this work.

## 9. Run strict final analysis

Always analyze through the frozen derived config, even when selection reused a
static run:

    python analyze.py \
      --config analysis/ma_optimized_exploratory.experiment.yaml

Do not use --allow-incomplete. Strict analysis must reject a wrong or missing
run, pilot link, manifest, environment lock, corpus fingerprint, architecture
topology, prompt bundle, selector trace, GPU identity, or finalized JSONL hash.

Preserve and back up:

    config/environment.lock.json
    config/manifests/*.json
    results/*.jsonl
    results/*.meta.json
    analysis/ma_optimized_exploratory.selection.json
    analysis/ma_optimized_exploratory.experiment.yaml
    analysis/report.json
    analysis/report.md
    logs/*.plan.json
    logs/*.log

## 10. Hard stops

Stop rather than improvising if:

- pilot GO is missing, failed, or mismatched;
- the final header or architecture fingerprint differs from section 1;
- the worktree is dirty at production start;
- environment, corpus, model, tokenizer, dataset, prompt, or manifest validation
  fails;
- a question-answering step does not retrieve top-k or does not create
  per-document Extractor records;
- an aggregate step retrieves passages;
- a multi-agent answer bypasses plan summary;
- a later step runs after QA success=no;
- a repeated stage uses a treatment different from its conceptual role;
- an arm OOMs at batch 32;
- a resume changes GPU, environment, canonical batch membership, or experiment
  fingerprint;
- a timing artifact spans sessions or devices;
- selector artifacts are edited or uncommitted before a distinct run; or
- strict final analysis raises an error.

Never fix a stop by changing the scientific contract, retrying a generation,
replacing a question, tuning a treatment-specific prompt, or claiming device
performance that was not measured. Preserve artifacts, diagnose on excluded
data, and repeat the relevant gate.
