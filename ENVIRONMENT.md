# A100 environment lock contract

Follow `RUNBOOK.md` for the exact ordered commands. This file defines what the
environment lock means; it is not a substitute for the operator runbook.

`requirements-core.txt` pins the direct Python dependencies needed by the
campaign, including Transformers, bitsandbytes, datasets, Accelerate, PyYAML,
NumPy, SciPy, huggingface-hub, and pytest. It is not a complete runtime lock:
PyTorch/CUDA must come from the chosen immutable A100 container, and every
transitive dependency, driver, platform field, and GPU property is captured from
that real worker in `config/environment.lock.json`.

Before any final-manifest run, select one immutable A100 container and create
`config/environment.lock.json` from the actual preflight worker. Commit it before
production. At minimum it must contain:

- container registry reference and immutable digest;
- operating system and Python version;
- complete `pip freeze --all` output (or equivalent fully resolved lock) and its
  SHA-256;
- PyTorch, CUDA runtime, cuDNN, driver, Transformers, bitsandbytes, datasets,
  Accelerate, and PyYAML versions;
- GPU name, SKU, total memory, UUID, and driver-visible CUDA version;
- repository commit and whether the worktree was dirty;
- all model/tokenizer resolved revisions and dataset revision;
- prompt version/hash, experiment-config hash, source-bundle hash, all frozen
  cohort file/ID hashes, and the retrieval/corpus contract through the config.

Generate it with `python -m src.runner --write-environment-lock --container-ref
... --container-digest sha256:...`. The lock binds a deterministic source-bundle
hash that excludes the lock file itself; this avoids an impossible Git/hash
self-reference while still detecting any source, SPEC, config, test, or manifest
edit. Production also refuses a dirty worktree. Validate every worker with
`python -m src.runner --validate-environment-lock`. Before validation or any
production run, export the same immutable identity as
`EXPERIMENT_CONTAINER_REF` and `EXPERIMENT_CONTAINER_DIGEST`; validation fails
closed when those values do not match the committed lock.

Every accuracy worker must compare itself with this artifact before loading a
model and fail on a mismatch. Timing uses one reserved uncontended A100 matching
the same lock. Worker UUIDs may differ, but GPU SKU, memory, compute capability,
driver, container, and software must match; a resumed arm remains bound to its
original physical GPU UUID.

Pinned direct-package values and the expected PyTorch container baseline,
pending confirmation by the generated lock:

```text
torch          2.13.0+cu130
CUDA runtime   13.0
transformers   5.14.1
bitsandbytes   0.50.0
datasets       5.0.1
accelerate     1.14.0
PyYAML         6.0.3
numpy          2.5.1
scipy          1.18.0
huggingface-hub 1.26.0
pytest         9.1.1
```

Do not create a lock by filling unknown fields with guessed versions. A lock is
valid only after a real A100 preflight records and verifies it.

`requirements-manifest.txt` is separate from the inference environment. It pins
the two libraries used by `scripts/freeze_final_sample.py`; neither is added to
the production stack solely for manifest generation.
