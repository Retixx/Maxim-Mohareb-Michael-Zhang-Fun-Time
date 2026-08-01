# marag-precision

*(repo: Maxim-Mohareb-Michael-Zhang-Fun-Time)*

Experimental harness for **Role-Aware Capacity Allocation in Multi-Agent RAG**
(NeurIPS 2026 workshop submission, deadline Aug 29 2026).

At SLM scale, is a four-agent RAG pipeline better served by spending its memory
budget unevenly across roles than evenly? Three sub-questions, in order:

1. Which role is most sensitive to **quantization**? — answered for model 1 (SPEC §14)
2. Which role is most sensitive to **parameter reduction**, measured in *this*
   pipeline rather than borrowed from another paper? — Phase S, not yet run
3. Does role-aware allocation beat uniform allocation at the same memory
   footprint, and does either beat plain single-call RAG? — Phase D, not yet run

**Read `SPEC.md` before changing anything.** The scientific design is locked, the
build order is fixed, and there are mandatory human approval gates.
`PROGRESS.md` is the session-to-session handoff.

## Status

| Phase | What | State |
|---|---|---|
| Q | Quantization ablation, one role at 4-bit | **complete**, model 1, n=750 seed 7 |
| S | Size ablation, one role at Qwen2.5-0.5B | wired, not run |
| H | Q-vs-S head-to-head (re-analysis, no new runs) | blocked on S |
| D | Role-aware vs uniform vs single-call RAG | blocked on H |

Build steps 1–11 of SPEC §11 are done. Next is step 12 (`analyze.py`), then
**Gate 3**. Phase Q's headline: quantizing the **Extractor** costs +3.20 EM pp
[+0.67, +5.87] — the only role resolved on answer EM at n=750.

## Setup

```bash
# Windows / local dev
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -U "transformers==5.14.1" "bitsandbytes==0.50.0" \
    "datasets==5.0.1" accelerate pyyaml
```

Versions are pinned because they affect numerics, and they are recorded in every
run's metadata. `transformers` 5.x is required: `models.py` passes `dtype=`,
which replaced `torch_dtype=` in v5.

Full sweeps run on Kaggle (T4), not locally — see `notebooks/kaggle_run.ipynb`.
Local VRAM readings are not trustworthy (SPEC §14).

## Running

```bash
# one run
python -m src.runner --config config/experiment.yaml --run stepdef_4bit

# size-ablation run — loads two different models in one run
python -m src.runner --config config/experiment.yaml --run stepdef_small

# smoke test: 10 questions, every raw output printed for inspection
python -X utf8 smoke_test.py --n 10

# ...at 4-bit throughout, if fp16 OOMs on a 4 GB card (SPEC §5b contingency)
python -X utf8 smoke_test.py --n 10 --run ma_uniform_4bit

# Gate 2 report over existing Phase Q results
python gate2_report.py results --n 750 --seed 7
```

Resume is the default, not a flag: kill the process and rerun the same command.
Never split one run across two machines (SPEC §6).

## Layout

| Path | Purpose |
|---|---|
| `SPEC.md` | The locked design. Read first. |
| `PROGRESS.md` | Handoff log — newest entry first. |
| `config/experiment.yaml` | Models, runs, n, seeds, batch size. Everything is driven from here. |
| `src/prompts.py` | One versioned prompt per role. **Frozen at v5.** Never tuned per precision or size. |
| `src/parsing.py` | Parsers and the six-label failure taxonomy. **No retry path — SPEC §5.** |
| `src/models.py` | Load a model at fp16/8bit/4bit; footprints; batched greedy generation. |
| `src/agents.py` | Role call wrappers; degraded-propagation policy on parse failure. |
| `src/pipeline.py` | Question sampling (incl. disjoint confirmation sets) and stage-major orchestration. |
| `src/metrics.py` | HotpotQA EM/F1 normalization, bootstrap CIs, AUROC, ECE. |
| `src/evidence.py` | Extraction accuracy vs gold `supporting_facts` (SPEC §5c). |
| `src/mechanism.py` | Strict format, verbatim rate, selection churn (SPEC §5b). |
| `src/runner.py` | Sweep driver: treatment resolution, checkpoint/resume, metadata. |
| `smoke_test.py` | Gate smoke-test driver. |
| `gate2_report.py` | Gate 2 report. **Superseded by `analyze.py`; kept for provenance.** |
| `notebooks/kaggle_run.ipynb` | Thin Kaggle launcher. No logic lives there. |
| `results/` | JSONL outputs. Committed — Kaggle pushes after every run. |

`analyze.py` (SPEC §8, §10) is build step 12 and does not exist yet.

## The rules that are easy to break

1. **Never retry, resample, or regex-rescue a failed parse.** The parse-failure
   rate is the experiment's mechanism evidence; a retry destroys it. SPEC §5.
2. **Never use constrained or grammar-based decoding.** It drives parse failures
   to zero by construction and deletes the same measurement. The temptation gets
   worse in Phase S, where the 0.5B model is *expected* to parse worse — that is
   the measurement, not a defect. SPEC §12.
3. **Never compare accuracy across arms without the footprint beside it,** and
   use `deduped_footprint_mb`, not `coresident_footprint_mb`. SPEC §5d explains
   why the naive sum inverts the sign of the memory claim.
4. **Never edit the prompts.** Frozen at v5; an edit invalidates every run
   already collected.
