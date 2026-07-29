# marag-precision

*(repo: Maxim-Mohareb-Michael-Zhang-Fun-Time — NeurIPS conf)*

Experimental harness for **Role-Aware Precision Allocation in Multi-Agent RAG**
(NeurIPS 2026 workshop submission).

Research question: at SLM scale, does quantizing one agent in a multi-agent RAG
pipeline hurt the same roles that *shrinking* one agent hurts?

**Read `SPEC.md` before changing anything.** The scientific design is locked, the
build order is fixed, and there are two mandatory human approval gates.
`PROGRESS.md` is the session-to-session handoff.

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -U transformers accelerate bitsandbytes datasets pyyaml
```

## Gate 1 smoke test

10 questions, FP16, all four agents, every raw output printed for inspection:

```bash
.venv/Scripts/python.exe -X utf8 smoke_test.py --n 10
```

Add `--precision 4bit` if FP16 OOMs on a 4 GB card (SPEC §5a).

## Layout

| Path | Purpose |
|---|---|
| `src/prompts.py` | One versioned prompt template per role. Never tuned per precision. |
| `src/parsing.py` | Parsers and the six-label failure taxonomy. **No retry path — see SPEC §5.** |
| `src/models.py` | Load the base model at fp16 / 8bit / 4bit; batched greedy generation. |
| `src/agents.py` | Role call wrappers; degraded-propagation policy on parse failure. |
| `src/pipeline.py` | Dataset loading and orchestration. |
| `src/metrics.py` | HotpotQA EM/F1 normalization, bootstrap CIs. |
| `smoke_test.py` | Gate 1 driver. |
| `results/` | JSONL outputs (gitignored). |

Build steps 3-12 in SPEC §11 are not implemented yet.

## The one rule that is easy to break

Never retry, resample, or regex-rescue a failed parse. The parse-failure rate is
the experiment's mechanism evidence, and a retry destroys it. SPEC §5.
