#!/usr/bin/env python3
"""Emit kaggle_qwen_arena.ipynb. Generated rather than hand-written so the
JSON is guaranteed valid."""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
ARENA = (HERE / "qwen_arena.py").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Qwen3 vs Qwen2.5 -- clean-room arena

Settle whether Qwen3 actually beats Qwen2.5 on short-form extractive QA, with a
harness that cannot inherit bugs from the `marag-precision` repo (it imports
nothing from it).

**Before you run:**

1. Notebook settings -> Accelerator -> **GPU T4 x2** (or P100).
2. Notebook settings -> **Internet: On** (needed for HF downloads).
3. Run cells in order. Total runtime at `N=500` across 4 models is roughly
   45-70 minutes on T4 x2.

## Why the Kaggle result may already be wrong

Kaggle's T4 is compute capability **7.5** and P100 is **6.0**. Neither has
native bfloat16. Qwen3 is a bf16-native family. Running it in fp16 on a T4 can
overflow activations in a way Qwen2.5 tolerates. Cell 4 tests this directly by
running the same weights in fp16 and bf16 and diffing the outputs. If they
disagree, precision is your confound, not the model.
"""),

    md("## 1. Environment"),
    code("""import subprocess, sys, torch, os

print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap",
                      "--format=csv"], capture_output=True, text=True).stdout)
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  |  {p.name}  {p.total_memory/1024**3:.1f} GiB  sm_{p.major}{p.minor}")
print(f"native bf16 supported: {torch.cuda.is_bf16_supported()}")
if not torch.cuda.is_bf16_supported():
    print("\\n>> NO NATIVE BF16. This is the single most likely cause of a spurious")
    print(">> Qwen3 regression on Kaggle. Do not skip cell 4.")
print(f"\\nvisible GPUs: {torch.cuda.device_count()}")
"""),

    code("""# Kaggle images are usually current enough; upgrade only if something is missing.
!pip -q install -U "transformers>=4.51" accelerate bitsandbytes datasets 2>&1 | tail -3
"""),

    md("## 2. The harness\n\nWritten to disk verbatim so it is byte-identical to the local run."),
    code("ARENA_SOURCE = r'''\n" + ARENA.replace("'''", "\\'\\'\\'") + "\n'''\n"
         "open('qwen_arena.py','w',encoding='utf-8').write(ARENA_SOURCE)\n"
         "print('wrote qwen_arena.py', len(ARENA_SOURCE), 'bytes')\n"),

    md("""## 3. Tier 0 -- harness self-test

Model-agnostic. Checks decoder-only left padding, batch-size invariance under
greedy decoding, thinking-mode suppression, and non-finite logits.

**Greedy decoding is mathematically batch-invariant.** If `batch_invariance`
comes back below ~0.90, the harness is provably broken and no accuracy number
from it means anything. This is the check that settles "harness or model"
without reference to any model comparison."""),
    code("""!python qwen_arena.py --tier 0 --n 16 \\
    --models qwen2.5-1.5b-fp16,qwen3-1.7b-fp16 \\
    --out tier0.json
"""),

    md("""## 4. The fp16 / bf16 confound test

Same weights, same prompts, two dtypes. On an A100 these should agree closely.
On a T4 (no native bf16) a large disagreement for Qwen3 but not Qwen2.5 is your
answer: the regression is numerical, not a property of the model."""),
    code("""import json, subprocess

for m in ["qwen2.5-1.5b", "qwen3-1.7b"]:
    for dt in ["fp16", "bf16"]:
        subprocess.run([sys.executable, "qwen_arena.py", "--tier", "2", "--n", "120",
                        "--batch-size", "8", "--models", f"{m}-{dt}",
                        "--out", f"dtype_{m}_{dt}.json"], check=False)

print(f"\\n{'model':22s} {'fp16 F1':>9s} {'bf16 F1':>9s} {'delta':>9s} {'nonfinite':>10s}")
print("-"*64)
for m in ["qwen2.5-1.5b", "qwen3-1.7b"]:
    try:
        a = json.load(open(f"dtype_{m}_fp16.json"))["eval"][0]
        b = json.load(open(f"dtype_{m}_bf16.json"))["eval"][0]
        print(f"{m:22s} {a['f1']:9.4f} {b['f1']:9.4f} {b['f1']-a['f1']:+9.4f}")
    except (IndexError, FileNotFoundError, KeyError) as e:
        print(f"{m:22s} incomplete ({type(e).__name__})")
print("\\nA delta above ~0.02 for one family only means dtype is a confound.")
"""),

    md("""## 5. Tier 2 -- the real comparison

Paired on identical questions, greedy, one generation path, HotpotQA distractor
context (all 10 paragraphs handed to every model, so retrieval is not a
variable).

`N=500` is a screening run. Note that detecting a 5-point F1 difference at 80%
power needs roughly 1,200-1,500 paired questions -- if the result lands inside
the CI, raise `N` rather than believing the point estimate."""),
    code("""N = 500              # raise to 1200+ before drawing a firm conclusion
BATCH = 8            # lower to 4 if you OOM

MODELS = ",".join([
    "qwen2.5-0.5b-fp16", "qwen3-0.6b-fp16",
    "qwen2.5-1.5b-fp16", "qwen3-1.7b-fp16",
])

!python qwen_arena.py --tier 2 --n {N} --batch-size {BATCH} \\
    --models {MODELS} --out arena_fp16.json
"""),

    md("""## 6. Quantization axis

4-bit is where the reported Qwen3 collapse was worst, so it gets its own pass.
`bnb_4bit_compute_dtype` is fp16 in the harness for T4 compatibility."""),
    code("""MODELS_4BIT = ",".join([
    "qwen2.5-1.5b-4bit", "qwen3-1.7b-4bit",
    "qwen2.5-3b-4bit",   "qwen3-4b-4bit",
])

!python qwen_arena.py --tier 2 --n {N} --batch-size {BATCH} \\
    --models {MODELS_4BIT} --out arena_4bit.json
"""),

    md("""## 7. The large pair, and 14B

Qwen3-8B at fp16 is ~16 GB and will not fit one T4. 4-bit is ~4.5 GB and fits
comfortably. Qwen2.5-14B at 4-bit is ~8.5 GB and also fits a single T4."""),
    code("""!python qwen_arena.py --tier 2 --n {N} --batch-size 4 \\
    --models qwen2.5-7b-4bit,qwen3-8b-4bit --out arena_8b.json
"""),
    code("""# 14B, 4-bit, ~8.5 GB. Single T4 is enough. Drop batch size -- the KV cache
# on a 1,600-token context dominates at this width.
!python qwen_arena.py --tier 2 --n {N} --batch-size 2 \\
    --models qwen2.5-14b-4bit --out arena_14b.json
"""),

    md("""## 7b. Retrieval headroom -- read this before believing any scale result

**This is the gate.** A multi-agent RAG system has exactly one mechanism through
which it can beat single-hop RAG: issuing a *second, better-informed query* that
retrieves evidence the first query could not reach. If the first query already
retrieves everything, decomposition is pure overhead and no model size fixes it.

That is a property of the **corpus**, not the model. It is measurable on CPU,
for free, in minutes -- and it should gate every GPU-hour you spend.

MA-RAG's published setup retrieves densely (gte-multilingual + FAISS) over the
full Karpukhin/DPR Wikipedia corpus, roughly 21M passages. A corpus of 72,094
passages constructed for 100% gold-title coverage is a fundamentally easier
retrieval problem. If single-query recall@10 for *both* gold paragraphs is
already high, the experiment cannot test its own hypothesis at any scale."""),
    code("""# CPU only. No model, no GPU. Run this before the 14B cell is worth anything.
!python retrieval_headroom.py --n 1000 --k 10 --out headroom.json
"""),

    md("## 8. Combined report"),
    code("""import json, glob
import numpy as np

evals, paired = [], []
for f in sorted(glob.glob("arena_*.json")):
    d = json.load(open(f))
    evals += d.get("eval", [])
    paired += d.get("paired", [])

print(f"{'model':26s} {'F1':>8s} {'EM':>8s} {'json':>7s} {'think':>6s} {'trunc':>6s} {'n':>5s}")
print("-"*72)
for e in sorted(evals, key=lambda x: -x["f1"]):
    print(f"{e['model']:26s} {e['f1']:8.4f} {e['em']:8.4f} {e['strict_json_rate']:7.3f} "
          f"{e['think_leak']:6d} {e['truncated']:6d} {e['n']:5d}")

print("\\npaired (positive delta = Qwen3 better)")
print("-"*72)
for p in paired:
    if "error" in p: continue
    tag = "SIGNIFICANT" if p["significant"] else "n.s."
    print(f"{p['a']:22s} -> {p['b']:22s} dF1={p['delta_f1']:+.4f} "
          f"CI={p['ci95']} p={p['mcnemar_p']:.4f} [{tag}]")

print("\\nRead this as: any comparison marked n.s. does not support a claim in")
print("either direction. Raise N and rerun before concluding.")
"""),

    md("""## 9. What each outcome means

| Tier 0 | Tier 2 | Conclusion |
|---|---|---|
| batch invariance < 0.90 | -- | Harness broken. Fix before anything else. |
| PASS | Qwen3 >= Qwen2.5 | Your Kaggle result was the harness. Migrate. |
| PASS | Qwen3 < Qwen2.5, significant, and cell 4 shows a dtype gap | Numerical, not the model. Rerun on Ampere+. |
| PASS | Qwen3 < Qwen2.5, significant, no dtype gap | Real. Qwen3 is genuinely weaker at this task and the migration is not justified on accuracy. |
| PASS | not significant | Underpowered. Raise N to 1200+. |

Note the last row is the most likely outcome at N=500 for small deltas, and it
is not a failure -- it is the measurement telling you it cannot resolve the
difference yet.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "kaggle_qwen_arena.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
