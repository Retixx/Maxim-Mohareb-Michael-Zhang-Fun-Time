# Clean-room arena

A model comparison harness that imports **nothing** from this repository.

That is the entire design constraint. If a regression reproduces here, it is a
property of the model or the hardware. If it does not reproduce here, it came
from the harness in `src/`. There is no third option, because nothing in `src/`
is on the import path.

## Files

| File | Purpose |
|---|---|
| `qwen_arena.py` | The harness. Runs locally or on Kaggle, unchanged. |
| `kaggle_qwen_arena.ipynb` | Kaggle wrapper, T4 x2 / P100. Embeds `qwen_arena.py` byte-identically. |
| `build_notebook.py` | Regenerates the notebook from `qwen_arena.py`. Run after editing the harness. |

Nothing here touches `config/experiment.yaml`, any frozen manifest, or any
pinned hash. It is inert with respect to the production experiment.

## Tiers

**Tier 0 — harness self-test.** Model-agnostic. The important check is
**batch-size invariance**: greedy decoding is mathematically independent of
batch size, so running the same prompts at `batch_size=1` and `batch_size=8`
must produce the same strings. Floating-point reduction order makes this
slightly noisy in practice, so the threshold is 0.90 rather than 1.00; a
right-padding bug on a decoder-only model collapses it toward zero.

Tier 0 also checks:
- `padding_side == "left"` (asserted at construction, not merely logged)
- `enable_thinking=False` actually changed the rendered template, rather than
  being silently swallowed
- `<think>` tags reaching the output
- non-finite logits, which on pre-Ampere hardware usually means fp16 overflow
- truncation at `max_new_tokens`

If Tier 0 fails, the run aborts rather than reporting accuracy.

**Tier 2 — task accuracy.** HotpotQA `distractor` validation. All 10 paragraphs
(2 gold, 8 distractors) go to every model, so **retrieval is not a variable**.
Canonical HotpotQA `normalize_answer` / token F1 / EM, transcribed from the
official evaluation script. Paired bootstrap CI on ΔF1 (10,000 replicates) plus
an exact McNemar on EM.

## Running

```bash
python qwen_arena.py --tier 0 --n 16 --models qwen2.5-1.5b-4bit,qwen3-1.7b-4bit
```

```bash
python qwen_arena.py --tier 2 --n 300 --batch-size 4 --models qwen2.5-1.5b-4bit,qwen3-1.7b-4bit
```

With no `--models`, it picks the largest matched pairs that fit the detected
VRAM, falling back to 4-bit per model.

## Hardware notes

**4 GB local (RTX 3050 Laptop).** 0.5B/0.6B fit at fp16; everything else needs
4-bit. Use `--batch-size 4` — the 10-paragraph context is ~1,600 tokens and the
KV cache dominates.

**Kaggle T4 (sm_75) and P100 (sm_60) have no native bf16.** Qwen3 is a
bf16-native family. Running it in fp16 on those cards risks activation overflow
that Qwen2.5 may tolerate. Notebook cell 4 tests this directly by running the
same weights in both dtypes and diffing. Treat a one-family-only gap as a
numerical confound, not a model property.

## Matched pairs

Qwen3 has no 0.5B and Qwen2.5 has no 0.6B/1.7B, so pairing is nearest-neighbour
by parameter count:

| Qwen2.5 | Qwen3 | size delta |
|---|---|---|
| 0.5B | 0.6B | +20% |
| 1.5B | 1.7B | +13% |
| 3B | 4B | +33% |
| 7B | 8B | +14% |

Qwen3 is the larger model in every pair. Disclose this — it biases *toward*
Qwen3, so a Qwen3 loss is the stronger finding and a Qwen3 win is partly a size
effect.

## Measured result (2026-08-07, n=600, k=10, pooled 72,094-passage corpus)

| stratum | n | SINGLE both-gold | ORACLE two-pass | headroom |
|---|---|---|---|---|
| hidden_bridge | 475 | 0.4716 | 0.8863 | **+0.4147** |
| fully_named | 125 | 0.8320 | 0.8880 | +0.0560 |

Verdict: **GO**. Headroom is large and lands exactly where the design predicts —
on hidden-bridge questions, which are 73% of the frozen cohort (1097/1500).

Note `single_any_recall` on hidden_bridge is 0.9684: one gold paragraph is almost
always retrieved. It is the *second*, hidden one that a single query misses. That
is precisely what a second hop exists to reach.

For contrast, `scripts/recall_check.py` on the real planner sub-questions from
`baseline_qwen2.5-1.5b_n750_seed7` (n=730, unstratified) shows the old
decomposition going **backwards**: both-gold recall@10 falls from 0.495 (single
query) to 0.296 (one query per sub-question, equal read budget). Even at double
budget it only reaches 0.410.

So the corpus supports a large multi-hop win and the old retrieval logic spent
that headroom in the wrong direction. That is a pipeline defect, not an SLM
capacity limit and not a corpus artifact.

## Power

Detecting a 5-point F1 difference at 80% power needs roughly **1,200–1,500
paired questions**, using the ~40% discordance rate measured in this project's
own `results/`. `n=300` is a screening run: it can catch a large effect and
nothing else. A non-significant result at n=300 means *underpowered*, not
*equal*.
