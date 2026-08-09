#!/usr/bin/env python3
"""Emit kaggle_passage_replay.ipynb — the frozen-upstream passage replay.

Generated so the JSON is guaranteed valid. Analysis cells are standalone and
read from disk, so any of them can be re-run alone.
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
BRANCH = "multihop-vs-single-hop-rag-bug-fix"
REPO = "https://github.com/Retixx/Maxim-Mohareb-Michael-Zhang-Fun-Time.git"
COMMIT = "f45cafc"
OUT = "analysis/passage_plus_spans_replay"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


cells = [
    md(f"""# Frozen-upstream passage replay — does the Extractor cause the gap?

**~30 minutes.** 1,038 scored generations plus a 21-call sentinel, versus ~5,200
for a full pipeline run.

## What this tests

Gate C failed at Qwen3-1.7B: MA **0.2312** vs single-hop **0.4213**, ΔF1
**−19.01 pts**, CI [−26.58, −11.56], p=0.00026. Retrieval was *not* the cause —
on the 128 questions where both gold passages were retrieved, MA still scored
0.3141 against single-hop's 0.5133.

The suspect is the Extractor. Measured on those 128 questions:

| | |
|---|---|
| gold answer in **raw** Extractor output | 84/128 |
| still present after normalization, in QA's prompt | **52/128** |
| spans rejected by the normalizer | 830/1,460 (56.8%) |
| rejected as `not_in_source` | 625 |
| of those, near-matches (token-F1 ≥ 0.80) | 228/625 |

So ~42 questions lose the answer at the model and ~34 at the normalizer.

## The intervention

Everything upstream is frozen and replayed from the recorded trace: Planner
output, Step Definer routes and tasks, retrieval titles and order, passages,
Extractor spans, executed-step topology, stop reasons. Only **QA** and the
strictly downstream **plan_summary** are regenerated.

Two treatment arms:

- `spans_plus_passages` (627 calls, all 200 questions) — QA sees the accepted
  spans **plus** the frozen passages
- `passages_only` (411 calls, 128 both-gold questions) — QA sees passages
  **instead of** spans, which separates passage access from repetition salience

The recorded single-hop arm is never regenerated; it's the comparator.

## Before running
1. Accelerator → **GPU T4** (the harness aborts on non-T4 before writing anything)
2. **Internet: On**
3. Run top to bottom.

This is a fixed-trace diagnostic, not an acceptance run. It cannot output
`PASS_GATE_C`, and no SPEC claim follows from it directly.
"""),

    md("## 1. Environment — must be T4"),
    code("""import subprocess, torch
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap",
                      "--format=csv"], capture_output=True, text=True).stdout)
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} | sm_{p.major}{p.minor}")
assert "T4" in p.name, f"harness requires a Tesla T4, got {p.name}"
print("T4 OK")
"""),

    md("## 2. Clone and renormalize\n\n"
       "`.gitattributes` only applies at checkout, and the manifests are pinned by "
       "raw SHA-256, so the tree is renormalized before anything reads it."),
    code(f"""!git clone -q --branch {BRANCH} {REPO} /kaggle/working/marag
%cd /kaggle/working/marag
!git checkout -q {COMMIT}
!git rm --cached -r -q . && git reset --hard -q
!git log --oneline -1
"""),

    md("## 3. Dependencies"),
    code("""!pip -q install -U "transformers>=4.51" accelerate bitsandbytes datasets 2>&1 | tail -3
"""),

    md("""## 4. CPU audit — fails closed before any GPU work

Verifies the four artifact hashes, reconstructs all 627 recorded prompt hashes,
checks the 4,260 Extractor joins, and confirms no gold field is read during
prompt assembly. Under a minute. Must print `AUDIT_PASS`."""),
    code("""!python -u clean_room/passage_replay.py --audit-only
"""),

    md(f"""## 5. Run the replay

Streams live — no `tail`, which would hold output until the process exits.

Order: 21-call reproducibility sentinel first (byte-identical reproduction of
recorded outputs; **any mismatch aborts**), then the 1,038 scored generations.
Batch 4, greedy, `enable_thinking=False`, `<think>` in any output aborts.

Outputs land in `{OUT}/`."""),
    code("""!python -u clean_room/passage_replay.py --execute 2>&1 | grep --line-buffered -vE "Loading weights"
"""),

    md("## 6. Result"),
    code(f'''import json, pathlib
s = json.loads(pathlib.Path("{OUT}/summary.json").read_text(encoding="utf-8"))
print(json.dumps(s, indent=2)[:6000])
'''),

    md("""## 7. Headline table

Recorded baselines for comparison:

| arm | overall F1 | both-gold F1 |
|---|---|---|
| MA, spans only (recorded) | 0.2312 | 0.3141 |
| single-hop (recorded) | 0.4213 | 0.5133 |

The gap to close is **19.01 pts** overall, **19.92** on the both-gold subset."""),
    code(f'''import json, pathlib

s = json.loads(pathlib.Path("{OUT}/summary.json").read_text(encoding="utf-8"))

def walk(node, depth=0, prefix=""):
    """Print any numeric leaf whose key looks like a metric."""
    keep = ("f1", "em", "delta", "gap", "recover", "ci", "p_value",
            "mcnemar", "wins", "losses", "ties", "n_")
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, depth + 1, f"{{prefix}}.{{k}}" if prefix else k)
    elif isinstance(node, list):
        if node and all(isinstance(x, (int, float)) for x in node):
            if any(t in prefix.lower() for t in keep):
                print(f"{{prefix:58s}} {{node}}")
    elif isinstance(node, (int, float)):
        if any(t in prefix.lower() for t in keep):
            print(f"{{prefix:58s}} {{node}}")

walk(s)
print()
print("Reference: MA 0.2312 / 0.3141, single-hop 0.4213 / 0.5133 (overall / both-gold)")
'''),

    md("""## 8. How to read it

| `passages_only` result | Meaning |
|---|---|
| Closes most of the gap | The Extractor contributes nothing over raw passages. Extract-then-answer is the defect. |
| Closes some | Extraction loses real information but isn't the whole story; QA/summary reasoning holds the rest. |
| Closes little | Extraction is **not** the bottleneck. Look at QA and the plan summary. |

Compare `spans_plus_passages` against `passages_only`: if adding passages helps
but removing spans doesn't hurt, the spans were never load-bearing.

If passages bring MA to roughly single-hop's score, the honest reading is
"extraction is not contributing", **not** "the §4.3 contract is repairable" —
QA with full passages is doing what a single call with full passages does.

This is labelled a `Gate-C-comparable fixed-trace diagnostic`. It was designed
after seeing Gate C fail and it violates the current §4.3 QA-input contract, so
an acceptance claim needs an approved SPEC change and a fresh full run."""),

    md("## 9. Save artifacts"),
    code(f"""!tar czf /kaggle/working/passage_replay_artifacts.tar.gz {OUT}/
!ls -lh /kaggle/working/passage_replay_artifacts.tar.gz
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = HERE / "kaggle_passage_replay.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
