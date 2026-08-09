#!/usr/bin/env python3
"""Emit kaggle_smoke_gate.ipynb — SPEC §15 acceptance gates on Kaggle.

Generated so the JSON is guaranteed valid. Every analysis cell is standalone:
no cell depends on a variable defined in another, so a failure or a re-run in
the middle does not cascade.
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
BRANCH = "multihop-vs-single-hop-rag-bug-fix"
REPO = "https://github.com/Retixx/Maxim-Mohareb-Michael-Zhang-Fun-Time.git"
COMMIT = "56cd63f"
ALIAS = "small"          # Qwen3-1.7B; ALLOWED_MODEL_ALIASES = ("tiny","small")
OUTDIR = "small_4bit_bs4"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


# Shared loader injected into every analysis cell so cells stay independent.
LOADER = f'''import json, glob, os, collections
import numpy as np

RESULTS = "analysis/local_smoke/{OUTDIR}/results/"

def _load(pattern):
    hits = glob.glob(RESULTS + pattern)
    if not hits:
        raise FileNotFoundError(f"no artifact matching {{pattern!r}} in {{RESULTS}} "
                                "- did cell 5 finish?")
    out = {{}}
    for line in open(hits[0], encoding="utf-8"):
        try: r = json.loads(line, strict=False)
        except Exception: continue
        if "f1" in r and "predicted_answer" in r:
            out[r["question_id"]] = r
    return out
'''

cells = [
    md(f"""# SPEC §15 acceptance gates — Qwen3-1.7B

Runs the multi-agent vs single-hop smoke gate at `--model {ALIAS}`
(Qwen3-1.7B, 4-bit), one size up from the local 0.6B run that failed Gate C.

**Why this run exists.** Local RTX 3050, Qwen3-0.6B 4-bit, n=200 paired:

| | MA pipeline | single-hop | Δ |
|---|---|---|---|
| F1 | 0.0592 | 0.2658 | **−0.2066** |
| EM | 0.0450 | 0.1750 | −0.1300 |

Gates A and B **passed** — retrieval recall 0.5077 → 0.9271, follow-up firing
1097/1097 — so the evidence reaches the model. At 5.9% F1 the 0.6B model could
not consume it: QA fired the semantic stop on 32% of step-1 calls, the Extractor
emitted malformed spans 16% of the time, and 26 of 200 plans were single-step.
This run tests whether 1.7B clears that floor.

**Before running**
1. Settings → Accelerator → **GPU T4 x2** (or P100)
2. Settings → **Internet: On**
3. Expect **3–5 hours**. Session limit is 12 h. Don't close the tab.

**Notes**
- Batch size is fixed at 4 by the smoke contract, validated in two places
  (`run_retrieval_smoke.py` and `_validate_local_smoke_contract`). Don't raise it.
- `{ALIAS}` is the largest alias the contract allows
  (`ALLOWED_MODEL_ALIASES = ("tiny","small")`).
- `check_pilot.py` fails closed on `model_revisions: TBD` (BUG-5) with no
  `--allow-unpinned-tbd` escape hatch, so cell 6 computes Gate C/D directly from
  the artifacts. Same arithmetic, same data.
- Cells 6–9 are standalone; each reloads from disk. Re-run any of them alone.
"""),

    md("## 1. Environment"),
    code("""import subprocess, torch
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,compute_cap",
                      "--format=csv"], capture_output=True, text=True).stdout)
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} {p.total_memory/1024**3:.1f} GiB sm_{p.major}{p.minor}")
print(f"native bf16: {torch.cuda.is_bf16_supported()}")
"""),

    md("## 2. Clone at the verified commit, and fix line endings\n\n"
       "The manifests are pinned by raw SHA-256 and `.gitattributes` only applies "
       "at checkout, so the tree is renormalized before anything reads it "
       "(SPEC §14 BUG-6). The assert fails loudly rather than 10 minutes into the run."),
    code(f"""!git clone -q --branch {BRANCH} {REPO} /kaggle/working/marag
%cd /kaggle/working/marag
!git checkout -q {COMMIT}
!git rm --cached -r -q . && git reset --hard -q
!git log --oneline -3

import hashlib, pathlib
h = hashlib.sha256(pathlib.Path("config/manifests/final_n1500_seed20260805.json").read_bytes()).hexdigest()
want = "ba2836fdcd180a16daae625c09ae6bc4f68aee26a072e9bbef14e44ec6868f90"
print("manifest sha256:", h)
assert h == want, "CRLF corruption - manifest hash does not match the pin"
print("manifest OK")
"""),

    md("## 3. Dependencies"),
    code("""!pip -q install -U "transformers>=4.51" accelerate bitsandbytes datasets 2>&1 | tail -3
"""),

    md("## 4. Assert thinking is OFF\n\n"
       "SPEC §14 BUG-10. Qwen3 defaults to hybrid thinking; with it on the Planner "
       "scored parse_ok **0.000**, burning its whole token ceiling on reasoning. "
       "`render_chat` must emit a pre-closed `<think></think>` block."),
    code("""from transformers import AutoTokenizer
from src import models

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
msgs = [{"role": "system", "content": "You answer in JSON."},
        {"role": "user", "content": "Which is older, A or B?"}]
rendered = models.render_chat(tok, msgs)
print(repr(rendered[-70:]))
assert rendered.rstrip().endswith("</think>"), "thinking NOT disabled - stop here"
print("thinking disabled OK")
"""),

    md("""## 5. Run the gates

Output streams live — `python -u` unbuffers Python, `grep --line-buffered`
flushes per line, and there is no `tail`, which would otherwise hold everything
until the process exits.

Gates A and B run first, on CPU, deterministically. They should reproduce the
local numbers exactly (0.5077 → 0.9271, 1097/1097); if they don't, the checkout
is wrong and nothing downstream is trustworthy.

Progress lines look like `N/2000 missing calls written`. The Extractor stage is
the long one — 10 calls per retrieved document."""),
    code(f"""!rm -f analysis/local_smoke/{OUTDIR}/results/*.lock 2>/dev/null
!python -u scripts/run_retrieval_smoke.py \\
    --model {ALIAS} --batch-size 4 --allow-unpinned-tbd --execute \\
    2>&1 | grep --line-buffered -vE "Loading weights"
"""),

    md("""## 6. Gate C — paired accuracy

Thresholds (SPEC §15.5): overall ΔF1 ≥ **+5.0** pts, bootstrap 95% CI lower
bound > **+2.0**, McNemar p < **0.01**."""),
    code(LOADER + '''from math import comb

ma, so = _load("baseline_*.jsonl"), _load("single_fp16_*.jsonl")
ids = sorted(set(ma) & set(so))
a = np.array([ma[i]["f1"] for i in ids])
b = np.array([so[i]["f1"] for i in ids])
diff = a - b

rng = np.random.default_rng(20260807)
boot = np.array([diff[rng.integers(0, len(ids), len(ids))].mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])

ma_only = sum(1 for i in ids if ma[i]["em"] > so[i]["em"])
so_only = sum(1 for i in ids if so[i]["em"] > ma[i]["em"])
disc = ma_only + so_only
k = min(ma_only, so_only)
pval = min(1.0, 2 * sum(comb(disc, j) for j in range(k + 1)) / 2 ** disc) if disc else 1.0

print(f"paired n = {len(ids)}")
print(f"MA      F1 {a.mean():.4f}   EM {np.mean([ma[i]['em'] for i in ids]):.4f}")
print(f"single  F1 {b.mean():.4f}   EM {np.mean([so[i]['em'] for i in ids]):.4f}")
print(f"dF1 {diff.mean()*100:+.2f} pts    95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")
print(f"McNemar p = {pval:.5f}   (MA-only wins {ma_only}, single-only {so_only})")
w = int((diff > 0).sum()); l = int((diff < 0).sum())
print(f"MA wins {w}, loses {l}, ties {len(ids)-w-l}")
print()
ok = diff.mean() >= 0.05 and lo > 0.02 and pval < 0.01
print("GATE C:", "PASS" if ok else "FAIL")
print("  0.6B reference: dF1 -20.66 pts, CI [-27.14, -14.13]")
'''),

    md("""## 7. Gate D — stratum sanity

`fully_named` must stay within **±2.0** pts. There is only +0.056 of retrieval
headroom on that stratum, so a large multi-agent win there indicates leakage,
not a fix."""),
    code(LOADER + '''ma, so = _load("baseline_*.jsonl"), _load("single_fp16_*.jsonl")
ids = sorted(set(ma) & set(so))

strata = collections.defaultdict(list)
for i in ids:
    strata[ma[i].get("retrieval_stratum", "unknown")].append(i)

print(f"{'stratum':16s} {'n':>5s} {'MA F1':>8s} {'single':>8s} {'delta pts':>10s}")
print("-" * 52)
for s, sub in sorted(strata.items()):
    am = np.mean([ma[i]["f1"] for i in sub])
    bm = np.mean([so[i]["f1"] for i in sub])
    print(f"{s:16s} {len(sub):5d} {am:8.4f} {bm:8.4f} {(am-bm)*100:+10.2f}")

fn = strata.get("fully_named", [])
if fn:
    dlt = np.mean([ma[i]["f1"] - so[i]["f1"] for i in fn]) * 100
    print()
    print("GATE D:", "PASS" if abs(dlt) <= 2.0 else "FAIL", f"(fully_named {dlt:+.2f} pts)")
'''),

    md("""## 8. Where the pipeline loses

Only meaningful if Gate C failed. This separates *"retrieval did not improve"*
from *"the model cannot consume good evidence"* — the distinction that decides
whether the next fix belongs in retrieval or in the Extractor/QA contract."""),
    code(LOADER + '''ma, so = _load("baseline_*.jsonl"), _load("single_fp16_*.jsonl")
ids = sorted(set(ma) & set(so))

stop = collections.Counter(ma[i].get("stop_reason", "?") for i in ids)
print("stop_reason      :", dict(stop))

depth = collections.Counter(ma[i].get("planner_emitted_depth") for i in ids)
print("planner depth    :", dict(sorted(depth.items(), key=lambda x: (x[0] is None, x[0]))))
print("executed steps   : mean %.2f" % np.mean([ma[i].get("executed_steps", 0) for i in ids]))

fired = sum(ma[i].get("retrieval_followup_fired_step_count", 0) for i in ids)
elig  = sum(ma[i].get("retrieval_followup_eligible_step_count", 0) for i in ids)
print(f"follow-up fired  : {fired}/{elig}" + (f" = {fired/elig:.3f}" if elig else ""))

rec = np.mean([ma[i].get("retrieval_gold_title_recall", 0) for i in ids])
allg = np.mean([bool(ma[i].get("retrieval_all_gold")) for i in ids])
print(f"gold title recall: {rec:.4f}   both-gold rate: {allg:.4f}")

ev = [ma[i].get("evidence_f1") for i in ids if ma[i].get("evidence_f1") is not None]
if ev: print(f"evidence F1      : {np.mean(ev):.4f}  (Extractor vs gold supporting sentences)")

src = collections.Counter(ma[i].get("final_answer_source", "?") for i in ids)
print("answer source    :", dict(src))

# The decisive split: retrieval succeeded, but did the answer?
good = [i for i in ids if ma[i].get("retrieval_all_gold")]
if good:
    print()
    print(f"On the {len(good)} questions where BOTH gold passages were retrieved:")
    print(f"   MA F1     {np.mean([ma[i]['f1'] for i in good]):.4f}")
    print(f"   single F1 {np.mean([so[i]['f1'] for i in good]):.4f}")
    print("   -> if MA still loses here, the bottleneck is downstream of retrieval.")
'''),

    md("## 9. Per-stage health"),
    code('''import json, glob, collections
f = glob.glob("analysis/local_smoke/''' + OUTDIR + '''/results/baseline_*.jsonl")[0]

by = collections.defaultdict(lambda: [0, 0, 0])
for line in open(f, encoding="utf-8"):
    try: r = json.loads(line, strict=False)
    except Exception: continue
    s = r.get("stage")
    if not s or "raw_output" not in r: continue
    row = by[s]; row[0] += 1
    if "<think>" in (r.get("raw_output") or ""): row[1] += 1
    if r.get("parse_status") == "ok": row[2] += 1

print(f"{'stage':20s} {'n':>6s} {'<think>':>8s} {'parse_ok':>9s}")
print("-" * 46)
for s, (n, t, ok) in sorted(by.items()):
    print(f"{s:20s} {n:6d} {t:8d} {ok/n:9.3f}")
print()
print("0.6B reference: planner 0.956, step_definer 1.000, extractor 0.841,")
print("                <think> zero everywhere.")
print("Any <think> > 0 means BUG-10 regressed and the run is void.")
'''),

    md("## 10. Save artifacts\n\nDownload from the **Output** tab on the right."),
    code(f"""!tar czf /kaggle/working/smoke_artifacts.tar.gz analysis/local_smoke/
!ls -lh /kaggle/working/smoke_artifacts.tar.gz
"""),

    md("""## Reading the result

| Outcome | Meaning | Next |
|---|---|---|
| Gate C PASS | Retrieval fix works; 0.6B was below the capability floor | Human gate, then merge |
| Gate C FAIL, MA F1 ≫ 0.059 | Scale helps but isn't sufficient | Check cell 8 stop_reason and evidence F1 |
| Gate C FAIL, MA F1 ≈ 0.059 | Bottleneck is not model size | Extractor/QA contract, not retrieval |
| Gate D FAIL (big fully_named win) | Suspect leakage | Investigate before believing Gate C |

The decisive line is at the bottom of cell 8: multi-agent versus single-hop **on
the subset where both gold passages were retrieved**. If MA loses there, the
evidence was present and something downstream discarded it — which is the same
failure mode as the original 1,588 → 376 token Extractor bottleneck, just at a
different layer.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = HERE / "kaggle_smoke_gate.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
