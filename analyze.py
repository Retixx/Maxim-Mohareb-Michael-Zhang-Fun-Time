"""Analysis for SPEC §10: every table and figure, from results/ alone.

    python analyze.py --n 3000 --seed 7
    python analyze.py --n 750 --seed 7 --gate3     # reproduce SPEC §14

Supersedes gate2_report.py, which is kept only for provenance (§8).

WHAT THIS ENFORCES, so it is not quietly undone later
-----------------------------------------------------
1. ONE primary test (§5f). The confirmatory result is the format-heavy vs
   knowledge-heavy contrast, pre-registered 2026-07-29. Everything per-role is
   printed as DESCRIPTIVE and is never labelled "significant". With 4 roles x
   3 treatments x 3 metrics there are up to 36 comparisons; calling per-role
   intervals significant without correction is the objection that sinks the
   paper (§13c.1).
2. Pooling resamples QUESTIONS, never stacks runs (§10). Stacking two runs'
   per-question deltas reuses the same questions against the same baseline, so
   the interval comes out too narrow. At n=750 that flipped a call.
3. Memory is reported deduplicated (§5d). `coresident` counts one instance per
   stage, which quadruple-counts a uniform run.
4. Bootstrap is vectorized and order-invariant. The pure-python double loop was
   O(resamples x n) per interval and order-sensitive at fixed seed (§13a.10-11).
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ROLES = ["planner", "step_definer", "extractor", "qa"]
ROLE_LABEL = {"planner": "Planner", "step_definer": "Step Definer",
              "extractor": "Extractor", "qa": "QA"}

# run-id prefix -> role
PREFIX_ROLE = {"planner": "planner", "stepdef": "step_definer",
               "extractor": "extractor", "qa": "qa"}
# run-id suffix -> treatment label
SUFFIX_TREATMENT = {"4bit": "4-bit", "8bit": "8-bit", "small": "0.5B"}

# SPEC §10 Figure 3 / §5b prediction 3 groupings.
FORMAT_HEAVY = ["step_definer", "extractor"]
KNOWLEDGE_HEAVY = ["planner", "qa"]

N_RESAMPLES = 10_000


# ---------------------------------------------------------------- loading ---

def discover(results_dir, n, seed):
    """Find every run file for this (n, seed), whatever model slug it carries.

    Slugs gained a model list when Phase S introduced multi-model runs, e.g.
    `stepdef_small_qwen2.5-1.5b+qwen2.5-0.5b_n3000_seed7`. Match on the run id
    and the n/seed tail, and treat everything between as the model tag.
    """
    out = {}
    pat = re.compile(r"^(.+?)_([a-z0-9.\-+]+)_n%d_seed%d\.jsonl$" % (n, seed))
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        m = pat.match(os.path.basename(path))
        if m:
            out[m.group(1)] = path
    return out


def load_run(path):
    answers, calls = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type") == "answer":
                answers[r["question_id"]] = r
            else:
                calls.append(r)
    meta_path = path[: -len(".jsonl")] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return {"answers": answers, "calls": calls, "meta": meta, "path": path}


def split_run_id(run_id):
    """`stepdef_small` -> ('step_definer', '0.5B'); baseline -> (None, None)."""
    for suffix, treat in SUFFIX_TREATMENT.items():
        if run_id.endswith("_" + suffix):
            prefix = run_id[: -len(suffix) - 1]
            if prefix in PREFIX_ROLE:
                return PREFIX_ROLE[prefix], treat
    return None, None


# ------------------------------------------------------------- statistics ---

def boot_ci(values, n_resamples=N_RESAMPLES, alpha=0.05, seed=0):
    """Percentile bootstrap over the mean. Returns (mean, lo, hi).

    Vectorized, and sorted first so the interval does not depend on the order
    records happened to be appended to a JSONL (§13a.10-11).
    """
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_delta(base, other, metric, seed=0):
    """Paired bootstrap on (baseline - run) per question. Returns pp units."""
    qids = sorted(set(base["answers"]) & set(other["answers"]))
    if not qids:
        return float("nan"), float("nan"), float("nan"), 0
    d = [base["answers"][q][metric] - other["answers"][q][metric] for q in qids]
    m, lo, hi = boot_ci(d, seed=seed)
    return 100 * m, 100 * lo, 100 * hi, len(qids)


def clustered_group_delta(base, runs_by_role, roles, metric, seed=0):
    """Mean delta across `roles`, bootstrapped by resampling QUESTIONS.

    SPEC §10: pooling must resample questions and carry every role's delta for
    the drawn question. Stacking the per-role delta vectors instead treats two
    measurements of the same question against the same baseline as independent,
    which understates the interval — at n=750 it flipped a significance call.
    """
    present = [r for r in roles if r in runs_by_role]
    if not present:
        return float("nan"), float("nan"), float("nan"), 0
    qids = sorted(set(base["answers"]).intersection(
        *[set(runs_by_role[r]["answers"]) for r in present]))
    # rows = questions, cols = roles
    mat = np.array([[base["answers"][q][metric] - runs_by_role[r]["answers"][q][metric]
                     for r in present] for q in qids], dtype=float)
    per_q = mat.mean(axis=1)              # one number per question
    m, lo, hi = boot_ci(per_q, seed=seed)
    return 100 * m, 100 * lo, 100 * hi, len(qids)


def clustered_contrast(base, runs_by_role, group_a, group_b, metric, seed=0):
    """(mean delta over group_a) - (mean delta over group_b), clustered by question."""
    a = [r for r in group_a if r in runs_by_role]
    b = [r for r in group_b if r in runs_by_role]
    if not a or not b:
        return float("nan"), float("nan"), float("nan"), 0
    qids = sorted(set(base["answers"]).intersection(
        *[set(runs_by_role[r]["answers"]) for r in a + b]))
    def col(roles):
        return np.array([[base["answers"][q][metric] - runs_by_role[r]["answers"][q][metric]
                          for r in roles] for q in qids], dtype=float).mean(axis=1)
    per_q = col(a) - col(b)
    m, lo, hi = boot_ci(per_q, seed=seed)
    return 100 * m, 100 * lo, 100 * hi, len(qids)


def spearman(x, y):
    """Spearman rho with an EXACT permutation p-value for small k.

    scipy's asymptotic p-value is not usable at k=4: a perfect correlation makes
    its t-statistic diverge and it reports p=0.000, which is impossible. With
    only 4! = 24 orderings the smallest attainable two-sided p is 2/24 = 0.083,
    so a perfect rank agreement over four roles can NEVER be significant at 0.05.
    Enumerate instead, and report rho as the descriptive quantity it is.
    """
    import math
    from itertools import permutations
    from scipy.stats import spearmanr

    rho = float(spearmanr(x, y).statistic)
    k = len(x)
    if k <= 8:
        hits = sum(1 for p in permutations(range(k))
                   if abs(float(spearmanr(list(p), y).statistic)) >= abs(rho) - 1e-12)
        return rho, hits / math.factorial(k)
    return rho, float(spearmanr(x, y).pvalue)


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. Returns a reject/accept list in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            out[i] = True
        else:
            break
    return out


def parse_rate(calls, stage):
    c = collections.Counter(r["parse_status"] for r in calls if r["stage"] == stage)
    total = sum(c.values())
    if not total:
        return None, 0
    return 100.0 * c["ok"] / total, total


def deduped_mb(meta):
    """SPEC §5d. Recomputed so runs predating the field still report it."""
    default = meta.get("model_id")
    seen = {}
    for s in (meta.get("stages") or {}).values():
        fp = s.get("weight_footprint_mb")
        if fp:
            seen[(s.get("model_id", default), s.get("precision"))] = fp
    return sum(seen.values()) if seen else None


# ----------------------------------------------------------------- report ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--figures", default=None, help="directory to write figures into")
    ap.add_argument("--gate3", action="store_true",
                    help="print the SPEC §14 comparison block")
    args = ap.parse_args()

    found = discover(args.results, args.n, args.seed)
    if "baseline" not in found:
        raise SystemExit(f"no baseline at n={args.n} seed={args.seed} in {args.results}")

    base = load_run(found["baseline"])
    runs = {rid: load_run(p) for rid, p in found.items() if rid != "baseline"}

    # role -> {treatment -> run}
    by_role = collections.defaultdict(dict)
    for rid, run in runs.items():
        role, treat = split_run_id(rid)
        if role:
            by_role[role][treat] = run
    treatments = sorted({t for d in by_role.values() for t in d},
                        key=lambda t: list(SUFFIX_TREATMENT.values()).index(t))

    m = base["meta"]
    print("=" * 78)
    print(f"ANALYSIS — n={args.n} seed={args.seed}   (SPEC §10)")
    print("=" * 78)
    print(f"model      : {m.get('model_id')}")
    print(f"prompts    : {m.get('prompt_version')}   batch {m.get('batch_size')}   "
          f"gpu {m.get('gpu_name')}")
    print(f"commit     : {(m.get('git_commit') or '')[:8]}")
    print(f"treatments : {', '.join(treatments) or '(none found)'}")

    b_em, b_lo, b_hi = boot_ci([a["em"] for a in base["answers"].values()])
    b_f1, _, _ = boot_ci([a["f1"] for a in base["answers"].values()])
    print(f"\nBASELINE   : EM {100*b_em:.2f}% [{100*b_lo:.2f}, {100*b_hi:.2f}]   "
          f"F1 {100*b_f1:.2f}%   n={len(base['answers'])}")

    # ---- Table 1 ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("TABLE 1 — per-run summary")
    print("-" * 78)
    print(f"{'run':<18}{'EM':>7}{'F1':>7}{'dedup MB':>10}{'cores MB':>10}  batch")
    def row(label, run):
        em = 100 * np.mean([a["em"] for a in run["answers"].values()])
        f1 = 100 * np.mean([a["f1"] for a in run["answers"].values()])
        d = deduped_mb(run["meta"]); c = run["meta"].get("coresident_footprint_mb")
        print(f"{label:<18}{em:>6.1f}%{f1:>6.1f}%"
              f"{(d if d else float('nan')):>10.1f}{(c if c else float('nan')):>10.1f}"
              f"  {run['meta'].get('batch_size')}")
    row("baseline", base)
    for rid in sorted(runs):
        row(rid, runs[rid])

    # ---- Table 2: Phase H ---------------------------------------------------
    print("\n" + "-" * 78)
    print("TABLE 2 — PHASE H: cost of degrading each agent (paired bootstrap 95% CI)")
    print("  positive = that treatment HURT accuracy.  DESCRIPTIVE — see §5f: these")
    print("  are not corrected for multiplicity and must not be called significant.")
    print("-" * 78)
    hdr = f"{'role':<14}"
    for t in treatments:
        hdr += f"{t + ' EM drop':<26}"
    print(hdr + "ratio")
    ranking = {t: [] for t in treatments}
    for role in ROLES:
        if role not in by_role:
            continue
        line = f"{ROLE_LABEL[role]:<14}"
        pts, resolved = {}, {}
        for t in treatments:
            if t not in by_role[role]:
                line += f"{'—':<26}"
                continue
            d, lo, hi, _ = paired_delta(base, by_role[role][t], "em")
            pts[t] = d
            resolved[t] = (lo > 0) or (hi < 0)
            ranking[t].append((role, d))
            line += f"{d:+6.2f} [{lo:+6.2f},{hi:+6.2f}]   "
        # A ratio of two costs is only interpretable when BOTH are resolved and
        # both point the same way. Planner's two nulls (-0.17 and -0.90) would
        # otherwise print a meaningless "5.4x".
        if ("0.5B" in pts and "4-bit" in pts
                and resolved.get("0.5B") and resolved.get("4-bit")
                and pts["0.5B"] > 0 and pts["4-bit"] > 0):
            line += f"{pts['0.5B'] / pts['4-bit']:.1f}x"
        elif "0.5B" in pts and "4-bit" in pts:
            line += "n/a (null)"
        print(line)

    # ---- Table 3: rankings --------------------------------------------------
    print("\n" + "-" * 78)
    print("TABLE 3 — ranking by sensitivity, and agreement between axes")
    print("-" * 78)
    orders = {}
    for t in treatments:
        if not ranking[t]:
            continue
        ordered = [r for r, _ in sorted(ranking[t], key=lambda x: -x[1])]
        orders[t] = ordered
        print(f"  {t:<8} {' > '.join(ROLE_LABEL[r] for r in ordered)}")
    if "4-bit" in orders and "0.5B" in orders:
        common = [r for r in orders["4-bit"] if r in orders["0.5B"]]
        if len(common) >= 3:
            rq = [orders["4-bit"].index(r) for r in common]
            rs = [orders["0.5B"].index(r) for r in common]
            rho, p = spearman(rq, rs)
            print(f"\n  Spearman(quantize, shrink) = {rho:+.2f}  (p={p:.3f}, k={len(common)})")
            print("  §5b prediction 6 said rho < +0.60 -> "
                  f"{'HOLDS' if rho < 0.60 else 'REFUTED'}")
    print("\n  MA-RAG (related work, NOT a control arm — §1): their HotpotQA size")
    print("  ablation is flat, drops 1.5/1.3/1.0/0.8 pp for Planner/Extractor/QA/")
    print("  StepDef, all within 0.7 pp and without CIs. The 'QA hurts most'")
    print("  ordering often quoted is their 2WikiMQA column, a different dataset.")

    # ---- PRIMARY TEST -------------------------------------------------------
    print("\n" + "=" * 78)
    print("PRIMARY TEST (§5f) — format-heavy vs knowledge-heavy, question-clustered")
    print("  The ONE confirmatory comparison. Pre-registered 2026-07-29 as §5b")
    print("  prediction 3, before any confirmatory data existed.")
    print("=" * 78)
    for t in treatments:
        rr = {r: by_role[r][t] for r in by_role if t in by_role[r]}
        if len(rr) < 4:
            print(f"  {t:<8} incomplete ({len(rr)}/4 roles) — contrast not computed")
            continue
        fa, flo, fhi, nq = clustered_group_delta(base, rr, FORMAT_HEAVY, "em")
        ka, klo, khi, _ = clustered_group_delta(base, rr, KNOWLEDGE_HEAVY, "em")
        c, clo, chi, _ = clustered_contrast(base, rr, FORMAT_HEAVY, KNOWLEDGE_HEAVY, "em")
        excl = (clo > 0) or (chi < 0)
        print(f"  {t}:  format-heavy {fa:+.2f} [{flo:+.2f},{fhi:+.2f}]   "
              f"knowledge-heavy {ka:+.2f} [{klo:+.2f},{khi:+.2f}]")
        print(f"    CONTRAST {c:+.2f} pp [{clo:+.2f}, {chi:+.2f}]  n={nq}  "
              f"-> {'EXCLUDES ZERO' if excl else 'includes zero'}")

    # ---- parse-failure ------------------------------------------------------
    print("\n" + "-" * 78)
    print("TABLE 4 — parse success of the degraded stage (§5b predictions 1 and 5)")
    print("-" * 78)
    print(f"{'role':<14}{'baseline':>10}" + "".join(f"{t:>10}" for t in treatments))
    for role in ROLES:
        if role not in by_role:
            continue
        br, _ = parse_rate(base["calls"], role)
        line = f"{ROLE_LABEL[role]:<14}{br:>9.1f}%"
        for t in treatments:
            if t in by_role[role]:
                r, _ = parse_rate(by_role[role][t]["calls"], role)
                line += f"{r:>9.1f}%"
            else:
                line += f"{'—':>10}"
        print(line)
    print("\n  prediction 1: quantization does NOT damage format")
    print("  prediction 5: shrinking DOES, on >=2 of 4 roles")

    if args.figures:
        make_figures(args.figures, base, by_role, treatments, args.n)

    if args.gate3:
        print("\n" + "=" * 78)
        print("GATE 3 — compare against SPEC §14 (n=750). Any mismatch beyond the")
        print("  documented Extractor lower-bound shift is a bug in this file.")
        print("=" * 78)
        print("  §14 expects, on EM:")
        print("    Extractor +3.20 [+0.53, +5.87] | QA +1.73 [-0.13, +3.60]")
        print("    Step Definer +0.53 [-1.60, +2.67] | Planner -1.73 [-4.53, +1.07]")


def make_figures(outdir, base, by_role, treatments, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    roles = [r for r in ROLES if r in by_role]
    x = np.arange(len(roles))
    w = 0.8 / max(len(treatments), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, t in enumerate(treatments):
        vals, errs = [], [[], []]
        for r in roles:
            if t in by_role[r]:
                d, lo, hi, _ = paired_delta(base, by_role[r][t], "em")
                vals.append(d); errs[0].append(d - lo); errs[1].append(hi - d)
            else:
                vals.append(np.nan); errs[0].append(0); errs[1].append(0)
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=3, label=t)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([ROLE_LABEL[r] for r in roles])
    ax.set_ylabel("EM drop from baseline (pp)")
    ax.set_title(f"Cost of degrading one agent (n={n})\npositive = hurt accuracy")
    ax.legend(title="treatment"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_em_drop.png"), dpi=180)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, t in enumerate(treatments):
        vals = []
        for r in roles:
            v, _ = parse_rate(by_role[r][t]["calls"], r) if t in by_role[r] else (np.nan, 0)
            vals.append(v)
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=t)
    bl = [parse_rate(base["calls"], r)[0] for r in roles]
    ax.plot(x, bl, "k--o", label="baseline (fp16 1.5B)")
    ax.set_xticks(x); ax.set_xticklabels([ROLE_LABEL[r] for r in roles])
    ax.set_ylabel("parse success (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Output format under each treatment (n={n})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_parse.png"), dpi=180)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    b_em = 100 * np.mean([a["em"] for a in base["answers"].values()])
    ax.scatter([deduped_mb(base["meta"])], [b_em], marker="*", s=220,
               c="k", label="baseline (all fp16 1.5B)", zorder=5)
    style = {"4-bit": ("o", "tab:blue"), "8-bit": ("s", "tab:green"),
             "0.5B": ("^", "tab:red")}
    for t in treatments:
        xs, ys = [], []
        for r in roles:
            if t not in by_role[r]:
                continue
            run = by_role[r][t]
            d = deduped_mb(run["meta"])
            if d:
                xs.append(d)
                ys.append(100 * np.mean([a["em"] for a in run["answers"].values()]))
        mk, c = style.get(t, ("D", "gray"))
        ax.scatter(xs, ys, marker=mk, c=c, s=60, label=t)
    ax.set_xlabel("deduplicated weight footprint (MB) — §5d")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Accuracy vs memory\n(one agent degraded per point)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_accuracy_vs_memory.png"), dpi=180)

    print(f"\n  figures -> {outdir}/fig1_em_drop.png, fig2_parse.png, "
          f"fig3_accuracy_vs_memory.png")


if __name__ == "__main__":
    main()
