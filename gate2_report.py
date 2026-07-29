"""GATE 2 report (SPEC §11a).

Ad-hoc by design: the spec puts analyze.py at build step 10, after model 2, so the
Gate 2 tables are produced without it. No figures here — those are step 10.

    python gate2.py <results_dir> [--n 300] [--seed 7]

Produces:
  - accuracy drop from baseline per role, with PAIRED bootstrap 95% CIs
  - parse-failure rate per role, baseline vs 4-bit
  - the resulting ranking of roles by quantization sensitivity
  - side-by-side with MA-RAG's published size-sensitivity ranking
  - automatic flags for results that look like bugs rather than findings
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\maxim\Projects\marag-precision")
from src.metrics import bootstrap_ci  # noqa: E402

# SPEC §2: run id -> the single stage quantized in that run.
QUANTIZED_STAGE = {
    "planner_4bit": "planner",
    "stepdef_4bit": "step_definer",
    "extractor_4bit": "extractor",
    "qa_4bit": "qa",
}
ROLE_LABEL = {
    "planner": "Planner",
    "step_definer": "Step Definer",
    "extractor": "Extractor",
    "qa": "QA",
}
# SPEC §11a: MA-RAG's published *size*-sensitivity ordering, most-hurt first.
MARAG_SIZE_RANKING = ["qa", "planner", "extractor", "step_definer"]
MARAG_NOTE = ("MA-RAG (size): QA hurts most -> Planner ~ Extractor -> "
              "Step Definer barely at all")

N_RESAMPLES = 10_000


def load_run(results_dir: Path, run_id: str, n: int, seed: int):
    stem = f"{run_id}_n{n}_seed{seed}"
    jsonl = results_dir / f"{stem}.jsonl"
    if not jsonl.exists():
        return None
    recs = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    answers = {r["question_id"]: r for r in recs if r.get("record_type") == "answer"}
    calls = [r for r in recs if r.get("record_type") != "answer"]
    meta_path = results_dir / f"{stem}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {"run_id": run_id, "answers": answers, "calls": calls, "meta": meta}


def paired_delta_ci(base: dict, other: dict, metric: str, seed: int = 0):
    """Paired bootstrap CI on (baseline - run) per question.

    Paired, not independent: every run scores the SAME question set, so pairing
    removes between-question variance and is both correct and far tighter than
    differencing two independent CIs.
    """
    qids = sorted(set(base) & set(other))
    deltas = [base[q][metric] - other[q][metric] for q in qids]
    if not deltas:
        return float("nan"), float("nan"), float("nan"), 0
    mean, lo, hi = bootstrap_ci(deltas, n_resamples=N_RESAMPLES, seed=seed)
    return mean, lo, hi, len(qids)


def parse_rate(calls, stage):
    c = collections.Counter(r["parse_status"] for r in calls if r["stage"] == stage)
    total = sum(c.values())
    if not total:
        return None, 0, {}
    fails = {k: v for k, v in c.items() if k != "ok"}
    return 100.0 * (total - c["ok"]) / total, total, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    base = load_run(rd, "baseline", args.n, args.seed)
    if base is None:
        raise SystemExit(f"no baseline at {rd}/baseline_n{args.n}_seed{args.seed}.jsonl")
    runs = {rid: load_run(rd, rid, args.n, args.seed) for rid in QUANTIZED_STAGE}
    missing = [r for r, v in runs.items() if v is None]
    if missing:
        print(f"!! MISSING RUNS: {missing} — report is incomplete\n")

    m = base["meta"]
    print("=" * 78)
    print("GATE 2 — one-agent-4bit tier")
    print("=" * 78)
    print(f"model        : {m.get('model_id')}")
    print(f"n / seed     : {m.get('n')} / {m.get('seed')}   prompts {m.get('prompt_version')}")
    print(f"gpu          : {m.get('gpu_name')}")
    print(f"git commit   : {(m.get('git_commit') or '')[:8]}")
    print(f"libs         : {m.get('library_versions')}")
    b_em = 100 * sum(a["em"] for a in base["answers"].values()) / len(base["answers"])
    b_f1 = 100 * sum(a["f1"] for a in base["answers"].values()) / len(base["answers"])
    _, lo, hi = bootstrap_ci([a["em"] for a in base["answers"].values()], N_RESAMPLES)
    print(f"\nbaseline     : EM {b_em:.1f}% [{100*lo:.1f}, {100*hi:.1f}]   F1 {b_f1:.1f}%   "
          f"n={len(base['answers'])}")

    # ---- Table 1: accuracy drop per role -------------------------------------
    print("\n" + "-" * 78)
    print("TABLE 1 — accuracy drop from baseline (paired bootstrap 95% CI)")
    print("           positive = quantizing that role HURT accuracy")
    print("-" * 78)
    print(f"{'role':<14} {'EM drop (pp)':<26} {'F1 drop (pp)':<26} {'run EM':>7}")
    rows = []
    for rid, stage in QUANTIZED_STAGE.items():
        r = runs[rid]
        if r is None:
            print(f"{ROLE_LABEL[stage]:<14} {'(run missing)':<26}")
            continue
        dem, dlo, dhi, npair = paired_delta_ci(base["answers"], r["answers"], "em")
        df1, flo, fhi, _ = paired_delta_ci(base["answers"], r["answers"], "f1")
        run_em = 100 * sum(a["em"] for a in r["answers"].values()) / len(r["answers"])
        print(f"{ROLE_LABEL[stage]:<14} "
              f"{100*dem:+6.2f}  [{100*dlo:+6.2f}, {100*dhi:+6.2f}]   "
              f"{100*df1:+6.2f}  [{100*flo:+6.2f}, {100*fhi:+6.2f}]   {run_em:6.1f}%")
        rows.append({
            "stage": stage, "run_id": rid, "em_drop": 100 * dem,
            "em_lo": 100 * dlo, "em_hi": 100 * dhi, "npair": npair,
            "significant": (100 * dlo > 0) or (100 * dhi < 0),
        })

    # ---- Table 2: parse-failure rate per role --------------------------------
    print("\n" + "-" * 78)
    print("TABLE 2 — parse-failure rate of each role, baseline vs when quantized")
    print("-" * 78)
    print(f"{'role':<14} {'baseline':>10} {'4-bit':>10} {'delta':>9}   4-bit failure types")
    for rid, stage in QUANTIZED_STAGE.items():
        r = runs[rid]
        b_rate, b_n, _ = parse_rate(base["calls"], stage)
        if r is None:
            print(f"{ROLE_LABEL[stage]:<14} {b_rate:9.1f}%  (run missing)")
            continue
        q_rate, q_n, q_fails = parse_rate(r["calls"], stage)
        print(f"{ROLE_LABEL[stage]:<14} {b_rate:9.1f}% {q_rate:9.1f}% "
              f"{q_rate - b_rate:+8.1f}   {q_fails or '-'}")

    # ---- Ranking -------------------------------------------------------------
    print("\n" + "-" * 78)
    print("RANKING by quantization sensitivity (largest EM drop first)")
    print("-" * 78)
    ranked = sorted(rows, key=lambda x: -x["em_drop"])
    for i, x in enumerate(ranked, 1):
        sig = "significant" if x["significant"] else "CI includes 0 — not distinguishable"
        print(f"  {i}. {ROLE_LABEL[x['stage']]:<14} {x['em_drop']:+.2f} pp   ({sig})")
    print(f"\n  ours (precision): {' -> '.join(ROLE_LABEL[x['stage']] for x in ranked)}")
    print(f"  {MARAG_NOTE}")

    sig_rows = [x for x in ranked if x["significant"]]
    if not sig_rows:
        verdict = ("INDISTINGUISHABLE — no role's drop has a CI excluding zero, so the "
                   "orderings cannot be compared. This is a real possible outcome, not a bug.")
    else:
        ours = [x["stage"] for x in ranked]
        verdict = ("MATCHES MA-RAG's size ordering" if ours == MARAG_SIZE_RANKING
                   else f"DIFFERS from MA-RAG's size ordering ({'>'.join(MARAG_SIZE_RANKING)})")
    print(f"\n  VERDICT: {verdict}")

    # ---- Bug sniffing (SPEC §11a) -------------------------------------------
    print("\n" + "-" * 78)
    print("LOOKS-LIKE-A-BUG CHECKS")
    print("-" * 78)
    flags = []
    for x in rows:
        if x["em_drop"] == 0.0:
            flags.append(f"{ROLE_LABEL[x['stage']]}: EM drop is EXACTLY 0.00 — suspect the "
                         f"quantized stage silently ran at fp16, or outputs were reused")
        if x["em_hi"] - x["em_lo"] > 60:
            flags.append(f"{ROLE_LABEL[x['stage']]}: CI spans {x['em_hi']-x['em_lo']:.0f} pp — "
                         f"far too wide to interpret")
        if x["npair"] < args.n:
            flags.append(f"{ROLE_LABEL[x['stage']]}: only {x['npair']}/{args.n} questions "
                         f"paired with baseline — run may be incomplete")
    for rid, stage in QUANTIZED_STAGE.items():
        if runs[rid] is None:
            continue
        for label, calls in (("baseline", base["calls"]), (rid, runs[rid]["calls"])):
            rate, n_calls, _ = parse_rate(calls, stage)
            if rate is None:
                continue
            if rate == 0.0:
                flags.append(f"{label}/{stage}: parse-failure rate exactly 0.0% over "
                             f"{n_calls} calls — verify the parser is actually running")
            if rate == 100.0:
                flags.append(f"{label}/{stage}: parse-failure rate exactly 100% over "
                             f"{n_calls} calls — stage is broken, not degraded")
    # Floor effect (SPEC §5a contingency)
    if rows and all(x["em_drop"] > 0 for x in rows):
        run_ems = []
        for x in rows:
            r = runs[x["run_id"]]
            run_ems.append(100 * sum(a["em"] for a in r["answers"].values()) / len(r["answers"]))
        if max(run_ems) < 5:
            flags.append("FLOOR EFFECT: every 4-bit run is under 5% EM. No ranking is "
                         "measurable — see SPEC §5a contingencies (try 8-bit tier next).")
    print("\n".join(f"  !! {f}" for f in flags) if flags else "  none — nothing looks like a bug")

    # ---- Table 2b: latency, the SPEC §10 Table 1 column I had been omitting --
    print("\n" + "-" * 78)
    print("TABLE 2b — per-call latency of the quantized stage, fp16 vs 4-bit (paired)")
    print("  CAVEAT: this is batch wall-time / batch size at the configured batch size,")
    print("  i.e. inverse THROUGHPUT, not user-facing latency. Whole-run wall times are")
    print("  NOT comparable across runs (shared-GPU session noise); only these paired")
    print("  per-call deltas are.")
    print("-" * 78)
    print(f"{'role':<14}{'fp16 s':>9}{'4bit s':>9}{'delta':>9}{'ratio':>8}   95% CI on delta")
    for rid, stage in QUANTIZED_STAGE.items():
        r = runs[rid]
        if r is None:
            continue
        b_idx = {(x["question_id"], x["call_index"]): x
                 for x in base["calls"] if x["stage"] == stage}
        q_idx = {(x["question_id"], x["call_index"]): x
                 for x in r["calls"] if x["stage"] == stage}
        keys = [k for k in b_idx if k in q_idx]
        if not keys:
            continue
        bl = [b_idx[k]["latency_s"] for k in keys]
        ql = [q_idx[k]["latency_s"] for k in keys]
        m, lo, hi = bootstrap_ci([y - x for x, y in zip(bl, ql)], N_RESAMPLES)
        mb, mq = sum(bl) / len(bl), sum(ql) / len(ql)
        flag = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"{ROLE_LABEL[stage]:<14}{mb:9.3f}{mq:9.3f}{m:+9.3f}{mq / mb:8.2f}   "
              f"[{lo:+.3f}, {hi:+.3f}]{flag}")

    # ---- Table 3: mechanism instruments (SPEC §5b) ---------------------------
    print("\n" + "-" * 78)
    print("TABLE 3 — mechanism instruments, baseline vs quantized (SPEC §5b)")
    print("  format damage was REFUTED at 4-bit on model 1; these test predictions 1-2")
    print("-" * 78)
    from src.mechanism import SELECTION_FIELD, selection_changed, strict_format_ok, verbatim_rate
    from src.pipeline import load_questions
    qmap = {q["question_id"]: q for q in load_questions(args.n, seed=args.seed)}
    print(f"{'role':<14}{'strict fmt base/4bit':>22}{'verbatim base/4bit':>22}{'selection churn':>17}")
    for rid, stage in QUANTIZED_STAGE.items():
        r = runs[rid]
        if r is None:
            continue
        b_idx = {(x["question_id"], x["call_index"]): x
                 for x in base["calls"] if x["stage"] == stage}
        q_idx = {(x["question_id"], x["call_index"]): x
                 for x in r["calls"] if x["stage"] == stage}
        keys = [k for k in b_idx if k in q_idx]
        if not keys:
            continue

        def fmt(src):
            return 100 * sum(1 for k in keys
                             if strict_format_ok(stage, src[k]["raw_output"])) / len(keys)

        def vb(src):
            if stage != "extractor":
                return None
            num = den = 0
            for k in keys:
                spans = (src[k].get("parsed") or {}).get("spans") or []
                rate = verbatim_rate(spans, qmap[k[0]]["paragraphs"])
                if rate is not None:
                    num += rate * len(spans)
                    den += len(spans)
            return 100 * num / den if den else None

        churn = 100 * sum(1 for k in keys
                          if selection_changed(b_idx[k], q_idx[k],
                                               SELECTION_FIELD[stage])) / len(keys)
        bv, qv = vb(b_idx), vb(q_idx)
        vtxt = f"{bv:9.1f} /{qv:6.1f}" if bv is not None else f"{'n/a':>16}"
        print(f"{ROLE_LABEL[stage]:<14}{fmt(b_idx):9.1f} /{fmt(q_idx):6.1f}   "
              f"{vtxt}   {churn:14.1f}%")

    # Calibration, only if the runs were generated with generation.log_confidence
    if base["meta"].get("log_confidence"):
        from src.metrics import auroc
        print("\n  CALIBRATION (SPEC §5b prediction 4) — AUROC of QA confidence vs EM")
        for label, run in [("baseline", base)] + [(k, v) for k, v in runs.items() if v]:
            qa = [x for x in run["calls"] if x["stage"] == "qa"]
            a = auroc([x.get("mean_logprob") for x in qa],
                      [run["answers"][x["question_id"]]["em"] for x in qa])
            print(f"    {label:<18} AUROC {a:.3f}   (0.5 = confidence says nothing)")
    else:
        print("\n  CALIBRATION: not measured — these runs were generated with")
        print("    generation.log_confidence = false. Enable it for the n=5000 rerun.")

    # ---- memory: the controlled constant ------------------------------------
    print("\n" + "-" * 78)
    print("MEMORY (SPEC §7: identical across runs 1-4 by construction)")
    print("-" * 78)
    print(f"  baseline coresident : {base['meta'].get('coresident_footprint_mb')} MB")
    for rid in QUANTIZED_STAGE:
        if runs[rid]:
            print(f"  {rid:<20}: {runs[rid]['meta'].get('coresident_footprint_mb')} MB")


if __name__ == "__main__":
    main()
