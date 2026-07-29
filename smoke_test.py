"""Smoke test / gate reporter.

Runs a small sweep through src.runner, then prints the Gate-style report over the
resulting JSONL: every agent's raw output, parse counts per agent broken down by
failure type, answers beside gold with EM, and measured throughput.

    python smoke_test.py                                 # baseline, n=10, dev seed
    python smoke_test.py --run qa_4bit --n 10
    python smoke_test.py --report-only --run baseline    # re-print, no GPU work

Prompt development must use --seed 0 or 1234, never the config's eval_seed
(see the seed-hygiene note in src/prompts.py).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src import prompts
from src.runner import run as run_sweep

ROOT = Path(__file__).parent
DEV_SEED = 1234


def report(records: list[dict], meta: dict, n_raw: int) -> None:
    calls = [r for r in records if r.get("record_type") != "answer"]
    answers = [r for r in records if r.get("record_type") == "answer"]

    print(f"\n{'#' * 78}\n# SMOKE REPORT — run={meta.get('run_id')} "
          f"model={meta.get('model_id')} prompts={meta.get('prompt_version')}\n{'#' * 78}")
    print(f"stage precision: {meta.get('stage_precision')}")

    print(f"\n--- Verbatim raw outputs ({n_raw} per agent) ---")
    for role in prompts.ROLES:
        rs = [r for r in calls if r["stage"] == role]
        print(f"\n=== {role.upper()} ===")
        for r in rs[:n_raw]:
            print(f"[qid={r['question_id']} call_index={r['call_index']} "
                  f"status={r['parse_status']}]")
            print(r["raw_output"])

    print("\n--- Parse status per agent ---")
    by_role = defaultdict(Counter)
    for r in calls:
        by_role[r["stage"]][r["parse_status"]] += 1
    print(f"{'role':<14} {'calls':>6} {'ok':>6} {'ok%':>7}   breakdown")
    for role in prompts.ROLES:
        c = by_role[role]
        total = sum(c.values())
        if not total:
            print(f"{role:<14} {0:>6}      -       -   (no calls)")
            continue
        rest = {k: v for k, v in sorted(c.items()) if k != "ok"}
        flag = "" if 100 * c["ok"] / total >= 90 else "   <-- BELOW 90% (SPEC §5a)"
        print(f"{role:<14} {total:>6} {c['ok']:>6} {100 * c['ok'] / total:>6.1f}%   "
              f"{rest if rest else '-'}{flag}")

    print("\n--- Answers vs gold ---")
    print(f"{'#':<3} {'EM':<3} {'F1':<5} {'predicted':<34} gold")
    for i, a in enumerate(answers, start=1):
        print(f"{i:<3} {a['em']:<3.0f} {a['f1']:<5.2f} "
              f"{(a['predicted_answer'] or '(empty)')[:32]:<34} {a['gold_answer'][:40]}")
    if answers:
        em = sum(a["em"] for a in answers) / len(answers)
        f1 = sum(a["f1"] for a in answers) / len(answers)
        print(f"\nEM = {100 * em:.1f}%   F1 = {100 * f1:.1f}%   (n={len(answers)})")

    print("\n--- Throughput & memory ---")
    for role in prompts.ROLES:
        lat = [r["latency_s"] for r in calls if r["stage"] == role]
        st = (meta.get("stages") or {}).get(role, {})
        if lat:
            print(f"    {role:<14} n={len(lat):<4} mean {sum(lat) / len(lat):.2f}s   "
                  f"peak_vram_mb={st.get('peak_vram_mb')}  "
                  f"weights_mb={st.get('weight_footprint_mb')}")
    wall = meta.get("total_wall_s") or 0
    if answers and wall:
        print(f"wall {wall:.0f}s for {len(answers)} questions, {len(calls)} agent calls")
        print(f"extrapolated GPU-hours for one 300-question run: "
              f"{(wall / len(answers)) * 300 / 3600:.2f}")
    print(f"coresident_footprint_mb={meta.get('coresident_footprint_mb')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=DEV_SEED)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--raw", type=int, default=2, help="raw outputs printed per agent")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print from existing JSONL without running the GPU")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir = ROOT / cfg.get("results_dir", "results")

    if not args.report_only:
        run_sweep(cfg, args.run, args.n, args.seed, args.batch_size)

    jsonl = results_dir / f"{args.run}.jsonl"
    meta_path = results_dir / f"{args.run}.meta.json"
    if not jsonl.exists():
        print(f"no results at {jsonl}")
        return 1
    records = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    report(records, meta, args.raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
