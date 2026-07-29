"""Gate 1 smoke test (SPEC §11 step 1-2, §11a).

10 questions, FP16, all four agents, locally. Prints every agent's raw output
verbatim for manual inspection, then a summary: parse counts per agent broken
down by failure type, the 10 answers beside gold, and measured throughput with
the extrapolated GPU-hours for one 300-question run.

    python smoke_test.py                     # FP16, 10 questions
    python smoke_test.py --precision 4bit    # §5a fallback if the 4 GB card OOMs

This file is scaffolding for Gate 1, not the sweep driver. src/runner.py is
build step 4.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src import models, prompts
from src.pipeline import load_questions, run_question_naive

RESULTS = Path(__file__).parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="fp16", choices=models.PRECISIONS)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    run_id = f"smoke_{args.precision}"

    print(f"=== SMOKE TEST: {args.model_id} @ {args.precision}, "
          f"n={args.n}, seed={args.seed}, prompts={prompts.PROMPT_VERSION} ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        print("WARNING: no CUDA device visible; this will be unusably slow on CPU.")

    print("\nLoading HotpotQA (distractor, dev)...")
    questions = load_questions(args.n, seed=args.seed)
    print(f"Loaded {len(questions)} questions.")

    print(f"\nLoading model at {args.precision}...")
    t_load = time.perf_counter()
    model, tok = models.load_model(args.model_id, args.precision)
    print(f"Loaded in {time.perf_counter() - t_load:.1f}s. "
          f"weight_footprint_mb={models.weight_footprint_mb(model, args.precision):.0f}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_records, all_answers = [], []
    failed_at = None
    t0 = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        print(f"\n{'=' * 78}\nQUESTION {i}/{len(questions)}  qid={q['question_id']}"
              f"  level={q.get('level')}  type={q.get('type')}"
              f"\nQ: {q['question']}\nGOLD: {q['answer']}\n{'=' * 78}")
        try:
            recs, ans = run_question_naive(model, tok, q, run_id, args.precision, verbose=True)
        except torch.cuda.OutOfMemoryError:
            print(f"\n!!! CUDA OOM on question {i}. "
                  f"Per SPEC §5a, re-run with --precision 4bit. !!!")
            failed_at = i
            break
        except Exception as e:  # noqa: BLE001 - the gate report needs the failure point
            print(f"\n!!! {type(e).__name__} on question {i}: {e}")
            failed_at = i
            break
        all_records.extend(recs)
        all_answers.append(ans)
        print(f"\n>>> PREDICTED: {ans['predicted_answer']!r}  |  GOLD: {ans['gold_answer']!r}"
              f"  |  EM={ans['em']:.0f} F1={ans['f1']:.2f}")
    wall = time.perf_counter() - t0

    # ---------------------------------------------------------------- summary
    print(f"\n\n{'#' * 78}\n# GATE 1 SUMMARY\n{'#' * 78}")
    completed = len(all_answers)
    if failed_at is None:
        print(f"\nCompleted end-to-end: {completed}/{len(questions)} questions.")
    else:
        print(f"\nBROKE at question {failed_at}. Completed {completed}/{len(questions)}.")

    print("\n--- Parse status per agent ---")
    by_role = defaultdict(Counter)
    for r in all_records:
        by_role[r["stage"]][r["parse_status"]] += 1
    print(f"{'role':<14} {'calls':>6} {'ok':>6} {'ok%':>7}   breakdown")
    for role in prompts.ROLES:
        c = by_role[role]
        total = sum(c.values())
        if not total:
            print(f"{role:<14} {0:>6}      -       -   (no calls)")
            continue
        rest = {k: v for k, v in sorted(c.items()) if k != "ok"}
        print(f"{role:<14} {total:>6} {c['ok']:>6} {100 * c['ok'] / total:>6.1f}%   "
              f"{rest if rest else '-'}")

    print("\n--- Answers vs gold ---")
    print(f"{'#':<3} {'EM':<3} {'F1':<5} {'predicted':<34} gold")
    for i, a in enumerate(all_answers, start=1):
        print(f"{i:<3} {a['em']:<3.0f} {a['f1']:<5.2f} "
              f"{(a['predicted_answer'] or '(empty)')[:32]:<34} {a['gold_answer']}")
    if all_answers:
        em = sum(a["em"] for a in all_answers) / len(all_answers)
        f1 = sum(a["f1"] for a in all_answers) / len(all_answers)
        print(f"\nEM = {100 * em:.1f}%   F1 = {100 * f1:.1f}%   (n={len(all_answers)}, "
              f"no CIs at this n — see SPEC §5)")

    print("\n--- Throughput ---")
    n_calls = len(all_records)
    if n_calls and completed:
        calls_per_q = n_calls / completed
        s_per_call = wall / n_calls
        gpu_hours_300 = (wall / completed) * 300 / 3600
        print(f"wall: {wall:.1f}s for {completed} questions, {n_calls} agent calls")
        print(f"calls/question: {calls_per_q:.1f}   s/agent call: {s_per_call:.2f}")
        for role in prompts.ROLES:
            lat = [r["latency_s"] for r in all_records if r["stage"] == role]
            if lat:
                print(f"    {role:<14} n={len(lat):<4} mean {sum(lat) / len(lat):.2f}s")
        print(f"extrapolated GPU-hours for one 300-question run: {gpu_hours_300:.2f}"
              f"  (batch_size=1, unbatched — an upper bound; SPEC §6 batching is build step 6)")
    if torch.cuda.is_available():
        print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1024**2:.0f}")

    out = RESULTS / f"{run_id}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for a in all_answers:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records) + len(all_answers)} records to {out}")
    return 0 if failed_at is None else 1


if __name__ == "__main__":
    sys.exit(main())
