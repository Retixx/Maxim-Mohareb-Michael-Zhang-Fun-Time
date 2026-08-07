"""Measured top-10 gold recall of the SHIPPED retrieval policy vs the baseline.

Reads the per-call JSONL emitted by smoke_local.py. No GPU, no model, no repo
mutation. This is the number the anchored-fusion design was gated on.

WHY IT MATTERS
--------------
An earlier probe (scripts/bridge_recall.py) reported hidden-bridge all-gold
recall rising 0.520 -> 0.678 under two-hop retrieval. That probe queried BM25
with the ORIGINAL QUESTION at hop 1 and a regex-derived entity at hop 2. The
shipped pipeline instead queries with the Step Definer's generated task text
(src/pipeline.py: search_titles(task["task"], k)), while the single-call control
queries with the original question (search_titles(q["question"], k)). Those are
different retrievers, so the earlier number never described the running system.
This recomputes recall from what the pipeline actually retrieved.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent


def load(name):
    return [json.loads(line) for line in (SCRATCH / name).open(encoding="utf-8")]


def titles_by_question(records):
    """Union of every passage title the arm actually retrieved, across all steps."""
    out = defaultdict(set)
    for rec in records:
        event = ((rec.get("consumer_input") or {}).get("retrieval") or {})
        for title in event.get("titles") or ():
            out[rec["question_id"]].add(title)
    return out


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-1.5B-Instruct_fp16"
    multi = load(f"local_smoke_multi_{tag}.jsonl")
    single = load(f"local_smoke_single_{tag}.jsonl")

    answers = {r["question_id"]: r for r in multi if r.get("record_type") == "answer"}
    gold = {q: {t for t, _ in (a.get("evidence_gold_labels") or [])}
            for q, a in answers.items()}
    stratum = {q: a["retrieval_stratum"] for q, a in answers.items()}
    got = {"single": titles_by_question(single), "multi": titles_by_question(multi)}

    def recall(arm, sel):
        qs = [q for q in gold if gold[q] and (sel == "ALL" or stratum[q] == sel)]
        partial = sum(len(gold[q] & got[arm][q]) / len(gold[q]) for q in qs) / len(qs)
        complete = sum(1 for q in qs if gold[q] <= got[arm][q]) / len(qs)
        return partial, complete, len(qs)

    print(f"{'stratum':<16}{'n':>4}{'SINGLE':>9}{'MULTI':>9}{'delta':>9}"
          f"{'  SINGLE all':>13}{'MULTI all':>11}{'delta':>9}")
    print("-" * 80)
    for sel in ("hidden_bridge", "fully_named", "ALL"):
        sp, sc, n = recall("single", sel)
        mp, mc, _ = recall("multi", sel)
        print(f"{sel:<16}{n:>4}{sp:>9.3f}{mp:>9.3f}{mp - sp:>+9.3f}"
              f"{sc:>13.3f}{mc:>11.3f}{mc - sc:>+9.3f}")

    for arm in ("single", "multi"):
        mean = sum(len(v) for v in got[arm].values()) / len(got[arm])
        print(f"\nmean unique passages read per question, {arm}: {mean:.1f}")

    # One retrieval query per (question, step): the event is duplicated onto every
    # Extractor record for that step, so dedupe before judging query diversity.
    per_step = defaultdict(dict)
    for rec in multi:
        event = ((rec.get("consumer_input") or {}).get("retrieval") or {})
        if event.get("query") and event.get("titles"):
            stage = rec.get("stage", "")
            step = 2 if "step2" in stage else 3 if "step3" in stage else 1
            per_step[rec["question_id"]][step] = event["query"]
    multi_step = [q for q in per_step if len(per_step[q]) > 1]
    identical = sum(1 for q in multi_step if len(set(per_step[q].values())) == 1)
    print(f"\nquestions executing >1 retrieval step: {len(multi_step)}")
    print(f"  every step issued an identical query: {identical} "
          f"({identical / max(len(multi_step), 1):.0%})")

    print("\nsample step-2 queries (the hop that must resolve the bridge):")
    shown = 0
    for q in multi_step:
        if 2 in per_step[q] and shown < 6:
            print(f"  [{stratum[q]}]")
            print(f"    step1: {per_step[q][1][:100]}")
            print(f"    step2: {per_step[q][2][:100]}")
            shown += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
