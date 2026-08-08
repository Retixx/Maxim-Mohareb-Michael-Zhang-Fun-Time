#!/usr/bin/env python3
"""Gate A (SPEC §15.5): does the repaired query policy actually recover recall?

CPU-only, no GPU, no model load. This is the cheapest decisive test of the
BUG-1/2/3 repair and it must pass before any GPU time is spent.

It drives the REAL `src.retrieval` index over the REAL 72,094-passage corpus and
reproduces the exact query construction of `pipeline._retrieval_decision` under
both the pre-repair (v1) and repaired (v2) policies, so the delta is attributable
to the policy and nothing else.

Follow-up task text comes from a completed run's Planner sub-questions, the same
proxy SPEC §15.2 used to measure the defect. Those artifacts are off-lineage
(§14 BUG-7) and their SUB-QUESTION TEXT is used only as realistic query input —
no accuracy number is taken from them.

    python scripts/gate_a_recall.py --n 1000

Gate A passes when hidden_bridge both-gold recall@10 >= 0.75 (from 0.4716,
toward the 0.8863 oracle).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src import retrieval

GATE_A_FLOOR = 0.75
SINGLE_QUERY_BASELINE = 0.4716   # SPEC §15.3, hidden_bridge
ORACLE = 0.8863                  # SPEC §15.3, reachable ceiling


def _plans(path: Path) -> dict[str, list[str]]:
    """Real Planner sub-questions keyed by question id."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"planner"' not in line:
            continue
        record = json.loads(line)
        if record.get("stage") != "planner":
            continue
        subs = (record.get("parsed") or {}).get("sub_questions")
        if subs:
            out[record["question_id"]] = subs[:3]
    return out


def _union(task_titles: list[str], anchor_titles: list[str], k: int) -> list[str]:
    """Byte-for-byte the interleave in pipeline._retrieval_decision."""
    titles: list[str] = []
    for task_title, anchor_title in zip_longest(task_titles, anchor_titles):
        for candidate in (task_title, anchor_title):
            if candidate is not None and candidate not in titles:
                titles.append(candidate)
    return titles[:k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--plans",
        default="results/baseline_qwen2.5-1.5b_n3000_seed7.jsonl",
        help="completed run supplying real Planner sub-questions",
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8")
    )["retrieval"]

    print("building the pinned corpus (CPU, no model)...", flush=True)
    corpus = retrieval.build_corpus(configs=tuple(config["corpus_configs"]))
    print(f"  {len(corpus):,} passages "
          f"(config expects {config['expected_corpus_passages']:,})")
    index = retrieval.BM25Index(corpus)
    k = args.k

    from datasets import load_dataset
    rows = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    plans = _plans(ROOT / args.plans)
    print(f"real planner sub-questions for {len(plans):,} questions\n")

    stats: dict[str, dict[str, list[int]]] = {
        "hidden_bridge": {"v1": [], "v2": [], "anchor": []},
        "fully_named": {"v1": [], "v2": [], "anchor": []},
    }

    evaluated = 0
    for row in rows:
        if evaluated >= args.n:
            break
        qid = row["id"]
        subs = plans.get(qid)
        if not subs or len(subs) < 2:
            continue
        gold = set(row["supporting_facts"]["title"])
        if len(gold) < 2:
            continue

        question = row["question"]
        stratum = ("fully_named"
                   if all(retrieval.title_is_mentioned(t, question) for t in gold)
                   else "hidden_bridge")

        anchor_titles = index.search_titles(question, k)
        task = subs[1]

        # v1 (pre-repair): follow-up REPLACES the anchor and drops the question.
        v1_titles = index.search_titles(task, k)
        # v2 (repaired): question anchors the follow-up, union of both, cap k.
        v2_query = " | ".join([question, task])
        v2_titles = _union(index.search_titles(v2_query, k), anchor_titles, k)

        for label, titles in (("anchor", anchor_titles),
                              ("v1", v1_titles), ("v2", v2_titles)):
            stats[stratum][label].append(int(gold <= set(titles)))
        evaluated += 1
        if evaluated % 200 == 0:
            print(f"  {evaluated}/{args.n}", flush=True)

    print(f"\nevaluated {evaluated} questions with a >=2-step plan and 2 gold titles\n")
    print(f"{'stratum':16s} {'n':>5s} {'anchor-only':>12s} {'v1 (pre-fix)':>13s} {'v2 (repaired)':>14s}")
    for stratum, arms in stats.items():
        n = len(arms["v2"])
        if not n:
            continue
        rate = {label: sum(v) / len(v) for label, v in arms.items() if v}
        print(f"{stratum:16s} {n:5d} {rate['anchor']:12.4f} "
              f"{rate['v1']:13.4f} {rate['v2']:14.4f}")

    hb = stats["hidden_bridge"]["v2"]
    if not hb:
        print("\nNO hidden_bridge questions evaluated — cannot score Gate A")
        return 2
    achieved = sum(hb) / len(hb)
    print(f"\nGate A: hidden_bridge both-gold recall@{k} = {achieved:.4f}")
    print(f"  single-query baseline {SINGLE_QUERY_BASELINE:.4f} | "
          f"oracle {ORACLE:.4f} | floor {GATE_A_FLOOR:.2f}")
    if achieved >= GATE_A_FLOOR:
        print("  PASS")
        return 0
    print("  FAIL — do not spend GPU time until the query policy recovers recall")
    return 1


if __name__ == "__main__":
    sys.exit(main())
