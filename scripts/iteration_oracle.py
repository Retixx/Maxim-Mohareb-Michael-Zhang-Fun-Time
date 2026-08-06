"""Oracle ceiling for iterative retrieval. CPU only. Run BEFORE building anything.

The previous check showed decomposition retrieves WORSE than a single query,
because SPEC §4 makes the pipeline non-iterative: the Planner writes hop-2 before
any evidence exists, so "Which university did *that director* attend?" is a query
containing a pronoun and retrieves nothing.

The proposed fix is a two-pass pipeline that rewrites hop-2 after hop-1 returns.
Before building that, measure its CEILING: give hop-2 PERFECT resolution by
splicing in the gold hop-1 entity, and see whether retrieval then beats a single
query. A real refiner can only do worse than this oracle.

    SINGLE     one query, the original multi-hop question
    NAIVE      per-sub-question queries as the current planner writes them
    ORACLE     hop-1 query as written; hop-2 query = sub-question 2 with the
               gold hop-1 title spliced in (perfect entity resolution)

If ORACLE does not clear SINGLE by a useful margin, iteration cannot rescue this
design and the build should not happen.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_TOKEN = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--plans", default="results/baseline_qwen2.5-1.5b_n3000_seed7.jsonl")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    from datasets import load_dataset
    from src.retrieval import BM25Index, build_corpus

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    passages = build_corpus(ds)
    titles = [p.title for p in passages]
    print(f"corpus {len(titles):,} passages; indexing...", flush=True)
    _index = BM25Index(passages)

    class _Shim:
        """rank_bm25-compatible surface over the fast sparse index."""

        def get_top_n(self, toks, _titles, n):
            return _index.search_titles(" ".join(toks), n)

    bm25 = _Shim()
    print("indexed\n")

    plans = {}
    for line in Path(args.plans).read_text(encoding="utf-8").splitlines():
        if '"planner"' not in line:
            continue
        r = json.loads(line)
        if r.get("stage") == "planner":
            got = (r.get("parsed") or {}).get("sub_questions")
            if got:
                plans[r["question_id"]] = got[:3]

    gold = {r["id"]: list(dict.fromkeys(r["supporting_facts"]["title"])) for r in ds}
    qtext = {r["id"]: r["question"] for r in ds}
    qids = [q for q in qtext if q in plans and len(gold[q]) >= 2][: args.n]
    print(f"evaluating {len(qids):,} two-hop questions\n")

    acc = defaultdict(lambda: defaultdict(float))
    for i, qid in enumerate(qids):
        if i and i % 200 == 0:
            print(f"  {i}/{len(qids)}", flush=True)
        g = set(gold[qid])
        subs = plans[qid]
        hop1_gold, hop2_gold = gold[qid][0], gold[qid][1]

        single = bm25.get_top_n(tok(qtext[qid]), titles, n=max(ks))
        naive = [bm25.get_top_n(tok(s), titles, n=max(ks)) for s in subs]
        # ORACLE: hop-2 query gets the gold hop-1 entity spliced in.
        q2 = subs[1] if len(subs) > 1 else qtext[qid]
        oracle_q2 = f"{hop1_gold} {q2}"
        oracle = [
            bm25.get_top_n(tok(subs[0]), titles, n=max(ks)),
            bm25.get_top_n(tok(oracle_q2), titles, n=max(ks)),
        ]

        for k in ks:
            per = max(1, k // max(len(naive), 1))
            s = set(single[:k])
            nv = set().union(*[set(r[:per]) for r in naive])
            orc = set(oracle[0][: k // 2]) | set(oracle[1][: k // 2])
            for name, got in (("single", s), ("naive", nv), ("oracle", orc)):
                acc[k][name] += len(g & got) / len(g)
                acc[k][name + "_both"] += float(g <= got)
            # did the oracle hop-2 query find the hop-2 gold specifically?
            acc[k]["oracle_hop2"] += float(hop2_gold in set(oracle[1][: k // 2]))
            acc[k]["naive_hop2"] += float(
                hop2_gold in set().union(*[set(r[:per]) for r in naive[1:]] or [set()])
            )

    n = len(qids)
    print(f"\n{'k':>4} {'SINGLE':>17} {'NAIVE multi':>17} {'ORACLE iterative':>19}")
    print(f"{'':>4} {'recall  both':>17} {'recall  both':>17} {'recall  both':>19}")
    for k in ks:
        a = acc[k]
        print(f"{k:>4} {a['single']/n:>8.3f}{a['single_both']/n:>9.3f} "
              f"{a['naive']/n:>8.3f}{a['naive_both']/n:>9.3f} "
              f"{a['oracle']/n:>10.3f}{a['oracle_both']/n:>9.3f}")

    print(f"\nHOP-2 gold found by the hop-2 query alone:")
    for k in ks:
        a = acc[k]
        print(f"  k={k:<3} naive {a['naive_hop2']/n:.3f}   oracle {a['oracle_hop2']/n:.3f}   "
              f"gain {a['oracle_hop2']/n - a['naive_hop2']/n:+.3f}")

    print("\n--- DECISION ---")
    for k in ks:
        a = acc[k]
        s, o = a["single"] / n, a["oracle"] / n
        sb, ob = a["single_both"] / n, a["oracle_both"] / n
        verdict = "ORACLE BEATS SINGLE" if o > s else "oracle loses"
        print(f"  k={k:<3} recall {s:.3f} -> {o:.3f} ({o-s:+.3f}) | "
              f"both-gold {sb:.3f} -> {ob:.3f} ({ob-sb:+.3f})  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
