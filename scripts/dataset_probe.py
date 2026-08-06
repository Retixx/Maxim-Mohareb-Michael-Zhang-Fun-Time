"""ARCHIVED PRE-DESIGN PROBE — not the production pipeline or current evidence.

This exploratory dataset comparison predates the frozen HotpotQA pooled corpus
and variable-depth MA-RAG executor.  Its claims and outputs are not part of the
final experiment.

Which dataset gives decomposition something to do? CPU only, no GPU.

HotpotQA distractor killed the multi-agent premise: its questions NAME their gold
entities, so one query carries the full lexical signal and splitting it strictly
discards information. Oracle-resolved decomposed retrieval lost to a single query
at every k, on two independent BM25 implementations.

MA-RAG evaluates on NQ, TriviaQA, HotpotQA, 2WikiMQA and FEVER. 2WikiMQA is the
one whose question types deliberately HIDE the bridge entity:

    compositional      "Who is the mother of the director of film X?"
                       gold = [X, <director>]  -- the director is never named
    inference          "Who is X's paternal grandfather?"
    bridge_comparison  four gold titles, only two named
    comparison         both named -- no headroom, same as HotpotQA

This measures, per question type:

    SINGLE  one query = the question
    ORACLE  hop-1 query = the question; hop-2 query = the gold hop-1 title
            (perfect bridge resolution -- the ceiling any refiner could reach)

and reports recall over ALL gold titles and, separately, over the HIDDEN gold
titles -- the ones absent from the question text. Hidden-title recall is the
number that matters: it is exactly what decomposition exists to reach.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT = "scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35"


def named_in(title: str, question: str) -> bool:
    """Is this gold title lexically present in the question?"""
    return title.lower().strip() in question.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--ks", default="5,10,20")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    from datasets import load_dataset

    from src.retrieval import BM25Index, Passage

    ds = load_dataset(args.dataset, split=args.split)
    print(f"{args.dataset}  n={len(ds):,}")

    # Pool every unique paragraph into one corpus, as with HotpotQA.
    seen: dict[str, str] = {}
    for row in ds:
        ctx = row["context"]
        # 2WikiMQA names the sentence list "content"; HotpotQA calls it
        # "sentences". Accept either so the probe runs on both.
        for title, sents in zip(ctx["title"], ctx.get("content") or ctx["sentences"]):
            if title not in seen:
                seen[title] = " ".join(s.strip() for s in sents).strip()
    passages = [Passage(t, x) for t, x in seen.items()]
    print(f"corpus: {len(passages):,} unique passages; indexing...", flush=True)
    index = BM25Index(passages)
    print("indexed\n")

    rows = list(ds)[: args.n]
    acc = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    for row in rows:
        q = row["question"]
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        if len(gold) < 2:
            continue
        qtype = row["type"]
        counts[qtype] += 1
        counts["ALL"] += 1

        hidden = [g for g in gold if not named_in(g, q)]
        named = [g for g in gold if named_in(g, q)]

        single = index.search_titles(q, max(ks))
        # ORACLE: one follow-up query per NAMED gold title, using that title as
        # the resolved bridge. This is the ceiling a perfect refiner reaches.
        follow = [index.search_titles(f"{t} {q}", max(ks)) for t in (named or gold[:1])]

        for k in ks:
            s = set(single[:k])
            per = max(1, k // max(len(follow) + 1, 1))
            o = set(single[:per])
            for f in follow:
                o |= set(f[:per])
            for scope, targets in (("all", gold), ("hidden", hidden)):
                if not targets:
                    continue
                tg = set(targets)
                for name, got in (("single", s), ("oracle", o)):
                    acc[(qtype, k, scope)][name] += len(tg & got) / len(tg)
                    acc[("ALL", k, scope)][name] += len(tg & got) / len(tg)
                acc[(qtype, k, scope)]["n"] += 1
                acc[("ALL", k, scope)]["n"] += 1

    print(f"{'type':<19}{'k':>4}{'scope':>8}{'n':>6}{'SINGLE':>9}{'ORACLE':>9}{'gain':>9}")
    for qtype in ["compositional", "inference", "bridge_comparison", "comparison", "ALL"]:
        for k in ks:
            for scope in ("all", "hidden"):
                a = acc.get((qtype, k, scope))
                if not a or not a["n"]:
                    continue
                n = a["n"]
                s, o = a["single"] / n, a["oracle"] / n
                mark = "  <<<" if (scope == "hidden" and o - s > 0.05) else ""
                print(f"{qtype:<19}{k:>4}{scope:>8}{int(n):>6}"
                      f"{s:>9.3f}{o:>9.3f}{o - s:>+9.3f}{mark}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
