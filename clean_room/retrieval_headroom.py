#!/usr/bin/env python3
"""
Retrieval headroom: can a second hop possibly help on this corpus?

Imports nothing from this repository. CPU only, no GPU, no model.

WHY THIS GATES EVERYTHING
-------------------------
A multi-agent RAG system has exactly one mechanism through which it can beat
single-hop RAG on a multi-hop question: issuing a second, better-informed query
that reaches evidence the first query could not. Every other difference
(planning, extraction, per-step QA) is overhead layered on top of that one
mechanism.

So if the first query already retrieves both gold paragraphs, decomposition has
nothing to buy, and no amount of model scale changes that. The gap between
single-hop and multi-hop will stay flat from 0.5B to 70B, because the ceiling is
set by the corpus, not the reader.

That makes retrieval headroom a *corpus property*, measurable without any LLM.
Measure it before renting a GPU.

THE STRATIFICATION THAT MATTERS
-------------------------------
Whether the question NAMES the entity whose page you need.

    "Which magazine started first, Arthur's Magazine or First for Women?"
        Both gold titles appear in the question. One query finds both.
        Splitting it strictly DISCARDS lexical signal. Decomposition can only
        lose here.

    "Who is the mother of the director of the film Polish-Russian War?"
        Gold = [Polish-Russian War, Xawery Zulawski]. The director is never
        named. No single query can retrieve his page. You must read hop 1,
        learn the name, then query again.

Only the second kind -- hidden bridge -- has headroom. Reporting an unstratified
average hides the entire effect.

ARMS (equal read budget: every arm returns at most k passages)
--------------------------------------------------------------
    SINGLE  one BM25 query = the question, top k.
    ORACLE  hop 1 = question, top k/2; then splice in a query for the hidden
            gold title *by name*, top k/2. Union capped at k.

ORACLE is an upper bound: it is handed the hidden title for free, which a real
pipeline must resolve by reading. If ORACLE does not beat SINGLE, a real
pipeline certainly cannot, and the experiment is untestable on this corpus.

DECISION RULE, fixed before looking
-----------------------------------
Let H = ORACLE both-gold recall@k minus SINGLE both-gold recall@k, on the
hidden-bridge stratum.

    H < 0.05   NO-GO. No headroom. Multi-agent cannot win here at any scale.
               Fix the corpus (bigger haystack / weaker retriever), not the
               pipeline.
    H >= 0.15  GO. Real headroom. A multi-agent win is achievable and failing
               to get one is a pipeline bug worth chasing.
    otherwise  MARGINAL. Any effect will be small and needs a large n.

Usage:
    python retrieval_headroom.py --n 1000 --k 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any

SEED = 20260807
_TOKEN = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class BM25:
    """Lucene-style BM25 over an in-memory inverted index. Deterministic."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avgdl = sum(self.lens) / max(1, self.n)
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.index[term].append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.index.items()
        }

    def top_k(self, query: str, k: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for term in tok(query):
            post = self.index.get(term)
            if not post:
                continue
            idf = self.idf[term]
            for doc, tf in post:
                dl = self.lens[doc]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc] += idf * tf * (self.k1 + 1) / denom
        # deterministic tie-break on doc id
        return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def build_corpus() -> tuple[BM25, list[str], dict[str, int]]:
    """First-occurrence union of distractor and fullwiki validation passages.
    distractor MUST come first -- 402 titles appear in both configs with
    different sentence splits, and supporting_facts sent_id is defined against
    the distractor split."""
    from datasets import load_dataset

    title_to_id: dict[str, int] = {}
    titles: list[str] = []
    raw: list[str] = []

    for config in ("distractor", "fullwiki"):
        try:
            ds = load_dataset("hotpotqa/hotpot_qa", config, split="validation")
        except Exception as e:
            print(f"  ! could not load {config}: {type(e).__name__}: {e}")
            continue
        for r in ds:
            ctx = r["context"]
            for t, sents in zip(ctx["title"], ctx["sentences"]):
                if t not in title_to_id:
                    title_to_id[t] = len(titles)
                    titles.append(t)
                    raw.append(t + " " + " ".join(sents))
        print(f"  after {config:11s}: {len(titles)} unique passages")

    print("  tokenizing and indexing ...")
    return BM25([tok(x) for x in raw]), titles, title_to_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="headroom.json")
    args = ap.parse_args()

    from datasets import load_dataset

    print("building pooled corpus ...")
    bm25, titles, title_to_id = build_corpus()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    idx = idx[: args.n]

    k, half = args.k, max(1, args.k // 2)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for count, i in enumerate(idx):
        r = ds[i]
        gold = sorted({t for t in r["supporting_facts"]["title"]})
        gold_ids = [title_to_id[t] for t in gold if t in title_to_id]
        if len(gold_ids) != len(gold) or not gold:
            skipped += 1
            continue

        q = r["question"]
        qtoks = set(tok(q))
        # A gold title is "hidden" when its content words are not in the question.
        hidden = [t for t in gold if not set(tok(t)) <= qtoks]
        stratum = "hidden_bridge" if hidden else "fully_named"

        single = set(bm25.top_k(q, k))

        oracle = set(bm25.top_k(q, half))
        for t in (hidden or gold):
            oracle |= set(bm25.top_k(t + " " + q, half))
        oracle = set(sorted(oracle)[:k]) if len(oracle) > k else oracle

        strata[stratum].append({
            "single_both": all(g in single for g in gold_ids),
            "oracle_both": all(g in oracle for g in gold_ids),
            "single_any": any(g in single for g in gold_ids),
        })

        if (count + 1) % 200 == 0:
            print(f"  {count + 1}/{len(idx)} ...")

    report: dict[str, Any] = {"k": k, "n_requested": args.n, "skipped": skipped, "strata": {}}
    print(f"\n{'stratum':16s} {'n':>6s} {'SINGLE both':>12s} {'ORACLE both':>12s} {'headroom':>10s}")
    print("-" * 62)

    for name in ("hidden_bridge", "fully_named"):
        rows = strata.get(name, [])
        if not rows:
            continue
        n = len(rows)
        s = sum(r["single_both"] for r in rows) / n
        o = sum(r["oracle_both"] for r in rows) / n
        report["strata"][name] = {
            "n": n, "single_both_recall": round(s, 4),
            "oracle_both_recall": round(o, 4), "headroom": round(o - s, 4),
            "single_any_recall": round(sum(r["single_any"] for r in rows) / n, 4),
        }
        print(f"{name:16s} {n:6d} {s:12.4f} {o:12.4f} {o - s:+10.4f}")

    hb = report["strata"].get("hidden_bridge")
    if hb:
        h = hb["headroom"]
        verdict = "NO-GO" if h < 0.05 else ("GO" if h >= 0.15 else "MARGINAL")
        report["headroom_hidden_bridge"] = h
        report["verdict"] = verdict
        print(f"\nhidden-bridge headroom = {h:+.4f}  ->  {verdict}")
        if verdict == "NO-GO":
            print(
                "\n  The first query already reaches both gold paragraphs on the\n"
                "  stratum where a second hop is supposed to matter. There is no\n"
                "  mechanism through which multi-agent RAG can win on this corpus,\n"
                "  at ANY model size. A flat single-vs-multi gap across 0.5B->14B\n"
                "  is the expected result, not evidence of a pipeline bug.\n"
                "  Fix: enlarge the haystack or weaken the retriever until a single\n"
                "  query stops finding the hidden bridge."
            )
        elif verdict == "GO":
            print(
                "\n  Real headroom exists. If the pipeline still loses to single-hop\n"
                "  here, that IS a pipeline bug and is worth chasing."
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
