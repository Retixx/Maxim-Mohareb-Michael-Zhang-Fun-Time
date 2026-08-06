"""Does iterative decomposition beat a single query on 2WikiMQA? CPU only.

WHY THIS DATASET
----------------
On HotpotQA distractor, decomposition lost at every k, on two independent BM25
implementations, even with oracle bridge splicing. Cause: HotpotQA questions NAME
both gold entities, so one query already carries the full lexical signal and
splitting it strictly discards information. There was no hidden entity to find.

2WikiMQA -- also one of MA-RAG's evaluation sets -- is built from Wikidata
entity->relation->entity chains, and three of its four question types leave the
bridge entity UNNAMED:

    compositional      "Who is the mother of the director of film X?"
                       evidences = (X, director, B), (B, mother, ANS)
                       B is never in the question; its page must be retrieved.
    inference          "Who is X's paternal grandfather?"
    bridge_comparison  four gold titles, two bridges, only two named
    comparison         both named -- the HotpotQA-like control

`evidences[i].entity` gives the bridge entity exactly, so the ceiling is
measurable without running a model.

ARMS (equal read budget: every arm returns at most k passages)
-------------------------------------------------------------
    SINGLE    one query = the question, top-k.
    ORACLE    hop-1 = question, top-k/2; hop-2 = gold bridge entity + relation,
              top-k/2. Bridge handed over for free. Upper bound only.
    GROUNDED  identical, EXCEPT the bridge is spliced only when it literally
              appears in the text of a passage hop-1 actually retrieved. This is
              what a perfect Extractor could recover -- a real one does worse.
              When the bridge is unrecoverable the arm falls back to top-k
              single, so it is never given free information.

GROUNDED is the number that decides the build. If it does not clear SINGLE on
the hidden-bridge types by a useful margin, iteration cannot be rescued on this
dataset either and the redesign stops here.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET = "scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35"
TYPES = ["compositional", "inference", "bridge_comparison", "comparison", "ALL"]


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def named_in(title: str, question: str) -> bool:
    """Is this gold title lexically present in the question itself?"""
    return norm(title) in norm(question)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--ks", default="10,20")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    from datasets import load_dataset

    from src.retrieval import BM25Index, Passage

    ds = load_dataset(DATASET, split="validation")

    seen: dict[str, str] = {}
    for row in ds:
        ctx = row["context"]
        for title, sents in zip(ctx["title"], ctx["content"]):
            if title not in seen:
                seen[title] = " ".join(s.strip() for s in sents).strip()
    passages = [Passage(t, x) for t, x in seen.items()]
    text_of = {p.title: norm(p.text) for p in passages}
    print(f"corpus: {len(passages):,} unique passages; indexing...", flush=True)
    index = BM25Index(passages)
    print("indexed\n", flush=True)

    acc: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = list(ds)[: args.n]

    for i, row in enumerate(rows):
        if i and i % 250 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

        q = row["question"]
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        if len(gold) < 2:
            continue
        qtype = row["type"]
        gset = set(gold)

        ev = row["evidences"]
        # A bridge is a hop-1 object entity that the question does not name but
        # whose page is gold. Those are exactly the titles decomposition exists
        # to reach.
        bridges = []
        for fact, rel, ent in zip(ev["fact"], ev["relation"], ev["entity"]):
            if named_in(ent, q):
                continue
            match = next((g for g in gold if norm(g).startswith(norm(ent))), None)
            if match and match not in bridges:
                bridges.append((match, ent, rel))
        hidden = {g for g in gold if not named_in(g, q)}

        single_full = index.search_titles(q, max(ks))

        for k in ks:
            half = max(1, k // 2)
            s_at_k = set(single_full[:k])

            hop1 = single_full[:half]
            hop1_text = " ".join(text_of.get(t, "") for t in hop1)

            def follow(pairs, use_relation=True):
                """Union hop-1 with one hop-2 query per resolved bridge."""
                out = set(hop1)
                if not pairs:
                    return set(single_full[:k])
                per = max(1, half // len(pairs))
                for _title, ent, rel in pairs:
                    # REALISTIC mode spends no gold relation string: a refiner
                    # only has the resolved entity and the question it started
                    # from.
                    tail = rel if use_relation else q
                    out |= set(index.search_titles(f"{ent} {tail}", per))
                return out

            oracle = follow(bridges)
            # GROUNDED: keep only bridges a reader could actually have extracted
            # from the passages hop-1 returned.
            recoverable = [b for b in bridges if norm(b[1]) in hop1_text]
            grounded = follow(recoverable)
            realistic = follow(recoverable, use_relation=False)

            for scope, targets in (("all", gset), ("hidden", hidden)):
                if not targets:
                    continue
                a = acc[(qtype, k, scope)]
                b = acc[("ALL", k, scope)]
                for key, got in (("single", s_at_k), ("oracle", oracle),
                                 ("grounded", grounded), ("realistic", realistic)):
                    v = len(targets & got) / len(targets)
                    a[key] += v
                    b[key] += v
                    a[key + "_full"] += float(targets <= got)
                    b[key + "_full"] += float(targets <= got)
                a["n"] += 1
                b["n"] += 1

            if bridges:
                for d in (acc[(qtype, k, "all")], acc[("ALL", k, "all")]):
                    d["bridged"] += 1
                    d["recoverable"] += len(recoverable) / len(bridges)

    hdr = (f"{'type':<19}{'k':>4}{'scope':>8}{'n':>6}{'SINGLE':>9}"
           f"{'REALISTIC':>11}{'gain':>8}{'GROUNDED':>10}{'ORACLE':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for qtype in TYPES:
        for k in ks:
            for scope in ("all", "hidden"):
                a = acc.get((qtype, k, scope))
                if not a or not a["n"]:
                    continue
                n = a["n"]
                s, r = a["single"] / n, a["realistic"] / n
                g, o = a["grounded"] / n, a["oracle"] / n
                mark = "  <<<" if scope == "hidden" and r - s > 0.05 else ""
                print(f"{qtype:<19}{k:>4}{scope:>8}{int(n):>6}{s:>9.3f}"
                      f"{r:>11.3f}{r - s:>+8.3f}{g:>10.3f}{o:>9.3f}{mark}")
        print()

    print(f"{'type':<19}{'k':>4}{'all-gold SINGLE':>17}{'REALISTIC':>11}{'gain':>8}"
          f"{'  bridge recoverable from hop-1':>32}")
    for qtype in TYPES:
        for k in ks:
            a = acc.get((qtype, k, "all"))
            if not a or not a["n"]:
                continue
            n = a["n"]
            rec = (a["recoverable"] / a["bridged"]) if a["bridged"] else float("nan")
            print(f"{qtype:<19}{k:>4}{a['single_full'] / n:>17.3f}"
                  f"{a['realistic_full'] / n:>11.3f}"
                  f"{a['realistic_full'] / n - a['single_full'] / n:>+8.3f}"
                  f"{rec:>32.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
