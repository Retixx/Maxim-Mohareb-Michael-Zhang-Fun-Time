"""Does iterative decomposition beat a single query? CPU only, no GPU.

THE BUG THIS DIAGNOSES
----------------------
SPEC §3 used HotpotQA's *distractor* setting, which hands over 10 paragraphs
containing both gold. MA-RAG uses the *open-domain* setting and issues repeated
targeted queries against millions of passages. With gold guaranteed present
there is nothing to retrieve, so decomposition has no mechanism through which to
help and the four agents are pure overhead -- hence single_fp16 winning by 9.2
EM. The fix is to restore the haystack, not to change the pipeline.

WHAT DECIDES WHETHER DECOMPOSITION PAYS
---------------------------------------
Whether the question NAMES the entity whose page you need.

    "Which magazine started first, Arthur's Magazine or First for Women?"
        both gold titles are in the question -- one query finds both, and
        splitting it strictly discards lexical signal.

    "Who is the mother of the director of film Polish-Russian War?"
        gold = [Polish-Russian War, Xawery Zulawski]. The director is NEVER
        named. No single query can retrieve his page. You must read hop-1,
        learn the name, and issue a second query.

So the arms are stratified by whether a gold title is HIDDEN (absent from the
question text), and hidden-title recall is the number that matters -- it is
precisely what the second hop exists to reach.

ARMS -- equal read budget, every arm returns at most k passages
---------------------------------------------------------------
    SINGLE     one query = the question, top-k.
    TWO_PASS   hop-1 = question, top-k/2. Then for each hidden gold title whose
               name literally occurs in the text of a passage hop-1 ACTUALLY
               RETRIEVED, issue a hop-2 query of that name plus the question.
               Union, capped at k.
               If nothing is recoverable it falls back to plain top-k, so the
               arm is never handed information it could not have read.
    ORACLE     same, but every hidden title is spliced whether or not hop-1
               surfaced it. Upper bound; the gap to TWO_PASS is the cost of
               imperfect resolution.

Bridges are derived identically for both datasets (hidden gold title), so the
comparison is like-for-like and 2WikiMQA gets no advantage from its `evidences`
annotations.

CAVEAT, stated because it bounds the claim
------------------------------------------
TWO_PASS assumes perfect entity SELECTION: a real Extractor reading hop-1 sees
many entities and must choose which to follow. This measures the ceiling of that
choice, not the choice itself. Degrading the Extractor is exactly what makes a
real system fall short of it -- which is the experiment.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SETS = {
    "hotpot": {
        "path": ("hotpotqa/hotpot_qa", "distractor"),
        "split": "validation",
        "sents": "sentences",
        "types": ["bridge", "comparison"],
    },
    "2wiki": {
        "path": ("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35", None),
        "split": "validation",
        "sents": "content",
        "types": ["compositional", "inference", "bridge_comparison", "comparison"],
    },
}


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="hotpot")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--ks", default="10,20")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    cfg = SETS[args.set]

    from datasets import load_dataset

    from src.retrieval import BM25Index, Passage

    name, sub = cfg["path"]
    ds = load_dataset(name, sub, split=cfg["split"]) if sub else \
        load_dataset(name, split=cfg["split"])

    seen: dict[str, str] = {}
    for row in ds:
        ctx = row["context"]
        for title, sents in zip(ctx["title"], ctx[cfg["sents"]]):
            if title not in seen:
                seen[title] = " ".join(s.strip() for s in sents).strip()
    passages = [Passage(t, x) for t, x in seen.items()]
    text_of = {p.title: norm(p.text) for p in passages}
    print(f"{args.set}: {len(ds):,} questions, {len(passages):,} unique passages")
    print("indexing...", flush=True)
    index = BM25Index(passages)
    print("indexed\n", flush=True)

    acc: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = list(ds)[: args.n]

    for i, row in enumerate(rows):
        if i and i % 250 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

        q = row["question"]
        qn = norm(q)
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        if len(gold) < 2:
            continue
        qtype = row["type"]
        gset = set(gold)
        hidden = [g for g in gold if norm(g) not in qn]

        single_full = index.search_titles(q, max(ks))
        strat = "hidden-bridge" if hidden else "fully-named"

        for k in ks:
            half = max(1, k // 2)
            s_at_k = set(single_full[:k])
            hop1 = single_full[:half]
            hop1_text = " ".join(text_of.get(t, "") for t in hop1)

            def follow(bridges):
                if not bridges:
                    return set(single_full[:k])
                out = set(hop1)
                per = max(1, half // len(bridges))
                for ent in bridges:
                    out |= set(index.search_titles(f"{ent} {q}", per))
                return out

            oracle = follow(hidden)
            # Only follow a bridge a reader could actually have read off hop-1.
            recoverable = [h for h in hidden if norm(h) in hop1_text]
            two_pass = follow(recoverable)

            for key in ((qtype, k), (strat, k), ("ALL", k)):
                a = acc[key]
                for arm, got in (("single", s_at_k), ("two_pass", two_pass),
                                 ("oracle", oracle)):
                    a[arm] += len(gset & got) / len(gset)
                    a[arm + "_full"] += float(gset <= got)
                    if hidden:
                        hs = set(hidden)
                        a[arm + "_hid"] += len(hs & got) / len(hs)
                a["n"] += 1
                if hidden:
                    a["n_hid"] += 1
                    a["recoverable"] += len(recoverable) / len(hidden)

    groups = cfg["types"] + ["hidden-bridge", "fully-named", "ALL"]
    hdr = (f"{'stratum':<19}{'k':>4}{'n':>6}{'%hid':>7}"
           f"{'SINGLE':>9}{'TWO_PASS':>10}{'gain':>8}{'ORACLE':>9}")
    print("\n=== recall over ALL gold titles ===")
    print(hdr)
    print("-" * len(hdr))
    for g in groups:
        for k in ks:
            a = acc.get((g, k))
            if not a or not a["n"]:
                continue
            n = a["n"]
            s, t, o = a["single"] / n, a["two_pass"] / n, a["oracle"] / n
            mark = "  <<<" if t - s > 0.05 else ""
            print(f"{g:<19}{k:>4}{int(n):>6}{a['n_hid'] / n:>7.2f}"
                  f"{s:>9.3f}{t:>10.3f}{t - s:>+8.3f}{o:>9.3f}{mark}")
        print()

    print("=== recall over HIDDEN gold titles only (what hop-2 exists for) ===")
    print(hdr.replace("%hid", "recov"))
    print("-" * len(hdr))
    for g in groups:
        for k in ks:
            a = acc.get((g, k))
            if not a or not a["n_hid"]:
                continue
            nh = a["n_hid"]
            s, t, o = a["single_hid"] / nh, a["two_pass_hid"] / nh, a["oracle_hid"] / nh
            mark = "  <<<" if t - s > 0.05 else ""
            print(f"{g:<19}{k:>4}{int(nh):>6}{a['recoverable'] / nh:>7.2f}"
                  f"{s:>9.3f}{t:>10.3f}{t - s:>+8.3f}{o:>9.3f}{mark}")
        print()

    print("=== ALL-gold-retrieved (can the question be answered at all?) ===")
    print(f"{'stratum':<19}{'k':>4}{'n':>6}{'SINGLE':>9}{'TWO_PASS':>10}{'gain':>8}")
    for g in groups:
        for k in ks:
            a = acc.get((g, k))
            if not a or not a["n"]:
                continue
            n = a["n"]
            s, t = a["single_full"] / n, a["two_pass_full"] / n
            print(f"{g:<19}{k:>4}{int(n):>6}{s:>9.3f}{t:>10.3f}{t - s:>+8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
