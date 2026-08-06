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
import re
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


# A run of capitalised words, optionally joined by lowercase particles ("of",
# "the"), is a good enough proxy for "name of a thing" on Wikipedia prose. This
# is deliberately NOT a model call: it must not itself degrade under
# quantization, or it would confound the thing being measured.
_NAME = re.compile(
    r"\b[A-Z][\w'’-]*(?:\s+(?:of|the|de|van|von|and|for|in|at|on)\s+[A-Z][\w'’-]*"
    r"|\s+[A-Z][\w'’-]*)*"
)
_STOP = {"the", "a", "an", "he", "she", "it", "they", "this", "that", "in", "on",
         "at", "for", "and", "but", "his", "her", "their", "its", "was", "were",
         "is", "are", "who", "which", "what", "when", "where", "after", "before"}


def candidate_names(spans: list[str]) -> list[str]:
    """Capitalised name phrases in the Extractor's spans, longest first.

    Sentence-initial single words are dropped: "The" and "After" start sentences
    and are not names, and a bare one-word capital is too ambiguous to search.
    """
    out: dict[str, int] = {}
    for sp in spans:
        for m in _NAME.finditer(sp or ""):
            c = m.group(0).strip()
            if len(c.split()) < 2 and c.lower() in _STOP:
                continue
            if len(c) < 4:
                continue
            out[c] = max(out.get(c, 0), len(c.split()))
    return sorted(out, key=lambda c: -out[c])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="hotpot")
    ap.add_argument("--corpus", choices=["distractor", "fullwiki", "union"],
                    default="union",
                    help="hotpot only. fullwiki holds a real IR system's top-10, which "
                         "misses gold for 39%% of questions; union adds the distractor "
                         "paragraphs so gold is always reachable while keeping "
                         "fullwiki's harder distractors.")
    ap.add_argument("--hop1", type=int, default=0,
                    help="passages hop-1 keeps (rest go to hop-2). 0 = k//2.")
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

    def pool(dataset, into: dict[str, str]) -> dict[str, str]:
        for row in dataset:
            ctx = row["context"]
            for title, sents in zip(ctx["title"], ctx[cfg["sents"]]):
                if title not in into:
                    into[title] = " ".join(s.strip() for s in sents).strip()
        return into

    seen: dict[str, str] = {}
    if args.set == "hotpot" and args.corpus in ("fullwiki", "union"):
        # fullwiki and distractor share ids, questions and supporting_facts; only
        # the 10 paragraphs differ. Questions still come from `ds` (distractor),
        # which carries the `type` field.
        pool(load_dataset(name, "fullwiki", split=cfg["split"]), seen)
    if not (args.set == "hotpot" and args.corpus == "fullwiki"):
        pool(ds, seen)
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

        # What a working Extractor returns from pass 1: the gold supporting
        # SENTENCES of whichever gold paragraphs hop-1 actually surfaced. The
        # real pipeline has spans, not entity names, so this is the query
        # material actually available to hop-2.
        sents_by_title = dict(zip(row["context"]["title"], row["context"][cfg["sents"]]))
        support: dict[str, list[str]] = {}
        sf = row["supporting_facts"]
        for t, sid in zip(sf["title"], sf["sent_id"]):
            sl = sents_by_title.get(t)
            if sl and 0 <= sid < len(sl):
                support.setdefault(t, []).append(sl[sid])

        single_full = index.search_titles(q, max(ks))
        strat = "hidden-bridge" if hidden else "fully-named"

        for k in ks:
            half = args.hop1 or max(1, k // 2)
            rest = max(1, k - half)
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

            # SPAN arm: hop-2 query is the pass-1 spans plus the question. No
            # entity names, nothing the Extractor cannot already produce.
            spans = [s for t in (set(hop1) & gset) for s in support.get(t, [])]
            span_pass = (
                set(hop1) | set(index.search_titles(" ".join(spans) + " " + q, half))
                if spans else set(single_full[:k])
            )

            # NER arm: pull capitalised name phrases out of the pass-1 spans and
            # search those instead of the whole sentence. Deterministic, no model
            # call. Names already in the question are dropped -- they are what
            # hop-1 was for, so re-searching them cannot surface anything new.
            # A candidate is worth a second query only if it is BOTH absent from
            # the question and not already the title of something hop-1 returned.
            # Without the second condition the arm fires on fully-named questions
            # -- where every needed page is already in hand -- and spends half the
            # budget re-finding what it has, costing 0.172 all-gold there. Both
            # conditions are checkable at run time with no gold labels.
            hop1_titles = [norm(t) for t in hop1]
            names = [
                c for c in candidate_names(spans)
                if norm(c) not in qn
                and not any(norm(c) in t for t in hop1_titles)
            ]
            ner_pass = (
                set(hop1) | set(index.search_titles(" ".join(names[:4]), rest))
                if names else set(single_full[:k])
            )

            for key in ((qtype, k), (strat, k), ("ALL", k)):
                a = acc[key]
                for arm, got in (("single", s_at_k), ("two_pass", two_pass),
                                 ("span_pass", span_pass), ("ner_pass", ner_pass), ("oracle", oracle)):
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
           f"{'SINGLE':>9}{'NER':>9}{'gain':>8}{'TWO_PASS':>10}{'ORACLE':>9}")
    print("\n=== recall over ALL gold titles ===")
    print(hdr)
    print("-" * len(hdr))
    for g in groups:
        for k in ks:
            a = acc.get((g, k))
            if not a or not a["n"]:
                continue
            n = a["n"]
            s, sp = a["single"] / n, a["ner_pass"] / n
            t, o = a["two_pass"] / n, a["oracle"] / n
            mark = "  <<<" if sp - s > 0.05 else ""
            print(f"{g:<19}{k:>4}{int(n):>6}{a['n_hid'] / n:>7.2f}"
                  f"{s:>9.3f}{sp:>9.3f}{sp - s:>+8.3f}{t:>10.3f}{o:>9.3f}{mark}")
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
            s, sp = a["single_hid"] / nh, a["ner_pass_hid"] / nh
            t, o = a["two_pass_hid"] / nh, a["oracle_hid"] / nh
            mark = "  <<<" if sp - s > 0.05 else ""
            print(f"{g:<19}{k:>4}{int(nh):>6}{a['recoverable'] / nh:>7.2f}"
                  f"{s:>9.3f}{sp:>9.3f}{sp - s:>+8.3f}{t:>10.3f}{o:>9.3f}{mark}")
        print()

    print("=== ALL-gold-retrieved (can the question be answered at all?) ===")
    print(f"{'stratum':<19}{'k':>4}{'n':>6}{'SINGLE':>9}{'NER':>9}{'gain':>8}"
          f"{'TWO_PASS':>10}")
    for g in groups:
        for k in ks:
            a = acc.get((g, k))
            if not a or not a["n"]:
                continue
            n = a["n"]
            s, sp, t = (a["single_full"] / n, a["ner_pass_full"] / n,
                        a["two_pass_full"] / n)
            print(f"{g:<19}{k:>4}{int(n):>6}{s:>9.3f}{sp:>9.3f}{sp - s:>+8.3f}"
                  f"{t:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
