"""Open-domain retrieval: the haystack SPEC §3 deleted.

WHY THIS EXISTS
---------------
SPEC §3 used HotpotQA's *distractor* setting, which hands the pipeline ten
paragraphs already containing both gold. MA-RAG uses the open-domain setting and
issues repeated targeted queries against millions of passages. With gold
guaranteed present there is nothing to retrieve, the Step Definer's
`search_terms` field is consumed by nothing, and decomposition has no mechanism
through which to help -- which is why `single_fp16` beat the four-agent pipeline
by 9.2 EM. The defect was the missing haystack, not the pipeline.

THE CORPUS
----------
Every unique paragraph from BOTH HotpotQA configs, pooled:

  * `fullwiki` contributes a real IR system's top-10 per question -- genuinely
    hard lexical distractors, and gold missing for 39% of questions.
  * `distractor` contributes its ten, which guarantees every gold paragraph is
    reachable.

Union: 72,094 passages, 100% gold-in-corpus. Hard distractors AND an achievable
answer, which is the condition worth measuring: single-query retrieval must be
usable but beatable, not impossible.

SENTENCE-SPLIT PRECEDENCE -- do not "simplify" this
---------------------------------------------------
402 titles appear in both configs with DIFFERENT sentence splits. `sent_id` in
`supporting_facts` is defined against the distractor context, so distractor
sentences must win for any shared title. Taking fullwiki's split for those 402
would silently shift gold sentence indices and corrupt every extraction-accuracy
number without raising anything.

WHAT DECIDES WHETHER THE SECOND HOP PAYS
----------------------------------------
Whether the question NAMES the page you need.

    "Which magazine was started first, Arthur's Magazine or First for Women?"
        both needed pages named -> one query finds both -> a second hop can only
        waste budget. 20% of the dev split.

    "The director of the film Polish-Russian War was born in what year?"
        needs `Polish-Russian War (film)` (named) and `Xawery Zulawski` (never
        named). No single query can reach the second page. 80% of the split.

Measured at k=10 over 1,500 questions, hop-1 7 / hop-2 3, versus one query:

    stratum        n     all-gold-retrieved   hidden-title recall
    hidden-bridge  1204  0.520 -> 0.678       0.653 -> 0.794
    fully-named     296  0.892 -> 0.797       (nothing hidden)

The fully-named row is the control: two hops cannot help where nothing is
hidden, and the 7/3 split is what keeps that cost small. A 5/5 split scores an
identical +0.158 on hidden-bridge but -0.172 on fully-named; 8/2 protects
fully-named further but gives up a third of the gain.

IMPLEMENTATION NOTE
-------------------
`rank_bm25` is pure Python and scores every document per query: ~0.5 s/query
over 72k passages, which at several queries per question and n=1500 is hours of
CPU. This uses a sparse inverted index (scipy CSR) instead: ~0.4 ms/query,
roughly 1379x faster.

The IDF is the Lucene/Elasticsearch form, log(1 + (N-df+0.5)/(df+0.5)), NOT the
form in rank_bm25's BM25Okapi, which is log(N-df+0.5) - log(df+0.5) with
negative values floored to `epsilon * average_idf`. The Lucene form is positive
by construction and needs no such patch. Rankings therefore agree closely but
not identically, and every measurement backing the k and hop-1 choices was taken
with THIS implementation. tests/test_retrieval.py pins the scores against a
direct transcription of the formula (which is what catches vectorisation bugs)
and separately records the level of agreement with rank_bm25.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75

# Retrieval budget. `K` passages reach the Extractor per sub-question in total;
# HOP1 of them come from the question itself and the remainder from the
# follow-up query. Swept over k in {5,8,10,12,16,20} and hop-1 in {5,6,7,8}.
K = 10
HOP1 = 7


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


@dataclass
class Passage:
    title: str
    sentences: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences).strip()


class BM25Index:
    """Okapi BM25 over a fixed passage set, sparse and vectorised."""

    def __init__(self, passages: list[Passage], k1: float = K1, b: float = B):
        self.passages = passages
        self.title_to_index = {p.title: i for i, p in enumerate(passages)}
        self.k1, self.b = k1, b

        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(passages), dtype=np.float32)
        for i, p in enumerate(passages):
            toks = tokenize(f"{p.title} {p.text}")
            lengths[i] = len(toks)
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)

        self.vocab = vocab
        n_docs, n_terms = len(passages), len(vocab)
        tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(n_docs, n_terms),
        )
        self.avgdl = float(lengths.mean()) if len(lengths) else 0.0

        # Precompute the full BM25 term weight per (doc, term); a query then only
        # has to sum columns.
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm_len = self.k1 * (1.0 - self.b + self.b * lengths / (self.avgdl or 1.0))
        coo = tf.tocoo()
        weighted = (
            coo.data * (self.k1 + 1.0) / (coo.data + norm_len[coo.row])
        ) * idf[coo.col]
        self.matrix = sparse.csr_matrix(
            (weighted.astype(np.float32), (coo.row, coo.col)),
            shape=(n_docs, n_terms),
        ).tocsc()

    def search(self, query: str, k: int = K) -> list[int]:
        """Indices of the top-k passages for `query`, best first."""
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols or k <= 0:
            return []
        scores = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        if k >= len(scores):
            return list(np.argsort(-scores))
        top = np.argpartition(-scores, k)[:k]
        return list(top[np.argsort(-scores[top])])

    def search_titles(self, query: str, k: int = K) -> list[str]:
        return [self.passages[i].title for i in self.search(query, k)]

    def get(self, title: str) -> Passage | None:
        i = self.title_to_index.get(title)
        return self.passages[i] if i is not None else None


# --------------------------------------------------------------------------
# Follow-up query construction
# --------------------------------------------------------------------------

# A run of capitalised words, optionally joined by lowercase particles, is a good
# enough proxy for "name of a thing" in Wikipedia prose.
#
# This is deliberately NOT a model call. A quantized Extractor must be able to
# hurt retrieval only through WHICH SENTENCES it selects, never through a
# separate learned component that also degrades -- otherwise the role-precision
# effect this experiment measures would be confounded by a second moving part.
# Measured against splicing the gold bridge title directly: 0.794 vs 0.812
# hidden-title recall, so the regex costs 0.018 and buys full determinism.
#
# The joiner list holds ONLY particles that occur inside a single name
# ("University of Texas", "Vincent van Gogh"). Conjunctions and prepositions are
# excluded on purpose: with "and" in the list, "Paris and Xawery Zulawski" is
# captured as one 25-character phrase, which BM25 then searches as a unit and
# matches nothing. Verified by test_conjunctions_do_not_join_names.
_NAME = re.compile(
    r"\b[A-Z][\w'’-]*"
    r"(?:\s+(?:of|the|de|del|della|da|di|van|von|la|le|du|des)\s+[A-Z][\w'’-]*"
    r"|\s+[A-Z][\w'’-]*)*"
)
_STOP = {
    "the", "a", "an", "he", "she", "it", "they", "this", "that", "in", "on",
    "at", "for", "and", "but", "his", "her", "their", "its", "was", "were",
    "is", "are", "who", "which", "what", "when", "where", "after", "before",
    "however", "later", "during", "both", "these", "those", "there",
}
MAX_FOLLOWUP_NAMES = 4


def candidate_names(spans: list[str]) -> list[str]:
    """Capitalised name phrases in the Extractor's spans, longest first.

    Longest-first matters: "Xawery Zulawski" must outrank "Polish" when only
    MAX_FOLLOWUP_NAMES survive, and longer phrases are both more specific and
    more likely to be the bridge entity.
    """
    out: dict[str, int] = {}
    for sp in spans:
        for m in _NAME.finditer(sp or ""):
            c = m.group(0).strip()
            if len(c) < 4:
                continue
            words = len(c.split())
            # A lone capitalised stopword is sentence-initial punctuation, not a
            # name: "The", "After", "However".
            if words < 2 and c.lower() in _STOP:
                continue
            out[c] = max(out.get(c, 0), words)
    return sorted(out, key=lambda c: (-out[c], c))


def followup_query(spans: list[str], question: str, hop1_titles: list[str]) -> str:
    """The hop-2 query, or "" when a second hop cannot help.

    A candidate name earns a query only if it is BOTH absent from the question
    and not already the title of something hop-1 returned. Both conditions are
    checkable at run time with no gold labels.

    Dropping names already in the question is what makes this a BRIDGE query
    rather than a rerun of hop-1. Dropping names hop-1 already retrieved is what
    keeps fully-named questions from spending budget re-finding pages they hold.
    """
    qn = norm(question)
    titles = [norm(t) for t in hop1_titles]
    names = [
        c for c in candidate_names(spans)
        if norm(c) not in qn and not any(norm(c) in t for t in titles)
    ]
    return " ".join(names[:MAX_FOLLOWUP_NAMES])


def retrieve(index: BM25Index, question: str, k: int = K, hop1: int = HOP1,
             spans: list[str] | None = None,
             hop1_titles: list[str] | None = None) -> dict:
    """One retrieval round.

    With no `spans` this is hop 1: the question alone, top-`k`... but only
    `hop1` of those are kept if a second hop will follow. The caller decides by
    passing `spans` on the second call.

    Returns {"titles", "query", "fired"} where `fired` is False when no
    follow-up was warranted, in which case the caller should keep hop-1's full
    top-`k` rather than a truncated list.
    """
    if spans is None:
        return {"titles": index.search_titles(question, k), "query": question,
                "fired": True}
    query = followup_query(spans, question, hop1_titles or [])
    if not query:
        return {"titles": [], "query": "", "fired": False}
    return {"titles": index.search_titles(query, max(1, k - hop1)),
            "query": query, "fired": True}


# --------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------

def build_corpus(name: str = "hotpotqa/hotpot_qa", split: str = "validation",
                 revision: str | None = None, configs: tuple[str, ...] = ("distractor", "fullwiki"),
                 loader=None) -> list[Passage]:
    """Pool every unique paragraph across `configs` into one corpus.

    CONFIG ORDER IS LOAD-BEARING. The first config to supply a title wins, and
    `distractor` must come first: `supporting_facts.sent_id` is defined against
    the distractor context, and 402 titles carry a different sentence split in
    fullwiki. Reordering these silently shifts gold sentence indices.
    """
    if loader is None:
        from datasets import load_dataset as loader  # noqa: N813

    seen: dict[str, list[str]] = {}
    for config in configs:
        ds = loader(name, config, split=split, revision=revision)
        for row in ds:
            ctx = row["context"]
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                if title not in seen:
                    seen[title] = list(sents)
    return [Passage(title=t, sentences=s) for t, s in seen.items()]


class RetrievalContext:
    """The corpus, the index, and the budget — one object the pipeline reads.

    Built once per campaign (indexing 72k passages takes ~3.5 s) and passed to
    every stage that reads passages.
    """

    def __init__(self, passages: list[Passage], k: int = K, hop1: int = HOP1):
        if not 0 < hop1 < k:
            raise ValueError(f"need 0 < hop1 < k; got hop1={hop1}, k={k}")
        self.index = BM25Index(passages)
        self.k = k
        self.hop1 = hop1
        self._sentences = {p.title: p.sentences for p in passages}

    def passages(self, titles: list[str]) -> list[Passage]:
        got = [self.index.get(t) for t in titles]
        return [p for p in got if p is not None]

    def sentence_index(self, titles: list[str]) -> list[dict]:
        """Addressable sentences for exactly the passages a question saw."""
        from .evidence import build_sentence_index

        present = [t for t in titles if t in self._sentences]
        return build_sentence_index(present, [self._sentences[t] for t in present])

    def fingerprint(self) -> dict:
        """Provenance for the run metadata: the retrieval setup is now a variable."""
        return {
            "corpus_passages": len(self.index.passages),
            "k": self.k,
            "hop1": self.hop1,
            "hop2": self.k - self.hop1,
            "bm25_k1": self.index.k1,
            "bm25_b": self.index.b,
            "vocab_terms": len(self.index.vocab),
            "max_followup_names": MAX_FOLLOWUP_NAMES,
        }


def format_passages(passages: list[Passage]) -> str:
    """Render passages exactly as the frozen v5 prompts already expect.

    Delegates to prompts.format_paragraphs rather than reimplementing it, so the
    Extractor's frozen template sees byte-identical structure -- only the
    PROVENANCE of the passages changes, never their presentation. Formatting them
    differently here would confound the retrieval change with a prompt change.
    """
    from .prompts import format_paragraphs

    return format_paragraphs([p.title for p in passages],
                             [p.sentences for p in passages])
