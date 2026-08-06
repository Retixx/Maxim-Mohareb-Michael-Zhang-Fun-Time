"""BM25 retrieval over a pooled HotpotQA corpus.

WHY THIS EXISTS
---------------
SPEC §3 deleted the retriever and handed the Extractor a fixed 10 paragraphs.
That made the Step Definer's `search_terms` field dead output — nothing consumed
it — and left decomposition with no mechanism through which to help, which is
why a single call beats the four-agent pipeline by 9.2 EM in that setting.

MA-RAG issues repeated targeted queries against ~21M DPR Wikipedia passages.
This module restores the mechanism at a smaller scale: every unique paragraph in
the HotpotQA dev split, pooled into one 66,581-passage corpus with 2 gold per
question. Not open-domain scale, but the same shape.

IMPLEMENTATION NOTE
-------------------
`rank_bm25` is pure Python and scores every document per query; over 66k
passages that is ~0.5 s/query, which at ~5 queries per question and n=1500 is
over an hour of CPU. This uses a sparse inverted index (scipy CSR) instead:
identical Okapi BM25 scores, ~1 ms/query. Verified against rank_bm25 in
tests/test_retrieval.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


@dataclass
class Passage:
    title: str
    text: str


class BM25Index:
    """Okapi BM25 over a fixed passage set, sparse and vectorised."""

    def __init__(self, passages: list[Passage], k1: float = K1, b: float = B):
        self.passages = passages
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

        # Precompute the full BM25 term weight per (doc, term); the query then
        # only has to sum columns.
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = (self.k1 * (1.0 - self.b + self.b * lengths / (self.avgdl or 1.0)))
        coo = tf.tocoo()
        weighted = (
            coo.data * (self.k1 + 1.0) / (coo.data + norm[coo.row])
        ) * idf[coo.col]
        self.matrix = sparse.csr_matrix(
            (weighted.astype(np.float32), (coo.row, coo.col)),
            shape=(n_docs, n_terms),
        ).tocsc()

    def search(self, query: str, k: int = 10) -> list[int]:
        """Indices of the top-k passages for `query`, best first."""
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return []
        scores = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        if k >= len(scores):
            return list(np.argsort(-scores))
        top = np.argpartition(-scores, k)[:k]
        return list(top[np.argsort(-scores[top])])

    def search_titles(self, query: str, k: int = 10) -> list[str]:
        return [self.passages[i].title for i in self.search(query, k)]


def build_corpus(dataset) -> list[Passage]:
    """Every unique paragraph in the split, keyed by title.

    HotpotQA reuses paragraphs across questions, so pooling the distractor
    contexts yields one shared corpus rather than per-question contexts. 66,581
    unique passages for the dev split.
    """
    seen: dict[str, str] = {}
    for row in dataset:
        ctx = row["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title not in seen:
                seen[title] = " ".join(s.strip() for s in sents).strip()
    return [Passage(title=t, text=x) for t, x in seen.items()]


def format_retrieved(passages: list[Passage]) -> str:
    """Render retrieved passages in the same shape prompts already expect.

    Identical formatting to prompts.format_paragraphs so the Extractor's frozen
    v5 template sees the same structure it always has — only the *provenance* of
    the passages changes, not their presentation.
    """
    return "\n".join(
        f"[{i}] {p.title}: {p.text}" for i, p in enumerate(passages, start=1)
    )
