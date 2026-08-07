"""Sparse retrieval for the controlled open-corpus MA-RAG experiment.

The production corpus pools unique HotpotQA ``distractor`` and ``fullwiki``
paragraphs into a frozen 72,094-passage haystack.  Distractor paragraphs are
inserted first because HotpotQA supporting-fact sentence IDs use those sentence
splits; changing precedence can silently corrupt evidence scoring.  The runner
asserts the passage count, corpus content hash, and exact normalized gold-sentence
coverage before any model is loaded.

Production retrieval follows the MA-RAG control loop.  Each Step Definer
``question-answering`` task becomes its own BM25 query and receives a fresh
top-10 result set.  ``aggregate`` tasks retrieve nothing.  The Planner may emit
one to five steps, later queries can use compact answers from earlier steps, QA
can stop the plan early, and the Step Definer summarizes the completed state.
Consequently, this module does not define a fixed one-hop/two-hop pipeline.

The final cohort is labelled after freezing as ``hidden_bridge`` (at least one
gold page title absent from the question) or ``fully_named`` (all gold titles
present).  Those labels are analysis-only controls and never enter prompts or
retrieval queries.

The BM25 IDF is the positive Lucene form
``log(1 + (N-df+0.5)/(df+0.5))``.  It is intentionally not rank_bm25's floored
Okapi variant, so internal ordering can differ.  Tests pin the vectorized scores
to a direct implementation.  Legacy entity-regex and 7/3 fusion helpers remain
below solely to reproduce archived retrieval probes; the production pipeline
does not call them.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
K1 = 1.5
B = 0.75

# Active MA-RAG retrieval budget: every question-answering plan step issues its
# own Step-Definer query and retrieves top-K. HOP1 remains only for archived
# two-query diagnostic helpers and is not part of the production query policy.
K = 10
HOP1 = 7
ANCHOR_K = 7
QUERY_POLICY = "anchored_original_question_7_plus_grounded_task_3_v1"


def fuse_rankings(
    anchor_titles: list[str],
    task_titles: list[str],
    *,
    k: int = K,
    anchor_k: int = ANCHOR_K,
) -> list[str]:
    """Fuse rankings with a fixed anchor quota and no duplicate exposure."""
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if not 0 < anchor_k <= k:
        raise ValueError(f"need 0 < anchor_k <= k; got anchor_k={anchor_k}, k={k}")

    fused: list[str] = []
    seen: set[str] = set()

    def extend(values: list[str], limit: int) -> None:
        for title in values:
            if len(fused) >= limit:
                return
            if title not in seen:
                seen.add(title)
                fused.append(title)

    extend(anchor_titles[:anchor_k], anchor_k)
    task_target = min(k, len(fused) + (k - anchor_k))
    extend(task_titles, task_target)
    extend(anchor_titles, k)
    extend(task_titles, k)
    return fused



def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return _TOKEN.findall(normalized)


def norm(s: str) -> str:
    return " ".join(tokenize(s))


def title_is_mentioned(title: str, question: str) -> bool:
    """Whether a Wikipedia title is explicitly named in the question.

    Terminal disambiguators such as ``(film)`` are metadata, not hidden bridge
    entities. Unicode normalization and token boundaries avoid both accent loss
    and accidental substring matches.
    """
    base = re.sub(r"\s*\([^()]+\)\s*$", "", title or "")
    title_tokens = tokenize(base)
    question_tokens = tokenize(question)
    if not title_tokens or len(title_tokens) > len(question_tokens):
        return False
    width = len(title_tokens)
    return any(
        question_tokens[start:start + width] == title_tokens
        for start in range(len(question_tokens) - width + 1)
    )


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
        if not passages:
            raise ValueError("BM25 corpus must be non-empty")
        titles = [p.title for p in passages]
        seen_titles: set[str] = set()
        duplicate_titles: set[str] = set()
        for title in titles:
            if title in seen_titles:
                duplicate_titles.add(title)
            else:
                seen_titles.add(title)
        duplicates = sorted(duplicate_titles)
        if duplicates:
            raise ValueError(
                "duplicate passage title(s) are ambiguous: "
                + ", ".join(repr(title) for title in duplicates[:5])
            )
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
        k = min(k, len(scores))
        if k == len(scores):
            selected = np.arange(len(scores), dtype=np.int64)
        else:
            # argpartition alone chooses arbitrary members at the boundary when
            # many documents share a score (especially the zero-score tail).
            # Select every strict winner, then fill the remaining slots from the
            # tied documents in corpus order. This stays O(N) while making the
            # result stable across NumPy implementations and workers.
            threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
            above = np.flatnonzero(scores > threshold)
            tied = np.flatnonzero(scores == threshold)
            selected = np.concatenate((above, tied[: k - len(above)]))
        order = np.lexsort((selected, -scores[selected]))
        return [int(i) for i in selected[order]]

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
# This archived helper is deliberately not a model call. Production retrieval
# uses the Step Definer's task directly and never calls this entity regex.
#
# The joiner list holds ONLY particles that occur inside a single name
# ("University of Texas", "Vincent van Gogh"). Conjunctions and prepositions are
# excluded on purpose: with "and" in the list, "Paris and Xawery Zulawski" is
# captured as one 25-character phrase, which BM25 then searches as a unit and
# matches nothing. Verified by test_conjunctions_do_not_join_names.
_NAME_WORD = re.compile(r"[^\W_][\w'’-]*", re.UNICODE)
_NAME_PARTICLES = {"of", "the", "de", "del", "della", "da", "di", "van", "von", "la", "le", "du", "des"}
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
        words = [match.group(0) for match in _NAME_WORD.finditer(sp or "")]
        for start, word in enumerate(words):
            if not word[0].isupper():
                continue
            if start and words[start - 1][0].isupper():
                continue
            if (
                start >= 2
                and words[start - 1].casefold() in _NAME_PARTICLES
                and words[start - 2][0].isupper()
            ):
                continue
            phrase = [word]
            cursor = start + 1
            while cursor < len(words):
                nxt = words[cursor]
                if nxt[0].isupper():
                    phrase.append(nxt)
                    cursor += 1
                    continue
                if (
                    nxt.casefold() in _NAME_PARTICLES
                    and cursor + 1 < len(words)
                    and words[cursor + 1][0].isupper()
                ):
                    phrase.extend((nxt, words[cursor + 1]))
                    cursor += 2
                    continue
                break
            c = " ".join(phrase)
            if len(c) < 4:
                continue
            word_count = len(phrase)
            # A lone capitalised stopword is sentence-initial punctuation, not a
            # name: "The", "After", "However".
            if word_count < 2 and c.casefold() in _STOP:
                continue
            out[c] = max(out.get(c, 0), word_count)
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
    needed = max(1, k - hop1)
    excluded = set(hop1_titles or ())
    # Search enough candidates to replace every possible hop-1 overlap, then
    # retain only unseen titles. A fired second hop must spend its budget on new
    # passages rather than showing the Extractor the same page twice.
    candidate_k = min(len(index.passages), needed + len(excluded))
    titles = [
        title for title in index.search_titles(query, candidate_k)
        if title not in excluded
    ][:needed]
    return {"titles": titles, "query": query, "fired": True}


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

    def __init__(
        self,
        passages: list[Passage],
        k: int = K,
        hop1: int | None = None,
        anchor_k: int | None = None,
    ):
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        if hop1 is not None and not 0 < hop1 < k:
            raise ValueError(f"need 0 < hop1 < k; got hop1={hop1}, k={k}")
        anchor_k = (
            min(ANCHOR_K, max(1, k - 1)) if anchor_k is None else anchor_k
        )
        if not 0 < anchor_k <= k:
            raise ValueError(f"need 0 < anchor_k <= k; got anchor_k={anchor_k}, k={k}")
        self.index = BM25Index(passages)
        self.k = k
        self.anchor_k = anchor_k
        self.task_k = k - anchor_k
        # Kept for archived two-pass probes; production leaves it None.
        self.hop1 = hop1
        self._sentences = {p.title: p.sentences for p in passages}
        corpus_payload = [
            {"title": passage.title, "sentences": list(passage.sentences)}
            for passage in passages
        ]
        self._corpus_sha256 = hashlib.sha256(
            json.dumps(
                corpus_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

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
            "corpus_sha256": self._corpus_sha256,
            "algorithm": "sparse_bm25_lucene_idf_v1",
            "tie_break": "score_desc_corpus_index_asc",
            "query_policy": QUERY_POLICY,
            "k_per_step": self.k,
            "anchor_k": self.anchor_k,
            "task_k": self.task_k,
            "bm25_k1": self.index.k1,
            "bm25_b": self.index.b,
            "vocab_terms": len(self.index.vocab),
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
