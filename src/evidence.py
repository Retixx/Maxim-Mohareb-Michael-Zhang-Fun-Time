"""Extraction accuracy: did the Extractor find the RIGHT evidence?

WHY THIS EXISTS
---------------
Until this module, nothing measured extraction correctness. `parse_status` says the
Extractor's JSON was well-formed. `mechanism.verbatim_rate` says its spans were
copied rather than paraphrased. `selection_changed` says the spans differed between
two runs. None of them say the evidence was *right* — a model that verbatim-copies
a completely irrelevant sentence scores perfectly on all three.

That made every claim about "extraction under quantization" unsupported: we could
show 73.8% of Extractor calls select different evidence at 4-bit, but not whether
the new selection was worse.

HotpotQA ships the ground truth for this — `supporting_facts`, a set of
(title, sent_id) labels marking the gold evidence sentences. 99.86% resolve to a
real sentence in the distractor context. We had been discarding it.

WHY IT IS SET-F1 AND NOT TOKEN OVERLAP
--------------------------------------
Token-overlap F1 against sentence-length references is broken in a specific way:
scoring an incomplete verbatim copy above a complete faithful paraphrase, and
scoring a half-copy-plus-fabrication above both, because precision taxes a synonym
exactly as hard as an invented fact. Worked example, gold = a 15-token sentence:

    complete faithful paraphrase   -> F1 0.46
    half copy + fabricated clause  -> F1 0.51
    incomplete verbatim copy       -> F1 0.59   <- best score, least information

So this module never compares text. It resolves each span to the sentence LABEL it
came from and compares label sets. Labels are discrete and unambiguous, so
fabrication cannot earn partial credit and paraphrase is not punished — the metric
is immune to all three failures above by construction.

This is also HotpotQA's own official supporting-facts metric, so the numbers are
comparable to published work.
"""

import re

from .mechanism import _norm


def build_sentence_index(titles: list[str], sentence_lists: list[list[str]]) -> list[dict]:
    """Flatten a question's context into addressable sentences.

    Returns [{"title", "sent_id", "text", "norm"}], covering all 10 paragraphs
    (gold and distractor alike — the Extractor sees all of them).
    """
    out = []
    for title, sents in zip(titles, sentence_lists):
        for i, s in enumerate(sents):
            n = _norm(s)
            if n:
                out.append({"title": title, "sent_id": i, "text": s, "norm": n})
    return out


def gold_evidence(supporting_facts: dict) -> set:
    """The gold (title, sent_id) label set for one question."""
    return set(zip(supporting_facts["title"], supporting_facts["sent_id"]))


# A span shorter than this is not attributed to any sentence. Short spans like
# "Richard Strauss" occur in many sentences, so attributing them would invent
# evidence the Extractor never really located and would corrupt precision. They
# are counted as unattributable instead — see attribution_stats().
MIN_ATTRIBUTABLE_CHARS = 25


def attribute_span(span: str, index: list[dict]) -> set:
    """Resolve one span to the sentence label(s) it was copied from.

    Matches in either direction: the span may be a fragment of a sentence, or may
    have swallowed a whole sentence plus neighbouring text.

    Limitation, verified by test: a span that straddles a sentence boundary only
    claims the sentences it *fully* contains. Given "…directed by Spielberg. It was
    based on a novel." it claims the second sentence but not the first, because
    neither string contains the other. The Extractor is instructed to return single
    sentences so this is an edge case, but it makes recall slightly pessimistic
    rather than optimistic — the safer direction for a metric.

    Returns an empty set when the span is too short to attribute or matches
    nothing (paraphrased or fabricated).
    """
    n = _norm(span)
    if len(n) < MIN_ATTRIBUTABLE_CHARS:
        return set()
    hits = set()
    for s in index:
        sn = s["norm"]
        if n in sn:
            # Span is a fragment of this sentence. The span already cleared the
            # length bar, so this direction is safe.
            hits.add((s["title"], s["sent_id"]))
        elif sn in n and len(sn) >= MIN_ATTRIBUTABLE_CHARS:
            # Span swallowed this whole sentence. SPEC §13b.2: the length guard
            # must apply to the SENTENCE here, not the span. Without it any
            # context sentence shorter than the span could be claimed by a
            # correct span — including sentences in *distractor* paragraphs. A
            # correct verbatim gold span was co-matching a 7-character distractor
            # ("in 1804"), dropping precision from 1.00 to 0.50 on that question.
            # The published MIN_ATTRIBUTABLE_CHARS sweep could not detect this:
            # sweeping the constant only ever moved the span-side guard.
            hits.add((s["title"], s["sent_id"]))
    return hits


def predicted_evidence(spans: list[str], index: list[dict]) -> set:
    """Union of labels across all spans the Extractor returned for a question."""
    out = set()
    for sp in spans:
        out |= attribute_span(sp, index)
    return out


def prf(pred: set, gold: set) -> dict:
    """Set precision/recall/F1 plus exact set match — HotpotQA's official form."""
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "em": 1.0}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "em": float(pred == gold)}


def attribution_stats(spans: list[str], index: list[dict]) -> dict:
    """How trustworthy the attribution was, so the metric's limits stay visible.

    `unattributable` spans (too short, paraphrased, or fabricated) and `ambiguous`
    spans (matching more than one sentence) both weaken the label mapping. Report
    these alongside the scores rather than hiding them.
    """
    n_short = n_none = n_ambig = n_unique = 0
    for sp in spans:
        if len(_norm(sp)) < MIN_ATTRIBUTABLE_CHARS:
            n_short += 1
            continue
        h = attribute_span(sp, index)
        if not h:
            n_none += 1
        elif len(h) > 1:
            n_ambig += 1
        else:
            n_unique += 1
    return {"spans": len(spans), "too_short": n_short, "unmatched": n_none,
            "ambiguous": n_ambig, "unique": n_unique}
