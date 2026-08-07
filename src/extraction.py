"""Deterministic Extractor-output normalization against source sentences.

The model output and parse status remain untouched. This module produces the
payload consumed by downstream agents and explicit telemetry describing every
accepted or rejected candidate span.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


MIN_FRAGMENT_CHARS = 25
MAX_SPANS = 3


def _norm(value: str) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", value or "")
    ).strip().casefold()


def normalize_spans(
    spans: list[str], source_sentences: list[str], *, limit: int = MAX_SPANS
) -> tuple[list[str], dict]:
    """Map candidate spans to unambiguous exact source sentences.

    Exact short sentences are allowed. A non-exact fragment must contain at
    least ``MIN_FRAGMENT_CHARS`` normalized characters and identify exactly one
    source sentence. A candidate that contains multiple complete sentences is
    a passage echo and is rejected rather than truncated arbitrarily.
    """
    if limit <= 0:
        raise ValueError(f"span limit must be positive; got {limit}")

    sources = [
        sentence.strip()
        for sentence in source_sentences
        if isinstance(sentence, str) and sentence.strip()
    ]
    normalized_sources = [_norm(sentence) for sentence in sources]
    accepted: list[str] = []
    seen: set[str] = set()
    reasons: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    accepted_inputs = 0

    for candidate in spans or []:
        if not isinstance(candidate, str) or not candidate.strip():
            reasons["empty_or_non_string"] += 1
            continue
        candidate_norm = _norm(candidate)

        exact = [
            index
            for index, source_norm in enumerate(normalized_sources)
            if candidate_norm == source_norm
        ]
        if len(exact) == 1:
            matches = exact
            mode = "exact"
        elif len(exact) > 1:
            reasons["ambiguous_sentence"] += 1
            continue
        else:
            contained_sources = [
                index
                for index, source_norm in enumerate(normalized_sources)
                if source_norm and source_norm in candidate_norm
            ]
            if len(contained_sources) > 1:
                reasons["multiple_sentences"] += 1
                continue
            fragment_matches = (
                [
                    index
                    for index, source_norm in enumerate(normalized_sources)
                    if candidate_norm in source_norm
                ]
                if len(candidate_norm) >= MIN_FRAGMENT_CHARS
                else []
            )
            matches = sorted(set(contained_sources + fragment_matches))
            if not matches:
                reasons[
                    "fragment_too_short"
                    if len(candidate_norm) < MIN_FRAGMENT_CHARS
                    else "not_in_source"
                ] += 1
                continue
            if len(matches) > 1:
                reasons["ambiguous_sentence"] += 1
                continue
            mode = "enclosing" if contained_sources else "fragment"

        sentence = sources[matches[0]]
        sentence_key = _norm(sentence)
        if sentence_key in seen:
            reasons["duplicate_sentence"] += 1
            continue
        if len(accepted) >= limit:
            reasons["over_limit"] += 1
            continue
        seen.add(sentence_key)
        accepted.append(sentence)
        accepted_inputs += 1
        modes[mode] += 1

    telemetry = {
        "source_sentence_count": len(sources),
        "input_span_count": len(spans or []),
        "accepted_input_count": accepted_inputs,
        "accepted_span_count": len(accepted),
        "rejected_input_count": sum(reasons.values()),
        "rejection_reasons": dict(sorted(reasons.items())),
        "normalization_modes": dict(sorted(modes.items())),
        "max_spans": limit,
    }
    return accepted, telemetry
