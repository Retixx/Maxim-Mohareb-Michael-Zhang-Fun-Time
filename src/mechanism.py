"""Mechanism metrics — the instruments that test SPEC §1's mechanism claim.

SPEC §5's parse-failure taxonomy answers a coarse question: did the output parse?
That turned out to be too blunt to test the mechanism. This module adds three
sharper instruments, all computed from fields already in the JSONL — no extra
generation, no GPU.

    strict_format_ok   parse WITHOUT the tolerance parsing.py grants. The real
                       parser strips markdown fences and surrounding prose, so a
                       model that starts padding its JSON with chatter still
                       scores `ok`. This catches that.
    verbatim_rate      fraction of Extractor spans that are exact substrings of
                       the paragraphs given to it. parsing.py deliberately does
                       not check this (a content error, not a format error), so
                       paraphrasing-instead-of-copying is invisible to the
                       taxonomy AND to EM.
    selection_churn    fraction of calls whose chosen content changed between two
                       runs, independent of whether it was well-formed.

At 4-bit on Qwen2.5-1.5B the first two showed no degradation and the third was
73.8%. See the pre-registered predictions in SPEC §5b before adding a fourth
instrument — "measure format a different way until it breaks" is not a method.
"""

import json
import re

from .parsing import _VALIDATORS


def strict_format_ok(role: str, raw_output: str) -> bool:
    """True iff the whole raw output is exactly valid JSON of the right schema.

    No fence stripping, no blob extraction, no prose tolerance. The gap between
    this and parse_status == "ok" is precisely what the parser's leniency hides.
    """
    text = (raw_output or "").strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return _VALIDATORS[role](obj)[0]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def verbatim_rate(spans: list[str], paragraphs: str) -> float | None:
    """Fraction of `spans` that appear verbatim in `paragraphs`.

    Whitespace-collapsed, case-insensitive — we are testing copying fidelity, not
    the model's shift key. Returns None when there are no spans to score, so
    empty-evidence calls do not silently count as perfect fidelity.
    """
    if not spans:
        return None
    hay = _norm(paragraphs)
    return sum(1 for s in spans if _norm(s) in hay) / len(spans)


def selection_changed(rec_a: dict, rec_b: dict, key: str) -> bool:
    """Did the chosen content differ between two records of the same call?

    `key` is the parsed field carrying the selection: "spans" for the Extractor,
    "sub_questions" for the Planner, "search_terms" for the Step Definer.
    Compares normalised content, so formatting-only differences do not count.
    """
    a = (rec_a.get("parsed") or {}).get(key) or []
    b = (rec_b.get("parsed") or {}).get(key) or []
    return [_norm(x) for x in a] != [_norm(x) for x in b]


# The selection field each role's output is judged on.
SELECTION_FIELD = {
    "planner": "sub_questions",
    "step_definer": "search_terms",
    "extractor": "spans",
    "qa": "answer",
}
