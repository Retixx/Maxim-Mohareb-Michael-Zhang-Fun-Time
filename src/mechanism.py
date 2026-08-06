"""Mechanism metrics — the instruments that test SPEC §1's mechanism claim.

SPEC §5's parse-failure taxonomy answers a coarse question: did the output parse?
That turned out to be too blunt to test the mechanism. This module adds three
sharper instruments, all computed from fields already in the JSONL — no extra
generation, no GPU.

    strict_format_ok   require the entire output to be the accepted JSON object,
                       WITHOUT the fence/prose tolerance parsing.py grants. The
                       role validators still intentionally tolerate extra keys
                       and coerce a numeric QA answer; this is container-format
                       compliance, not an exact-schema validator.
    verbatim_rate      fraction of Extractor spans that are exact substrings of
                       the paragraphs given to it. parsing.py deliberately does
                       not check this (a content error, not a format error), so
                       paraphrasing-instead-of-copying is invisible to the
                       taxonomy AND to EM.
    selection_churn    fraction of calls whose chosen content changed between two
                       runs, independent of whether it was well-formed.

The current campaign reports these instruments for every concrete stage and
also rolls repeated stages up to the four conceptual agents.  The 0.5B
enumeration failsafe uses the question-clustered lower confidence bound of
``protocol_ok``; surviving candidates must still pass the selector's paired-F1
noninferiority constraint, so format compliance alone cannot win an allocation.
"""

import json
import re

from .parsing import _VALIDATORS


def strict_format_ok(role: str, raw_output: str) -> bool:
    """True iff the whole raw output is accepted JSON for the role.

    No fence stripping, no blob extraction, and no surrounding-prose tolerance.
    Extra keys remain allowed and numeric QA answers remain coercible because
    those are documented parser conventions, not container-format failures. The
    gap between this and ``parse_status == "ok"`` therefore isolates wrapper and
    chatter tolerance rather than claiming exact JSON-schema compliance.
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


def protocol_ok(
    role: str,
    raw_output: str,
    parsed: dict | None,
    *,
    paragraphs: str | None = None,
    max_plan_steps: int = 5,
) -> bool:
    """Strict functional-output compliance, separate from tolerant parsing."""
    if parsed is None or not strict_format_ok(role, raw_output):
        return False
    if role == "planner":
        steps = parsed.get("sub_questions") or []
        return 1 <= len(steps) <= max_plan_steps
    if role == "step_definer":
        return parsed.get("type") in {"aggregate", "question-answering"} and bool(
            parsed.get("task")
        )
    if role == "extractor":
        spans = parsed.get("spans") or []
        fidelity = verbatim_rate(spans, paragraphs or "")
        return len(spans) <= 3 and (fidelity is None or fidelity == 1.0)
    if role == "qa":
        answer = parsed.get("answer") or ""
        return len(answer.split()) <= 12 and parsed.get("success") in {"yes", "no"}
    if role == "plan_summary":
        answer = parsed.get("answer") or ""
        return (
            parsed.get("output") in {"Successful", "Unsuccessful"}
            and len(answer.split()) <= 12
        )
    if role == "solo":
        return len((parsed.get("answer") or "").split()) <= 12
    return False


def selection_changed(rec_a: dict, rec_b: dict, key: str) -> bool:
    """Did the chosen content differ between two records of the same call?

    `key` is the parsed field carrying the selection: "spans" for the Extractor,
    "sub_questions" for the Planner, "task" for the Step Definer,
    "answer" for QA. Compares normalised content, so formatting-only differences
    do not count.

    QA's field is a scalar, not a list. Iterating it would compare lists of
    *characters* — which happens to give the right answer most of the time and
    the wrong one whenever the two strings differ only in whitespace, since
    `_norm` maps a lone space to "" but collapses interior runs. Scalars are
    normalised whole.
    """
    def sel(rec):
        v = (rec.get("parsed") or {}).get(key)
        if v is None:
            return []
        return [_norm(x) for x in v] if isinstance(v, list) else [_norm(v)]

    return sel(rec_a) != sel(rec_b)


def selection_changed_set(rec_a: dict, rec_b: dict, key: str) -> bool:
    """Set-based counterpart to `selection_changed`. SPEC §13b.3.

    `selection_changed` compares ordered lists, so a pure REORDERING of the same
    spans counts as churn — which makes the published 73.8% / 75.9% figures upper
    bounds by an unmeasured amount. SPEC §5b prediction 2 is worded as a
    "different span *set*", and that wording is pre-registered, so it cannot be
    edited after seeing data to match whichever number is more convenient.

    Both are therefore reported: the sequence figure for continuity with what was
    published, and this one for the prediction as actually written.
    """
    def sel(rec):
        v = (rec.get("parsed") or {}).get(key)
        if v is None:
            return frozenset()
        return frozenset(_norm(x) for x in v) if isinstance(v, list) else frozenset([_norm(v)])

    return sel(rec_a) != sel(rec_b)


# The selection field each role's output is judged on.
SELECTION_FIELD = {
    "planner": "sub_questions",
    "step_definer": "task",
    "extractor": "spans",
    "qa": "answer",
    "solo": "answer",
}
