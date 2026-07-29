"""Output parsers and the parse-failure taxonomy.

SPEC §5: the parse-failure rate is the mechanism evidence and is as important
as accuracy.

    !!! THERE IS NO RETRY PATH IN THIS MODULE AND THERE MUST NEVER BE ONE. !!!

A failed parse is a *measurement*, not an error to recover from. Retrying a
failed generation — or resampling at a higher temperature, or falling back to a
regex "rescue" that pulls an answer out of malformed output — destroys the
secondary measurement the whole experiment rests on. Classify it, log it, return
None, and let the question score as-is.

The same applies upstream, and more dangerously: **constrained/grammar-based
decoding is forbidden** (SPEC §12). Making the model physically unable to emit
invalid JSON would send every number this module produces to zero and quietly
erase the paper's mechanism argument. If these rates look bad, report them.
"""

import json
import re

# The six labels from SPEC §5. Order here is documentation only.
PARSE_STATUSES = (
    "ok",
    "malformed_json",
    "schema_mismatch",
    "empty_output",
    "truncated",
    "refusal_or_offtopic",
)

# Phrases that mark a refusal or a meta-comment instead of an attempt at the
# task. Only consulted when no JSON object could be located at all.
_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i can not", "i am unable", "i'm unable",
    "i apologize", "i'm sorry", "i am sorry", "as an ai", "as a language model",
    "i do not have", "i don't have access", "cannot assist", "can't help with",
)


def _find_json_blob(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, or None.

    Tolerant of markdown fences and of prose before/after the JSON, because
    those are formatting noise rather than a structural failure. Not tolerant
    of actually-broken JSON — that is what `malformed_json` is for.
    """
    if not text:
        return None
    # Strip markdown fences if present; the content inside is what we want.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    start = None
    opener = closer = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            closer = "}" if ch == "{" else "]"
            break
    if start is None:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # never closed — truncated or malformed


def _is_str_list(v, allow_empty: bool = False) -> bool:
    if not isinstance(v, list):
        return False
    if not v:
        return allow_empty
    return all(isinstance(x, str) and x.strip() for x in v)


# --------------------------------------------------------------------------
# Per-role schema validators.
#
# Each returns (ok: bool, normalised_payload: dict | None).
#
# Convention: required keys must be present with the right type. Extra keys are
# tolerated — the downstream stage only reads what it needs, so an extra key is
# not a functional failure. Semantic quality is deliberately NOT checked here:
# the Extractor's spans are not verified to be verbatim, and the Step Definer's
# answer_type is not checked against its enum. Those are content errors, not
# format errors, and folding them into the taxonomy would make the parse-failure
# rate measure two different things at once.
# --------------------------------------------------------------------------

def _validate_planner(obj):
    if not isinstance(obj, dict):
        return False, None
    subs = obj.get("sub_questions")
    if not _is_str_list(subs):
        return False, None
    return True, {"sub_questions": [s.strip() for s in subs]}


def _validate_step_definer(obj):
    if not isinstance(obj, dict):
        return False, None
    terms = obj.get("search_terms")
    entity = obj.get("target_entity")
    atype = obj.get("answer_type")
    if not _is_str_list(terms):
        return False, None
    if not isinstance(entity, str) or not entity.strip():
        return False, None
    if not isinstance(atype, str) or not atype.strip():
        return False, None
    return True, {
        "search_terms": [t.strip() for t in terms],
        "target_entity": entity.strip(),
        "answer_type": atype.strip().lower(),
    }


def _validate_extractor(obj):
    if not isinstance(obj, dict):
        return False, None
    spans = obj.get("spans")
    # An empty span list is a legitimate answer ("nothing here supports this").
    if not _is_str_list(spans, allow_empty=True):
        return False, None
    return True, {"spans": [s.strip() for s in spans]}


def _validate_qa(obj):
    if not isinstance(obj, dict):
        return False, None
    ans = obj.get("answer")
    if isinstance(ans, (int, float)) and not isinstance(ans, bool):
        ans = str(ans)  # {"answer": 1969} is a well-formed answer
    if not isinstance(ans, str) or not ans.strip():
        return False, None
    return True, {"answer": ans.strip()}


_VALIDATORS = {
    "planner": _validate_planner,
    "step_definer": _validate_step_definer,
    "extractor": _validate_extractor,
    "qa": _validate_qa,
}


def parse_output(role: str, raw_output: str, hit_token_cap: bool) -> tuple[str, dict | None]:
    """Classify one agent call's raw output. Returns (parse_status, parsed).

    `parsed` is None for every status except "ok".
    `hit_token_cap` must be True iff generation stopped on max_new_tokens
    rather than on EOS.

    Precedence, in order:
      1. empty_output        — nothing but whitespace came back
      2. refusal_or_offtopic — no JSON at all, and refusal language present
      3. truncated           — ran out of tokens and the result does not parse.
                               Checked before malformed_json because truncation
                               is the *cause* of the malformation; labelling it
                               malformed would hide a budget problem as a format
                               problem. Note that hitting the cap while still
                               emitting complete valid JSON scores "ok" — the
                               output was usable, which is what we are measuring.
      4. malformed_json      — no locatable JSON, or json.loads failed
      5. schema_mismatch     — valid JSON, wrong fields or wrong types
      6. ok
    """
    if role not in _VALIDATORS:
        raise KeyError(f"unknown role {role!r}")

    text = (raw_output or "").strip()
    if not text:
        return "empty_output", None

    blob = _find_json_blob(text)

    if blob is None:
        lowered = text.lower()
        if any(m in lowered for m in _REFUSAL_MARKERS):
            return "refusal_or_offtopic", None
        if hit_token_cap:
            return "truncated", None
        return "malformed_json", None

    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        if hit_token_cap:
            return "truncated", None
        return "malformed_json", None

    ok, payload = _VALIDATORS[role](obj)
    if not ok:
        if hit_token_cap:
            return "truncated", None
        return "schema_mismatch", None

    return "ok", payload
