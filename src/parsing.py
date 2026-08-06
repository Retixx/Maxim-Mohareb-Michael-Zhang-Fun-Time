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


def _balanced_from(text: str, start: int) -> str | None:
    """The balanced {...} / [...] substring beginning at `start`, or None."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
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


def _iter_json_blobs(text: str):
    """Yield candidate balanced JSON substrings, best-first.

    Tolerant of markdown fences and of prose before/after the JSON, because
    those are formatting noise rather than a structural failure. Not tolerant
    of actually-broken JSON — that is what `malformed_json` is for.

    Two things this deliberately does NOT do, both of which it used to:

    1. **It no longer throws away everything outside the first fence.** The old
       code replaced the whole output with the first ```...``` block, so a model
       that reasoned inside a fence and then emitted correct JSON after it scored
       `malformed_json`. Fenced content is still tried first — it is the most
       likely place for the payload — but the full text is tried after.

    2. **It no longer commits to the first `{` or `[` in the string.** A brace in
       prose ("use the shape {sub_questions: [...]}") or a preamble object
       ({"note": "thinking"}) hijacked the parse, in the second case producing
       `schema_mismatch` for a model whose very next object had exactly the right
       fields. Every opening bracket is now a candidate and the caller takes the
       first that satisfies the role's schema.

    Both were latent on model 1 — replaying all 23,486 model-1 call records
    produces byte-identical statuses before and after — but they penalise
    *chatty* models specifically. Phase S's 0.5B model and Llama-3.2 are chattier
    than Qwen2.5-1.5B, so the old behaviour would have surfaced as format damage
    under the treatment, i.e. as a spurious confirmation of SPEC §5b prediction 5.
    """
    if not text:
        return
    seen = set()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for chunk in [*fenced, text]:
        i = 0
        while i < len(chunk):
            if chunk[i] not in "{[":
                i += 1
                continue
            blob = _balanced_from(chunk, i)
            if blob is None:
                # Unterminated from here. Stop scanning this chunk rather than
                # descending inside it: a truncated object very often contains a
                # COMPLETE inner array, and treating that as the payload would
                # relabel a genuine `truncated` as `schema_mismatch`. Replaying
                # the model-1 corpus, descending changed 388 published statuses
                # exactly this way. Candidates must be top level.
                break
            if blob not in seen:
                seen.add(blob)
                yield blob
            i += len(blob)  # continue AFTER this blob, never inside it


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
    # SPEC §4a: the single-call baseline emits the same {"answer": "..."} shape
    # as QA, so it reuses QA's validator. Identical contract, identical taxonomy
    # — the comparison isolates decomposition, not output format.
    "solo": _validate_qa,
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
      3. truncated           — ran out of tokens and nothing DECODED. Checked
                               before malformed_json because truncation is the
                               *cause* of the malformation; labelling it
                               malformed would hide a budget problem as a format
                               problem. Note that hitting the cap while still
                               emitting complete valid JSON scores "ok" — the
                               output was usable, which is what we are measuring.
      4. malformed_json      — no locatable JSON, or json.loads failed
      5. schema_mismatch     — valid JSON, wrong fields or wrong types
      6. ok

    `truncated` outranks `malformed_json` but NOT `schema_mismatch`. If a blob
    decoded cleanly and merely had the wrong fields, truncation did not cause
    that — the model emitted a complete, well-formed object of the wrong shape,
    and the cap being hit alongside it is a coincidence. The old code returned
    `truncated` for that case, which contradicted the precedence documented right
    here and bled schema errors into the truncation bucket for any role whose cap
    bites (QA's is 48 tokens). Verified not to fire on model 1: no QA call reached
    its cap at n=750.
    """
    if role not in _VALIDATORS:
        raise KeyError(f"unknown role {role!r}")

    text = (raw_output or "").strip()
    if not text:
        return "empty_output", None

    # Take the first candidate satisfying the role's schema. Remember whether
    # anything decoded at all, to tell "wrong shape" from "not JSON".
    decoded_something = False
    for blob in _iter_json_blobs(text):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        decoded_something = True
        ok, payload = _VALIDATORS[role](obj)
        if ok:
            return "ok", payload

    if decoded_something:
        return "schema_mismatch", None

    lowered = text.lower()
    if any(m in lowered for m in _REFUSAL_MARKERS):
        return "refusal_or_offtopic", None
    if hit_token_cap:
        return "truncated", None
    return "malformed_json", None
