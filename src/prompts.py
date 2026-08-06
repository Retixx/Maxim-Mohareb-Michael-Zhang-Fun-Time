"""Prompt templates, one per agent role.

SPEC §4: prompts live in one module, versioned, one per role.
SPEC §12: do NOT tune prompts per precision level. The same template string is
used at FP16, 8-bit and 4-bit. Any change here must bump PROMPT_VERSION and be
re-run across *all* precisions, otherwise it confounds the experiment.
"""

PROMPT_VERSION = "v5"

# ==========================================================================
#  PROMPTS ARE FROZEN AT v5. DO NOT EDIT THEM.
#
#  Human decision, recorded: the four templates below are final for the whole
#  experiment. Any edit invalidates every run already collected, because a
#  measured role difference would then partly reflect prompt changes rather than
#  precision. If you believe a prompt is wrong, stop and raise it — do not edit.
#
#  Related and absolute: no constrained/grammar-based decoding, ever (SPEC §12).
#  It would drive parse failures to zero by construction and delete the paper's
#  mechanism evidence. Prompt wording is the ONLY lever that was ever legitimate
#  here, and it is now closed.
# ==========================================================================
#
# v4 -> v5: reverted the Extractor to its original v1 wording — no worked
# example, and the `{"spans": []}` literal restored.
#
# Measured head-to-head, three variants replayed over the SAME 68 extractor
# calls (FP16, greedy, so fully deterministic):
#
#     v1 original (no example)   65/68 = 95.6%   stray-`[]}` tic  3
#     v4 current  (w/ example)   49/68 = 72.1%   stray-`[]}` tic 13
#     C  bare JSON array         61/68 = 89.7%   stray-`[]}` tic  6
#
# The conclusion is the reverse of what v2 assumed. The one-shot example did not
# help the Extractor, it WAS the regression — 23 points of it — and v1's 95.5%
# on an earlier 22-call sample was real, not small-sample luck. Two intervening
# hypotheses about the cause (v3: the "close the array exactly once" line primed
# the tic; v4: the `{"spans": []}` literal was being welded onto real arrays)
# were both tested and both wrong; they moved the rate by less than noise.
#
# Variant C was authorised for adoption if it won. It did not win, so the object
# schema stays and parsing.py is unchanged.
#
# NOTE ON ASYMMETRY: the Extractor is now the only role without a worked example.
# v2's reasoning for adding examples everywhere — that unequal prompt engineering
# across roles is itself a confound in a per-role comparison — still holds, but
# the requirement it implies is that no role is left *under*-optimised relative
# to the others, not that the templates look alike. Extractor at 95.6% is the
# best wording found for it; forcing structural symmetry would mean knowingly
# shipping a 23-point-worse prompt for one of the four roles under study.
#
# Failure sets are prompt-dependent, not item-dependent: across the four variants
# tested, ZERO calls failed under all of them (Jaccard 0.00, union 24, breakdown
# by number-of-variants-failed {1: 7, 2: 12, 3: 5}). There is no systematically
# unparseable subset of HotpotQA items for this model — so the baseline should be
# reported as a prompt-sensitive rate, not as "N% of items are impossible".

# SEED HYGIENE. Prompt development looked at failures from seed 0 (n=10) and
# seed 1234 (n=30 / the 68-call replays). The n=300 experimental runs therefore
# use a seed prompt work has never seen — `dataset.eval_seed: 7` in
# config/experiment.yaml.
#
# v1 -> v2 (Gate 1 fix, human-approved): added a one-shot worked example and an
# explicit "quote every string value" rule to all four roles.
#
# Motivation: at FP16 the QA role parsed at only 70% (SPEC §5a floor is 90%) and
# every failure was the same defect — an unquoted JSON string value, e.g.
# `{"answer": Richard Strauss}`. SPEC §5a sanctions prompt work to bring baseline
# parse failure under 10%.
#
# The example was added to ALL FOUR roles, not just the two that were failing.
# The experiment compares roles against each other, so unequal prompt engineering
# across roles is itself a confound: giving QA a one-shot example while leaving
# the Planner zero-shot would mean a measured role difference partly reflects
# prompt quality rather than quantization sensitivity. All four now share the
# same instruction scaffold and the same worked example (a single invented
# question, threaded through all four roles).
#
# The example is invented, NOT drawn from the HotpotQA dev split, so no eval
# question leaks into the prompt.
#
# This is uniform across precisions, which SPEC §12 requires. Any future change
# must bump this version and be re-run at every precision.

# Roles, in pipeline order. Used everywhere as the canonical stage names.
ROLES = ["planner", "step_definer", "extractor", "qa", "solo"]

# `solo` is the single-call RAG baseline (SPEC §4a). It is NOT part of the
# four-agent pipeline and never runs alongside the others — a `single_*` run
# defines only this stage. It exists to answer the question the Planner result
# raised: degrading the Planner never hurt accuracy, and at 0.5B it failed to
# parse 37.6% of the time (falling back to no decomposition at all) while
# accuracy went UP. If one call matches four agents, that is the most useful
# finding in the project, and it has to be measured rather than inferred.

# Generation budget per role. `truncated` in the failure taxonomy means the
# generation hit exactly this cap without emitting EOS.
MAX_NEW_TOKENS = {
    "solo": 48,      # same budget as QA: both emit one short answer
    "planner": 160,
    "step_definer": 160,
    "extractor": 320,
    "qa": 48,
}


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the Planner in a multi-agent question-answering system.
You decompose a multi-hop question into an ordered list of simpler sub-questions.

Rules:
- Emit 2 or 3 sub-questions. Never more than 3.
- Each sub-question must be answerable from a short encyclopedia passage.
- Order them so that later sub-questions may depend on earlier answers.
- Reply with JSON only. No explanation, no markdown fences.
- Every string value must be wrapped in double quotes.

Reply with exactly this shape:
{"sub_questions": ["...", "..."]}

Example
Question: Which university did the director of Jaws attend?
JSON:
{"sub_questions": ["Who directed the film Jaws?", "Which university did that director attend?"]}"""

PLANNER_USER = """Question: {question}

JSON:"""


# --------------------------------------------------------------------------
# Step Definer  (format-heavy role, see SPEC §1)
# --------------------------------------------------------------------------

STEP_DEFINER_SYSTEM = """You are the Step Definer in a multi-agent question-answering system.
Given one sub-question, you emit a structured specification telling the Extractor
what evidence to pull from a document set.

Rules:
- "search_terms": 2 to 4 short keyword strings likely to appear in the source text.
- "target_entity": the entity or topic the evidence must be about, as a string.
- "answer_type": exactly one of "person", "place", "date", "number", "title", "other".
- Reply with JSON only. No explanation, no markdown fences.
- Every string value must be wrapped in double quotes.

Reply with exactly this shape:
{"search_terms": ["...", "..."], "target_entity": "...", "answer_type": "..."}

Example
Overall question: Which university did the director of Jaws attend?
Sub-question: Who directed the film Jaws?
JSON:
{"search_terms": ["Jaws", "directed by", "1975 film"], "target_entity": "Jaws", "answer_type": "person"}"""

STEP_DEFINER_USER = """Overall question: {question}
Sub-question: {sub_question}

JSON:"""


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------

EXTRACTOR_SYSTEM = """You are the Extractor in a multi-agent question-answering system.
You return supporting evidence copied VERBATIM from the provided paragraphs.

Rules:
- Every string in "spans" must be an exact substring of the paragraphs above. Copy, never paraphrase.
- Return 1 to 3 spans, each a single sentence.
- If no paragraph supports the sub-question, return {"spans": []}.
- Never invent facts that are not in the paragraphs.
- Reply with JSON only. No explanation, no markdown fences.

Reply with exactly this shape:
{"spans": ["...", "..."]}"""
# ^ Deliberately has NO worked example, unlike the other three roles. Measured,
# not stylistic: see the v4 -> v5 note at the top of this file. Do not "make it
# consistent" with the others.

EXTRACTOR_USER = """Paragraphs:
{paragraphs}

Sub-question: {sub_question}
Look for: {target_entity}
Keywords: {search_terms}

JSON:"""


# --------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------

QA_SYSTEM = """You are the QA agent in a multi-agent question-answering system.
You give the final answer using the collected evidence.

Rules:
- The answer must be SHORT: a name, a date, a number, a title, or "yes" / "no".
- Never answer in a sentence. Never explain. Never restate the question.
- Base the answer on the evidence. If the evidence is insufficient, give your best short guess anyway.
- Reply with JSON only. No explanation, no markdown fences.
- The value of "answer" must be wrapped in double quotes, even for a single word.

Reply with exactly this shape:
{"answer": "..."}

Example
Evidence:
1. Who directed the film Jaws?
   - Jaws is a 1975 American thriller film directed by Steven Spielberg.
2. Which university did that director attend?
   - Spielberg attended California State University, Long Beach.
Question: Which university did the director of Jaws attend?
JSON:
{"answer": "California State University, Long Beach"}"""

QA_USER = """Evidence:
{evidence}

Question: {question}

JSON:"""


# --------------------------------------------------------------------------
# Solo  (single-call RAG baseline, SPEC §4a)
# --------------------------------------------------------------------------
# Deliberately mirrors QA's contract, rules and worked example so the comparison
# isolates DECOMPOSITION, not prompt quality. The only difference is what it
# reads: raw paragraphs instead of extracted evidence. Same output schema, same
# parser, same failure taxonomy, same token budget.

SOLO_SYSTEM = """You are a question-answering system.
You give the final answer using the provided paragraphs.

Rules:
- The answer must be SHORT: a name, a date, a number, a title, or "yes" / "no".
- Never answer in a sentence. Never explain. Never restate the question.
- Base the answer on the paragraphs. If they are insufficient, give your best short guess anyway.
- Reply with JSON only. No explanation, no markdown fences.
- The value of "answer" must be wrapped in double quotes, even for a single word.

Reply with exactly this shape:
{"answer": "..."}

Example
Paragraphs:
[1] Jaws: Jaws is a 1975 American thriller film directed by Steven Spielberg.
[2] Steven Spielberg: Spielberg attended California State University, Long Beach.
Question: Which university did the director of Jaws attend?
JSON:
{"answer": "California State University, Long Beach"}"""

SOLO_USER = """Paragraphs:
{paragraphs}

Question: {question}

JSON:"""


SYSTEM_PROMPTS = {
    "solo": SOLO_SYSTEM,
    "planner": PLANNER_SYSTEM,
    "step_definer": STEP_DEFINER_SYSTEM,
    "extractor": EXTRACTOR_SYSTEM,
    "qa": QA_SYSTEM,
}

USER_PROMPTS = {
    "solo": SOLO_USER,
    "planner": PLANNER_USER,
    "step_definer": STEP_DEFINER_USER,
    "extractor": EXTRACTOR_USER,
    "qa": QA_USER,
}


def build_messages(role: str, **fields) -> list[dict]:
    """Return chat messages for `role` with the template fields filled in."""
    if role not in SYSTEM_PROMPTS:
        raise KeyError(f"unknown role {role!r}; expected one of {ROLES}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[role]},
        {"role": "user", "content": USER_PROMPTS[role].format(**fields)},
    ]


def format_paragraphs(titles: list[str], sentence_lists: list[list[str]]) -> str:
    """Render HotpotQA distractor context as numbered paragraphs.

    SPEC §3: no retriever. All 10 paragraphs (2 gold + 8 distractors) go in as-is.
    """
    out = []
    for i, (title, sents) in enumerate(zip(titles, sentence_lists), start=1):
        body = " ".join(s.strip() for s in sents).strip()
        out.append(f"[{i}] {title}: {body}")
    return "\n".join(out)
