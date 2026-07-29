"""Prompt templates, one per agent role.

SPEC §4: prompts live in one module, versioned, one per role.
SPEC §12: do NOT tune prompts per precision level. The same template string is
used at FP16, 8-bit and 4-bit. Any change here must bump PROMPT_VERSION and be
re-run across *all* precisions, otherwise it confounds the experiment.
"""

PROMPT_VERSION = "v4"

# SEED HYGIENE. Prompt development has looked at failures from seed 0 (n=10) and
# seed 1234 (n=30). The n=300 experimental runs therefore use a seed that prompt
# work has never seen — `dataset.eval_seed: 7` in config/experiment.yaml. Iterate
# on 0 or 1234; never on 7.
#
# v3 -> v4: removed the literal `{"spans": []}` from the Extractor's rules.
# Measured cause, not a guess: 11 of 17 v3 extractor failures were a stray empty
# array welded onto a real one, `{"spans": [...][]}`, and that exact `[]}`
# sequence appeared nowhere in the model's output space except that rule. The
# model was blending the memorized empty-case literal onto the end of a populated
# array. The rule now states the empty case in words instead.
#
# v2 -> v3 (did NOT work, kept for the record): hypothesised that the v2 line
# "Close the array exactly once" primed the tic. Dropping it changed the
# extractor parse rate from 77.3% to 75.0% — no effect. The two-span example and
# "stop after the closing brace" introduced in v3 are retained as harmless.
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
ROLES = ["planner", "step_definer", "extractor", "qa"]

# Generation budget per role. `truncated` in the failure taxonomy means the
# generation hit exactly this cap without emitting EOS.
MAX_NEW_TOKENS = {
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
- If no paragraph supports the sub-question, return an empty list of spans.
- Never invent facts that are not in the paragraphs.
- Reply with JSON only. No explanation, no markdown fences.
- Every string value must be wrapped in double quotes.
- Stop immediately after the closing brace. Output nothing after it.

Reply with exactly this shape:
{"spans": ["...", "..."]}

Example
Paragraphs:
[1] Jaws: Jaws is a 1975 American thriller film directed by Steven Spielberg. It is based on the 1974 novel by Peter Benchley.
[2] Peter Benchley: Peter Bradford Benchley was an American author born in New York City.
Sub-question: Who directed the film Jaws, and who wrote the novel?
JSON:
{"spans": ["Jaws is a 1975 American thriller film directed by Steven Spielberg.", "It is based on the 1974 novel by Peter Benchley."]}"""

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


SYSTEM_PROMPTS = {
    "planner": PLANNER_SYSTEM,
    "step_definer": STEP_DEFINER_SYSTEM,
    "extractor": EXTRACTOR_SYSTEM,
    "qa": QA_SYSTEM,
}

USER_PROMPTS = {
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
