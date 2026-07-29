"""Prompt templates, one per agent role.

SPEC §4: prompts live in one module, versioned, one per role.
SPEC §12: do NOT tune prompts per precision level. The same template string is
used at FP16, 8-bit and 4-bit. Any change here must bump PROMPT_VERSION and be
re-run across *all* precisions, otherwise it confounds the experiment.
"""

PROMPT_VERSION = "v1"

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

Reply with exactly this shape:
{"sub_questions": ["...", "..."]}"""

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

Reply with exactly this shape:
{"search_terms": ["...", "..."], "target_entity": "...", "answer_type": "..."}"""

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

Reply with exactly this shape:
{"answer": "..."}"""

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
