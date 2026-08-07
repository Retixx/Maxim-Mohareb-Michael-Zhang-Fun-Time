"""Prompt templates, one per agent role.

SPEC §4: prompts live in one module, versioned, one per role.
SPEC §12: do NOT tune prompts per precision level. The same template string is
used at FP16, 8-bit and 4-bit. Any change here must bump PROMPT_VERSION and be
re-run across *all* precisions, otherwise it confounds the experiment.
"""

import hashlib

# The open-corpus MA-RAG redesign changes the Planner, Step Definer, and QA
# contracts because downstream steps must consume prior step results.  The
# Extractor remains byte-for-byte v5 and the architecture control remains
# solo-v1.  Every arm in the redesigned experiment uses this same bundle.
PROMPT_VERSION = "marag-v1"
PROMPT_BUNDLE_VERSION = "marag-v1+extractor-v5+solo-v1"

# ==========================================================================
#  PROMPTS ARE FROZEN FOR THE OPEN-CORPUS CAMPAIGN. DO NOT EDIT THEM.
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

# SEED HYGIENE. Prompt development looked at failures from seed 0 (n=10), seed
# 1234 (n=30 / the 68-call replays), and an earlier seed-7 result. Those IDs are
# part of the committed exclusion union. The final n=1,500 cohort is frozen at
# seed 20260805 and is never used for prompt development.
#
# v1 -> v2 (Gate 1 fix, human-approved): added a one-shot worked example and an
# explicit "quote every string value" rule to all four roles.
#
# Historical motivation: at FP16 the QA role parsed at only 70% and
# every failure was the same defect — an unquoted JSON string value, e.g.
# `{"answer": Richard Strauss}`. SPEC §5a sanctions prompt work to bring baseline
# parse failure under 10%.
#
# At v2 the example was added to all four roles, not just the two that failed.
# The experiment compares roles against each other, so unequal prompt engineering
# across roles is itself a confound: giving QA a one-shot example while leaving
# the Planner zero-shot would mean a measured role difference partly reflects
# prompt quality rather than quantization sensitivity. Extractor v5 was later
# reverted to the empirically stronger no-example wording documented above;
# the other role examples remain a single invented question.
#
# The example is invented, NOT drawn from the HotpotQA dev split, so no eval
# question leaks into the prompt.
#
# This is uniform across precisions, which SPEC §12 requires. Any future change
# must bump this version and be re-run at every precision.

# Roles, in pipeline order. A ROLE is one conceptual agent and prompt template;
# a STAGE is one invocation round. A planned step invokes Step Definer,
# Extractor, then QA, so those roles have up to five stage labels while still
# sharing one treatment in the frozen edge campaign.
ROLES = ["planner", "step_definer", "extractor", "qa"]
SOLO_ROLE = "solo"
ALL_ROLES = [*ROLES, SOLO_ROLE]
# A configurable executor would be unbounded in principle; this frozen edge
# campaign uses five as a fail-closed resource ceiling because the official
# MA-RAG examples extend through five steps. The emitted depth and any clamp are
# logged per question.
MAX_PLAN_STEPS = 5
ROUND_ROLES = ("step_definer", "extractor", "qa")


def _round_stage(role: str, step_index: int) -> str:
    if role not in ROUND_ROLES:
        raise KeyError(f"{role!r} is not a repeated MA-RAG role")
    if not 0 <= step_index < MAX_PLAN_STEPS:
        raise IndexError(f"plan step {step_index} is outside [0, {MAX_PLAN_STEPS})")
    return role if step_index == 0 else f"{role}_step{step_index + 1}"


PIPELINE_STAGES = ["planner"]
for _step_index in range(MAX_PLAN_STEPS):
    PIPELINE_STAGES.extend(_round_stage(role, _step_index) for role in ROUND_ROLES)
PLAN_SUMMARY_STAGE = "plan_summary"
PIPELINE_STAGES.append(PLAN_SUMMARY_STAGE)
PIPELINE_STAGES.append(SOLO_ROLE)

STAGE_ROLE = {
    "planner": "planner",
    PLAN_SUMMARY_STAGE: "step_definer",
    SOLO_ROLE: SOLO_ROLE,
}
for _step_index in range(MAX_PLAN_STEPS):
    for _role in ROUND_ROLES:
        STAGE_ROLE[_round_stage(_role, _step_index)] = _role

# A repeated stage is not independently configurable. It mirrors the first
# stage of the same conceptual agent.
STAGE_MIRRORS = {
    _round_stage(role, step_index): role
    for role in ROUND_ROLES
    for step_index in range(1, MAX_PLAN_STEPS)
}
STAGE_MIRRORS[PLAN_SUMMARY_STAGE] = "step_definer"

def role_for(stage: str) -> str:
    """The prompt role a pipeline stage uses."""
    try:
        return STAGE_ROLE[stage]
    except KeyError:
        raise KeyError(
            f"unknown stage {stage!r}; expected one of {PIPELINE_STAGES}"
        ) from None


def stage_for(role: str, step_index: int = 0) -> str:
    """Return the concrete stage label for a conceptual role and plan step."""
    if role in ("planner", SOLO_ROLE):
        if step_index != 0:
            raise IndexError(f"{role} has no repeated plan-step stages")
        return role
    return _round_stage(role, step_index)


def step_index_for(stage: str) -> int | None:
    """Return a repeated stage's zero-based plan index, or ``None``."""
    role = role_for(stage)
    if stage == PLAN_SUMMARY_STAGE or role not in ROUND_ROLES:
        return None
    return next(
        index for index in range(MAX_PLAN_STEPS)
        if _round_stage(role, index) == stage
    )


def stages_for_role(role: str) -> tuple[str, ...]:
    """All concrete stages that execute one conceptual role."""
    if role in ("planner", SOLO_ROLE):
        return (role,)
    if role not in ROUND_ROLES:
        raise KeyError(f"unknown conceptual role {role!r}")
    stages = tuple(_round_stage(role, index) for index in range(MAX_PLAN_STEPS))
    return stages + ((PLAN_SUMMARY_STAGE,) if role == "step_definer" else ())


def prompt_for(stage: str) -> str:
    """Prompt/schema key used by a concrete execution stage."""
    role_for(stage)
    return PLAN_SUMMARY_STAGE if stage == PLAN_SUMMARY_STAGE else role_for(stage)

# Generation budget per role. `truncated` in the failure taxonomy means the
# generation hit exactly this cap without emitting EOS.
MAX_NEW_TOKENS = {
    "planner": 160,
    "step_definer": 160,
    "extractor": 320,
    "qa": 96,
    "plan_summary": 128,
    "solo": 48,
}


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the Planner in a multi-agent question-answering system.
You decompose a multi-hop question into an ordered list of simpler sub-questions.

Rules:
- Emit the shortest sufficient plan: 1 to 5 sub-questions. Never more than 5.
- Each sub-question must be answerable from a short encyclopedia passage.
- Order them so that later sub-questions may depend on earlier answers.
- Make the final sub-question yield the short answer to the overall question.
- Reply with JSON only. No explanation, no markdown fences.
- Every string value must be wrapped in double quotes.

Reply with exactly this shape:
{"analysis": "one short planning sentence", "sub_questions": ["...", "..."]}

Example
Question: Which university did the director of Jaws attend?
JSON:
{"analysis": "Resolve the director, then the university.", "sub_questions": ["Who directed the film Jaws?", "Which university did that director attend?"]}"""

PLANNER_USER = """Question: {question}

JSON:"""


# --------------------------------------------------------------------------
# Step Definer  (format-heavy role, see SPEC §1)
# --------------------------------------------------------------------------

STEP_DEFINER_SYSTEM = """You are the Step Definer in a multi-agent question-answering system.
Given the full plan, one current step, and earlier answers, define the next task.

Rules:
- "type" must be exactly "question-answering" when new documents are needed, or
  "aggregate" when the answer can be computed entirely from prior step answers.
- "task" must be a self-contained question. Resolve references using prior
  answers. For question-answering, this exact task drives retrieval.
- Reply with JSON only. No explanation, no markdown fences.
- Every string value must be wrapped in double quotes.

Reply with exactly this shape:
{"type": "question-answering", "task": "..."}

Example
Overall question: Which university did the director of Jaws attend?
Sub-question: Who directed the film Jaws?
JSON:
{"type": "question-answering", "task": "Who directed the 1975 film Jaws?"}"""

STEP_DEFINER_USER = """Overall question: {question}
Full plan:
{full_plan}
Plan step: {step_number} of {plan_steps}
Current sub-question: {sub_question}
Prior step results:
{prior_state}

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
You answer the current plan step using the evidence collected so far. Your short
answer will be passed to the next agent; the final plan step answers the overall
question.

Rules:
- The answer must be SHORT: a name, a date, a number, a title, or "yes" / "no".
- Never answer in a sentence. Never explain. Never restate the question.
- Base the answer on the evidence. If it is insufficient, set "success" to "no".
- "success" is exactly "yes" or "no". "rating" is an integer confidence
  score from 0 to 10. "analysis" is one short sentence.
- Reply with JSON only. No explanation, no markdown fences.
- The value of "answer" must be wrapped in double quotes, even for a single word.

Reply with exactly this shape:
{"analysis": "...", "answer": "...", "success": "yes", "rating": 8}

Example
Evidence:
1. Who directed the film Jaws?
   - Jaws is a 1975 American thriller film directed by Steven Spielberg.
2. Which university did that director attend?
   - Spielberg attended California State University, Long Beach.
Question: Which university did the director of Jaws attend?
JSON:
{"analysis": "The evidence directly names the university.", "answer": "California State University, Long Beach", "success": "yes", "rating": 10}"""

QA_USER = """Overall question: {question}
Plan step: {step_number} of {plan_steps}
Current sub-question: {sub_question}
Evidence from completed and current steps:
{evidence}

JSON:"""


# --------------------------------------------------------------------------
# Plan summary (same Step Definer agent/treatment, distinct prompt contract)
# --------------------------------------------------------------------------

PLAN_SUMMARY_SYSTEM = """You finalize a multi-agent plan after its step answers.
Use the completed results to answer the original question. If an earlier step
reported failure but the available results still support an answer, succeed.

Rules:
- "output" is exactly "Successful" or "Unsuccessful".
- "answer" is a short name, date, number, title, "yes", "no", or empty string.
- "score" is an integer from 0 to 10; use 0 when unsuccessful.
- Reply with JSON only. No explanation or markdown fences.

Reply with exactly this shape:
{"output": "Successful", "answer": "...", "score": 8}"""

PLAN_SUMMARY_USER = """Original question: {question}
Plan:
{full_plan}
Completed step history:
{prior_state}
Stop reason: {stop_reason}

JSON:"""


# --------------------------------------------------------------------------
# Single-agent control
# --------------------------------------------------------------------------

# Frozen before the final evaluation manifest is generated. The control issues
# one original-question top-10 retrieval and has the same short-answer JSON
# contract and answer budget as the final multi-agent summary. It performs one
# generation per question, so retrieval queries/passages/tokens are co-reported
# rather than pretending the unequal adaptive exposure is budget matched.
SOLO_SYSTEM = """You are a single-agent question-answering system.
Answer the question using the provided encyclopedia paragraphs.

Rules:
- Read all provided paragraphs and combine facts when the question requires multiple hops.
- The answer must be SHORT: a name, a date, a number, a title, or "yes" / "no".
- Never answer in a sentence. Never explain. Never restate the question.
- If the paragraphs are insufficient, give your best short guess anyway.
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
    "planner": PLANNER_SYSTEM,
    "step_definer": STEP_DEFINER_SYSTEM,
    "extractor": EXTRACTOR_SYSTEM,
    "qa": QA_SYSTEM,
    "plan_summary": PLAN_SUMMARY_SYSTEM,
    "solo": SOLO_SYSTEM,
}

USER_PROMPTS = {
    "planner": PLANNER_USER,
    "step_definer": STEP_DEFINER_USER,
    "extractor": EXTRACTOR_USER,
    "qa": QA_USER,
    "plan_summary": PLAN_SUMMARY_USER,
    "solo": SOLO_USER,
}

ROLE_PROMPT_VERSIONS = {
    "planner": "marag-v1",
    "step_definer": "marag-v1",
    "extractor": "v5",
    "qa": "marag-v1",
    "plan_summary": "marag-v1",
    "solo": "solo-v1",
}


def prompt_template_sha256(role: str) -> str:
    """Hash the exact system/user template pair used for ``role``."""
    if role not in SYSTEM_PROMPTS:
        raise KeyError(f"unknown prompt {role!r}; expected one of {tuple(SYSTEM_PROMPTS)}")
    payload = (SYSTEM_PROMPTS[role] + "\0" + USER_PROMPTS[role]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_template_hashes() -> dict[str, str]:
    """Content hashes persisted in every run's metadata."""
    return {role: prompt_template_sha256(role) for role in SYSTEM_PROMPTS}


def build_messages(role: str, **fields) -> list[dict]:
    """Return chat messages for `role` with the template fields filled in."""
    if role not in SYSTEM_PROMPTS:
        raise KeyError(f"unknown prompt {role!r}; expected one of {tuple(SYSTEM_PROMPTS)}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[role]},
        {"role": "user", "content": USER_PROMPTS[role].format(**fields)},
    ]


def format_paragraphs(titles: list[str], sentence_lists: list[list[str]]) -> str:
    """Render exactly the retrieved passages as numbered prompt paragraphs."""
    out = []
    for i, (title, sents) in enumerate(zip(titles, sentence_lists), start=1):
        body = " ".join(s.strip() for s in sents).strip()
        out.append(f"[{i}] {title}: {body}")
    return "\n".join(out)
