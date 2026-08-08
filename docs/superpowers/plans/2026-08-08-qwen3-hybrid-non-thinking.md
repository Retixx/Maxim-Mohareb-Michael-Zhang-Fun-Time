# Qwen3 Hybrid Non-Thinking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce one original Qwen3 hybrid-capable model lineage across every
parameter-size arm and disable thinking explicitly in every execution renderer.

**Architecture:** Put the exact five-repository mapping in the immutable
contract, validate it before execution, and route generation plus auxiliary
token sizing through one renderer that always passes `enable_thinking=False`.
Fingerprint the family and mode so stale thinking-on artifacts fail closed.

**Tech Stack:** Python 3, PyTorch, Hugging Face tokenizers/transformers,
PyYAML, unittest/pytest, JSONL metadata.

---

### Task 1: Pin model-family and rendering regressions

**Files:**
- Create: `tests/test_thinking_mode.py`
- Modify: `tests/test_execution_integrity.py`
- Modify: `tests/test_local_smoke_gate.py`

- [ ] **Step 1: Add a tokenizer spy and CPU fake generation model**

The spy records every chat-template keyword and returns deterministic tensors;
the fake model appends one generated token. Use them to call
`src.models.generate_batch` without a GPU or transformers download.

- [ ] **Step 2: Assert explicit non-thinking rendering**

```python
rows = models.generate_batch(model, tokenizer, conversations, 1, batch_size=2)
self.assertEqual(len(rows), len(conversations))
self.assertTrue(tokenizer.template_calls)
self.assertTrue(all(call["enable_thinking"] is False
                    for call in tokenizer.template_calls))
```

- [ ] **Step 3: Assert the exact hybrid family contract**

Load `config/experiment.yaml`, validate it, then mutate one field at a time.
Missing/true `thinking_mode`, `Qwen/Qwen3-8B-Instruct-2507`, a wrong alias, and
a literal foreign run model must each raise `ValueError`.

- [ ] **Step 4: Assert no runner-local template rendering remains**

Exercise largest-first ordering with a renderer spy and inspect the runner AST
so production calls to `apply_chat_template` can only reside in
`src.models.render_chat`.

- [ ] **Step 5: Run the focused tests and confirm red**

Run:

```bash
python -m pytest tests/test_thinking_mode.py tests/test_execution_integrity.py tests/test_local_smoke_gate.py -q
```

Expected: failures identify the missing renderer, family validator, fingerprint
fields, and smoke contract.

### Task 2: Implement the immutable family and renderer

**Files:**
- Modify: `src/contracts.py`
- Modify: `src/models.py`
- Modify: `src/runner.py`
- Modify: `scripts/run_retrieval_smoke.py`

- [ ] **Step 1: Define the exact family contract**

```python
QWEN3_HYBRID_MODELS = {
    "large": "Qwen/Qwen3-14B",
    "base": "Qwen/Qwen3-8B",
    "mid": "Qwen/Qwen3-4B",
    "small": "Qwen/Qwen3-1.7B",
    "tiny": "Qwen/Qwen3-0.6B",
}
QWEN3_THINKING_MODE = False
```

- [ ] **Step 2: Add one production renderer**

```python
def render_chat(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
```

- [ ] **Step 3: Replace all three production render paths**

Use `models.render_chat` in `generate_batch`,
`_order_batches_largest_first`, and `_preflight_stage.rendered_tokens`.

- [ ] **Step 4: Validate config before treatment resolution**

Require the exact base ID, exact alias mapping, exact false boolean, approved
model names in every run definition, and revision keys for all five repositories.
Call the validator before data/model work and from smoke validation.

- [ ] **Step 5: Run the focused tests until green**

Run the Task 1 command. Expected: all selected tests pass.

### Task 3: Pin mode and family in artifact identity

**Files:**
- Modify: `src/contracts.py`
- Modify: `src/runner.py`
- Modify: `scripts/check_pilot.py`
- Modify: `scripts/run_campaign.py`
- Modify: `tests/test_pilot_gate.py`
- Modify: `tests/test_campaign.py`
- Modify: `tests/test_analyze.py`

- [ ] **Step 1: Advance the experiment schema**

Change the experiment schema identifier from version 3 to version 4 so prior
artifacts cannot resume under the repaired renderer.

- [ ] **Step 2: Extend the experiment fingerprint payload**

```python
"thinking_mode": False,
"model_family": {
    "name": "Qwen3-hybrid",
    "models": QWEN3_HYBRID_MODELS,
},
```

Write the same false boolean at top-level metadata and make it immutable during
resume.

- [ ] **Step 3: Enforce identity during pilot and campaign discovery**

Require top-level false, payload false, family name, and exact family mapping.
Reject missing fields, true mode, or any mapping drift.

- [ ] **Step 4: Update fixture payloads and add stale-artifact tests**

Every synthetic current artifact includes the repaired fields. Dedicated tests
mutate each field and assert that pilot loading or campaign completion discovery
rejects it.

- [ ] **Step 5: Run focused provenance tests**

Run:

```bash
python -m pytest tests/test_pilot_gate.py tests/test_campaign.py tests/test_analyze.py -q
```

Expected: all selected tests pass.

### Task 4: Verify, document, commit, and bundle

**Files:**
- Modify surgically: `SPEC.md`
- Create: `handoff/multihop-rag-fix-thinking-off.bundle`

- [ ] **Step 1: Run the entire CPU suite**

```bash
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Audit protected experiment inputs**

```bash
git diff f92391b..HEAD -- config/manifests/
git diff --check
git status --short
```

Expected: no manifest diff, no whitespace errors, and only intentional tracked
changes plus preserved pre-existing untracked artifacts.

- [ ] **Step 3: Append the diagnosis to SPEC**

Record that YAML-only configuration left thinking enabled at all runtime render
sites, identify the affected roles/sizes, state the exact family invariant, and
record fresh test totals. Do not edit retrieval behavior or start additive work.

- [ ] **Step 4: Commit scoped changes**

Commit tests separately from implementation/provenance where practical, then
commit the SPEC audit. Use messages that identify the Qwen3 non-thinking
contract.

- [ ] **Step 5: Create and verify a complete branch bundle**

```bash
git bundle create handoff/multihop-rag-fix-thinking-off.bundle multihop-vs-single-hop-rag-bug-fix
git bundle verify handoff/multihop-rag-fix-thinking-off.bundle
sha256sum handoff/multihop-rag-fix-thinking-off.bundle
```

Expected: verification reports a complete bundle and the feature branch ref at
the final commit.
