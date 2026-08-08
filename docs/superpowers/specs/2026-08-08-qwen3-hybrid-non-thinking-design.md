# Qwen3 hybrid non-thinking execution design

**Date:** 2026-08-08
**Scope:** Model-family and chat-rendering integrity only. Retrieval, frozen
cohorts, manifests, model revisions, and SPEC §16 additive work are excluded.

## Diagnosis

The Qwen3 migration declared `thinking_mode: false` in
`config/experiment.yaml`, but production Python never reads that field. The
three production chat-template call sites omit `enable_thinking`, so the Qwen3
tokenizer default enables thinking during actual generation, batch ordering,
and preflight sizing.

All model-backed roles converge on the defective generation path: planner,
step definer, extractor, QA, plan summary, and solo. All five configured model
sizes are affected. This is a hidden experimental variable because thinking
tokens consume each role's frozen output budget and change latency, parsing,
and truncation behavior.

Live run evidence confirms the effect rather than merely implying it from the
source: 204 of 256 planner outputs contained `<think>` and planner parse-OK was
0.000; 100 of 126 step-definer outputs contained `<think>` and parse-OK was
0.063. Both roles were observed truncating mid-reasoning against their frozen
token ceilings.

## Model-family invariant

Every arm must use the original April 2025 Qwen3 unified checkpoints that can
switch between thinking and non-thinking behavior. The exact allowed mapping
is:

| Tier | Repository |
|---|---|
| large | `Qwen/Qwen3-14B` |
| base | `Qwen/Qwen3-8B` |
| mid | `Qwen/Qwen3-4B` |
| small | `Qwen/Qwen3-1.7B` |
| tiny | `Qwen/Qwen3-0.6B` |

Later dedicated Instruct or Thinking repositories are not equivalent and are
rejected. Literal per-run repositories and CLI substitutions are also rejected
unless they resolve to the exact mapping above. Parameter size and numerical
precision remain experimental variables; checkpoint subtype and training
lineage do not.

The authoritative mapping lives in the immutable contracts module. Runner and
campaign validation compare the full configured mapping before any model or
dataset work. The existing unresolved revision markers remain unchanged in
this pass.

## Non-thinking rendering

`src.models.render_chat` is the only production chat-template renderer. It
always invokes:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

Actual generation, largest-first batch ordering, and preflight token sizing all
use this helper. Prompts are unchanged, no `/no_think` suffix is added, and the
tokenizer or model is not mutated. The runner fails closed unless the config
contains the exact boolean `thinking_mode: false`.

## Artifact identity

The experiment schema advances so old thinking-on artifacts cannot resume or
be discovered as current. The experiment fingerprint payload and top-level
metadata record:

- `thinking_mode: false`;
- the exact Qwen3 hybrid family mapping.

Pilot and campaign validators compare both fields against the active config and
immutable family contract. Scientific JSONL records remain tied to that payload
through their experiment fingerprint.

## Regression proof

CPU-only tests must prove:

1. the authoritative config has the exact five-model family and false thinking
   mode;
2. missing/true thinking mode, a dedicated checkpoint variant, an alias drift,
   or a literal foreign run model is rejected;
3. generation passes `enable_thinking=False` for every rendered conversation;
4. batch ordering and preflight rendering use the same helper;
5. smoke derivation and validation preserve the contract;
6. experiment fingerprints and metadata pin both invariants;
7. pilot and campaign discovery reject absent, true, or stale values.

No dataset block, manifest, retrieval policy, plan ceiling, retrieval `k`, or
manifest hash changes.
