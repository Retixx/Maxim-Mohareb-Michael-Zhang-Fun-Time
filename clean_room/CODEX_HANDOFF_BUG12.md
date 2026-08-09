# Codex handoff — Gate C fails on a mechanically healthy pipeline

Evidence measured at `56cd63f`: `analysis/local_smoke/small_4bit_bs4/`
(Kaggle T4, Qwen3-1.7B 4-bit, frozen excluded n=200, batch 4).
Branch head is now `bddec43`.

## Read this first: `bddec43` optimizes a variable that is no longer binding

`bddec43` adds BUG-4 entity linking and reports hidden_bridge both-gold
recall@10 rising 0.517 → 0.671 (oracle 0.869). That is real and the measurement
is sound. It will not move Gate C.

The decisive figure below is computed on the **128 of 200 questions where both
gold passages were already retrieved**. On those, with the evidence fully in
hand, MA scores 0.3141 against single-hop's 0.5133. Improving retrieval moves
more questions into that bucket; it does not change the outcome once they are
there. So:

    MA ceiling at 100% both-gold retrieval   ~0.3141
    single-hop, measured today                0.4213

Even perfect retrieval leaves MA roughly 10 points below where single-hop
already is. **Further retrieval work cannot pass Gate C.** The remaining loss is
entirely downstream.

`bddec43` also corroborates this from the other direction. Its own comment:
*"Linking the passages beats narrowing to Extractor-selected sentences first
(0.619): the bridge entity is often in a sentence the Extractor did not
choose."* That is the same defect as `evidence_f1 = 0.2160` below — and the fix
routed around the Extractor rather than repairing it.

One correction to the record: `bddec43` removed a duplicate `"solo":
_validate_qa` key in `parsing.py`. Python keeps the last duplicate, so the
correct solo validator was active during the run below and the single-hop
number is valid. Had the dead key won, every solo call would have failed
validation and handed MA an unearned win.

## Result

Gate C **FAILS**, and not marginally.

| | MA pipeline | single-hop | Δ |
|---|---:|---:|---:|
| F1 | 0.2312 | 0.4213 | **−19.01 pts** |
| EM | 0.1650 | 0.3100 | −14.50 pts |

95% bootstrap CI `[−26.58, −11.56]`, McNemar **p = 0.00026**. MA wins 29,
loses 70, ties 101, n=200 paired.

Both strata lose: hidden_bridge **−20.32**, fully_named **−13.78**.

## The retrieval repair is NOT the problem — it works

Everything the previous fix targeted is now healthy:

| Signal | Value | Reading |
|---|---|---|
| Gate A recall@10 | 0.5077 → **0.9271** | mechanism repaired |
| Follow-up firing | **226/226 = 1.000** | BUG-2 gone; second hop always fires |
| `stop_reason` | 191 `plan_complete`, 9 `semantic_inability` | semantic stop is no longer killing runs (was 32% at 0.6B) |
| `executed_steps` | 2.13 mean | genuine multi-hop execution |
| Planner depth | {2: 160, 3: 26, 4: 7, 1: 3, 0: 3, 6: 1} | plans are sane |
| Gold title recall | 0.8175 | retrieval delivers |
| Both-gold retrieved | 0.6400 | 128/200 questions have full evidence available |

## The decisive measurement

Restricting to the **128 questions where BOTH gold passages were retrieved**:

```
MA      F1 0.3141
single  F1 0.5133
delta      -19.92 pts
```

The evidence was in hand. The multi-agent pipeline read it and produced a worse
answer than a single call reading the same passages. **The bottleneck is
downstream of retrieval.** No further retrieval work will move Gate C.

## Prime suspect: the Extractor destroys the evidence

```
evidence_f1 = 0.2160
```

That is Extractor-selected spans scored against gold supporting sentences. The
Extractor is discarding ~78% of the relevant evidence before QA sees anything.
QA and the plan summary only ever see extracted spans — never the passages — so
whatever the Extractor drops is unrecoverable.

Single-hop has no Extractor. It reads the top-10 passages directly. That is the
entire difference, and it is worth ~20 F1 points.

This is the same information-destruction mechanism measured on the pre-repair
architecture (Extractor saw 1,588 prompt tokens, passed 376 to QA; gold answer
absent in 69% of MA losses). The per-document refactor changed its shape, not
its effect.

## Scale does not fix it

| Model | MA F1 | single F1 | Δ |
|---|---:|---:|---:|
| Qwen3-0.6B 4-bit (local 3050) | 0.0592 | 0.2658 | −20.66 |
| Qwen3-1.7B 4-bit (Kaggle T4) | 0.2312 | 0.4213 | **−19.01** |

Both arms improved ~2–4× with model size. **The gap moved 1.65 points.** That is
the signature of an architectural loss, not a capacity limit — consistent with
the previously reported flat gap across 0.5B → 7B.

## What to investigate

Diagnose independently; this is a lead, not a verdict.

1. **Quantify the Extractor bottleneck directly.** For each of the 128
   both-gold-retrieved questions, check whether the gold answer string survives
   into the spans QA receives. If it is absent in most MA losses, the Extractor
   is confirmed as the defect.
2. **`evidence_f1 = 0.216` — is it precision or recall?** The record carries
   `evidence_precision`, `evidence_recall`, `evidence_status`,
   `extractor_normalization_rejected_span_count`, and
   `extractor_normalization_rejection_reasons`. If rejection counts are high,
   the normalizer (exact-source-sentence matching) may be discarding valid spans
   rather than the model failing to find them.
3. **Test the contract, not just the model.** SPEC §4.3 requires only normalized
   exact source sentences reach QA. Consider whether passing the retrieved
   passage alongside the extracted spans closes the gap. If it does, the
   extract-then-answer contract is the defect and it is a SPEC change, not a
   bug fix.
4. **Check the plan summary path.** `final_answer_source` = {summary_parsed 157,
   qa_fallback 28, none 10, summary_salvaged 5}. The scored answer comes from a
   Step Definer summary over accumulated state. If that state is poor, the
   summary cannot recover. Compare summary-derived answers against the best
   intermediate QA answer.

## Constraints

- Do not touch retrieval. Gates A/B pass; the +41.5 pt headroom is being
  reached.
- Batch stays at 4; enforced in `run_retrieval_smoke.py` and
  `_validate_local_smoke_contract` alongside model identity, precision, and the
  10/7/3 split.
- `enable_thinking=False` must hold — one renderer, `models.render_chat`, three
  call sites. Any `<think>` in output voids the run.
- Renormalize line endings after checkout (`git rm --cached -r . && git reset
  --hard`) or manifest hashes fail (BUG-6).
- `model_revisions` are all `TBD` (BUG-5). `check_pilot.py` fails closed on this
  and has no `--allow-unpinned-tbd` escape hatch, unlike the runner — so Gate C
  must be computed directly from artifacts.

## Acceptance

Unchanged from SPEC §15.5: overall ΔF1 ≥ +5.0, CI lower bound > +2.0,
McNemar p < 0.01, hidden_bridge ΔF1 ≥ +8.0, fully_named within ±2.0.

Current distance to threshold: **−19.01 vs +5.0 required, a 24-point swing.**

If that is unreachable, the honest outcome is a negative result: multi-agent
extract-then-answer underperforms single-call RAG at SLM scale because the
extraction bottleneck costs more than decomposition gains. That is publishable
and well-evidenced — but it should be a conclusion reached after testing the
Extractor contract, not before.
