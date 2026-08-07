# Local smoke rehearsal — artifacts for Codex

Run on an RTX 3050 laptop (4 GiB) against `no-bs` @ 8cefd92, read-only.
NOT the pilot. NOT campaign evidence. 100 questions drawn with seed 20260807
from the 2,874 dev IDs remaining after removing all 4,531 reserved IDs
(final 1500 + its 3031 embedded exclusions + pilot 200 + timing 128 +
preflight 32). Overlap with reserved = 0, asserted before and after load.

## Files
- CODEX_BRIEF.md ....... findings + decisions needed before launch
- smoke_local.py ....... the driver (now also persists JSONL + meta.json)
- smoke100.json ........ the 100-question sample manifest (70/30)
- reserved_ids.json .... the 4,531 IDs excluded from sampling
- smoke_4bit_*.json .... metrics, parse counts, sample failed generations
- run_1.5b_4bit.log .... full stage-by-stage log of the 1.5B-4bit run
- quant.py / quant.log . empty-answer decomposition (n=40)
- diag.py .............. prediction dump (multi vs single, per-step QA state)
- cap_probe.py ......... extractor 128-vs-320 token-cap probe (not yet run)

## MISSING and why
No `.jsonl` / `.meta.json` for the 1.5B run. The driver invoked
`src/pipeline` + `src/agents` directly rather than `src/runner.py`, because the
runner validates frozen manifests against config hashes and using it would have
required mutating campaign config. Per-call records therefore stayed in memory.
The driver has since been patched to persist both; any re-run produces them.

## Headline (Qwen2.5-1.5B-Instruct @ 4bit, n=100)
F1  hidden_bridge  multi 8.1%  single 35.5%   -27.4pp
F1  fully_named    multi 37.8% single 68.8%   -31.1pp
F1  ALL            multi 17.0% single 45.5%   -28.5pp
EM  ALL            multi 11.0% single 32.0%   -21.0pp

Parse rates 95-100% at every stage -> not a parsing failure.
Cost: multi 2300 calls/1658s vs single 100 calls/472s (23 calls/question,
mean plan depth 1.75: 100 questions -> 64 reached step2 -> 11 reached step3).

## Decomposition of the gap (n=40)
32% of multi answers are empty strings. Single scored mean F1 0.464 on those
same questions.
  multi all            0.168
  multi non-empty only 0.249   (n=27)
  single all           0.455
  single same subset   0.450
=> ~0.08 of the gap is the fail-closed empty-answer policy (fixable),
   ~0.20 is genuinely wrong answers (capability).

## Context
Qwen2.5-0.5B-Instruct scores 0.007 EM on HotpotQA as a search agent;
Search-R1-PPO-3B scores 0.340 (arXiv 2508.20324). This 1.5B-4bit result
(0.11 EM multi) sits between them, which is evidence the harness measures
reality rather than a bug.

---

# SECOND COMMIT — measured retrieval recall + repaired driver

## The number the anchored-fusion design was gated on

Top-10 gold recall of the SHIPPED pipeline, Qwen2.5-1.5B-Instruct @ fp16, n=100:

  stratum          n   SINGLE   MULTI    delta   SINGLE all-gold  MULTI all-gold  delta
  hidden_bridge   70    0.729   0.571   -0.157        0.514           0.257      -0.257
  fully_named     30    0.883   0.800   -0.083        0.800           0.667      -0.133
  ALL            100    0.775   0.640   -0.135        0.600           0.380      -0.220

  mean unique passages read per question: single 10.0, multi 15.2

MULTI READS 52% MORE TEXT AND RETRIEVES LESS GOLD.

## Why the earlier +0.158 number did not transfer

scripts/bridge_recall.py measured hidden-bridge all-gold 0.520 -> 0.678 using
the ORIGINAL QUESTION as the hop-1 query and a regex-derived entity as hop-2.
The shipped pipeline queries with generated Step Definer text:

    src/pipeline.py:526   search_titles(task["task"], k)      <- multi
    src/pipeline.py:680   search_titles(q["question"], k)      <- solo

Different retrievers. The earlier figure never described the running system.
It was produced and reported by Claude, and using it to argue the pilot would
likely pass was wrong.

## Mechanism: step-2 queries never resolve the bridge

Queries do vary across steps (only 7% of multi-step questions repeat a query),
but step 2 carries unresolved references instead of the bridge entity:

    step1: What is the name of the animated series based on the Teen Titans?
    step2: Who played the character in the series?          <- "the character"

    step1: Where did John MacGregor study before becoming Baron MacGregor...?
    step2: ...other university he attended after...          <- "he"

    step1: What American country music group does Candy Coburn play with?
    step2: In which band does Candy Coburn perform?          <- restated, not a hop

This is the naive per-sub-question failure measured before the redesign
(recall@10 0.743 vs 0.794 for question-as-query). It was never fixed; it was
inherited by the variable-depth executor.

## Answer quality across three configs (same 100 questions)

  config             hidden_bridge F1        ALL F1
  1.5B 4bit          8.1 vs 35.5  (-27.4)   17.0 vs 45.5  (-28.5)
  3B   4bit         15.2 vs 39.6  (-24.3)   16.9 vs 49.1  (-32.3)
  1.5B fp16         15.3 vs 35.2  (-20.0)   19.2 vs 46.5  (-27.3)

Doubling parameters did not close the gap; fp16 over 4bit buys ~7pp on
hidden_bridge. Consistent with a mechanical retrieval deficit rather than a
pure capability floor.

## New files

- retrieval_recall.py / .txt ... the recall computation and its output
- local_smoke_{multi,single}_Qwen2.5-1.5B-Instruct_fp16.jsonl / .meta.json
                                 per-call records (agent_call + answer)
- smoke_fp16_*.json, smoke_4bit_Qwen2.5-3B-Instruct.json, run logs
- smoke_local.py .............. REPAIRED (the version in a68bbd1 had a
                                SyntaxError at line 189 and could not run)
