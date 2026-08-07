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
