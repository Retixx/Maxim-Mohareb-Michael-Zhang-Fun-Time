# Pre-launch findings for `no-bs` @ 8cefd92

All verified locally against the pushed branch. Nothing here was fixed — these
are decisions and risks for you to action before the A100 campaign.

---

## 1. BLOCKING DECISION: the pilot gate can STOP on a stratum you already know is worse

`config/experiment.yaml` requires the pilot to pass on **both**:

```yaml
pass_if_multiagent_minus_single_overall_at_least: 0.0
pass_if_multiagent_minus_single_hidden_bridge_at_least: 0.0
```

The pilot cohort is 160 `hidden_bridge` / 40 `fully_named`.

Measured retrieval (n=1500, k=10, equal read budget, all-gold-retrieved):

| stratum | single | two-hop | delta |
|---|---|---|---|
| hidden_bridge | 0.520 | 0.678 | **+0.158** |
| fully_named | 0.892 | 0.797 | **−0.095** |

`fully_named` is worse *by construction* — when every gold page is named in the
question, a second retrieval hop has nothing to resolve and spends budget
re-finding pages already held. Those 40 questions drag the **overall** contrast
toward negative.

So the campaign can pass `hidden_bridge` cleanly and still trip the `overall`
gate at a threshold of exactly 0.0, with n=200.

**Decide before launch, not after seeing a STOP:**
- (a) keep `overall` as a co-primary gate and accept the risk, or
- (b) make `hidden_bridge` primary and report `overall` as a reported-but-not-gating
  secondary.

(b) is defensible and pre-registerable. Changing the rule *after* a STOP is not.

---

## 2. Extractor token cap: 320 → 128 against a frozen prompt

```
f8c6334 (pre-merge)   MAX_NEW_TOKENS["extractor"] = 320
8cefd92 (current)     MAX_NEW_TOKENS["extractor"] = 128
```

The cut is reasonable in principle — the Extractor now sees **one passage per
call** instead of ten. But `src/prompts.py` is frozen at extractor-v5, whose
95.6% parse rate was measured on **ten-paragraph inputs at 320 tokens**. That
pairing (v5 prompt + 1 passage + 128 cap) has never been measured.

Observed on Qwen2.5-1.5B-Instruct @ 4bit, local cohort:
`truncated` on roughly 10–15% of Extractor calls, across all plan steps.

The rate is not the concern. The concern is **selectivity**: a gold passage
contains material the Extractor is asked to copy, a distractor usually does not,
so truncation may land preferentially on gold — losing exactly the evidence the
pipeline exists to gather, while distractors return a tidy empty list and score
`ok`. Measurement of gold-vs-distractor truncation and of whether raising the
cap recovers spans is in progress; numbers to follow.

Note the baseline is 3B-fp16, which should ramble less than 1.5B-4bit, so expect
a lower rate — but the cap and input shape are identical.

---

## 3. Stratum split is the dataset's natural rate — good, say so in the paper

Using `retrieval.title_is_mentioned` (which correctly strips `(film)`-style
disambiguators and matches on token boundaries), over the 2,874 dev questions
not reserved by any frozen cohort:

```
hidden_bridge 2142   fully_named 732   ->  74.5% / 25.5%
```

The frozen final manifest is 1097/403 = **73.1% hidden**. That matches the
natural distribution, so the split is **not enrichment** — it is stratified
reporting of HotpotQA as it comes. This is worth stating explicitly; a reviewer
will otherwise assume the cohort was stacked.

---

## 4. Supporting citation from MA-RAG's own ablation (Table 1, Llama3-70B)

| | NQ | TriviaQA | HotpotQA | 2WikimQA |
|---|---|---|---|---|
| MA-RAG | 58.1 | 85.4 | 50.7 | 43.1 |
| − Planner | 57.9 | 80.3 | 36.2 | 26.4 |
| **Planner worth** | **+0.2** | +5.1 | **+14.5** | **+16.7** |

The decomposition agent is worth +0.2 on single-hop NQ and +14.5/+16.7 on
multi-hop. MA-RAG measured this effect but never isolated it — the
hidden_bridge/fully_named stratification is the isolation. Cite this.

Caveat to state honestly: HotpotQA is 100% multi-hop by construction, so the
paper should not claim a general RAG benefit. Real-world RAG workloads are
mostly single-hop, where the Planner is worth ~nothing by MA-RAG's own numbers.

---

## 5. Measured cost per question

Local run, 3 executed plan steps:

```
planner 1 + 3 x (step_definer 1 + extractor 10 + qa 1) + plan_summary 1 = 32 calls
```

vs 1 call for `single_fp16`. At n=1500 across 21 multi-agent arms this is the
dominant cost. There is no call budget or cap anywhere in the runner — worth
confirming the fleet wall-clock projection uses a measured s/call from the pilot
rather than an estimate.

---

## 6. Lower-severity items

- **`.gitattributes` is absent** and `core.autocrlf=true` on Windows. Your
  canonical LF manifest hashes are correct for Linux, but any Windows checkout
  injects CRLF and fails `test_contract` (2 failures). Adding
  `config/manifests/*.json -text` would stop that recurring. Blob content and
  hashes are unaffected.
- **`_write_json_atomic` leaks a `*.tmp`** when handed a non-serialisable
  payload. Target file stays intact and the leftover cannot be mistaken for
  state (`*.json` glob is clean). Cosmetic.
- **Timing phase vs fleet co-tenancy.** `timing.benchmark_device_policy` is
  `one_reserved_uncontended_a100`, but the accuracy phase runs 6 concurrent
  workers. Confirm the runbook pins timing to a dedicated idle GPU, or
  throughput ratios are contaminated.

---

## Verified as sound (no action needed)

- 22-run shard plan is an exact partition: no duplicates, none missing, 4/4/4/4/3/3
- Pilot gate cannot be forged — recomputing `artifact_sha256` over a tampered
  payload still refuses (cross-checked against config hash, manifest hashes,
  decision rule, n, strata counts, environment lock)
- Claim ledger: 8 concurrent same-worker launches → exactly 1 wins; hard-kill
  leaves a stale claim that fails closed and needs explicit `--recover-stale-claim`
- Torn JSONL tail is truncated, logged in-band as a `store_repair` record with
  exact `removed_bytes`, and appends cleanly afterwards
- 106 unit tests + 35 adversarial checks pass under an LF checkout
