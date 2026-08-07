"""How much does the 128-token Extractor cap actually cost? READ-ONLY.

THE CONCERN
-----------
`MAX_NEW_TOKENS["extractor"]` was cut 320 -> 128 when the Extractor moved to one
passage per call. The prompt itself is frozen at v5 and was tuned at 320 over
ten-paragraph inputs, so the pairing has never been measured.

Raw truncation rate is the wrong number to panic about. What matters is whether
truncation is RANDOM or SELECTIVE. A gold passage contains material the
Extractor is being asked to copy; a distractor usually does not. So the failure
mode that would actually damage the experiment is truncation landing
preferentially on gold passages -- losing precisely the evidence the pipeline
exists to collect, while distractors return a tidy empty list and score "ok".

WHAT THIS MEASURES
------------------
For the same Extractor calls, generated twice (cap 128 and cap 320):

  * parse status and output length at each cap;
  * whether the passage is gold for that question;
  * truncation rate on gold vs distractor passages -- the selectivity test;
  * whether salvage still recovers spans from a truncated call;
  * whether raising the cap recovers spans that 128 lost.

Runs on the local smoke cohort only (seed 20260807, disjoint from every frozen
campaign cohort). Nothing is written into the repo.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(r"C:\Users\maxim\Projects\marag-precision")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src import agents, models, parsing, pipeline, prompts, retrieval  # noqa: E402

N_QUESTIONS = 30          # 30 questions x 10 passages = 300 extractor calls per cap
CAPS = (128, 320)


def main() -> int:
    spec = json.loads((SCRATCH / "smoke100.json").read_text(encoding="utf-8"))
    reserved = set(json.loads((SCRATCH / "reserved_ids.json").read_text()))
    wanted = set(spec["question_ids"])
    assert not (wanted & reserved), "CONTAMINATION"

    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      revision="1908d6afbbead072334abe2965f91bd2709910ab")
    exclude = {r["id"] for r in ds} - wanted
    questions = pipeline.load_questions(
        len(wanted), seed=20260807, exclude=exclude, name="hotpotqa/hotpot_qa",
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
    )[:N_QUESTIONS]

    corpus = retrieval.build_corpus(
        name="hotpotqa/hotpot_qa", split="validation",
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
        configs=("distractor", "fullwiki"),
    )
    retriever = retrieval.RetrievalContext(corpus, k=10)
    gold_of = {q["question_id"]: set((q["supporting_facts"] or {}).get("title", ()))
               for q in questions}

    model, tok = models.load_model("Qwen/Qwen2.5-1.5B-Instruct", "4bit")

    # Reproduce the real hop-1 Extractor calls exactly as the pipeline builds them.
    idx = {}
    for stage in ("planner", "step_definer"):
        calls = pipeline.build_stage_calls(stage, questions, idx, retriever=retriever)
        for rec in agents.run_calls(model, tok, stage, calls, "4bit",
                                    run_id="cap_probe", batch_size=8):
            idx[(rec["question_id"], rec["stage"], rec["call_index"])] = rec

    calls = pipeline.build_stage_calls("extractor", questions, idx, retriever=retriever)
    print(f"extractor calls: {len(calls)}")
    messages = [prompts.build_messages("extractor", **c["fields"]) for c in calls]

    per_cap = {}
    for cap in CAPS:
        t0 = time.perf_counter()
        gens = models.generate_batch(model, tok, messages, cap, batch_size=8)
        per_cap[cap] = gens
        print(f"  cap {cap}: {time.perf_counter() - t0:.0f}s")

    rows = []
    for i, call in enumerate(calls):
        title = (call["consumer_input"] or {}).get("document_title")
        is_gold = title in gold_of.get(call["question_id"], set())
        row = {"gold": is_gold, "title": title}
        for cap in CAPS:
            g = per_cap[cap][i]
            status, parsed = parsing.parse_output("extractor", g["raw_output"],
                                                  g["hit_token_cap"])
            salvaged = None if status == "ok" else parsing.salvage("extractor",
                                                                   g["raw_output"])
            spans = (parsed or salvaged or {}).get("spans", [])
            row[cap] = {"status": status, "tokens": g["output_tokens"],
                        "n_spans": len(spans), "usable": bool(spans)}
        rows.append(row)

    models.unload(model)

    def rate(sel, cap, key="status", val="truncated"):
        s = [r for r in rows if sel(r)]
        if not s:
            return float("nan"), 0
        return sum(r[cap][key] == val for r in s) / len(s), len(s)

    print("\n" + "=" * 70)
    print("SELECTIVITY: does truncation land on gold passages?")
    print(f"{'cap':>5}{'gold trunc':>14}{'distractor trunc':>20}{'ratio':>10}")
    for cap in CAPS:
        g, ng = rate(lambda r: r["gold"], cap)
        d, nd = rate(lambda r: not r["gold"], cap)
        ratio = (g / d) if d else float("inf")
        print(f"{cap:>5}{g:>13.1%}{d:>19.1%}{ratio:>10.2f}x   (n_gold={ng}, n_dist={nd})")

    print("\nUSABLE SPANS (parsed or salvaged) — what actually reaches QA")
    print(f"{'cap':>5}{'gold':>12}{'distractor':>14}{'overall':>10}")
    for cap in CAPS:
        g, _ = rate(lambda r: r["gold"], cap, "usable", True)
        d, _ = rate(lambda r: not r["gold"], cap, "usable", True)
        o, _ = rate(lambda r: True, cap, "usable", True)
        print(f"{cap:>5}{g:>11.1%}{d:>13.1%}{o:>9.1%}")

    print("\nRAISING THE CAP 128 -> 320")
    fixed = [r for r in rows if not r[128]["usable"] and r[320]["usable"]]
    lost = [r for r in rows if r[128]["usable"] and not r[320]["usable"]]
    fixed_gold = [r for r in fixed if r["gold"]]
    print(f"  calls that gain usable spans at 320 : {len(fixed)}/{len(rows)}"
          f"  ({len(fixed_gold)} of them on GOLD passages)")
    print(f"  calls that lose usable spans at 320 : {len(lost)}")

    print("\nOUTPUT LENGTH (tokens)")
    for cap in CAPS:
        toks = sorted(r[cap]["tokens"] for r in rows)
        med = toks[len(toks) // 2]
        p90 = toks[int(len(toks) * 0.9)]
        at_cap = sum(t >= cap for t in toks) / len(toks)
        print(f"  cap {cap:>3}: median={med:>4}  p90={p90:>4}  at-cap={at_cap:.1%}")

    print("\nPARSE STATUS")
    for cap in CAPS:
        print(f"  cap {cap:>3}: {dict(Counter(r[cap]['status'] for r in rows))}")

    (SCRATCH / "cap_probe.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {SCRATCH / 'cap_probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
