"""Local pilot rehearsal on a 4 GiB laptop GPU. READ-ONLY against the repo.

WHY THIS EXISTS
---------------
The A100 pilot gates the whole campaign on multi-agent F1 >= single-call F1.
This rehearses that contrast on hardware we already have, so a wiring fault in
the variable-depth executor is found here rather than on the rented GPU.

CONTAMINATION FIREWALL -- the entire point of the design below
--------------------------------------------------------------
Nothing here may touch a question the real campaign will score. The 100 IDs are
drawn from the 2,874 that remain after removing the union of every frozen
cohort (final 1,500, its 3,031 embedded exclusions, pilot 200, timing 128,
preflight 32 = 4,531 reserved). The draw uses seed 20260807, which no frozen
cohort uses. Overlap with the reserved union is asserted to be zero before a
model loads, and again after loading questions.

Results are written to the scratchpad, never to results/. The repo working tree
must remain clean; `--assert-clean` enforces that.

WHAT IT IS NOT
--------------
Not the pilot, and not evidence for the paper. The models are smaller than the
campaign's, the corpus build is identical but n is 100, and a laptop GPU forces
small batches. It answers one question only: does the multi-agent pipeline
convert its retrieval advantage into answer quality, or does stage-wise
degradation eat it?
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

SAMPLES = defaultdict(list)
from pathlib import Path

REPO = Path(r"C:\Users\maxim\Projects\marag-precision")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src import agents, models, pipeline, prompts, retrieval  # noqa: E402
from src.metrics import exact_match, f1_score  # noqa: E402


def assert_repo_clean():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    if out:
        raise SystemExit(f"repo is dirty; refusing to run:\n{out}")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    print(f"repo clean at {head}")


def load_smoke_questions(reserved: set[str]) -> list[dict]:
    """Exactly the 100 smoke IDs, via the real loader, proven disjoint."""
    spec = json.loads((SCRATCH / "smoke100.json").read_text(encoding="utf-8"))
    wanted = set(spec["question_ids"])
    if wanted & reserved:
        raise SystemExit("smoke set intersects a frozen cohort — aborting")

    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation",
                      revision="1908d6afbbead072334abe2965f91bd2709910ab")
    # Exclude everything that is not wanted, so the loader's own sampling path
    # can only return the smoke set.
    exclude = {r["id"] for r in ds} - wanted
    qs = pipeline.load_questions(
        len(wanted), seed=20260807, exclude=exclude,
        name="hotpotqa/hotpot_qa",
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
    )
    got = {q["question_id"] for q in qs}
    if got != wanted:
        raise SystemExit("loader did not return the smoke set exactly")
    if got & reserved:
        raise SystemExit("CONTAMINATION after load — aborting")
    return qs


def run_stage(model, tok, stage, questions, idx, retriever, batch_size, precision,
              model_id, stats):
    calls = pipeline.build_stage_calls(stage, questions, idx, retriever=retriever)
    if not calls:
        return 0
    t0 = time.perf_counter()
    records = agents.run_calls(
        model, tok, stage, calls, precision, run_id="local_smoke",
        batch_size=batch_size, model_id=model_id,
    )
    for rec in records:
        idx[(rec["question_id"], rec["stage"], rec["call_index"])] = rec
        stats[(stage, rec["parse_status"])] += 1
        # Keep a few failing generations verbatim: the Extractor's frozen v5
        # prompt was tuned on ten-paragraph inputs and now sees one passage, so
        # its failure MODE matters more than its rate.
        if rec["parse_status"] != "ok" and len(SAMPLES[rec["parse_status"]]) < 3:
            SAMPLES[rec["parse_status"]].append(
                {"stage": stage, "output_tokens": rec.get("output_tokens"),
                 "raw": (rec.get("raw_output") or "")[:600]})
    dt = time.perf_counter() - t0
    print(f"    {stage:<22} {len(calls):5d} calls  {dt:7.1f}s  "
          f"{dt / max(len(calls), 1):5.2f}s/call", flush=True)
    return len(calls)


def score(questions, idx, retriever, arm):
    answers = pipeline.build_answer_records(
        questions, idx, run_id=f"local_smoke_{arm}", retriever=retriever
    )
    by = defaultdict(list)
    for a in answers:
        by[a["retrieval_stratum"]].append(a)
        by["ALL"].append(a)
    return answers, by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="4bit", choices=["fp16", "8bit", "4bit"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap questions for a dry run")
    ap.add_argument("--arms", default="multi,single")
    args = ap.parse_args()

    assert_repo_clean()
    reserved = set(json.loads((SCRATCH / "reserved_ids.json").read_text()))
    print(f"reserved (must never be touched): {len(reserved)}")

    questions = load_smoke_questions(reserved)
    if args.limit:
        questions = questions[: args.limit]
    strata = Counter(q["retrieval_stratum"] for q in questions)
    print(f"smoke questions: {len(questions)}  {dict(strata)}")

    print("building pooled corpus (distractor + fullwiki)...", flush=True)
    corpus = retrieval.build_corpus(
        name="hotpotqa/hotpot_qa", split="validation",
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
        configs=("distractor", "fullwiki"),
    )
    retriever = retrieval.RetrievalContext(corpus, k=10)
    print(f"  {len(corpus):,} passages  {retriever.fingerprint()}")
    missing = [t for q in questions
               for t in (q["supporting_facts"] or {}).get("title", ())
               if t not in retriever.index.title_to_index]
    print(f"  gold titles absent from corpus: {len(missing)}")

    results = {}
    for arm in args.arms.split(","):
        stages = ([s for s in prompts.PIPELINE_STAGES if s != prompts.SOLO_ROLE]
                  if arm == "multi" else [prompts.SOLO_ROLE])
        print(f"\n=== ARM {arm}  ({args.model} @ {args.precision}) ===", flush=True)
        t_load = time.perf_counter()
        model, tok = models.load_model(args.model, args.precision)
        mem = models.memory_footprint_mib(model)
        shown = {k: round(v, 1) for k, v in mem.items() if isinstance(v, (int, float))}
        print(f"  loaded in {time.perf_counter() - t_load:.1f}s  {shown}", flush=True)

        idx, stats, total_calls = {}, Counter(), 0
        t0 = time.perf_counter()
        for stage in stages:
            total_calls += run_stage(model, tok, stage, questions, idx, retriever,
                                     args.batch_size, args.precision, args.model, stats)
        wall = time.perf_counter() - t0
        models.unload(model)
        model = tok = None
        torch.cuda.empty_cache()

        answers, by = score(questions, idx, retriever, arm)
        results[arm] = {"answers": answers, "stats": stats,
                        "calls": total_calls, "wall_s": wall}
        print(f"  {total_calls} calls in {wall:.0f}s")

        # Persist in the same shape src/runner.py writes, so existing analysis
        # tooling can read these without a bespoke parser. Written to the
        # scratchpad only -- never into results/.
        slug = f"local_smoke_{arm}_{args.model.split('/')[-1]}_{args.precision}"
        jsonl = SCRATCH / f"{slug}.jsonl"
        with jsonl.open("w", encoding="utf-8") as fh:
            for rec in idx.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for ans in answers:
                fh.write(json.dumps(ans, ensure_ascii=False) + "\n")
        meta = {
            "run_id": slug, "arm": arm, "record_counts":
                {"agent_call": len(idx), "answer": len(answers)},
            "model_id": args.model, "precision": args.precision,
            "batch_size": args.batch_size, "stages": stages,
            "n": len(questions), "strata": dict(strata),
            "question_ids_sha256": pipeline.question_ids_sha256(
                [q["question_id"] for q in questions]),
            "sampling": {"seed": 20260807, "source": "smoke100.json",
                         "disjoint_from_frozen_cohorts": True,
                         "reserved_ids_count": len(reserved)},
            "retrieval": retriever.fingerprint(),
            "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO,
                capture_output=True, text=True).stdout.strip(),
            "calls": total_calls, "wall_s": round(wall, 1),
            "NOT_THE_PILOT": "local rehearsal on excluded IDs; not campaign evidence",
        }
        (SCRATCH / f"{slug}.meta.json").write_text(
            json.dumps(meta, indent=1), encoding="utf-8")
        print(f"  wrote {jsonl.name} ({len(idx)} agent_call + {len(answers)} answer)")

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"{'stratum':<16}{'n':>5}" + "".join(f"{a:>12}" for a in results)
          + f"{'delta':>10}")
    print("-" * 74)
    out = {}
    for metric in ("f1", "em"):
        print(f"\n-- {metric.upper()} --")
        for stratum in ("hidden_bridge", "fully_named", "ALL"):
            row, vals = [], {}
            for arm, res in results.items():
                sel = [a for a in res["answers"]
                       if stratum == "ALL" or a["retrieval_stratum"] == stratum]
                vals[arm] = sum(a[metric] for a in sel) / max(len(sel), 1)
                row.append(vals[arm])
            n = len([a for a in results[list(results)[0]]["answers"]
                     if stratum == "ALL" or a["retrieval_stratum"] == stratum])
            delta = (vals.get("multi", 0) - vals.get("single", 0)) * 100
            print(f"{stratum:<16}{n:>5}" + "".join(f"{v * 100:>11.1f}%" for v in row)
                  + f"{delta:>+9.1f}pp")
            out[f"{metric}_{stratum}"] = {**{k: v for k, v in vals.items()},
                                          "delta_pp": delta}

    print("\n-- parse status by stage (multi arm) --")
    if "multi" in results:
        per_stage = defaultdict(Counter)
        for (stage, status), c in results["multi"]["stats"].items():
            per_stage[stage][status] += c
        for stage, c in per_stage.items():
            tot = sum(c.values())
            ok = c.get("ok", 0)
            bad = {k: v for k, v in c.items() if k != "ok"}
            print(f"  {stage:<22} {ok:5d}/{tot:<5d} ok ({ok / tot:5.1%})"
                  + (f"   {dict(bad)}" if bad else ""))

    print("\n-- cost --")
    for arm, res in results.items():
        print(f"  {arm:<8} {res['calls']:6d} calls  {res['wall_s']:7.0f}s")

    payload = {
        "model": args.model, "precision": args.precision,
        "n": len(questions), "strata": dict(strata),
        "seed": 20260807, "metrics": out,
        "calls": {a: r["calls"] for a, r in results.items()},
        "wall_s": {a: r["wall_s"] for a, r in results.items()},
        "samples": {k: v for k, v in SAMPLES.items()},
        "parse": {f"{s}|{st}": c for (s, st), c in
                  (results.get("multi", {}).get("stats") or {}).items()},
    }
    dest = SCRATCH / f"smoke_{args.precision}_{args.model.split('/')[-1]}.json"
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    assert_repo_clean()
    return 0


if __name__ == "__main__":
    sys.exit(main())
