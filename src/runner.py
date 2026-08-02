"""Sweep driver: stage-major execution with checkpoint/resume.

    python -m src.runner --config config/experiment.yaml --run stepdef_4bit

SPEC §6. Walks the four stages in order. For each stage it loads the base model
at the precision this run assigns to that stage, processes every question, writes
records to disk, and unloads before moving on — so exactly one model is resident
at any moment.

Resuming is the default, not a flag. On startup the existing JSONL is read and
any (question_id, stage, call_index) already present is skipped. Killing the
process and rerunning the same command picks up where it stopped.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from . import agents, models, prompts
from .pipeline import (
    build_answer_records,
    build_stage_calls,
    index_records,
    load_questions,
)

FLUSH_EVERY = 50  # SPEC §6


def resolve_treatments(cfg: dict, run_id: str) -> dict:
    """Turn a run's stage table into {stage: {"model_id", "precision"}}.

    SPEC §8. Two capacity axes, one config schema, backward compatible:

        planner: fp16                            -> base model at fp16
        step_definer: {model: small, precision: fp16}   -> the `small` alias
        extractor: {model: Qwen/Qwen2.5-3B-Instruct, precision: 4bit}

    A bare string is a precision on the run's base model, so every Phase Q run
    definition written against the v1 schema still resolves unchanged. A `model`
    is looked up in `cfg["models"]` and falls through to a literal HF repo id if
    it contains a "/", so a one-off model needs no config entry.
    """
    if run_id not in cfg["runs"]:
        raise SystemExit(f"unknown run {run_id!r}; config defines {list(cfg['runs'])}")

    aliases = cfg.get("models") or {}
    base = cfg["model_id"]

    out = {}
    for stage, spec in cfg["runs"][run_id].items():
        if stage not in prompts.ROLES:
            raise SystemExit(
                f"run {run_id!r}: unknown stage {stage!r}; "
                f"expected one of {prompts.ROLES}"
            )
        if isinstance(spec, str):
            model_id, precision = base, spec
        elif isinstance(spec, dict):
            precision = spec.get("precision", "fp16")
            name = spec.get("model", "base")
            if name in aliases:
                model_id = aliases[name]
            elif name == "base":
                model_id = base
            elif "/" in name:
                model_id = name  # literal repo id, no alias needed
            else:
                raise SystemExit(
                    f"run {run_id!r} stage {stage!r}: unknown model alias {name!r}; "
                    f"config defines {sorted(aliases)} (or use a literal 'org/repo')"
                )
        else:
            raise SystemExit(
                f"run {run_id!r} stage {stage!r}: expected a precision string or a "
                f"{{model, precision}} mapping, got {type(spec).__name__}"
            )

        if precision not in models.PRECISIONS:
            raise SystemExit(
                f"run {run_id!r} stage {stage!r}: unknown precision {precision!r}; "
                f"expected one of {models.PRECISIONS}"
            )
        out[stage] = {"model_id": model_id, "precision": precision}
    return out


def deduped_footprint_mb(stage_meta: dict) -> float:
    """Weight bytes for a deployment that loads each distinct config once.

    SPEC §5d. `coresident_footprint_mb` sums over stages, which is correct only
    if every agent runs as its own model server. All four roles share one set of
    weights in a uniform run, so summing over stages counts four copies of one
    model and reports an 1874 MB "saving" that is an artifact of the arithmetic.

    This sums over distinct (model_id, precision) pairs instead — the number the
    Pareto frontier in SPEC §10 Figure 4 is plotted against. Both are logged;
    neither is an estimate; they answer different deployment questions.
    """
    seen = {}
    for s in stage_meta.values():
        fp = s.get("weight_footprint_mb")
        if fp:
            seen[(s.get("model_id"), s.get("precision"))] = fp
    return sum(seen.values())


def model_slug(model_id: str) -> str:
    """Short filename-safe tag for a base model.

    Qwen/Qwen2.5-1.5B-Instruct     -> qwen2.5-1.5b
    meta-llama/Llama-3.2-3B-Instruct -> llama-3.2-3b
    """
    name = model_id.split("/")[-1].lower()
    for suffix in ("-instruct", "-chat", "-it"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def result_slug(run_id: str, n: int, seed: int, model_id: str,
                treatments: dict | None = None) -> str:
    """Filename stem for a run's outputs.

    Carries model, n and seed — not just run_id. Resume keys on
    (question_id, stage, call_index), which is blind to all three, so any of them
    sharing a file would silently interleave incompatible data with no key
    collision to flag it. Model matters as much as n: `baseline` on Qwen and
    `baseline` on Llama are different experiments answering to the same run_id.

    Since Phase S a run can use more than one model, and the base model alone no
    longer identifies it. Every additional model is appended:

        stepdef_4bit   -> stepdef_4bit_qwen2.5-1.5b_n750_seed7
        stepdef_small  -> stepdef_small_qwen2.5-1.5b+qwen2.5-0.5b_n750_seed7

    Single-model runs are unaffected, so every Phase Q file keeps its name. This
    is what stops `--model-id meta-llama/Llama-3.2-3B-Instruct --run stepdef_small`
    from writing Qwen-generated small-stage records into a file named `llama-3.2-3b`
    — the exact silent interleave this function exists to prevent.
    """
    slug = model_slug(model_id)
    if treatments:
        extra = sorted({model_slug(t["model_id"]) for t in treatments.values()} - {slug})
        if extra:
            slug = "+".join([slug, *extra])
    return f"{run_id}_{slug}_n{n}_seed{seed}"


class JsonlStore:
    """Append-only record store with resume (SPEC §6)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._since_flush = 0

    def read_existing(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A kill mid-write can leave one torn final line. Drop it and
                    # carry on; that call is simply redone.
                    print(f"  [resume] dropping torn line {line_no} of {self.path.name}")
        return out

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # If a previous process was killed mid-write the file can end without a
        # newline. Appending straight onto that torn line would glue the first
        # resumed record onto it and lose BOTH. Terminate it first so the torn
        # line stays isolated (read_existing drops it) and the new record lands
        # on a line of its own.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as fh:
                fh.seek(-1, 2)
                needs_newline = fh.read(1) != b"\n"
            if needs_newline:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write("\n")
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def write(self, records: list[dict]) -> None:
        for r in records:
            self._fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._since_flush += 1
            if self._since_flush >= FLUSH_EVERY:
                self.flush()

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()
            self._since_flush = 0

    def close(self) -> None:
        if self._fh:
            self.flush()
            self._fh.close()
            self._fh = None


def _run_stage(model, tok, stage, pending, precision, run_id,
               batch_size, min_batch_size, store, idx, log_confidence=False,
               model_id=None) -> int:
    """Process `pending` in batches, halving the batch size on CUDA OOM.

    SPEC §6: start at 16, auto-reduce on OOM. The reduction sticks for the rest
    of the stage rather than being retried upward — a stage that OOMed once at 16
    will OOM again, and the retry costs more than the throughput it buys.

    Returns the batch size the stage finished on (recorded in run metadata).

    This retries a *batch*, never a parse. Recovering from an allocator failure
    and recovering from a malformed generation are different things; see SPEC §5.
    """
    bs = batch_size
    done = 0
    start = 0
    while start < len(pending):
        chunk = pending[start : start + bs]
        try:
            recs = agents.run_calls(
                model, tok, stage, chunk, precision, run_id, batch_size=bs,
                log_confidence=log_confidence, model_id=model_id,
            )
        except torch.cuda.OutOfMemoryError:
            if bs <= min_batch_size:
                raise
            torch.cuda.empty_cache()
            bs = max(min_batch_size, bs // 2)
            print(f"\n    [oom] reducing batch size to {bs} and retrying")
            continue
        store.write(recs)
        for r in recs:
            idx[(r["question_id"], r["stage"], r["call_index"])] = r
        start += len(chunk)
        done += len(chunk)
        print(f"    {done}/{len(pending)} (bs={bs})", end="\r", flush=True)
    return bs


def _library_versions() -> dict:
    """Versions that affect numerics, for reproducibility (SPEC §7).

    Runs happen on two machines (local dev, Kaggle T4); if results ever disagree
    this is the first thing to check.
    """
    import bitsandbytes
    import datasets
    import transformers
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "datasets": datasets.__version__,
        "cuda": torch.version.cuda,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - metadata only
        return "unknown"


def run(cfg: dict, run_id: str, n: int | None, seed: int | None, batch_size: int | None):
    treatments = resolve_treatments(cfg, run_id)
    stage_precision = {s: t["precision"] for s, t in treatments.items()}
    model_id = cfg["model_id"]
    n = n if n is not None else cfg["dataset"]["n"]
    seed = seed if seed is not None else cfg["dataset"]["eval_seed"]
    batch_size = batch_size if batch_size is not None else cfg["generation"]["batch_size"]
    min_batch_size = cfg["generation"].get("min_batch_size", 1)
    # SPEC §5b prediction 4. Off by default: costs an extra prefill per batch, and
    # enabling it mid-project would make runs non-comparable in cost accounting.
    log_confidence = bool(cfg["generation"].get("log_confidence", False))
    results_dir = Path(cfg.get("results_dir", "results"))

    print(f"=== run {run_id} | {model_id} | n={n} seed={seed} "
          f"batch={batch_size} prompts={prompts.PROMPT_VERSION} ===")
    for s, t in treatments.items():
        print(f"    {s:<14} {t['model_id']} @ {t['precision']}")

    # SPEC §8: everything is driven from the config. These three keys used to be
    # present but dead — load_questions hardcoded them and never read the file.
    ds_cfg = cfg.get("dataset", {})
    questions = load_questions(
        n,
        seed=seed,
        split=ds_cfg.get("split", "validation"),
        config=ds_cfg.get("config", "distractor"),
        name=ds_cfg.get("name"),
    )

    # n and seed are part of the filename, not just the metadata. Resume keys on
    # (question_id, stage, call_index), which says nothing about which sample the
    # question came from — so a dev run at n=30/seed=1234 sharing a file with the
    # real n=300/seed=7 run would silently interleave two different datasets into
    # one results file and no key collision would ever flag it.
    slug = result_slug(run_id, n, seed, model_id, treatments)
    store = JsonlStore(results_dir / f"{slug}.jsonl")

    existing = store.read_existing()
    idx = index_records(existing)
    if existing:
        print(f"    [resume] found {len(idx)} completed agent calls on disk")

    meta_path = results_dir / f"{slug}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    stage_meta = meta.get("stages", {})

    store.open()
    t_run = time.perf_counter()
    # SPEC §6: consecutive stages sharing a (model, precision) reuse the loaded
    # model instead of reloading it. Pure wall-time — it changes no output. The
    # peak counter is still reset per stage, and since max_memory_allocated is a
    # high-water mark over *currently allocated* bytes, a resident model still
    # counts toward the stage that reuses it. Peaks stay comparable to runs made
    # before this optimization existed.
    model = tok = None
    loaded: tuple | None = None
    # Walk only the stages this run defines, in canonical pipeline order. A
    # single-call run (SPEC §4a) defines one stage, not four.
    stages = [r for r in prompts.ROLES if r in treatments]
    did_work = False
    try:
        for stage in stages:
            treatment = treatments[stage]
            precision, stage_model_id = treatment["precision"], treatment["model_id"]
            calls = build_stage_calls(stage, questions, idx)
            pending = [
                c for c in calls
                if (c["question_id"], stage, c["call_index"]) not in idx
            ]
            print(f"\n--- stage {stage} @ {stage_model_id} {precision}: "
                  f"{len(calls)} calls, {len(pending)} pending ---")

            if not pending:
                print("    (already complete, skipping model load)")
                continue
            did_work = True

            # Reset the peak counter only once nothing stale is resident, and
            # BEFORE the incoming model is allocated, so a stage's peak means the
            # same thing whether the model was loaded fresh or carried over:
            # its own weights plus its own activations.
            #
            # Resetting earlier is wrong. reset_peak_memory_stats() sets the peak
            # to *currently allocated*, so with the outgoing model still resident
            # its bytes become the incoming stage's peak floor — e.g. qa_small's
            # QA stage (0.5B) would report the 1.5B footprint it replaced. SPEC §6
            # requires the reuse optimization to leave this accounting untouched.
            want = (stage_model_id, precision)
            if loaded != want:
                if model is not None:
                    # Drop our references FIRST, then reclaim — otherwise the old
                    # model is still reachable and survives until the new one has
                    # already been allocated. See models.unload's docstring.
                    model = tok = None
                    loaded = None
                    models.unload()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t_load = time.perf_counter()
                model, tok = models.load_model(stage_model_id, precision)
                loaded = want
                print(f"    loaded in {time.perf_counter() - t_load:.1f}s")
            else:
                # Reused: the weights are already allocated, so the reset floor is
                # exactly those weights — the same floor a fresh load produces.
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                print(f"    reusing loaded {stage_model_id} @ {precision}")

            footprint = models.weight_footprint_mb(model, precision)
            census = models.param_census(model)
            t_stage = time.perf_counter()
            bs = _run_stage(
                model, tok, stage, pending, precision, run_id,
                batch_size, min_batch_size, store, idx,
                log_confidence=log_confidence, model_id=stage_model_id,
            )
            store.flush()
            elapsed = time.perf_counter() - t_stage

            peak = (torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available() else None)
            stage_meta[stage] = {
                **models.quant_config_metadata(precision),
                "model_id": stage_model_id,
                "peak_vram_mb": round(peak, 1) if peak else None,
                "weight_footprint_mb": round(footprint, 1),
                "calls": len(calls),
                "stage_wall_s": round(elapsed, 1),
                "final_batch_size": bs,
                "oom_autotuned": bs != batch_size,
                **census,
            }
            vram = f"  peak_vram_mb={peak:.0f}" if peak is not None else ""
            print(f"\n    done in {elapsed:.1f}s  final_batch_size={bs}{vram}")

        if model is not None:
            model = tok = None
            models.unload()

        # Answer records (SPEC §7: one per question, with EM and F1).
        answers = build_answer_records(questions, idx, run_id)
        have = {r["question_id"] for r in existing if r.get("record_type") == "answer"}
        new_answers = [a for a in answers if a["question_id"] not in have]
        store.write(new_answers)
        store.flush()
    finally:
        store.close()
        if model is not None:
            model = tok = None
            models.unload()

    wall = time.perf_counter() - t_run

    # SPEC §5d: two footprints, two deployment topologies. `coresident` sums over
    # stages (one model server per agent); `deduped` sums over distinct
    # (model_id, precision) pairs (one process that loads each config once).
    # They differ for every mixed run, and the paper leads with `deduped`.
    coresident = sum(
        s["weight_footprint_mb"] for s in stage_meta.values()
        if s.get("weight_footprint_mb")
    )
    # A stage that resumed as already-complete contributes no stage_meta, so a
    # run finished across two invocations can end up with footprints summed over
    # a SUBSET of its stages — silently, and smaller than the truth. That is the
    # Kaggle idle-timeout path PROGRESS.md records as having already happened
    # once, and SPEC §7 calls the footprint "the number the paper reports".
    # Refuse to report a number computed from partial metadata.
    missing = [s for s in stages if s not in stage_meta]
    if missing:
        coresident = deduped = None
        print(f"\n  !! metadata incomplete: no stage record for {missing}. Footprints "
              f"suppressed rather than under-reported — rerun with results/{slug}.jsonl "
              f"removed, or merge in the meta.json from the earlier invocation.")
    else:
        deduped = deduped_footprint_mb(stage_meta)
    meta.update({
        "run_id": run_id,
        "model_id": model_id,
        "n": n,
        "seed": seed,
        "batch_size": batch_size,
        "log_confidence": log_confidence,
        "prompt_version": prompts.PROMPT_VERSION,
        "stage_precision": stage_precision,
        "stage_models": {s: t["model_id"] for s, t in treatments.items()},
        "stages": stage_meta,
        "coresident_footprint_mb": round(coresident, 1) if coresident else None,
        "deduped_footprint_mb": round(deduped, 1) if deduped else None,
        "metadata_complete": not missing,
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # Only count invocations that actually generated something. This used to
        # accumulate unconditionally, so every no-op resume inflated it while
        # `stage_wall_s` stayed put — and smoke_test.py extrapolates GPU-hours
        # from it, which is what SPEC §13's budget criterion is judged on.
        "total_wall_s": round(meta.get("total_wall_s", 0.0) + (wall if did_work else 0.0), 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    em = sum(a["em"] for a in answers) / len(answers)
    f1 = sum(a["f1"] for a in answers) / len(answers)
    print(f"\n\n=== {run_id} complete: EM {100*em:.1f}%  F1 {100*f1:.1f}%  "
          f"n={len(answers)}  wall {wall:.0f}s ===")
    fmt = lambda v: f"{v:.0f}" if v else "unavailable (incomplete metadata)"  # noqa: E731
    print(f"    deduped_footprint_mb={fmt(deduped)}  (SPEC §5d — the paper's number)")
    print(f"    coresident_footprint_mb={fmt(coresident)}  (one server per agent)")
    print(f"    -> {store.path}")
    print(f"    -> {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None, help="override dataset.n")
    ap.add_argument("--seed", type=int, default=None, help="override dataset.eval_seed")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--model-id", default=None,
                    help="override the base model (SPEC §3: a model is a config value, "
                         "not a plugin — this is just so one config can drive both)")
    ap.add_argument("--small-model-id", default=None,
                    help="override the `small` alias. Pair this with --model-id when "
                         "running a Phase S definition on model 2, or the small stage "
                         "silently keeps model 1's sibling.")
    ap.add_argument("--log-confidence", dest="log_confidence", action="store_true",
                    default=None, help="force confidence logging on for this run")
    ap.add_argument("--no-log-confidence", dest="log_confidence", action="store_false",
                    help="force it off. The confidence pass costs ~4.7 GB at batch 64 "
                         "and is what pushes a 3B extractor stage toward the 40 GB "
                         "ceiling. It is teacher-forced and runs strictly AFTER "
                         "generation, so disabling it cannot change any accuracy "
                         "number — only the calibration fields are lost.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model_id:
        cfg["model_id"] = args.model_id
    if args.small_model_id:
        cfg.setdefault("models", {})["small"] = args.small_model_id
    if args.log_confidence is not None:
        cfg.setdefault("generation", {})["log_confidence"] = args.log_confidence
    run(cfg, args.run, args.n, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
