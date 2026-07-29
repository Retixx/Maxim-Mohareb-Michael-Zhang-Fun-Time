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


def result_slug(run_id: str, n: int, seed: int) -> str:
    """Filename stem for a run's outputs. Identifies the dataset sample too."""
    return f"{run_id}_n{n}_seed{seed}"


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
               batch_size, min_batch_size, store, idx, log_confidence=False) -> int:
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
                log_confidence=log_confidence,
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
    if run_id not in cfg["runs"]:
        raise SystemExit(f"unknown run {run_id!r}; config defines {list(cfg['runs'])}")

    stage_precision = cfg["runs"][run_id]
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
    print(f"    stage precision: {stage_precision}")

    questions = load_questions(n, seed=seed)

    # n and seed are part of the filename, not just the metadata. Resume keys on
    # (question_id, stage, call_index), which says nothing about which sample the
    # question came from — so a dev run at n=30/seed=1234 sharing a file with the
    # real n=300/seed=7 run would silently interleave two different datasets into
    # one results file and no key collision would ever flag it.
    slug = result_slug(run_id, n, seed)
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
    try:
        for stage in prompts.ROLES:
            precision = stage_precision[stage]
            calls = build_stage_calls(stage, questions, idx)
            pending = [
                c for c in calls
                if (c["question_id"], stage, c["call_index"]) not in idx
            ]
            print(f"\n--- stage {stage} @ {precision}: "
                  f"{len(calls)} calls, {len(pending)} pending ---")

            if not pending:
                print("    (already complete, skipping model load)")
                continue

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_load = time.perf_counter()
            model, tok = models.load_model(model_id, precision)
            print(f"    loaded in {time.perf_counter() - t_load:.1f}s")

            footprint = models.weight_footprint_mb(model, precision)
            census = models.param_census(model)
            t_stage = time.perf_counter()
            bs = _run_stage(
                model, tok, stage, pending, precision, run_id,
                batch_size, min_batch_size, store, idx,
                log_confidence=log_confidence,
            )
            store.flush()
            elapsed = time.perf_counter() - t_stage

            peak = (torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available() else None)
            stage_meta[stage] = {
                **models.quant_config_metadata(precision),
                "peak_vram_mb": round(peak, 1) if peak else None,
                "weight_footprint_mb": round(footprint, 1),
                "calls": len(calls),
                "stage_wall_s": round(elapsed, 1),
                "final_batch_size": bs,
                **census,
            }
            vram = f"  peak_vram_mb={peak:.0f}" if peak is not None else ""
            print(f"\n    done in {elapsed:.1f}s  final_batch_size={bs}{vram}")

            models.unload(model)

        # Answer records (SPEC §7: one per question, with EM and F1).
        answers = build_answer_records(questions, idx, run_id)
        have = {r["question_id"] for r in existing if r.get("record_type") == "answer"}
        new_answers = [a for a in answers if a["question_id"] not in have]
        store.write(new_answers)
        store.flush()
    finally:
        store.close()

    wall = time.perf_counter() - t_run

    # SPEC §7: memory is a controlled constant across runs 1-4, logged to prove it.
    coresident = sum(
        s["weight_footprint_mb"] for s in stage_meta.values()
        if s.get("weight_footprint_mb")
    )
    meta.update({
        "run_id": run_id,
        "model_id": model_id,
        "n": n,
        "seed": seed,
        "batch_size": batch_size,
        "log_confidence": log_confidence,
        "prompt_version": prompts.PROMPT_VERSION,
        "stage_precision": stage_precision,
        "stages": stage_meta,
        "coresident_footprint_mb": round(coresident, 1),
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_wall_s": round(meta.get("total_wall_s", 0.0) + wall, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    em = sum(a["em"] for a in answers) / len(answers)
    f1 = sum(a["f1"] for a in answers) / len(answers)
    print(f"\n\n=== {run_id} complete: EM {100*em:.1f}%  F1 {100*f1:.1f}%  "
          f"n={len(answers)}  wall {wall:.0f}s ===")
    print(f"    coresident_footprint_mb={coresident:.0f}")
    print(f"    -> {store.path}")
    print(f"    -> {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=None, help="override dataset.n")
    ap.add_argument("--seed", type=int, default=None, help="override dataset.eval_seed")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, args.run, args.n, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
