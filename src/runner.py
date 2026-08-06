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
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from . import agents, models, prompts, retrieval
from .pipeline import (
    build_answer_records,
    build_stage_calls,
    index_records,
    load_id_manifest,
    load_questions,
    question_ids_sha256,
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
        if stage not in prompts.ALL_ROLES:
            raise SystemExit(
                f"run {run_id!r}: unknown stage {stage!r}; "
                f"expected one of {prompts.ALL_ROLES}"
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
        revision = (cfg.get("model_revisions") or {}).get(model_id)
        tokenizer_revision = (cfg.get("tokenizer_revisions") or {}).get(
            model_id, revision
        )
        if not revision:
            raise SystemExit(
                f"run {run_id!r} stage {stage!r}: no pinned revision for {model_id}; "
                "add it to model_revisions"
            )
        fingerprint, fingerprint_payload = models.config_fingerprint(
            model_id, precision, revision, tokenizer_revision
        )
        out[stage] = {
            "model_id": model_id,
            "precision": precision,
            "model_revision": revision,
            "tokenizer_revision": tokenizer_revision,
            "config_fingerprint": fingerprint,
            "config_fingerprint_payload": fingerprint_payload,
        }

    # The Extractor's second retrieval pass is the SAME agent at the SAME
    # precision, so its treatment is mirrored rather than configured. Letting a
    # run set it independently would imply the two passes could be quantized
    # apart, which is not what the ablation measures.
    for mirror, source in prompts.STAGE_MIRRORS.items():
        if mirror in cfg["runs"][run_id]:
            raise SystemExit(
                f"run {run_id!r}: stage {mirror!r} is not configurable; it mirrors "
                f"{source!r} because it is the same agent making a second pass"
            )
        if source in out:
            out[mirror] = dict(out[source])
    return out


def deduplicated_footprint_mib(stage_meta: dict) -> float:
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
        fp = s.get("model_footprint_mib", s.get("weight_footprint_mb"))
        if fp:
            key = s.get("config_fingerprint") or (
                s.get("model_id"), s.get("precision"), s.get("model_revision")
            )
            seen[key] = fp
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
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self._fh = None
        self._since_flush = 0
        self._owns_lock = False

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
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"JSONL corruption at non-torn line {line_no} of {self.path}: {exc}"
                    ) from exc
        return out

    def open(self, execution_session_id: str | None = None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_payload = json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "execution_session_id": execution_session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = self.lock_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"exclusive output lock already exists: {self.lock_path}\nowner: {owner}\n"
                "Do not run the same arm on two workers. Remove a stale lock only "
                "after verifying that its process is dead."
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as lock_fh:
            lock_fh.write(lock_payload + "\n")
        self._owns_lock = True
        repair = None
        # Only a non-newline-terminated tail can be a kill-torn write. Truncate
        # it to the last durable newline; malformed complete lines are corruption
        # and read_existing() fails rather than silently dropping them.
        if self.path.exists() and self.path.stat().st_size:
            raw = self.path.read_bytes()
            if not raw.endswith(b"\n"):
                cut = raw.rfind(b"\n") + 1
                removed = len(raw) - cut
                with self.path.open("r+b") as fh:
                    fh.truncate(cut)
                repair = {
                    "record_type": "store_repair",
                    "repair": "truncated_non_newline_tail",
                    "removed_bytes": removed,
                    "execution_session_id": execution_session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                print(f"  [resume] truncated {removed} torn tail bytes from {self.path.name}")
        self._fh = self.path.open("a", encoding="utf-8")
        if repair:
            self.write([repair])
            self.durable_flush()
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

    def durable_flush(self) -> None:
        """Flush Python and OS buffers before writing a completion checkpoint."""
        if self._fh:
            self.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh:
            self.flush()
            self._fh.close()
            self._fh = None
        if self._owns_lock:
            self.lock_path.unlink(missing_ok=True)
            self._owns_lock = False


def _member_key(call: dict) -> tuple[str, int]:
    return call["question_id"], call["call_index"]


def _batch_hash(calls: list[dict]) -> str:
    payload = "".join(
        f"{call['question_id']}\t{call['call_index']}\n" for call in calls
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _execution_batch_id(
    stage: str, batch_no: int, phase: str, timing_repeat: int | None
) -> str:
    if phase == "timing_benchmark":
        return f"{stage}:{batch_no:06d}:timing:r{timing_repeat}"
    return f"{stage}:{batch_no:06d}"


def _order_batches_largest_first(
    tok, stage: str, batches: list[tuple[int, list[dict], list[dict]]]
) -> list[tuple[int, list[dict], list[dict]]]:
    """Execute the highest estimated padded-token batch first, IDs unchanged."""
    scored = []
    try:
        for batch_no, chunk, missing in batches:
            texts = [
                tok.apply_chat_template(
                    prompts.build_messages(stage, **call["fields"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for call in chunk
            ]
            encoded = tok(
                texts, padding=False, truncation=False, add_special_tokens=False
            )["input_ids"]
            max_len = max(len(ids) for ids in encoded)
            padded_tokens = max_len * len(chunk)
            scored.append((padded_tokens, batch_no, chunk, missing))
    except Exception:
        # Unit-test fakes and unusual tokenizers fall back to canonical order;
        # actual A100 preflight uses the pinned HF tokenizer and records lengths.
        return batches
    return [
        (batch_no, chunk, missing)
        for _, batch_no, chunk, missing in sorted(
            scored, key=lambda item: (-item[0], item[1])
        )
    ]


def _certified_batch_ids(
    records: list[dict], stage: str, calls: list[dict], batch_size: int,
    *, config_fingerprint: str, question_manifest_sha256: str,
) -> set[str]:
    """Validate durable scored-batch certificates already on disk."""
    by_id: dict[str, list[dict]] = {}
    for record in records:
        if (
            record.get("record_type") == "batch"
            and record.get("stage") == stage
            and record.get("phase") == "scored"
            and not record.get("oom")
        ):
            by_id.setdefault(record.get("batch_id"), []).append(record)

    certified = set()
    for batch_no, start in enumerate(range(0, len(calls), batch_size)):
        chunk = calls[start : start + batch_size]
        batch_id = f"{stage}:{batch_no:06d}"
        found = by_id.get(batch_id, [])
        if len(found) > 1:
            raise RuntimeError(f"duplicate completion certificates for {batch_id}")
        if not found:
            continue
        record = found[0]
        expected_members = [
            {"question_id": c["question_id"], "call_index": c["call_index"]}
            for c in chunk
        ]
        expected = {
            "canonical_batch_sha256": _batch_hash(chunk),
            "members": expected_members,
            "config_fingerprint": config_fingerprint,
            "question_manifest_sha256": question_manifest_sha256,
            "batch_size_requested": batch_size,
            "batch_ordinal": batch_no,
        }
        bad = [key for key, value in expected.items() if record.get(key) != value]
        if bad:
            raise RuntimeError(
                f"invalid completion certificate for {batch_id}: {', '.join(bad)}"
            )
        certified.add(batch_id)
    return certified


def _validate_existing_stage_calls(
    records: list[dict], stage: str, calls: list[dict], batch_size: int,
    *, run_id: str, treatment: dict, question_manifest_sha256: str,
) -> None:
    """Validate every persisted call before it can influence resume/upstream state."""
    expected = {}
    for ordinal, call in enumerate(calls):
        key = (call["question_id"], call["call_index"])
        expected[key] = {
            "run_id": run_id,
            "stage": stage,
            "model_id": treatment["model_id"],
            "precision": treatment["precision"],
            "model_revision": treatment["model_revision"],
            "tokenizer_revision": treatment["tokenizer_revision"],
            "config_fingerprint": treatment["config_fingerprint"],
            "question_manifest_sha256": question_manifest_sha256,
            "batch_id": f"{stage}:{ordinal // batch_size:06d}",
            "batch_member_index": ordinal % batch_size,
            "prompt_version": prompts.ROLE_PROMPT_VERSIONS[stage],
            "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
            "prompt_template_sha256": prompts.prompt_template_sha256(stage),
            "rendered_prompt_sha256": agents.rendered_prompt_sha256(
                prompts.build_messages(stage, **call["fields"])
            ),
        }
    seen = set()
    for record in records:
        if record.get("record_type") != "agent_call" or record.get("stage") != stage:
            continue
        key = (record.get("question_id"), record.get("call_index"))
        if key in seen:
            raise RuntimeError(f"duplicate persisted agent call for {stage}/{key}")
        seen.add(key)
        if key not in expected:
            raise RuntimeError(f"unexpected persisted agent call for {stage}/{key}")
        bad = [
            field for field, value in expected[key].items()
            if record.get(field) != value
        ]
        required_payload = (
            "raw_output", "parse_status", "parsed", "salvaged",
            "prompt_tokens", "output_tokens",
        )
        bad.extend(field for field in required_payload if field not in record)
        if bad:
            raise RuntimeError(
                f"persisted call integrity failure for {stage}/{key}: "
                + ", ".join(sorted(set(bad)))
            )


def _assert_regenerated_matches(
    existing: dict, regenerated: dict, *, allow_repeat_batch_id: bool = False
) -> None:
    fields = (
        "raw_output", "parse_status", "parsed", "salvaged",
        "rendered_prompt_sha256", "batch_member_index",
        "config_fingerprint", "question_manifest_sha256", "model_revision",
        "tokenizer_revision", "prompt_template_sha256", "prompt_bundle_version",
    )
    if not allow_repeat_batch_id:
        fields += ("batch_id",)
    bad = [field for field in fields if existing.get(field) != regenerated.get(field)]
    if bad:
        key = (regenerated.get("question_id"), regenerated.get("call_index"))
        raise RuntimeError(
            f"regenerated canonical member differs from persisted {key}: "
            + ", ".join(bad)
        )


def _oom_batch_record(
    *, stage: str, run_id: str, model_id: str, precision: str,
    batch_id: str, calls: list[dict], phase: str,
    execution_session_id: str, gpu_metadata: dict, error: Exception,
    batch_size_requested: int, batch_ordinal: int,
    config_fingerprint: str | None, model_revision: str | None,
    tokenizer_revision: str | None, question_manifest_sha256: str | None,
    timing_repeat: int | None = None,
) -> dict:
    return {
        "record_type": "batch", "run_id": run_id, "stage": stage,
        "model_id": model_id, "precision": precision, "batch_id": batch_id,
        "phase": phase, "timing_eligible": False,
        "execution_session_id": execution_session_id,
        "members": [
            {"question_id": c["question_id"], "call_index": c["call_index"]}
            for c in calls
        ],
        "canonical_batch_sha256": _batch_hash(calls),
        "batch_size_actual": len(calls),
        "batch_size_requested": batch_size_requested,
        "batch_ordinal": batch_ordinal,
        "timing_repeat": timing_repeat,
        "oom": True, "retry_count": 0,
        "config_fingerprint": config_fingerprint,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "question_manifest_sha256": question_manifest_sha256,
        "error": f"{type(error).__name__}: {error}",
        "timestamp": datetime.now(timezone.utc).isoformat(), **gpu_metadata,
    }


def _run_stage(
    model, tok, stage, calls, precision, run_id,
    batch_size, min_batch_size, store, idx, log_confidence=False,
    model_id=None, execution_session_id=None, gpu_metadata=None,
    timing_eligible=False,
    config_fingerprint=None, model_revision=None, tokenizer_revision=None,
    question_manifest_sha256=None, certified_batch_ids=None,
    phase="scored", timing_repeat=None,
) -> int:
    """Run fixed canonical batches, regenerating a whole partial resume batch.

    The batch is part of the treatment. Scored execution never autotunes after
    OOM: a warm-up/preflight must pass, or the arm fails loudly. On resume,
    completed members are regenerated with their original neighbors and
    discarded; only missing call records are appended.
    """
    if batch_size != min_batch_size:
        raise ValueError(
            f"fail-closed batching requires batch_size={batch_size} to equal "
            f"min_batch_size={min_batch_size}"
        )
    gpu_metadata = gpu_metadata or {}
    canonical = [
        (batch_no, calls[start : start + batch_size])
        for batch_no, start in enumerate(range(0, len(calls), batch_size))
    ]
    certified_batch_ids = set(certified_batch_ids or ())
    work = []
    for batch_no, chunk in canonical:
        canonical_batch_id = _execution_batch_id(
            stage, batch_no, phase, timing_repeat
        )
        missing = [
            c for c in chunk
            if (c["question_id"], stage, c["call_index"]) not in idx
        ]
        if missing or canonical_batch_id not in certified_batch_ids:
            work.append((batch_no, chunk, missing))
    if not work:
        return batch_size
    work = _order_batches_largest_first(tok, stage, work)

    missing_total = sum(len(missing) for _, _, missing in work)
    completed = 0
    for execution_ordinal, (batch_no, chunk, missing) in enumerate(work):
        batch_id = _execution_batch_id(stage, batch_no, phase, timing_repeat)
        try:
            recs, batch_record = agents.run_calls(
                model, tok, stage, chunk, precision, run_id,
                batch_size=batch_size, log_confidence=log_confidence,
                model_id=model_id, batch_id=batch_id,
                execution_session_id=execution_session_id,
                gpu_metadata=gpu_metadata, timing_eligible=timing_eligible,
                phase=phase, return_batch_record=True,
                config_fingerprint=config_fingerprint,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                question_manifest_sha256=question_manifest_sha256,
                batch_ordinal=batch_no,
                timing_repeat=timing_repeat,
                execution_ordinal=execution_ordinal,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            store.write([_oom_batch_record(
                stage=stage, run_id=run_id, model_id=model_id,
                precision=precision, batch_id=batch_id, calls=chunk,
                phase=phase, execution_session_id=execution_session_id,
                gpu_metadata=gpu_metadata, error=exc,
                batch_size_requested=batch_size, batch_ordinal=batch_no,
                config_fingerprint=config_fingerprint,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                question_manifest_sha256=question_manifest_sha256,
                timing_repeat=timing_repeat,
            )])
            store.durable_flush()
            raise RuntimeError(
                f"{run_id}/{stage} OOM at fixed {batch_id}; refusing to autotune"
            ) from exc

        missing_keys = {_member_key(c) for c in missing}
        for record in recs:
            existing_record = idx.get(
                (record["question_id"], stage, record["call_index"])
            )
            if existing_record and existing_record.get("record_type") == "agent_call":
                _assert_regenerated_matches(existing_record, record)
        new_recs = [
            r for r in recs
            if (r["question_id"], r["call_index"]) in missing_keys
        ]
        batch_record["canonical_batch_sha256"] = _batch_hash(chunk)
        batch_record["written_members"] = [
            {"question_id": r["question_id"], "call_index": r["call_index"]}
            for r in new_recs
        ]
        batch_record["resume_regenerated_members"] = len(chunk) - len(new_recs)
        # Call records land first; the trailing batch record is the durable
        # completion certificate. A torn write can never falsely certify work.
        store.write([*new_recs, batch_record])
        store.durable_flush()
        for r in new_recs:
            idx[(r["question_id"], r["stage"], r["call_index"])] = r
        completed += len(new_recs)
        print(
            f"    {completed}/{missing_total} missing calls written "
            f"(fixed bs={batch_size})", end="\r", flush=True,
        )
    return batch_size


def _preflight_stage(
    model, tok, stage: str, calls: list[dict], precision: str, run_id: str,
    batch_size: int, store: JsonlStore, warmup_idx: dict, *, model_id: str,
    execution_session_id: str, gpu_metadata: dict, config_fingerprint: str,
    model_revision: str, tokenizer_revision: str,
    question_manifest_sha256: str, preflight_manifest_sha256: str,
) -> None:
    """Warm up and prove batch-32 viability only on frozen excluded questions."""
    chunk = calls[:batch_size]
    if len(chunk) != batch_size:
        raise RuntimeError(
            f"{stage} preflight produced {len(chunk)} calls; exactly {batch_size} "
            "excluded calls are required"
        )
    batch_id = f"{stage}:excluded-preflight:{execution_session_id}"
    try:
        recs, batch_record = agents.run_calls(
            model, tok, stage, chunk, precision, run_id,
            batch_size=batch_size, log_confidence=False, model_id=model_id,
            batch_id=batch_id, execution_session_id=execution_session_id,
            gpu_metadata=gpu_metadata, timing_eligible=False,
            phase="excluded_preflight_warmup", return_batch_record=True,
            config_fingerprint=config_fingerprint,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            question_manifest_sha256=question_manifest_sha256,
            batch_ordinal=0,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        store.write([_oom_batch_record(
            stage=stage, run_id=run_id, model_id=model_id, precision=precision,
            batch_id=batch_id, calls=chunk, phase="excluded_preflight_warmup",
            execution_session_id=execution_session_id,
            gpu_metadata=gpu_metadata, error=exc,
            batch_size_requested=batch_size, batch_ordinal=0,
            config_fingerprint=config_fingerprint,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            question_manifest_sha256=question_manifest_sha256,
        )])
        store.durable_flush()
        raise RuntimeError(
            f"{run_id}/{stage} failed excluded fixed batch-{batch_size} preflight"
        ) from exc

    batch_record["canonical_batch_sha256"] = _batch_hash(chunk)
    batch_record["preflight_manifest_sha256"] = preflight_manifest_sha256
    for record in recs:
        record["record_type"] = "warmup_call"
        record["preflight_manifest_sha256"] = preflight_manifest_sha256
        warmup_idx[(record["question_id"], stage, record["call_index"])] = record
    store.write([*recs, batch_record])
    store.durable_flush()


def _run_timing_repeat(
    model, tok, stage: str, calls: list[dict], precision: str, run_id: str,
    batch_size: int, store: JsonlStore, idx: dict, *, repeat: int,
    model_id: str, execution_session_id: str, gpu_metadata: dict,
    config_fingerprint: str, model_revision: str, tokenizer_revision: str,
    question_manifest_sha256: str,
) -> None:
    """Replay identical canonical timing calls without duplicating call records."""
    batches = [
        (batch_no, calls[start : start + batch_size], [])
        for batch_no, start in enumerate(range(0, len(calls), batch_size))
    ]
    batches = _order_batches_largest_first(tok, stage, batches)
    for execution_ordinal, (batch_no, chunk, _) in enumerate(batches):
        batch_id = _execution_batch_id(stage, batch_no, "timing_benchmark", repeat)
        try:
            recs, batch_record = agents.run_calls(
                model, tok, stage, chunk, precision, run_id,
                batch_size=batch_size, log_confidence=False, model_id=model_id,
                batch_id=batch_id, execution_session_id=execution_session_id,
                gpu_metadata=gpu_metadata, timing_eligible=True,
                phase="timing_benchmark", return_batch_record=True,
                config_fingerprint=config_fingerprint,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                question_manifest_sha256=question_manifest_sha256,
                batch_ordinal=batch_no, timing_repeat=repeat,
                execution_ordinal=execution_ordinal,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            store.write([_oom_batch_record(
                stage=stage, run_id=run_id, model_id=model_id,
                precision=precision, batch_id=batch_id, calls=chunk,
                phase="timing_benchmark",
                execution_session_id=execution_session_id,
                gpu_metadata=gpu_metadata, error=exc,
                batch_size_requested=batch_size, batch_ordinal=batch_no,
                config_fingerprint=config_fingerprint,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                question_manifest_sha256=question_manifest_sha256,
                timing_repeat=repeat,
            )])
            store.durable_flush()
            raise RuntimeError(
                f"{run_id}/{stage} timing repeat {repeat} OOM at {batch_id}"
            ) from exc
        for record in recs:
            first = idx[(record["question_id"], stage, record["call_index"])]
            _assert_regenerated_matches(
                first, record, allow_repeat_batch_id=True
            )
        batch_record["canonical_batch_sha256"] = _batch_hash(chunk)
        batch_record["repeat_of"] = 0
        batch_record["duplicate_call_records_omitted"] = True
        store.write([batch_record])
        store.durable_flush()


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


def _gpu_metadata() -> dict:
    """Stable execution-device fields copied into every timing batch."""
    base = {
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if not torch.cuda.is_available():
        return {**base, "gpu_available": False}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    uuid_value = getattr(props, "uuid", None)
    result = {
        **base,
        "gpu_available": True,
        "gpu_logical_index": device,
        "gpu_name": props.name,
        "gpu_uuid": str(uuid_value) if uuid_value is not None else None,
        "gpu_total_memory_mib": round(props.total_memory / 1024**2, 1),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
    }
    selector = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi", "-i", selector,
                "--query-gpu=uuid,driver_version,pci.bus_id,pstate,power.limit,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        values = [x.strip() for x in line.split(",")]
        if len(values) == 6:
            result.update({
                "gpu_uuid": values[0],
                "gpu_driver_version": values[1],
                "gpu_pci_bus_id": values[2],
                "gpu_pstate_at_start": values[3],
                "gpu_power_limit_w": values[4],
                "gpu_sm_clock_mhz_at_start": values[5],
            })
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return result


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


def _resolve_repo_path(cfg: dict, value: str | Path) -> Path:
    """Resolve data artifacts predictably from either repo root or config dir."""
    path = Path(value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / path]
    if cfg.get("_config_dir"):
        candidates.append(Path(cfg["_config_dir"]) / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _validate_resume_meta(
    meta: dict, expected: dict, path: Path, *, require_present: bool = False
) -> None:
    """Refuse to append if immutable treatment/sample identity changed."""
    for key, value in expected.items():
        if require_present and key not in meta:
            raise RuntimeError(f"resume metadata {path} is missing immutable key {key}")
        if key in meta and meta[key] != value:
            raise RuntimeError(
                f"resume metadata mismatch for {key} in {path}: "
                f"on disk {meta[key]!r}, requested {value!r}"
            )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bundle_sha256() -> str:
    """Hash all experiment-defining source/artifacts, excluding the lock itself."""
    root = Path(__file__).resolve().parent.parent
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        include = (
            rel.startswith("src/") and rel.endswith(".py")
            or rel.startswith("scripts/") and rel.endswith(".py")
            or rel.startswith("tests/") and rel.endswith(".py")
            or rel.startswith("config/manifests/") and rel.endswith(".json")
            or rel in {
                "analyze.py", "smoke_test.py", "SPEC.md", "README.md",
                "ENVIRONMENT.md", "config/experiment.yaml",
                "requirements-core.txt", "requirements-manifest.txt",
            }
        )
        if include and rel != "config/environment.lock.json":
            candidates.append((rel, path))
    digest = hashlib.sha256()
    for rel, path in sorted(candidates):
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _content_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _derived_config_matches_lock(cfg: dict, lock: dict, artifact: dict) -> bool:
    """Allow only the analyzer's cryptographically linked allocation delta."""
    current_cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    frozen = current_cfg.get("frozen_allocation") or {}
    base_cfg = lock.get("experiment_config_contract")
    if not frozen or not isinstance(base_cfg, dict):
        return False

    semantic = dict(artifact)
    executable_hash = semantic.pop("executable_config_sha256", None)
    artifact_hash = semantic.pop("artifact_sha256", None)
    execution_run_id = artifact.get("execution_run_id")
    actual_run = (current_cfg.get("runs") or {}).get(execution_run_id)
    if not all((
        artifact_hash == _content_hash(semantic),
        executable_hash == _content_hash(current_cfg),
        frozen.get("selection_artifact_sha256") == artifact_hash,
        artifact.get("source_config_sha256")
        == lock.get("experiment_config_content_sha256"),
        frozen.get("run_config_sha256") == artifact.get("run_config_sha256"),
        actual_run == artifact.get("run_definition"),
        actual_run is not None,
        _content_hash(actual_run) == artifact.get("run_config_sha256"),
        frozen.get("question_manifest_sha256")
        == lock.get("final_manifest_ids_sha256"),
    )):
        return False

    # Reduce the derived config back to the locked base. No model alias,
    # revision, prompt, dataset, static arm, or memory/timing field may change.
    reduced = copy.deepcopy(current_cfg)
    reduced.pop("frozen_allocation", None)
    reduced["allocation_selector"] = copy.deepcopy(base_cfg["allocation_selector"])
    if execution_run_id not in (base_cfg.get("runs") or {}):
        reduced.get("runs", {}).pop(execution_run_id, None)
    return reduced == base_cfg


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pip_freeze() -> tuple[list[str], str]:
    text = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    lines = sorted(line.strip() for line in text.splitlines() if line.strip())
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    return lines, digest


def _environment_snapshot(
    cfg: dict,
    config_path: Path,
    manifest_path: Path,
    *,
    container_ref: str | None = None,
    container_digest: str | None = None,
    include_freeze: bool = False,
) -> dict:
    freeze, freeze_hash = _pip_freeze()
    gpu = _gpu_metadata()
    snapshot = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "container_ref": container_ref or os.environ.get("EXPERIMENT_CONTAINER_REF"),
        "container_digest": (
            container_digest or os.environ.get("EXPERIMENT_CONTAINER_DIGEST")
        ),
        "pip_freeze_sha256": freeze_hash,
        "core_packages": {
            name: _package_version(name)
            for name in (
                "torch", "transformers", "bitsandbytes", "datasets",
                "accelerate", "PyYAML", "huggingface-hub",
            )
        },
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "experiment_config_sha256": _sha256_file(config_path),
        "experiment_config_content_sha256": _content_hash({
            key: value for key, value in cfg.items() if not key.startswith("_")
        }),
        "experiment_config_contract": {
            key: copy.deepcopy(value)
            for key, value in cfg.items() if not key.startswith("_")
        },
        "final_manifest_file_sha256": _sha256_file(manifest_path),
        "final_manifest_ids_sha256": cfg["dataset"]["manifest_sha256"],
        "final_exclusion_ids_sha256": cfg["dataset"]["exclusion_sha256"],
        "model_revisions": cfg.get("model_revisions") or {},
        "tokenizer_revisions": cfg.get("tokenizer_revisions") or cfg.get("model_revisions") or {},
        "dataset_revision": cfg["dataset"].get("revision"),
        "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
        "prompt_template_sha256": prompts.prompt_template_hashes(),
        "source_bundle_sha256": _source_bundle_sha256(),
        "preflight_manifest_file_sha256": _sha256_file(
            _resolve_repo_path(
                cfg,
                cfg.get("dataset", {}).get(
                    "preflight_manifest_path",
                    "config/manifests/preflight_excluded32.json",
                ),
            )
        ),
        "preflight_manifest_ids_sha256": cfg.get("dataset", {}).get(
            "preflight_manifest_sha256"
        ),
        "timing_manifest_file_sha256": _sha256_file(
            _resolve_repo_path(cfg, cfg.get("timing", {})["manifest_path"])
        ),
        "timing_manifest_ids_sha256": cfg.get("timing", {}).get("manifest_sha256"),
        "timing_repetitions": cfg.get("timing", {}).get("repetitions"),
    }
    if include_freeze:
        snapshot["pip_freeze"] = freeze
    return snapshot


def write_environment_lock(
    cfg: dict,
    config_path: Path,
    output_path: Path,
    *,
    container_ref: str,
    container_digest: str,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip():
        raise RuntimeError(
            "generate the environment lock only from a clean committed source tree"
        )
    gpu = _gpu_metadata()
    if not gpu.get("gpu_available") or "A100" not in (gpu.get("gpu_name") or ""):
        raise RuntimeError("environment lock must be generated on the selected A100 worker")
    if not container_ref or not container_digest:
        raise ValueError("immutable --container-ref and --container-digest are required")
    manifest_path = _resolve_repo_path(cfg, cfg["dataset"]["manifest_path"])
    snapshot = _environment_snapshot(
        cfg, config_path, manifest_path,
        container_ref=container_ref, container_digest=container_digest,
        include_freeze=True,
    )
    snapshot.update({
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent, text=True,
        ).strip()),
    })
    _write_json_atomic(output_path, snapshot)


def validate_environment_lock(cfg: dict, config_path: Path, lock_path: Path) -> dict:
    if not lock_path.exists():
        raise RuntimeError(
            f"production environment lock is missing: {lock_path}. Generate and "
            "commit it on the chosen A100 before any final-manifest run."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_path = _resolve_repo_path(cfg, cfg["dataset"]["manifest_path"])
    current = _environment_snapshot(cfg, config_path, manifest_path)
    # UUID is deliberately not a global equality constraint: multiple A100s
    # have different UUIDs. SKU/memory/software/container must match, while
    # each run records its UUID and resume enforces it separately.
    derived_config_ok = False
    if lock.get("experiment_config_sha256") != current.get("experiment_config_sha256"):
        frozen = cfg.get("frozen_allocation") or {}
        selector = cfg.get("allocation_selector") or {}
        trace_value = (selector.get("output_artifacts") or {}).get("selection_trace")
        if frozen and trace_value:
            trace_path = _resolve_repo_path(cfg, trace_value)
            if trace_path.exists():
                artifact = json.loads(trace_path.read_text(encoding="utf-8"))
                derived_config_ok = _derived_config_matches_lock(
                    cfg, lock, artifact
                )

    checks = (
        "python_version", "platform", "container_ref", "container_digest",
        "pip_freeze_sha256", "core_packages", "torch_cuda_runtime", "torch_cudnn",
        "final_manifest_file_sha256",
        "final_manifest_ids_sha256", "final_exclusion_ids_sha256",
        "model_revisions", "tokenizer_revisions", "dataset_revision",
        "prompt_bundle_version", "prompt_template_sha256",
        "source_bundle_sha256", "preflight_manifest_file_sha256",
        "preflight_manifest_ids_sha256",
        "timing_manifest_file_sha256", "timing_manifest_ids_sha256",
        "timing_repetitions",
    )
    mismatches = [key for key in checks if lock.get(key) != current.get(key)]
    if (
        lock.get("experiment_config_sha256") != current.get("experiment_config_sha256")
        and not derived_config_ok
    ):
        mismatches.append("experiment_config_sha256/derived_selection_link")
    for key in (
        "gpu_name", "gpu_total_memory_mib", "gpu_compute_capability",
        "gpu_driver_version",
    ):
        if (lock.get("gpu") or {}).get(key) != (current.get("gpu") or {}).get(key):
            mismatches.append(f"gpu.{key}")
    if not current["gpu"].get("gpu_available") or "A100" not in (
        current["gpu"].get("gpu_name") or ""
    ):
        mismatches.append("gpu.A100_required")
    if mismatches:
        raise RuntimeError(
            "worker does not match config/environment.lock.json: "
            + ", ".join(mismatches)
        )
    repo_root = Path(__file__).resolve().parent.parent
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip():
        raise RuntimeError("production worktree is dirty; commit the exact locked source first")
    return lock


def run(
    cfg: dict,
    run_id: str,
    n: int | None,
    seed: int | None,
    batch_size: int | None,
    *,
    use_manifest: bool | None = None,
    timing_mode: bool = False,
):
    treatments = resolve_treatments(cfg, run_id)
    stage_precision = {s: t["precision"] for s, t in treatments.items()}
    model_id = cfg["model_id"]
    n = n if n is not None else cfg["dataset"]["n"]
    seed = seed if seed is not None else cfg["dataset"]["eval_seed"]
    batch_size = batch_size if batch_size is not None else cfg["generation"]["batch_size"]
    min_batch_size = cfg["generation"].get("min_batch_size", batch_size)
    # SPEC §5b prediction 4. Off by default: costs an extra prefill per batch, and
    # enabling it mid-project would make runs non-comparable in cost accounting.
    log_confidence = bool(cfg["generation"].get("log_confidence", False))
    results_dir = _resolve_repo_path(cfg, cfg.get("results_dir", "results"))

    ds_cfg = cfg.get("dataset", {})
    configured_manifest = ds_cfg.get("manifest_path")
    if use_manifest is None:
        use_manifest = bool(configured_manifest)
    if use_manifest:
        if not configured_manifest:
            raise ValueError("final execution requires dataset.manifest_path")
        if n != ds_cfg["n"] or seed != ds_cfg["eval_seed"]:
            raise ValueError(
                "--n/--seed cannot override a frozen final manifest; use the "
                "explicit development-sample mode for prompt smoke tests"
            )
        if batch_size != 32 or min_batch_size != 32:
            raise ValueError(
                "final scored execution is pinned fail-closed to batch_size == "
                "min_batch_size == 32"
            )
        config_path = Path(cfg.get("_config_path", "config/experiment.yaml")).resolve()
        lock_path = _resolve_repo_path(
            cfg, cfg.get("environment_lock_path", "config/environment.lock.json")
        )
        environment_lock = validate_environment_lock(cfg, config_path, lock_path)
    else:
        # Development runs are outside the final experiment and may be smaller,
        # but they still never autotune within a run.
        min_batch_size = batch_size
        environment_lock = None

    print(f"=== run {run_id} | {model_id} | n={n} seed={seed} "
          f"batch={batch_size} prompts={prompts.PROMPT_BUNDLE_VERSION} ===")
    for s, t in treatments.items():
        print(f"    {s:<14} {t['model_id']} @ {t['precision']}")

    manifest_path = (
        _resolve_repo_path(cfg, configured_manifest)
        if use_manifest else None
    )
    manifest_ids = manifest_meta = None
    exclusions = set()
    warmup_ids = []
    timing_ids = []
    timing_meta = None
    timing_repetitions = 1
    if manifest_path is not None:
        manifest_ids, manifest_meta = load_id_manifest(manifest_path)
        exclusions_meta = manifest_meta.get("exclusions") or {}
        exclusion_ids = exclusions_meta.get("question_ids") or []
        exclusions = set(exclusion_ids)
        preflight_path = _resolve_repo_path(
            cfg,
            ds_cfg.get(
                "preflight_manifest_path",
                "config/manifests/preflight_excluded32.json",
            ),
        )
        warmup_ids, preflight_meta = load_id_manifest(preflight_path)
        if len(warmup_ids) != batch_size or not set(warmup_ids) <= exclusions:
            raise ValueError(
                "preflight manifest must contain exactly batch_size unique IDs, "
                "all drawn from the frozen final exclusion set"
            )
        if preflight_meta["question_ids_sha256"] != ds_cfg.get("preflight_manifest_sha256"):
            raise ValueError("configured preflight ordered-ID hash does not match manifest")
        if preflight_meta["manifest_file_sha256"] != ds_cfg.get(
            "preflight_manifest_file_sha256"
        ):
            raise ValueError("configured preflight file-byte hash does not match manifest")
        actual_exclusion_hash = exclusions_meta.get("ordered_ids_sha256")
        if ds_cfg.get("exclusion_sha256") != actual_exclusion_hash:
            raise ValueError(
                "dataset.exclusion_sha256 does not match the frozen manifest: "
                f"{ds_cfg.get('exclusion_sha256')} != {actual_exclusion_hash}"
            )
        if ds_cfg.get("manifest_file_sha256") != manifest_meta.get("manifest_file_sha256"):
            raise ValueError(
                "dataset.manifest_file_sha256 does not match manifest bytes: "
                f"{ds_cfg.get('manifest_file_sha256')} != "
                f"{manifest_meta.get('manifest_file_sha256')}"
            )
        if timing_mode:
            timing_cfg = cfg.get("timing") or {}
            allowed_timing_runs = set(timing_cfg.get("run_ids") or [])
            allowed_timing_runs.add(timing_cfg.get("post_selection_run_id"))
            allowed_timing_runs.discard(None)
            if run_id not in allowed_timing_runs:
                raise ValueError(
                    f"run {run_id!r} is excluded from the timing benchmark"
                )
            timing_path = _resolve_repo_path(cfg, timing_cfg["manifest_path"])
            timing_ids, timing_meta = load_id_manifest(timing_path)
            if len(timing_ids) != timing_cfg.get("n"):
                raise ValueError("timing manifest n does not match timing.n")
            if timing_meta["question_ids_sha256"] != timing_cfg.get("manifest_sha256"):
                raise ValueError("timing manifest ordered-ID hash mismatch")
            if timing_meta["manifest_file_sha256"] != timing_cfg.get(
                "manifest_file_sha256"
            ):
                raise ValueError("timing manifest file-byte hash mismatch")
            if not set(timing_ids) <= exclusions:
                raise ValueError("timing cohort must be entirely within final exclusions")
            if set(timing_ids) & set(warmup_ids):
                raise ValueError("timing and preflight cohorts must be disjoint")
            timing_source = timing_meta.get("source") or {}
            if (
                timing_source.get("final_question_ids_sha256")
                != ds_cfg.get("manifest_sha256")
                or timing_source.get("exclusion_ids_sha256")
                != ds_cfg.get("exclusion_sha256")
                or timing_meta.get("preflight_question_ids_sha256")
                != ds_cfg.get("preflight_manifest_sha256")
            ):
                raise ValueError("timing manifest source linkage does not match frozen cohorts")
            timing_repetitions = int(timing_cfg.get("repetitions", 1))
            if timing_repetitions < 2:
                raise ValueError("timing benchmark requires at least two repetitions")
    dataset_meta = {}
    auxiliary_questions = []
    questions = load_questions(
        n,
        seed=seed,
        split=ds_cfg.get("split", "validation"),
        config=ds_cfg.get("config", "distractor"),
        name=ds_cfg.get("name"),
        exclude=exclusions,
        manifest_path=manifest_path,
        manifest_sha256=ds_cfg.get("manifest_sha256") if use_manifest else None,
        revision=ds_cfg.get("revision"),
        metadata=dataset_meta,
        warmup_question_ids=(warmup_ids + timing_ids) if use_manifest else None,
        warmup_out=auxiliary_questions if use_manifest else None,
    )
    warmup_questions = auxiliary_questions[:len(warmup_ids)]
    timing_questions = auxiliary_questions[len(warmup_ids):]

    # SPEC §3 (revised): build the open-domain corpus before any model loads.
    # Failing here costs seconds; failing after a 3B model is resident costs the
    # GPU allocation. ~3.5 s to index 72k passages, CPU only.
    retr_cfg = cfg.get("retrieval") or {}
    print("building retrieval corpus...", flush=True)
    corpus = retrieval.build_corpus(
        name=ds_cfg.get("name") or "hotpotqa/hotpot_qa",
        split=ds_cfg.get("split", "validation"),
        revision=ds_cfg.get("revision"),
        configs=tuple(retr_cfg.get("corpus_configs", ("distractor", "fullwiki"))),
    )
    retriever = retrieval.RetrievalContext(
        corpus,
        k=int(retr_cfg.get("k", retrieval.K)),
        hop1=int(retr_cfg.get("hop1", retrieval.HOP1)),
    )
    print(f"  {len(corpus):,} passages, k={retriever.k}, "
          f"hop1={retriever.hop1}, hop2={retriever.k - retriever.hop1}")
    missing_gold = [
        t for q in questions for t in (q["supporting_facts"] or {}).get("title", ())
        if t not in retriever.index.title_to_index
    ]
    if missing_gold:
        # Gold outside the corpus is unreachable by ANY arm, capping every result
        # for a reason unrelated to the experiment.
        raise AssertionError(
            f"{len(missing_gold)} gold passages absent from the corpus; "
            f"examples: {sorted(set(missing_gold))[:3]}"
        )
    if manifest_ids is not None and [q["question_id"] for q in questions] != manifest_ids:
        raise AssertionError("loaded questions do not exactly match frozen manifest order")
    final_eval_question_ids_sha256 = dataset_meta["question_ids_sha256"]
    if timing_mode:
        # Timing never regenerates or scores the final 1,500. It uses the fixed
        # excluded batch-32 cohort as a separate systems benchmark.
        questions = timing_questions
        active_question_manifest_sha256 = timing_meta["question_ids_sha256"]
        artifact_n = len(questions)
        cohort_kind = "excluded_timing_benchmark"
    else:
        active_question_manifest_sha256 = final_eval_question_ids_sha256
        artifact_n = n
        cohort_kind = "final_evaluation" if use_manifest else "development"

    # n and seed are part of the filename, not just the metadata. Resume keys on
    # (question_id, stage, call_index), which says nothing about which sample the
    # question came from — so a dev run at n=30/seed=1234 sharing a file with the
    # real n=300/seed=7 run would silently interleave two different datasets into
    # one results file and no key collision would ever flag it.
    slug = result_slug(run_id, artifact_n, seed, model_id, treatments)
    if timing_mode:
        slug += "_timing"
    store = JsonlStore(results_dir / f"{slug}.jsonl")
    execution_session_id = uuid.uuid4().hex
    gpu_metadata = _gpu_metadata()
    store.open(execution_session_id)
    try:
        existing = store.read_existing()
    except Exception:
        store.close()
        raise
    if timing_mode and existing:
        store.close()
        raise RuntimeError(
            f"timing replay must be one fresh uncontended session; {store.path} "
            "already contains records. Move the old timing artifact aside before replay."
        )
    prior_uuids = {
        r.get("gpu_uuid") for r in existing
        if r.get("gpu_uuid") and r.get("record_type") in (
            "agent_call", "batch", "model_load"
        )
    }
    prior_names = {
        r.get("gpu_name") for r in existing
        if r.get("gpu_name") and r.get("record_type") in (
            "agent_call", "batch", "model_load"
        )
    }
    if prior_uuids and prior_uuids != {gpu_metadata.get("gpu_uuid")}:
        store.close()
        raise RuntimeError(
            f"resume would change GPU UUID: on disk {sorted(prior_uuids)}, "
            f"current {gpu_metadata.get('gpu_uuid')}"
        )
    if prior_names and prior_names != {gpu_metadata.get("gpu_name")}:
        store.close()
        raise RuntimeError(
            f"resume would change GPU SKU: on disk {sorted(prior_names)}, "
            f"current {gpu_metadata.get('gpu_name')}"
        )
    idx = index_records(existing)
    if existing:
        print(f"    [resume] found {len(idx)} completed agent calls on disk")

    meta_path = results_dir / f"{slug}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    frozen_allocation = cfg.get("frozen_allocation") or {}
    immutable_meta = {
        "run_id": run_id,
        "n": artifact_n,
        "final_eval_n": n,
        "seed": seed,
        "question_ids_sha256": active_question_manifest_sha256,
        "batch_size": batch_size,
        "timing_mode": timing_mode,
        "environment_lock_sha256": (
            _sha256_file(lock_path) if environment_lock is not None else None
        ),
        "git_commit": _git_commit(),
        "allocation_selection_artifact_sha256": frozen_allocation.get(
            "selection_artifact_sha256"
        ),
        "allocation_run_config_sha256": frozen_allocation.get(
            "run_config_sha256"
        ),
        "allocation_claim_status": frozen_allocation.get("claim_status"),
        "prompt_template_sha256": prompts.prompt_template_hashes(),
        "stage_config_fingerprints": {
            s: t["config_fingerprint"] for s, t in treatments.items()
        },
    }
    try:
        meaningful_existing = any(
            r.get("record_type") in ("agent_call", "batch", "model_load", "answer")
            for r in existing
        )
        _validate_resume_meta(
            meta, immutable_meta, meta_path,
            require_present=meaningful_existing,
        )
    except Exception:
        store.close()
        raise
    stage_meta = meta.get("stages", {})
    sessions = list(meta.get("execution_sessions") or [])
    sessions.append({
        "execution_session_id": execution_session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "timing_mode": timing_mode,
        **gpu_metadata,
    })
    meta.update({
        **immutable_meta,
        "model_id": model_id,
        "dataset": dataset_meta,
        # Retrieval is now an experimental variable, so its setup is provenance.
        "retrieval": retriever.fingerprint(),
        "cohort_kind": cohort_kind,
        "final_eval_question_ids_sha256": final_eval_question_ids_sha256,
        "timing_manifest_path": (
            str(timing_path) if timing_mode else None
        ),
        "timing_manifest_file_sha256": (
            timing_meta.get("manifest_file_sha256") if timing_mode else None
        ),
        "timing_repetitions": timing_repetitions if timing_mode else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "preflight_manifest_path": (
            str(preflight_path) if use_manifest else None
        ),
        "preflight_manifest_sha256": (
            preflight_meta.get("question_ids_sha256") if use_manifest else None
        ),
        "preflight_manifest_file_sha256": (
            preflight_meta.get("manifest_file_sha256") if use_manifest else None
        ),
        "manifest_file_sha256": (
            manifest_meta.get("manifest_file_sha256") if manifest_meta else None
        ),
        "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
        "stage_precision": stage_precision,
        "stage_models": {s: t["model_id"] for s, t in treatments.items()},
        "stage_model_revisions": {
            s: t["model_revision"] for s, t in treatments.items()
        },
        "stages": stage_meta,
        "execution_sessions": sessions,
        "git_commit": _git_commit(),
        "environment_lock_sha256": (
            _sha256_file(lock_path) if environment_lock is not None else None
        ),
    })
    try:
        _write_json_atomic(meta_path, meta)
    except Exception:
        store.close()
        raise
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
    stages = [r for r in prompts.PIPELINE_STAGES if r in treatments]
    warmup_idx = {}
    preflight_manifest_sha256 = question_ids_sha256(
        [q["question_id"] for q in warmup_questions]
    ) if warmup_questions else "development-sample"
    did_work = False
    try:
        for stage in stages:
            treatment = treatments[stage]
            precision, stage_model_id = treatment["precision"], treatment["model_id"]
            revision = treatment["model_revision"]
            tokenizer_revision = treatment["tokenizer_revision"]
            calls = build_stage_calls(stage, questions, idx, retriever=retriever)
            _validate_existing_stage_calls(
                existing, stage, calls, batch_size,
                run_id=run_id, treatment=treatment,
                question_manifest_sha256=active_question_manifest_sha256,
            )
            pending = [
                c for c in calls
                if (c["question_id"], stage, c["call_index"]) not in idx
            ]
            certified_batch_ids = _certified_batch_ids(
                existing, stage, calls, batch_size,
                config_fingerprint=treatment["config_fingerprint"],
                question_manifest_sha256=active_question_manifest_sha256,
            )
            expected_batch_count = (len(calls) + batch_size - 1) // batch_size
            print(f"\n--- stage {stage} @ {stage_model_id} {precision}: "
                  f"{len(calls)} calls, {len(pending)} pending ---")

            if (
                not pending
                and len(certified_batch_ids) == expected_batch_count
                and stage in stage_meta
            ):
                print("    (already complete, skipping model load)")
                continue
            if not pending:
                print(
                    "    (call records complete; repairing missing certificate "
                    "or stage metadata)"
                )
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
            want = treatment["config_fingerprint"]
            load_elapsed = 0.0
            load_reused = loaded == want
            pre_load_allocated = None
            if loaded != want:
                if model is not None:
                    # Drop our references FIRST, then reclaim — otherwise the old
                    # model is still reachable and survives until the new one has
                    # already been allocated. See models.unload's docstring.
                    model = tok = None
                    loaded = None
                    models.unload()
                pre_load_allocated = (
                    torch.cuda.memory_allocated() if torch.cuda.is_available() else None
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t_load = time.perf_counter()
                model, tok = models.load_model(
                    stage_model_id, precision, revision=revision,
                    tokenizer_revision=tokenizer_revision,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                loaded = want
                load_elapsed = time.perf_counter() - t_load
                print(f"    loaded in {load_elapsed:.1f}s")
            else:
                # Reused: the weights are already allocated, so the reset floor is
                # exactly those weights — the same floor a fresh load produces.
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    pre_load_allocated = torch.cuda.memory_allocated()
                print(f"    reusing loaded {stage_model_id} @ {precision}")

            post_load_allocated = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else None
            )
            resolved_revisions = models.resolved_revision_metadata(
                model, tok, revision, tokenizer_revision
            )
            load_delta = (
                (post_load_allocated - pre_load_allocated) / 1024**2
                if post_load_allocated is not None and pre_load_allocated is not None
                else None
            )
            store.write([{
                "record_type": "model_load",
                "run_id": run_id,
                "stage": stage,
                "model_id": stage_model_id,
                "precision": precision,
                "model_revision": revision,
                "tokenizer_revision": tokenizer_revision,
                **resolved_revisions,
                "config_fingerprint": treatment["config_fingerprint"],
                "execution_session_id": execution_session_id,
                "load_reused": load_reused,
                "load_wall_s": load_elapsed,
                "cuda_allocated_delta_mib": load_delta,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **gpu_metadata,
            }])
            store.durable_flush()

            memory = models.memory_footprint_mib(model)
            census = models.param_census(model)
            if use_manifest:
                warmup_calls = build_stage_calls(
                    stage, warmup_questions, warmup_idx, retriever=retriever
                )
                _preflight_stage(
                    model, tok, stage, warmup_calls, precision, run_id,
                    batch_size, store, warmup_idx, model_id=stage_model_id,
                    execution_session_id=execution_session_id,
                    gpu_metadata=gpu_metadata,
                    config_fingerprint=treatment["config_fingerprint"],
                    model_revision=revision,
                    tokenizer_revision=tokenizer_revision,
                    question_manifest_sha256=preflight_manifest_sha256,
                    preflight_manifest_sha256=preflight_manifest_sha256,
                )
            t_stage = time.perf_counter()
            bs = _run_stage(
                model, tok, stage, calls, precision, run_id,
                batch_size, min_batch_size, store, idx,
                log_confidence=log_confidence, model_id=stage_model_id,
                execution_session_id=execution_session_id,
                gpu_metadata=gpu_metadata,
                timing_eligible=timing_mode,
                config_fingerprint=treatment["config_fingerprint"],
                model_revision=revision,
                tokenizer_revision=tokenizer_revision,
                question_manifest_sha256=active_question_manifest_sha256,
                certified_batch_ids=certified_batch_ids,
                phase="timing_benchmark" if timing_mode else "scored",
                timing_repeat=0 if timing_mode else None,
            )
            if timing_mode:
                for repeat in range(1, timing_repetitions):
                    _run_timing_repeat(
                        model, tok, stage, calls, precision, run_id,
                        batch_size, store, idx, repeat=repeat,
                        model_id=stage_model_id,
                        execution_session_id=execution_session_id,
                        gpu_metadata=gpu_metadata,
                        config_fingerprint=treatment["config_fingerprint"],
                        model_revision=revision,
                        tokenizer_revision=tokenizer_revision,
                        question_manifest_sha256=active_question_manifest_sha256,
                    )
            store.durable_flush()
            elapsed = time.perf_counter() - t_stage

            peak = (torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available() else None)
            peak_reserved = (torch.cuda.max_memory_reserved() / 1024**2
                             if torch.cuda.is_available() else None)
            stage_meta[stage] = {
                **models.quant_config_metadata(precision),
                "model_id": stage_model_id,
                "model_revision": revision,
                "tokenizer_revision": tokenizer_revision,
                **resolved_revisions,
                "config_fingerprint": treatment["config_fingerprint"],
                "peak_vram_allocated_mib": round(peak, 1) if peak is not None else None,
                "peak_vram_reserved_mib": (
                    round(peak_reserved, 1) if peak_reserved is not None else None
                ),
                **{k: round(v, 1) if isinstance(v, float) else v
                   for k, v in memory.items()},
                "model_load_wall_s": round(load_elapsed, 3),
                "model_load_reused": load_reused,
                "model_load_allocated_delta_mib": (
                    round(load_delta, 1) if load_delta is not None else None
                ),
                "calls": len(calls),
                "stage_wall_s": round(elapsed, 1),
                "final_batch_size": bs,
                "oom_autotuned": False,
                **census,
            }
            meta["stages"] = stage_meta
            meta["last_completed_stage"] = stage
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(meta_path, meta)
            vram = f"  peak_vram_mib={peak:.0f}" if peak is not None else ""
            print(f"\n    done in {elapsed:.1f}s  final_batch_size={bs}{vram}")

        if model is not None:
            model = tok = None
            models.unload()

        if timing_mode:
            # Systems benchmark only: no inferential F1/EM records are created.
            answers = []
        else:
            answers = build_answer_records(
                questions, idx, run_id,
                question_manifest_sha256=active_question_manifest_sha256,
                retriever=retriever,
            )
            for answer in answers:
                answer.update({
                    "question_manifest_sha256": active_question_manifest_sha256,
                    "execution_session_id": execution_session_id,
                    "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
                })
            have = {
                r["question_id"] for r in existing
                if r.get("record_type") == "answer"
            }
            new_answers = [a for a in answers if a["question_id"] not in have]
            store.write(new_answers)
            store.durable_flush()
    finally:
        store.close()
        if model is not None:
            model = tok = None
            models.unload()

    wall = time.perf_counter() - t_run

    # Distinct-weight concurrent residency is primary; isolated role services
    # and the actually executed sequential peak answer different topologies.
    isolated_role_service_weight_mib = sum(
        s["model_footprint_mib"] for s in stage_meta.values()
        if s.get("model_footprint_mib") is not None
    )
    # A stage that resumed as already-complete contributes no stage_meta, so a
    # run finished across two invocations can end up with footprints summed over
    # a SUBSET of its stages — silently, and smaller than the truth. That is the
    # Kaggle idle-timeout path PROGRESS.md records as having already happened
    # once, and SPEC §7 calls the footprint "the number the paper reports".
    # Refuse to report a number computed from partial metadata.
    missing = [s for s in stages if s not in stage_meta]
    if missing:
        isolated_role_service_weight_mib = deduped = None
        print(f"\n  !! metadata incomplete: no stage record for {missing}. Footprints "
              f"suppressed rather than under-reported — rerun with results/{slug}.jsonl "
              f"removed, or merge in the meta.json from the earlier invocation.")
    else:
        deduped = deduplicated_footprint_mib(stage_meta)
    sequential_peak_allocated_mib = max(
        (s.get("peak_vram_allocated_mib") or 0 for s in stage_meta.values()),
        default=0,
    ) or None
    sequential_peak_reserved_mib = max(
        (s.get("peak_vram_reserved_mib") or 0 for s in stage_meta.values()),
        default=0,
    ) or None
    meta.update({
        "run_id": run_id,
        "model_id": model_id,
        "n": artifact_n,
        "final_eval_n": n,
        "seed": seed,
        "batch_size": batch_size,
        "log_confidence": log_confidence,
        "prompt_bundle_version": prompts.PROMPT_BUNDLE_VERSION,
        "prompt_template_sha256": prompts.prompt_template_hashes(),
        "stage_precision": stage_precision,
        "stage_models": {s: t["model_id"] for s, t in treatments.items()},
        "stages": stage_meta,
        "isolated_role_service_weight_mib": (
            round(isolated_role_service_weight_mib, 1)
            if isolated_role_service_weight_mib is not None else None
        ),
        "deduplicated_distinct_weight_mib": (
            round(deduped, 1) if deduped is not None else None
        ),
        "deduplicated_concurrent_model_footprint_mib": (
            round(deduped, 1) if deduped is not None else None
        ),
        "isolated_role_service_model_footprint_mib": (
            round(isolated_role_service_weight_mib, 1)
            if isolated_role_service_weight_mib is not None else None
        ),
        "sequential_peak_vram_mib": (
            round(sequential_peak_allocated_mib, 1)
            if sequential_peak_allocated_mib is not None else None
        ),
        "sequential_peak_vram_allocated_mib": (
            round(sequential_peak_allocated_mib, 1)
            if sequential_peak_allocated_mib is not None else None
        ),
        "sequential_peak_vram_reserved_mib": (
            round(sequential_peak_reserved_mib, 1)
            if sequential_peak_reserved_mib is not None else None
        ),
        # The store is closed and durably flushed before this point, so this
        # binds analysis to the exact finalized record stream.
        "jsonl_sha256": _sha256_file(store.path),
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
    _write_json_atomic(meta_path, meta)

    if answers:
        em = sum(a["em"] for a in answers) / len(answers)
        f1 = sum(a["f1"] for a in answers) / len(answers)
        print(f"\n\n=== {run_id} complete: EM {100*em:.1f}%  F1 {100*f1:.1f}%  "
              f"n={len(answers)}  wall {wall:.0f}s ===")
    else:
        print(
            f"\n\n=== {run_id} timing benchmark complete: excluded cohort "
            f"n={artifact_n} wall {wall:.0f}s; no F1/EM records ==="
        )
    fmt = lambda v: f"{v:.0f}" if v else "unavailable (incomplete metadata)"  # noqa: E731
    print(f"    deduplicated_distinct_weight_mib={fmt(deduped)}  (primary topology)")
    print(
        "    isolated_role_service_weight_mib="
        f"{fmt(isolated_role_service_weight_mib)}"
    )
    print(f"    -> {store.path}")
    print(f"    -> {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--run", default=None)
    ap.add_argument("--n", type=int, default=None, help="override dataset.n")
    ap.add_argument("--seed", type=int, default=None, help="override dataset.eval_seed")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--dev-sample", action="store_true",
        help="explicitly bypass final manifest/environment lock for seed 0/1234 smoke tests",
    )
    ap.add_argument(
        "--timing-mode", action="store_true",
        help="fresh reserved-A100 timing replay; writes a separate *_timing artifact",
    )
    ap.add_argument(
        "--write-environment-lock", nargs="?", const="config/environment.lock.json",
        help="capture the selected A100 software/container/GPU lock and exit",
    )
    ap.add_argument("--container-ref", default=None)
    ap.add_argument("--container-digest", default=None)
    ap.add_argument(
        "--validate-environment-lock", action="store_true",
        help="validate this worker against config/environment.lock.json and exit",
    )
    ap.add_argument("--model-id", default=None,
                    help="override the base model (SPEC §3: a model is a config value, "
                         "not a plugin — this is just so one config can drive both)")
    ap.add_argument("--small-model-id", default=None,
                    help="override the `small` alias. Pair this with --model-id when "
                         "running a Phase S definition on model 2, or the small stage "
                         "silently keeps model 1's sibling.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    if args.model_id:
        cfg["model_id"] = args.model_id
    if args.small_model_id:
        cfg.setdefault("models", {})["small"] = args.small_model_id
    if args.write_environment_lock:
        if not args.container_ref or not args.container_digest:
            ap.error("--write-environment-lock requires --container-ref and --container-digest")
        write_environment_lock(
            cfg, config_path, _resolve_repo_path(cfg, args.write_environment_lock),
            container_ref=args.container_ref,
            container_digest=args.container_digest,
        )
        return
    if args.validate_environment_lock:
        validate_environment_lock(
            cfg, config_path,
            _resolve_repo_path(
                cfg, cfg.get("environment_lock_path", "config/environment.lock.json")
            ),
        )
        print("environment lock matches this worker")
        return
    if not args.run:
        ap.error("--run is required unless writing/validating the environment lock")
    if args.dev_sample:
        dev_seed = args.seed if args.seed is not None else 1234
        dev_n = args.n if args.n is not None else 10
        if dev_seed not in (0, 1234) or dev_n > 30:
            ap.error("--dev-sample is restricted to seed 0/1234 and n <= 30")
    if args.timing_mode and args.dev_sample:
        ap.error("--timing-mode is a final-manifest production replay, not a dev smoke test")
    run(
        cfg, args.run, args.n, args.seed, args.batch_size,
        use_manifest=not args.dev_sample,
        timing_mode=args.timing_mode,
    )


if __name__ == "__main__":
    main()
