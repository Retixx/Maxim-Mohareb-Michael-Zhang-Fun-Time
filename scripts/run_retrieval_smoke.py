"""Prepare or execute the frozen local-GPU retrieval accuracy smoke gate.

This path is deliberately separate from the A100 campaign. It derives a
two-arm configuration from the authoritative experiment, preserves the frozen
excluded n=200 cohort and retrieval contract, and compares uniform MA-RAG to a
one-call control using the same Qwen3 model/revision/4-bit configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SMOKE_PROFILE = "local_gpu_memory_matched_4bit_v1"
ALLOWED_MODEL_ALIASES = ("tiny", "small")
MAX_BATCH_SIZE = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_analysis_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts[:2] == ("analysis", "local_smoke")
    )


def derive_smoke_config(
    source: dict,
    *,
    source_config_path: Path,
    model_alias: str = "tiny",
    batch_size: int = MAX_BATCH_SIZE,
) -> dict:
    """Return an isolated local-smoke config without mutating ``source``."""
    from src.contracts import validate_model_contract

    validate_model_contract(source)
    if model_alias not in ALLOWED_MODEL_ALIASES:
        raise ValueError(
            f"smoke model must be one of {ALLOWED_MODEL_ALIASES}, got {model_alias!r}"
        )
    if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= MAX_BATCH_SIZE:
        raise ValueError(f"smoke batch size must be in [1, {MAX_BATCH_SIZE}]")
    batch_size = int(batch_size)
    models = source.get("models") or {}
    if model_alias not in models:
        raise ValueError(f"source config does not define model alias {model_alias!r}")
    model_id = models[model_alias]
    revision = (source.get("model_revisions") or {}).get(model_id)
    if not revision:
        raise ValueError(f"source config has no revision entry for {model_id}")

    derived = copy.deepcopy(source)
    label = f"{model_alias}_4bit_bs{batch_size}"
    derived["model_id"] = model_id
    derived["generation"]["batch_size"] = batch_size
    derived["generation"]["min_batch_size"] = batch_size
    derived["results_dir"] = f"analysis/local_smoke/{label}/results"
    derived["pilot"]["gate_artifact"] = (
        f"analysis/local_smoke/{label}/gate.json"
    )
    derived["runs"]["baseline"] = {
        "planner": "4bit",
        "step_definer": "4bit",
        "extractor": "4bit",
        "qa": "4bit",
    }
    derived["runs"]["single_fp16"] = {"solo": "4bit"}
    try:
        source_label = source_config_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_label = source_config_path.name
    derived["local_smoke"] = {
        "profile": SMOKE_PROFILE,
        "source_config_path": source_label,
        "source_config_sha256": _sha256(source_config_path),
        "model_alias": model_alias,
        "model_id": model_id,
        "precision": "4bit",
        "batch_size": batch_size,
        "max_batch_size": MAX_BATCH_SIZE,
        "model_revision_status": (
            "unpinned_TBD" if revision == "TBD" else "pinned"
        ),
        "publication_claim": False,
        "production_gate_authority": False,
    }
    validate_smoke_config(derived)
    return derived


def validate_smoke_config(config: dict) -> dict:
    """Fail closed unless ``config`` is exactly the bounded smoke treatment."""
    from src.contracts import validate_model_contract

    validate_model_contract(config, allow_local_smoke=True)
    smoke = config.get("local_smoke") or {}
    if smoke.get("profile") != SMOKE_PROFILE:
        raise ValueError("local smoke profile is missing or stale")
    alias = smoke.get("model_alias")
    if alias not in ALLOWED_MODEL_ALIASES:
        raise ValueError("local smoke model alias is not approved")
    model_id = (config.get("models") or {}).get(alias)
    if not model_id or model_id != smoke.get("model_id") or config.get("model_id") != model_id:
        raise ValueError("local smoke model identity drifted")
    if smoke.get("precision") != "4bit":
        raise ValueError("local smoke precision must be 4bit")
    batch_size = (config.get("generation") or {}).get("batch_size")
    min_batch_size = (config.get("generation") or {}).get("min_batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= MAX_BATCH_SIZE
        or min_batch_size != batch_size
        or smoke.get("batch_size") != batch_size
        or smoke.get("max_batch_size") != MAX_BATCH_SIZE
    ):
        raise ValueError("local smoke batch contract drifted")
    pilot = config.get("pilot") or {}
    if pilot.get("run_ids") != ["baseline", "single_fp16"]:
        raise ValueError("local smoke must compare baseline to single_fp16")
    if int(pilot.get("n", -1)) < 200:
        raise ValueError("local smoke requires at least 200 frozen excluded questions")
    retrieval = config.get("retrieval") or {}
    if (
        retrieval.get("k") != 10
        or retrieval.get("anchor_k") != 7
        or retrieval.get("task_k") != 3
        or retrieval.get("grounded_followup_requires_evidence") is not False
    ):
        raise ValueError("local smoke retrieval contract drifted")
    if not _safe_analysis_path(config.get("results_dir")) or not _safe_analysis_path(
        pilot.get("gate_artifact")
    ):
        raise ValueError("local smoke outputs must remain under analysis/local_smoke")
    source_hash = smoke.get("source_config_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise ValueError("local smoke source config hash is malformed")
    expected_ma = {
        "planner": "4bit",
        "step_definer": "4bit",
        "extractor": "4bit",
        "qa": "4bit",
    }
    if (config.get("runs") or {}).get("baseline") != expected_ma:
        raise ValueError("local smoke MA treatment is not uniform 4bit")
    if (config.get("runs") or {}).get("single_fp16") != {"solo": "4bit"}:
        raise ValueError("local smoke single-hop treatment is not 4bit")

    # Import lazily so config-only callers do not pay the runner import cost.
    from src.runner import resolve_treatments

    ma = resolve_treatments(config, "baseline")
    single = resolve_treatments(config, "single_fp16")
    ma_fingerprints = {row["config_fingerprint"] for row in ma.values()}
    single_fingerprints = {row["config_fingerprint"] for row in single.values()}
    if (
        {row["model_id"] for row in ma.values()} != {model_id}
        or {row["model_id"] for row in single.values()} != {model_id}
        or {row["precision"] for row in ma.values()} != {"4bit"}
        or {row["precision"] for row in single.values()} != {"4bit"}
        or len(ma_fingerprints) != 1
        or ma_fingerprints != single_fingerprints
    ):
        raise ValueError("local smoke treatments are not memory matched")
    return {
        "profile": SMOKE_PROFILE,
        "model_alias": alias,
        "model_id": model_id,
        "batch_size": batch_size,
        "n": int(pilot["n"]),
        "config_fingerprint": next(iter(ma_fingerprints)),
        "model_revision_status": smoke.get("model_revision_status"),
    }


def _write_yaml_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_committed_tracked_source() -> None:
    for command in (
        ["git", "diff", "--quiet", "HEAD", "--"],
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--model", choices=ALLOWED_MODEL_ALIASES, default="tiny")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--output-config", default=None)
    parser.add_argument(
        "--allow-unpinned-tbd",
        action="store_true",
        help="acknowledge that this local smoke is non-publication evidence when the configured revision is TBD",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.config).resolve()
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    derived = derive_smoke_config(
        source,
        source_config_path=source_path,
        model_alias=args.model,
        batch_size=args.batch_size,
    )
    label = f"{args.model}_4bit_bs{args.batch_size}"
    output_path = (
        Path(args.output_config).resolve()
        if args.output_config
        else ROOT / "analysis" / "local_smoke" / label / "experiment.yaml"
    )
    _write_yaml_atomic(output_path, derived)
    summary = validate_smoke_config(derived)
    print(f"prepared {SMOKE_PROFILE}: {summary}")
    print(f"derived config: {output_path}")
    if not args.execute:
        tbd_flag = (
            " --allow-unpinned-tbd"
            if summary["model_revision_status"] == "unpinned_TBD"
            else ""
        )
        print(
            "execute with: "
            f"{sys.executable} scripts/run_retrieval_smoke.py --model {args.model} "
            f"--batch-size {args.batch_size}{tbd_flag} --execute"
        )
        return 0

    _require_committed_tracked_source()
    if (
        summary["model_revision_status"] == "unpinned_TBD"
        and not args.allow_unpinned_tbd
    ):
        parser.error(
            "model revision is TBD; pass --allow-unpinned-tbd only for this "
            "explicitly non-publication local smoke"
        )
    retrieval_gate_path = (
        ROOT / "analysis" / "local_smoke" / label / "retrieval_gate.json"
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/check_retrieval_gate.py",
            "--config",
            str(source_path),
            "--output",
            str(retrieval_gate_path),
        ],
        cwd=ROOT,
        check=True,
    )
    for run_id in derived["pilot"]["run_ids"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.runner",
                "--config",
                str(output_path),
                "--run",
                run_id,
                "--pilot-mode",
                "--local-smoke-mode",
                "--batch-size",
                str(args.batch_size),
            ],
            cwd=ROOT,
            check=True,
        )
    gate_path = ROOT / derived["pilot"]["gate_artifact"]
    subprocess.run(
        [
            sys.executable,
            "scripts/check_pilot.py",
            "--config",
            str(output_path),
            "--output",
            str(gate_path),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
