"""Fail-closed A100 operator entrypoint for preflight, pilot, and campaigns."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner import _gpu_metadata  # noqa: E402


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def require_a100() -> dict:
    gpu = _gpu_metadata()
    if gpu.get("gpu_available") is not True or "A100" not in (gpu.get("gpu_name") or ""):
        raise RuntimeError(f"one NVIDIA A100 is required; detected {gpu}")
    return gpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "accuracy", "timing"))
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--container-ref")
    parser.add_argument("--container-digest")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gpu = require_a100()
    print(json.dumps({"phase": args.phase, "gpu": gpu}, indent=2))

    if args.phase == "prepare":
        if not args.container_ref or not args.container_digest:
            parser.error("prepare requires --container-ref and --container-digest")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip():
            raise RuntimeError("commit the exact source before A100 prepare")
        run([sys.executable, "scripts/prefetch_assets.py", "--config", str(config_path),
             "--offline-verify-only"])
        run([sys.executable, "-m", "pytest", "-q"])
        run([
            sys.executable, "-m", "src.runner", "--config", str(config_path),
            "--write-environment-lock", "config/environment.lock.json",
            "--container-ref", args.container_ref,
            "--container-digest", args.container_digest,
        ])
        print("commit and push config/environment.lock.json before running the pilot")
        return 0

    if args.container_ref:
        os.environ["EXPERIMENT_CONTAINER_REF"] = args.container_ref
    if args.container_digest:
        os.environ["EXPERIMENT_CONTAINER_DIGEST"] = args.container_digest
    run([
        sys.executable, "-m", "src.runner", "--config", str(config_path),
        "--validate-environment-lock",
    ])
    command = [
        sys.executable, "scripts/run_campaign.py", "--config", str(config_path),
        "--kind", args.phase, "--workers", str(args.workers),
        "--worker-index", str(args.worker_index), "--execute",
    ]
    run(command)
    if args.phase == "pilot":
        print(
            "pilot GO must now be committed as analysis/pilot_gate.json and pulled "
            "unchanged by every final worker"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
