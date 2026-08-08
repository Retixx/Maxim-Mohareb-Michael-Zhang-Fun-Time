"""Prefetch every pinned experiment asset and verify it without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import retrieval  # noqa: E402
from src.contracts import validate_model_contract  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest_files(config: dict) -> None:
    contracts = (
        (config["dataset"], "manifest_path", "manifest_file_sha256"),
        (config["dataset"], "preflight_manifest_path", "preflight_manifest_file_sha256"),
        (config["timing"], "manifest_path", "manifest_file_sha256"),
        (config["pilot"], "manifest_path", "manifest_file_sha256"),
    )
    for section, path_key, hash_key in contracts:
        path = ROOT / section[path_key]
        if not path.exists() or _sha256(path) != section[hash_key]:
            raise RuntimeError(f"manifest cache contract failed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument(
        "--offline-verify-only", action="store_true",
        help="forbid network and prove all snapshots/dataset shards are already cached",
    )
    parser.add_argument("--minimum-free-gib", type=float, default=15.0)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_model_contract(config)
    _verify_manifest_files(config)
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < args.minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free; require {args.minimum_free_gib:.1f} GiB"
        )

    dataset = config["dataset"]
    retrieval_config = config["retrieval"]
    old_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    old_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
    if args.offline_verify_only:
        # These variables are read when the libraries are imported, so set them
        # before importing either package. ``local_files_only`` alone protects
        # model snapshots but not dataset metadata/shard lookups.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        for dataset_config in retrieval_config["corpus_configs"]:
            loaded = load_dataset(
                dataset["name"], dataset_config,
                split=dataset["split"], revision=dataset["revision"],
                download_mode="reuse_cache_if_exists",
            )
            if len(loaded) <= 0:
                raise RuntimeError(f"cached dataset is empty: {dataset_config}")
        snapshots = {}
        model_ids = [config["model_id"], *config["models"].values()]
        for model_id in model_ids:
            revision = config["model_revisions"][model_id]
            snapshot = snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=args.offline_verify_only,
            )
            snapshots[model_id] = str(snapshot)

        corpus = retrieval.build_corpus(
            name=dataset["name"], split=dataset["split"],
            revision=dataset["revision"],
            configs=tuple(retrieval_config["corpus_configs"]),
        )
        context = retrieval.RetrievalContext(
            corpus,
            k=int(retrieval_config["k"]),
            anchor_k=int(retrieval_config["anchor_k"]),
        )
        fingerprint = context.fingerprint()
        if (
            fingerprint["corpus_passages"]
            != int(retrieval_config["expected_corpus_passages"])
            or fingerprint["corpus_sha256"]
            != retrieval_config["expected_corpus_sha256"]
            or fingerprint["algorithm"] != retrieval_config["algorithm"]
            or fingerprint["query_policy"] != retrieval_config["query_policy"]
            or fingerprint["initial_query_source"]
            != retrieval_config["initial_query_source"]
            or int(fingerprint["grounded_followup_k"])
            != int(retrieval_config["grounded_followup_k"])
            or int(fingerprint["anchor_k"])
            != int(retrieval_config["anchor_k"])
            or int(fingerprint["task_k"])
            != int(retrieval_config["task_k"])
            or fingerprint["grounded_followup_requires_evidence"] is not False
            or retrieval_config["grounded_followup_requires_evidence"] is not False
        ):
            raise RuntimeError(f"cached corpus fingerprint changed: {fingerprint}")
    finally:
        if old_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hub_offline
        if old_datasets_offline is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = old_datasets_offline

    report = {
        "schema_version": 1,
        "offline_verify_only": args.offline_verify_only,
        "free_gib_after": shutil.disk_usage(ROOT).free / 1024**3,
        "models": snapshots,
        "retrieval": fingerprint,
    }
    output = ROOT / "logs" / "prefetch_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"asset verification passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
