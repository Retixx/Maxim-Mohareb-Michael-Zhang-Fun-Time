"""Freeze the final HotpotQA question-ID manifest.

This is a provenance tool, not a runtime sampler.  It deliberately reads the
old pilot IDs from the committed result blob instead of reconstructing them
from ``random.sample``.  The final runner consumes the generated manifest and
must never draw a replacement sample on its own.

Run from the repository root:

    python scripts/freeze_final_sample.py
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

import requests


DATASET = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
FINAL_N = 1500
FINAL_SEED = 20260805

PILOT_COMMIT = "0e71d4125c6430c5dea1a385ed322f225b446899"
PILOT_PATH = "results/baseline_qwen2.5-1.5b_n3000_seed7.jsonl"
PILOT_BLOB = "a3cf03540ffc80db5fd6f0dee7d5e7bac87a678b"

DEV_COMMIT = "078c699e119a6d026249ae8e1c19b5cc59558670"
DEV_PATH = "results/baseline_n30_seed1234.jsonl"
DEV_BLOB = "2eb24d2db2813ebbd8da9d05dfe547dcc8594b47"

OUTPUT = Path("config/manifests/final_n1500_seed20260805.json")
PREFLIGHT_OUTPUT = Path("config/manifests/preflight_excluded32.json")
TIMING_OUTPUT = Path("config/manifests/timing_excluded128_seed20260805.json")
PREFLIGHT_N = 32
TIMING_N = 128
TIMING_SEED = 20260805


def _sha256_lines(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_blob(commit: str, path: str, expected_blob: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"], text=True
    ).strip()
    if actual != expected_blob:
        raise RuntimeError(
            f"source blob changed for {commit}:{path}: "
            f"expected {expected_blob}, got {actual}"
        )
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], text=True, encoding="utf-8"
    )


def _question_ids_from_jsonl(text: str) -> set[str]:
    ids = set()
    for line in text.splitlines():
        if line.strip():
            ids.add(json.loads(line)["question_id"])
    return ids


def _dataset_ids() -> list[str]:
    """Read validation IDs in row order from the pinned Parquet artifact."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "freezing requires pyarrow (runtime evaluation does not): "
            "python -m pip install pyarrow"
        ) from exc

    url = (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/"
        "distractor/validation-00000-of-00001.parquet"
    )
    temp_path: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as parquet_file:
                temp_path = Path(parquet_file.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    parquet_file.write(chunk)
        ids = pq.read_table(temp_path, columns=["id"])["id"].to_pylist()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    if len(ids) != 7405 or len(set(ids)) != len(ids):
        raise RuntimeError(
            f"expected 7,405 unique validation IDs, got {len(ids)} rows / "
            f"{len(set(ids))} unique"
        )
    return ids


def _write_auxiliary_manifests(final_manifest: dict) -> None:
    """Freeze non-scored A100 warm-up and timing cohorts from exclusions.

    Neither artifact can overlap the final 1,500.  The timing cohort also
    excludes the warm-up batch so measured calls are not warmed up in advance.
    Its sampled order is retained as the canonical timing-batch order.
    """
    exclusions = list(final_manifest["exclusions"]["question_ids"])
    final_ids = set(final_manifest["question_ids"])
    if len(exclusions) != len(set(exclusions)):
        raise RuntimeError("final manifest exclusion IDs are not unique")

    preflight_ids = exclusions[:PREFLIGHT_N]
    timing_pool = [qid for qid in exclusions if qid not in set(preflight_ids)]
    timing_ids = random.Random(TIMING_SEED).sample(timing_pool, TIMING_N)
    if final_ids & (set(preflight_ids) | set(timing_ids)):
        raise RuntimeError("auxiliary cohort overlaps the final evaluation sample")
    if set(preflight_ids) & set(timing_ids):
        raise RuntimeError("timing cohort overlaps the warm-up cohort")

    source_file_sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    common_source = {
        "final_manifest_path": OUTPUT.as_posix(),
        "final_manifest_file_sha256": source_file_sha256,
        "final_question_ids_sha256": final_manifest["question_ids_sha256"],
        "exclusion_ids_sha256": final_manifest["exclusions"]["ordered_ids_sha256"],
    }
    preflight = {
        "schema_version": 1,
        "purpose": "fixed excluded batch-32 A100 warm-up and smoke check; never scored",
        "source": common_source,
        "selection": {
            "n": PREFLIGHT_N,
            "algorithm": "first 32 IDs in final manifest exclusions.question_ids",
        },
        "question_ids_sha256": _sha256_lines(preflight_ids),
        "question_ids": preflight_ids,
    }
    timing = {
        "schema_version": 1,
        "purpose": "fixed excluded A100 steady-state timing cohort; never scored for accuracy",
        "source": common_source,
        "selection": {
            "n": TIMING_N,
            "seed": TIMING_SEED,
            "algorithm": (
                "CPython random.Random(seed).sample over the ordered exclusion list "
                "after removing the 32 preflight IDs; sampled order retained"
            ),
            "eligible_count": len(timing_pool),
        },
        "preflight_question_ids_sha256": preflight["question_ids_sha256"],
        "question_ids_sha256": _sha256_lines(timing_ids),
        "question_ids": timing_ids,
    }
    for path, payload in ((PREFLIGHT_OUTPUT, preflight), (TIMING_OUTPUT, timing)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path} with {len(payload['question_ids'])} IDs")


def main() -> None:
    dataset_ids = _dataset_ids()
    dataset_id_set = set(dataset_ids)

    pilot_ids = _question_ids_from_jsonl(
        _git_blob(PILOT_COMMIT, PILOT_PATH, PILOT_BLOB)
    )
    if len(pilot_ids) != 3000:
        raise RuntimeError(f"expected 3,000 exact pilot IDs, got {len(pilot_ids)}")

    dev_seed1234_ids = _question_ids_from_jsonl(
        _git_blob(DEV_COMMIT, DEV_PATH, DEV_BLOB)
    )
    if len(dev_seed1234_ids) != 30:
        raise RuntimeError(
            f"expected 30 exact seed-1234 development IDs, got "
            f"{len(dev_seed1234_ids)}"
        )

    # No seed-0/n=10 result with unambiguous seed metadata survives in git.
    # Reconstruct this development set from the pinned dataset row order and
    # the original implementation.  This provenance is intentionally distinct
    # from the exact result-derived exclusions above.
    dev_seed0_indices = sorted(random.Random(0).sample(range(len(dataset_ids)), 10))
    dev_seed0_ids = {dataset_ids[index] for index in dev_seed0_indices}

    excluded = pilot_ids | dev_seed1234_ids | dev_seed0_ids
    if not excluded <= dataset_id_set:
        raise RuntimeError("one or more exclusion IDs are absent from the pinned dataset")

    eligible_indices = [
        index for index, question_id in enumerate(dataset_ids)
        if question_id not in excluded
    ]
    final_indices = sorted(
        random.Random(FINAL_SEED).sample(eligible_indices, FINAL_N)
    )
    final_ids = [dataset_ids[index] for index in final_indices]
    if set(final_ids) & excluded:
        raise RuntimeError("final sample overlaps an excluded question")

    manifest = {
        "schema_version": 1,
        "purpose": "single final evaluation sample shared by every arm",
        "dataset": {
            "name": DATASET,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "row_count": len(dataset_ids),
            "ordered_ids_sha256": _sha256_lines(dataset_ids),
        },
        "sampling": {
            "n": FINAL_N,
            "seed": FINAL_SEED,
            "algorithm": (
                "CPython random.Random(seed).sample over eligible dataset row "
                "indices, followed by ascending row-index order"
            ),
            "eligible_count": len(eligible_indices),
        },
        "exclusions": {
            "unique_count": len(excluded),
            "ordered_ids_sha256": _sha256_lines(sorted(excluded)),
            "question_ids": sorted(excluded),
            "sources": [
                {
                    "purpose": "old n=3000 seed=7 design pilot",
                    "method": "exact unique question IDs read from committed JSONL",
                    "git_commit": PILOT_COMMIT,
                    "path": PILOT_PATH,
                    "git_blob": PILOT_BLOB,
                    "id_count": len(pilot_ids),
                },
                {
                    "purpose": "prompt development, seed=1234 n=30",
                    "method": "exact unique question IDs read from committed JSONL",
                    "git_commit": DEV_COMMIT,
                    "path": DEV_PATH,
                    "git_blob": DEV_BLOB,
                    "id_count": len(dev_seed1234_ids),
                },
                {
                    "purpose": "prompt development, seed=0 n=10",
                    "method": (
                        "reconstructed from pinned dataset row order and original "
                        "random.Random(0).sample(range(7405), 10) implementation; "
                        "no unambiguously labelled result blob survives"
                    ),
                    "id_count": len(dev_seed0_ids),
                    "ordered_ids_sha256": _sha256_lines(sorted(dev_seed0_ids)),
                },
            ],
        },
        "question_ids_sha256": _sha256_lines(final_ids),
        "question_ids": final_ids,
        "generator": "scripts/freeze_final_sample.py",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_auxiliary_manifests(manifest)
    print(f"wrote {OUTPUT} with {len(final_ids)} IDs")
    print(f"final IDs sha256: {manifest['question_ids_sha256']}")
    print(f"excluded unique IDs: {len(excluded)}")


if __name__ == "__main__":
    main()
