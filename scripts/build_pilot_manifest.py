"""Deterministically freeze the exclusion-only n=200 pilot cohort."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "config" / "manifests"
FINAL = MANIFESTS / "final_n1500_seed20260805.json"
PREFLIGHT = MANIFESTS / "preflight_excluded32.json"
TIMING = MANIFESTS / "timing_excluded128_seed20260805.json"
OUTPUT = MANIFESTS / "pilot_excluded200_seed20260806.json"
SEED = 20260806
N = 200
DATA_QUALITY_REPLACEMENTS = {
    # HotpotQA labels this five-sentence page with sent_id=902. Keeping it would
    # make every system fail the exact gold-sentence reachability contract for a
    # dataset annotation error unrelated to retrieval quality.
    "5ae61bfd5542992663a4f261": "5ae622495542995703ce8b20",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("utf-8")
    ).hexdigest()


def main() -> None:
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    timing = json.loads(TIMING.read_text(encoding="utf-8"))
    reserved = set(preflight["question_ids"]) | set(timing["question_ids"])
    eligible = [
        question_id for question_id in final["exclusions"]["question_ids"]
        if question_id not in reserved
    ]
    selected = random.Random(SEED).sample(eligible, N)
    for malformed_id, replacement_id in DATA_QUALITY_REPLACEMENTS.items():
        if malformed_id not in selected:
            raise RuntimeError(f"expected malformed pilot ID was not sampled: {malformed_id}")
        if replacement_id not in eligible or replacement_id in selected:
            raise RuntimeError(f"invalid deterministic pilot replacement: {replacement_id}")
        selected[selected.index(malformed_id)] = replacement_id
    payload = {
        "schema_version": 1,
        "purpose": (
            "fixed exclusion-only baseline-vs-single pilot; scored only for the "
            "pre-campaign stop/go gate and never included in final inference"
        ),
        "source": {
            "final_manifest_path": "config/manifests/final_n1500_seed20260805.json",
            "final_manifest_file_sha256": file_sha256(FINAL),
            "final_question_ids_sha256": final["question_ids_sha256"],
            "exclusion_ids_sha256": final["exclusions"]["ordered_ids_sha256"],
            "preflight_question_ids_sha256": preflight["question_ids_sha256"],
            "timing_question_ids_sha256": timing["question_ids_sha256"],
        },
        "selection": {
            "n": N,
            "seed": SEED,
            "algorithm": (
                "CPython random.Random(seed).sample over ordered final exclusions "
                "after removing preflight and timing IDs; sampled order retained"
            ),
            "eligible_count": len(eligible),
        },
        "expected_retrieval_strata_counts": {
            "hidden_bridge": 160,
            "fully_named": 40,
        },
        "expected_hotpot_type_counts": {"bridge": 168, "comparison": 32},
        "data_quality_replacements": {
            malformed: {
                "replacement_id": replacement,
                "reason": "supporting sent_id 902 is outside a five-sentence gold page",
                "policy": "next ordered eligible nonsampled exclusion with valid labels",
            }
            for malformed, replacement in DATA_QUALITY_REPLACEMENTS.items()
        },
        "question_ids_sha256": ids_sha256(selected),
        "question_ids": selected,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"file_sha256={file_sha256(OUTPUT)}")
    print(f"ids_sha256={payload['question_ids_sha256']}")


if __name__ == "__main__":
    main()
