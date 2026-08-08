"""Fail-closed tests for the frozen experiment matrix and data cohorts."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "experiment.yaml"
FINAL = ROOT / "config" / "manifests" / "final_n1500_seed20260805.json"
PREFLIGHT = ROOT / "config" / "manifests" / "preflight_excluded32.json"
TIMING = ROOT / "config" / "manifests" / "timing_excluded128_seed20260805.json"
PILOT = ROOT / "config" / "manifests" / "pilot_excluded200_seed20260806.json"

TINY_RUNS = {
    "planner_tiny",
    "stepdef_tiny",
    "extractor_tiny",
    "qa_tiny",
    "ma_uniform_tiny",
}
ROLES = ("planner", "stepdef", "extractor", "qa")
STATIC_TIERS = ("8bit", "4bit", "mid", "small", "tiny", "large")
STATIC_RUNS = {
    "baseline",
    "single_fp16",
    *(f"{role}_{tier}" for tier in STATIC_TIERS for role in ROLES),
    *(f"ma_uniform_{tier}" for tier in STATIC_TIERS),
}
SECTION_16_ADDITIONS = {
    *(f"{role}_14b_4bit" for role in ROLES),
    "ma_uniform_14b_4bit",
    "single_8bit",
    "single_4bit",
    "single_mid",
    "single_small",
    "single_tiny",
    "single_large",
    "single_14b_4bit",
}


def _line_hash(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.final = json.loads(FINAL.read_text(encoding="utf-8"))
        cls.preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        cls.timing = json.loads(TIMING.read_text(encoding="utf-8"))
        cls.pilot = json.loads(PILOT.read_text(encoding="utf-8"))

    def test_static_matrix_and_selector_are_closed(self) -> None:
        runs = self.config["runs"]
        self.assertEqual(len(runs), 32)
        self.assertEqual(set(runs), STATIC_RUNS)
        self.assertEqual(set(runs) & SECTION_16_ADDITIONS, set())
        self.assertEqual(set(runs) & TINY_RUNS, TINY_RUNS)
        self.assertEqual(runs["single_fp16"], {"solo": "fp16"})

        selector = self.config["allocation_selector"]
        self.assertEqual(
            set(selector["candidates"]),
            {
                "large_fp16",
                "base_fp16",
                "base_8bit",
                "base_4bit",
                "mid_fp16",
                "small_fp16",
                "tiny_fp16",
            },
        )
        self.assertEqual(selector["candidate_allocation_count"], 7**4)
        self.assertEqual(
            selector["tiny_eligibility_gate"]["ineligible_action"],
            "remove_tiny_for_that_role_only",
        )

    def test_frozen_cohorts_are_unique_disjoint_and_hashed(self) -> None:
        final_ids = self.final["question_ids"]
        excluded_ids = self.final["exclusions"]["question_ids"]
        preflight_ids = self.preflight["question_ids"]
        timing_ids = self.timing["question_ids"]
        pilot_ids = self.pilot["question_ids"]

        for values, expected_n in (
            (final_ids, 1500),
            (excluded_ids, 3031),
            (preflight_ids, 32),
            (timing_ids, 128),
            (pilot_ids, 200),
        ):
            self.assertEqual(len(values), expected_n)
            self.assertEqual(len(set(values)), expected_n)

        self.assertFalse(set(final_ids) & set(excluded_ids))
        self.assertFalse(set(preflight_ids) & set(timing_ids))
        self.assertFalse(set(pilot_ids) & set(preflight_ids))
        self.assertFalse(set(pilot_ids) & set(timing_ids))
        self.assertFalse(set(pilot_ids) & set(final_ids))
        self.assertLessEqual(set(pilot_ids), set(excluded_ids))
        self.assertLessEqual(set(preflight_ids) | set(timing_ids), set(excluded_ids))
        self.assertEqual(_line_hash(final_ids), self.final["question_ids_sha256"])
        self.assertEqual(
            _line_hash(excluded_ids), self.final["exclusions"]["ordered_ids_sha256"]
        )
        self.assertEqual(_line_hash(preflight_ids), self.preflight["question_ids_sha256"])
        self.assertEqual(_line_hash(timing_ids), self.timing["question_ids_sha256"])
        self.assertEqual(_line_hash(pilot_ids), self.pilot["question_ids_sha256"])
        for auxiliary in (self.preflight, self.timing, self.pilot):
            source = auxiliary["source"]
            self.assertEqual(source["final_manifest_file_sha256"], _file_hash(FINAL))
            self.assertEqual(
                source["final_question_ids_sha256"], self.final["question_ids_sha256"]
            )
            self.assertEqual(
                source["exclusion_ids_sha256"],
                self.final["exclusions"]["ordered_ids_sha256"],
            )

    def test_config_binds_every_manifest_byte_and_id_hash(self) -> None:
        dataset = self.config["dataset"]
        timing = self.config["timing"]
        self.assertEqual(dataset["manifest_file_sha256"], _file_hash(FINAL))
        self.assertEqual(dataset["manifest_sha256"], self.final["question_ids_sha256"])
        self.assertEqual(
            dataset["exclusion_sha256"], self.final["exclusions"]["ordered_ids_sha256"]
        )
        self.assertEqual(dataset["preflight_manifest_file_sha256"], _file_hash(PREFLIGHT))
        self.assertEqual(
            dataset["preflight_manifest_sha256"], self.preflight["question_ids_sha256"]
        )
        self.assertEqual(timing["manifest_file_sha256"], _file_hash(TIMING))
        self.assertEqual(timing["manifest_sha256"], self.timing["question_ids_sha256"])
        pilot = self.config["pilot"]
        self.assertEqual(pilot["manifest_file_sha256"], _file_hash(PILOT))
        self.assertEqual(pilot["manifest_sha256"], self.pilot["question_ids_sha256"])
        self.assertEqual(pilot["retrieval_strata_counts"], {"hidden_bridge": 160, "fully_named": 40})

    def test_timing_is_excluded_data_and_never_uses_tiny_runs(self) -> None:
        timing = self.config["timing"]
        self.assertEqual(timing["n"], 128)
        self.assertGreaterEqual(timing["repetitions"], 2)
        self.assertTrue(set(timing["run_ids"]).isdisjoint(TINY_RUNS))
        self.assertEqual(set(timing["excluded_runs"]), TINY_RUNS)
        self.assertNotIn("ma_optimized_exploratory", timing["run_ids"])
        self.assertEqual(timing["post_selection_run_id"], "ma_optimized_exploratory")


if __name__ == "__main__":
    unittest.main()
