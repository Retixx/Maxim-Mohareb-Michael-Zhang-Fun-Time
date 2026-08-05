from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from scripts.run_campaign import STATIC_RUNS, TINY_RUNS, build_plan


ROOT = Path(__file__).resolve().parents[1]


class CampaignPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8")
        )

    def test_accuracy_plan_assigns_every_static_arm_once(self) -> None:
        plan = build_plan(self.config, kind="accuracy", workers=7, seed=42)
        assigned = [run_id for values in plan["assignments"].values() for run_id in values]
        self.assertEqual(len(assigned), 22)
        self.assertEqual(set(assigned), STATIC_RUNS)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_plan_is_worker_count_independent_in_global_order(self) -> None:
        one = build_plan(self.config, kind="accuracy", workers=1, seed=7)
        many = build_plan(self.config, kind="accuracy", workers=5, seed=7)
        self.assertEqual(one["ordered_run_ids"], many["ordered_run_ids"])

    def test_timing_is_one_worker_and_excludes_tiny(self) -> None:
        plan = build_plan(self.config, kind="timing", workers=1, seed=99)
        self.assertTrue(set(plan["ordered_run_ids"]).isdisjoint(TINY_RUNS))
        with self.assertRaises(ValueError):
            build_plan(self.config, kind="timing", workers=2, seed=99)


if __name__ == "__main__":
    unittest.main()
