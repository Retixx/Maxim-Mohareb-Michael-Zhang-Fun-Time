import math
import unittest

from src.metrics import (
    bootstrap_ci,
    clustered_ratio_bootstrap,
    exact_mcnemar,
    holm_adjust,
    joint_paired_bootstrap,
)


class MetricsTests(unittest.TestCase):
    def test_exact_mcnemar_counts_discordance_and_computes_two_sided_p(self):
        result = exact_mcnemar(
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
        )

        self.assertEqual(result["n_pairs"], 7)
        self.assertEqual(result["a_only_wins"], 4)
        self.assertEqual(result["b_only_wins"], 1)
        self.assertEqual(result["n_discordant"], 5)
        self.assertAlmostEqual(result["p_value"], 0.375)

    def test_exact_mcnemar_is_symmetric_and_handles_no_discordance(self):
        forward = exact_mcnemar([1, 1, 1, 0], [0, 0, 1, 1])
        reverse = exact_mcnemar([0, 0, 1, 1], [1, 1, 1, 0])

        self.assertEqual(forward["p_value"], reverse["p_value"])
        self.assertEqual(forward["a_only_wins"], reverse["b_only_wins"])
        self.assertEqual(forward["b_only_wins"], reverse["a_only_wins"])
        self.assertEqual(exact_mcnemar([0, 1], [0, 1])["p_value"], 1.0)

    def test_exact_mcnemar_rejects_invalid_pairs(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            exact_mcnemar([0], [0, 1])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            exact_mcnemar([], [])

        invalid_cases = [
            ([0, 2], [0, 1], "outcomes_a\\[1\\]"),
            ([0, 1], [0, -1], "outcomes_b\\[1\\]"),
            ([0, "1"], [0, 1], "outcomes_a\\[1\\]"),
        ]
        for outcomes_a, outcomes_b, message in invalid_cases:
            with self.subTest(outcomes_a=outcomes_a, outcomes_b=outcomes_b):
                with self.assertRaisesRegex(ValueError, message):
                    exact_mcnemar(outcomes_a, outcomes_b)

    def test_bootstrap_ci_is_order_invariant(self):
        forward = bootstrap_ci([0.0, 1.0, 3.0, 8.0], n_resamples=500, seed=9)
        reverse = bootstrap_ci([8.0, 3.0, 1.0, 0.0], n_resamples=500, seed=9)
        self.assertEqual(forward, reverse)

    def test_joint_bootstrap_uses_identical_draws_and_requires_same_ids(self):
        result = joint_paired_bootstrap(
            {
                "a": {"q2": 2.0, "q1": 1.0, "q3": 3.0},
                "b": {"q3": 6.0, "q2": 4.0, "q1": 2.0},
            },
            n_resamples=500,
            seed=4,
        )
        self.assertAlmostEqual(result["a"]["estimate"], 2.0)
        self.assertAlmostEqual(result["b"]["estimate"], 4.0)
        # b is exactly 2*a, including under every shared bootstrap resample.
        self.assertAlmostEqual(
            result["b"]["ci_lower"], 2 * result["a"]["ci_lower"]
        )
        self.assertAlmostEqual(
            result["b"]["ci_upper"], 2 * result["a"]["ci_upper"]
        )

        with self.assertRaisesRegex(ValueError, "different question set"):
            joint_paired_bootstrap(
                {"a": {"q1": 1.0}, "b": {"q2": 1.0}},
                n_resamples=20,
            )

    def test_clustered_ratio_keeps_calls_from_a_question_together(self):
        # q1 contributes 100 failed calls and q2 contributes one successful call.
        # The point estimand is still per-call, but n_clusters is two (not 101).
        result = clustered_ratio_bootstrap(
            {"q1": (100, 100), "q2": (0, 1)},
            n_resamples=500,
            seed=0,
        )
        self.assertAlmostEqual(result["estimate"], 100 / 101)
        self.assertEqual(result["n_clusters"], 2)
        self.assertLessEqual(result["ci_lower"], result["estimate"])
        self.assertLessEqual(result["estimate"], result["ci_upper"])

    def test_holm_step_down_is_monotonic_in_rank(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04, "d": 0.5})
        expected = {"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.5}
        for key, value in expected.items():
            self.assertAlmostEqual(adjusted[key], value)
        self.assertTrue(all(math.isfinite(value) for value in adjusted.values()))


if __name__ == "__main__":
    unittest.main()

