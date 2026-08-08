import unittest
import json
import tempfile
import hashlib
from pathlib import Path
from unittest import mock

from analyze import (
    AnalysisError,
    EXPECTED_FINAL_MANIFEST_SHA256,
    RunData,
    SELECTOR_CONFIGS,
    STATIC_RUN_IDS,
    analyze_call_counts,
    analyze_evidence,
    analyze_latency_batches,
    analyze_optimized_system,
    analyze_role_costs,
    analyze_role_families,
    analyze_selection_churn,
    analyze_solo,
    _complete_matching_accuracy_run,
    _validate_analysis_environment_lock,
    content_hash,
    build_report,
    discover_runs,
    load_run,
    paired_differences,
    pareto_frontier,
    near_memory_match,
    role_run,
    select_allocation,
    verify_selection_artifact,
    validate_campaign_completeness,
)


ROLES = ("planner", "step_definer", "extractor", "qa")
CONFIGS = (
    "large_fp16",
    "base_fp16",
    "base_8bit",
    "base_4bit",
    "mid_fp16",
    "small_fp16",
    "tiny_fp16",
)
FOOTPRINTS = {
    "large_fp16": 200.0,
    "base_fp16": 100.0,
    "base_8bit": 60.0,
    "base_4bit": 40.0,
    "mid_fp16": 55.0,
    "small_fp16": 50.0,
    "tiny_fp16": 10.0,
}
TIMING_HASH = "1" * 64


def make_run(
    run_id,
    scores=None,
    *,
    calls=(),
    batches=(),
    timing_meta=None,
    timing_n_questions=None,
    timing_repetitions=None,
):
    scores = scores or {"q1": (1.0, 1.0), "q2": (0.0, 0.0)}
    answers = {
        question_id: {
            "question_id": question_id,
            "record_type": "answer",
            "f1": f1,
            "em": em,
        }
        for question_id, (f1, em) in scores.items()
    }
    return RunData(
        run_id=run_id,
        answers=answers,
        calls=tuple(calls),
        batches=tuple(batches),
        meta={"run_id": run_id, "n": len(answers)},
        jsonl_path=Path(f"{run_id}.jsonl"),
        meta_path=Path(f"{run_id}.meta.json"),
        timing_meta=timing_meta,
        timing_n_questions=timing_n_questions,
        timing_repetitions=timing_repetitions,
    )


class AnalyzeTests(unittest.TestCase):
    def test_selector_reuses_complete_static_accuracy_without_timing(self):
        definition = {"planner": {"model": "tiny", "precision": "fp16"}}
        config = {"runs": {"planner_tiny": definition}}
        complete = make_run("planner_tiny", timing_meta=None)
        self.assertEqual(
            _complete_matching_accuracy_run(config, {"planner_tiny": complete}, definition),
            "planner_tiny",
        )
        incomplete = RunData(
            **{**complete.__dict__, "answers": {}},
        )
        self.assertIsNone(
            _complete_matching_accuracy_run(
                config, {"planner_tiny": incomplete}, definition
            )
        )

    def test_analysis_binds_results_to_present_lock_and_current_source(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "environment.lock.json"
            lock_path.write_text(
                json.dumps({"source_bundle_sha256": "source-bundle"}) + "\n",
                encoding="utf-8",
            )
            config = {"environment_lock_path": str(lock_path)}
            with mock.patch(
                "src.runner.source_bundle_sha256", return_value="source-bundle"
            ):
                self.assertEqual(
                    _validate_analysis_environment_lock(config),
                    hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                )
            with mock.patch(
                "src.runner.source_bundle_sha256", return_value="changed-source"
            ):
                with self.assertRaisesRegex(AnalysisError, "source bundle differs"):
                    _validate_analysis_environment_lock(config)

    def test_final_completeness_gate_requires_full_accuracy_and_timing_matrix(self):
        self.assertEqual(len(STATIC_RUN_IDS), 32)
        config = {
            "runs": {run_id: {} for run_id in STATIC_RUN_IDS},
            "timing": {"run_ids": ["baseline"]},
            "allocation_selector": {"output_artifacts": {}},
        }
        runs = {run_id: make_run(run_id) for run_id in STATIC_RUN_IDS}
        with self.assertRaisesRegex(AnalysisError, "timing campaign"):
            validate_campaign_completeness(config, runs)
        baseline = runs["baseline"]
        runs["baseline"] = RunData(
            **{
                **baseline.__dict__,
                "timing_meta": {"timing_mode": True},
            }
        )
        validate_campaign_completeness(config, runs)
        runs.pop("qa_tiny")
        with self.assertRaisesRegex(AnalysisError, "accuracy campaign"):
            validate_campaign_completeness(config, runs)

    def test_base_completeness_ignores_future_run_from_existing_selector_trace(self):
        config = {
            "runs": {run_id: {} for run_id in STATIC_RUN_IDS},
            "timing": {"run_ids": ["baseline"]},
            "allocation_selector": {"output_artifacts": {}},
        }
        runs = {run_id: make_run(run_id) for run_id in STATIC_RUN_IDS}
        baseline = runs["baseline"]
        runs["baseline"] = RunData(
            **{**baseline.__dict__, "timing_meta": {"timing_mode": True}},
        )
        trace = {
            "source_config_sha256": content_hash(config),
            "executable_config_sha256": "future-derived-config",
            "execution_run_id": "ma_optimized_exploratory",
            "materialized_new_run": True,
        }
        with mock.patch("analyze._existing_selection_artifact", return_value=trace):
            validate_campaign_completeness(config, runs)

        derived = json.loads(json.dumps(config))
        derived["runs"]["ma_optimized_exploratory"] = {"qa": "4bit"}
        derived["frozen_allocation"] = {
            "execution_run_id": "ma_optimized_exploratory"
        }
        trace["executable_config_sha256"] = content_hash(derived)
        with mock.patch("analyze._existing_selection_artifact", return_value=trace):
            with self.assertRaisesRegex(AnalysisError, "accuracy campaign"):
                validate_campaign_completeness(derived, runs)

    def test_derived_static_selection_requires_its_missing_timing(self):
        config = {
            "runs": {run_id: {} for run_id in STATIC_RUN_IDS},
            "timing": {"run_ids": ["baseline"]},
            "allocation_selector": {"output_artifacts": {}},
            "frozen_allocation": {"execution_run_id": "planner_tiny"},
        }
        runs = {run_id: make_run(run_id) for run_id in STATIC_RUN_IDS}
        baseline = runs["baseline"]
        runs["baseline"] = RunData(
            **{**baseline.__dict__, "timing_meta": {"timing_mode": True}},
        )
        trace = {
            "source_config_sha256": "base-config",
            "executable_config_sha256": content_hash(config),
            "execution_run_id": "planner_tiny",
            "materialized_new_run": False,
        }
        with mock.patch("analyze._existing_selection_artifact", return_value=trace):
            with self.assertRaisesRegex(AnalysisError, "timing campaign"):
                validate_campaign_completeness(config, runs)

    def test_discovery_merges_excluded_timing_without_using_timing_for_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accuracy_meta = root / "baseline_accuracy.meta.json"
            timing_meta = root / "baseline_timing.meta.json"
            accuracy_meta.write_text(json.dumps({
                "run_id": "baseline", "n": 2, "seed": 7,
                "question_ids_sha256": EXPECTED_FINAL_MANIFEST_SHA256,
            }), encoding="utf-8")
            timing_meta.write_text(json.dumps({
                "run_id": "baseline", "n": 2, "seed": 7,
                "timing_mode": True, "question_ids_sha256": TIMING_HASH,
                "timing_repetitions": 2,
            }), encoding="utf-8")
            accuracy_meta.with_name("baseline_accuracy.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in (
                    {"record_type": "answer", "run_id": "baseline",
                     "question_id": "q1", "f1": 1.0, "em": 1.0},
                    {"record_type": "answer", "run_id": "baseline",
                     "question_id": "q2", "f1": 0.0, "em": 0.0},
                )), encoding="utf-8",
            )
            timing_meta.with_name("baseline_timing.jsonl").write_text(
                json.dumps({
                    "record_type": "agent_call", "run_id": "baseline",
                    "question_id": "excluded-q1", "stage": "qa", "call_index": 0,
                }) + "\n",
                encoding="utf-8",
            )
            runs = discover_runs(
                root, n=2, seed=7, timing_manifest_sha256=TIMING_HASH,
                timing_n=2, timing_seed=7, timing_repetitions=2,
            )
            self.assertEqual(set(runs["baseline"].answers), {"q1", "q2"})
            self.assertEqual(runs["baseline"].timing_calls[0]["question_id"], "excluded-q1")
            self.assertEqual(runs["baseline"].timing_n_questions, 4)

    def test_paired_answer_comparison_refuses_silent_intersection(self):
        a = make_run("a", {"q1": (1.0, 1.0), "q2": (0.0, 0.0)})
        b = make_run("b", {"q1": (1.0, 1.0)})
        with self.assertRaisesRegex(AnalysisError, "not paired"):
            paired_differences(a, b, "f1")

    def test_selector_enumerates_seven_configs_and_fixed_charges(self):
        self.assertEqual(SELECTOR_CONFIGS, CONFIGS)
        effects = {
            role: {
                "large_fp16": -2.0,
                "base_fp16": 0.0,
                "base_8bit": -0.2,
                "base_4bit": -0.3,
                "mid_fp16": -2.0,
                "small_fp16": -0.4,
                "tiny_fp16": -2.0,
            }
            for role in ROLES
        }
        result = select_allocation(
            baseline_f1_points=80.0,
            role_effects_points=effects,
            role_effect_draws_points={
                role: {configuration: [effect] * 100
                       for configuration, effect in effects[role].items()}
                for role in ROLES
            },
            footprints_mib=FOOTPRINTS,
            margin_points=1.0,
        )
        self.assertEqual(result["candidate_count"], 7**4)
        self.assertEqual(
            result["criterion"]["bootstrap_shared_across_all_24_nonzero_role_effects"],
            True,
        )
        # all@4bit and all@small violate the margin. all@8bit is feasible and
        # costs one 60-MiB fixed charge, less than feasible mixed configs.
        self.assertEqual(
            result["selected"]["allocation"], {role: "base_8bit" for role in ROLES}
        )
        self.assertEqual(result["selected"]["deduplicated_memory_mib"], 60.0)

        with self.assertRaisesRegex(AnalysisError, "exactly"):
            select_allocation(
                baseline_f1_points=80.0,
                role_effects_points=effects,
                role_effect_draws_points={
                    role: {configuration: [effect] * 100
                           for configuration, effect in effects[role].items()}
                    for role in ROLES
                },
                footprints_mib={key: value for key, value in FOOTPRINTS.items()
                                if key != "large_fp16"},
            )

    def test_selector_allows_uniform_no_change(self):
        effects = {
            role: {
                configuration: (-1.0 if configuration != "base_fp16" else 0.0)
                for configuration in CONFIGS
            }
            for role in ROLES
        }
        result = select_allocation(
            baseline_f1_points=80.0,
            role_effects_points=effects,
            role_effect_draws_points={
                role: {configuration: [effect] * 100
                       for configuration, effect in effects[role].items()}
                for role in ROLES
            },
            footprints_mib=FOOTPRINTS,
            margin_points=0.5,
        )
        self.assertEqual(
            result["selected"]["allocation"], {role: "base_fp16" for role in ROLES}
        )

    def test_selector_uses_ni_lower_bound_not_only_point_estimate(self):
        effects = {
            role: {
                "large_fp16": -2.0,
                "base_fp16": 0.0,
                "base_8bit": -0.2,
                "base_4bit": -2.0,
                "mid_fp16": -2.0,
                "small_fp16": -2.0,
                "tiny_fp16": -2.0,
            }
            for role in ROLES
        }
        # all@8bit has a point prediction of -0.8 (inside the -1 margin), but
        # its lower bootstrap bound is -4.0 and therefore cannot pass NI.
        draws = {
            role: {
                "large_fp16": [-2.0] * 100,
                "base_fp16": [0.0] * 100,
                "base_8bit": [-1.0] * 50 + [0.6] * 50,
                "base_4bit": [-2.0] * 100,
                "mid_fp16": [-2.0] * 100,
                "small_fp16": [-2.0] * 100,
                "tiny_fp16": [-2.0] * 100,
            }
            for role in ROLES
        }
        result = select_allocation(
            baseline_f1_points=80.0,
            role_effects_points=effects,
            role_effect_draws_points=draws,
            footprints_mib=FOOTPRINTS,
            margin_points=1.0,
        )
        self.assertNotEqual(
            result["selected"]["allocation"], {role: "base_8bit" for role in ROLES}
        )
        self.assertGreaterEqual(
            result["selected"]["predicted_effect_ci_lower"], -1.0
        )

    def test_selector_applies_tiny_eligibility_per_role(self):
        effects = {
            role: {configuration: 0.0 for configuration in CONFIGS}
            for role in ROLES
        }
        result = select_allocation(
            baseline_f1_points=80.0,
            role_effects_points=effects,
            role_effect_draws_points={
                role: {configuration: [effect] * 100
                       for configuration, effect in effects[role].items()}
                for role in ROLES
            },
            footprints_mib=FOOTPRINTS,
            eligible_tiny_roles={"qa"},
        )
        self.assertEqual(result["eligible_tiny_roles"], ["qa"])
        self.assertEqual(result["candidate_count"], 2401)
        self.assertEqual(result["eligible_candidate_count"], 6**3 * 7)
        self.assertEqual(result["eligibility_filtered_count"], 2401 - 6**3 * 7)
        self.assertTrue(all(
            configuration != "tiny_fp16"
            for role, configuration in result["selected"]["allocation"].items()
            if role != "qa"
        ))

    def test_role_families_use_mid_not_small_as_near_memory_peer(self):
        runs = {
            role_run(role, suffix): make_run(role_run(role, suffix))
            for role in ROLES
            for suffix in ("8bit", "4bit", "mid")
        }
        families = analyze_role_families(runs, n_resamples=100, seed=0)
        self.assertEqual(
            families["primary"]["family"], "primary_8b8_vs_4b_fp16"
        )
        self.assertEqual(
            families["primary"]["contrast_direction"], "8bit_minus_mid"
        )
        self.assertEqual(
            families["secondary_4bit"]["family"], "secondary_8b4_vs_4b_fp16"
        )
        self.assertEqual(
            families["secondary_4bit"]["contrast_direction"], "4bit_minus_mid"
        )

    def test_report_role_family_readiness_requires_mid_arms(self):
        runs = {
            role_run(role, suffix): make_run(role_run(role, suffix))
            for role in ROLES
            for suffix in ("8bit", "4bit", "mid")
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "analyze.validate_final_cohort"
        ), mock.patch("analyze.near_memory_match", return_value={"status": "tested"}):
            report = build_report(
                {},
                runs,
                Path(directory),
                n_resamples=100,
                freeze_selection=False,
                require_complete=False,
            )
        self.assertIn("role_analysis", report)
        self.assertNotIn("role_analysis_status", report)

    def test_role_costs_cover_all_six_treatment_axes(self):
        runs = {"baseline": make_run("baseline")}
        for role in ROLES:
            for suffix in ("8bit", "4bit", "mid", "small", "tiny", "large"):
                run_id = role_run(role, suffix)
                runs[run_id] = make_run(run_id)
        costs = analyze_role_costs(runs, n_resamples=100, seed=0)
        self.assertEqual(set(costs), {
            "8b_8bit", "8b_4bit", "4b_fp16", "1p7b_fp16", "0p6b_fp16",
            "14b_fp16",
        })
        for family in costs.values():
            self.assertEqual(family["contrast_direction"], "baseline_minus_treatment")
            self.assertEqual(set(family["metrics"]["f1"]), set(ROLES))

    def test_system_contrast_directions_name_the_8b_reference(self):
        baseline = make_run("baseline")
        solo = analyze_solo(
            {"baseline": baseline, "single_fp16": make_run("single_fp16")},
            n_resamples=100,
            seed=0,
        )
        self.assertEqual(
            solo["contrast_direction"],
            "single_8b_fp16_minus_uniform_multiagent_8b_fp16",
        )
        optimized = analyze_optimized_system(
            {"baseline": baseline, "ma_optimized_exploratory": make_run(
                "ma_optimized_exploratory"
            )},
            n_resamples=100,
            seed=0,
        )
        self.assertEqual(
            optimized["contrast_direction"],
            "optimized_minus_uniform_multiagent_8b_fp16",
        )

    def test_near_memory_match_uses_mid_runs_and_current_memory_keys(self):
        runs = {}
        for role in ROLES:
            for suffix, treated_mib, system_mib in (
                ("8bit", 8500.0, 24500.0),
                ("mid", 8000.0, 24000.0),
            ):
                run_id = role_run(role, suffix)
                run = make_run(run_id)
                runs[run_id] = RunData(**{
                    **run.__dict__,
                    "meta": {
                        **run.meta,
                        "deduplicated_concurrent_model_footprint_mib": system_mib,
                        "stages": {role: {"model_footprint_mib": treated_mib}},
                    },
                })
        result = near_memory_match(runs)
        expected_keys = {
            "8b_8bit_mib",
            "4b_fp16_mib",
            "gap_mib_8b8_minus_4b_fp16",
            "gap_percent_of_4b_fp16",
            "lower_memory_arm",
        }
        for comparison in result["per_role"].values():
            self.assertEqual(set(comparison["treated_model"]), expected_keys)
            self.assertEqual(
                set(comparison["one_role_full_system_deduplicated_concurrent"]),
                expected_keys,
            )
            self.assertEqual(comparison["treated_model"]["lower_memory_arm"], "4b_fp16")

    def test_pareto_is_measured_nondominance_without_interpolation(self):
        points = [
            {"run_id": "low", "memory_mib": 10, "f1": 70},
            {"run_id": "dominated", "memory_mib": 20, "f1": 69},
            {"run_id": "high", "memory_mib": 30, "f1": 80},
        ]
        self.assertEqual(
            [point["run_id"] for point in pareto_frontier(points)], ["low", "high"]
        )

    def test_latency_ignores_call_latency_and_uses_batches(self):
        call_only = make_run(
            "baseline",
            calls=({"question_id": "q1", "stage": "qa", "latency_s": 999.0},),
        )
        unavailable = analyze_latency_batches(
            {"baseline": call_only}, n_resamples=100, seed=0
        )
        self.assertEqual(unavailable["status"], "unavailable_no_batch_records")
        self.assertFalse(unavailable["call_latency_fields_used"])

        batched = make_run(
            "baseline",
            calls=(
                {
                    "record_type": "agent_call",
                    "question_id": "q1",
                    "stage": "qa",
                    "call_index": 0,
                    "timing_eligible": True,
                    "execution_session_id": "session-1",
                    "question_manifest_sha256": TIMING_HASH,
                    "config_fingerprint": "fingerprint-1",
                },
                {
                    "record_type": "agent_call",
                    "question_id": "q2",
                    "stage": "qa",
                    "call_index": 0,
                    "timing_eligible": True,
                    "execution_session_id": "session-1",
                    "question_manifest_sha256": TIMING_HASH,
                    "config_fingerprint": "fingerprint-1",
                },
            ),
            batches=(
                {
                    "record_type": "batch",
                    "stage": "qa",
                    "batch_id": "qa:000000:timing:r0",
                    "batch_wall_s": 2.0,
                    "n_calls": 1,
                    "latency_eligible": True,
                    "phase": "timing_benchmark",
                    "execution_session_id": "session-1",
                    "timing_repeat": 0,
                    "members": [{"question_id": "q1", "call_index": 0}],
                    "question_manifest_sha256": TIMING_HASH,
                    "config_fingerprint": "fingerprint-1",
                },
                {
                    "record_type": "batch",
                    "stage": "qa",
                    "batch_id": "qa:000001:timing:r0",
                    "batch_wall_s": 3.0,
                    "n_calls": 1,
                    "latency_eligible": True,
                    "phase": "timing_benchmark",
                    "execution_session_id": "session-1",
                    "timing_repeat": 0,
                    "members": [{"question_id": "q2", "call_index": 0}],
                    "question_manifest_sha256": TIMING_HASH,
                    "config_fingerprint": "fingerprint-1",
                },
                {
                    "record_type": "batch",
                    "stage": "qa",
                    "batch_id": "qa:000000:timing:r1",
                    "batch_wall_s": 2.0,
                    "n_calls": 1,
                    "timing_eligible": True,
                    "phase": "timing_benchmark",
                    "execution_session_id": "session-1",
                    "timing_repeat": 1,
                    "members": [{"question_id": "q1", "call_index": 0}],
                    "question_manifest_sha256": TIMING_HASH,
                    "config_fingerprint": "fingerprint-1",
                },
                {
                    "record_type": "batch",
                    "stage": "qa",
                    "batch_id": "qa:000001:timing:r1",
                    "batch_wall_s": 3.0,
                    "n_calls": 1,
                    "timing_eligible": True,
                    "phase": "timing_benchmark",
                    "execution_session_id": "session-1",
                    "timing_repeat": 1,
                    "members": [{"question_id": "q2", "call_index": 0}],
                    "question_manifest_sha256": TIMING_HASH,
                    "config_fingerprint": "fingerprint-1",
                },
            ),
            timing_meta={
                "n": 2,
                "question_ids_sha256": TIMING_HASH,
                "timing_repetitions": 2,
                "stage_config_fingerprints": {"qa": "fingerprint-1"},
                "stages": {},
            },
            timing_n_questions=4,
            timing_repetitions=2,
        )
        batched.meta["stage_config_fingerprints"] = {"qa": "fingerprint-1"}
        available = analyze_latency_batches(
            {"baseline": batched}, n_resamples=100, seed=0
        )
        self.assertEqual(available["status"], "available")
        self.assertAlmostEqual(
            available["runs"]["baseline"]["stages"]["qa"]["estimate"], 2.5
        )
        self.assertEqual(
            available["runs"]["baseline"]["stages"]["qa"]["ratio_to_uniform_ma_8b_fp16"],
            1.0,
        )

        service_batches = tuple(
            {**record, "service_wall_s": record["batch_wall_s"] + 1.0}
            for record in batched.batches
        )
        experiment_fingerprint = "experiment-v1"
        orchestration = tuple({
            "record_type": "stage_timing",
            "stage": "qa",
            "timing_repeat": repeat,
            "orchestration_wall_s": 1.0,
            "timing_eligible": True,
            "phase": "timing_benchmark",
            "question_manifest_sha256": TIMING_HASH,
            "experiment_fingerprint": experiment_fingerprint,
            "config_fingerprint": "fingerprint-1",
            "call_graph_sha256": "same-graph",
        } for repeat in range(2))
        current = RunData(**{
            **batched.__dict__,
            "batches": service_batches,
            "timing_orchestration": orchestration,
            "timing_meta": {
                **batched.timing_meta,
                "experiment_fingerprint": experiment_fingerprint,
                "experiment_fingerprint_payload": {"schema": "open_corpus_marag_v3"},
            },
        })
        service = analyze_latency_batches(
            {"baseline": current}, n_resamples=100, seed=0
        )["runs"]["baseline"]["system"]
        self.assertAlmostEqual(service["estimate_seconds_per_question"], 4.0)
        self.assertAlmostEqual(
            service["generation_only"]["estimate_seconds_per_question"], 2.5
        )
        self.assertAlmostEqual(service["orchestration_seconds_per_question"], 0.5)

    def test_loader_excludes_warmup_and_model_load_from_agent_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta_path = root / "sample.meta.json"
            jsonl_path = root / "sample.jsonl"
            meta_path.write_text(
                json.dumps({"run_id": "baseline", "n": 1}), encoding="utf-8"
            )
            records = [
                {"record_type": "warmup_call", "run_id": "baseline", "stage": "qa"},
                {"record_type": "model_load", "run_id": "baseline", "stage": "qa"},
                {"record_type": "agent_call", "run_id": "baseline", "stage": "qa",
                 "question_id": "q1", "parse_status": "ok"},
                {"record_type": "answer", "run_id": "baseline", "question_id": "q1",
                 "f1": 1.0, "em": 1.0},
            ]
            jsonl_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            run = load_run(meta_path)
            self.assertEqual(len(run.calls), 1)
            self.assertEqual(run.calls[0]["record_type"], "agent_call")

    def test_evidence_uses_precomputed_consumed_span_scores_and_skips_solo(self):
        multi = make_run("baseline")
        for answer in multi.answers.values():
            answer.update({
                "evidence_status": "applicable",
                "evidence_f1": answer["f1"],
                "evidence_em": answer["em"],
            })
        solo = make_run("single_fp16")
        for answer in solo.answers.values():
            answer.update({"evidence_status": "not_applicable"})
        result = analyze_evidence(
            {"baseline": multi, "single_fp16": solo}, n_resamples=100, seed=0
        )
        self.assertEqual(result["status"], "available")
        self.assertIn("baseline", result["main"])
        self.assertNotIn("single_fp16", result["main"])

    def test_churn_and_call_counts_are_question_clustered(self):
        baseline = make_run(
            "baseline",
            calls=(
                {"question_id": "q1", "stage": "planner", "call_index": 0,
                 "parsed": {"sub_questions": ["a"]}, "prompt_tokens": 10,
                 "output_tokens": 2},
                {"question_id": "q2", "stage": "planner", "call_index": 0,
                 "parsed": {"sub_questions": ["b"]}, "prompt_tokens": 20,
                 "output_tokens": 3},
            ),
        )
        treatment = make_run(
            "planner_8bit",
            calls=(
                {"question_id": "q1", "stage": "planner", "call_index": 0,
                 "parsed": {"sub_questions": ["changed"]}},
                {"question_id": "q2", "stage": "planner", "call_index": 0,
                 "parsed": {"sub_questions": ["b"]}},
            ),
        )
        churn = analyze_selection_churn(
            {"baseline": baseline, "planner_8bit": treatment},
            n_resamples=100,
            seed=0,
        )
        estimate = churn["main"]["planner_8bit"]["parser_success_payload_churn"]
        self.assertEqual(estimate["n_clusters"], 2)
        self.assertAlmostEqual(estimate["estimate"], 0.5)
        counts = analyze_call_counts({"baseline": baseline}, n_resamples=100, seed=0)
        self.assertAlmostEqual(counts["main"]["baseline"]["calls"]["estimate"], 1.0)
        self.assertAlmostEqual(
            counts["main"]["baseline"]["prompt_tokens"]["estimate"], 15.0
        )

    def test_repeated_ma_rag_stages_roll_up_without_call_key_collisions(self):
        calls = (
            {"question_id": "q1", "stage": "extractor", "conceptual_role": "extractor",
             "prompt_role": "extractor", "call_index": 0,
             "parsed": {"spans": ["first"]}},
            {"question_id": "q1", "stage": "extractor_step2",
             "conceptual_role": "extractor", "prompt_role": "extractor",
             "call_index": 0, "parsed": {"spans": ["second"]}},
        )
        changed = tuple(
            {**record, "parsed": {"spans": ["changed"]}}
            if record["stage"] == "extractor_step2" else record
            for record in calls
        )
        baseline = make_run("baseline", calls=calls)
        treatment = make_run("extractor_8bit", calls=changed)
        churn = analyze_selection_churn(
            {"baseline": baseline, "extractor_8bit": treatment},
            n_resamples=100,
            seed=0,
        )["main"]["extractor_8bit"]
        self.assertEqual(churn["shared_calls"], 2)
        self.assertAlmostEqual(
            churn["downstream_effective_payload_churn"]["estimate"], 0.5
        )
        counts = analyze_call_counts(
            {"baseline": baseline}, n_resamples=100, seed=0
        )["main"]["baseline"]
        self.assertAlmostEqual(
            counts["calls_by_conceptual_role"]["extractor"]["estimate"], 1.0
        )

    def test_selection_artifact_hash_has_a_non_circular_verifier(self):
        artifact = {
            "question_manifest_sha256": EXPECTED_FINAL_MANIFEST_SHA256,
            "execution_run_id": "baseline",
            "run_definition": {"qa": "fp16"},
            "source_config_sha256": content_hash({"runs": {"baseline": {}}}),
        }
        artifact["run_config_sha256"] = content_hash(artifact["run_definition"])
        artifact["artifact_sha256"] = content_hash(artifact)
        executable = {
            "runs": {"baseline": artifact["run_definition"]},
            "frozen_allocation": {
                "selection_artifact_sha256": artifact["artifact_sha256"],
                "run_config_sha256": artifact["run_config_sha256"],
            },
        }
        artifact["executable_config_sha256"] = content_hash(executable)
        self.assertTrue(verify_selection_artifact(artifact, executable))
        broken_executable = {**executable, "runs": {"baseline": {}, "extra": {}}}
        self.assertFalse(verify_selection_artifact(artifact, broken_executable))
        artifact["run_definition"]["qa"] = "4bit"
        self.assertFalse(verify_selection_artifact(artifact, executable))


if __name__ == "__main__":
    unittest.main()
