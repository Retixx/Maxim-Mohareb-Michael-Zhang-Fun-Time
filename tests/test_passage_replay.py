import ast
import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from clean_room import passage_replay_core as core
from clean_room import passage_replay as runtime
from src import agents, prompts


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "gate_c_1.7b"


class PassageReplaySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = core.load_source_bundle(EVIDENCE)

    def test_committed_sources_match_all_pins_and_cohorts(self):
        source = self.source
        self.assertEqual(source.question_ids_sha256, core.FULL_IDS_SHA256)
        self.assertEqual(len(source.question_ids), 200)
        self.assertEqual(
            Counter(question.stratum for question in source.scoring.values()),
            {"hidden_bridge": 160, "fully_named": 40},
        )
        self.assertEqual(len(source.both_gold_ids), 128)
        self.assertEqual(
            core.ordered_ids_sha256(source.both_gold_ids),
            core.BOTH_GOLD_IDS_SHA256,
        )
        self.assertEqual(
            Counter(source.scoring[qid].stratum for qid in source.both_gold_ids),
            {"hidden_bridge": 95, "fully_named": 33},
        )
        self.assertEqual(source.baseline_meta["git_commit"], core.SOURCE_COMMIT)
        self.assertEqual(
            sum(
                source.single_answers[qid].get("retrieval_all_gold") is True
                for qid in source.question_ids
            ),
            108,
        )
        for frozen in source.questions.values():
            self.assertFalse(hasattr(frozen, "gold_answer"))
            self.assertFalse(hasattr(frozen, "stratum"))

    def test_hash_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            target = Path(raw)
            for name in core.SOURCE_SHA256:
                shutil.copy2(EVIDENCE / name, target / name)
            path = target / core.BASELINE_META
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(core.IntegrityError, "SHA-256"):
                core.load_source_bundle(target)


class PassageReplayTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = core.load_source_bundle(EVIDENCE)

    def test_committed_trace_reconstructs_all_source_prompts(self):
        audit = core.audit_source(self.source)
        self.assertEqual(
            audit.qa_stage_counts,
            {"qa": 200, "qa_step2": 187, "qa_step3": 32, "qa_step4": 7, "qa_step5": 1},
        )
        self.assertEqual(audit.qa_prompt_hash_matches, 427)
        self.assertEqual(audit.summary_prompt_hash_matches, 200)
        self.assertEqual(audit.question_answering_calls, 426)
        self.assertEqual(audit.aggregate_calls, 1)
        self.assertEqual(audit.extractor_joins, 4260)

    def test_context_conditions_have_exact_text_and_duplication(self):
        original = "1. Document 1: Alpha\n   - Selected sentence."
        join = core.PassageJoin(
            ("Alpha", "Beta"),
            (("Selected sentence.", "Remaining sentence."), ("Other sentence.",)),
        )
        plus = core.render_treated_evidence(original, join, core.SPANS_PLUS_PASSAGES)
        only = core.render_treated_evidence(original, join, core.PASSAGES_ONLY)
        passage = (
            "[1] Alpha: Selected sentence. Remaining sentence.\n"
            "[2] Beta: Other sentence."
        )
        self.assertEqual(
            plus,
            original + "\n\nRetrieved passages for the current step:\n" + passage,
        )
        self.assertEqual(
            only,
            "Retrieved passages for the current step:\n" + passage,
        )
        self.assertEqual(plus.count("Selected sentence."), 2)
        self.assertEqual(only.count("Selected sentence."), 1)

    def _first_retrieval_qa(self, source):
        return next(
            record
            for record in core.source_qa_records(source)
            if record["consumer_input"]["task_type"] == "question-answering"
        )

    def test_real_passage_join_is_exact(self):
        qa = self._first_retrieval_qa(self.source)
        joined = core.join_recorded_passages(self.source, qa)
        self.assertEqual(len(joined.titles), 10)
        self.assertEqual(
            joined.titles,
            tuple(qa["consumer_input"]["retrieval"]["titles"]),
        )
        self.assertEqual(len(joined.sentence_lists), 10)

    def test_passage_join_rejects_relational_mutations(self):
        mutations = ("missing_rank", "duplicate_rank", "swapped_title", "task", "retrieval")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                source = copy.deepcopy(self.source)
                qa = self._first_retrieval_qa(source)
                qid = qa["question_id"]
                step = int(qa["call_index"])
                ext_stage = prompts.stage_for("extractor", step)
                if mutation == "missing_rank":
                    del source.baseline_index[(qid, ext_stage, 9)]
                elif mutation == "duplicate_rank":
                    source.baseline_index[(qid, ext_stage, 1)]["consumer_input"][
                        "document_rank"
                    ] = 0
                elif mutation == "swapped_title":
                    source.baseline_index[(qid, ext_stage, 0)]["consumer_input"][
                        "document_title"
                    ] = "Wrong title"
                elif mutation == "task":
                    source.baseline_index[(qid, ext_stage, 0)]["consumer_input"][
                        "step_definition"
                    ]["task"] = "Wrong task"
                else:
                    titles = qa["consumer_input"]["retrieval"]["titles"]
                    titles[0], titles[1] = titles[1], titles[0]
                with self.assertRaises(core.IntegrityError):
                    core.join_recorded_passages(source, qa)

    def test_condition_populations_calls_and_batches_are_exact(self):
        plus_calls = core.condition_call_keys(self.source, core.SPANS_PLUS_PASSAGES)
        only_calls = core.condition_call_keys(self.source, core.PASSAGES_ONLY)
        plus_batches = core.condition_batches(self.source, core.SPANS_PLUS_PASSAGES)
        only_batches = core.condition_batches(self.source, core.PASSAGES_ONLY)

        self.assertEqual(len(plus_calls), 627)
        self.assertEqual(len(only_calls), 411)
        self.assertEqual(len(plus_batches), 158)
        self.assertEqual(len(only_batches), 104)
        self.assertEqual(sum(len(batch.members) for batch in plus_batches), 627)
        self.assertEqual(sum(len(batch.members) for batch in only_batches), 411)
        self.assertEqual(len(set(plus_calls)), 627)
        self.assertEqual(len(set(only_calls)), 411)
        self.assertFalse(any(key.stage == "solo" for key in plus_calls + only_calls))
        self.assertEqual(
            Counter(key.stage for key in only_calls),
            {
                "qa": 128,
                "qa_step2": 123,
                "qa_step3": 24,
                "qa_step4": 7,
                "qa_step5": 1,
                "plan_summary": 128,
            },
        )

    def test_plus_batches_preserve_source_members_exactly(self):
        plus = core.condition_batches(self.source, core.SPANS_PLUS_PASSAGES)
        expected = []
        for stage in (*core.QA_STAGES, "plan_summary"):
            expected.extend(core.source_scored_batches(self.source, stage))
        self.assertEqual(
            [(batch.stage, batch.ordinal, batch.members) for batch in plus],
            [(batch.stage, batch.ordinal, batch.members) for batch in expected],
        )

    def test_prompt_reconstruction_is_invariant_to_poisoned_scoring_fields(self):
        qa = self._first_retrieval_qa(self.source)
        original_fields = core.reconstruct_source_qa_fields(self.source, qa)
        original_messages = prompts.build_messages("qa", **original_fields)

        poisoned = copy.deepcopy(self.source)
        qid = qa["question_id"]
        poisoned.scoring[qid] = replace(
            poisoned.scoring[qid],
            gold_answer="POISON GOLD",
            stratum="fully_named",
            both_gold=False,
        )
        poisoned.baseline_answers[qid]["gold_answer"] = "POISON GOLD"
        poisoned.baseline_answers[qid]["gold_titles"] = ["POISON TITLE"]
        poisoned.baseline_answers[qid]["supporting_facts"] = {"POISON": [999]}
        poisoned.single_answers[qid]["retrieval_all_gold"] = not bool(
            poisoned.single_answers[qid].get("retrieval_all_gold")
        )

        poisoned_fields = core.reconstruct_source_qa_fields(poisoned, qa)
        poisoned_messages = prompts.build_messages("qa", **poisoned_fields)
        self.assertEqual(poisoned_fields, original_fields)
        self.assertEqual(poisoned_messages, original_messages)
        self.assertEqual(
            agents.rendered_prompt_sha256(poisoned_messages),
            agents.rendered_prompt_sha256(original_messages),
        )


class PassageReplayTreatedStateTests(unittest.TestCase):
    condition_fingerprint_sha256 = "f" * 64

    @classmethod
    def setUpClass(cls):
        cls.source = core.load_source_bundle(EVIDENCE)
        cls.aggregate_qa = next(
            record
            for record in core.source_qa_records(cls.source)
            if record["consumer_input"]["task_type"] == "aggregate"
        )
        cls.qid = cls.aggregate_qa["question_id"]
        cls.source_qas = {
            record["call_index"]: record
            for record in core.source_qa_records(cls.source)
            if record["question_id"] == cls.qid
        }

    def _treated(self, step, answer, *, success="yes", mode="parsed"):
        record = copy.deepcopy(self.source_qas[step])
        payload = {
            "analysis": "treated",
            "answer": answer,
            "success": success,
            "rating": 7,
        }
        record["parsed"] = payload if mode == "parsed" else None
        record["salvaged"] = payload if mode == "salvaged" else None
        if mode == "fallback":
            record["parsed"] = None
            record["salvaged"] = None
        return record

    def _index(self, step_records, condition=core.SPANS_PLUS_PASSAGES):
        output = {}
        for original in step_records:
            record = copy.deepcopy(original)
            record["condition"] = condition
            record["condition_fingerprint_sha256"] = (
                self.condition_fingerprint_sha256
            )
            output[(condition, self.qid, record["stage"], record["call_index"])] = record
        return output

    def test_aggregate_uses_treated_grounded_answers_without_retrieval(self):
        treated_index = self._index(
            [self._treated(0, "Motörhead"), self._treated(1, "ImpossibleValue")]
        )
        call = core.build_treated_qa_call(
            self.source,
            self.aggregate_qa,
            core.SPANS_PLUS_PASSAGES,
            treated_index,
            self.condition_fingerprint_sha256,
        )
        source_input = self.aggregate_qa["consumer_input"]
        self.assertEqual(
            call["consumer_input"]["step_definition"],
            source_input["step_definition"],
        )
        self.assertEqual(call["consumer_input"]["retrieval"], source_input["retrieval"])
        self.assertFalse(call["consumer_input"]["retrieval"]["attempted"])
        self.assertIn("Prior step answer: Motörhead", call["fields"]["evidence"])
        self.assertNotIn(
            "Prior step answer: Noel Gallagher's High Flying Birds",
            call["fields"]["evidence"],
        )
        self.assertNotIn("ImpossibleValue", call["fields"]["evidence"])

    def test_aggregate_withholds_ungrounded_treated_answers(self):
        treated_index = self._index(
            [self._treated(0, "ImpossibleOne"), self._treated(1, "ImpossibleTwo")]
        )
        call = core.build_treated_qa_call(
            self.source,
            self.aggregate_qa,
            core.SPANS_PLUS_PASSAGES,
            treated_index,
            self.condition_fingerprint_sha256,
        )
        self.assertEqual(call["fields"]["evidence"], "(no evidence collected)")
        self.assertEqual(call["consumer_input"]["evidence_prompt_block_count"], 0)

    def test_treated_history_and_summary_are_coherent_without_truncation(self):
        treated_index = self._index(
            [
                self._treated(0, "Motörhead", success="no"),
                self._treated(1, "ImpossibleValue"),
                self._treated(2, "1980", mode="salvaged"),
            ]
        )
        history = core.rebuild_treated_history(
            self.source,
            self.qid,
            treated_index,
            core.SPANS_PLUS_PASSAGES,
            self.condition_fingerprint_sha256,
        )
        self.assertEqual(len(history), 3)
        self.assertEqual([item["answer"] for item in history], ["Motörhead", "ImpossibleValue", "1980"])
        self.assertEqual(history[0]["success"], "no")
        self.assertTrue(history[0]["answer_grounded"])
        self.assertEqual(history[2]["qa_source"], "salvaged")
        self.assertEqual(
            [item["task"] for item in history],
            [item["task"] for item in self.source.baseline_index[(self.qid, "plan_summary", 0)]["consumer_input"]["completed_steps"]],
        )

        call = core.build_treated_summary_call(
            self.source,
            self.qid,
            history,
            core.SPANS_PLUS_PASSAGES,
        )
        rendered = prompts.build_messages("plan_summary", **call["fields"])[1]["content"]
        self.assertIn("Motörhead", rendered)
        self.assertNotIn("Noel Gallagher's High Flying Birds", call["fields"]["prior_state"])
        self.assertEqual(
            call["consumer_input"]["stop_reason"],
            self.source.questions[self.qid].stop_reason,
        )

    def test_treated_history_rejects_cross_condition_or_fingerprint_rows(self):
        treated_index = self._index([self._treated(0, "Motörhead")])
        key = next(iter(treated_index))
        treated_index[key]["condition"] = core.PASSAGES_ONLY
        with self.assertRaisesRegex(core.IntegrityError, "condition"):
            core.rebuild_treated_history(
                self.source,
                self.qid,
                treated_index,
                core.SPANS_PLUS_PASSAGES,
                self.condition_fingerprint_sha256,
                before_step=1,
            )

    def test_aggregate_grounding_uses_raw_answers_not_prompt_wrapper_words(self):
        source_history = self.source.baseline_index[
            (self.qid, "plan_summary", 0)
        ]["consumer_input"]["completed_steps"]
        self.assertNotIn(
            "prior",
            " ".join(str(item.get("answer") or "") for item in source_history[:2]).casefold(),
        )
        treated_index = self._index(
            [
                self._treated(0, source_history[0]["answer"]),
                self._treated(1, source_history[1]["answer"]),
                self._treated(2, "prior"),
            ]
        )
        history = core.rebuild_treated_history(
            self.source,
            self.qid,
            treated_index,
            core.SPANS_PLUS_PASSAGES,
            self.condition_fingerprint_sha256,
        )
        self.assertFalse(history[2]["answer_grounded"])

        treated_index = self._index([self._treated(0, "Motörhead")])
        treated_index[next(iter(treated_index))]["condition_fingerprint_sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(core.IntegrityError, "fingerprint"):
            core.rebuild_treated_history(
                self.source,
                self.qid,
                treated_index,
                core.SPANS_PLUS_PASSAGES,
                self.condition_fingerprint_sha256,
                before_step=1,
            )

    def test_effective_payload_covers_parsed_salvaged_and_fallback(self):
        parsed = self._treated(0, "Parsed", mode="parsed")
        salvaged = self._treated(0, "Salvaged", mode="salvaged")
        fallback = self._treated(0, "Ignored", mode="fallback")
        self.assertEqual(core.effective_payload(parsed)[1], "parsed")
        self.assertEqual(core.effective_payload(salvaged)[1], "salvaged")
        self.assertEqual(core.effective_payload(fallback), ({}, "fallback"))

    def test_finalizer_uses_summary_then_reverse_qa_precedence(self):
        history = [
            {"step_number": 1, "answer": "First", "answer_grounded": True},
            {"step_number": 2, "answer": "Final QA", "answer_grounded": False},
        ]
        parsed = core.resolve_treated_answer({"parsed": {"answer": "Summary"}}, history)
        salvaged = core.resolve_treated_answer(
            {"parsed": None, "salvaged": {"answer": "Salvaged summary"}}, history
        )
        fallback = core.resolve_treated_answer(
            {"parsed": {"answer": "unknown"}, "salvaged": None}, history
        )
        empty = core.resolve_treated_answer(None, [])
        self.assertEqual((parsed["answer"], parsed["source"]), ("Summary", "summary_parsed"))
        self.assertEqual(
            (salvaged["answer"], salvaged["source"]),
            ("Salvaged summary", "summary_salvaged"),
        )
        self.assertEqual((fallback["answer"], fallback["source"]), ("Final QA", "qa_fallback"))
        self.assertEqual((empty["answer"], empty["source"]), ("", "none"))


class PassageReplayScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = core.load_source_bundle(EVIDENCE)

    def test_survival_headline_reproduces_artifact(self):
        self.assertEqual(
            core.extractor_survival_headline(self.source),
            {
                "n": 128,
                "raw_producer_present": 84,
                "normalized_qa_prompt_present": 52,
                "raw_and_prompt_present": 50,
                "absent_from_raw_and_prompt": 42,
                "present_raw_removed_by_normalizer": 34,
                "absent_raw_recovered_by_normalizer": 2,
                "method": "contiguous normalize_answer token phrase",
            },
        )

    def test_normalizer_near_match_diagnostic_separates_mixed_loss(self):
        report = core.extractor_near_match_diagnostic(self.source)
        self.assertEqual(
            report["normalizer_totals"],
            {
                "input_spans": 1460,
                "rejected_spans": 830,
                "rejected_fraction": 0.568493,
                "rejection_reasons": {
                    "duplicate_sentence": 4,
                    "fragment_too_short": 82,
                    "multiple_sentences": 119,
                    "not_in_source": 625,
                },
            },
        )
        self.assertEqual(
            report["all_questions"]["at_or_above"],
            {"0.50": 430, "0.70": 293, "0.80": 228, "0.90": 164},
        )
        self.assertEqual(
            report["both_gold_questions"]["at_or_above"],
            {"0.50": 302, "0.70": 210, "0.80": 162, "0.90": 117},
        )
        self.assertIn("cannot distinguish", report["interpretation"])
        self.assertIn("does not assign loss", report["interpretation"])

    def test_paired_metrics_include_f1_em_and_declared_direction(self):
        a = {
            "q1": {"f1": 1.0, "em": 1.0},
            "q2": {"f1": 0.0, "em": 0.0},
            "q3": {"f1": 0.5, "em": 0.0},
        }
        b = {
            "q1": {"f1": 0.0, "em": 0.0},
            "q2": {"f1": 0.5, "em": 1.0},
            "q3": {"f1": 0.5, "em": 0.0},
        }
        report = core.paired_comparison(a, b, ("q1", "q2", "q3"))
        self.assertEqual(report["n"], 3)
        self.assertAlmostEqual(report["a_f1"], 0.5)
        self.assertAlmostEqual(report["b_f1"], 1 / 3)
        self.assertAlmostEqual(report["delta_f1_points"], 100 / 6)
        self.assertAlmostEqual(report["a_em"], 1 / 3)
        self.assertAlmostEqual(report["b_em"], 1 / 3)
        self.assertAlmostEqual(report["delta_em_points"], 0.0)
        self.assertEqual((report["wins"], report["losses"], report["ties"]), (1, 1, 1))
        self.assertEqual(report["mcnemar"]["a_only_wins"], 1)
        self.assertEqual(report["mcnemar"]["b_only_wins"], 1)
        self.assertIn("f1_points", report["bootstrap"])
        self.assertIn("em_points", report["bootstrap"])

    def test_paired_metrics_reject_incomplete_cohort(self):
        with self.assertRaisesRegex(core.IntegrityError, "incomplete"):
            core.paired_comparison(
                {"q1": {"f1": 1.0, "em": 1.0}},
                {},
                ("q1",),
            )

    def test_practical_equivalence_includes_exact_two_point_boundary(self):
        self.assertTrue(core.practically_equivalent(0.4213, 0.4013))
        self.assertFalse(core.practically_equivalent(0.4213, 0.4012))

    def test_all_predeclared_interpretation_branches(self):
        near = core.interpret_scores(single_f1=0.4213, plus_f1=0.4090, only_f1=0.4110)
        self.assertIn(
            "extraction is not contributing a net advantage over raw retrieval",
            near,
        )
        self.assertIn("selected Extractor spans add no measurable value", near)

        salience = core.interpret_scores(single_f1=0.50, plus_f1=0.45, only_f1=0.42)
        self.assertIn("selected-span/repetition-salience increment", salience)

        distraction = core.interpret_scores(single_f1=0.50, plus_f1=0.42, only_f1=0.45)
        self.assertIn("retained spans or their duplication distract QA", distraction)

        insufficient = core.interpret_scores(single_f1=0.50, plus_f1=0.40, only_f1=0.39)
        self.assertIn("passage access alone is insufficient", insufficient)

        decomposition = core.interpret_scores(single_f1=0.42, plus_f1=0.45, only_f1=0.44)
        self.assertIn("decomposition value conditional on the frozen trace", decomposition)

        combined = " ".join((near, salience, distraction, insufficient, decomposition))
        self.assertNotIn("§4.3 is repairable", combined)
        self.assertNotIn("PASS_GATE_C", combined)
        self.assertNotIn("GO", combined)

    def test_report_label_and_gate_thresholds_are_diagnostic_only(self):
        self.assertEqual(
            core.REPORT_LABEL,
            "Gate-C-comparable fixed-trace diagnostic",
        )
        self.assertEqual(
            core.GATE_C_THRESHOLDS,
            {
                "overall_delta_f1_points_min": 5.0,
                "overall_ci_lower_points_strictly_greater_than": 2.0,
                "mcnemar_p_strictly_less_than": 0.01,
                "hidden_bridge_delta_f1_points_min": 8.0,
                "fully_named_delta_f1_points_range": [-2.0, 2.0],
            },
        )

    def test_source_outputs_round_trip_through_complete_report(self):
        fingerprints = {
            core.SPANS_PLUS_PASSAGES: "a" * 64,
            core.PASSAGES_ONLY: "b" * 64,
        }
        plus_index = {}
        only_index = {}
        both_gold = set(self.source.both_gold_ids)
        for source_record in core.source_qa_records(self.source):
            record = copy.deepcopy(source_record)
            key = (record["question_id"], record["stage"], record["call_index"])
            record["condition"] = core.SPANS_PLUS_PASSAGES
            record["condition_fingerprint_sha256"] = fingerprints[
                core.SPANS_PLUS_PASSAGES
            ]
            plus_index[(core.SPANS_PLUS_PASSAGES, *key)] = record
            if record["question_id"] in both_gold:
                only_record = copy.deepcopy(source_record)
                only_record["condition"] = core.PASSAGES_ONLY
                only_record["condition_fingerprint_sha256"] = fingerprints[
                    core.PASSAGES_ONLY
                ]
                only_index[(core.PASSAGES_ONLY, *key)] = only_record
        plus_summaries = {}
        only_summaries = {}
        for source_record in core.source_summary_records(self.source):
            plus_record = copy.deepcopy(source_record)
            plus_record["condition"] = core.SPANS_PLUS_PASSAGES
            plus_record["condition_fingerprint_sha256"] = fingerprints[
                core.SPANS_PLUS_PASSAGES
            ]
            plus_summaries[source_record["question_id"]] = plus_record
            if source_record["question_id"] in both_gold:
                only_record = copy.deepcopy(source_record)
                only_record["condition"] = core.PASSAGES_ONLY
                only_record["condition_fingerprint_sha256"] = fingerprints[
                    core.PASSAGES_ONLY
                ]
                only_summaries[source_record["question_id"]] = only_record
        plus = core.build_condition_result(
            self.source,
            core.SPANS_PLUS_PASSAGES,
            plus_index,
            plus_summaries,
            fingerprints[core.SPANS_PLUS_PASSAGES],
        )
        only = core.build_condition_result(
            self.source,
            core.PASSAGES_ONLY,
            only_index,
            only_summaries,
            fingerprints[core.PASSAGES_ONLY],
        )
        report = core.score_replay(
            self.source,
            {core.SPANS_PLUS_PASSAGES: plus, core.PASSAGES_ONLY: only},
        )

        self.assertEqual(len(plus.answer_records), 200)
        self.assertEqual(len(only.answer_records), 128)
        for qid, record in plus.answer_records.items():
            self.assertEqual(record["predicted_answer"], self.source.baseline_answers[qid]["predicted_answer"])
            self.assertEqual(record["f1"], self.source.baseline_answers[qid]["f1"])
            self.assertEqual(record["em"], self.source.baseline_answers[qid]["em"])
        self.assertEqual(report["report_label"], core.REPORT_LABEL)
        self.assertEqual(
            report["final_answer"][core.SPANS_PLUS_PASSAGES]["versus_source_ma"]["overall"][
                "delta_f1_points"
            ],
            0.0,
        )
        self.assertEqual(
            report["final_answer"]["three_way_both_gold"][
                "spans_plus_passages_minus_spans_only"
            ]["delta_f1_points"],
            0.0,
        )
        spans_only_vs_single = report["final_answer"]["three_way_both_gold"][
            "spans_only_minus_single"
        ]
        self.assertEqual(spans_only_vs_single["n"], 128)
        self.assertAlmostEqual(
            spans_only_vs_single["a_f1"],
            sum(
                self.source.baseline_answers[qid]["f1"]
                for qid in self.source.both_gold_ids
            )
            / 128,
        )
        sensitivity = report["qa_level"][core.SPANS_PLUS_PASSAGES][
            "earliest_new_no_sensitivity"
        ]
        self.assertEqual(sensitivity["full_cohort"]["means"]["overall"]["n"], 200)
        self.assertEqual(
            sensitivity["full_cohort"]["versus_source_ma"]["overall"]["n"],
            200,
        )
        self.assertIn("reverse_usable", report["qa_level"][core.SPANS_PLUS_PASSAGES])
        self.assertIn("success_drift", report["qa_level"][core.SPANS_PLUS_PASSAGES])
        self.assertIn("right_censoring", report["qa_level"][core.SPANS_PLUS_PASSAGES])
        per_step = report["qa_level"][core.SPANS_PLUS_PASSAGES][
            "parse_salvage_by_question_step"
        ]
        self.assertEqual(len(per_step), 427)
        self.assertEqual(
            set(per_step[0]),
            {
                "question_id",
                "stage",
                "call_index",
                "step_number",
                "parse_status",
                "parsed_present",
                "salvaged_present",
                "effective_payload_source",
            },
        )
        self.assertTrue(
            all(item["step_number"] == item["call_index"] + 1 for item in per_step)
        )


class FakeReplayTokenizer:
    chat_template = "fake-chat-template"

    def __init__(self, token_count=10):
        self.token_count = token_count
        self.calls = []

    def save_pretrained(self, path):
        path = Path(path)
        (path / "nested").mkdir()
        (path / "z.json").write_text('{"z":1}', encoding="utf-8")
        (path / "nested" / "a.txt").write_text("alpha", encoding="utf-8")

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": list(range(self.token_count))}


class PassageReplayRuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = core.load_source_bundle(EVIDENCE)

    @staticmethod
    def _persistence_identity():
        return {
            "model_id": runtime.MODEL_ID,
            "model_revision": runtime.MODEL_REVISION,
            "tokenizer_revision": runtime.MODEL_REVISION,
            "precision": runtime.PRECISION,
        }

    @staticmethod
    def _execution_identity():
        return {
            "model_id": runtime.MODEL_ID,
            "model_revision": runtime.MODEL_REVISION,
            "tokenizer_revision": runtime.MODEL_REVISION,
            "precision": runtime.PRECISION,
            "census": dict(runtime.EXPECTED_CENSUS),
            "gpu": {
                "gpu_available": True,
                "gpu_name": runtime.EXPECTED_GPU_NAME,
                "gpu_compute_capability": runtime.EXPECTED_GPU_COMPUTE_CAPABILITY,
                "gpu_driver_version": "580.159.04",
            },
            "batch_size": 4,
            "enable_thinking": False,
            "tokenizer_identity": {
                "chat_template_type": "str",
                "chat_template_sha256": "a" * 64,
                "snapshot_sha256": "b" * 64,
                "files": {
                    "tokenizer.json": {"sha256": "c" * 64, "size_bytes": 123}
                },
            },
            "library_versions": {
                "python": "3.12.3",
                "torch": "warning-only-version",
            },
            "source_rendered_chat_manifest_sha256": "d" * 64,
            "source_rendered_chat_manifest_count": 627,
            "replay_git_commit": "e" * 40,
        }

    def _synthetic_batch(
        self,
        ordinal,
        *,
        phase="scored",
        size=4,
        condition=core.SPANS_PLUS_PASSAGES,
    ):
        if phase == runtime.SENTINEL_PHASE:
            condition = runtime.SENTINEL_CONDITION
        members = tuple(
            runtime.MemberSpec(
                condition=condition,
                question_id=f"q-{phase}-{ordinal}-{index}",
                stage="qa",
                call_index=ordinal * 4 + index,
                batch_member_index=index,
                parent_key=(f"q-{phase}-{ordinal}-{index}", "qa", ordinal * 4 + index),
                parent_record_sha256="1" * 64,
                source_message_sha256="2" * 64,
                treatment_message_sha256="3" * 64,
                rendered_chat_sha256="4" * 64,
            )
            for index in range(size)
        )
        return runtime.make_batch_spec(
            phase=phase,
            condition=condition,
            condition_fingerprint_sha256="5" * 64,
            stage="qa",
            batch_ordinal=ordinal,
            members=members,
            batch_id=f"{phase}:{condition}:qa:{ordinal:06d}",
        )

    def _synthetic_call(self, member, batch, *, answer="ok"):
        generated = {
            "question_id": member.question_id,
            "stage": member.stage,
            "call_index": member.call_index,
            "raw_output": json.dumps({"answer": answer}),
            "parse_status": "ok",
            "parsed": {"answer": answer, "success": "yes", "rating": 7},
            "salvaged": None,
            "consumer_payload": None,
            "consumer_input": {"frozen": True},
            "consumer_payload_source": "replay-test",
            "message_object_sha256": member.treatment_message_sha256,
            "rendered_prompt_sha256": member.treatment_message_sha256,
            "prompt_tokens": 10,
            "output_ceiling_tokens": 96,
            "prompt_plus_ceiling_tokens": 106,
            "output_tokens": 5,
            "generated_sequence_tokens": 6,
            "context_window_tokens": 40960,
            "forced_full_generation": False,
            "strict_format_ok": True,
            "protocol_ok": True,
        }
        return runtime.augment_generated_call(
            generated,
            member,
            batch,
            identity=self._persistence_identity(),
        )

    def _model_load_record(self):
        return {
            "record_type": "model_load",
            "status": "success",
            "execution_fingerprint_sha256": "6" * 64,
            "condition_fingerprints": {
                core.SPANS_PLUS_PASSAGES: "7" * 64,
                core.PASSAGES_ONLY: "8" * 64,
            },
            "source_experiment_fingerprint": core.SOURCE_EXPERIMENT_FINGERPRINT,
            "source_artifact_sha256": dict(core.SOURCE_SHA256),
            "source_question_ids_sha256": core.FULL_IDS_SHA256,
            "source_both_gold_ids_sha256": core.BOTH_GOLD_IDS_SHA256,
            **self._persistence_identity(),
        }

    def _minimal_publication_validation(self):
        manifest = {"schema": "atomic-publication-test-v1"}
        return runtime.PublicationValidationPayload(
            source=self.source,
            execution_fingerprint_sha256="6" * 64,
            replay_fingerprint_sha256=core.canonical_json_sha256(manifest),
            condition_fingerprints={
                core.SPANS_PLUS_PASSAGES: "7" * 64,
                core.PASSAGES_ONLY: "8" * 64,
            },
            batch_specs=(),
            identity=self._persistence_identity(),
            expected_manifest=manifest,
        )

    def test_package_mismatch_warns_but_sentinel_is_hard_gate(self):
        warnings = runtime.version_warnings(
            {"torch": "2.10.0+cu128", "transformers": "5.14.1"},
            {"torch": "2.11.0+cu128", "transformers": "5.15.0"},
        )
        self.assertEqual(len(warnings), 2)

        source_records = runtime.sentinel_source_calls(self.source)
        replay_records = copy.deepcopy(source_records)
        replay_records[0]["parse_status"] = "recorded-but-not-gating"
        report = runtime.compare_sentinel(source_records, replay_records)
        self.assertFalse(report["parse_status_all_match"])
        replay_records[0]["output_tokens"] += 1
        with self.assertRaisesRegex(core.IntegrityError, "sentinel.*output_tokens"):
            runtime.compare_sentinel(source_records, replay_records)

    def test_replay_modules_cannot_import_or_call_live_retrieval_or_solo_paths(self):
        forbidden_calls = {
            "build_stage_calls",
            "_run_stage",
            "build_solo_fields",
            "search_titles",
            "retrieve",
            "run_solo",
        }
        for relative in (
            "clean_room/passage_replay.py",
            "clean_room/passage_replay_core.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = {alias.name for alias in node.names}
                    self.assertFalse(
                        any(
                            name == "src.pipeline"
                            or name == "src.retrieval"
                            or name.endswith(".retrieval")
                            for name in imported
                        ),
                        f"forbidden import in {relative}: {sorted(imported)}",
                    )
                elif isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                    module = node.module or ""
                    self.assertFalse(
                        module in {"src.pipeline", "src.retrieval"}
                        or (module == "src" and names.intersection({"pipeline", "retrieval"})),
                        f"forbidden import in {relative}: {module} {sorted(names)}",
                    )
                elif isinstance(node, ast.Call):
                    called = (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else node.func.id
                        if isinstance(node.func, ast.Name)
                        else None
                    )
                    self.assertNotIn(
                        called,
                        forbidden_calls,
                        f"forbidden call in {relative}: {called}",
                    )

    def test_sentinel_selects_exactly_21_source_members(self):
        batches = runtime.sentinel_batches(self.source)
        self.assertEqual([len(batch.members) for batch in batches], [4, 4, 4, 4, 1, 4])
        self.assertEqual(sum(len(batch.members) for batch in batches), 21)
        self.assertEqual(
            [batch.stage for batch in batches],
            ["qa", "qa_step2", "qa_step3", "qa_step4", "qa_step5", "plan_summary"],
        )
        source_telemetry = runtime.sentinel_source_batch_records(self.source)
        self.assertEqual(len(source_telemetry), 6)
        self.assertEqual(
            [record["stage"] for record in source_telemetry],
            [batch.stage for batch in batches],
        )
        self.assertTrue(
            all(record["gpu_driver_version"] == "580.159.04" for record in source_telemetry)
        )

    def test_sentinel_report_records_paired_batch_telemetry_and_think_checks(self):
        from unittest.mock import patch

        batches = runtime.sentinel_batches(self.source)
        source_calls = runtime.sentinel_source_calls(self.source)
        sizes = [len(batch.members) for batch in batches]
        generated = []
        offset = 0
        for index, size in enumerate(sizes):
            generated.append(
                (tuple(copy.deepcopy(source_calls[offset : offset + size])), {"ordinal": index})
            )
            offset += size
        plan = tuple((batch, ()) for batch in batches)
        with patch.object(runtime, "_run_prepared_batch", side_effect=generated):
            buffered, report = runtime._run_sentinel_buffer(
                object(),
                object(),
                self.source,
                plan,
                run_calls_fn=object(),
                run_id="test",
                execution_session_id="session",
                gpu={},
                model_config_fingerprint="config",
                execution_fingerprint_sha256="fingerprint",
            )
        self.assertEqual(report["think_tag_checks"]["source"]["checked_calls"], 21)
        self.assertEqual(report["think_tag_checks"]["replay"]["checked_calls"], 21)
        self.assertEqual(len(report["batch_telemetry"]), 6)
        self.assertEqual(report["batch_telemetry"][3]["replay"], {"ordinal": 3})
        self.assertIn("source_batch", buffered[0][2])
        self.assertIn("replay_batch", buffered[0][2])

    def test_cpu_audit_reports_exact_frozen_topology_and_treatments(self):
        report = runtime.build_cpu_audit(self.source)
        self.assertEqual(report["source_questions"], 200)
        self.assertEqual(report["source_strata"], {"hidden_bridge": 160, "fully_named": 40})
        self.assertEqual(report["both_gold_questions"], 128)
        self.assertEqual(
            report["both_gold_strata"],
            {"hidden_bridge": 95, "fully_named": 33},
        )
        self.assertEqual(
            report["source_qa_stage_counts"],
            {"qa": 200, "qa_step2": 187, "qa_step3": 32, "qa_step4": 7, "qa_step5": 1},
        )
        self.assertEqual(report["retrieval_depth_distribution"], {1: 13, 2: 156, 3: 24, 4: 6, 5: 1})
        self.assertEqual(report["source_prompt_hashes"], {"matched": 627, "expected": 627})
        self.assertEqual(
            report["conditions"],
            {
                core.SPANS_PLUS_PASSAGES: {"calls": 627, "batches": 158},
                core.PASSAGES_ONLY: {"calls": 411, "batches": 104},
            },
        )
        self.assertEqual(report["scored_total"], {"calls": 1038, "batches": 262})
        self.assertEqual(
            report["answer_survival"],
            core.extractor_survival_headline(self.source),
        )

    def test_sentinel_rejects_thinking_tags_case_insensitively(self):
        source_records = runtime.sentinel_source_calls(self.source)
        replay_records = copy.deepcopy(source_records)
        replay_records[0]["raw_output"] += "<ThInK>bad</tHiNk>"
        with self.assertRaisesRegex(core.IntegrityError, "thinking tag"):
            runtime.compare_sentinel(source_records, replay_records)

    def test_tokenizer_snapshot_hashes_files_and_chat_template(self):
        tokenizer = FakeReplayTokenizer()
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            scratch = Path(raw)
            first = runtime.capture_tokenizer_identity(
                tokenizer,
                repo_root=ROOT,
                scratch_parent=scratch,
            )
            second = runtime.capture_tokenizer_identity(
                tokenizer,
                repo_root=ROOT,
                scratch_parent=scratch,
            )
        self.assertEqual(first, second)
        self.assertEqual(list(first.files), ["nested/a.txt", "z.json"])
        self.assertEqual(first.chat_template_type, "str")
        self.assertEqual(len(first.snapshot_sha256), 64)
        self.assertEqual(len(first.chat_template_sha256), 64)

    def test_runtime_prompt_audit_uses_one_renderer_and_hard_context_ceiling(self):
        tokenizer = FakeReplayTokenizer(token_count=100)
        messages = [{"role": "user", "content": "x"}]
        from unittest.mock import patch

        with patch("src.models.render_chat", return_value="rendered") as render:
            audit = runtime.audit_prompt_runtime(
                tokenizer,
                messages,
                prompt_role="qa",
                recorded_context_window_tokens=196,
            )
        render.assert_called_once_with(tokenizer, messages)
        self.assertEqual(audit.prompt_tokens, 100)
        self.assertEqual(audit.output_ceiling_tokens, 96)
        self.assertEqual(audit.prompt_plus_ceiling_tokens, 196)
        self.assertEqual(
            tokenizer.calls[0][1],
            {"padding": False, "truncation": False, "add_special_tokens": False},
        )

        with patch("src.models.render_chat", return_value="rendered"):
            with self.assertRaisesRegex(core.IntegrityError, "context window"):
                runtime.audit_prompt_runtime(
                    tokenizer,
                    messages,
                    prompt_role="qa",
                    recorded_context_window_tokens=195,
                )

    def test_condition_fingerprints_bind_condition_ids_and_batches(self):
        base = {"schema": "test", "axis": {"value": 1}}
        execution_sha = core.canonical_json_sha256(base)
        plus_sha, plus_payload = runtime.condition_fingerprint(
            execution_sha,
            core.SPANS_PLUS_PASSAGES,
            self.source.question_ids,
            core.condition_batches(self.source, core.SPANS_PLUS_PASSAGES),
        )
        only_sha, _ = runtime.condition_fingerprint(
            execution_sha,
            core.PASSAGES_ONLY,
            self.source.both_gold_ids,
            core.condition_batches(self.source, core.PASSAGES_ONLY),
        )
        self.assertNotEqual(plus_sha, only_sha)
        changed = copy.deepcopy(plus_payload)
        changed["ordered_ids"] = list(reversed(changed["ordered_ids"]))
        self.assertNotEqual(plus_sha, core.canonical_json_sha256(changed))
        self.assertNotIn("output", json.dumps(plus_payload).lower())

    def test_execution_identity_hard_gates_model_gpu_census_batch_and_thinking(self):
        valid = self._execution_identity()
        runtime.validate_execution_identity(valid)
        mutations = {
            "model": ("model_revision", "wrong"),
            "precision": ("precision", "8bit"),
            "batch": ("batch_size", 3),
            "thinking": ("enable_thinking", True),
        }
        for label, (key, value) in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(valid)
                changed[key] = value
                with self.assertRaises(core.IntegrityError):
                    runtime.validate_execution_identity(changed)
        for key, value in (
            ("gpu_name", "A100"),
            ("gpu_compute_capability", "8.0"),
            ("gpu_driver_version", ""),
        ):
            changed = copy.deepcopy(valid)
            changed["gpu"][key] = value
            with self.assertRaises(core.IntegrityError):
                runtime.validate_execution_identity(changed)
        changed = copy.deepcopy(valid)
        changed["census"]["nominal_params"] += 1
        with self.assertRaises(core.IntegrityError):
            runtime.validate_execution_identity(changed)
        for missing in (
            "tokenizer_identity",
            "library_versions",
            "source_rendered_chat_manifest_sha256",
            "replay_git_commit",
        ):
            changed = copy.deepcopy(valid)
            changed.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(core.IntegrityError):
                runtime.validate_execution_identity(changed)
        changed = copy.deepcopy(valid)
        changed["census"]["quantized_fraction"] = 0.5
        with self.assertRaises(core.IntegrityError):
            runtime.validate_execution_identity(changed)
        changed = copy.deepcopy(valid)
        changed["library_versions"]["torch"] = "different-but-present"
        runtime.validate_execution_identity(changed)

    def test_post_load_validation_failure_unloads_model(self):
        from unittest.mock import patch

        resolved = {
            "resolved_model_revision": runtime.MODEL_REVISION,
            "resolved_tokenizer_revision": runtime.MODEL_REVISION,
        }
        with (
            patch("src.models.load_model", return_value=(object(), object())),
            patch("src.models.resolved_revision_metadata", return_value=resolved),
            patch(
                "src.models.validate_loaded_precision",
                side_effect=core.IntegrityError("injected census failure"),
            ),
            patch("src.models.unload") as unload,
            self.assertRaisesRegex(core.IntegrityError, "census failure"),
        ):
            runtime._load_replay_model(self.source, self._execution_identity()["gpu"])
        unload.assert_called_once_with()

    def test_execution_fingerprint_binds_static_contract_without_outputs(self):
        identity = self._execution_identity()
        sha, payload = runtime.build_execution_fingerprint_payload(self.source, identity)
        self.assertEqual(sha, core.canonical_json_sha256(payload))
        serialized = json.dumps(payload, sort_keys=True).lower()
        for required in (
            "frozen_trace_sha256",
            "passage_formatter",
            "prompt_contract",
            "history_policy",
            "condition_batch_manifests",
            "replay_code_sha256",
            "bootstrap",
        ):
            self.assertIn(required, serialized)
        self.assertNotIn("calls_sha256", serialized)
        self.assertNotIn("answers_sha256", serialized)
        self.assertEqual(payload["replay_git_commit"], identity["replay_git_commit"])
        self.assertEqual(payload["source"]["python_version"], None)
        self.assertEqual(
            payload["source"]["gpu"]["gpu_driver_version"],
            "580.159.04",
        )
        self.assertEqual(
            payload["source_rendered_chat_manifest"],
            {
                "sha256": identity["source_rendered_chat_manifest_sha256"],
                "count": 627,
            },
        )
        changed = copy.deepcopy(payload)
        changed["runtime_identity"]["library_versions"]["torch"] = "another"
        self.assertNotEqual(sha, core.canonical_json_sha256(changed))

    def test_resume_admits_only_certified_prefix_and_regenerates_full_orphan_batch(self):
        first = self._synthetic_batch(0)
        second = self._synthetic_batch(1)
        first_calls = tuple(
            self._synthetic_call(member, first) for member in first.members
        )
        first_cert = runtime.build_batch_certificate(
            first,
            first_calls,
            written_members=[member.key for member in first.members],
            identity=self._persistence_identity(),
        )
        second_calls = tuple(
            self._synthetic_call(member, second) for member in second.members
        )
        records = [self._model_load_record(), *first_calls, first_cert, second_calls[0]]
        audit = runtime.audit_resume(
            records,
            (first, second),
            identity=self._persistence_identity(),
        )
        self.assertEqual(len(audit.certified_calls), 4)
        self.assertNotIn(second.members[0].key, audit.certified_calls)
        self.assertEqual(
            audit.orphan_calls_by_batch[second.key],
            (second_calls[0],),
        )

        missing, certificate = runtime.reconcile_orphan_batch(
            second,
            (second_calls[0],),
            second_calls,
            identity=self._persistence_identity(),
        )
        self.assertEqual(missing, second_calls[1:])
        self.assertEqual(certificate["resume_regenerated_members"], 1)
        changed = list(copy.deepcopy(second_calls))
        changed[0]["raw_output"] = "different"
        with self.assertRaisesRegex(core.IntegrityError, "changed output field"):
            runtime.reconcile_orphan_batch(
                second,
                (second_calls[0],),
                changed,
                identity=self._persistence_identity(),
            )
        changed = list(copy.deepcopy(second_calls))
        changed[0]["consumer_input"] = {"frozen": False}
        with self.assertRaisesRegex(core.IntegrityError, "deterministic metadata"):
            runtime.reconcile_orphan_batch(
                second,
                (second_calls[0],),
                changed,
                identity=self._persistence_identity(),
            )

    def test_resume_rejects_duplicate_calls_and_wrong_certificate_hash(self):
        batch = self._synthetic_batch(0)
        calls = tuple(self._synthetic_call(member, batch) for member in batch.members)
        certificate = runtime.build_batch_certificate(
            batch,
            calls,
            written_members=[member.key for member in batch.members],
            identity=self._persistence_identity(),
        )
        with self.assertRaisesRegex(core.IntegrityError, "duplicate"):
            runtime.audit_resume(
                [self._model_load_record(), calls[0], calls[0]],
                (batch,),
                identity=self._persistence_identity(),
            )
        changed = copy.deepcopy(certificate)
        changed["canonical_treatment_batch_sha256"] = "0" * 64
        with self.assertRaisesRegex(core.IntegrityError, "hash|mismatch"):
            runtime.audit_resume(
                [self._model_load_record(), *calls, changed],
                (batch,),
                identity=self._persistence_identity(),
            )

    def test_jsonl_store_repairs_only_torn_tail_and_excludes_concurrent_writer(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            path = Path(raw) / "calls.jsonl.partial"
            path.write_bytes(b'{"record_type":"model_load"}\n{"torn":')
            store = runtime.JsonlStore(path).open("session-a")
            try:
                records = store.read_existing()
                self.assertEqual(records[0]["record_type"], "model_load")
                self.assertEqual(records[1]["record_type"], "store_repair")
                with self.assertRaisesRegex(RuntimeError, "exclusive output lock"):
                    runtime.JsonlStore(path).open("session-b")
            finally:
                store.close()

    def test_jsonl_lock_does_not_create_calls_file_before_data_open(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            directory = Path(raw)
            path = directory / "calls.jsonl.partial"
            store = runtime.JsonlStore(
                path,
                lock_path=directory / ".replay.lock",
            ).acquire_lock("sentinel-pending")
            try:
                self.assertFalse(path.exists())
                store.open_data("sentinel-passed")
                self.assertTrue(path.is_file())
            finally:
                store.close()

    def test_jsonl_store_rejects_symlinked_calls_leaf(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            directory = Path(raw)
            target = directory / "outside.jsonl"
            target.write_text("untouched\n", encoding="utf-8")
            path = directory / "calls.jsonl.partial"
            path.symlink_to(target)
            with self.assertRaisesRegex(core.IntegrityError, "regular file|safely open"):
                runtime.JsonlStore(path).open("session")
            self.assertEqual(target.read_text(encoding="utf-8"), "untouched\n")

    def test_meta_is_published_last_and_failed_final_rename_is_recoverable(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            output = Path(raw)
            for name in runtime.PUBLISHED_DATA_ARTIFACTS:
                (output / f"{name}.partial").write_bytes(f"{name}\n".encode())
            destinations = []
            validation = self._minimal_publication_validation()

            def fail_final_meta(source, destination):
                source = Path(source)
                destination = Path(destination)
                destinations.append(destination.name)
                if destination.name == "meta.json":
                    raise OSError("injected final meta rename failure")
                source.replace(destination)

            def validated_meta(directory, **_kwargs):
                return json.loads((Path(directory) / "meta.json").read_text())

            with (
                patch.object(runtime, "_validate_publication_data", return_value={}),
                patch.object(runtime, "validate_published_result", side_effect=validated_meta),
                self.assertRaisesRegex(OSError, "injected"),
            ):
                runtime.publish_meta_last(
                    output,
                    {"experiment": "test"},
                    validation_payload=validation,
                    replace_fn=fail_final_meta,
                )
            self.assertFalse((output / "meta.json").exists())
            self.assertTrue((output / "meta.json.partial").exists())
            self.assertEqual(destinations[-1], "meta.json")

            with (
                patch.object(runtime, "_validate_publication_data", return_value={}),
                patch.object(runtime, "validate_published_result", side_effect=validated_meta),
            ):
                meta = runtime.publish_meta_last(
                    output,
                    {"experiment": "test"},
                    validation_payload=validation,
                )
            self.assertEqual(meta["status"], "complete")
            self.assertTrue((output / "meta.json").is_file())
            self.assertFalse((output / "meta.json.partial").exists())

    def test_publication_rejects_symlinked_final_artifact(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            output = Path(raw)
            target = output / "outside"
            target.write_text("do-not-certify\n", encoding="utf-8")
            (output / runtime.PUBLISHED_DATA_ARTIFACTS[0]).symlink_to(target)
            for name in runtime.PUBLISHED_DATA_ARTIFACTS[1:]:
                (output / f"{name}.partial").write_text(name, encoding="utf-8")
            with self.assertRaisesRegex(core.IntegrityError, "regular file"):
                runtime._publication_artifact_hashes(output)
            self.assertEqual(target.read_text(encoding="utf-8"), "do-not-certify\n")

    def test_failed_temporary_meta_rename_cleans_retry_blocker(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            output = Path(raw)
            for name in runtime.PUBLISHED_DATA_ARTIFACTS:
                (output / f"{name}.partial").write_text(name, encoding="utf-8")

            def fail_replace(_source, _destination):
                raise OSError("injected first meta rename failure")

            with (
                patch.object(runtime, "_validate_publication_data", return_value={}),
                self.assertRaisesRegex(OSError, "first meta rename"),
            ):
                runtime.publish_meta_last(
                    output,
                    {"experiment": "test"},
                    validation_payload=self._minimal_publication_validation(),
                    replace_fn=fail_replace,
                )
            self.assertEqual(list(output.glob("meta.json.partial.*.tmp")), [])
            self.assertFalse((output / "meta.json.partial").exists())

    def test_corrupted_partial_meta_is_rejected_before_promotion(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            output = Path(raw)
            for name in runtime.PUBLISHED_DATA_ARTIFACTS:
                (output / f"{name}.partial").write_text(name, encoding="utf-8")
            validation = self._minimal_publication_validation()
            expected_base = {"experiment": "test"}
            certificate = {
                "experiment": "tampered",
                "status": "complete",
                "artifact_sha256": runtime._publication_artifact_hashes(output),
                "meta_payload_sha256": core.canonical_json_sha256(expected_base),
                "validation_contract_sha256": validation.contract_sha256,
            }
            (output / "meta.json.partial").write_text(
                json.dumps(certificate),
                encoding="utf-8",
            )
            with (
                patch.object(runtime, "_validate_publication_data", return_value={}),
                self.assertRaisesRegex(core.IntegrityError, "different replay payload"),
            ):
                runtime.publish_meta_last(
                    output,
                    expected_base,
                    validation_payload=validation,
                )
            self.assertFalse((output / "meta.json").exists())
            self.assertTrue((output / "meta.json.partial").is_file())
            self.assertTrue(
                all(
                    (output / f"{name}.partial").is_file()
                    for name in runtime.PUBLISHED_DATA_ARTIFACTS
                )
            )

    def test_successful_publication_removes_identical_duplicate_partials(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            output = Path(raw)
            for name in runtime.PUBLISHED_DATA_ARTIFACTS:
                payload = (name + "\n").encode()
                (output / name).write_bytes(payload)
                (output / f"{name}.partial").write_bytes(payload)

            def validated_meta(directory, **_kwargs):
                return json.loads((Path(directory) / "meta.json").read_text())

            with (
                patch.object(runtime, "_validate_publication_data", return_value={}),
                patch.object(
                    runtime,
                    "validate_published_result",
                    side_effect=validated_meta,
                ),
            ):
                meta = runtime.publish_meta_last(
                    output,
                    {"experiment": "test"},
                    validation_payload=self._minimal_publication_validation(),
                )
            self.assertEqual(meta["status"], "complete")
            self.assertTrue(
                all(
                    not (output / f"{name}.partial").exists()
                    for name in runtime.PUBLISHED_DATA_ARTIFACTS
                )
            )

    def test_semantic_publisher_rejects_arbitrary_artifacts_before_meta(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as raw:
            output = Path(raw)
            validation = self._minimal_publication_validation()
            (output / "manifest.json.partial").write_text(
                json.dumps(dict(validation.expected_manifest)),
                encoding="utf-8",
            )
            for name in ("calls.jsonl", "answers.jsonl", "summary.json"):
                (output / f"{name}.partial").write_text(name + "\n", encoding="utf-8")
            with self.assertRaises(core.IntegrityError):
                runtime.publish_meta_last(
                    output,
                    {"experiment": "invalid"},
                    validation_payload=validation,
                )
            self.assertFalse((output / "meta.json").exists())
            self.assertFalse((output / "meta.json.partial").exists())

    def test_complete_stream_requires_exact_preregistered_cardinalities(self):
        specs = []
        calls = [self._model_load_record()]
        for ordinal, size in enumerate((4, 4, 4, 4, 1, 4)):
            batch = self._synthetic_batch(
                ordinal,
                phase=runtime.SENTINEL_PHASE,
                size=size,
            )
            records = tuple(
                self._synthetic_call(member, batch) for member in batch.members
            )
            specs.append(batch)
            calls.extend(records)
            calls.append(
                runtime.build_batch_certificate(
                    batch,
                    records,
                    written_members=[member.key for member in batch.members],
                    identity=self._persistence_identity(),
                )
            )
        for condition in core.CONDITIONS:
            for ordinal, frozen in enumerate(
                core.condition_batches(self.source, condition)
            ):
                batch = self._synthetic_batch(
                    ordinal,
                    size=len(frozen.members),
                    condition=condition,
                )
                records = tuple(
                    self._synthetic_call(member, batch) for member in batch.members
                )
                specs.append(batch)
                calls.extend(records)
                calls.append(
                    runtime.build_batch_certificate(
                        batch,
                        records,
                        written_members=[member.key for member in batch.members],
                        identity=self._persistence_identity(),
                    )
                )

        def answer(condition, question_id):
            return {
                "record_type": "answer",
                "condition": condition,
                "condition_fingerprint_sha256": (
                    "7" * 64 if condition == core.SPANS_PLUS_PASSAGES else "8" * 64
                ),
                "question_id": question_id,
                "question": "question",
                "gold_answer": "gold",
                "retrieval_stratum": "hidden_bridge",
                "source_retrieval_all_gold": condition == core.PASSAGES_ONLY,
                "predicted_answer": "prediction",
                "final_answer_source": "summary_parsed",
                "final_answer_grounded": None,
                "final_answer_qa_step": None,
                "f1": 0.0,
                "em": 0.0,
                "qa_reverse_usable": {},
                "qa_last_executed": {},
                "qa_best_intermediate_oracle": {},
                "executed_steps": 1,
                "stop_reason": "plan_complete",
            }

        answers = [
            answer(core.SPANS_PLUS_PASSAGES, f"plus-{index}")
            for index in range(200)
        ] + [
            answer(core.PASSAGES_ONLY, f"only-{index}")
            for index in range(128)
        ]
        counts = runtime.validate_complete_streams(
            calls,
            answers,
            specs,
            identity=self._persistence_identity(),
        )
        self.assertEqual(counts, runtime.EXPECTED_COMPLETE_COUNTS)
        with self.assertRaisesRegex(core.IntegrityError, "cardinality"):
            runtime.validate_complete_streams(
                calls,
                answers[:-1],
                specs,
                identity=self._persistence_identity(),
            )


if __name__ == "__main__":
    unittest.main()
