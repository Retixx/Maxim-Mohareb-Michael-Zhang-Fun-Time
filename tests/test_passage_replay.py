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
        return {
            (condition, self.qid, record["stage"], record["call_index"]): record
            for record in step_records
        }

    def test_aggregate_uses_treated_grounded_answers_without_retrieval(self):
        treated_index = self._index(
            [self._treated(0, "Motörhead"), self._treated(1, "ImpossibleValue")]
        )
        call = core.build_treated_qa_call(
            self.source,
            self.aggregate_qa,
            core.SPANS_PLUS_PASSAGES,
            treated_index,
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
        plus_index = {}
        only_index = {}
        both_gold = set(self.source.both_gold_ids)
        for record in core.source_qa_records(self.source):
            key = (record["question_id"], record["stage"], record["call_index"])
            plus_index[(core.SPANS_PLUS_PASSAGES, *key)] = record
            if record["question_id"] in both_gold:
                only_index[(core.PASSAGES_ONLY, *key)] = record
        summaries = {
            record["question_id"]: record
            for record in core.source_summary_records(self.source)
        }
        plus = core.build_condition_result(
            self.source,
            core.SPANS_PLUS_PASSAGES,
            plus_index,
            summaries,
        )
        only = core.build_condition_result(
            self.source,
            core.PASSAGES_ONLY,
            only_index,
            {qid: summaries[qid] for qid in self.source.both_gold_ids},
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
        self.assertIn("reverse_usable", report["qa_level"][core.SPANS_PLUS_PASSAGES])
        self.assertIn("success_drift", report["qa_level"][core.SPANS_PLUS_PASSAGES])
        self.assertIn("right_censoring", report["qa_level"][core.SPANS_PLUS_PASSAGES])


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

    def test_sentinel_selects_exactly_21_source_members(self):
        batches = runtime.sentinel_batches(self.source)
        self.assertEqual([len(batch.members) for batch in batches], [4, 4, 4, 4, 1, 4])
        self.assertEqual(sum(len(batch.members) for batch in batches), 21)
        self.assertEqual(
            [batch.stage for batch in batches],
            ["qa", "qa_step2", "qa_step3", "qa_step4", "qa_step5", "plan_summary"],
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
        valid = {
            "model_id": runtime.MODEL_ID,
            "model_revision": runtime.MODEL_REVISION,
            "tokenizer_revision": runtime.MODEL_REVISION,
            "precision": runtime.PRECISION,
            "census": dict(runtime.EXPECTED_CENSUS),
            "gpu": {
                "gpu_available": True,
                "gpu_name": runtime.EXPECTED_GPU_NAME,
                "gpu_compute_capability": runtime.EXPECTED_GPU_COMPUTE_CAPABILITY,
            },
            "batch_size": 4,
            "enable_thinking": False,
        }
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
        ):
            changed = copy.deepcopy(valid)
            changed["gpu"][key] = value
            with self.assertRaises(core.IntegrityError):
                runtime.validate_execution_identity(changed)
        changed = copy.deepcopy(valid)
        changed["census"]["nominal_params"] += 1
        with self.assertRaises(core.IntegrityError):
            runtime.validate_execution_identity(changed)

    def test_execution_fingerprint_binds_static_contract_without_outputs(self):
        identity = {
            "model_id": runtime.MODEL_ID,
            "model_revision": runtime.MODEL_REVISION,
            "tokenizer_revision": runtime.MODEL_REVISION,
            "precision": runtime.PRECISION,
            "census": dict(runtime.EXPECTED_CENSUS),
            "gpu": {
                "gpu_available": True,
                "gpu_name": runtime.EXPECTED_GPU_NAME,
                "gpu_compute_capability": runtime.EXPECTED_GPU_COMPUTE_CAPABILITY,
            },
            "batch_size": 4,
            "enable_thinking": False,
            "tokenizer_identity": {"snapshot_sha256": "a" * 64},
            "library_versions": {"torch": "different-is-warning-only"},
            "source_rendered_prompt_manifest_sha256": "b" * 64,
        }
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
        changed = copy.deepcopy(payload)
        changed["runtime_identity"]["library_versions"]["torch"] = "another"
        self.assertNotEqual(sha, core.canonical_json_sha256(changed))


if __name__ == "__main__":
    unittest.main()
