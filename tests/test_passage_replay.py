import copy
import shutil
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from clean_room import passage_replay_core as core
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


if __name__ == "__main__":
    unittest.main()
