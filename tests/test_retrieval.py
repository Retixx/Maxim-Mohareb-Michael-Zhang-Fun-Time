"""Tests for open-domain retrieval (SPEC §3 revised).

Every assertion here guards a defect that would be silent in production: a
corrupted gold sentence index, drift in archived query-fusion helpers, a prompt
that changed shape with passages, or a BM25 implementation that drifted from its
reference formula.
"""

import unittest

from src import prompts, retrieval
from src.pipeline import build_answer_records, build_stage_calls, load_questions


def _passages(spec: dict[str, list[str]]) -> list[retrieval.Passage]:
    return [retrieval.Passage(t, s) for t, s in spec.items()]


class BM25Tests(unittest.TestCase):
    # Vocabulary is mostly doc-specific, so query terms have low document
    # frequency -- the regime real retrieval runs in, and the only one where
    # comparing IDF variants says anything.
    DOCS = {
        f"Doc {i}": [f"topic{i} concerns subject{i} and region{i % 11}",
                     f"a further note about theme{i % 17} and person{i % 23}"]
        for i in range(60)
    }
    QUERIES = ("topic7 subject7", "region3 theme14", "person5 topic41 region9")

    def test_scores_match_a_direct_transcription_of_the_formula(self):
        """Pins the vectorised index against a plain loop over the same maths.

        The CSC column-sum trick is where a scoring bug would actually hide, and
        it is what this catches. Equivalence to rank_bm25 is NOT claimed: that
        library floors negative IDFs from a different formula (see the module
        docstring), so it is a different scorer, not a reference implementation.
        """
        import math

        passages = _passages(self.DOCS)
        index = retrieval.BM25Index(passages)
        toks = [retrieval.tokenize(f"{p.title} {p.text}") for p in passages]
        n_docs = len(toks)
        avgdl = sum(len(t) for t in toks) / n_docs

        def naive(query: str) -> list[float]:
            terms = retrieval.tokenize(query)
            out = []
            for doc in toks:
                score = 0.0
                for t in terms:
                    tf = doc.count(t)
                    if not tf:
                        continue
                    df = sum(1 for d in toks if t in d)
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    denom = tf + retrieval.K1 * (
                        1 - retrieval.B + retrieval.B * len(doc) / avgdl
                    )
                    score += idf * tf * (retrieval.K1 + 1.0) / denom
                out.append(score)
            return out

        for query in self.QUERIES:
            want = naive(query)
            order = sorted(range(n_docs), key=lambda i: -want[i])[:5]
            got = index.search(query, 5)
            for a, b in zip(got, order):
                self.assertAlmostEqual(want[a], want[b], places=5,
                                       msg=f"score mismatch for {query!r}")

    def test_agreement_with_rank_bm25_is_recorded_not_assumed(self):
        """Pins the ACTUAL relationship to rank_bm25: same documents, own order.

        Both scorers select an identical candidate set — no document is reachable
        by one and not the other — but they rank within it differently, because
        the Lucene IDF and rank_bm25's floored IDF weight terms differently. That
        distinction is why src/retrieval.py must not claim equivalence, and why
        every k/hop-1 measurement was taken with this implementation rather than
        carried over from the rank_bm25 probes.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.skipTest("rank_bm25 not installed")

        passages = _passages(self.DOCS)
        index = retrieval.BM25Index(passages)
        titles = [p.title for p in passages]
        docs = [retrieval.tokenize(f"{p.title} {p.text}") for p in passages]
        reference = BM25Okapi(docs)

        for query in self.QUERIES:
            terms = set(retrieval.tokenize(query))
            # Compare only documents that actually contain a query term. Beyond
            # those every score is zero, so the tail is arbitrary tie-breaking in
            # both libraries and comparing it measures nothing.
            matched = sum(1 for d in docs if terms & set(d))
            got = index.search_titles(query, matched)
            want = reference.get_top_n(retrieval.tokenize(query), titles, n=matched)
            self.assertEqual(set(got), set(want),
                             f"candidate sets diverged for {query!r}")

    def test_search_returns_at_most_k_best_first(self):
        index = retrieval.BM25Index(_passages({
            "Alpha": ["alpha alpha alpha unique"],
            "Beta": ["alpha beta"],
            "Gamma": ["gamma only"],
        }))
        got = index.search_titles("alpha", 2)
        self.assertEqual(got[0], "Alpha")
        self.assertEqual(len(got), 2)

    def test_unknown_terms_return_empty_not_everything(self):
        index = retrieval.BM25Index(_passages({"A": ["hello"]}))
        self.assertEqual(index.search_titles("zzzznotaword", 5), [])

    def test_equal_scores_are_broken_by_corpus_order(self):
        passages = [
            retrieval.Passage(title=f"Doc {i}", sentences=["shared text"])
            for i in range(12)
        ]
        index = retrieval.BM25Index(passages)
        self.assertEqual(index.search("shared", 3), [0, 1, 2])

    def test_rejects_empty_and_duplicate_title_corpora(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrieval.BM25Index([])
        with self.assertRaisesRegex(ValueError, "duplicate passage title"):
            retrieval.BM25Index([
                retrieval.Passage("Same", ["a"]),
                retrieval.Passage("Same", ["b"]),
            ])


class CorpusTests(unittest.TestCase):
    def test_distractor_sentences_win_for_shared_titles(self):
        """402 real titles differ between configs; sent_id follows distractor.

        Taking fullwiki's split for a shared title would shift gold sentence
        indices and corrupt every extraction-accuracy number without erroring.
        """
        rows = {
            "distractor": [{"context": {
                "title": ["Shared", "OnlyD"],
                "sentences": [["d one", "d two"], ["only d"]],
            }}],
            "fullwiki": [{"context": {
                "title": ["Shared", "OnlyF"],
                "sentences": [["f one", "f two", "f three"], ["only f"]],
            }}],
        }

        def loader(_name, config, split=None, revision=None):
            return rows[config]

        corpus = retrieval.build_corpus(loader=loader)
        by_title = {p.title: p for p in corpus}
        self.assertEqual(by_title["Shared"].sentences, ["d one", "d two"])
        self.assertEqual(sorted(by_title), ["OnlyD", "OnlyF", "Shared"])

    def test_reversing_config_order_would_change_gold_split(self):
        """Guards the ordering itself, so a future 'tidy-up' cannot silently swap it."""
        rows = {
            "distractor": [{"context": {"title": ["Shared"], "sentences": [["d"]]}}],
            "fullwiki": [{"context": {"title": ["Shared"], "sentences": [["f"]]}}],
        }

        def loader(_name, config, split=None, revision=None):
            return rows[config]

        flipped = retrieval.build_corpus(
            configs=("fullwiki", "distractor"), loader=loader
        )
        self.assertEqual(flipped[0].sentences, ["f"])
        self.assertEqual(retrieval.build_corpus(loader=loader)[0].sentences, ["d"])


class FollowupQueryTests(unittest.TestCase):
    QUESTION = "The director of the film Polish-Russian War was born in what year?"

    def test_extracts_the_bridge_name_from_a_span(self):
        spans = ["Polish-Russian War is a 2009 film directed by Xawery Zulawski."]
        got = retrieval.followup_query(spans, self.QUESTION, ["Polish-Russian War"])
        self.assertIn("Xawery Zulawski", got)

    def test_drops_names_already_in_the_question(self):
        """Re-searching a name the question already gave is what hop 1 did."""
        spans = ["Polish-Russian War is a film."]
        self.assertEqual(
            retrieval.followup_query(spans, self.QUESTION, []), ""
        )

    def test_drops_names_hop1_already_retrieved(self):
        """Archived fusion helper must not re-query pages already in hand."""
        spans = ["Arthur's Magazine was first published in 1844."]
        got = retrieval.followup_query(
            spans, "Which was started first, Arthur's Magazine or First for Women?",
            ["Arthur's Magazine", "First for Women"],
        )
        self.assertEqual(got, "")

    def test_no_candidates_means_no_second_hop(self):
        hop = retrieval.retrieve(
            retrieval.BM25Index(_passages({"A": ["text"]})),
            self.QUESTION, spans=["nothing capitalised here at all"],
            hop1_titles=[],
        )
        self.assertFalse(hop["fired"])
        self.assertEqual(hop["titles"], [])

    def test_second_hop_excludes_passages_already_seen(self):
        index = retrieval.BM25Index(_passages({
            "Seen": ["Hidden Person profile"],
            "New A": ["Hidden Person biography"],
            "New B": ["Hidden Person career"],
            "New C": ["Hidden Person history"],
            "Noise": ["unrelated"],
        }))
        hop = retrieval.retrieve(
            index,
            "What year was the director born?",
            k=4,
            hop1=1,
            spans=["The director was Hidden Person."],
            hop1_titles=["Seen"],
        )
        self.assertTrue(hop["fired"])
        self.assertEqual(len(hop["titles"]), 3)
        self.assertNotIn("Seen", hop["titles"])

    def test_sentence_initial_stopwords_are_not_names(self):
        names = retrieval.candidate_names(["The film was directed by Xawery Zulawski."])
        self.assertNotIn("The", names)
        self.assertIn("Xawery Zulawski", names)

    def test_longer_names_rank_first(self):
        names = retrieval.candidate_names(["Xawery Zulawski met Paris."])
        self.assertLess(names.index("Xawery Zulawski"), names.index("Paris"))

    def test_conjunctions_do_not_join_names(self):
        """"and"/"for" as joiners produced one bogus phrase from two names."""
        self.assertEqual(
            retrieval.candidate_names(["Paris and Xawery Zulawski met."]),
            ["Xawery Zulawski", "Paris"],
        )
        self.assertNotIn(
            "Berlin and Rome for Sony",
            retrieval.candidate_names(["He worked in Berlin and Rome for Sony."]),
        )

    def test_internal_particles_still_join(self):
        self.assertIn(
            "University of Texas",
            retrieval.candidate_names(["She attended the University of Texas."]),
        )

    def test_unicode_tokens_and_names_are_preserved(self):
        self.assertEqual(retrieval.tokenize("Xawery Żuławski"), ["xawery", "żuławski"])
        self.assertIn(
            "Xawery Żuławski",
            retrieval.candidate_names(["It was directed by Xawery Żuławski."]),
        )

    def test_terminal_disambiguator_does_not_create_a_hidden_bridge(self):
        self.assertTrue(retrieval.title_is_mentioned(
            "Polish-Russian War (film)",
            "Who directed the film Polish-Russian War?",
        ))
        self.assertFalse(retrieval.title_is_mentioned(
            "Xawery Żuławski",
            "Who directed the film Polish-Russian War?",
        ))


class AnchoredFusionTests(unittest.TestCase):
    def test_fusion_preserves_seven_anchor_slots_and_adds_three_unique_task_slots(self):
        anchor = [f"A{i}" for i in range(10)]
        task = ["A0", "T0", "T1", "A1", "T2", "T3"]
        self.assertEqual(
            retrieval.fuse_rankings(anchor, task, k=10, anchor_k=7),
            ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "T0", "T1", "T2"],
        )

    def test_fusion_fills_unused_task_slots_from_anchor_depth(self):
        anchor = [f"A{i}" for i in range(10)]
        self.assertEqual(
            retrieval.fuse_rankings(anchor, ["A0", "A1"], k=10, anchor_k=7),
            anchor,
        )

    def test_fusion_never_exceeds_the_frozen_exposure_budget(self):
        got = retrieval.fuse_rankings(
            [f"A{i}" for i in range(30)],
            [f"T{i}" for i in range(30)],
            k=10,
            anchor_k=7,
        )
        self.assertEqual(len(got), 10)
        self.assertEqual(len(got), len(set(got)))


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.ctx = retrieval.RetrievalContext(
            _passages({f"T{i}": [f"sentence {i} a", f"sentence {i} b"]
                       for i in range(20)}),
            k=10, hop1=7,
        )

    def test_rejects_a_budget_that_leaves_no_second_hop(self):
        with self.assertRaises(ValueError):
            retrieval.RetrievalContext(_passages({"A": ["x"]}), k=10, hop1=10)

    def test_sentence_index_covers_only_what_was_seen(self):
        idx = self.ctx.sentence_index(["T1", "T2", "Missing"])
        self.assertEqual({s["title"] for s in idx}, {"T1", "T2"})
        self.assertEqual([s["sent_id"] for s in idx if s["title"] == "T1"], [0, 1])

    def test_fingerprint_binds_ordered_corpus_content(self):
        a = retrieval.RetrievalContext(
            [retrieval.Passage("T", ["alpha"])], k=2, hop1=1
        ).fingerprint()
        b = retrieval.RetrievalContext(
            [retrieval.Passage("T", ["beta"])], k=2, hop1=1
        ).fingerprint()
        self.assertIn("corpus_sha256", a)
        self.assertNotEqual(a["corpus_sha256"], b["corpus_sha256"])
        self.assertEqual(a["anchor_k"], 1)
        self.assertEqual(a["task_k"], 1)
        self.assertIn("anchored", a["query_policy"])

    def test_format_passages_is_byte_identical_to_the_frozen_renderer(self):
        """The prompt shape must not change along with the passages' provenance."""
        passages = self.ctx.passages(["T1", "T2"])
        self.assertEqual(
            retrieval.format_passages(passages),
            prompts.format_paragraphs([p.title for p in passages],
                                      [p.sentences for p in passages]),
        )


class StageWiringTests(unittest.TestCase):
    def setUp(self):
        self.ctx = retrieval.RetrievalContext(
            _passages({
                "Polish-Russian War (film)": [
                    "Polish-Russian War is a 2009 Polish film.",
                    "It was directed by Xawery Zulawski.",
                ],
                # Shares no term with the question, so a single query cannot
                # reach it -- exactly the hidden-bridge condition.
                "Xawery Zulawski": ["Educated at Lodz, graduating in 1996."],
                # Noise that matches the QUESTION strongly, so hop-1's top-7 is
                # saturated by it and the bridge page is genuinely out of reach
                # for a single query -- which is the condition under test.
                **{f"Noise {i}": [
                    f"polish russian war film director feature {i}"
                ] for i in range(30)},
            }),
            k=10, hop1=7,
        )
        self.q = {
            "question_id": "q1",
            "question": "Who directed the film Polish-Russian War?",
            "answer": "1971",
            "supporting_facts": {"title": ["Polish-Russian War (film)"], "sent_id": [1]},
            "retrieval_stratum": "hidden_bridge",
        }
        self.idx = {("q1", "planner", 0): {
            "parsed": {"sub_questions": [
                "Who directed it?",
                "When was that director born?",
                "Where was that director educated?",
            ]},
            "salvaged": None,
        }}

    def test_stages_that_read_passages_require_a_retriever(self):
        """SPEC §3 no longer hands anyone the dataset's ten paragraphs."""
        for stage in prompts.stages_for_role("extractor") + ("solo",):
            with self.assertRaises(ValueError, msg=stage):
                build_stage_calls(stage, [self.q], self.idx)

    def test_solo_retrieves_and_is_not_handed_gold(self):
        call = build_stage_calls("solo", [self.q], self.idx, retriever=self.ctx)[0]
        titles = call["consumer_input"]["retrieval"]["titles"]
        self.assertEqual(len(titles), 10)
        self.assertEqual(call["consumer_payload_source"], "retrieved")

        self.idx[("q1", "step_definer", 0)] = {
            "parsed": {"type": "question-answering", "task": "Who directed it?"}
        }
        multi = build_stage_calls(
            "extractor", [self.q], self.idx, retriever=self.ctx
        )[0]
        event = multi["consumer_input"]["retrieval"]
        self.assertEqual(event["titles"], titles)
        self.assertEqual(event["query_count"], 1)
        self.assertFalse(event["task_component_attempted"])

    def test_second_round_consumes_prior_qa_and_extractor_state(self):
        self.idx[("q1", "step_definer", 0)] = {
            "parsed": {
                "type": "question-answering",
                "task": "Who directed the film Polish-Russian War?",
            },
            "salvaged": None,
        }
        first = build_stage_calls("extractor", [self.q], self.idx, retriever=self.ctx)[0]
        self.assertIn("directed", first["consumer_input"]["retrieval"]["query"])
        self.assertEqual(len(first["consumer_input"]["retrieval"]["titles"]), 10)
        self.assertEqual(
            first["consumer_input"]["document_sentences"],
            self.ctx.index.get(first["consumer_input"]["document_title"]).sentences,
        )
        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["It was directed by Xawery Zulawski."]},
            "salvaged": None,
        }
        self.idx[("q1", "qa", 0)] = {
            "parsed": {"answer": "Xawery Zulawski", "success": "yes", "rating": 9},
            "salvaged": None,
        }
        step2 = build_stage_calls(
            "step_definer_step2", [self.q], self.idx, retriever=self.ctx
        )[0]
        self.assertIn("Xawery Zulawski", step2["fields"]["prior_state"])
        self.idx[("q1", "step_definer_step2", 1)] = {
            "parsed": {
                "type": "question-answering",
                "task": "When was that director born?",
            },
            "salvaged": None,
        }
        second = build_stage_calls(
            "extractor_step2", [self.q], self.idx, retriever=self.ctx
        )[0]
        got = second["consumer_input"]["retrieval"]
        self.assertEqual(got["step"], 2)
        self.assertEqual(got["anchor_query"], self.q["question"])
        self.assertIn("Xawery Zulawski", got["task_query"])
        self.assertEqual(got["grounded_prior_answers"], ["Xawery Zulawski"])
        self.assertEqual(got["query_count"], 2)
        self.assertEqual(
            got["titles"][:7],
            self.ctx.index.search_titles(self.q["question"], 10)[:7],
        )
        self.assertIn("Xawery Zulawski", got["titles"])

    def test_salvaged_step_task_keeps_its_query_when_type_is_missing(self):
        self.idx[("q1", "step_definer", 0)] = {
            "parsed": None,
            "salvaged": {"task": "Find the director named by the film page."},
            "parse_status": "schema_mismatch",
        }
        call = build_stage_calls(
            "extractor", [self.q], self.idx, retriever=self.ctx
        )[0]
        self.assertEqual(
            call["consumer_input"]["step_definition"],
            {
                "type": "question-answering",
                "task": "Find the director named by the film page.",
            },
        )
        self.assertEqual(call["consumer_payload_source"], "salvaged")
        self.assertEqual(
            self.idx[("q1", "step_definer", 0)]["parse_status"],
            "schema_mismatch",
        )

    def test_missing_step_task_fallback_carries_feedback_into_next_query(self):
        self.idx[("q1", "qa", 0)] = {
            "parsed": {
                "answer": "Xawery Zulawski", "success": "yes", "rating": 9,
            },
        }
        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["It was directed by Xawery Zulawski."]},
        }
        self.idx[("q1", "step_definer_step2", 1)] = {
            "parsed": None, "salvaged": None, "parse_status": "invalid_json",
        }
        call = build_stage_calls(
            "extractor_step2", [self.q], self.idx, retriever=self.ctx
        )[0]
        query = call["consumer_input"]["retrieval"]["query"]
        self.assertIn("When was that director born?", query)
        self.assertIn("Xawery Zulawski", query)
        self.assertIn(self.q["question"], query)
        self.assertEqual(call["consumer_payload_source"], "fallback")

    def test_unsupported_qa_guess_is_withheld_from_state_and_targeted_retrieval(self):
        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["It was directed by Xawery Zulawski."]},
        }
        self.idx[("q1", "qa", 0)] = {
            "parsed": {"answer": "Bay of Biscay", "success": "yes", "rating": 10},
        }
        step2 = build_stage_calls("step_definer_step2", [self.q], self.idx)[0]
        self.assertNotIn("Bay of Biscay", step2["fields"]["prior_state"])
        self.assertIn("withheld", step2["fields"]["prior_state"])

        self.idx[("q1", "step_definer_step2", 1)] = {
            "parsed": {
                "type": "question-answering",
                "task": "Which coast is Bay of Biscay on?",
            }
        }
        call = build_stage_calls(
            "extractor_step2", [self.q], self.idx, retriever=self.ctx
        )[0]
        event = call["consumer_input"]["retrieval"]
        self.assertEqual(event["grounded_prior_answers"], [])
        self.assertFalse(event["task_component_attempted"])
        self.assertNotIn("Bay of Biscay", event["anchor_query"])

    def test_qa_keeps_extractor_evidence_grouped_by_document(self):
        self.idx[("q1", "step_definer", 0)] = {
            "parsed": {
                "type": "question-answering",
                "task": "Who directed Polish-Russian War?",
            },
        }
        event = {
            "step": 1,
            "task_type": "question-answering",
            "attempted": True,
            "query": "Who directed Polish-Russian War?",
            "titles": ["Film page", "Director page"],
        }
        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["Directed by Xawery.", "Directed by Xawery."]},
            "consumer_input": {
                "retrieval": event,
                "document_rank": 0,
                "document_title": "Film page",
            },
        }
        self.idx[("q1", "extractor", 1)] = {
            "parsed": {"spans": ["Directed by Xawery.", "Xawery was a Polish director."]},
            "consumer_input": {
                "retrieval": event,
                "document_rank": 1,
                "document_title": "Director page",
            },
        }
        self.idx[("q1", "extractor", 2)] = {
            "parsed": {"spans": []},
            "consumer_input": {
                "retrieval": event,
                "document_rank": 2,
                "document_title": "Empty page",
            },
        }

        class DirectLookupOnly(dict):
            def items(self):
                raise AssertionError("feedback paths must not scan the full index")

        self.idx = DirectLookupOnly(self.idx)
        call = build_stage_calls("qa", [self.q], self.idx)[0]
        blocks = call["consumer_input"]["evidence_blocks"]
        self.assertEqual(
            [(b["document_title"], b["document_rank"]) for b in blocks],
            [("Film page", 0), ("Director page", 1), ("Empty page", 2)],
        )
        self.assertEqual(blocks[0]["spans"], ["Directed by Xawery."])
        self.assertEqual(
            blocks[1]["spans"],
            ["Directed by Xawery.", "Xawery was a Polish director."],
        )
        self.assertEqual(
            blocks[1]["prompt_spans"], ["Xawery was a Polish director."]
        )
        self.assertFalse(blocks[2]["included_in_prompt"])
        self.assertIn("Film page", call["fields"]["evidence"])
        self.assertIn("Director page", call["fields"]["evidence"])
        self.assertNotIn("Empty page", call["fields"]["evidence"])
        self.assertNotIn("no supporting text found", call["fields"]["evidence"])
        self.assertEqual(call["fields"]["evidence"].count("Directed by Xawery."), 1)

        self.idx[("q1", "qa", 0)] = {
            "stage": "qa",
            "parsed": {"answer": "Xawery", "success": "yes", "rating": 9},
            "consumer_input": call["consumer_input"],
        }
        self.idx[("q1", "plan_summary", 0)] = {
            "stage": "plan_summary",
            "parsed": {"output": "Successful", "answer": "Xawery", "score": 9},
            "consumer_input": {"stop_reason": "plan_complete"},
        }
        answer = build_answer_records([self.q], self.idx, "baseline")[0]
        self.assertEqual(answer["retrieved_titles"], ["Film page", "Director page"])

    def test_zero_result_query_survives_on_qa_and_answer_telemetry(self):
        task = "zzzznotaword"
        question = dict(self.q, question=task)
        self.idx[("q1", "step_definer", 0)] = {
            "parsed": {"type": "question-answering", "task": task},
        }
        self.assertEqual(
            build_stage_calls("extractor", [question], self.idx, self.ctx), []
        )
        qa_call = build_stage_calls("qa", [question], self.idx, self.ctx)[0]
        event = qa_call["consumer_input"]["retrieval"]
        self.assertTrue(event["attempted"])
        self.assertEqual(event["query"], task)
        self.assertEqual(event["query_count"], 1)
        self.assertEqual(event["titles"], [])

        self.idx[("q1", "qa", 0)] = {
            "stage": "qa",
            "parsed": {"answer": "unknown", "success": "no", "rating": 0},
            "consumer_input": qa_call["consumer_input"],
        }
        self.idx[("q1", "plan_summary", 0)] = {
            "stage": "plan_summary",
            "parsed": {"output": "Unsuccessful", "answer": "", "score": 0},
            "consumer_input": {"stop_reason": "semantic_inability"},
        }
        answer = build_answer_records([question], self.idx, "baseline")[0]
        self.assertEqual(answer["retrieval_query_count"], 1)
        self.assertEqual(answer["retrieval_step_count"], 1)
        self.assertEqual(answer["retrieval_zero_result_query_count"], 1)
        self.assertEqual(answer["retrieval_aggregate_step_count"], 0)
        self.assertEqual(answer["retrieval_events"], [event])

        self.assertEqual(
            qa_call["fields"]["evidence"], "(no evidence collected)"
        )
        self.assertFalse(qa_call["consumer_input"]["evidence_blocks"][0]["included_in_prompt"])
    def test_rounds_are_question_dependent_and_keep_agent_order(self):
        one_step = dict(self.q, question_id="one")
        three_step = dict(self.q, question_id="three")
        idx = {
            ("one", "planner", 0): {"parsed": {"sub_questions": ["Only step"]}},
            ("three", "planner", 0): {"parsed": {"sub_questions": ["A", "B", "C"]}},
        }
        expected = ["planner"]
        for step in range(prompts.MAX_PLAN_STEPS):
            expected.extend(prompts.stage_for(role, step) for role in prompts.ROUND_ROLES)
        expected.extend(["plan_summary", "solo"])
        self.assertEqual(prompts.PIPELINE_STAGES, expected)
        # Step 2 cannot run until step 1 QA has appended state.
        self.assertEqual(build_stage_calls("step_definer_step2", [one_step, three_step], idx), [])
        idx[("three", "qa", 0)] = {
            "parsed": {"answer": "bridge", "success": "yes", "rating": 8}
        }
        later = build_stage_calls("step_definer_step2", [one_step, three_step], idx)
        self.assertEqual([call["question_id"] for call in later], ["three"])

    def test_repeated_stages_map_to_four_conceptual_agents(self):
        self.assertEqual(prompts.role_for("step_definer_step3"), "step_definer")
        self.assertEqual(prompts.role_for("extractor_step2"), "extractor")
        self.assertEqual(prompts.role_for("qa_step3"), "qa")
        self.assertEqual(prompts.role_for("plan_summary"), "step_definer")

    def test_aggregate_route_skips_retrieval_and_extractor(self):
        self.idx[("q1", "step_definer", 0)] = {
            "parsed": {"type": "aggregate", "task": "Compare the known dates."}
        }
        self.assertEqual(build_stage_calls("extractor", [self.q], self.idx, self.ctx), [])
        qa_call = build_stage_calls("qa", [self.q], self.idx, self.ctx)[0]
        self.assertEqual(qa_call["consumer_input"]["task_type"], "aggregate")
        self.assertEqual(qa_call["consumer_input"]["retrieval"], {
            "step": 1,
            "task_type": "aggregate",
            "attempted": False,
            "query": None,
            "query_count": 0,
            "components": [],
            "titles": [],
        })

    def test_aggregate_qa_renders_only_grounded_prior_answers(self):
        self.idx[("q1", "qa", 0)] = {
            "parsed": {"answer": "Bay of Biscay", "success": "yes", "rating": 10},
        }
        self.idx[("q1", "step_definer_step2", 1)] = {
            "parsed": {"type": "aggregate", "task": "Use the prior answer."},
        }
        unsupported = build_stage_calls("qa_step2", [self.q], self.idx)[0]
        self.assertEqual(
            unsupported["fields"]["evidence"], "(no evidence collected)"
        )
        self.assertFalse(
            unsupported["consumer_input"]["evidence_blocks"][0][
                "included_in_prompt"
            ]
        )

        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["It was directed by Xawery Zulawski."]},
        }
        self.idx[("q1", "qa", 0)]["parsed"]["answer"] = "Xawery Zulawski"
        grounded = build_stage_calls("qa_step2", [self.q], self.idx)[0]
        self.assertIn("Prior step answer: Xawery Zulawski", grounded["fields"]["evidence"])
        self.assertTrue(
            grounded["consumer_input"]["evidence_blocks"][0][
                "included_in_prompt"
            ]
        )

    def test_semantic_failure_stops_later_steps_and_summarizes(self):
        self.idx[("q1", "qa", 0)] = {
            "parsed": {"answer": "unknown", "success": "no", "rating": 0}
        }
        self.assertEqual(build_stage_calls("step_definer_step2", [self.q], self.idx), [])
        summary = build_stage_calls("plan_summary", [self.q], self.idx)[0]
        self.assertEqual(summary["consumer_input"]["stop_reason"], "semantic_inability")


class StratifierTests(unittest.TestCase):
    def test_stratum_is_derived_from_question_text_and_gold_titles(self):
        rows = [
            {"id": "a", "question": "Which started first, Arthur's Magazine or Vogue?",
             "answer": "x", "level": "hard", "type": "comparison",
             "supporting_facts": {"title": ["Arthur's Magazine", "Vogue"],
                                  "sent_id": [0, 0]},
             "context": {"title": ["Arthur's Magazine", "Vogue"],
                         "sentences": [["s"], ["s"]]}},
            {"id": "b", "question": "Who directed Polish-Russian War?",
             "answer": "y", "level": "hard", "type": "bridge",
             "supporting_facts": {"title": ["Polish-Russian War", "Xawery Zulawski"],
                                  "sent_id": [0, 0]},
             "context": {"title": ["Polish-Russian War", "Xawery Zulawski"],
                         "sentences": [["s"], ["s"]]}},
        ]

        class FakeDS(list):
            def __getitem__(self, i):
                return list.__getitem__(self, i)

        import src.pipeline as P
        real = P.load_questions.__globals__.get("load_dataset")
        try:
            import datasets
            datasets.load_dataset = lambda *a, **k: FakeDS(rows)
            got = {q["question_id"]: q for q in load_questions(2, seed=0)}
        finally:
            if real is not None:
                P.load_questions.__globals__["load_dataset"] = real
        self.assertEqual(got["a"]["retrieval_stratum"], "fully_named")
        self.assertEqual(got["b"]["retrieval_stratum"], "hidden_bridge")
        self.assertEqual(got["b"]["hidden_gold_titles"], ["Xawery Zulawski"])


if __name__ == "__main__":
    unittest.main()
