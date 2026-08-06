"""Tests for open-domain retrieval (SPEC §3 revised).

Every assertion here guards a defect that would be silent in production: a
corrupted gold sentence index, a second hop that fires when it cannot help, a
prompt that changed shape along with the passages, or a BM25 implementation that
drifted from the reference scores it claims to reproduce.
"""

import unittest

from src import prompts, retrieval
from src.pipeline import build_stage_calls, load_questions


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
        """The fully-named case: every needed page is in hand, so do not fire.

        Without this the second hop spends budget re-finding held pages and costs
        0.172 all-gold on the 20% of questions that name everything.
        """
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
            "parsed": {"sub_questions": ["Who directed it?"]}, "salvaged": None,
        }}

    def test_stages_that_read_passages_require_a_retriever(self):
        """SPEC §3 no longer hands anyone the dataset's ten paragraphs."""
        for stage in ("extractor", prompts.EXTRACTOR_HOP2, "solo"):
            with self.assertRaises(ValueError, msg=stage):
                build_stage_calls(stage, [self.q], self.idx)

    def test_solo_retrieves_and_is_not_handed_gold(self):
        call = build_stage_calls("solo", [self.q], self.idx, retriever=self.ctx)[0]
        titles = call["consumer_input"]["retrieval"]["titles"]
        self.assertEqual(len(titles), 10)
        self.assertEqual(call["consumer_payload_source"], "retrieved")

    def test_hop2_searches_the_name_hop1_revealed(self):
        calls = build_stage_calls("extractor", [self.q], self.idx, retriever=self.ctx)
        hop1_titles = calls[0]["consumer_input"]["retrieval"]["titles"]
        self.assertEqual(len(hop1_titles), 7)

        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["It was directed by Xawery Zulawski."]},
            "salvaged": None,
        }
        hop2 = build_stage_calls(
            prompts.EXTRACTOR_HOP2, [self.q], self.idx, retriever=self.ctx
        )[0]
        got = hop2["consumer_input"]["retrieval"]
        self.assertTrue(got["fired"])
        self.assertIn("Xawery Zulawski", got["query"])
        self.assertIn("Xawery Zulawski", got["titles"])

    def test_hop2_falls_back_to_held_back_depth_when_it_cannot_help(self):
        """A fully-named question must not be punished for the machinery."""
        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["lowercase text with no names at all"]},
            "salvaged": None,
        }
        hop2 = build_stage_calls(
            prompts.EXTRACTOR_HOP2, [self.q], self.idx, retriever=self.ctx
        )[0]
        got = hop2["consumer_input"]["retrieval"]
        self.assertFalse(got["fired"])
        # The budget hop-1 held back is spent on more depth instead of wasted.
        self.assertEqual(len(got["titles"]), self.ctx.k - self.ctx.hop1)

    def test_both_hops_use_the_same_step_definition(self):
        """The two passes must differ ONLY in which passages they see."""
        self.idx[("q1", "step_definer", 0)] = {
            "parsed": {"target_entity": "director", "search_terms": ["directed by"]},
            "salvaged": None,
        }
        self.idx[("q1", "extractor", 0)] = {
            "parsed": {"spans": ["It was directed by Xawery Zulawski."]},
            "salvaged": None,
        }
        a = build_stage_calls("extractor", [self.q], self.idx, retriever=self.ctx)[0]
        b = build_stage_calls(
            prompts.EXTRACTOR_HOP2, [self.q], self.idx, retriever=self.ctx
        )[0]
        for field in ("sub_question", "target_entity", "search_terms"):
            self.assertEqual(a["fields"][field], b["fields"][field], field)
        self.assertNotEqual(a["fields"]["paragraphs"], b["fields"]["paragraphs"])


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
