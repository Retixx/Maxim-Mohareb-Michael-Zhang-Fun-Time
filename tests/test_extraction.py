"""Regression tests for deterministic Extractor consumer normalization."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src import agents
from src.extraction import normalize_spans
from src.pipeline import spans_of


class SentenceNormalizationTests(unittest.TestCase):
    SENTENCES = [
        "Apollo 8 launched on December 21, 1968.",
        "It was the first crewed spacecraft to orbit the Moon.",
        "The crew returned safely to Earth.",
        "Frank Borman commanded the mission.",
    ]

    def test_exact_sentence_is_preserved(self):
        spans, telemetry = normalize_spans([self.SENTENCES[0]], self.SENTENCES)
        self.assertEqual(spans, [self.SENTENCES[0]])
        self.assertEqual(telemetry["accepted_input_count"], 1)
        self.assertEqual(telemetry["rejected_input_count"], 0)

    def test_unique_long_fragment_expands_to_exact_source_sentence(self):
        spans, telemetry = normalize_spans(
            ["first crewed spacecraft to orbit the Moon"], self.SENTENCES
        )
        self.assertEqual(spans, [self.SENTENCES[1]])
        self.assertEqual(telemetry["normalization_modes"], {"fragment": 1})

    def test_multi_sentence_echo_is_rejected(self):
        echo = " ".join(self.SENTENCES[:3])
        spans, telemetry = normalize_spans([echo], self.SENTENCES)
        self.assertEqual(spans, [])
        self.assertEqual(telemetry["rejection_reasons"], {"multiple_sentences": 1})

    def test_fragment_matching_two_source_sentences_is_rejected_as_ambiguous(self):
        source = [
            "The city is located beside the same unusually named Blue River delta.",
            "The port is located beside the same unusually named Blue River delta.",
        ]
        spans, telemetry = normalize_spans(
            ["located beside the same unusually named Blue River delta"], source
        )
        self.assertEqual(spans, [])
        self.assertEqual(telemetry["rejection_reasons"], {"ambiguous_sentence": 1})

    def test_output_is_deduplicated_and_capped_at_three(self):
        spans, telemetry = normalize_spans(
            [self.SENTENCES[0], self.SENTENCES[0], *self.SENTENCES[1:]],
            self.SENTENCES,
        )
        self.assertEqual(spans, self.SENTENCES[:3])
        self.assertEqual(telemetry["rejection_reasons"], {
            "duplicate_sentence": 1,
            "over_limit": 1,
        })

    def test_downstream_prefers_normalized_payload_without_hiding_parse_source(self):
        record = {
            "parsed": {"spans": ["first. second."]},
            "salvaged": None,
            "consumer_payload": {"spans": ["first."]},
        }
        spans, source = spans_of(record)
        self.assertEqual(spans, ["first."])
        self.assertEqual(source, "normalized_parsed")



class AgentNormalizationIntegrationTests(unittest.TestCase):
    def test_raw_payload_is_preserved_while_consumer_payload_is_normalized(self):
        source = [
            "Apollo 8 launched on December 21, 1968.",
            "It was the first crewed spacecraft to orbit the Moon.",
        ]
        raw_span = " ".join(source)
        generated = [{
            "raw_output": json.dumps({"spans": [raw_span]}),
            "hit_token_cap": False,
            "prompt_tokens": 20,
            "output_tokens": 20,
            "latency_s": 0.01,
        }]
        call = {
            "question_id": "q1",
            "call_index": 0,
            "fields": {
                "document": "[1] Apollo 8: " + raw_span,
                "sub_question": "When did Apollo 8 launch?",
            },
            "consumer_input": {
                "document_title": "Apollo 8",
                "document_sentences": source,
            },
        }
        with patch("src.agents.generate_batch", return_value=generated):
            record = agents.run_calls(
                object(), object(), "extractor", [call], "fp16", "run"
            )[0]

        self.assertEqual(record["parsed"], {"spans": [raw_span]})
        self.assertEqual(record["consumer_payload"], {"spans": []})
        self.assertEqual(
            record["extractor_normalization"]["rejection_reasons"],
            {"multiple_sentences": 1},
        )
        self.assertFalse(record["protocol_ok"])


if __name__ == "__main__":
    unittest.main()
