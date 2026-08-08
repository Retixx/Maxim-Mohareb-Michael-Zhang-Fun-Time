"""Regression tests for deterministic Extractor consumer normalization."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch


import yaml

from src import agents, models, parsing, prompts, runner
from src.parsing import _VALIDATORS, parse_output
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
        self.assertEqual(record["verbatim_copy_rate"], 1.0)


class SoloValidatorContract(unittest.TestCase):
    """SPEC §4a: solo must be parsed against its OWN schema, never QA's.

    The two prompts ask for different shapes — solo for a bare
    {"answer": ...}, QA for {analysis, answer, success, rating}. `_VALIDATORS`
    once carried a duplicate "solo" key, so the entry that actually took effect
    depended on ordering. If solo were ever validated with QA's validator every
    solo call would fail to parse, the single-call baseline would score zero,
    and the multi-agent arm would win by default.
    """

    def test_solo_accepts_the_bare_answer_shape_its_prompt_asks_for(self):
        status, parsed = parse_output("solo", '{"answer": "Steven Spielberg"}', False)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed, {"answer": "Steven Spielberg"})

    def test_solo_validator_is_not_qas(self):
        # QA's validator would reject the bare shape; solo's must not.
        self.assertIsNot(_VALIDATORS["solo"], _VALIDATORS["qa"])
        qa_ok, _ = _VALIDATORS["qa"]({"answer": "Steven Spielberg"})
        solo_ok, _ = _VALIDATORS["solo"]({"answer": "Steven Spielberg"})
        self.assertFalse(qa_ok, "QA validator should require its richer schema")
        self.assertTrue(solo_ok, "solo validator must accept the bare shape")

    def test_qa_still_requires_its_full_schema(self):
        status, _ = parse_output("qa", '{"answer": "Steven Spielberg"}', False)
        self.assertNotEqual(status, "ok")
        status, parsed = parse_output(
            "qa",
            '{"analysis": "named directly", "answer": "Steven Spielberg",'
            ' "success": "yes", "rating": 9}',
            False,
        )
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["answer"], "Steven Spielberg")

    def test_every_pipeline_stage_role_has_exactly_one_validator(self):
        # A duplicate key is invisible in a dict literal, so assert the source
        # text declares each role once.
        source = Path(parsing.__file__).read_text(encoding="utf-8")
        table = source.split("_VALIDATORS = {", 1)[1].split("\n}", 1)[0]
        declared = re.findall(r'^\s*"([a-z_]+)":', table, flags=re.MULTILINE)
        self.assertEqual(
            sorted(declared), sorted(set(declared)),
            f"duplicate validator keys: {declared}",
        )
        for stage in prompts.PIPELINE_STAGES:
            role = prompts.STAGE_ROLE.get(stage, stage)
            self.assertIn(role, _VALIDATORS, f"{stage} -> {role} has no validator")


class ThinkingModeContract(unittest.TestCase):
    """Qwen3 must be driven in non-thinking mode (config `thinking_mode: false`).

    The chat template branches on `enable_thinking`: absent or True emits an
    OPEN '<think>' tag that forces reasoning; False emits a closed block. Every
    role's budget is 48-320 new tokens, which a reasoning block routinely
    exceeds — generation hits the cap, emits no JSON, parses `truncated`, and the
    answer degrades to "". That is EM/F1 near zero in all 32 arms, silently.

    Asserts the CONTRACT, not one implementation of it, so either an inline
    `enable_thinking=False` or a wrapper helper satisfies it.
    """

    QWEN3_BRANCH = (
        "{%- if enable_thinking is defined and enable_thinking is false %}"
        "{{- '<think>\\n\\n</think>\\n\\n' }}"
        "{%- else %}{{- '<think>\\n' }}{%- endif %}"
    )

    def test_template_branch_behaves_as_assumed(self):
        from jinja2 import Template
        t = Template(self.QWEN3_BRANCH)
        self.assertIn("</think>", t.render(enable_thinking=False))
        self.assertNotIn("</think>", t.render())

    def test_every_render_site_passes_the_flag(self):
        offenders = []
        for path in sorted(Path(models.__file__).parent.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            for number, line in enumerate(lines, start=1):
                if "apply_chat_template(" not in line:
                    continue
                if "enable_thinking" not in "\n".join(lines[number - 1:number + 8]):
                    offenders.append(f"{path.name}:{number}")
        self.assertEqual(offenders, [],
                         f"renders defaulting to chain-of-thought: {offenders}")

    def test_config_declares_non_thinking_and_is_a_literal_bool(self):
        config = yaml.safe_load(
            (Path(models.__file__).parents[1] / "config" / "experiment.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertIsInstance(config["thinking_mode"], bool)
        self.assertFalse(config["thinking_mode"])


class PreflightPadFieldContract(unittest.TestCase):
    """Every preflight pad field must exist in that role's prompt template.

    `pad_fields["extractor"]` was left at "paragraphs" after the v5->v6 prompt
    renamed the slot to "document", so the preflight padding loop raised
    KeyError on every Extractor stage of every manifest-driven run — after the
    model was resident and two stages had already consumed GPU time. Smoke runs
    never reach that code (`use_manifest` only), so nothing caught it.
    """

    def test_pad_fields_match_the_prompt_templates(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        table = source.split("pad_fields = {", 1)[1].split("}", 1)[0]
        pad_fields = dict(re.findall(r'"(\w+)":\s*"(\w+)"', table))
        self.assertTrue(pad_fields, "could not parse pad_fields")

        for role, field in pad_fields.items():
            slots = set(re.findall(r"\{(\w+)\}", prompts.USER_PROMPTS[role]))
            self.assertIn(
                field, slots,
                f"pad field {field!r} is not rendered by the {role!r} prompt "
                f"(slots: {sorted(slots)})",
            )

    def test_every_prompt_role_has_a_pad_field(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        table = source.split("pad_fields = {", 1)[1].split("}", 1)[0]
        pad_fields = dict(re.findall(r'"(\w+)":\s*"(\w+)"', table))
        missing = sorted(set(prompts.USER_PROMPTS) - set(pad_fields))
        self.assertEqual(missing, [], f"roles with no preflight pad field: {missing}")


class SpanSourceLabelContract(unittest.TestCase):
    """A total Extractor failure must not be telemetried as a salvage.

    `consumer_payload` is ALWAYS populated for the Extractor ({"spans": []}
    even when parsing and salvage both failed), so a two-way parsed/salvaged
    split labelled every hard failure "salvaged". That label flows into each QA
    record's consumer_payload_source, which is the field used to attribute
    downstream degradation to parsing versus salvage.
    """

    def test_parsed_salvaged_and_total_failure_are_distinguished(self):
        parsed = spans_of(
            {"parsed": {"spans": ["a"]}, "salvaged": None,
             "consumer_payload": {"spans": ["a"]}}
        )
        salvaged = spans_of(
            {"parsed": None, "salvaged": {"spans": ["b"]},
             "consumer_payload": {"spans": ["b"]}}
        )
        failed = spans_of(
            {"parsed": None, "salvaged": None, "consumer_payload": {"spans": []}}
        )
        self.assertEqual(parsed, (["a"], "normalized_parsed"))
        self.assertEqual(salvaged, (["b"], "normalized_salvaged"))
        self.assertEqual(failed, ([], "normalized_fallback"))
        self.assertNotEqual(
            failed[1], "normalized_salvaged",
            "a hard failure counted as a salvage inflates salvage contribution",
        )


if __name__ == "__main__":
    unittest.main()
