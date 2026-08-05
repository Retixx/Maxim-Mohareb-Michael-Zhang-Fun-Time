import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import prompts
from src.pipeline import (
    build_answer_records,
    build_stage_calls,
    load_id_manifest,
)
from src.runner import (
    JsonlStore,
    _batch_hash,
    _certified_batch_ids,
    _content_hash,
    _derived_config_matches_lock,
    _run_stage,
    result_slug,
)


ROOT = Path(__file__).resolve().parents[1]


class MemoryStore:
    def __init__(self):
        self.records = []

    def write(self, records):
        self.records.extend(records)

    def flush(self):
        pass

    def durable_flush(self):
        pass


class ExecutionIntegrityTests(unittest.TestCase):
    def test_frozen_manifest_and_exclusions_validate(self):
        ids, meta = load_id_manifest(
            ROOT / "config/manifests/final_n1500_seed20260805.json"
        )
        self.assertEqual(len(ids), 1500)
        self.assertEqual(
            meta["question_ids_sha256"],
            "5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af",
        )
        excluded = meta["exclusions"]["question_ids"]
        self.assertEqual(len(excluded), 3031)
        self.assertFalse(set(ids) & set(excluded))

    def test_qa_consumes_salvaged_extractor_payload(self):
        question = {
            "question_id": "q1", "question": "Q?", "answer": "A",
            "paragraphs": "[1] T: supporting sentence", "level": "easy",
            "type": "bridge", "supporting_facts": {"title": [], "sent_id": []},
            "sentence_index": [],
        }
        idx = {
            ("q1", "planner", 0): {
                "parsed": {"sub_questions": ["SQ?"]}, "salvaged": None,
            },
            ("q1", "extractor", 0): {
                "parsed": None,
                "salvaged": {"spans": ["supporting sentence"]},
            },
        }
        call = build_stage_calls("qa", [question], idx)[0]
        self.assertEqual(call["consumer_payload_source"], "salvaged")
        self.assertIn("supporting sentence", call["fields"]["evidence"])
        self.assertEqual(
            call["consumer_input"]["evidence_blocks"][0]["spans"],
            ["supporting sentence"],
        )

    def test_solo_answer_has_no_fake_subquestions_or_evidence_metric(self):
        question = {
            "question_id": "q1", "question": "Q?", "answer": "A",
            "paragraphs": "[1] T: A", "level": "easy", "type": "bridge",
            "supporting_facts": {"title": ["T"], "sent_id": [0]},
            "sentence_index": [],
        }
        idx = {("q1", "solo", 0): {
            "stage": "solo", "parsed": {"answer": "A"}, "salvaged": None,
        }}
        answer = build_answer_records([question], idx, "single_fp16", "manifest")[0]
        self.assertIsNone(answer["n_sub_questions"])
        self.assertEqual(answer["evidence_status"], "not_applicable")
        self.assertEqual(answer["question_manifest_sha256"], "manifest")

    def test_partial_resume_regenerates_canonical_neighbors(self):
        calls = [
            {"question_id": f"q{i}", "call_index": 0, "fields": {}}
            for i in range(4)
        ]
        idx = {("q0", "qa", 0): {"question_id": "q0", "stage": "qa", "call_index": 0}}
        generated = []

        def fake_run_calls(_model, _tok, role, chunk, precision, run_id, **kwargs):
            generated.append([c["question_id"] for c in chunk])
            records = [{
                "record_type": "agent_call", "run_id": run_id,
                "question_id": c["question_id"], "stage": role,
                "call_index": c["call_index"],
            } for c in chunk]
            batch = {
                "record_type": "batch", "stage": role,
                "batch_id": kwargs["batch_id"], "phase": "scored", "oom": False,
                "members": [
                    {"question_id": c["question_id"], "call_index": c["call_index"]}
                    for c in chunk
                ],
                "batch_size_requested": kwargs["batch_size"],
                "batch_ordinal": kwargs["batch_ordinal"],
                "config_fingerprint": kwargs["config_fingerprint"],
                "question_manifest_sha256": kwargs["question_manifest_sha256"],
            }
            return records, batch

        store = MemoryStore()
        with patch("src.runner.agents.run_calls", side_effect=fake_run_calls):
            _run_stage(
                object(), object(), "qa", calls, "fp16", "r", 2, 2,
                store, idx, model_id="m", config_fingerprint="fp",
                model_revision="rev", tokenizer_revision="rev",
                question_manifest_sha256="manifest", certified_batch_ids=set(),
            )
        self.assertEqual(generated, [["q0", "q1"], ["q2", "q3"]])
        written_calls = [r["question_id"] for r in store.records if r.get("record_type") == "agent_call"]
        self.assertEqual(written_calls, ["q1", "q2", "q3"])
        certificates = [r for r in store.records if r.get("record_type") == "batch"]
        self.assertEqual(len(certificates), 2)
        self.assertEqual(certificates[0]["canonical_batch_sha256"], _batch_hash(calls[:2]))
        self.assertEqual(
            _certified_batch_ids(
                certificates, "qa", calls, 2,
                config_fingerprint="fp", question_manifest_sha256="manifest",
            ),
            {"qa:000000", "qa:000001"},
        )

    def test_jsonl_repairs_only_torn_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_bytes(b'{"ok": 1}\n{"torn":')
            store = JsonlStore(path).open("session")
            records = store.read_existing()
            store.close()
            self.assertEqual(records[0], {"ok": 1})
            self.assertEqual(records[1]["record_type"], "store_repair")
            self.assertFalse(Path(str(path) + ".lock").exists())

    def test_jsonl_rejects_complete_corrupt_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_bytes(b'{"ok": 1}\nnot-json\n{"ok": 2}\n')
            store = JsonlStore(path).open("session")
            try:
                with self.assertRaises(RuntimeError):
                    store.read_existing()
            finally:
                store.close()

    def test_mixed_model_slug_and_frozen_prompt_hashes(self):
        treatments = {
            "planner": {"model_id": "Qwen/Qwen2.5-3B-Instruct"},
            "qa": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
        }
        self.assertIn(
            "qwen2.5-3b+qwen2.5-1.5b",
            result_slug("x", 10, 1234, "Qwen/Qwen2.5-3B-Instruct", treatments),
        )
        self.assertEqual(prompts.prompt_template_hashes() | {}, {
            "planner": "35f6d9e8bc089bbaf8a7ad6dc722f380892a1d7a0af7a4bd04ad01355cc59614",
            "step_definer": "8fe9b8a6ba665ae608601e296b6001ec07485d1615ea0ef826f17bbf2489a695",
            "extractor": "8f84d86ae84278916dcb483b5ba8891f65360327701048d48c4a7d4df403018c",
            "qa": "ef42a3d19aef7c9669407bc5478bf680ac8fc62d8dd2dd070543af200b0971a6",
            "solo": "337626135fa3a5054bb5a065cc638a9ca05c4e1f21b44977b4700c5a0cba94cb",
        })

    def test_derived_allocation_link_rejects_unrelated_config_tamper(self):
        base = {
            "models": {"small": "m1"},
            "dataset": {"manifest_sha256": "manifest"},
            "allocation_selector": {"status": "pending"},
            "runs": {"baseline": {"qa": "fp16"}},
        }
        run_def = {"qa": "8bit"}
        artifact = {
            "source_config_sha256": _content_hash(base),
            "execution_run_id": "ma_optimized_exploratory",
            "run_definition": run_def,
            "run_config_sha256": _content_hash(run_def),
        }
        artifact["artifact_sha256"] = _content_hash(artifact)
        derived = json.loads(json.dumps(base))
        derived["runs"]["ma_optimized_exploratory"] = run_def
        derived["allocation_selector"] = {"status": "frozen"}
        derived["frozen_allocation"] = {
            "selection_artifact_sha256": artifact["artifact_sha256"],
            "run_config_sha256": artifact["run_config_sha256"],
            "question_manifest_sha256": "manifest",
        }
        artifact["executable_config_sha256"] = _content_hash(derived)
        lock = {
            "experiment_config_contract": base,
            "experiment_config_content_sha256": _content_hash(base),
            "final_manifest_ids_sha256": "manifest",
        }
        self.assertTrue(_derived_config_matches_lock(derived, lock, artifact))
        tampered = json.loads(json.dumps(derived))
        tampered["models"]["small"] = "malicious-other-model"
        tampered_artifact = dict(artifact)
        tampered_artifact["executable_config_sha256"] = _content_hash(tampered)
        self.assertFalse(
            _derived_config_matches_lock(tampered, lock, tampered_artifact)
        )


if __name__ == "__main__":
    unittest.main()
