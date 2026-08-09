import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from clean_room import passage_replay_core as core


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


if __name__ == "__main__":
    unittest.main()
