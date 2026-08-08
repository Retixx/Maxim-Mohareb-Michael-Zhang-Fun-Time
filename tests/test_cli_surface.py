"""Every script must expose a safe, working `--help`.

Two generators here rewrite frozen, SHA-256-pinned cohort manifests. They had
no argument parsing at all, so ANY invocation — including `--help` — rewrote
them. They are deterministic, so no damage was done, but a changed upstream
dataset revision or a touched sampler would have silently re-frozen the cohort
and invalidated every artifact keyed to it.

Two others imported `scripts.*` without bootstrapping sys.path, so direct
invocation died with ModuleNotFoundError.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py")) + [
    ROOT / "analyze.py", ROOT / "smoke_test.py",
]
# Generators whose output is pinned in config/experiment.yaml.
FROZEN = sorted((ROOT / "config" / "manifests").glob("*.json"))


def _digests() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN}


class CliSurfaceTests(unittest.TestCase):
    def test_every_script_has_a_working_help(self):
        failures = []
        for script in SCRIPTS:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
            if "usage:" not in result.stdout:
                head = (result.stderr or result.stdout).strip().splitlines()
                failures.append(f"{script.name}: {head[-1] if head else 'no usage line'}")
        self.assertEqual(failures, [], f"scripts without a usable --help: {failures}")

    def test_help_never_rewrites_a_frozen_manifest(self):
        before = _digests()
        for script in SCRIPTS:
            subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
        self.assertEqual(
            before, _digests(),
            "a --help invocation rewrote a SHA-256-pinned cohort manifest",
        )

    def test_manifest_generators_refuse_to_overwrite_without_force(self):
        for name in ("build_pilot_manifest.py", "freeze_final_sample.py"):
            before = _digests()
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name)],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
            self.assertNotEqual(result.returncode, 0, f"{name} overwrote without --force")
            self.assertIn("refusing to overwrite", result.stderr + result.stdout)
            self.assertEqual(before, _digests(), f"{name} mutated a frozen manifest")


if __name__ == "__main__":
    unittest.main()
