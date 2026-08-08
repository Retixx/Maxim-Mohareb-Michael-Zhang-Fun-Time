#!/usr/bin/env python3
"""Fast standalone check: config arm matrix still matches src/contracts.py.

This is the SPEC §14 BUG-9 failure class — the Qwen3 migration moved the config
to 32 arms and 7 selector tiers but left the campaign runner asserting 22 and 5,
so `validate_matrix` rejected the config the repo shipped and no campaign could
launch. The unit tests passed the whole time.

Deliberately imports only `yaml` and `src.contracts` — no torch, no scipy — so
it runs in well under a second and is cheap enough for a pre-commit hook.
`scripts/healthcheck.py` supersets this and pulls the heavy deps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import (
    CANDIDATE_ALLOCATION_COUNT,
    OPTIMIZED_RUN,
    SELECTOR_CONFIGS,
    STATIC_RUN_COUNT,
    STATIC_RUNS,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?",
                        default=str(ROOT / "config" / "experiment.yaml"),
                        help="path to experiment.yaml")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    problems: list[str] = []

    runs = set(config.get("runs") or {})
    if runs not in (set(STATIC_RUNS), set(STATIC_RUNS) | {OPTIMIZED_RUN}):
        problems.append(
            f"arm drift: missing={sorted(set(STATIC_RUNS) - runs)} "
            f"extra={sorted(runs - set(STATIC_RUNS))}"
        )

    selector = config.get("allocation_selector") or {}
    candidates = set(selector.get("candidates") or {})
    if candidates != set(SELECTOR_CONFIGS):
        problems.append(
            f"selector tier drift: missing={sorted(set(SELECTOR_CONFIGS) - candidates)} "
            f"extra={sorted(candidates - set(SELECTOR_CONFIGS))}"
        )
    if selector.get("candidate_allocation_count") != CANDIDATE_ALLOCATION_COUNT:
        problems.append(
            f"allocation count drift: config says "
            f"{selector.get('candidate_allocation_count')}, "
            f"contract says {CANDIDATE_ALLOCATION_COUNT}"
        )

    if problems:
        print("arm contract FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nThe contract lives in src/contracts.py. Update it there and every "
            "consumer follows; do not re-hardcode counts.",
            file=sys.stderr,
        )
        return 1

    print(
        f"arm contract OK ({STATIC_RUN_COUNT} arms, "
        f"{len(SELECTOR_CONFIGS)} tiers, {CANDIDATE_ALLOCATION_COUNT} allocations)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
