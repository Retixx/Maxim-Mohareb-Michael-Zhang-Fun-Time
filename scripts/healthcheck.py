#!/usr/bin/env python3
"""Repo health check: every launch-blocking invariant, CPU-only, no network.

This exists because the failures that actually hurt this project are not test
failures — they are silent contract drift. SPEC §14 BUG-9 shipped a config the
campaign runner rejected, and the frozen plan silently scheduled 22 of 32 arms.
Neither showed up as a crash; both would have burned A100 hours.

Run before any campaign, and in CI on every push:

    python scripts/healthcheck.py            # all checks
    python scripts/healthcheck.py --strict   # also fail on warnings

Exit codes: 0 all passed, 1 one or more failed, 2 the check itself broke.

Every check is pure CPU: no GPU, no model download, no dataset download. A check
that cannot run offline does not belong here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import retrieval
from src.contracts import (
    CANDIDATE_ALLOCATION_COUNT,
    OPTIMIZED_RUN,
    ROLES,
    SELECTOR_CONFIGS,
    STATIC_RUN_COUNT,
    STATIC_RUNS,
    TINY_RUNS,
)

CONFIG_PATH = ROOT / "config" / "experiment.yaml"


class Report:
    """Collects check outcomes so one failure does not hide the rest."""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.passed: list[str] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, detail: str) -> None:
        self.failures.append((name, detail))
        print(f"  FAIL  {name}\n          {detail}")

    def warn(self, name: str, detail: str) -> None:
        self.warnings.append((name, detail))
        print(f"  WARN  {name}\n          {detail}")

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.ok(name)
        else:
            self.fail(name, detail)
        return condition


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_hash(value) -> str:
    """Canonical JSON hash — must match src/runner._content_hash exactly."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _sha256_ids(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_arm_contract(config: dict, report: Report) -> None:
    """Config arms must be exactly the frozen contract in src/contracts.py."""
    print("\n[arm contract]")
    runs = set(config.get("runs") or {})
    allowed = {frozenset(STATIC_RUNS), frozenset(STATIC_RUNS | {OPTIMIZED_RUN})}
    report.check(
        f"config declares the {STATIC_RUN_COUNT}-arm contract",
        frozenset(runs) in allowed,
        f"missing={sorted(STATIC_RUNS - runs)} extra={sorted(runs - STATIC_RUNS)}",
    )
    report.check(
        "tiny appendix-floor arms are all present",
        set(TINY_RUNS) <= runs,
        f"missing={sorted(set(TINY_RUNS) - runs)}",
    )

    selector = config.get("allocation_selector") or {}
    report.check(
        f"selector declares all {len(SELECTOR_CONFIGS)} tiers",
        set(selector.get("candidates") or {}) == set(SELECTOR_CONFIGS),
        f"got={sorted(selector.get('candidates') or {})}",
    )
    report.check(
        f"selector enumerates {CANDIDATE_ALLOCATION_COUNT} allocations",
        selector.get("candidate_allocation_count") == CANDIDATE_ALLOCATION_COUNT,
        f"got={selector.get('candidate_allocation_count')} "
        f"(expected {len(SELECTOR_CONFIGS)}**{len(ROLES)})",
    )

    # An arm defined but never scheduled is invisible waste; an arm scheduled
    # but never defined crashes mid-campaign.
    timing_ids = set(config.get("timing", {}).get("run_ids") or [])
    report.check(
        "every timing run_id is a defined arm",
        timing_ids <= runs,
        f"undefined={sorted(timing_ids - runs)}",
    )
    expected_timing = set(STATIC_RUNS) - set(TINY_RUNS)
    missing_timing = expected_timing - timing_ids
    if missing_timing:
        report.warn(
            "timing covers every non-tiny arm",
            f"not timed: {sorted(missing_timing)}",
        )
    else:
        report.ok("timing covers every non-tiny arm")


def check_manifest_pins(config: dict, report: Report) -> None:
    """Every committed manifest must match its pinned hashes byte for byte."""
    print("\n[manifest hash pins]")
    dataset = config.get("dataset") or {}
    specs = [
        ("final", dataset.get("manifest_path"),
         dataset.get("manifest_file_sha256"), dataset.get("manifest_sha256")),
        ("preflight", dataset.get("preflight_manifest_path"),
         dataset.get("preflight_manifest_file_sha256"),
         dataset.get("preflight_manifest_sha256")),
        ("pilot", (config.get("pilot") or {}).get("manifest_path"),
         (config.get("pilot") or {}).get("manifest_file_sha256"),
         (config.get("pilot") or {}).get("manifest_sha256")),
        ("timing", (config.get("timing") or {}).get("manifest_path"),
         (config.get("timing") or {}).get("manifest_file_sha256"),
         (config.get("timing") or {}).get("manifest_sha256")),
    ]
    for label, rel, file_sha, ids_sha in specs:
        if not rel:
            report.fail(f"{label} manifest path is declared", "path missing from config")
            continue
        path = ROOT / rel
        if not path.exists():
            report.fail(f"{label} manifest exists", f"{rel} not found")
            continue
        report.check(
            f"{label} manifest file hash matches pin",
            _sha256_file(path) == file_sha,
            f"{rel}: computed {_sha256_file(path)[:16]}... pinned {str(file_sha)[:16]}...",
        )
        blob = json.loads(path.read_text(encoding="utf-8"))
        ids = blob.get("question_ids") or []
        report.check(
            f"{label} manifest id-list hash matches pin",
            _sha256_ids(ids) == ids_sha,
            f"{rel}: {len(ids)} ids hash to {_sha256_ids(ids)[:16]}..., "
            f"pinned {str(ids_sha)[:16]}...",
        )
        report.check(
            f"{label} manifest ids are unique",
            len(ids) == len(set(ids)),
            f"{len(ids) - len(set(ids))} duplicate ids",
        )


def check_cohort_disjointness(config: dict, report: Report) -> None:
    """Selection and confirmation cohorts must not overlap (SPEC §5e)."""
    print("\n[cohort disjointness]")
    dataset = config.get("dataset") or {}
    final_path = ROOT / (dataset.get("manifest_path") or "")
    if not final_path.exists():
        report.fail("final manifest readable", "cannot check disjointness")
        return
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final_ids = set(final.get("question_ids") or [])
    excluded = set((final.get("exclusions") or {}).get("question_ids") or [])

    report.check(
        "final cohort does not intersect its own exclusion set",
        not (final_ids & excluded),
        f"{len(final_ids & excluded)} ids appear in both",
    )

    for label, key, section in (
        ("preflight", "preflight_manifest_path", dataset),
        ("pilot", "manifest_path", config.get("pilot") or {}),
        ("timing", "manifest_path", config.get("timing") or {}),
    ):
        rel = section.get(key)
        if not rel:
            continue
        path = ROOT / rel
        if not path.exists():
            continue
        ids = set(json.loads(path.read_text(encoding="utf-8")).get("question_ids") or [])
        report.check(
            f"{label} cohort is disjoint from the final 1,500",
            not (ids & final_ids),
            f"{len(ids & final_ids)} ids leak into the scored cohort",
        )


def check_model_pins(config: dict, report: Report) -> None:
    """No model may load from a moving ref."""
    print("\n[model revision pins]")
    revisions = config.get("model_revisions") or {}
    report.check("model_revisions is populated", bool(revisions), "block is empty")

    unpinned = [m for m, r in revisions.items()
                if not r or str(r).strip().upper() in {"TBD", "MAIN", "NONE"}]
    report.check(
        "no model revision is TBD or a moving ref",
        not unpinned,
        f"unpinned: {unpinned}",
    )
    bad = [f"{m}={r}" for m, r in revisions.items()
           if r and str(r).strip().upper() not in {"TBD", "MAIN", "NONE"}
           and (len(str(r)) != 40 or not all(c in "0123456789abcdef" for c in str(r)))]
    report.check(
        "every revision is a 40-hex commit SHA",
        not bad,
        f"malformed: {bad}",
    )

    declared = {config.get("model_id"), *(config.get("models") or {}).values()}
    declared.discard(None)
    missing = sorted(declared - set(revisions))
    report.check(
        "every model in use has a pinned revision",
        not missing,
        f"unpinned models: {missing}",
    )


def check_retrieval_policy(config: dict, report: Report) -> None:
    """Config and implementation must agree on the retrieval contract."""
    print("\n[retrieval policy]")
    retr = config.get("retrieval") or {}
    report.check(
        "query_policy matches the implementation",
        retr.get("query_policy") == retrieval.QUERY_POLICY,
        f"config={retr.get('query_policy')} code={retrieval.QUERY_POLICY}",
    )
    report.check(
        "initial_query_source matches the implementation",
        retr.get("initial_query_source") == retrieval.INITIAL_QUERY_SOURCE,
        f"config={retr.get('initial_query_source')} code={retrieval.INITIAL_QUERY_SOURCE}",
    )
    # A YAML "true" string would coerce past a truthiness test and silently
    # change the policy, so require a real boolean (SPEC §14 BUG-2).
    value = retr.get("grounded_followup_requires_evidence")
    report.check(
        "grounded_followup_requires_evidence is a literal boolean",
        isinstance(value, bool),
        f"got {value!r} of type {type(value).__name__}",
    )
    report.check(
        "grounded_followup_requires_evidence matches the implementation",
        value is retrieval.GROUNDED_FOLLOWUP_REQUIRES_EVIDENCE,
        f"config={value!r} code={retrieval.GROUNDED_FOLLOWUP_REQUIRES_EVIDENCE!r}",
    )
    report.check(
        "grounded follow-up owns the full per-step budget",
        retr.get("grounded_followup_k") == retr.get("k"),
        f"followup_k={retr.get('grounded_followup_k')} k={retr.get('k')}",
    )

    # This guard is duplicated across six modules. When BUG-2 flipped the pinned
    # value from True to False, four sites were updated and two were not —
    # analyze.py then refused to produce a report for a correct artifact, and
    # prefetch_assets.py made `prepare` unable to succeed at all. Neither is
    # covered by a unit test. Assert no site still hardcodes `is not True`.
    stale = []
    for rel in (
        "analyze.py", "src/runner.py", "scripts/check_pilot.py",
        "scripts/run_campaign.py", "scripts/prefetch_assets.py",
    ):
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if ("grounded_followup_requires_evidence" in line
                    and "is not True" in line):
                stale.append(f"{rel}:{line_no}")
    report.check(
        "no module hardcodes grounded_followup_requires_evidence is True",
        not stale,
        f"stale guards at {stale} — these reject correct artifacts now that "
        f"the pinned value is {retrieval.GROUNDED_FOLLOWUP_REQUIRES_EVIDENCE}",
    )


def check_generation_mode(config: dict, report: Report) -> None:
    """Qwen3 must be driven in the mode the config declares."""
    print("\n[generation mode]")
    value = config.get("thinking_mode")
    report.check(
        "thinking_mode is a literal boolean",
        isinstance(value, bool),
        f"got {value!r} of type {type(value).__name__}",
    )
    report.check(
        "thinking_mode is disabled",
        value is False,
        "Qwen3 reasoning blocks exceed every role's 48-320 token budget; a "
        "thinking-mode run truncates before emitting JSON",
    )

    # Qwen3's template emits an OPEN <think> tag unless the flag is passed, so
    # a render that bypasses the helper silently re-enables chain-of-thought.
    # That is how every generation came to be implicit reasoning: the config
    # declared thinking_mode from the migration onward, and no code read it.
    # Every apply_chat_template call must pass enable_thinking explicitly.
    # Qwen3 renders an OPEN <think> tag without it, and a reasoning block blows
    # every role's 48-320 token budget -> truncated -> no JSON -> EM/F1 near
    # zero in all 32 arms, failing silently.
    offenders = []
    for path in sorted((ROOT / "src").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "apply_chat_template(" not in line:
                continue
            window = "\n".join(text.splitlines()[number - 1:number + 8])
            if "enable_thinking" not in window:
                offenders.append(f"src/{path.name}:{number}")
    report.check(
        "every chat-template render passes enable_thinking",
        not offenders,
        f"renders that would default to chain-of-thought: {offenders}",
    )

    try:
        import inspect

        from src import agents, models

        source = Path(models.__file__).read_text(encoding="utf-8")
        report.check(
            "generation renders in non-thinking mode",
            "enable_thinking=False" in source,
            "models.py never pins enable_thinking=False",
        )
        del agents, inspect
    except Exception as exc:
        report.fail("generation mode wiring is inspectable",
                    f"{type(exc).__name__}: {exc}")


def check_frozen_plan(config: dict, report: Report) -> None:
    """The frozen accuracy plan must schedule every arm exactly once."""
    print("\n[frozen accuracy plan]")
    try:
        from scripts.a100_production import (
            FROZEN_PLAN_PATH,
            FROZEN_PLAN_SHA256,
            load_frozen_accuracy_plan,
        )
    except Exception as exc:
        report.fail("frozen plan module imports", f"{type(exc).__name__}: {exc}")
        return

    path = Path(FROZEN_PLAN_PATH)
    if not report.check("frozen plan file exists", path.exists(), f"{path} missing"):
        return
    report.check(
        "frozen plan file hash matches its pin",
        _sha256_file(path) == FROZEN_PLAN_SHA256,
        f"computed {_sha256_file(path)[:16]}... pinned {FROZEN_PLAN_SHA256[:16]}...",
    )
    try:
        plan = load_frozen_accuracy_plan(config)
    except Exception as exc:
        report.fail("frozen plan validates against the live config",
                    f"{type(exc).__name__}: {exc}")
        return
    report.ok("frozen plan validates against the live config")

    scheduled = [r for values in plan["assignments"].values() for r in values]
    report.check(
        "frozen plan schedules every arm exactly once",
        sorted(scheduled) == sorted(STATIC_RUNS),
        f"scheduled {len(scheduled)} arms, contract has {STATIC_RUN_COUNT}; "
        f"missing={sorted(set(STATIC_RUNS) - set(scheduled))}",
    )


def check_campaign_planning(config: dict, report: Report) -> None:
    """The campaign runner must accept the config it ships with."""
    print("\n[campaign planning]")
    try:
        from scripts.run_campaign import build_plan, validate_matrix
    except Exception as exc:
        report.fail("campaign module imports", f"{type(exc).__name__}: {exc}")
        return
    try:
        validate_matrix(config)
        report.ok("validate_matrix accepts the shipped config")
    except Exception as exc:
        report.fail("validate_matrix accepts the shipped config",
                    f"{type(exc).__name__}: {exc}")
        return

    for workers in (1, 4, 6):
        try:
            plan = build_plan(config, kind="accuracy", workers=workers, seed=20260805)
        except Exception as exc:
            report.fail(f"accuracy plan builds for {workers} worker(s)",
                        f"{type(exc).__name__}: {exc}")
            continue
        assigned = [r for v in plan["assignments"].values() for r in v]
        report.check(
            f"accuracy plan covers every arm once at {workers} worker(s)",
            sorted(assigned) == sorted(STATIC_RUNS),
            f"got {len(assigned)} arms ({len(set(assigned))} unique)",
        )

    # Sharding must not change WHICH arms run, only who runs them.
    orders = {
        w: build_plan(config, kind="accuracy", workers=w, seed=20260805)["ordered_run_ids"]
        for w in (1, 4, 6)
    }
    report.check(
        "global arm order is independent of worker count",
        len({tuple(o) for o in orders.values()}) == 1,
        "sharding changed the global order",
    )


def check_line_endings(report: Report) -> None:
    """Hash-pinned files must be LF, or Windows checkouts fail integrity."""
    print("\n[line endings]")
    attributes = ROOT / ".gitattributes"
    if not report.check(".gitattributes exists", attributes.exists(),
                        "hash pins are not portable without it"):
        return
    text = attributes.read_text(encoding="utf-8")
    for pattern in ("*.json", "*.yaml"):
        report.check(
            f".gitattributes pins {pattern} to LF",
            f"{pattern} text eol=lf" in text,
            f"no eol=lf rule for {pattern}",
        )
    crlf = [
        path.relative_to(ROOT)
        for path in [*(ROOT / "config").rglob("*.json"), CONFIG_PATH]
        if b"\r\n" in path.read_bytes()
    ]
    report.check(
        "no hash-pinned file has CRLF in the working tree",
        not crlf,
        f"CRLF found in: {crlf}",
    )


def check_prompt_bundle(report: Report) -> None:
    """Prompt hashes must be reproducible: they key every artifact."""
    print("\n[prompt bundle]")
    try:
        from src import prompts
    except Exception as exc:
        report.fail("prompts module imports", f"{type(exc).__name__}: {exc}")
        return
    first = prompts.prompt_template_hashes()
    second = prompts.prompt_template_hashes()
    report.check("prompt template hashes are deterministic", first == second,
                 "two calls disagreed")
    report.check("prompt bundle version is set",
                 bool(getattr(prompts, "PROMPT_BUNDLE_VERSION", None)),
                 "PROMPT_BUNDLE_VERSION is empty")
    # Per-step stages (extractor_step2, ...) reuse their base role's template,
    # so resolve through STAGE_ROLE; plan_summary carries its own.
    unresolved = sorted(
        stage for stage in prompts.PIPELINE_STAGES
        if stage not in first and prompts.STAGE_ROLE.get(stage) not in first
    )
    report.check(
        "every pipeline stage resolves to a hashed template",
        not unresolved,
        f"stages with no template via STAGE_ROLE: {unresolved}",
    )
    orphan_roles = sorted(set(prompts.STAGE_ROLE) - set(prompts.PIPELINE_STAGES))
    report.check(
        "STAGE_ROLE declares no stage outside the pipeline",
        not orphan_roles,
        f"orphan stages: {orphan_roles}",
    )


def check_results_lineage(config: dict, report: Report) -> None:
    """Off-lineage artifacts must not be silently analysed (SPEC §14 BUG-7)."""
    print("\n[results lineage]")
    results_dir = ROOT / (config.get("results_dir") or "results")
    if not results_dir.exists():
        report.ok("results directory is absent (nothing to mis-analyse)")
        return
    metas = sorted(results_dir.glob("*.meta.json"))
    if not metas:
        report.ok("no result metadata present")
        return

    from src.contracts import EXPERIMENT_SCHEMA

    # Being off-lineage is NOT a launch blocker: analysis and resume both key on
    # the experiment fingerprint, so an artifact on an older schema is excluded
    # deterministically. What IS dangerous is an artifact whose lineage cannot
    # be determined at all — no fingerprint payload, or unreadable — because
    # nothing downstream can prove it should be skipped. Only that is a warning,
    # so `--strict` stays a gate that can actually pass.
    current, excluded = [], []
    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            excluded.append(meta_path.name)
            continue
        payload = meta.get("experiment_fingerprint_payload")
        # A consumer accepts an artifact only on a PRESENT, MATCHING payload.
        # Absence is therefore a deterministic rejection, not a hole: verified
        # against completed_run_ids(), which accepts none of the legacy
        # artifacts for any cohort, and validate_final_cohort(), which fails
        # closed rather than averaging them in.
        if isinstance(payload, dict) and payload.get("schema") == EXPERIMENT_SCHEMA:
            current.append(meta_path.name)
        else:
            excluded.append(meta_path.name)

    # The real hazard is the reverse of "old artifacts exist": an artifact that
    # claims the CURRENT schema while the run that produced it is not
    # reproducible from this tree. That is what a mismatched fingerprint means.
    mismatched = []
    for name in current:
        meta = json.loads((results_dir / name).read_text(encoding="utf-8"))
        payload = meta["experiment_fingerprint_payload"]
        if meta.get("experiment_fingerprint") != content_hash(payload):
            mismatched.append(name)
    report.check(
        "no artifact claims the current lineage with a broken fingerprint",
        not mismatched,
        f"{len(mismatched)} artifact(s) self-inconsistent: {mismatched[:5]}",
    )
    if excluded:
        print(
            f"  INFO  {len(excluded)} of {len(metas)} artifacts are off-lineage "
            f"(schema != {EXPERIMENT_SCHEMA!r} or pre-fingerprint).\n"
            f"          Every consumer rejects them deterministically; they are "
            f"diagnostic only (SPEC §14 BUG-7)."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    print(f"marag healthcheck — {config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path}")

    report = Report()
    check_arm_contract(config, report)
    check_manifest_pins(config, report)
    check_cohort_disjointness(config, report)
    check_model_pins(config, report)
    check_retrieval_policy(config, report)
    check_generation_mode(config, report)
    check_frozen_plan(config, report)
    check_campaign_planning(config, report)
    check_line_endings(report)
    check_prompt_bundle(report)
    check_results_lineage(config, report)

    total = len(report.passed) + len(report.failures)
    print(f"\n{'=' * 68}")
    print(f"{len(report.passed)}/{total} checks passed, "
          f"{len(report.warnings)} warning(s)")
    if report.failures:
        print("\nFAILED:")
        for name, detail in report.failures:
            print(f"  - {name}: {detail}")
    if report.warnings:
        print("\nWARNINGS:")
        for name, detail in report.warnings:
            print(f"  - {name}: {detail}")

    if report.failures:
        return 1
    if args.strict and report.warnings:
        print("\n--strict: warnings are failures")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
