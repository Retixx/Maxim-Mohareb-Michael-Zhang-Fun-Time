"""Run deterministic SPEC §15 Gate A against the frozen final cohort.

This CPU-only gate tests the repaired two-component retrieval mechanism, not
live Step Definer quality.  The follow-up ranking is deliberately given each
question's hidden gold title as an oracle bridge term.  Production never sees
that field; Gate C is the end-to-end test of whether model-generated tasks
resolve the bridge without gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import retrieval  # noqa: E402
from src.pipeline import load_questions  # noqa: E402


GATE_SCHEMA_VERSION = 1
MIN_HIDDEN_QUESTIONS = 1000
MIN_HIDDEN_BOTH_GOLD_RECALL = 0.75


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(rows: list[dict], field: str) -> float:
    if not rows:
        raise RuntimeError(f"retrieval gate has no rows for {field}")
    return sum(bool(row[field]) for row in rows) / len(rows)


def run_gate(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    retrieval_config = config["retrieval"]
    manifest_path = ROOT / dataset["manifest_path"]
    if _sha256(manifest_path) != dataset["manifest_file_sha256"]:
        raise RuntimeError("final cohort manifest file hash mismatch")
    if int(retrieval_config["k"]) != 10:
        raise RuntimeError("Gate A requires the frozen retrieval k=10")

    dataset_metadata: dict = {}
    questions = load_questions(
        int(dataset["n"]),
        seed=int(dataset["eval_seed"]),
        split=dataset["split"],
        config=dataset["config"],
        name=dataset["name"],
        exclude=set(),
        manifest_path=manifest_path,
        manifest_sha256=dataset["manifest_sha256"],
        revision=dataset["revision"],
        metadata=dataset_metadata,
    )
    observed_strata = {
        name: sum(q["retrieval_stratum"] == name for q in questions)
        for name in ("hidden_bridge", "fully_named")
    }
    if observed_strata != dataset["retrieval_strata_counts"]:
        raise RuntimeError(
            f"frozen retrieval strata changed: {observed_strata}"
        )

    corpus = retrieval.build_corpus(
        name=dataset["name"],
        split=dataset["split"],
        revision=dataset["revision"],
        configs=tuple(retrieval_config["corpus_configs"]),
    )
    context = retrieval.RetrievalContext(
        corpus,
        k=int(retrieval_config["k"]),
        anchor_k=int(retrieval_config["anchor_k"]),
    )
    fingerprint = context.fingerprint()
    expected_fingerprint = {
        "corpus_passages": int(retrieval_config["expected_corpus_passages"]),
        "corpus_sha256": retrieval_config["expected_corpus_sha256"],
        "algorithm": retrieval_config["algorithm"],
        "query_policy": retrieval_config["query_policy"],
        "k_per_step": int(retrieval_config["k"]),
        "initial_query_source": retrieval_config["initial_query_source"],
        "grounded_followup_k": int(retrieval_config["grounded_followup_k"]),
        "anchor_k": int(retrieval_config["anchor_k"]),
        "task_k": int(retrieval_config["task_k"]),
        "grounded_followup_requires_evidence": retrieval_config[
            "grounded_followup_requires_evidence"
        ],
    }
    for key, expected in expected_fingerprint.items():
        if fingerprint.get(key) != expected:
            raise RuntimeError(
                f"retrieval fingerprint mismatch for {key}: "
                f"{fingerprint.get(key)!r} != {expected!r}"
            )

    # Freeze rankings first. Gold titles are read below only for scoring. The
    # oracle bridge titles are the one explicit exception, declared in output.
    rankings = []
    max_unique_titles = 0
    for question in questions:
        safe_question = str(question["question"])
        hidden_titles = [str(title) for title in question["hidden_gold_titles"]]
        oracle_task_query = (
            f"{safe_question} | Oracle bridge title: {' | '.join(hidden_titles)}"
            if hidden_titles else safe_question
        )
        single = retrieval.search_anchored_union(
            context.index,
            safe_question,
            None,
            k=context.k,
            anchor_k=context.anchor_k,
        )["titles"]
        repaired = retrieval.search_anchored_union(
            context.index,
            safe_question,
            oracle_task_query,
            k=context.k,
            anchor_k=context.anchor_k,
        )["titles"]
        max_unique_titles = max(max_unique_titles, len(set(repaired)))
        rankings.append({
            "question_id": question["question_id"],
            "stratum": question["retrieval_stratum"],
            "single_titles": single,
            "repaired_titles": repaired,
        })
    if max_unique_titles > context.k:
        raise RuntimeError("repaired retrieval exceeded the k=10 exposure ceiling")

    gold_by_id = {
        question["question_id"]: set(
            (question.get("supporting_facts") or {}).get("title") or ()
        )
        for question in questions
    }
    scored = []
    for row in rankings:
        gold = gold_by_id[row["question_id"]]
        if not gold:
            raise RuntimeError(f"{row['question_id']}: missing gold titles")
        scored.append({
            "stratum": row["stratum"],
            "single_both_gold": gold <= set(row["single_titles"]),
            "repaired_both_gold": gold <= set(row["repaired_titles"]),
        })

    strata = {}
    for name in ("hidden_bridge", "fully_named"):
        rows = [row for row in scored if row["stratum"] == name]
        single_recall = _rate(rows, "single_both_gold")
        repaired_recall = _rate(rows, "repaired_both_gold")
        strata[name] = {
            "n": len(rows),
            "single_both_gold_recall_at_10": single_recall,
            "repaired_both_gold_recall_at_10": repaired_recall,
            "delta": repaired_recall - single_recall,
        }

    hidden = strata["hidden_bridge"]
    checks = {
        "hidden_bridge_n_at_least_1000": hidden["n"] >= MIN_HIDDEN_QUESTIONS,
        "k_is_10": context.k == 10,
        "hidden_bridge_both_gold_recall_at_least_0_75": (
            hidden["repaired_both_gold_recall_at_10"]
            >= MIN_HIDDEN_BOTH_GOLD_RECALL
        ),
        "exposure_never_exceeds_k": max_unique_titles <= context.k,
        "fully_named_ranking_is_unchanged": (
            strata["fully_named"]["delta"] == 0.0
        ),
    }
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "SPEC_15_GATE_A",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "oracle_query_uses_hidden_gold_titles": True,
        "claim_boundary": (
            "mechanism-only upper bound; not live Step Definer or accuracy evidence"
        ),
        "source_config_sha256": _sha256(config_path),
        "manifest_file_sha256": _sha256(manifest_path),
        "question_ids_sha256": dataset_metadata["question_ids_sha256"],
        "retrieval_fingerprint": fingerprint,
        "checks": checks,
        "strata": strata,
        "max_unique_titles": max_unique_titles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    parser.add_argument("--output", default=str(ROOT / "analysis" / "retrieval_gate.json"))
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    report = run_gate(Path(args.config).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hidden = report["strata"]["hidden_bridge"]
    fully_named = report["strata"]["fully_named"]
    print(
        f"Gate A {report['status']}: hidden_bridge n={hidden['n']}, "
        f"single={hidden['single_both_gold_recall_at_10']:.4f}, "
        f"repaired={hidden['repaired_both_gold_recall_at_10']:.4f}; "
        f"fully_named delta={fully_named['delta']:+.4f}"
    )
    print(f"wrote {output_path}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
