"""Scoring and resampling utilities used by the final experiment.

Answer EM/F1 follows the official HotpotQA normalization.  Inferential helpers
resample the experimental unit (a question, or a recorded timing batch) rather
than individual agent calls.  That distinction matters because the step-definer
and extractor can emit several correlated calls for one question.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence

import numpy as np


DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc((s or "").lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)

    # HotpotQA scores yes/no/noanswer as exact-match-only.
    if norm_pred in ("yes", "no", "noanswer") and norm_pred != norm_gold:
        return 0.0
    if norm_gold in ("yes", "no", "noanswer") and norm_pred != norm_gold:
        return 0.0

    pred_toks = norm_pred.split()
    gold_toks = norm_gold.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_mcnemar(
    outcomes_a: Sequence[int | float | bool],
    outcomes_b: Sequence[int | float | bool],
) -> dict[str, float | int]:
    """Return the exact two-sided McNemar test for paired binary outcomes.

    ``a_only_wins`` counts pairs where A is one and B is zero;
    ``b_only_wins`` counts the reverse.  Conditional on the number of
    discordant pairs, the null distribution is Binomial(n, 0.5).  The exact
    two-sided p-value is twice the lower tail at the smaller discordance count,
    capped at one.  With no discordant pairs, the p-value is one.
    """
    try:
        a_values = tuple(outcomes_a)
        b_values = tuple(outcomes_b)
    except TypeError as exc:
        raise ValueError("paired outcomes must be finite sequences") from exc

    if len(a_values) != len(b_values):
        raise ValueError("paired outcomes must have the same length")
    if not a_values:
        raise ValueError("paired outcomes must not be empty")

    for name, values in (("outcomes_a", a_values), ("outcomes_b", b_values)):
        for index, value in enumerate(values):
            if isinstance(value, (str, bytes)) or value not in (0, 1):
                raise ValueError(f"{name}[{index}] is not binary: {value!r}")

    a_only_wins = sum(
        value_a == 1 and value_b == 0
        for value_a, value_b in zip(a_values, b_values)
    )
    b_only_wins = sum(
        value_a == 0 and value_b == 1
        for value_a, value_b in zip(a_values, b_values)
    )
    n_discordant = a_only_wins + b_only_wins
    if n_discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(
            math.comb(n_discordant, successes)
            for successes in range(min(a_only_wins, b_only_wins) + 1)
        )
        p_value = min(1.0, (2 * lower_tail) / (1 << n_discordant))

    return {
        "n_pairs": len(a_values),
        "a_only_wins": a_only_wins,
        "b_only_wins": b_only_wins,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def auroc(scores: list[float], labels: list[float]) -> float:
    """AUROC via average ranks (the Mann-Whitney form)."""
    pairs = [(float(s), float(label)) for s, label in zip(scores, labels)
             if s is not None]
    pos = sum(label == 1 for _, label in pairs)
    neg = sum(label == 0 for _, label in pairs)
    if not pos or not neg:
        return float("nan")

    order = sorted(pairs, key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        average_rank = (i + j) / 2 + 1
        rank_sum += average_rank * sum(order[k][1] == 1 for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def expected_calibration_error(
    confidences: list[float],
    labels: list[float],
    n_bins: int = 10,
) -> float:
    """Standard binned ECE; confidences must be probabilities in ``[0, 1]``."""
    pairs = [(c, label) for c, label in zip(confidences, labels) if c is not None]
    if not pairs:
        return float("nan")
    bad = [c for c, _ in pairs if not 0.0 <= c <= 1.0]
    if bad:
        raise ValueError(
            f"{len(bad)} confidence values outside [0, 1] (e.g. {bad[0]!r}). "
            "ECE is defined on probabilities; pass exp(mean_logprob), not a raw "
            "logprob."
        )
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    total = len(pairs)
    ece = 0.0
    for bin_index in range(n_bins):
        lo, hi = bin_index / n_bins, (bin_index + 1) / n_bins
        bucket = [
            (confidence, label)
            for confidence, label in pairs
            if lo < confidence <= hi or (bin_index == 0 and confidence == 0)
        ]
        if bucket:
            confidence = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(label for _, label in bucket) / len(bucket)
            ece += len(bucket) / total * abs(accuracy - confidence)
    return ece


def _validate_resampling(n_resamples: int, alpha: float) -> None:
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")


def _percentile_interval(
    samples: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Order-statistic percentile interval, matching the project's old rule."""
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(ordered)
    lo = ordered[int((alpha / 2) * n)]
    hi = ordered[min(int((1 - alpha / 2) * n), n - 1)]
    return float(lo), float(hi)


def _bootstrap_means(
    matrix: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    chunk_size: int = 256,
) -> np.ndarray:
    """Jointly bootstrap row means without allocating a huge index tensor."""
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("matrix must be non-empty and two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("bootstrap input contains NaN or infinity")

    n_rows, n_columns = matrix.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, n_columns), dtype=np.float64)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, n_rows, size=(stop - start, n_rows))
        out[start:stop] = matrix[indices].mean(axis=1)
    return out


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over a mean.

    Values are sorted before resampling, making the result independent of JSONL
    append order.  Generation is vectorized in bounded chunks; 10,000 resamples
    at n=1,500 therefore takes a fraction of the time of the former Python
    double loop.
    """
    _validate_resampling(n_resamples, alpha)
    if len(values) == 0:
        return (float("nan"),) * 3
    array = np.sort(np.asarray(values, dtype=np.float64))
    if not np.isfinite(array).all():
        raise ValueError("bootstrap input contains NaN or infinity")
    estimate = float(array.mean())
    if len(array) == 1:
        return estimate, estimate, estimate
    draws = _bootstrap_means(
        array[:, None], n_resamples=n_resamples, seed=seed
    )[:, 0]
    lower, upper = _percentile_interval(draws, alpha)
    return estimate, lower, upper


def joint_paired_bootstrap(
    differences: Mapping[str, Mapping[Hashable, float]],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap several paired contrasts using the same question draws.

    ``differences`` maps contrast name to ``{question_id: paired_difference}``.
    All contrasts must contain exactly the same IDs; silently intersecting them
    would change the estimand and can hide an incomplete run.  The returned
    two-sided p-value uses the centered bootstrap null distribution.  Use
    :func:`holm_adjust` on a predeclared family before interpreting it.
    """
    if not differences:
        return {}
    names, estimates, draws, question_ids = joint_bootstrap_draws(
        differences, n_resamples=n_resamples, alpha=alpha, seed=seed
    )

    output: dict[str, dict[str, float | int]] = {}
    for column, name in enumerate(names):
        lower, upper = _percentile_interval(draws[:, column], alpha)
        # Center the estimated sampling distribution at zero to approximate the
        # null distribution.  The +1 correction prevents a reported p-value of
        # exactly zero with a finite Monte-Carlo sample.
        centered = draws[:, column] - estimates[column]
        extreme = int(np.count_nonzero(np.abs(centered) >= abs(estimates[column])))
        p_value = min(1.0, (extreme + 1) / (n_resamples + 1))
        output[name] = {
            "estimate": float(estimates[column]),
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value": p_value,
            "n_questions": len(question_ids),
            "n_resamples": n_resamples,
        }
    return output


def joint_bootstrap_draws(
    differences: Mapping[str, Mapping[Hashable, float]],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray, tuple[Hashable, ...]]:
    """Return shared question-bootstrap draws for a vector of paired effects.

    This lower-level companion to :func:`joint_paired_bootstrap` is useful when
    a prespecified downstream function must combine effects while retaining
    their covariance (the allocation selector sums up to four role effects).
    The ID-set equality check is intentionally the same strict check.
    """
    _validate_resampling(n_resamples, alpha)
    if not differences:
        raise ValueError("joint bootstrap has no contrasts")
    names = list(differences)
    reference_ids = set(differences[names[0]])
    if not reference_ids:
        raise ValueError("paired bootstrap has no question IDs")
    for name in names[1:]:
        ids = set(differences[name])
        if ids != reference_ids:
            raise ValueError(
                f"paired contrast {name!r} has a different question set: "
                f"missing={len(reference_ids - ids)}, extra={len(ids - reference_ids)}"
            )

    question_ids = tuple(sorted(reference_ids, key=str))
    matrix = np.asarray(
        [[differences[name][question_id] for name in names]
         for question_id in question_ids],
        dtype=np.float64,
    )
    estimates = matrix.mean(axis=0)
    if len(question_ids) == 1:
        draws = np.repeat(estimates[None, :], n_resamples, axis=0)
    else:
        draws = _bootstrap_means(matrix, n_resamples=n_resamples, seed=seed)
    return names, estimates, draws, question_ids


def clustered_ratio_bootstrap(
    clusters: Mapping[Hashable, tuple[float, float]],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int]:
    """CI for ``sum(numerator) / sum(denominator)``, resampling clusters.

    A cluster is normally one question for parse/churn metrics, or one batch for
    timing.  All calls from a sampled question travel together, avoiding the
    pseudo-replication caused by treating fan-out calls as independent.
    """
    _validate_resampling(n_resamples, alpha)
    if not clusters:
        raise ValueError("clustered bootstrap has no clusters")
    ordered_ids = sorted(clusters, key=str)
    values = np.asarray([clusters[cluster_id] for cluster_id in ordered_ids], dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("clusters must map to finite (numerator, denominator) pairs")
    if (values[:, 1] < 0).any() or values[:, 1].sum() <= 0:
        raise ValueError("cluster denominators must be non-negative with a positive total")

    estimate = float(values[:, 0].sum() / values[:, 1].sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    n_clusters = len(values)
    for start in range(0, n_resamples, 256):
        stop = min(start + 256, n_resamples)
        indices = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        sampled = values[indices].sum(axis=1)
        denominator = sampled[:, 1]
        draws[start:stop] = np.divide(
            sampled[:, 0],
            denominator,
            out=np.full(stop - start, np.nan),
            where=denominator > 0,
        )
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        raise ValueError("all clustered bootstrap draws had zero denominator")
    lower, upper = _percentile_interval(finite, alpha)
    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_clusters": n_clusters,
        "n_resamples": n_resamples,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error correction, keyed like the input."""
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {name!r}: {value!r}")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, (m - rank) * p_value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in p_values}
