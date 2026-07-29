"""EM / F1 with HotpotQA's official normalization, plus bootstrap CIs.

SPEC §5. The normalization below is transcribed from the official HotpotQA
evaluation script (lowercase, strip punctuation, strip articles, collapse
whitespace) so our numbers are comparable to published ones.
"""

import random
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


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


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over the mean. Returns (mean, lo, hi).

    SPEC §5: 10k resamples, 95% interval, on every reported number.
    """
    if not values:
        return (float("nan"),) * 3
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return mean, lo, hi
