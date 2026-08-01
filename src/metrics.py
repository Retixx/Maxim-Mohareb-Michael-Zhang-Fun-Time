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


def auroc(scores: list[float], labels: list[float]) -> float:
    """AUROC of `scores` against binary `labels`, via the rank/Mann-Whitney form.

    Used for calibration (SPEC §5b prediction 4): does answer confidence predict
    correctness? 0.5 means confidence carries no information about whether the
    answer is right. A drop under quantization with accuracy held constant is
    exactly the "calibration damaged, knowledge intact" claim in SPEC §1.
    """
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(pairs, key=lambda x: x[0])
    ranks = {}
    i = 0
    while i < len(order):  # average ranks within ties
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks.setdefault(id(order[k]), avg)
        i = j + 1
    rank_sum = sum(r for (o, r) in ((o, ranks[id(o)]) for o in order) if o[1] == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def expected_calibration_error(confidences: list[float], labels: list[float],
                               n_bins: int = 10) -> float:
    """Standard binned ECE. `confidences` must already be on a [0, 1] scale.

    That precondition is now enforced. The bin predicate assigns only values in
    [0, 1] to a bin while `total` counts every pair, so out-of-range inputs were
    silently dropped from the numerator and kept in the denominator — pushing the
    result toward 0.0, which reads as "perfectly calibrated". Every confidence
    field this pipeline logs (`mean_logprob`, `min_logprob`, `mean_entropy`) is
    out of range, so feeding the obvious field to this function for SPEC §5b
    prediction 4 returned a clean-looking 0.0. Fail instead: exp() a mean logprob
    to get a probability first.
    """
    pairs = [(c, l) for c, l in zip(confidences, labels) if c is not None]
    if not pairs:
        return float("nan")
    bad = [c for c, _ in pairs if not 0.0 <= c <= 1.0]
    if bad:
        raise ValueError(
            f"{len(bad)} confidence values outside [0, 1] (e.g. {bad[0]!r}). ECE is "
            "defined on probabilities; pass exp(mean_logprob), not a raw logprob."
        )
    total = len(pairs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [(c, l) for c, l in pairs if (lo < c <= hi or (b == 0 and c == 0))]
        if not bucket:
            continue
        conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(l for _, l in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(acc - conf)
    return ece


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over the mean. Returns (mean, lo, hi).

    SPEC §5: 10k resamples, 95% interval, on every reported number.

    `values` is sorted first. The RNG draws *indices*, so without this the same
    multiset in a different order gives a different interval at the same seed —
    the 750 baseline F1 values return [38.21, 44.64] in file order and
    [38.16, 44.50] sorted by question id. The shift is only Monte-Carlo sized
    (~0.15 pp) and moved no conclusion, but a number in the paper must not
    depend on the order records happened to be appended to a JSONL. Sorting is
    safe: the bootstrap treats the sample as exchangeable, so the estimator is
    unchanged — this pins *which* resamples a given seed produces, nothing else.
    """
    if not values:
        return (float("nan"),) * 3
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean

    values = sorted(values)
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
