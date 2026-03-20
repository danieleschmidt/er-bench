"""
Standard entity resolution metrics.

Metrics:
    precision       — fraction of predicted pairs that are true matches
    recall          — fraction of true matches that were predicted
    f1_score        — harmonic mean of precision and recall
    reduction_ratio — fraction of all possible pairs pruned (blocking quality)
    pairs_completeness — fraction of true matches in candidate set (= recall)

References:
    Köpcke, H., Thor, A., Rahm, E. (2010). Evaluation of entity resolution
    approaches on real-world match problems. PVLDB.

    Christen, P. (2012). Data Matching. Springer.
"""

from __future__ import annotations

from typing import Set, Tuple, Dict, Optional


Pair = Tuple[str, str]
PairSet = Set[Pair]


def _normalize_pairs(pairs: PairSet) -> PairSet:
    """Ensure all pairs have id1 <= id2."""
    return {(a, b) if a <= b else (b, a) for a, b in pairs}


def precision(predicted: PairSet, ground_truth: PairSet) -> float:
    """
    Precision = |predicted ∩ ground_truth| / |predicted|

    The fraction of predicted candidate pairs that are true matches.
    High precision = few false positives.

    Returns 0.0 if predicted is empty.
    """
    predicted = _normalize_pairs(predicted)
    ground_truth = _normalize_pairs(ground_truth)
    if not predicted:
        return 0.0
    return len(predicted & ground_truth) / len(predicted)


def recall(predicted: PairSet, ground_truth: PairSet) -> float:
    """
    Recall = |predicted ∩ ground_truth| / |ground_truth|

    The fraction of true matches that were found.
    High recall = few false negatives.

    Returns 0.0 if ground_truth is empty, 1.0 if ground_truth ⊆ predicted.
    """
    predicted = _normalize_pairs(predicted)
    ground_truth = _normalize_pairs(ground_truth)
    if not ground_truth:
        return 0.0
    return len(predicted & ground_truth) / len(ground_truth)


def f1_score(predicted: PairSet, ground_truth: PairSet) -> float:
    """
    F1 = 2 * precision * recall / (precision + recall)

    Harmonic mean of precision and recall. Balanced metric.

    Returns 0.0 if both precision and recall are 0.
    """
    p = precision(predicted, ground_truth)
    r = recall(predicted, ground_truth)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reduction_ratio(predicted: PairSet, n_records: int) -> float:
    """
    Reduction Ratio = 1 - |predicted| / |all_pairs|

    Measures blocking effectiveness: fraction of all possible pairs
    that were pruned (not in candidate set).

    Higher is better (more pruning), but must be balanced with recall.

    Returns 0.0 if n_records < 2.
    """
    if n_records < 2:
        return 0.0
    all_pairs = n_records * (n_records - 1) / 2
    return 1.0 - len(predicted) / all_pairs


def pairs_completeness(predicted: PairSet, ground_truth: PairSet) -> float:
    """
    Pairs Completeness = recall (alias).

    Standard ER metric: the fraction of true match pairs included
    in the candidate set. Equivalent to recall.
    """
    return recall(predicted, ground_truth)


def compute_all(
    predicted: PairSet,
    ground_truth: PairSet,
    n_records: int,
) -> Dict[str, float]:
    """
    Compute all standard ER metrics in one call.

    Returns:
        dict with keys: precision, recall, f1, reduction_ratio,
                         pairs_completeness, tp, fp, fn, n_predicted, n_ground_truth
    """
    p_norm = _normalize_pairs(predicted)
    g_norm = _normalize_pairs(ground_truth)

    tp = len(p_norm & g_norm)
    fp = len(p_norm - g_norm)
    fn = len(g_norm - p_norm)

    prec = tp / len(p_norm) if p_norm else 0.0
    rec = tp / len(g_norm) if g_norm else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    rr = reduction_ratio(p_norm, n_records)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "reduction_ratio": rr,
        "pairs_completeness": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predicted": len(p_norm),
        "n_ground_truth": len(g_norm),
        "n_records": n_records,
    }
