"""Metrics for entity resolution benchmarking."""

from er_bench.metrics.standard import (
    precision,
    recall,
    f1_score,
    reduction_ratio,
    pairs_completeness,
    compute_all,
)
from er_bench.metrics.report import BenchmarkReport

__all__ = [
    "precision",
    "recall",
    "f1_score",
    "reduction_ratio",
    "pairs_completeness",
    "compute_all",
    "BenchmarkReport",
]
