"""
BenchmarkRunner: orchestrates dataset loading, matching, and metric computation.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, Type

from er_bench.datasets import DATASETS
from er_bench.matchers import MATCHERS, EntityMatcher
from er_bench.metrics.standard import compute_all
from er_bench.metrics.report import BenchmarkReport


class BenchmarkRunner:
    """
    Orchestrates a full benchmark run.

    Usage:
        runner = BenchmarkRunner(dataset="febrl", matcher="cascade")
        report = runner.run()
        print(report)

        # Or with custom instances:
        from er_bench.datasets.febrl import FebrlDataset
        from er_bench.matchers.cascade import CascadeMatcher
        runner = BenchmarkRunner(
            dataset_instance=FebrlDataset(n_records=500),
            matcher_instance=CascadeMatcher(threshold_jw=0.85),
        )
        report = runner.run()
    """

    def __init__(
        self,
        dataset: Optional[str] = None,
        matcher: Optional[str] = None,
        dataset_instance=None,
        matcher_instance: Optional[EntityMatcher] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        matcher_kwargs: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ):
        """
        Args:
            dataset: Name of the dataset ("febrl", "icij", "dblp_acm").
            matcher: Name of the matcher ("exact", "fuzzy", "embedding", "cascade").
            dataset_instance: Pre-constructed dataset object (overrides dataset name).
            matcher_instance: Pre-constructed matcher object (overrides matcher name).
            dataset_kwargs: Keyword arguments for the dataset constructor.
            matcher_kwargs: Keyword arguments for the matcher constructor.
            notes: Optional notes for the report.
        """
        self.dataset_name = dataset
        self.matcher_name = matcher
        self.dataset_instance = dataset_instance
        self.matcher_instance = matcher_instance
        self.dataset_kwargs = dataset_kwargs or {}
        self.matcher_kwargs = matcher_kwargs or {}
        self.notes = notes

    def _get_dataset(self):
        if self.dataset_instance:
            return self.dataset_instance
        if not self.dataset_name:
            raise ValueError("Provide either dataset name or dataset_instance.")
        if self.dataset_name not in DATASETS:
            raise ValueError(
                f"Unknown dataset: '{self.dataset_name}'. "
                f"Available: {list(DATASETS.keys())}"
            )
        return DATASETS[self.dataset_name](**self.dataset_kwargs)

    def _get_matcher(self) -> EntityMatcher:
        if self.matcher_instance:
            return self.matcher_instance
        if not self.matcher_name:
            raise ValueError("Provide either matcher name or matcher_instance.")
        if self.matcher_name not in MATCHERS:
            raise ValueError(
                f"Unknown matcher: '{self.matcher_name}'. "
                f"Available: {list(MATCHERS.keys())}"
            )
        return MATCHERS[self.matcher_name](**self.matcher_kwargs)

    def run(self) -> BenchmarkReport:
        """
        Run the full benchmark pipeline.

        Returns:
            BenchmarkReport with all metrics populated.
        """
        # Load dataset
        ds = self._get_dataset()
        records, ground_truth = ds.as_dicts() if hasattr(ds, "as_dicts") else ds.load()

        # Run matcher
        matcher = self._get_matcher()
        t0 = time.perf_counter()
        match_results = matcher.match(records)
        elapsed = time.perf_counter() - t0

        # Extract pairs
        predicted_pairs = {r.as_pair() for r in match_results}

        # Compute metrics
        metrics = compute_all(predicted_pairs, ground_truth, len(records))

        # Tier breakdown (only for cascade)
        tier_breakdown: Dict[int, int] = {}
        for r in match_results:
            tier_breakdown[r.tier] = tier_breakdown.get(r.tier, 0) + 1

        return BenchmarkReport(
            dataset=getattr(ds, "name", self.dataset_name or "unknown"),
            matcher=getattr(matcher, "name", self.matcher_name or "unknown"),
            n_records=len(records),
            n_ground_truth=len(ground_truth),
            n_predicted=len(predicted_pairs),
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            reduction_ratio=metrics["reduction_ratio"],
            pairs_completeness=metrics["pairs_completeness"],
            tp=metrics["tp"],
            fp=metrics["fp"],
            fn=metrics["fn"],
            tier_breakdown=tier_breakdown,
            elapsed_seconds=elapsed,
            notes=self.notes,
        )
