"""Tests for metrics and BenchmarkReport."""

import json
import os
import tempfile
import pytest

from er_bench.metrics.standard import precision, recall, f1_score, reduction_ratio, compute_all
from er_bench.metrics.report import BenchmarkReport


PREDICTED = {("A", "B"), ("B", "C"), ("A", "D")}
GROUND_TRUTH = {("A", "B"), ("B", "C"), ("E", "F")}


class TestMetrics:
    def test_metrics_recall(self):
        """Known ground truth → verify recall calculation."""
        r = recall(PREDICTED, GROUND_TRUTH)
        # GT has 3 pairs, 2 are in predicted → 2/3
        assert abs(r - 2 / 3) < 1e-9

    def test_metrics_precision(self):
        """Predicted has 3 pairs, 2 are correct → 2/3."""
        p = precision(PREDICTED, GROUND_TRUTH)
        assert abs(p - 2 / 3) < 1e-9

    def test_metrics_f1(self):
        """F1 of equal precision and recall."""
        p = precision(PREDICTED, GROUND_TRUTH)
        r = recall(PREDICTED, GROUND_TRUTH)
        f1 = f1_score(PREDICTED, GROUND_TRUTH)
        expected = 2 * p * r / (p + r)
        assert abs(f1 - expected) < 1e-9

    def test_metrics_recall_perfect(self):
        """Perfect recall when predicted == ground truth."""
        assert recall(GROUND_TRUTH, GROUND_TRUTH) == 1.0

    def test_metrics_recall_zero(self):
        """Zero recall when no true pairs predicted."""
        wrong = {("X", "Y")}
        assert recall(wrong, GROUND_TRUTH) == 0.0

    def test_metrics_reduction_ratio(self):
        """Reduction ratio test."""
        predicted = {("A", "B")}  # 1 pair out of 10 possible (5 records)
        rr = reduction_ratio(predicted, 5)
        # All pairs = 5*4/2 = 10, predicted = 1 → rr = 1 - 1/10 = 0.9
        assert abs(rr - 0.9) < 1e-9

    def test_metrics_reduction_ratio_zero_records(self):
        assert reduction_ratio(set(), 0) == 0.0
        assert reduction_ratio(set(), 1) == 0.0

    def test_compute_all_keys(self):
        """compute_all returns all expected keys."""
        result = compute_all(PREDICTED, GROUND_TRUTH, n_records=10)
        for key in ["precision", "recall", "f1", "reduction_ratio", "tp", "fp", "fn"]:
            assert key in result

    def test_compute_all_tp_fp_fn(self):
        result = compute_all(PREDICTED, GROUND_TRUTH, n_records=10)
        assert result["tp"] == 2  # A-B and B-C
        assert result["fp"] == 1  # A-D
        assert result["fn"] == 1  # E-F


class TestBenchmarkReport:
    def _make_report(self, **kwargs) -> BenchmarkReport:
        defaults = dict(
            dataset="febrl",
            matcher="cascade",
            n_records=100,
            n_ground_truth=15,
            n_predicted=20,
            precision=0.75,
            recall=1.0,
            f1=0.857,
            reduction_ratio=0.96,
            pairs_completeness=1.0,
            tp=15,
            fp=5,
            fn=0,
            tier_breakdown={0: 5, 2: 10},
            elapsed_seconds=0.5,
        )
        defaults.update(kwargs)
        return BenchmarkReport(**defaults)

    def test_report_json_export(self):
        """BenchmarkReport serializes to valid JSON."""
        report = self._make_report()
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["dataset"] == "febrl"
        assert parsed["precision"] == 0.75
        # tier_breakdown keys become strings in JSON
        assert "0" in parsed["tier_breakdown"] or 0 in parsed["tier_breakdown"]

    def test_report_roundtrip(self):
        """BenchmarkReport can be deserialized from JSON."""
        report = self._make_report()
        j = report.to_json()
        report2 = BenchmarkReport.from_json(j)
        assert report2.dataset == report.dataset
        assert report2.precision == report.precision
        assert report2.tier_breakdown == report.tier_breakdown

    def test_report_save_json(self):
        """save_json writes a readable JSON file."""
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.save_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["matcher"] == "cascade"
        finally:
            os.unlink(path)

    def test_report_save_csv(self):
        """save_csv creates a CSV file."""
        import csv
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        # Delete first so append mode creates fresh
        os.unlink(path)
        try:
            report.save_csv(path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["dataset"] == "febrl"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_report_summary_contains_key_info(self):
        report = self._make_report()
        summary = report.summary()
        assert "febrl" in summary
        assert "cascade" in summary
        assert "Precision" in summary
