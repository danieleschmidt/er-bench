"""
BenchmarkReport: structured results from a benchmark run.

Supports JSON and CSV export.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class BenchmarkReport:
    """
    Results from a single benchmark run.

    Attributes:
        dataset: Name of the dataset used (e.g., "febrl", "icij").
        matcher: Name of the matcher used (e.g., "cascade", "fuzzy").
        n_records: Number of records in the dataset.
        n_ground_truth: Number of true matching pairs.
        n_predicted: Number of predicted candidate pairs.
        precision: Precision score (0.0–1.0).
        recall: Recall score (0.0–1.0).
        f1: F1 score (0.0–1.0).
        reduction_ratio: Fraction of all possible pairs pruned.
        pairs_completeness: Alias for recall (ER convention).
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tier_breakdown: Dict mapping tier number → count of pairs caught.
        elapsed_seconds: Wall-clock time for the run.
        timestamp: ISO 8601 timestamp of the run.
        notes: Optional free-form notes.
        extra: Dict for any additional metrics or metadata.
    """

    dataset: str
    matcher: str
    n_records: int
    n_ground_truth: int
    n_predicted: int
    precision: float
    recall: float
    f1: float
    reduction_ratio: float
    pairs_completeness: float
    tp: int
    fp: int
    fn: int
    tier_breakdown: Dict[int, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        d = asdict(self)
        # JSON-safe tier_breakdown keys
        d["tier_breakdown"] = {str(k): v for k, v in d["tier_breakdown"].items()}
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Write report to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def save_csv(self, path: str) -> None:
        """Append (or create) a CSV file with one row per report."""
        d = self.to_dict()
        # Flatten tier_breakdown into a single string for CSV
        d["tier_breakdown"] = json.dumps(d["tier_breakdown"])
        d["extra"] = json.dumps(d["extra"])

        file_exists = os.path.isfile(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(d.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkReport":
        """Deserialize from a dict."""
        # Convert string keys back to ints for tier_breakdown
        tb = d.get("tier_breakdown", {})
        d["tier_breakdown"] = {int(k): v for k, v in tb.items()}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "BenchmarkReport":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    @classmethod
    def load_json(cls, path: str) -> "BenchmarkReport":
        """Load report from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"=== BenchmarkReport ===",
            f"Dataset:  {self.dataset}",
            f"Matcher:  {self.matcher}",
            f"Records:  {self.n_records}",
            f"GT pairs: {self.n_ground_truth}",
            f"",
            f"Precision:         {self.precision:.4f}",
            f"Recall:            {self.recall:.4f}",
            f"F1:                {self.f1:.4f}",
            f"Reduction Ratio:   {self.reduction_ratio:.4f}",
            f"Pairs Completeness:{self.pairs_completeness:.4f}",
            f"",
            f"TP: {self.tp}  FP: {self.fp}  FN: {self.fn}",
        ]
        if self.tier_breakdown:
            lines.append(f"Tier breakdown: {self.tier_breakdown}")
        if self.elapsed_seconds:
            lines.append(f"Elapsed: {self.elapsed_seconds:.2f}s")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
