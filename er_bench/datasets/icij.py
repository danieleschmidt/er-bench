"""
ICIJ Offshore Leaks dataset loader.

Loads entity records from the ICIJ Offshore Leaks database export (CSV format).
Expects a CSV with the public field schema:
    node_id, name, country, entity_type

For full dataset download, see: https://offshoreleaks.icij.org/pages/database

A 100-record sample is included at data/icij_sample.csv.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple, Set, Optional


DEFAULT_SAMPLE_PATH = Path(__file__).parent.parent.parent / "data" / "icij_sample.csv"

REQUIRED_FIELDS = {"node_id", "name", "country", "entity_type"}


class ICIJDataset:
    """
    Loader for ICIJ Offshore Leaks entity data.

    The full ICIJ Offshore Leaks database is available at:
    https://offshoreleaks.icij.org/pages/database

    Download the 'Nodes' CSV, which contains node_id, name, country, entity_type.

    A 100-record sample is bundled at data/icij_sample.csv for quick testing.

    Example:
        ds = ICIJDataset()  # uses bundled sample
        records, ground_truth = ds.load()

        ds = ICIJDataset(path="/path/to/nodes.csv", ground_truth_path="/path/to/gt.csv")
        records, ground_truth = ds.load()
    """

    name = "icij"
    description = "ICIJ Offshore Leaks entity dataset (sample bundled, full via download)"

    def __init__(
        self,
        path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
    ):
        """
        Args:
            path: Path to ICIJ nodes CSV. Defaults to bundled sample.
            ground_truth_path: Path to ground truth pairs CSV (node_id_1, node_id_2).
                               If None, returns an empty ground truth set.
        """
        self.path = Path(path) if path else DEFAULT_SAMPLE_PATH
        self.ground_truth_path = Path(ground_truth_path) if ground_truth_path else None

    def load(self) -> Tuple[List[dict], Set[Tuple[str, str]]]:
        """
        Load records from CSV.

        Returns:
            records: List of dicts with keys: node_id, name, country, entity_type.
            ground_truth: Set of (id1, id2) pairs (empty if no GT file provided).
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"ICIJ data file not found: {self.path}\n"
                "Download from https://offshoreleaks.icij.org/pages/database\n"
                "Or use the bundled sample: ICIJDataset() (no args)"
            )

        records = []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                actual = set(reader.fieldnames)
                missing = REQUIRED_FIELDS - actual
                if missing:
                    raise ValueError(
                        f"ICIJ CSV missing required columns: {missing}\n"
                        f"Expected: {REQUIRED_FIELDS}\n"
                        f"Found: {actual}"
                    )
            for row in reader:
                records.append({
                    "record_id": row["node_id"],
                    "node_id": row["node_id"],
                    "name": row.get("name", "").strip(),
                    "country": row.get("country", "").strip(),
                    "entity_type": row.get("entity_type", "").strip(),
                })

        ground_truth: Set[Tuple[str, str]] = set()
        if self.ground_truth_path and self.ground_truth_path.exists():
            with open(self.ground_truth_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    id1 = row.get("node_id_1", "").strip()
                    id2 = row.get("node_id_2", "").strip()
                    if id1 and id2:
                        ground_truth.add(tuple(sorted([id1, id2])))  # type: ignore[arg-type]

        return records, ground_truth
