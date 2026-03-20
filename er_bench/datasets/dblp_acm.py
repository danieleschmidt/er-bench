"""
DBLP-ACM dataset loader.

Loads the DBLP-ACM publication matching dataset in Magellan format.
This dataset contains academic publication records from DBLP and ACM
that need to be matched across the two sources.

For full dataset: https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
Or via Magellan data repository: https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository

A 50-record sample is bundled at data/dblp_acm_sample.csv for quick testing.

Schema:
    id, title, authors, venue, year, source  (source = 'dblp' or 'acm')

Ground truth format:
    ltable_id, rtable_id  (matching pairs across sources)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Set, Optional


DEFAULT_SAMPLE_PATH = Path(__file__).parent.parent.parent / "data" / "dblp_acm_sample.csv"
DEFAULT_GT_PATH = Path(__file__).parent.parent.parent / "data" / "dblp_acm_gt.csv"

REQUIRED_FIELDS = {"id", "title", "authors", "venue", "year", "source"}


class DBLPACMDataset:
    """
    Loader for the DBLP-ACM publication matching dataset.

    The full dataset is available via the Magellan data repository:
    https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository

    A 50-record sample (25 DBLP + 25 ACM, with known matches) is bundled
    at data/dblp_acm_sample.csv.

    Example:
        ds = DBLPACMDataset()  # uses bundled sample
        records, ground_truth = ds.load()

        ds = DBLPACMDataset(path="/path/to/dblp_acm.csv", gt_path="/path/to/gt.csv")
        records, ground_truth = ds.load()
    """

    name = "dblp_acm"
    description = "DBLP-ACM publication matching dataset (Magellan format)"

    def __init__(
        self,
        path: Optional[str] = None,
        gt_path: Optional[str] = None,
    ):
        """
        Args:
            path: Path to records CSV. Defaults to bundled sample.
            gt_path: Path to ground truth CSV. Defaults to bundled GT if it exists.
        """
        self.path = Path(path) if path else DEFAULT_SAMPLE_PATH
        self.gt_path = Path(gt_path) if gt_path else DEFAULT_GT_PATH

    def load(self) -> Tuple[List[dict], Set[Tuple[str, str]]]:
        """
        Load records from CSV.

        Returns:
            records: List of dicts with keys: id, title, authors, venue, year, source.
                     Also includes 'record_id' (= id) and 'name' (= title) for
                     compatibility with matchers that expect these fields.
            ground_truth: Set of (id1, id2) pairs.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"DBLP-ACM data file not found: {self.path}\n"
                "Download from: https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository\n"
                "Or use the bundled sample: DBLPACMDataset() (no args)"
            )

        records = []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "record_id": row.get("id", "").strip(),
                    "id": row.get("id", "").strip(),
                    "name": row.get("title", "").strip(),  # alias for matchers
                    "title": row.get("title", "").strip(),
                    "authors": row.get("authors", "").strip(),
                    "venue": row.get("venue", "").strip(),
                    "year": row.get("year", "").strip(),
                    "source": row.get("source", "").strip(),
                    "entity_type": "Publication",  # uniform type for all
                })

        ground_truth: Set[Tuple[str, str]] = set()
        if self.gt_path.exists():
            with open(self.gt_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    id1 = row.get("ltable_id", row.get("id1", "")).strip()
                    id2 = row.get("rtable_id", row.get("id2", "")).strip()
                    if id1 and id2:
                        ground_truth.add(tuple(sorted([id1, id2])))  # type: ignore[arg-type]

        return records, ground_truth
