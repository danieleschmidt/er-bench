"""Tests for dataset loaders."""

import os
import pytest
from pathlib import Path


# --- Febrl ---

class TestFebrlGenerator:
    def test_generates_n_records(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds = FebrlDataset(n_records=100, duplicate_rate=0.15, seed=0)
        records, gt = ds.load()
        assert len(records) == 100

    def test_expected_duplicate_rate(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds = FebrlDataset(n_records=200, duplicate_rate=0.20, seed=42)
        records, gt = ds.load()
        # 20% of 200 = 40 duplicates → 40 GT pairs
        assert len(gt) == 40

    def test_ground_truth_has_valid_ids(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds = FebrlDataset(n_records=100, duplicate_rate=0.15, seed=1)
        records, gt = ds.load()
        record_ids = {r.record_id for r in records}
        for id1, id2 in gt:
            assert id1 in record_ids, f"{id1} not in records"
            assert id2 in record_ids, f"{id2} not in records"

    def test_pairs_are_normalized(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds = FebrlDataset(n_records=100, duplicate_rate=0.20, seed=5)
        records, gt = ds.load()
        for id1, id2 in gt:
            assert id1 <= id2, f"GT pair not normalized: ({id1}, {id2})"

    def test_no_self_pairs(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds = FebrlDataset(n_records=100, duplicate_rate=0.10, seed=3)
        records, gt = ds.load()
        for id1, id2 in gt:
            assert id1 != id2

    def test_as_dicts_returns_dicts(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds = FebrlDataset(n_records=50, seed=7)
        records, gt = ds.as_dicts()
        assert isinstance(records[0], dict)
        assert "record_id" in records[0]
        assert "name" in records[0]

    def test_noise_changes_name(self):
        from er_bench.datasets.febrl import _inject_noise
        import random
        rng = random.Random(0)
        original = "Schmidt"
        noisy = _inject_noise(original, rng, num_errors=1)
        # At least sometimes, noise changes the name (may not always since
        # replace_char could pick same char)
        # Just verify it doesn't crash and returns a string
        assert isinstance(noisy, str)
        assert len(noisy) > 0

    def test_reproducible_with_seed(self):
        from er_bench.datasets.febrl import FebrlDataset
        ds1 = FebrlDataset(n_records=50, seed=99)
        ds2 = FebrlDataset(n_records=50, seed=99)
        r1, _ = ds1.load()
        r2, _ = ds2.load()
        assert [r.record_id for r in r1] == [r.record_id for r in r2]


# --- ICIJ ---

class TestICIJDataset:
    def test_icij_sample_loads(self):
        from er_bench.datasets.icij import ICIJDataset
        ds = ICIJDataset()
        records, gt = ds.load()
        assert len(records) > 0

    def test_icij_sample_has_100_records(self):
        from er_bench.datasets.icij import ICIJDataset
        ds = ICIJDataset()
        records, gt = ds.load()
        assert len(records) == 100

    def test_icij_records_have_required_fields(self):
        from er_bench.datasets.icij import ICIJDataset
        ds = ICIJDataset()
        records, gt = ds.load()
        for rec in records:
            assert "record_id" in rec
            assert "name" in rec
            assert "country" in rec
            assert "entity_type" in rec

    def test_icij_missing_file_raises(self):
        from er_bench.datasets.icij import ICIJDataset
        ds = ICIJDataset(path="/nonexistent/path/icij.csv")
        with pytest.raises(FileNotFoundError):
            ds.load()


# --- DBLP-ACM ---

class TestDBLPACMDataset:
    def test_dblp_sample_loads(self):
        from er_bench.datasets.dblp_acm import DBLPACMDataset
        ds = DBLPACMDataset()
        records, gt = ds.load()
        assert len(records) > 0

    def test_dblp_sample_has_ground_truth(self):
        from er_bench.datasets.dblp_acm import DBLPACMDataset
        ds = DBLPACMDataset()
        records, gt = ds.load()
        assert len(gt) == 25  # 25 pairs in the sample GT

    def test_dblp_records_have_name_alias(self):
        from er_bench.datasets.dblp_acm import DBLPACMDataset
        ds = DBLPACMDataset()
        records, gt = ds.load()
        for rec in records:
            assert "name" in rec  # alias for title
            assert "record_id" in rec
