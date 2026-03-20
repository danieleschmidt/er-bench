"""Tests for the 5-tier cascade matcher."""

import pytest


class TestCascadeMatcher:
    def _records(self, *name_pairs):
        """Build a record list from (id, name, entity_type) tuples."""
        return [
            {
                "record_id": rid,
                "name": name,
                "entity_type": etype,
            }
            for rid, name, etype in name_pairs
        ]

    def test_cascade_tier_tracking_exact(self):
        """Exact-normalized pairs should be caught by tier 0."""
        from er_bench.matchers.cascade import CascadeMatcher
        records = [
            {"record_id": "A", "name": "James Smith", "entity_type": "Individual"},
            {"record_id": "B", "name": "james smith", "entity_type": "Individual"},
        ]
        matcher = CascadeMatcher()
        results = matcher.match(records)
        pairs = {r.as_pair(): r for r in results}
        assert ("A", "B") in pairs
        assert pairs[("A", "B")].tier == 0

    def test_cascade_tier_tracking_fuzzy(self):
        """Typo pairs should be caught by tier 2 (Jaro-Winkler)."""
        from er_bench.matchers.cascade import CascadeMatcher
        records = [
            {"record_id": "A", "name": "Schmidt", "entity_type": "Individual"},
            {"record_id": "B", "name": "Schmdit", "entity_type": "Individual"},
        ]
        matcher = CascadeMatcher()
        results = matcher.match(records)
        pairs = {r.as_pair(): r for r in results}
        assert ("A", "B") in pairs
        assert pairs[("A", "B")].tier == 2

    def test_cascade_type_gating(self):
        """Cross-type pairs should be skipped when use_type_gating=True."""
        from er_bench.matchers.cascade import CascadeMatcher
        records = [
            {"record_id": "A", "name": "Smith Holdings", "entity_type": "Individual"},
            {"record_id": "B", "name": "Smith Holdings", "entity_type": "Company"},
        ]
        matcher = CascadeMatcher(use_type_gating=True)
        results = matcher.match(records)
        pairs = {r.as_pair() for r in results}
        # Cross-type should be skipped
        assert ("A", "B") not in pairs

    def test_cascade_type_gating_disabled(self):
        """With type gating off, cross-type exact matches should be found."""
        from er_bench.matchers.cascade import CascadeMatcher
        records = [
            {"record_id": "A", "name": "Smith Holdings", "entity_type": "Individual"},
            {"record_id": "B", "name": "Smith Holdings", "entity_type": "Company"},
        ]
        matcher = CascadeMatcher(use_type_gating=False)
        results = matcher.match(records)
        pairs = {r.as_pair() for r in results}
        assert ("A", "B") in pairs

    def test_cascade_no_duplicate_pairs(self):
        """Each pair should appear at most once in results."""
        from er_bench.matchers.cascade import CascadeMatcher
        records = [
            {"record_id": "A", "name": "apex holdings", "entity_type": "Company"},
            {"record_id": "B", "name": "apex holdings", "entity_type": "Company"},
            {"record_id": "C", "name": "apex holdings ltd", "entity_type": "Company"},
        ]
        matcher = CascadeMatcher()
        results = matcher.match(records)
        pairs = [r.as_pair() for r in results]
        assert len(pairs) == len(set(pairs)), "Duplicate pairs found in cascade results"

    def test_cascade_recall_on_febrl(self):
        """Cascade should achieve > 85% recall on a small Febrl dataset."""
        from er_bench.datasets.febrl import FebrlDataset
        from er_bench.matchers.cascade import CascadeMatcher
        from er_bench.metrics.standard import recall

        ds = FebrlDataset(n_records=200, duplicate_rate=0.20, seed=0)
        records, gt = ds.as_dicts()

        matcher = CascadeMatcher()
        predicted = matcher.match_pairs(records)
        r = recall(predicted, gt)
        assert r > 0.85, f"Cascade recall too low: {r:.4f} (expected > 0.85)"

    def test_cascade_tier_breakdown_keys(self):
        """tier_breakdown should return int keys."""
        from er_bench.matchers.cascade import CascadeMatcher
        records = [
            {"record_id": "A", "name": "James Smith", "entity_type": "Individual"},
            {"record_id": "B", "name": "james smith", "entity_type": "Individual"},
        ]
        matcher = CascadeMatcher()
        breakdown = matcher.tier_breakdown(records)
        for k in breakdown.keys():
            assert isinstance(k, int), f"Expected int key, got {type(k)}"

    def test_cascade_empty_records(self):
        from er_bench.matchers.cascade import CascadeMatcher
        matcher = CascadeMatcher()
        results = matcher.match([])
        assert results == []

    def test_cascade_single_record(self):
        from er_bench.matchers.cascade import CascadeMatcher
        records = [{"record_id": "A", "name": "Lone Entity", "entity_type": "Individual"}]
        matcher = CascadeMatcher()
        results = matcher.match(records)
        assert results == []
