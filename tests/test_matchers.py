"""Tests for entity matchers."""

import pytest
from er_bench.matchers.base import normalize_text, MatchResult


# --- Normalization ---

class TestNormalization:
    def test_lowercase(self):
        assert normalize_text("SCHMIDT") == "schmidt"

    def test_strip_punctuation(self):
        assert normalize_text("Ltd.") == "ltd"

    def test_normalize_unicode(self):
        result = normalize_text("Müller")
        assert "u" in result  # accent stripped
        assert "ü" not in result

    def test_extra_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"


# --- Exact Matcher ---

class TestExactMatcher:
    def _make_records(self, pairs):
        records = []
        for i, (name,) in enumerate([(n,) for n in pairs]):
            records.append({"record_id": f"R{i}", "name": name})
        return records

    def test_exact_matcher_precision(self):
        """Exact matches on known identical pairs → all found."""
        from er_bench.matchers.exact import ExactMatcher
        records = [
            {"record_id": "A", "name": "James Smith"},
            {"record_id": "B", "name": "James Smith"},
            {"record_id": "C", "name": "Mary Jones"},
        ]
        matcher = ExactMatcher()
        results = matcher.match(records)
        pairs = {r.as_pair() for r in results}
        assert ("A", "B") in pairs
        # C should not be in any pair
        assert not any("C" in p for p in pairs)

    def test_exact_matcher_scores_one(self):
        from er_bench.matchers.exact import ExactMatcher
        records = [
            {"record_id": "X", "name": "apex holdings ltd"},
            {"record_id": "Y", "name": "Apex Holdings Ltd"},
        ]
        matcher = ExactMatcher()
        results = matcher.match(records)
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_exact_matcher_no_false_positives_on_different(self):
        from er_bench.matchers.exact import ExactMatcher
        records = [
            {"record_id": "A", "name": "Schmidt"},
            {"record_id": "B", "name": "Schmdit"},  # typo — should NOT match
        ]
        matcher = ExactMatcher()
        results = matcher.match(records)
        assert len(results) == 0

    def test_exact_matcher_empty(self):
        from er_bench.matchers.exact import ExactMatcher
        matcher = ExactMatcher()
        results = matcher.match([])
        assert results == []


# --- Fuzzy Matcher ---

class TestFuzzyMatcher:
    def test_fuzzy_matcher_catches_typos(self):
        """'Schmidt' vs 'Schmdit' (transposition) → should match."""
        from er_bench.matchers.fuzzy import FuzzyMatcher, jaro_winkler_similarity
        # Verify the similarity is above threshold
        sim = jaro_winkler_similarity("schmidt", "schmdit")
        assert sim >= 0.88, f"Expected ≥0.88 but got {sim:.4f}"

        records = [
            {"record_id": "A", "name": "Schmidt"},
            {"record_id": "B", "name": "Schmdit"},
        ]
        matcher = FuzzyMatcher(threshold=0.88)
        results = matcher.match(records)
        pairs = {r.as_pair() for r in results}
        assert ("A", "B") in pairs

    def test_fuzzy_matcher_rejects_different_names(self):
        from er_bench.matchers.fuzzy import FuzzyMatcher
        records = [
            {"record_id": "A", "name": "James Smith"},
            {"record_id": "B", "name": "Mary Jones"},
        ]
        matcher = FuzzyMatcher(threshold=0.88)
        results = matcher.match(records)
        assert len(results) == 0

    def test_fuzzy_matcher_threshold_parameter(self):
        """Lower threshold should find more pairs."""
        from er_bench.matchers.fuzzy import FuzzyMatcher
        records = [
            {"record_id": "A", "name": "apex holdings"},
            {"record_id": "B", "name": "apex holding"},
        ]
        matcher_high = FuzzyMatcher(threshold=0.99)
        matcher_low = FuzzyMatcher(threshold=0.80)
        high_results = matcher_high.match(records)
        low_results = matcher_low.match(records)
        assert len(low_results) >= len(high_results)

    def test_pure_python_jaro_winkler(self):
        """Pure-Python implementation gives reasonable results."""
        from er_bench.matchers.fuzzy import _jaro_winkler
        sim = _jaro_winkler("schmidt", "schmidt")
        assert sim == 1.0
        sim2 = _jaro_winkler("schmidt", "jones")
        assert sim2 < 0.80

    def test_fuzzy_tier_is_2(self):
        from er_bench.matchers.fuzzy import FuzzyMatcher
        records = [
            {"record_id": "A", "name": "Schmidt"},
            {"record_id": "B", "name": "Schmdit"},
        ]
        matcher = FuzzyMatcher(threshold=0.88)
        results = matcher.match(records)
        if results:
            assert results[0].tier == 2


# --- Embedding Matcher ---

class TestEmbeddingMatcher:
    def test_embedding_matcher_finds_similar(self):
        from er_bench.matchers.embedding import EmbeddingMatcher
        records = [
            {"record_id": "A", "name": "global asset management"},
            {"record_id": "B", "name": "global asset mgmt"},
            {"record_id": "C", "name": "pacific rim investments"},
        ]
        matcher = EmbeddingMatcher(threshold=0.5)
        results = matcher.match(records)
        pairs = {r.as_pair() for r in results}
        # A and B should be similar (abbreviation variant)
        assert ("A", "B") in pairs

    def test_embedding_returns_match_results(self):
        from er_bench.matchers.embedding import EmbeddingMatcher
        records = [
            {"record_id": "X", "name": "test entity one"},
            {"record_id": "Y", "name": "test entity one"},
        ]
        matcher = EmbeddingMatcher(threshold=0.5)
        results = matcher.match(records)
        assert all(isinstance(r, MatchResult) for r in results)
