"""
Jaro-Winkler fuzzy matcher.

Uses Jaro-Winkler string similarity for fuzzy name matching.
Default threshold: 0.88 (validated in DocGraph dissertation on ICIJ data).

Uses jellyfish if available, falls back to pure-Python implementation.
"""

from __future__ import annotations

from typing import List

from er_bench.matchers.base import EntityMatcher, MatchResult, normalize_text


# ---------------------------------------------------------------------------
# Pure-Python Jaro-Winkler (fallback — no external dependencies required)
# ---------------------------------------------------------------------------

def _jaro(s1: str, s2: str) -> float:
    """Compute Jaro similarity."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    match_dist = max(len(s1), len(s2)) // 2 - 1
    match_dist = max(0, match_dist)

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    transpositions = 0

    for i, ch in enumerate(s1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len(s2))
        for j in range(start, end):
            if s2_matches[j] or ch != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if not matches:
        return 0.0

    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches) / 3


def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity (p = prefix scale factor)."""
    jaro_sim = _jaro(s1, s2)
    # Common prefix (up to 4 chars)
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return jaro_sim + prefix * p * (1 - jaro_sim)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    Prefers jellyfish library for speed; falls back to pure Python.
    """
    try:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(s1, s2)
    except ImportError:
        return _jaro_winkler(s1, s2)


class FuzzyMatcher(EntityMatcher):
    """
    Jaro-Winkler fuzzy name matcher.

    Computes Jaro-Winkler similarity between all pairs of normalized names
    and returns pairs above the threshold.

    Caution: O(n²) complexity — use blocking for large datasets (>10k records).

    Default threshold: 0.88 (from DocGraph dissertation validation on ICIJ data).

    Expected performance:
        - Precision: ~0.85–0.95 (some false positives with common names)
        - Recall: ~0.70–0.85 (catches most typos and transpositions)
    """

    name = "fuzzy"

    def __init__(self, threshold: float = 0.88, max_records: int = 5000):
        """
        Args:
            threshold: Jaro-Winkler similarity threshold for a match (0.0–1.0).
            max_records: Safety limit to prevent O(n²) blowup.
        """
        self.threshold = threshold
        self.max_records = max_records

    def match(self, records: List[dict]) -> List[MatchResult]:
        """Find record pairs with Jaro-Winkler similarity >= threshold."""
        if len(records) > self.max_records:
            raise ValueError(
                f"FuzzyMatcher: {len(records)} records exceeds max_records={self.max_records}. "
                "Use blocking or increase max_records."
            )

        normalized = [
            (rec["record_id"], normalize_text(rec.get("name", "")))
            for rec in records
        ]

        results = []
        for i in range(len(normalized)):
            rid1, name1 = normalized[i]
            if not name1:
                continue
            for j in range(i + 1, len(normalized)):
                rid2, name2 = normalized[j]
                if not name2:
                    continue
                score = jaro_winkler_similarity(name1, name2)
                if score >= self.threshold:
                    id1, id2 = (rid1, rid2) if rid1 <= rid2 else (rid2, rid1)
                    results.append(MatchResult(
                        id1=id1,
                        id2=id2,
                        score=score,
                        tier=2,
                        matcher=self.name,
                    ))

        return results
