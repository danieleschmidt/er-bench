"""
Exact match entity matcher.

Matches records with identical normalized names. Uses Tier 0 normalization
(lowercase, strip punctuation, normalize unicode) before comparison.

High precision, low recall — useful as a baseline and as the first step
in a cascade pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from er_bench.matchers.base import EntityMatcher, MatchResult, normalize_text


class ExactMatcher(EntityMatcher):
    """
    Exact match on normalized name.

    Two records match if their Tier 0 normalized names are identical.

    Expected performance:
        - Precision: ~1.0 (very few false positives)
        - Recall: ~0.3–0.5 (misses typos, abbreviations, etc.)
    """

    name = "exact"

    def match(self, records: List[dict]) -> List[MatchResult]:
        """Find records with identical normalized names."""
        # Group records by normalized name
        buckets: defaultdict = defaultdict(list)
        for rec in records:
            key = normalize_text(rec.get("name", ""))
            if key:
                buckets[key].append(rec["record_id"])

        results = []
        for ids in buckets.values():
            if len(ids) < 2:
                continue
            # All pairs within the bucket
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    if id1 > id2:
                        id1, id2 = id2, id1
                    results.append(MatchResult(
                        id1=id1,
                        id2=id2,
                        score=1.0,
                        tier=0,
                        matcher=self.name,
                    ))

        return results
