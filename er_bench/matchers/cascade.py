"""
5-Tier Cascade Entity Matcher.

Implements the entity fusion cascade from the DocGraph dissertation
(Schmidt, D., 2024, "DocGraph: Entity-Centric Knowledge Graph Construction
from Heterogeneous Document Collections").

Tiers:
    0: Alias normalization (lowercase, strip punctuation, normalize unicode)
    1: Type normalization (skip cross-type pairs if entity_type present)
    2: Jaro-Winkler fuzzy matching (θ=0.88)
    3: TF-IDF cosine similarity (θ=0.75)
    4: Token set ratio (θ=0.90) — stub for LLM verification

Each tier processes candidate pairs not yet resolved by earlier tiers.
This progressive refinement maximizes recall while controlling precision.

Validated at 97.6% recall on 125,456 ICIJ Offshore Leaks identity pairs
using Tiers 0–2 only (Schmidt, 2024).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional

from er_bench.matchers.base import EntityMatcher, MatchResult, normalize_text
from er_bench.matchers.fuzzy import jaro_winkler_similarity
from er_bench.matchers.embedding import EmbeddingMatcher, _build_tfidf_vectors, _cosine


def _token_set_ratio(s1: str, s2: str) -> float:
    """
    Token set ratio similarity.

    Splits strings into token sets and computes Jaccard similarity
    on the sorted token intersection/union.

    This is the Tier 4 stub. In DocGraph, this tier is replaced by
    LLM-based verification for ambiguous pairs.
    """
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


class CascadeMatcher(EntityMatcher):
    """
    5-Tier Cascade Entity Matcher (DocGraph architecture).

    Processes all record pairs through successive tiers:

        Tier 0: Exact match on normalized names
                (capitalizes easy wins with zero computation)
        Tier 1: Type gating — skip cross-type pairs (e.g., Individual vs Company)
        Tier 2: Jaro-Winkler similarity ≥ threshold_jw (default 0.88)
        Tier 3: TF-IDF cosine similarity ≥ threshold_cosine (default 0.75)
        Tier 4: Token set ratio ≥ threshold_token_set (default 0.90)
                [Stub: in production, this is LLM verification]

    Each matched pair is annotated with the tier that caught it.
    Pairs already matched by earlier tiers are not re-evaluated.

    Usage:
        matcher = CascadeMatcher()
        results = matcher.match(records)
        for r in results:
            print(f"{r.id1} <-> {r.id2} score={r.score:.3f} tier={r.tier}")
    """

    name = "cascade"

    def __init__(
        self,
        threshold_jw: float = 0.88,
        threshold_cosine: float = 0.75,
        threshold_token_set: float = 0.90,
        use_type_gating: bool = True,
        max_records: int = 5000,
    ):
        """
        Args:
            threshold_jw: Jaro-Winkler threshold for Tier 2 (default 0.88).
            threshold_cosine: TF-IDF cosine threshold for Tier 3 (default 0.75).
            threshold_token_set: Token set ratio threshold for Tier 4 (default 0.90).
            use_type_gating: If True, skip cross-type pairs in Tier 1.
            max_records: Safety limit for O(n²) pairwise comparison.
        """
        self.threshold_jw = threshold_jw
        self.threshold_cosine = threshold_cosine
        self.threshold_token_set = threshold_token_set
        self.use_type_gating = use_type_gating
        self.max_records = max_records

    def match(self, records: List[dict]) -> List[MatchResult]:
        """
        Run the 5-tier cascade on a list of records.

        Returns:
            List of MatchResult, each annotated with the tier that matched it.
        """
        if len(records) > self.max_records:
            raise ValueError(
                f"CascadeMatcher: {len(records)} records exceeds max_records={self.max_records}."
            )

        if len(records) < 2:
            return []

        results: List[MatchResult] = []
        matched_pairs: Set[Tuple[str, str]] = set()  # dedup tracker

        def _add(id1: str, id2: str, score: float, tier: int) -> None:
            pair = (id1, id2) if id1 <= id2 else (id2, id1)
            if pair not in matched_pairs:
                matched_pairs.add(pair)
                results.append(MatchResult(
                    id1=pair[0], id2=pair[1], score=score, tier=tier, matcher=self.name
                ))

        # Tier 0: Normalize all names
        normalized: List[Tuple[str, str, str]] = []  # (record_id, norm_name, entity_type)
        for rec in records:
            rid = rec["record_id"]
            norm = normalize_text(rec.get("name", ""))
            etype = rec.get("entity_type", "")
            normalized.append((rid, norm, etype))

        # Tier 0 + Exact match: group by normalized name
        buckets: defaultdict = defaultdict(list)
        for rid, norm, etype in normalized:
            if norm:
                buckets[norm].append((rid, etype))

        for name_key, bucket_entries in buckets.items():
            if len(bucket_entries) < 2:
                continue
            for i in range(len(bucket_entries)):
                for j in range(i + 1, len(bucket_entries)):
                    rid1, et1 = bucket_entries[i]
                    rid2, et2 = bucket_entries[j]
                    # Tier 1: type gating
                    if self.use_type_gating and et1 and et2 and et1 != et2:
                        continue
                    _add(rid1, rid2, 1.0, tier=0)

        # Tier 2: Jaro-Winkler on all unmatched pairs
        for i in range(len(normalized)):
            rid1, norm1, et1 = normalized[i]
            if not norm1:
                continue
            for j in range(i + 1, len(normalized)):
                rid2, norm2, et2 = normalized[j]
                if not norm2:
                    continue
                pair = (rid1, rid2) if rid1 <= rid2 else (rid2, rid1)
                if pair in matched_pairs:
                    continue
                # Tier 1: type gating
                if self.use_type_gating and et1 and et2 and et1 != et2:
                    continue
                score = jaro_winkler_similarity(norm1, norm2)
                if score >= self.threshold_jw:
                    _add(rid1, rid2, score, tier=2)

        # Tier 3: TF-IDF cosine on all unmatched pairs
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [n[1] for n in normalized]
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
            matrix = vectorizer.fit_transform(texts)
            norms_arr = np.array(matrix.multiply(matrix).sum(axis=1)).flatten() ** 0.5
            norms_arr[norms_arr == 0] = 1.0
            use_sklearn = True
        except ImportError:
            texts = [n[1] for n in normalized]
            vectors = _build_tfidf_vectors(texts)
            use_sklearn = False

        for i in range(len(normalized)):
            rid1, norm1, et1 = normalized[i]
            if not norm1:
                continue
            for j in range(i + 1, len(normalized)):
                rid2, norm2, et2 = normalized[j]
                if not norm2:
                    continue
                pair = (rid1, rid2) if rid1 <= rid2 else (rid2, rid1)
                if pair in matched_pairs:
                    continue
                if self.use_type_gating and et1 and et2 and et1 != et2:
                    continue
                if use_sklearn:
                    import numpy as np
                    row_i = matrix[i].toarray().flatten() / norms_arr[i]  # type: ignore
                    row_j = matrix[j].toarray().flatten() / norms_arr[j]  # type: ignore
                    score = float(np.dot(row_i, row_j))
                else:
                    score = _cosine(vectors[i], vectors[j])  # type: ignore
                if score >= self.threshold_cosine:
                    _add(rid1, rid2, score, tier=3)

        # Tier 4: Token set ratio (stub for LLM verification)
        # In production (DocGraph): unresolved high-confidence candidates
        # are sent to an LLM for final verification. Here we use token set
        # ratio as a deterministic proxy.
        for i in range(len(normalized)):
            rid1, norm1, et1 = normalized[i]
            if not norm1:
                continue
            for j in range(i + 1, len(normalized)):
                rid2, norm2, et2 = normalized[j]
                if not norm2:
                    continue
                pair = (rid1, rid2) if rid1 <= rid2 else (rid2, rid1)
                if pair in matched_pairs:
                    continue
                if self.use_type_gating and et1 and et2 and et1 != et2:
                    continue
                # TODO: Replace with LLM call for production use
                # Candidate prompt: "Are '{name1}' and '{name2}' the same entity? Answer YES/NO."
                score = _token_set_ratio(norm1, norm2)
                if score >= self.threshold_token_set:
                    _add(rid1, rid2, score, tier=4)

        return results

    def tier_breakdown(self, records: List[dict]) -> Dict[int, int]:
        """Return count of matches caught by each tier."""
        results = self.match(records)
        counts: Dict[int, int] = defaultdict(int)
        for r in results:
            counts[r.tier] += 1
        return dict(counts)
