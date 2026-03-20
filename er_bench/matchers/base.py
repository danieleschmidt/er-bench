"""
Abstract base class for entity matchers.

All matchers implement:
    match(records) -> List[MatchResult]

where MatchResult is a (id1, id2, score, tier) named tuple.
"""

from __future__ import annotations

import unicodedata
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional


@dataclass
class MatchResult:
    """A candidate match between two records."""
    id1: str
    id2: str
    score: float
    tier: int  # Which cascade tier produced this match (0–4)
    matcher: str = ""

    def as_pair(self) -> Tuple[str, str]:
        """Return the pair with smaller id first."""
        a, b = self.id1, self.id2
        return (a, b) if a <= b else (b, a)


def normalize_text(text: str) -> str:
    """
    Tier 0 normalization: lowercase, strip punctuation, normalize unicode.

    This is the baseline normalization applied before any matching.
    """
    # Normalize unicode (NFKD → strip accents)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    # Strip punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EntityMatcher(ABC):
    """
    Abstract base class for entity matchers.

    Subclasses must implement `match(records)`.

    records: List of dicts, each with at minimum:
        - record_id (str): unique identifier
        - name (str): entity name

    Optional fields used by some matchers:
        - entity_type (str): used for Tier 1 type filtering
        - country (str): geographic attribute

    Returns:
        List[MatchResult]: candidate matching pairs with scores and tier labels.
    """

    name: str = "base"

    @abstractmethod
    def match(self, records: List[dict]) -> List[MatchResult]:
        """
        Find candidate matching pairs among records.

        Args:
            records: List of record dicts.

        Returns:
            List of MatchResult objects.
        """
        ...

    def match_pairs(self, records: List[dict]) -> Set[Tuple[str, str]]:
        """Convenience: return just the set of (id1, id2) pairs."""
        results = self.match(records)
        return {r.as_pair() for r in results}
