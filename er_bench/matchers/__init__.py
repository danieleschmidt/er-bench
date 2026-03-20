"""Entity matchers for er-bench."""

from er_bench.matchers.base import EntityMatcher, MatchResult
from er_bench.matchers.exact import ExactMatcher
from er_bench.matchers.fuzzy import FuzzyMatcher
from er_bench.matchers.embedding import EmbeddingMatcher
from er_bench.matchers.cascade import CascadeMatcher

MATCHERS = {
    "exact": ExactMatcher,
    "fuzzy": FuzzyMatcher,
    "embedding": EmbeddingMatcher,
    "cascade": CascadeMatcher,
}

__all__ = [
    "EntityMatcher",
    "MatchResult",
    "ExactMatcher",
    "FuzzyMatcher",
    "EmbeddingMatcher",
    "CascadeMatcher",
    "MATCHERS",
]
