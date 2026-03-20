"""
TF-IDF cosine similarity matcher.

Computes cosine similarity between TF-IDF vectors of entity names.
No external ML model required — uses scikit-learn's TfidfVectorizer.

Falls back to manual TF-IDF if scikit-learn is not available.

Default threshold: 0.75 (Tier 3 in the DocGraph cascade).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Tuple

from er_bench.matchers.base import EntityMatcher, MatchResult, normalize_text


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors (dicts)."""
    dot = sum(v1.get(k, 0.0) * v for k, v in v2.items())
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _build_tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    """
    Build TF-IDF vectors without scikit-learn (char n-gram based).
    Uses character trigrams for better fuzzy matching.
    """
    def ngrams(text: str, n: int = 3) -> List[str]:
        padded = f"{'_' * (n-1)}{text}{'_' * (n-1)}"
        return [padded[i:i+n] for i in range(len(padded) - n + 1)]

    # Build term-document frequency
    doc_term_counts = []
    doc_freq: Counter = Counter()
    for text in texts:
        grams = ngrams(text)
        tc: Counter = Counter(grams)
        doc_term_counts.append(tc)
        doc_freq.update(set(tc.keys()))

    n_docs = len(texts)
    vectors = []
    for tc in doc_term_counts:
        total = sum(tc.values()) or 1
        vec = {}
        for term, count in tc.items():
            tf = count / total
            idf = math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1.0
            vec[term] = tf * idf
        vectors.append(vec)

    return vectors


class EmbeddingMatcher(EntityMatcher):
    """
    TF-IDF cosine similarity matcher (Tier 3 in DocGraph cascade).

    Vectorizes entity names using TF-IDF and finds pairs with cosine
    similarity above the threshold.

    Uses scikit-learn's TfidfVectorizer with char_wb analyzer if available;
    falls back to manual char-trigram TF-IDF.

    Note: O(n²) for dense comparison. For large datasets, use LSH or
    approximate nearest-neighbor search.

    Expected performance:
        - Complements fuzzy matching by catching abbreviation and word-order variants.
        - Threshold 0.75 gives good precision/recall tradeoff.
    """

    name = "embedding"

    def __init__(self, threshold: float = 0.75, max_records: int = 3000):
        """
        Args:
            threshold: Cosine similarity threshold (0.0–1.0).
            max_records: Safety limit for O(n²) comparison.
        """
        self.threshold = threshold
        self.max_records = max_records

    def _vectorize_sklearn(self, texts: List[str]):
        """Vectorize using scikit-learn TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return matrix

    def match(self, records: List[dict]) -> List[MatchResult]:
        """Find pairs with cosine TF-IDF similarity >= threshold."""
        if len(records) > self.max_records:
            raise ValueError(
                f"EmbeddingMatcher: {len(records)} records exceeds max_records={self.max_records}."
            )

        ids = [rec["record_id"] for rec in records]
        texts = [normalize_text(rec.get("name", "")) for rec in records]

        results = []

        try:
            import numpy as np
            matrix = self._vectorize_sklearn(texts)
            # Compute pairwise cosine similarities
            # matrix shape: (n, vocab)
            norms = np.array(matrix.multiply(matrix).sum(axis=1)).flatten() ** 0.5
            norms[norms == 0] = 1.0

            for i in range(len(ids)):
                row_i = matrix[i].toarray().flatten() / norms[i]
                for j in range(i + 1, len(ids)):
                    row_j = matrix[j].toarray().flatten() / norms[j]
                    score = float(np.dot(row_i, row_j))
                    if score >= self.threshold:
                        id1, id2 = (ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i])
                        results.append(MatchResult(
                            id1=id1, id2=id2, score=score, tier=3, matcher=self.name
                        ))

        except ImportError:
            # Fallback: manual char-trigram TF-IDF
            vectors = _build_tfidf_vectors(texts)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    score = _cosine(vectors[i], vectors[j])
                    if score >= self.threshold:
                        id1, id2 = (ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i])
                        results.append(MatchResult(
                            id1=id1, id2=id2, score=score, tier=3, matcher=self.name
                        ))

        return results
