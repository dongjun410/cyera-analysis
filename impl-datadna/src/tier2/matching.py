"""Tier 2: KnownTypeMatcher — 3-signal weighted scoring against known type library.

Determines if a cluster matches an existing document type without LLM invocation.
Per spec section 4.3 Step 2.
"""

from __future__ import annotations

import numpy as np

from src.types import ClusterInfo, KnownType, MatchResult


class KnownTypeMatcher:
    """Matches clusters to known document types via 3-signal weighted scoring.

    Signals:
      1. Structure signature exact match (weight=0.5)
      2. TF-IDF keyword Jaccard overlap (weight=0.3)
      3. PII distribution cosine similarity (weight=0.2)

    Method thresholds:
      - score >= high_match_threshold  → "known_match"
      - low_match_threshold <= score < high_match_threshold → "llm_confirm"
      - score < low_match_threshold    → "unknown"
    """

    def __init__(
        self,
        known_types: list[KnownType] | None = None,
        config: dict | None = None,
    ) -> None:
        self._types: dict[str, KnownType] = {}
        if known_types:
            for kt in known_types:
                self._types[kt.type_id] = kt

        # Default configuration
        self._structure_weight: float = 0.5
        self._tfidf_weight: float = 0.3
        self._pii_weight: float = 0.2
        self._high_match_threshold: float = 0.8
        self._low_match_threshold: float = 0.5

        if config:
            self._structure_weight = config.get(
                "structure_signature_weight", self._structure_weight
            )
            self._tfidf_weight = config.get(
                "tfidf_overlap_weight", self._tfidf_weight
            )
            self._pii_weight = config.get(
                "pii_distribution_weight", self._pii_weight
            )
            self._high_match_threshold = config.get(
                "high_match_threshold", self._high_match_threshold
            )
            self._low_match_threshold = config.get(
                "low_match_threshold", self._low_match_threshold
            )

    # ── Public API ────────────────────────────────────────────

    def match(self, cluster: ClusterInfo) -> MatchResult:
        """Compute 3-signal weighted score against each known type,
        returning the best MatchResult.

        The returned MatchResult includes:
          - matched_type: the best KnownType (or None if score < low_match_threshold)
          - score: weighted composite score [0.0, 1.0]
          - method: "known_match" | "llm_confirm" | "unknown"
          - match_details: per-signal breakdown dict
        """
        if not self._types:
            return MatchResult(
                matched_type=None,
                score=0.0,
                method="unknown",
                match_details={},
            )

        best_type: KnownType | None = None
        best_score: float = -1.0
        best_details: dict = {}

        for kt in self._types.values():
            signal_structure = self._compute_structure_signal(cluster, kt)
            signal_tfidf = self._compute_tfidf_signal(cluster, kt)
            signal_pii = self._compute_pii_signal(cluster, kt)

            composite = (
                self._structure_weight * signal_structure
                + self._tfidf_weight * signal_tfidf
                + self._pii_weight * signal_pii
            )

            if composite > best_score:
                best_score = composite
                best_type = kt
                best_details = {
                    "structure": signal_structure,
                    "tfidf": signal_tfidf,
                    "pii": signal_pii,
                    "composite": composite,
                }

        # best_score is guaranteed >= 0 here (all signals >= 0)

        if best_score >= self._high_match_threshold:
            method = "known_match"
        elif best_score >= self._low_match_threshold:
            method = "llm_confirm"
        else:
            method = "unknown"
            best_type = None  # No type returned for "unknown"

        return MatchResult(
            matched_type=best_type,
            score=best_score,
            method=method,
            match_details=best_details,
        )

    def register_type(self, known_type: KnownType) -> None:
        """Add a known type to the library (idempotent — overwrites by type_id)."""
        self._types[known_type.type_id] = known_type

    def get_type(self, type_id: str) -> KnownType | None:
        """Retrieve a known type by its type_id, or None if not found."""
        return self._types.get(type_id)

    def type_count(self) -> int:
        """Return the number of registered known types."""
        return len(self._types)

    # ── Signal computation ────────────────────────────────────

    @staticmethod
    def _compute_structure_signal(
        cluster: ClusterInfo, known_type: KnownType
    ) -> float:
        """Exact string comparison of structural_bucket vs structural_signature."""
        if cluster.structural_bucket == known_type.structural_signature:
            return 1.0
        return 0.0

    @staticmethod
    def _compute_tfidf_signal(
        cluster: ClusterInfo, known_type: KnownType
    ) -> float:
        """Jaccard similarity between cluster.tfidf_keywords and known_type keywords."""
        set_a = set(cluster.tfidf_keywords)
        set_b = set(known_type.tfidf_keywords)

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _compute_pii_signal(
        cluster: ClusterInfo, known_type: KnownType
    ) -> float:
        """Cosine similarity between PII distribution vectors.

        Vectors are built over the union of keys; missing keys → 0.
        If both distributions are empty (or either has zero norm) → 0.0.
        """
        cluster_pii = cluster.pii_distribution
        known_pii = known_type.pii_distribution

        # Collect all keys from both distributions
        all_keys = sorted(set(cluster_pii.keys()) | set(known_pii.keys()))

        if not all_keys:
            return 0.0

        vec_a = np.array(
            [cluster_pii.get(k, 0) for k in all_keys], dtype=np.float64
        )
        vec_b = np.array(
            [known_pii.get(k, 0) for k in all_keys], dtype=np.float64
        )

        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cosine = float(np.dot(vec_a, vec_b)) / (norm_a * norm_b)

        # Clamp to [0, 1] to handle floating-point edge cases
        if cosine < 0.0:
            cosine = 0.0
        elif cosine > 1.0:
            cosine = 1.0

        return cosine
