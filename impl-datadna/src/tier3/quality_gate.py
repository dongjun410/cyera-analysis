"""Tier 3: QualityGate — final precision defense line using LLM verification.

Only ~2% of documents reach this tier. Five triggers per spec section 4.4
determine whether LLM verification is needed:

  1. High sensitivity: label is in high_sensitivity_types
  2. Semantic outlier: cosine distance from doc to cluster centroid > 2σ
  3. NER-rule contradiction: NER found types Tier 0 missed AND vice versa
  4. Low LLM confidence: confidence in [lower, upper) configurable range
  5. Marked for verification: classification.needs_manual_review == True

The verify() method calls the LLM for deep analysis and returns a final
ClassificationResult with method="llm_tier3" and a reasoning chain.
"""

from __future__ import annotations

import numpy as np

from src.llm.client import MistralClient
from src.types import ClassificationResult, ClusterInfo, Document


class QualityGate:
    """Final precision defense line — LLM verification for flagged documents.

    All external dependencies (LLM) are injected via constructor (DI),
    enabling full testability with mocks.

    Parameters
    ----------
    llm : MistralClient
        Mistral-7B LLM client for Tier 3 verification.
    config : dict, optional
        - high_sensitivity_types : list[str]
            Labels that always trigger verification.
            Default: ["SSN", "CREDIT_CARD", "MEDICAL", "IBAN"]
        - semantic_distance_sigma : float (default 2.0)
            Multiplier for cluster_radius to define outlier threshold.
        - outlier_sigma : float (default 3.0)
            Reserved for future use (extreme outlier detection).
        - ner_rule_contradiction : bool (default True)
            Whether to enable NER-rule contradiction detection.
        - llm_confidence_range : list[float] (default [0.5, 0.8])
            [lower, upper) — confidence in this range triggers verification.
    """

    def __init__(
        self,
        llm: MistralClient,
        config: dict | None = None,
    ) -> None:
        self._llm = llm
        cfg = config or {}

        self._high_sensitivity_types: list[str] = cfg.get(
            "high_sensitivity_types",
            ["SSN", "CREDIT_CARD", "MEDICAL", "IBAN"],
        )
        self._semantic_distance_sigma: float = cfg.get(
            "semantic_distance_sigma", 2.0
        )
        self._outlier_sigma: float = cfg.get("outlier_sigma", 3.0)
        self._ner_rule_contradiction: bool = cfg.get(
            "ner_rule_contradiction", True
        )
        confidence_range = cfg.get("llm_confidence_range", [0.5, 0.8])
        self._confidence_lower: float = confidence_range[0]
        self._confidence_upper: float = confidence_range[1]

    # ── Public API ──────────────────────────────────────────────

    def should_trigger(
        self,
        doc: Document,
        cluster: ClusterInfo,
        classification: ClassificationResult,
        ner_results: list | None = None,
        tier0_features: dict | None = None,
    ) -> bool:
        """Determine whether the quality gate should be triggered.

        Returns True if ANY of the five trigger conditions are met.

        Parameters
        ----------
        doc : Document
            The document being classified.
        cluster : ClusterInfo
            The cluster the document belongs to.
        classification : ClassificationResult
            The current classification from Tier 2.
        ner_results : list, optional
            NER results as list of PIIFeature (or dict-like with entity_type).
        tier0_features : dict, optional
            Tier 0 feature dict (pii_type_distribution or similar).

        Returns
        -------
        bool
            True if verification is needed.
        """
        # Trigger 1: High sensitivity label
        if self._check_high_sensitivity(classification):
            return True

        # Trigger 2: Semantic outlier
        if self._check_semantic_outlier(doc, cluster):
            return True

        # Trigger 3: NER-rule contradiction
        if self._ner_rule_contradiction and self._check_ner_contradiction(
            ner_results, tier0_features
        ):
            return True

        # Trigger 4: Low LLM confidence
        if self._check_low_confidence(classification):
            return True

        # Trigger 5: Marked for manual review
        if classification.needs_manual_review:
            return True

        return False

    def verify(
        self,
        doc: Document,
        cluster: ClusterInfo,
        current_classification: ClassificationResult,
    ) -> ClassificationResult:
        """LLM deep analysis to confirm or correct a classification.

        Builds cluster context, calls llm.verify(), and returns a final
        ClassificationResult with method="llm_tier3".

        Parameters
        ----------
        doc : Document
            The document to verify.
        cluster : ClusterInfo
            The cluster context for the LLM.
        current_classification : ClassificationResult
            The current classification to verify.

        Returns
        -------
        ClassificationResult
            Final classification with method="llm_tier3" and reasoning chain.
        """
        # 1. Build cluster context dict
        cluster_context = {
            "cluster_label": cluster.label,
            "cluster_size": len(cluster.doc_ids),
            "keywords": cluster.tfidf_keywords,
        }

        # 2. Call LLM verification
        llm_response = self._llm.verify(
            doc.text,
            current_classification.label,
            cluster_context,
        )

        # 3. Parse response
        label = llm_response.get("label", current_classification.label)
        confidence = float(llm_response.get("confidence", 0.0))
        reasoning = llm_response.get("reasoning_chain", "")
        llm_needs_review = bool(llm_response.get("needs_manual_review", False))

        # 4. If LLM confidence < 0.6 → force manual review
        needs_manual_review = confidence < 0.6 or llm_needs_review

        return ClassificationResult(
            doc_id=doc.doc_id,
            label=label,
            confidence=confidence,
            method="llm_tier3",
            is_new_type=current_classification.is_new_type,
            needs_manual_review=needs_manual_review,
            rationale=reasoning,
        )

    def verify_batch(
        self,
        docs: list[Document],
        cluster: ClusterInfo,
        classifications: list[ClassificationResult],
    ) -> list[ClassificationResult]:
        """Batch verification with shared cluster context.

        Parameters
        ----------
        docs : list[Document]
            Documents to verify.
        cluster : ClusterInfo
            The shared cluster context.
        classifications : list[ClassificationResult]
            Current classifications for each document.

        Returns
        -------
        list[ClassificationResult]
            One ClassificationResult per input document, with method="llm_tier3".
        """
        if not docs:
            return []

        results: list[ClassificationResult] = []
        for doc, classification in zip(docs, classifications):
            result = self.verify(doc, cluster, classification)
            results.append(result)
        return results

    # ── Trigger Checks (private) ────────────────────────────────

    def _check_high_sensitivity(self, classification: ClassificationResult) -> bool:
        """Trigger 1: label is in high_sensitivity_types."""
        return classification.label in self._high_sensitivity_types

    def _check_semantic_outlier(
        self, doc: Document, cluster: ClusterInfo
    ) -> bool:
        """Trigger 2: cosine distance > semantic_distance_sigma * cluster_radius."""
        if doc.embedding is None or cluster.centroid_embedding is None:
            return False
        if cluster.cluster_radius <= 0:
            return False

        distance = self._cosine_distance(doc.embedding, cluster.centroid_embedding)
        threshold = self._semantic_distance_sigma * cluster.cluster_radius
        return distance > threshold

    def _check_ner_contradiction(
        self,
        ner_results: list | None,
        tier0_features: dict | None,
    ) -> bool:
        """Trigger 3: NER and Tier 0 detect different PII type sets.

        A contradiction exists when NER finds types that Tier 0 missed AND
        Tier 0 found types that NER missed.

        Heuristic:
            len(type_set_ner - type_set_tier0) > 0
            AND len(type_set_tier0 - type_set_ner) > 0
        """
        if not ner_results or not tier0_features:
            return False

        # Extract entity types from NER results
        type_set_ner: set[str] = set()
        for item in ner_results:
            if item is not None:
                # Handle both PIIFeature objects and dict-like items
                if hasattr(item, "entity_type"):
                    type_set_ner.add(item.entity_type)
                elif isinstance(item, dict) and "entity_type" in item:
                    type_set_ner.add(item["entity_type"])

        # Extract entity types from tier0_features (dict keys = entity types)
        type_set_tier0: set[str] = set(tier0_features.keys())

        # Filter out empty strings and None
        type_set_ner.discard("")
        type_set_ner.discard(None)
        type_set_tier0.discard("")
        type_set_tier0.discard(None)

        if not type_set_ner or not type_set_tier0:
            return False

        diff_ner_minus_tier0 = type_set_ner - type_set_tier0
        diff_tier0_minus_ner = type_set_tier0 - type_set_ner

        return len(diff_ner_minus_tier0) > 0 and len(diff_tier0_minus_ner) > 0

    def _check_low_confidence(self, classification: ClassificationResult) -> bool:
        """Trigger 4: confidence in [lower, upper)."""
        return (
            self._confidence_lower <= classification.confidence < self._confidence_upper
        )

    # ── Math Utility ────────────────────────────────────────────

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance = 1 - cosine_similarity between two vectors.

        Parameters
        ----------
        a : np.ndarray
            First embedding vector.
        b : np.ndarray
            Second embedding vector.

        Returns
        -------
        float
            Cosine distance in [0.0, 2.0].
        """
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cosine_similarity = dot / (norm_a * norm_b)
        # Clamp to [-1.0, 1.0] to avoid floating-point drift
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
        return 1.0 - cosine_similarity
