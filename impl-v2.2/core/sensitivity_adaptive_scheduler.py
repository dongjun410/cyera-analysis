"""
Sensitivity-Aware Adaptive Clustering Granularity

Core innovation: Instead of using uniform clustering parameters for all
documents, partition documents into sensitivity tiers and apply tier-specific
clustering granularity. High-sensitivity documents get finer-grained clusters
(lower risk of misclassification), low-sensitivity documents get coarser
clusters (higher efficiency).

This module is a key differentiating feature of the system.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

from core.clustering_engine import ClusteringEngine

logger = logging.getLogger(__name__)


class SensitivityAdaptiveScheduler:

    # Default sensitivity weights per PII type (index in structure feature vector)
    # Order matches COUNTING_PATTERNS in structure_feature_extractor.py:
    # email, phone, credit_card, ssn, iban, url, ip, date, amount, long_number
    DEFAULT_WEIGHTS = [2.0, 3.0, 8.0, 10.0, 7.0, 0.5, 1.0, 1.0, 4.0, 1.5]

    # Tier-specific clustering parameter overrides
    TIER_CONFIGS = {
        "high": {
            "k_range": [20, 300],
            "k_step": 3,
            "split_threshold": 0.75,        # More aggressive splitting
            "min_cluster_size": 2,
            "small_cluster_threshold": 3,
            "small_cluster_min_coherence": 0.85,
        },
        "medium": {
            "k_range": [10, 200],
            "k_step": 5,
            "split_threshold": 0.65,
            "min_cluster_size": 2,
            "small_cluster_threshold": 5,
            "small_cluster_min_coherence": 0.80,
        },
        "low": {
            "k_range": [5, 100],
            "k_step": 10,
            "split_threshold": 0.55,         # Less aggressive splitting
            "min_cluster_size": 3,
            "small_cluster_threshold": 8,
            "small_cluster_min_coherence": 0.70,
        },
    }

    def __init__(self, config: dict):
        self.enabled = config.get("sensitivity_adaptive", {}).get("enabled", True)
        self.weights = config.get("sensitivity_adaptive", {}).get(
            "weights", self.DEFAULT_WEIGHTS
        )
        self.tier_thresholds = config.get("sensitivity_adaptive", {}).get(
            "tier_thresholds", [0.3, 0.7]
        )
        self.base_clustering_config = config

    def compute_sensitivity_scores(
        self, structure_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute a sensitivity score for each document based on its
        structure feature vector (PII type distribution).

        Only uses the first N columns (PII counts) of the structure vector.
        """
        n_pii_features = len(self.weights)
        pii_features = structure_vectors[:, :n_pii_features]

        # Weighted sum of PII counts
        weights = np.array(self.weights, dtype=np.float32)
        raw_scores = pii_features @ weights

        # Normalize to [0, 1] range
        max_score = raw_scores.max()
        if max_score > 0:
            scores = raw_scores / max_score
        else:
            scores = raw_scores

        return scores

    def partition_into_tiers(
        self, scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Partition documents into sensitivity tiers based on scores.
        Returns dict of tier_name → boolean mask.
        """
        low_threshold, high_threshold = self.tier_thresholds
        return {
            "high": scores >= high_threshold,
            "medium": (scores >= low_threshold) & (scores < high_threshold),
            "low": scores < low_threshold,
        }

    def fit(
        self,
        embeddings: np.ndarray,
        structure_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Run sensitivity-adaptive clustering.
        Returns label array (same length as embeddings).
        """
        if not self.enabled:
            engine = ClusteringEngine(self.base_clustering_config)
            return engine.fit(embeddings)

        scores = self.compute_sensitivity_scores(structure_vectors)
        tier_masks = self.partition_into_tiers(scores)

        all_labels = np.full(len(embeddings), -1, dtype=int)
        label_offset = 0

        for tier_name in ["high", "medium", "low"]:
            mask = tier_masks[tier_name]
            tier_count = mask.sum()

            if tier_count < 2:
                continue

            # Merge base config with tier-specific overrides
            tier_config = {**self.base_clustering_config}
            tier_config.update(self.TIER_CONFIGS.get(tier_name, {}))

            logger.info(
                f"Tier '{tier_name}': {tier_count} docs, "
                f"k_range={tier_config.get('k_range')}"
            )

            engine = ClusteringEngine(tier_config)
            tier_labels = engine.fit(embeddings[mask])

            # Offset labels to avoid collision across tiers
            valid_mask = tier_labels >= 0
            tier_labels[valid_mask] += label_offset
            label_offset = tier_labels[valid_mask].max() + 1 if valid_mask.any() else label_offset

            # Write back
            indices = np.where(mask)[0]
            for i, idx in enumerate(indices):
                all_labels[idx] = tier_labels[i]

        n_clusters = len(set(all_labels)) - (1 if -1 in all_labels else 0)
        logger.info(
            f"Sensitivity-adaptive clustering complete: "
            f"{n_clusters} total clusters across {len(tier_masks)} tiers"
        )
        return all_labels
