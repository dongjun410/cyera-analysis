"""Tier 2: LabelPropagator — propagate cluster labels to member documents.

Assigns a cluster-determined label to every document in the cluster,
with optional sampling verification and inconsistency detection.
Per spec section 4.3 Step 5.
"""

from __future__ import annotations

import random

from src.types import ClassificationResult, ClusterInfo, Document


class LabelPropagator:
    """Propagate cluster-level labels to all member documents.

    Sampling strategy: inverse_cluster_size — smaller clusters get a larger
    fraction of their documents sampled for verification, because small
    clusters carry more risk of misclassification.

    Parameters
    ----------
    config : dict, optional
        - sample_strategy: str (default "inverse_cluster_size")
        - min_samples: int (default 3)
        - inconsistency_threshold: float (default 0.15)
        - seed: int, optional — random seed for reproducibility
    """

    VALID_STRATEGIES = {"inverse_cluster_size"}

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}

        self._sample_strategy: str = cfg.get(
            "sample_strategy", "inverse_cluster_size"
        )
        if self._sample_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown sample_strategy: {self._sample_strategy!r}. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )

        self._min_samples: int = cfg.get("min_samples", 3)
        self._inconsistency_threshold: float = cfg.get(
            "inconsistency_threshold", 0.15
        )

        # Seed from config for reproducibility
        if "seed" in cfg:
            random.seed(cfg["seed"])

    # ── Public API ──────────────────────────────────────────────

    def propagate(
        self,
        cluster: ClusterInfo,
        label: str,
        confidence: float,
        documents: list[Document],
    ) -> tuple[list[ClassificationResult], bool]:
        """Assign a label to every document in the cluster, with sampling verification.

        1. Assign the cluster label to ALL documents in cluster.doc_ids.
        2. Select a sample of documents for manual verification using the
           inverse_cluster_size strategy.
        3. Mark sampled documents for review when confidence < 0.85.
        4. Return all ClassificationResults and a needs_resplit flag.

        Parameters
        ----------
        cluster : ClusterInfo
            The cluster whose documents receive the label.
        label : str
            The label to assign to all cluster documents.
        confidence : float
            Confidence of the cluster-level classification.
        documents : list[Document]
            All documents (to look up those not in cluster.doc_ids but
            sharing the same label — they are part of the cluster).

        Returns
        -------
        tuple[list[ClassificationResult], bool]
            - List of ClassificationResult, one per document in the cluster.
            - needs_resplit: True if sampled verification inconsistency
              exceeds the threshold. In this implementation, always False
              (Tier 3 handles splits).
        """
        # Build a lookup: doc_id → Document
        doc_lookup: dict[str, Document] = {d.doc_id: d for d in documents}

        # 1. Assign label to EVERY document in the cluster's doc_ids list
        results: list[ClassificationResult] = []
        for doc_id in cluster.doc_ids:
            results.append(
                ClassificationResult(
                    doc_id=doc_id,
                    label=label,
                    confidence=confidence,
                    method="propagated",
                    is_new_type=False,
                    needs_manual_review=False,
                    rationale=(
                        f"Label propagated from cluster {cluster.cluster_id} "
                        f"(structural_bucket={cluster.structural_bucket})"
                    ),
                )
            )

        # Also capture any documents that match this label but are NOT in
        # cluster.doc_ids — they belong to the same cluster via label affinity.
        for doc in documents:
            if doc.doc_id not in cluster.doc_ids and doc.label == label:
                results.append(
                    ClassificationResult(
                        doc_id=doc.doc_id,
                        label=label,
                        confidence=confidence,
                        method="propagated",
                        is_new_type=False,
                        needs_manual_review=False,
                        rationale=(
                            f"Label propagated via label affinity from "
                            f"cluster {cluster.cluster_id}"
                        ),
                    )
                )

        # 2. Select verification sample
        cluster_doc_ids = cluster.doc_ids
        sample_size = self._compute_sample_size(len(cluster_doc_ids))

        if sample_size > 0 and len(cluster_doc_ids) > 0:
            # Randomly select documents for verification
            sample_ids = random.sample(
                cluster_doc_ids,
                k=min(sample_size, len(cluster_doc_ids)),
            )

            # 3. Mark sampled documents for review if confidence < 0.85
            if confidence < 0.85:
                for result in results:
                    if result.doc_id in sample_ids:
                        result.needs_manual_review = True

        # 4. needs_resplit is always False — Tier 3 handles splits
        needs_resplit = False

        return results, needs_resplit

    def compute_inconsistency(
        self,
        verified_labels: list[str],
        propagated_label: str,
    ) -> float:
        """Compute the fraction of verified labels that differ from the propagated label.

        Parameters
        ----------
        verified_labels : list[str]
            Manually verified labels for the sampled documents.
        propagated_label : str
            The label that was originally propagated to the cluster.

        Returns
        -------
        float
            Inconsistency ratio in [0.0, 1.0]. A value of 0.0 means perfect
            agreement; 1.0 means all verified labels disagree.
        """
        if not verified_labels:
            return 0.0

        mismatches = sum(
            1 for lbl in verified_labels if lbl != propagated_label
        )
        return mismatches / len(verified_labels)

    # ── Internal helpers ─────────────────────────────────────────

    def _compute_sample_size(self, cluster_size: int) -> int:
        """Determine sample size using the configured strategy.

        inverse_cluster_size: smaller clusters get more coverage.
            sample_size = max(min_samples, int(cluster_size * 0.2))
            Capped at min(cluster_size, 10) to avoid over-sampling large clusters.
        """
        if self._sample_strategy == "inverse_cluster_size":
            # The "inverse" effect: 20% floor ensures small clusters get
            # proportionally more scrutiny.
            raw = int(cluster_size * 0.2)
            size = max(self._min_samples, raw)
            # Cap at min(cluster_size, 10)
            size = min(size, cluster_size, 10)
            return size

        # Fallback (should not be reached — validated in __init__)
        return min(cluster_size, self._min_samples)
