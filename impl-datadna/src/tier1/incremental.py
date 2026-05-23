"""Tier 1 Incremental Document Assignment.

O(1) structural hash lookup + O(log N) FAISS nearest-neighbor assignment
for new documents. Detects intra-bucket outliers and new-structure candidates,
triggering re-clustering when outlier ratios exceed thresholds.

Per spec section 4.2 Incremental.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.types import AssignmentResult, ClusterInfo, Document

if TYPE_CHECKING:
    from src.embeddings.bge_m3 import BgeM3Embedder
    from src.tier1.semantic import SemanticRefiner
    from src.tier1.structural import StructuralClusterer

# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "outlier_radius_multiplier": 1.5,
    "outlier_trigger_ratio": 0.2,
    "new_bucket_trigger": 50,
}

# ──────────────────────────────────────────────────────────────
# Optional FAISS import
# ──────────────────────────────────────────────────────────────

try:
    import faiss  # noqa: F401

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# ──────────────────────────────────────────────────────────────
# IncrementalAssigner
# ──────────────────────────────────────────────────────────────


class IncrementalAssigner:
    """Incremental document assignment via structural hash + nearest centroid.

    For each new document:
    1. Hash structural features → O(1) bucket lookup
    2. If bucket exists: embed doc, find nearest sub-cluster centroid
       via FAISS (if available) or brute-force cosine similarity
    3. If within radius threshold → assign to that cluster
    4. If outside radius → mark as intra_bucket_outlier
    5. If bucket doesn't exist → mark as new_structure_candidate

    Tracks per-bucket outlier counts and signals when re-clustering is warranted.
    """

    def __init__(
        self,
        structural: StructuralClusterer,
        refiner: SemanticRefiner,
        embedder: BgeM3Embedder,
        config: dict | None = None,
    ) -> None:
        """Initialize with structural hasher, semantic refiner, and embedder.

        Args:
            structural: StructuralClusterer for structural feature hashing.
            refiner: SemanticRefiner used for re-clustering triggers.
            embedder: BgeM3Embedder for encoding document text.
            config: Optional dict overriding any DEFAULT_CONFIG keys.
        """
        self._structural = structural
        self._refiner = refiner
        self._embedder = embedder

        cfg = dict(DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)
        self._config = cfg

        # Per-bucket outlier tracking: bucket_id → outlier count
        self._outlier_counts: dict[str, int] = {}

        # Per-bucket total assignment count: bucket_id → total assigned
        self._bucket_totals: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────

    def assign(
        self,
        doc: Document,
        known_buckets: dict[str, list[ClusterInfo]],
    ) -> AssignmentResult:
        """Assign a single document to a cluster in the known bucket map.

        Workflow:
        1. Extract structural features from doc → hash → bucket_id
        2. Look up bucket_id in known_buckets
        3. If bucket exists:
           a. Embed doc text
           b. Find nearest sub-cluster centroid (FAISS or brute-force)
           c. Compute cosine distance to nearest centroid
           d. If distance <= outlier_radius_multiplier * cluster_radius → assign
           e. Else → mark as intra_bucket_outlier
        4. If bucket doesn't exist → new_structure_candidate
        5. Update outlier tracking counters

        Args:
            doc: The document to assign.
            known_buckets: Mapping of bucket_id → list of ClusterInfo sub-clusters.

        Returns:
            AssignmentResult with assigned cluster, outlier flags, and
            re-clustering signal.
        """
        # Step 1: Hash structural features
        bucket_id = self._structural.extract_features(doc)

        # Step 2: Look up bucket in known_buckets
        clusters = known_buckets.get(bucket_id, [])

        # Step 3: If bucket doesn't exist → new structure candidate
        if not clusters:
            return AssignmentResult(
                doc_id=doc.doc_id,
                assigned_cluster_id=None,
                is_outlier=True,
                outlier_reason="new_structure_candidate",
                needs_reclustering=False,
            )

        # Step 3a: Embed document text
        try:
            embeddings = self._embedder.encode([doc.text])
            doc_embedding = embeddings[0]
        except Exception:
            # If embedding fails, treat as outlier — can't assign semantically
            self._record_outlier(bucket_id, is_outlier=True)
            return AssignmentResult(
                doc_id=doc.doc_id,
                assigned_cluster_id=None,
                is_outlier=True,
                outlier_reason="intra_bucket_outlier",
                needs_reclustering=self.should_recluster(bucket_id),
            )

        # Step 3b: Find nearest sub-cluster centroid
        nearest_cluster, min_distance = self._find_nearest(
            doc_embedding, clusters
        )

        # No cluster had a valid centroid
        if nearest_cluster is None:
            self._record_outlier(bucket_id, is_outlier=True)
            return AssignmentResult(
                doc_id=doc.doc_id,
                assigned_cluster_id=None,
                is_outlier=True,
                outlier_reason="intra_bucket_outlier",
                needs_reclustering=self.should_recluster(bucket_id),
            )

        # Step 3d: Check distance against outlier threshold
        threshold = (
            self._config["outlier_radius_multiplier"]
            * nearest_cluster.cluster_radius
        )

        if min_distance <= threshold:
            # Within radius — assign to this cluster
            self._record_outlier(bucket_id, is_outlier=False)
            return AssignmentResult(
                doc_id=doc.doc_id,
                assigned_cluster_id=nearest_cluster.cluster_id,
                is_outlier=False,
                outlier_reason="",
                needs_reclustering=False,
            )

        # Step 3e: Outside radius — intra_bucket_outlier
        self._record_outlier(bucket_id, is_outlier=True)
        return AssignmentResult(
            doc_id=doc.doc_id,
            assigned_cluster_id=None,
            is_outlier=True,
            outlier_reason="intra_bucket_outlier",
            needs_reclustering=self.should_recluster(bucket_id),
        )

    def should_recluster(self, bucket_id: str) -> bool:
        """Check whether outlier ratio exceeds trigger threshold.

        Returns True when:
            outlier_count / total_in_bucket > outlier_trigger_ratio

        Args:
            bucket_id: The structural bucket ID to check.

        Returns:
            True if re-clustering should be triggered for this bucket.
        """
        total = self._bucket_totals.get(bucket_id, 0)
        if total == 0:
            return False

        outliers = self._outlier_counts.get(bucket_id, 0)
        ratio = outliers / total
        return ratio > self._config["outlier_trigger_ratio"]

    def get_outlier_count(self, bucket_id: str) -> int:
        """Return the outlier count for a specific bucket.

        Args:
            bucket_id: The structural bucket ID.

        Returns:
            Number of outliers recorded for this bucket.
        """
        return self._outlier_counts.get(bucket_id, 0)

    def get_bucket_total(self, bucket_id: str) -> int:
        """Return the total assignment count for a specific bucket.

        Args:
            bucket_id: The structural bucket ID.

        Returns:
            Total number of documents assigned (including outliers) to this bucket.
        """
        return self._bucket_totals.get(bucket_id, 0)

    def reset_outlier_counts(self, bucket_id: str | None = None) -> None:
        """Reset outlier tracking counters.

        Args:
            bucket_id: If provided, reset only this bucket.
                       If None, reset all buckets.
        """
        if bucket_id is not None:
            self._outlier_counts.pop(bucket_id, None)
            self._bucket_totals.pop(bucket_id, None)
        else:
            self._outlier_counts.clear()
            self._bucket_totals.clear()

    # ── Internal helpers ──────────────────────────────────────

    def _record_outlier(self, bucket_id: str, is_outlier: bool) -> None:
        """Update per-bucket tracking counters.

        Args:
            bucket_id: The structural bucket ID.
            is_outlier: Whether this assignment is an outlier.
        """
        self._bucket_totals[bucket_id] = (
            self._bucket_totals.get(bucket_id, 0) + 1
        )
        if is_outlier:
            self._outlier_counts[bucket_id] = (
                self._outlier_counts.get(bucket_id, 0) + 1
            )

    def _find_nearest(
        self,
        doc_embedding: np.ndarray,
        clusters: list[ClusterInfo],
    ) -> tuple[ClusterInfo | None, float]:
        """Find the cluster with centroid nearest to doc_embedding.

        Computes cosine distance (1 - cosine_similarity) to each valid
        cluster centroid. Uses FAISS if available, otherwise brute-force.

        Args:
            doc_embedding: (D,) float32 embedding of the document.
            clusters: List of ClusterInfo sub-clusters to search.

        Returns:
            (nearest_cluster, min_cosine_distance).
            nearest_cluster is None if no cluster has a valid centroid.
        """
        # Collect centroids from clusters that have them
        valid_clusters: list[ClusterInfo] = []
        centroid_list: list[np.ndarray] = []

        for c in clusters:
            if c.centroid_embedding is not None:
                valid_clusters.append(c)
                centroid_list.append(c.centroid_embedding)

        if not centroid_list:
            return None, float("inf")

        centroids = np.stack(centroid_list)  # (K, D)

        # Compute cosine distances
        if _HAS_FAISS and len(centroids) >= 100:
            distances = self._faiss_nearest(doc_embedding, centroids)
        else:
            distances = self._brute_force_nearest(doc_embedding, centroids)

        best_idx = int(np.argmin(distances))
        return valid_clusters[best_idx], float(distances[best_idx])

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine distance between vectors.

        cosine_distance = 1 - dot(a, b) / (||a|| * ||b||)

        When both a and b are unit-normalized, this simplifies to 1 - dot(a, b).

        Args:
            a: (D,) single vector or (N, D) array.
            b: (K, D) array of centroid vectors.

        Returns:
            (K,) array of cosine distances, or scalar if a is 1-D and b is 1-D.
        """
        # L2 norms
        a_norm = np.linalg.norm(a)
        b_norms = np.linalg.norm(b, axis=-1)  # (K,) or scalar

        # Dot products
        dot = np.dot(b, a) if a.ndim == 1 else np.sum(b * a, axis=-1)

        # Avoid division by zero
        denom = a_norm * b_norms
        denom = np.where(denom == 0, 1.0, denom)

        return 1.0 - dot / denom

    def _brute_force_nearest(
        self,
        doc_embedding: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine distances to all centroids via brute-force.

        Args:
            doc_embedding: (D,) float32 embedding.
            centroids: (K, D) float32 centroid matrix.

        Returns:
            (K,) float32 array of cosine distances.
        """
        return self._cosine_distance(doc_embedding, centroids)

    def _faiss_nearest(
        self,
        doc_embedding: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine distances via FAISS IndexFlatIP.

        Uses inner-product search (equivalent to cosine similarity for
        unit-normalized vectors), then converts to distance.

        Args:
            doc_embedding: (D,) float32 embedding.
            centroids: (K, D) float32 centroid matrix.

        Returns:
            (K,) float32 array of cosine distances.
        """
        import faiss

        d = centroids.shape[1]
        # Ensure float32 contiguous
        centroids32 = np.ascontiguousarray(centroids.astype(np.float32))
        query32 = np.ascontiguousarray(
            doc_embedding.astype(np.float32).reshape(1, -1)
        )

        index = faiss.IndexFlatIP(d)
        index.add(centroids32)
        similarities, _ = index.search(query32, centroids.shape[0])
        # cosine distance = 1 - similarity (valid for normalized vectors)
        return 1.0 - similarities[0]
