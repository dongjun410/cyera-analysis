"""Type discovery loop — outlier buffer → periodic re-clustering → new type registration.

Per spec section 4.5.

Collects documents that could not be confidently classified by Tier 1/2/3,
periodically re-clusters them, evaluates candidate clusters against
coherence and distance-to-known criteria, and registers viable new types.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from src.types import Document, KnownType

if TYPE_CHECKING:
    from src.embeddings.bge_m3 import BgeM3Embedder
    from src.tier1.semantic import SemanticRefiner
    from src.tier1.structural import StructuralClusterer
    from src.tier2.matching import KnownTypeMatcher


# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "min_trigger_count": 100,
    "same_pattern_threshold": 5,
    "time_trigger_hours": 24,
    "min_coherence": 0.75,
    "min_distance_to_known": 0.3,
    "min_cluster_size": 3,
}


# ──────────────────────────────────────────────────────────────
# DiscoveryLoop
# ──────────────────────────────────────────────────────────────

class DiscoveryLoop:
    """Outlier accumulation → periodic re-clustering → new type evaluation → registration.

    Collects outliers from Tier 2 (low match scores), Tier 3 (is_new_type),
    and distance-based rejection. Periodically re-clusters the accumulated
    outliers to discover emerging document types.

    All dependencies are injected (DI pattern).
    """

    def __init__(
        self,
        structural: StructuralClusterer,
        refiner: SemanticRefiner,
        embedder: BgeM3Embedder,
        matcher: KnownTypeMatcher,
        config: dict | None = None,
    ) -> None:
        """Initialize the discovery loop with injected dependencies.

        Args:
            structural: StructuralClusterer for Stage A structural hashing.
            refiner: SemanticRefiner for Stage B semantic sub-clustering.
            embedder: BgeM3Embedder for encoding document text to embeddings.
            matcher: KnownTypeMatcher for registering newly discovered types.
            config: Optional overrides for DEFAULT_CONFIG keys.
        """
        self._structural = structural
        self._refiner = refiner
        self._embedder = embedder
        self._matcher = matcher

        # Merge config
        cfg = dict(DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)

        self._min_trigger_count: int = cfg["min_trigger_count"]
        self._same_pattern_threshold: int = cfg["same_pattern_threshold"]
        self._time_trigger_hours: int = cfg["time_trigger_hours"]
        self._min_coherence: float = cfg["min_coherence"]
        self._min_distance_to_known: float = cfg["min_distance_to_known"]
        self._min_cluster_size: int = cfg["min_cluster_size"]

        # Internal state
        self._buffer: list[tuple[Document, str]] = []  # (doc, reason)
        self._pattern_counts: dict[tuple[str, str], int] = {}  # (bucket, label) → count

    # ── Public API ────────────────────────────────────────────

    def collect_outlier(self, doc: Document, reason: str) -> None:
        """Store an outlier document with its rejection reason.

        Also updates pattern tracking for (structural_bucket, label) tuples
        to detect emerging patterns before the count threshold is hit.

        Sources: Tier 2 match score < 0.5, LLM is_new_type=True,
        Tier 3 distance > 3σ.

        Args:
            doc: The outlier document.
            reason: Why the document was flagged as an outlier.
        """
        # Compute structural bucket for pattern tracking
        try:
            bucket = self._structural.assign_bucket(doc)
        except Exception:
            bucket = "unknown"

        label = doc.label or ""

        self._buffer.append((doc, reason))
        pattern_key = (bucket, label)
        self._pattern_counts[pattern_key] = (
            self._pattern_counts.get(pattern_key, 0) + 1
        )

    def should_run(self) -> bool:
        """Check whether the discovery loop should execute.

        Returns True when EITHER:
          - Outlier buffer size >= min_trigger_count (default: 100)
          - Any (structural_bucket, label) pattern count >= same_pattern_threshold (default: 5)
        """
        if len(self._buffer) >= self._min_trigger_count:
            return True
        if any(
            count >= self._same_pattern_threshold
            for count in self._pattern_counts.values()
        ):
            return True
        return False

    def run(self) -> list[KnownType]:
        """Execute the discovery loop on accumulated outliers.

        Workflow:
        1. Extract documents from outlier buffer
        2. Run Tier 1 two-stage clustering (structural → semantic)
        3. For each candidate cluster:
           a. Check intra-cluster coherence > min_coherence (0.75)
           b. Check distance to nearest known type > min_distance_to_known (0.3)
           c. Check cluster size >= min_cluster_size (3)
        4. For passing candidates: generate type name, description
        5. Register new types via matcher.register_type()
        6. Clear processed outliers from buffer

        Returns:
            List of newly discovered KnownType objects (may be empty).
        """
        if not self._buffer:
            return []

        # Extract documents from buffer
        docs = [doc for doc, _ in self._buffer]
        doc_map: dict[str, Document] = {d.doc_id: d for d in docs}

        # Step 2: Tier 1 two-stage clustering
        buckets = self._structural.cluster(docs)  # {bucket_id: [doc_id, ...]}

        discovered: list[KnownType] = []
        timestamp = int(time.time())

        for bucket_id, doc_ids in buckets.items():
            # Resolve doc_ids to Document objects
            bucket_docs = [doc_map[did] for did in doc_ids if did in doc_map]
            if len(bucket_docs) < self._min_cluster_size:
                continue

            # Stage B: Semantic refinement within this structural bucket
            sub_clusters = self._refiner.refine(bucket_id, bucket_docs)

            for cluster in sub_clusters:
                # Resolve cluster members
                cluster_docs = [
                    doc_map[did] for did in cluster.doc_ids if did in doc_map
                ]
                if len(cluster_docs) < self._min_cluster_size:
                    continue

                # ── Step 3a: Intra-cluster coherence ─────────────
                embeddings = self._embedder.encode(
                    [d.text for d in cluster_docs]
                )
                # embeddings are unit-normalized (BGE-M3)
                centroid = embeddings.mean(axis=0).astype(np.float32)
                centroid_norm = float(np.linalg.norm(centroid))
                if centroid_norm > 0:
                    centroid = centroid / centroid_norm

                # Cosine similarity = dot product for unit vectors
                cos_sims = embeddings @ centroid  # (M,)
                coherence = float(np.mean(cos_sims))

                if coherence < self._min_coherence:
                    continue

                # ── Step 3b: Distance to nearest known type ──────
                min_distance = self._compute_min_distance_to_known(centroid)
                if min_distance < self._min_distance_to_known:
                    continue

                # ── Step 4: Create new KnownType ─────────────────
                idx = len(discovered) + 1
                new_type = KnownType(
                    type_id=f"discovered_{timestamp}_{idx}",
                    type_name=f"Unknown-Type-{idx}",
                    description=(
                        f"Auto-discovered type from cluster {cluster.cluster_id}"
                    ),
                    structural_signature=bucket_id,
                    tfidf_keywords=list(cluster.tfidf_keywords),
                    pii_distribution=dict(cluster.pii_distribution),
                    semantic_centroid=centroid.copy(),
                    detection_rules=[],
                    status="pending_review",
                    sample_count=len(cluster_docs),
                )

                self._matcher.register_type(new_type)
                discovered.append(new_type)

        # Step 5: Clear processed outliers
        self._buffer.clear()
        self._pattern_counts.clear()

        return discovered

    def get_buffer_size(self) -> int:
        """Return the number of documents currently in the outlier buffer."""
        return len(self._buffer)

    def get_buffer_docs(self) -> list[Document]:
        """Return the list of documents currently in the outlier buffer."""
        return [doc for doc, _ in self._buffer]

    # ── Internal helpers ──────────────────────────────────────

    def _compute_min_distance_to_known(self, centroid: np.ndarray) -> float:
        """Compute the minimum cosine distance from centroid to any known type.

        Iterates over all registered known types. Only considers types
        that have a semantic_centroid set.

        Returns:
            Minimum cosine distance (1.0 - max cosine similarity).
            Returns 1.0 if no known types have semantic_centroid set.
        """
        min_distance = 1.0
        for kt in self._matcher._types.values():  # noqa: SLF001
            if kt.semantic_centroid is not None:
                # Cosine similarity = dot product (both are unit-normalized)
                cos_sim = float(np.dot(centroid, kt.semantic_centroid))
                # Clamp to [-1, 1] for numerical stability
                cos_sim = max(-1.0, min(1.0, cos_sim))
                distance = 1.0 - cos_sim
                if distance < min_distance:
                    min_distance = distance
        return min_distance
