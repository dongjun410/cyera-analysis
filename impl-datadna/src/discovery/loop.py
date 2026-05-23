"""Type discovery loop — outlier buffer → periodic re-clustering → new type registration.

Collects documents that could not be confidently classified by the fusion voter,
periodically re-clusters them using BGE-M3 embeddings + cosine distance clustering,
evaluates candidate clusters, and registers viable new types into the TypeLibrary.

Triggers (any of):
  - Buffer size >= min_trigger_count (default 100)
  - Same structural pattern >= same_pattern_threshold (default 5)
  - Time since last run >= time_trigger_hours (default 24)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from typing import Any

import numpy as np

from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, FusionResult

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "min_trigger_count": 100,
    "same_pattern_threshold": 5,
    "time_trigger_hours": 24,
    "min_coherence": 0.75,
    "min_distance_to_known": 0.3,
    "min_cluster_size": 3,
}


class DiscoveryLoop:
    """Outlier accumulation → periodic re-clustering → new type registration.

    Collects outliers from fusion results (low confidence, LLM is_new_type,
    unclassified), periodically re-clusters the accumulated outliers to
    discover emerging document types. Registers viable new types into
    the TypeLibrary singleton.

    All dependencies are injected for testability.
    """

    def __init__(
        self,
        embedder=None,
        type_library: TypeLibrary | None = None,
        config: dict | None = None,
        llm_client=None,
    ) -> None:
        """Initialize the discovery loop.

        Args:
            embedder: BgeM3Embedder instance for encoding documents.
            type_library: TypeLibrary for registration. Uses singleton if None.
            config: Optional overrides for DEFAULT_CONFIG.
            llm_client: Optional MistralClient for descriptive type naming.
        """
        self._embedder = embedder
        self._type_library = type_library or get_type_library()
        self._llm = llm_client

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
        self._pattern_counts: dict[str, int] = {}  # structural_pattern → count
        self._last_run_time: float | None = None

    # ── Public API ────────────────────────────────────────────

    def collect_outlier(self, doc: Document, reason: str) -> None:
        """Store an outlier document with its rejection reason.

        Args:
            doc: The outlier document.
            reason: Why the document was flagged (e.g. "low_confidence:0.35").
        """
        pattern = self._structural_pattern(doc)
        self._buffer.append((doc, reason))
        self._pattern_counts[pattern] = self._pattern_counts.get(pattern, 0) + 1

    def collect_from_result(self, result: FusionResult, doc: Document) -> bool:
        """Collect outlier from a FusionResult if it qualifies.

        A document is an outlier if:
          - composite_confidence < 0.4 (manual_review)
          - final_label == "unclassified"
          - E6 LLM flagged is_new_type
          - degraded mode was used

        Returns True if the document was collected as an outlier.
        """
        reasons = []
        if result.composite_confidence < 0.4:
            reasons.append(f"low_confidence:{result.composite_confidence:.2f}")
        if result.final_label == "unclassified":
            reasons.append("unclassified")
        if result.degraded:
            reasons.append("degraded")

        # Check E6 LLM metadata for is_new_type flag
        e6_out = result.engine_outputs.get("E6_llm")
        if e6_out is not None and e6_out.metadata.get("is_new_type"):
            reasons.append("llm_is_new_type")

        if reasons:
            self.collect_outlier(doc, "; ".join(reasons))
            return True
        return False

    def should_run(self) -> bool:
        """Check whether the discovery loop should execute."""
        if len(self._buffer) >= self._min_trigger_count:
            return True
        if any(
            count >= self._same_pattern_threshold
            for count in self._pattern_counts.values()
        ):
            return True
        if self._last_run_time is not None and len(self._buffer) > 0:
            elapsed_hours = (time.time() - self._last_run_time) / 3600.0
            if elapsed_hours >= self._time_trigger_hours:
                return True
        return False

    def run(self) -> list[dict]:
        """Execute the discovery loop on accumulated outliers.

        Workflow:
        1. Extract outlier documents
        2. Embed all documents with BGE-M3
        3. Simple agglomerative clustering on cosine distance
        4. For each candidate cluster:
           a. Check intra-cluster coherence > min_coherence
           b. Check distance to nearest known type > min_distance_to_known
           c. Check size >= min_cluster_size
        5. Register passing candidates in TypeLibrary

        Returns:
            List of dicts with type_id, type_name, sample_count for new types.
        """
        if not self._buffer:
            return []

        self._last_run_time = time.time()

        docs = [doc for doc, _ in self._buffer]
        if len(docs) < self._min_cluster_size:
            self._buffer.clear()
            self._pattern_counts.clear()
            return []

        # Step 2: Embed all documents
        if self._embedder is None:
            logger.warning("No embedder available — cannot run discovery")
            return []

        try:
            embeddings = self._embedder.encode([d.text for d in docs])
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return []

        # Step 3: Simple clustering — group by cosine similarity > 0.7
        clusters = self._simple_cluster(embeddings, threshold=0.7)

        # Step 4: Evaluate each cluster
        discovered: list[dict] = []
        timestamp = int(time.time())

        for idx, cluster_indices in enumerate(clusters):
            if len(cluster_indices) < self._min_cluster_size:
                continue

            cluster_docs = [docs[i] for i in cluster_indices]
            cluster_embs = embeddings[cluster_indices]

            # 4a: Intra-cluster coherence
            centroid = cluster_embs.mean(axis=0).astype(np.float32)
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            cos_sims = cluster_embs @ centroid
            coherence = float(np.mean(cos_sims))
            if coherence < self._min_coherence:
                continue

            # 4b: Distance to nearest known type
            min_dist = self._min_distance_to_known_type(centroid)
            if min_dist < self._min_distance_to_known:
                continue

            # 4c: Generate type name and register
            keywords = self._extract_keywords(cluster_docs)
            type_name, type_desc = self._generate_name(
                keywords, cluster_docs, idx + 1,
            )

            type_id = f"discovered_{timestamp}_{idx + 1}"
            self._type_library.register(
                type_id=type_id,
                type_name=type_name,
                source="discovery",
                centroid=centroid,
                keywords=keywords,
            )

            discovered.append({
                "type_id": type_id,
                "type_name": type_name,
                "description": type_desc,
                "sample_count": len(cluster_docs),
                "coherence": round(coherence, 4),
                "distance_to_known": round(min_dist, 4),
            })

            logger.info(
                "New type discovered: %s (%d docs, coherence=%.3f, dist=%.3f)",
                type_name, len(cluster_docs), coherence, min_dist,
            )

        # Step 5: Clear processed outliers
        self._buffer.clear()
        self._pattern_counts.clear()

        return discovered

    def get_buffer_size(self) -> int:
        return len(self._buffer)

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _structural_pattern(doc: Document) -> str:
        """Create a simple structural pattern hash for the document."""
        meta = doc.metadata or {}
        file_type = meta.get("file_type", "")
        path_depth = meta.get("path_depth", 0)
        pattern_str = f"{file_type}:{path_depth}"
        return hashlib.md5(pattern_str.encode()).hexdigest()[:12]

    def _simple_cluster(
        self, embeddings: np.ndarray, threshold: float = 0.7
    ) -> list[list[int]]:
        """Simple agglomerative clustering on cosine similarity.

        Greedy: each point joins the first cluster where it has
        similarity > threshold with the existing centroid.
        If no cluster matches, it starts a new one.
        """
        n = embeddings.shape[0]
        if n == 0:
            return []

        clusters: list[list[int]] = []
        centroids: list[np.ndarray] = []

        for i in range(n):
            assigned = False
            for ci, centroid in enumerate(centroids):
                sim = float(np.dot(embeddings[i], centroid))
                if sim > threshold:
                    clusters[ci].append(i)
                    # Update centroid incrementally
                    k = len(clusters[ci])
                    centroids[ci] = (centroid * (k - 1) + embeddings[i]) / k
                    # Re-normalize
                    norm = float(np.linalg.norm(centroids[ci]))
                    if norm > 0:
                        centroids[ci] = centroids[ci] / norm
                    assigned = True
                    break

            if not assigned:
                clusters.append([i])
                centroids.append(embeddings[i].copy())

        return clusters

    def _min_distance_to_known_type(self, centroid: np.ndarray) -> float:
        """Compute minimum cosine distance from centroid to any known type."""
        min_dist = 1.0
        for info in self._type_library.list_active():
            if info.centroid is not None:
                cos_sim = float(np.dot(centroid, info.centroid))
                cos_sim = max(-1.0, min(1.0, cos_sim))
                dist = 1.0 - cos_sim
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    @staticmethod
    def _extract_keywords(docs: list[Document], top_n: int = 10) -> list[str]:
        """Extract simple TF-based keywords from document texts."""
        word_counts: Counter = Counter()
        for doc in docs:
            words = doc.text.lower().split()
            # Filter short words and pure numbers
            filtered = [
                w.strip(".,;:()[]{}!?\"'") for w in words
                if len(w) > 3 and not w.isdigit()
            ]
            word_counts.update(filtered)
        return [w for w, _ in word_counts.most_common(top_n)]

    def _generate_name(
        self,
        keywords: list[str],
        cluster_docs: list[Document],
        fallback_idx: int,
    ) -> tuple[str, str]:
        """Generate a human-readable type name."""
        if self._llm is not None:
            try:
                return self._llm_generate_name(keywords, cluster_docs)
            except Exception:
                pass

        if keywords:
            name = " ".join(w.capitalize() for w in keywords[:3])
        else:
            name = f"Unknown-Type-{fallback_idx}"

        desc = (
            f"Auto-discovered type with keywords: {', '.join(keywords[:5])}. "
            f"Sample count: {len(cluster_docs)}."
        )
        return name, desc

    def _llm_generate_name(
        self, keywords: list[str], cluster_docs: list[Document],
    ) -> tuple[str, str]:
        """Use LLM to generate a descriptive type name."""
        assert self._llm is not None

        sample_texts = [d.text[:500] for d in cluster_docs[:3]]
        combined = "\n---\n".join(sample_texts)

        prompt = (
            "<instruction>Name and describe a newly discovered document type "
            "based on its characteristics.</instruction>\n"
            f"<keywords>{', '.join(keywords[:15])}</keywords>\n"
            f"<sample_documents>{combined[:1500]}</sample_documents>\n"
            "<output_schema>"
            '{"type_name": "Short-Descriptive-Name", '
            '"description": "One-sentence description."}'
            "</output_schema>"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You name and describe newly discovered document types. "
                    "Generate a concise, descriptive name (2-4 words). "
                    "Always respond with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        result = self._llm._call_llm(messages)
        name = result.get("type_name", f"Type-{len(cluster_docs)}")
        desc = result.get("description", "Auto-discovered document type.")
        return str(name), str(desc)
