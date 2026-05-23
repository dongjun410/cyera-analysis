"""E4: Semantic kNN engine.

BGE-M3 embedding → cosine similarity to all known type centroids →
nearest centroid label + distance-based confidence.

Activation: type library must have >= 5 active types with centroids.
Coverage: broadest engine — can match any document type with a centroid.

Degradation: if BGE-M3 is unavailable, this engine returns "unavailable".
Only E4 is affected — all other engines continue independently.
"""

from __future__ import annotations

import numpy as np

from src.engines.base import BaseEngine
from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, EngineOutput


class E4kNNEngine(BaseEngine):
    """Semantic kNN engine — BGE-M3 embedding + centroid similarity.

    Attributes:
        engine_id: "E4_knn"
        weight: 1.0 (semantic similarity, broad coverage)
    """

    engine_id = "E4_knn"

    def __init__(
        self,
        embedder=None,
        type_library: TypeLibrary | None = None,
        min_types: int = 5,
    ) -> None:
        """Initialize the kNN engine.

        Args:
            embedder: BgeM3Embedder instance (or any object with encode()).
            type_library: TypeLibrary for centroid lookup.
            min_types: Minimum active types with centroids to activate.
        """
        self._embedder = embedder
        self._type_library = type_library or get_type_library()
        self._min_types = min_types

    @property
    def weight(self) -> float:
        return 1.0

    @property
    def is_available(self) -> bool:
        if self._embedder is None:
            return False
        centroids = self._type_library.list_centroids()
        return len(centroids) >= 1  # Only need 1 centroid to be useful

    def bootstrap_centroids(self, force: bool = False) -> int:
        """Bootstrap centroids from type keywords when no real samples exist.

        Encodes the concatenated keywords of each type as a zero-shot
        centroid. Only bootstraps types that don't already have a centroid.

        Args:
            force: If True, overwrite existing centroids too.

        Returns:
            Number of centroids bootstrapped.
        """
        if self._embedder is None:
            return 0

        count = 0
        for info in self._type_library.list_active():
            if info.centroid is not None and not force:
                continue

            if not info.keywords:
                continue

            # Build a pseudo-document from keywords
            pseudo_text = f"Document type: {info.type_name}. "
            pseudo_text += "Keywords: " + ", ".join(info.keywords) + ". "
            pseudo_text += f"This is a {info.type_name.lower()} document."

            try:
                embedding = self._embedder.encode([pseudo_text])[0]
                info.centroid = embedding.astype(np.float32)
                info.sample_count = max(info.sample_count, 1)
                count += 1
            except Exception:
                continue

        return count

    def analyze(self, doc: Document) -> EngineOutput:
        """Embed document + find nearest type centroid.

        Returns:
            EngineOutput with nearest type label and cosine similarity
            as confidence, or "unavailable"/"no_match".
        """
        if not self.is_available or self._embedder is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )

        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        centroids = self._type_library.list_centroids()
        if not centroids:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        try:
            # Reuse existing embedding if available on doc
            if doc.embedding is not None:
                embedding = doc.embedding
            else:
                embedding = self._embedder.encode([text])[0]

            best_label = None
            best_similarity = -1.0

            for label, centroid in centroids:
                if centroid is None:
                    continue
                sim = float(np.dot(embedding, centroid))
                if sim > best_similarity:
                    best_similarity = sim
                    best_label = label

            if best_label is None:
                return EngineOutput(
                    engine_id=self.engine_id,
                    status="no_match",
                )

            return EngineOutput(
                engine_id=self.engine_id,
                label=best_label,
                confidence=round(max(0.0, best_similarity), 4),
                status="matched",
                metadata={"cosine_similarity": round(best_similarity, 4)},
            )
        except Exception:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )
