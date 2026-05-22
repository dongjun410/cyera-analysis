"""Tier 1 Stage A: Deterministic Structural Hashing.

O(N) time, no K parameter. Produces a SHA256 bucket ID from document
structural features. Same structure → same bucket.

Also supports database column structural hashing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any

from src.types import Document, StructuralFeatures


# ──────────────────────────────────────────────────────────────
# Default feature set used for canonical hashing
# ──────────────────────────────────────────────────────────────

DEFAULT_FEATURE_CONFIG: list[str] = [
    "file_type",
    "file_size_quantile",
    "page_count",
    "paragraph_count",
    "table_count",
    "has_images",
    "header_pattern",
    "json_schema_signature",
    "path_depth",
]


# ──────────────────────────────────────────────────────────────
# StructuralClusterer
# ──────────────────────────────────────────────────────────────

class StructuralClusterer:
    """Deterministic structural hashing for document bucketing.

    Extracts a canonical feature string from each document's
    StructuralFeatures (or metadata), hashes it with SHA256, and
    assigns documents with identical structure to the same bucket.

    O(N) time. No K parameter. Fully deterministic.
    """

    def __init__(self, feature_config: list[str] | None = None) -> None:
        """Initialize with optional feature subset configuration.

        Args:
            feature_config: List of feature keys to use for hashing.
                Default includes all 9 structural features.
        """
        self._feature_config: list[str] = (
            feature_config if feature_config is not None
            else list(DEFAULT_FEATURE_CONFIG)
        )

    # ── Public API ────────────────────────────────────────────

    def extract_features(self, doc: Document) -> str:
        """Extract a structural hash string from a document.

        Canonicalizes the feature set, sorts keys, builds a stable
        string representation, and returns its SHA256 hex digest.

        Args:
            doc: Document with structural_features or metadata set.

        Returns:
            64-character SHA256 hex digest (the structural hash).
        """
        canonical = self._build_canonical_string(doc)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def assign_bucket(self, doc: Document) -> str:
        """Assign a document to a structural bucket.

        Same as extract_features — the hash IS the bucket ID.

        Args:
            doc: Document to assign.

        Returns:
            Bucket ID (64-char SHA256 hex digest).
        """
        return self.extract_features(doc)

    def cluster(self, documents: list[Document]) -> dict[str, list[str]]:
        """Group documents into structural buckets.

        Args:
            documents: List of Document objects.

        Returns:
            Mapping: {bucket_id: [doc_id, ...]}
        """
        buckets: dict[str, list[str]] = {}
        for doc in documents:
            bucket_id = self.assign_bucket(doc)
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(doc.doc_id)
        return buckets

    # ── Internal helpers ──────────────────────────────────────

    def _build_canonical_string(self, doc: Document) -> str:
        """Build a canonical, sorted feature string for hashing.

        If the document has structural_features set, those values are
        used. Otherwise, features are extracted from doc.metadata.

        Only features in self._feature_config are included.

        Format: "key1:value1|key2:value2|..." with keys sorted
        alphabetically for determinism.
        """
        if doc.structural_features is not None:
            features = self._from_structural_features(doc.structural_features)
        else:
            features = self._from_metadata(doc.metadata)

        # Filter to configured keys only
        filtered = {
            k: features.get(k, self._default_for_key(k))
            for k in self._feature_config
        }

        # Build sorted canonical string
        parts = []
        for key in sorted(filtered):
            value = filtered[key]
            parts.append(f"{key}:{self._format_value(value)}")

        return "|".join(parts)

    @staticmethod
    def _from_structural_features(sf: StructuralFeatures) -> dict[str, Any]:
        """Extract a flat dict from a StructuralFeatures dataclass."""
        raw = asdict(sf)
        # Remove 'extra' — not part of the core structural signature
        raw.pop("extra", None)
        return raw

    @staticmethod
    def _from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract structural features from a metadata dict.

        Handles:
        - Direct key matches (file_type, page_count, etc.)
        - file_size → file_size_quantile (log-bucketed)
        """
        features: dict[str, Any] = {}

        for key in (
            "file_type", "file_size_quantile", "page_count",
            "paragraph_count", "table_count", "has_images",
            "header_pattern", "json_schema_signature", "path_depth",
        ):
            if key in metadata:
                features[key] = metadata[key]

        # If file_size_quantile is missing but file_size is present, compute it
        if "file_size_quantile" not in features and "file_size" in metadata:
            features["file_size_quantile"] = StructuralClusterer._compute_size_quantile(
                metadata["file_size"]
            )

        return features

    @staticmethod
    def _compute_size_quantile(file_size: int) -> int:
        """Log-bucket a file size in bytes.

        Buckets:
          0:      0 – 1 KB
          1:   1 KB – 10 KB
          2:  10 KB – 100 KB
          3: 100 KB – 1 MB
          4:   1 MB – 10 MB
          5:  10 MB – 100 MB
          6: 100 MB+
        """
        if file_size <= 0:
            return 0
        return min(int(math.log10(file_size)), 9)

    @staticmethod
    def _default_for_key(key: str) -> Any:
        """Return the default value for a feature key when it's missing."""
        defaults: dict[str, Any] = {
            "file_type": "",
            "file_size_quantile": 0,
            "page_count": 0,
            "paragraph_count": 0,
            "table_count": 0,
            "has_images": False,
            "header_pattern": "",
            "json_schema_signature": "",
            "path_depth": 0,
        }
        return defaults.get(key, "")

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a feature value as a deterministic string.

        Bools → "1"/"0", ints → decimal, strings → as-is.
        """
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            # Avoid floating-point instability in hashes — round
            return f"{value:.6f}"
        return str(value)
