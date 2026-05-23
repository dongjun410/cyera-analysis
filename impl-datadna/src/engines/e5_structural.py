"""E5: Structural signature engine.

Extracts document structural features (file type, size quantile, page/para
counts, etc.), hashes them with SHA256, and matches against known structural
signatures in the type library.

Coverage: narrow — only distinguishes file FORMAT, not content type.
Weight: lowest (0.8) — auxiliary signal, not primary classifier.
"""

from __future__ import annotations

import hashlib
import math

from src.engines.base import BaseEngine
from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, EngineOutput


class E5StructuralEngine(BaseEngine):
    """Structural signature engine — file metadata hash matching.

    Attributes:
        engine_id: "E5_structural"
        weight: 0.8 (lowest — format-level only, not content)
    """

    engine_id = "E5_structural"

    _FEATURE_KEYS = [
        "file_type", "file_size_quantile", "page_count",
        "paragraph_count", "table_count", "has_images",
        "header_pattern", "json_schema_signature", "path_depth",
    ]

    def __init__(self, type_library: TypeLibrary | None = None) -> None:
        self._type_library = type_library or get_type_library()

    @property
    def weight(self) -> float:
        return 0.8

    def analyze(self, doc: Document) -> EngineOutput:
        """Extract structural signature, hash, and match against type library.

        Returns:
            EngineOutput with matched type or status="no_match".
        """
        meta = doc.metadata or {}
        if not meta:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        # Build structural feature string
        signature = self._build_signature(meta)
        sig_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()

        # Check against known structural signatures in type library
        for info in self._type_library.list_active():
            if sig_hash in info.structural_signatures:
                return EngineOutput(
                    engine_id=self.engine_id,
                    label=info.type_name,
                    confidence=1.0,
                    status="matched",
                    metadata={"signature_hash": sig_hash},
                )

        return EngineOutput(
            engine_id=self.engine_id,
            status="no_match",
            metadata={"signature_hash": sig_hash},
        )

    def _build_signature(self, metadata: dict) -> str:
        """Build canonical structural feature string for hashing."""
        parts = []
        for key in sorted(self._FEATURE_KEYS):
            val = metadata.get(key)
            if val is None:
                val = self._default_for_key(key)
            parts.append(f"{key}:{self._format_value(val)}")
        return "|".join(parts)

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    @staticmethod
    def _default_for_key(key: str):
        defaults = {
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
