"""Type library — central registry of known document types.

Manages the lifecycle of document types: registration, centroid updates,
keyword management, PII distribution tracking, and expiration.

Per spec section 5, types have a source (builtin/llm/discovery), status
(active/deprecated), and optional expiration for dynamic types.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class TypeInfo:
    """Metadata for a single known document type.

    Attributes:
        type_id: Unique identifier (snake_case).
        type_name: Human-readable display name.
        source: "builtin", "llm", or "discovery".
        status: "active" or "deprecated".
        centroid: Mean BGE-M3 embedding of type exemplars (1024-dim).
        keywords: Representative keywords for this type.
        pii_distribution: Typical PII type frequency distribution.
        structural_signatures: Structural hashes seen for this type.
        sample_count: Number of documents labeled as this type.
        created_at: Unix timestamp of first registration.
        last_seen_at: Unix timestamp of most recent sample.
        rules: E1 regex rule IDs associated with this type.
        template_hashes: E2 template hashes for this type.
    """

    __slots__ = (
        "type_id", "type_name", "source", "status", "centroid",
        "keywords", "pii_distribution", "structural_signatures",
        "sample_count", "created_at", "last_seen_at",
        "rules", "template_hashes",
    )

    def __init__(
        self,
        type_id: str,
        type_name: str,
        source: str = "builtin",
        centroid: np.ndarray | None = None,
        keywords: list[str] | None = None,
        pii_distribution: dict[str, float] | None = None,
        structural_signatures: list[str] | None = None,
        rules: list[str] | None = None,
        template_hashes: list[str] | None = None,
    ) -> None:
        self.type_id = type_id
        self.type_name = type_name
        self.source = source
        self.status = "active"
        self.centroid = centroid
        self.keywords = keywords or []
        self.pii_distribution = pii_distribution or {}
        self.structural_signatures = structural_signatures or []
        self.sample_count = 0
        self.created_at = time.time()
        self.last_seen_at = self.created_at
        self.rules = rules or []
        self.template_hashes = template_hashes or []

    def is_deprecated(self, max_idle_days: int = 180) -> bool:
        """Check if type should be deprecated (no new samples)."""
        if self.source == "builtin":
            return False
        idle = time.time() - self.last_seen_at
        return idle > max_idle_days * 86400

    def is_expired(self, max_idle_days: int = 365) -> bool:
        """Check if type should be deleted."""
        if self.source == "builtin":
            return False
        return time.time() - self.last_seen_at > max_idle_days * 86400

    def update_centroid(self, new_embedding: np.ndarray) -> None:
        """Incremental centroid update with a new sample embedding."""
        if self.centroid is None:
            self.centroid = np.asarray(new_embedding, dtype=np.float32)
        else:
            n = float(self.sample_count)
            self.centroid = (self.centroid * n + np.asarray(new_embedding, dtype=np.float32)) / (n + 1.0)
        self.sample_count += 1
        self.last_seen_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (centroid omitted)."""
        return {
            "type_id": self.type_id,
            "type_name": self.type_name,
            "source": self.source,
            "status": self.status,
            "keywords": self.keywords,
            "pii_distribution": self.pii_distribution,
            "structural_signatures": self.structural_signatures,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
        }


class TypeLibrary:
    """Central registry of known document types.

    Provides registration, lookup, centroid management, and type
    lifecycle operations (deprecation, expiration, dedup).

    Usage:
        lib = TypeLibrary()
        lib.register("hr_payroll", "HR & Payroll", source="builtin")
        info = lib.get("hr_payroll")
        all_types = lib.list_active()
    """

    def __init__(self) -> None:
        self._types: dict[str, TypeInfo] = {}

    def register(
        self,
        type_id: str,
        type_name: str,
        source: str = "builtin",
        centroid: np.ndarray | None = None,
        keywords: list[str] | None = None,
        pii_distribution: dict[str, float] | None = None,
        structural_signatures: list[str] | None = None,
        rules: list[str] | None = None,
        template_hashes: list[str] | None = None,
    ) -> TypeInfo:
        """Register a new type or return existing one.

        If a type with the same type_id already exists, it is returned
        unchanged (no overwrite). Use update_centroid() to add samples.
        """
        if type_id in self._types:
            return self._types[type_id]

        info = TypeInfo(
            type_id=type_id,
            type_name=type_name,
            source=source,
            centroid=centroid,
            keywords=keywords,
            pii_distribution=pii_distribution,
            structural_signatures=structural_signatures,
            rules=rules,
            template_hashes=template_hashes,
        )
        self._types[type_id] = info
        return info

    def get(self, type_id: str) -> TypeInfo | None:
        """Look up a type by ID."""
        return self._types.get(type_id)

    def get_by_name(self, type_name: str) -> TypeInfo | None:
        """Look up a type by display name."""
        for info in self._types.values():
            if info.type_name == type_name:
                return info
        return None

    def list_active(self) -> list[TypeInfo]:
        """Return all active (non-deprecated) types."""
        return [t for t in self._types.values() if t.status == "active"]

    def list_centroids(self) -> list[tuple[str, np.ndarray | None]]:
        """Return (type_name, centroid) for all active types with centroids."""
        return [
            (t.type_name, t.centroid)
            for t in self._types.values()
            if t.status == "active" and t.centroid is not None
        ]

    def check_deprecation(self) -> list[str]:
        """Return type_ids that should be deprecated (>180 days idle)."""
        return [tid for tid, info in self._types.items() if info.is_deprecated()]

    def check_expiration(self) -> list[str]:
        """Return type_ids that should be deleted (>365 days idle)."""
        return [tid for tid, info in self._types.items() if info.is_expired()]

    def deprecate(self, type_id: str) -> None:
        """Mark a type as deprecated."""
        info = self._types.get(type_id)
        if info:
            info.status = "deprecated"

    def remove(self, type_id: str) -> None:
        """Remove a type from the library."""
        self._types.pop(type_id, None)

    def suggest_merges(
        self,
        name_threshold: float = 0.8,
        semantic_threshold: float = 0.15,
    ) -> list[dict]:
        """Find candidate type pairs for merging.

        Uses normalized edit distance for name similarity and cosine
        distance between centroids for semantic similarity.

        Never auto-merges — suggestions must be confirmed by an operator.

        Args:
            name_threshold: Minimum name similarity to flag (0.0-1.0).
            semantic_threshold: Maximum semantic distance to flag.

        Returns:
            List of dicts with keys: type_id_a, type_id_b, name_a, name_b,
            name_similarity, semantic_distance, suggested_action.
        """
        active = self.list_active()
        suggestions: list[dict] = []

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]

                # Name similarity via normalized edit distance
                name_sim = TypeLibrary._name_similarity(
                    a.type_name, b.type_name
                )
                if name_sim < name_threshold:
                    continue

                # Semantic distance via centroids
                sem_dist = 1.0
                if a.centroid is not None and b.centroid is not None:
                    cos_sim = float(np.dot(a.centroid, b.centroid))
                    cos_sim = max(-1.0, min(1.0, cos_sim))
                    sem_dist = 1.0 - cos_sim

                if sem_dist < semantic_threshold:
                    # Keep the type with more samples as the primary
                    if a.sample_count >= b.sample_count:
                        keeper, merged = a, b
                    else:
                        keeper, merged = b, a

                    suggestions.append({
                        "type_id_a": keeper.type_id,
                        "type_id_b": merged.type_id,
                        "name_a": keeper.type_name,
                        "name_b": merged.type_name,
                        "name_similarity": round(name_sim, 4),
                        "semantic_distance": round(sem_dist, 4),
                        "suggested_action": "merge_b_into_a",
                        "keeper_sample_count": keeper.sample_count,
                        "merged_sample_count": merged.sample_count,
                    })

        return suggestions

    def merge_types(self, type_id_a: str, type_id_b: str) -> TypeInfo | None:
        """Merge type B into type A.

        Combines centroids (weighted average), merges keyword lists,
        structural signatures, rules, and template hashes.
        Type B is deprecated after merging.

        Args:
            type_id_a: The keeper type ID.
            type_id_b: The type to merge and deprecate.

        Returns:
            The merged TypeInfo (type A), or None if either type not found.
        """
        a = self._types.get(type_id_a)
        b = self._types.get(type_id_b)
        if a is None or b is None:
            return None

        # Merge centroids (weighted by sample count)
        if a.centroid is not None and b.centroid is not None:
            total = a.sample_count + b.sample_count
            if total > 0:
                a.centroid = (
                    a.centroid * a.sample_count + b.centroid * b.sample_count
                ) / total
        elif b.centroid is not None:
            a.centroid = b.centroid.copy()

        # Merge keywords (deduplicated, preserving order)
        seen = set(a.keywords)
        for kw in b.keywords:
            if kw not in seen:
                a.keywords.append(kw)
                seen.add(kw)

        # Merge structural signatures
        for sig in b.structural_signatures:
            if sig not in a.structural_signatures:
                a.structural_signatures.append(sig)

        # Merge rules and template hashes
        a.rules = list(set(a.rules + b.rules))
        a.template_hashes = list(set(a.template_hashes + b.template_hashes))

        # Merge PII distribution
        for pii_type, freq in b.pii_distribution.items():
            a.pii_distribution[pii_type] = (
                a.pii_distribution.get(pii_type, 0.0) + freq
            )

        # Update counts
        a.sample_count += b.sample_count
        a.last_seen_at = max(a.last_seen_at, b.last_seen_at)

        # Deprecate B
        self.deprecate(type_id_b)

        return a

    @staticmethod
    def _name_similarity(name_a: str, name_b: str) -> float:
        """Compute normalized edit distance similarity between two names.

        Returns 1.0 for identical strings, ~0.0 for completely different.
        """
        if name_a == name_b:
            return 1.0
        if not name_a or not name_b:
            return 0.0

        # Levenshtein distance
        m, n = len(name_a), len(name_b)
        if m == 0 or n == 0:
            return 0.0

        # Use 2-row DP for memory efficiency
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if name_a[i - 1] == name_b[j - 1] else 1
                curr[j] = min(
                    curr[j - 1] + 1,      # insert
                    prev[j] + 1,           # delete
                    prev[j - 1] + cost,    # substitute
                )
            prev, curr = curr, prev

        distance = prev[n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len)

    @property
    def count(self) -> int:
        return len(self._types)

    @property
    def active_count(self) -> int:
        return len(self.list_active())


# ── Builtin types — 13 document types registered at factory ──

BUILTIN_TYPE_NAMES: list[tuple[str, str, list[str]]] = [
    ("hr_payroll", "HR & Payroll", ["payroll", "salary", "W2", "SSN", "benefits", "compensation"]),
    ("financial_report", "Financial Report", ["revenue", "invoice", "balance", "expense", "budget", "tax"]),
    ("medical_record", "Medical Record", ["diagnosis", "patient", "prescribed", "HIPAA", "lab", "NPI"]),
    ("legal_document", "Legal Document", ["contract", "agreement", "NDA", "compliance", "litigation"]),
    ("api_log", "API Log", ["timestamp", "endpoint", "status", "request", "response"]),
    ("technical_document", "Technical Document", ["config", "server", "database", "function", "README"]),
    ("identity_document", "Identity Document", ["passport", "license", "birth", "resume", "application"]),
    ("email_communication", "Email / Communication", ["From:", "Subject:", "meeting", "memo", "agenda"]),
    ("government_form", "Government Form", ["form", "grant", "permit", "census", "federal"]),
    ("education_record", "Education Record", ["transcript", "diploma", "GPA", "enrollment", "semester"]),
    ("real_estate", "Real Estate Document", ["deed", "mortgage", "lease", "property", "escrow"]),
    ("marketing", "Marketing Document", ["campaign", "analytics", "pitch", "CTR", "conversion"]),
    ("scientific_paper", "Scientific Paper", ["abstract", "methodology", "clinical", "patent", "journal"]),
]

# Global singleton — initialized once at startup
_default_library: TypeLibrary | None = None


def get_type_library() -> TypeLibrary:
    """Return the global TypeLibrary singleton, initializing if needed."""
    global _default_library
    if _default_library is None:
        _default_library = TypeLibrary()
        for tid, tname, keywords in BUILTIN_TYPE_NAMES:
            _default_library.register(
                type_id=tid,
                type_name=tname,
                source="builtin",
                keywords=keywords,
            )
    return _default_library
