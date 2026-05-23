"""Pre-computed template hashes for E2 template engine.

Each template is a SHA256 hash of a PII-replaced document sample.
When E2 processes a document, it replaces PII entities with type
placeholders, hashes the result, and checks against this library.

Templates are organized by document type for O(1) lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemplateEntry:
    """A single template hash entry in the library.

    Attributes:
        template_hash: SHA256 hex digest of PII-replaced sample text.
        label: Document type this template represents.
        sample_count: How many samples contributed to this template.
        source: "builtin" (pre-computed) or "learned" (from customer data).
    """

    template_hash: str
    label: str
    sample_count: int = 1
    source: str = "builtin"


class TemplateLibrary:
    """In-memory template hash library with O(1) lookup.

    Usage:
        lib = TemplateLibrary()
        lib.add("abc123...", "HR & Payroll")
        result = lib.lookup("abc123...")  # → TemplateEntry or None
    """

    def __init__(self) -> None:
        self._templates: dict[str, TemplateEntry] = {}
        self._by_label: dict[str, list[str]] = {}

    def add(
        self, template_hash: str, label: str, source: str = "builtin"
    ) -> None:
        """Add or update a template hash entry.

        If the hash already exists with the same label, increment sample_count.
        """
        if template_hash in self._templates:
            existing = self._templates[template_hash]
            if existing.label == label:
                existing.sample_count += 1
            return

        entry = TemplateEntry(
            template_hash=template_hash,
            label=label,
            sample_count=1,
            source=source,
        )
        self._templates[template_hash] = entry
        self._by_label.setdefault(label, []).append(template_hash)

    def lookup(self, template_hash: str) -> TemplateEntry | None:
        """Look up a template hash. Returns None if not found."""
        return self._templates.get(template_hash)

    def partial_match(
        self, template_hash: str, prefix_len: int = 16
    ) -> TemplateEntry | None:
        """Look up by hash prefix (partial match fallback)."""
        for h, entry in self._templates.items():
            if h[:prefix_len] == template_hash[:prefix_len]:
                return entry
        return None

    def hashes_for_label(self, label: str) -> list[str]:
        """Return all template hashes for a given label."""
        return list(self._by_label.get(label, []))

    @property
    def count(self) -> int:
        return len(self._templates)

    @property
    def labels(self) -> list[str]:
        return sorted(self._by_label.keys())


# ── Builtin template library (empty at factory — populated from samples) ──

BUILTIN_TEMPLATES = TemplateLibrary()

# Pre-populate with placeholder entries for common types.
# Real hashes will be computed from actual sample documents during
# Phase 0 knowledge building. These labels ensure the template library
# is aware of all types even before real templates are loaded.
_BUILTIN_LABELS = [
    "HR & Payroll",
    "Financial Report",
    "Medical Record",
    "Legal Document",
    "API Log",
    "Technical Document",
    "Identity Document",
    "Email / Communication",
    "Government Form",
    "Education Record",
    "Real Estate Document",
    "Marketing Document",
    "Scientific Paper",
]

for _lbl in _BUILTIN_LABELS:
    BUILTIN_TEMPLATES._by_label.setdefault(_lbl, [])
