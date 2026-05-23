"""E2: Template hash engine.

PII detection → entity type placeholder replacement → SHA256 hash →
template library lookup.

Inspired by Cyera's metadata replacement scheme (patent US12026123B2).
If the document has < 3 PII entities, template match is unlikely and
this engine naturally produces no output — the fusion voter handles it.

Dependencies:
  - PII_PATTERNS from knowledge/rules.py (shared regex patterns, not runtime)
  - TemplateLibrary from knowledge/templates.py
"""

from __future__ import annotations

import hashlib

from src.engines.base import BaseEngine
from src.knowledge.rules import PII_PATTERNS
from src.knowledge.templates import BUILTIN_TEMPLATES, TemplateLibrary
from src.types import Document, EngineOutput


class E2TemplateEngine(BaseEngine):
    """Template hash engine — PII replacement + SHA256 + template lookup.

    Independent of E1 regex engine. PII detection here uses the same
    pattern definitions but runs independently — if PII_PATTERNS is
    corrupted, only E2 is affected, not E1.

    Attributes:
        engine_id: "E2_template"
        weight: 1.0 (deterministic, match = high confidence)
    """

    engine_id = "E2_template"

    def __init__(self, template_library: TemplateLibrary | None = None) -> None:
        self._library = template_library or BUILTIN_TEMPLATES
        self._pii_patterns = PII_PATTERNS

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, doc: Document) -> EngineOutput:
        """Replace PII entities, hash, and look up in template library.

        Returns:
            EngineOutput with matched label and confidence=1.0, or
            status="no_match" if hash not found in library.
        """
        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        # Step 1: Detect and replace PII entities with type placeholders
        replaced_text, pii_count = self._replace_pii(text)

        # Step 2: SHA256 hash of the replaced text
        template_hash = hashlib.sha256(
            replaced_text.encode("utf-8")
        ).hexdigest()

        # Step 3: Look up in template library
        entry = self._library.lookup(template_hash)
        if entry is not None:
            return EngineOutput(
                engine_id=self.engine_id,
                label=entry.label,
                confidence=1.0,
                status="matched",
                metadata={
                    "template_hash": template_hash,
                    "pii_count": pii_count,
                    "source": entry.source,
                },
            )

        # Try partial match (first 16 hex chars) as fallback
        partial = self._library.partial_match(template_hash, prefix_len=16)
        if partial is not None:
            return EngineOutput(
                engine_id=self.engine_id,
                label=partial.label,
                confidence=0.5,
                status="matched",
                metadata={
                    "template_hash": template_hash,
                    "pii_count": pii_count,
                    "match_type": "partial",
                },
            )

        return EngineOutput(
            engine_id=self.engine_id,
            status="no_match",
            metadata={"pii_count": pii_count},
        )

    def _replace_pii(self, text: str) -> tuple[str, int]:
        """Replace detected PII entities with type placeholders.

        Returns (replaced_text, entity_count).
        """
        count = 0
        # Collect all matches with positions
        matches: list[tuple[int, int, str]] = []
        for pii_type, pattern in self._pii_patterns.items():
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), pii_type))

        if not matches:
            return text, 0

        # Sort by start, then by end descending (longest match first)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Build replaced text, skipping overlapping matches
        result: list[str] = []
        pos = 0
        last_end = 0
        for start, end, pii_type in matches:
            if start < last_end:
                continue  # Skip overlapping
            result.append(text[pos:start])
            result.append(f"[{pii_type}]")
            pos = end
            last_end = end
            count += 1

        result.append(text[pos:])
        return "".join(result), count
