"""Tier 0: Deterministic PII feature extraction engine.

Consumes document text, applies builtin + custom regex patterns from
`src.tier0.patterns`, checks context windows for boost/penalty terms,
runs validators, and outputs `PIIFeatureVector` (dataclass from `src.types`).

This is FEATURE EXTRACTION, not final classification. Low-confidence
matches are never discarded — they carry a context_flag for Tier 2.
"""

from __future__ import annotations

import re
from typing import Callable

from src.tier0.patterns import BUILTIN_PATTERNS, PENALTY_TERMS, BOOST_TERMS, VALIDATORS
from src.types import PIIFeature, PIIFeatureVector


# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict = {
    "context_window": 100,
    "confidence_threshold": 0.8,
    "context_penalty_terms": [],
    "context_boost_terms": [],
}


# ──────────────────────────────────────────────────────────────
# Tier0Engine
# ──────────────────────────────────────────────────────────────

class Tier0Engine:
    """PII feature extractor using compiled regex + context analysis.

    Consumes document text, applies patterns, checks context windows,
    runs validators, and outputs PIIFeatureVector. Produces features,
    NOT final labels — never discards low-confidence matches.

    Usage::

        engine = Tier0Engine({"context_window": 100})
        vec = engine.extract("doc_01", "SSN: 123-45-6789")
        print(vec.pii_features[0].confidence)  # e.g. 0.84
    """

    def __init__(
        self,
        config: dict,
        custom_patterns: list[dict] | None = None,
    ) -> None:
        """Initialise the engine, merge + compile all patterns.

        Args:
            config: Dictionary with optional keys:
                - context_window (int, default 100)
                - confidence_threshold (float, default 0.8)
                - context_penalty_terms (list[str], merged globally)
                - context_boost_terms (list[str], merged globally)
            custom_patterns: Additional regex patterns in the same format
                as BUILTIN_PATTERNS entries.
        """
        # Merge config with defaults
        self._config: dict = {**_DEFAULT_CONFIG, **config}

        # Global extra terms from config
        extra_penalty: list[str] = list(
            self._config.get("context_penalty_terms") or []
        )
        extra_boost: list[str] = list(
            self._config.get("context_boost_terms") or []
        )

        # Merge builtin + custom patterns
        all_patterns: list[dict] = list(BUILTIN_PATTERNS)
        if custom_patterns:
            all_patterns.extend(custom_patterns)

        # Compile every pattern at init time (one-off cost)
        self._compiled: list[dict] = []
        for p in all_patterns:
            validator_name: str | None = p.get("validation")
            validator_fn: Callable[[str], bool] | None = (
                VALIDATORS.get(validator_name) if validator_name else None
            )

            self._compiled.append({
                "entity_type": p["entity_type"],
                "regex": re.compile(p["regex"]),
                "validator": validator_fn,
                "boost_terms": (
                    list(p.get("context_boost_terms") or []) + extra_boost
                ),
                "penalty_terms": (
                    list(p.get("context_penalty_terms") or []) + extra_penalty
                ),
                "min_confidence": float(p.get("min_confidence", 0.5)),
            })

    # ── Public API ──────────────────────────────────────────

    def extract(self, doc_id: str, text: str) -> PIIFeatureVector:
        """Extract PII features from a single document.

        Args:
            doc_id: Unique document identifier.
            text: Raw document text.

        Returns:
            PIIFeatureVector with detected entities, type distribution,
            and summary flags.
        """
        features: list[PIIFeature] = []

        if not text:
            return PIIFeatureVector(
                doc_id=doc_id,
                pii_features=[],
                pii_type_distribution={},
                has_high_conf_pii=False,
                has_penalty_terms=False,
            )

        window: int = int(self._config.get("context_window", 100))
        text_len: int = len(text)

        for cp in self._compiled:
            entity_type: str = cp["entity_type"]
            regex: re.Pattern = cp["regex"]
            validator: Callable[[str], bool] | None = cp["validator"]
            boost_terms: list[str] = cp["boost_terms"]
            penalty_terms: list[str] = cp["penalty_terms"]
            base_conf: float = cp["min_confidence"]

            for match in regex.finditer(text):
                start, end = match.span()
                matched_text: str = match.group()

                # ── Context window check ──
                ctx_start: int = max(0, start - window)
                ctx_end: int = min(text_len, end + window)
                context_slice: str = text[ctx_start:ctx_end]

                context_flag, context_modifier = self._check_context(
                    context_slice, penalty_terms, boost_terms
                )

                # ── Validator ──
                # If validator returns False, the match is structurally
                # invalid (e.g. failed Luhn checksum) — discard it entirely.
                # This differs from penalty context (which lowers confidence
                # but keeps the feature for Tier 2 review).
                if validator is not None and not validator(matched_text):
                    continue

                # ── Confidence ──
                confidence: float = base_conf * context_modifier
                confidence = max(0.0, min(1.0, confidence))

                features.append(PIIFeature(
                    entity_type=entity_type,
                    span=(start, end),
                    confidence=round(confidence, 4),
                    context_flag=context_flag,
                ))

        # ── Build feature vector ──
        distribution: dict[str, int] = {}
        has_penalty: bool = False
        has_high: bool = False
        threshold: float = float(self._config.get("confidence_threshold", 0.8))

        for f in features:
            distribution[f.entity_type] = distribution.get(f.entity_type, 0) + 1
            if f.context_flag == "penalty_term_present":
                has_penalty = True
            if f.confidence >= threshold:
                has_high = True

        return PIIFeatureVector(
            doc_id=doc_id,
            pii_features=features,
            pii_type_distribution=distribution,
            has_high_conf_pii=has_high,
            has_penalty_terms=has_penalty,
        )

    def extract_batch(
        self, docs: list[tuple[str, str]]
    ) -> list[PIIFeatureVector]:
        """Extract PII features from a batch of documents.

        Args:
            docs: List of (doc_id, text) tuples.

        Returns:
            List of PIIFeatureVector, one per input document, in order.
        """
        return [self.extract(doc_id, text) for doc_id, text in docs]

    # ── Internal helpers ────────────────────────────────────

    @staticmethod
    def _check_context(
        context_slice: str,
        penalty_terms: list[str],
        boost_terms: list[str],
    ) -> tuple[str, float]:
        """Scan context window for penalty/boost terms.

        Penalty overrides boost — if both are present, penalty wins.

        Args:
            context_slice: The text window around the match.
            penalty_terms: Terms that lower confidence.
            boost_terms: Terms that raise confidence.

        Returns:
            (context_flag, context_modifier) tuple.
            flag is one of "penalty_term_present", "boost_term_present", "clean".
            modifier is 0.5, 1.2, or 1.0 respectively.
        """
        # Check penalty terms first (they override boost)
        for term in penalty_terms:
            if _term_in_text(term, context_slice):
                return ("penalty_term_present", 0.5)

        # Check boost terms
        for term in boost_terms:
            if _term_in_text(term, context_slice):
                return ("boost_term_present", 1.2)

        return ("clean", 1.0)


# ──────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────

def _term_in_text(term: str, text: str) -> bool:
    """Case-insensitive word-boundary check for a term in text.

    Args:
        term: The term to search for.
        text: The text to search within.

    Returns:
        True if the term appears as a whole word in text.
    """
    # Escape special regex chars, wrap with word boundaries, case-insensitive
    try:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        return bool(pattern.search(text))
    except re.error:
        # Fallback: simple case-insensitive substring check
        return term.lower() in text.lower()
