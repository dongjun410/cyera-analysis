"""Tests for Tier 0 PII feature extraction engine.

Test contracts (TDD — write these BEFORE implementation):
  1. SSN detection with clean context (no boost/penalty terms nearby)
  2. SSN detection with penalty terms (lowered confidence, NOT discarded)
  3. Credit card with valid Luhn checksum
  4. Credit card with invalid Luhn checksum (validator rejects)
  5. Multiple entity types in a single document
  6. Empty document (no crash, empty feature vector)
  7. Context window boundary (inclusive at ±100 chars)
  8. Batch extraction (multiple docs, correct doc_ids)
"""

from __future__ import annotations

import pytest

from src.tier0.engine import Tier0Engine
from src.types import PIIFeature, PIIFeatureVector


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> Tier0Engine:
    """Default engine with context_window=100, confidence_threshold=0.8."""
    config = {
        "context_window": 100,
        "confidence_threshold": 0.8,
    }
    return Tier0Engine(config)


@pytest.fixture
def low_threshold_engine() -> Tier0Engine:
    """Engine with lowered confidence threshold so clean SSN clears it."""
    config = {
        "context_window": 100,
        "confidence_threshold": 0.4,
    }
    return Tier0Engine(config)


# ──────────────────────────────────────────────────────────────
# Test 1: SSN detection — clean context
# ──────────────────────────────────────────────────────────────

def test_ssn_detection_clean_context(engine: Tier0Engine) -> None:
    """SSN detected with high confidence when no boost/penalty terms nearby."""
    text = "The number 123-45-6789 is in this document."
    result = engine.extract("doc1", text)

    # SSN feature present
    ssn_features = [f for f in result.pii_features if f.entity_type == "SSN"]
    assert len(ssn_features) == 1, f"Expected 1 SSN feature, got {len(ssn_features)}"

    ssn = ssn_features[0]
    assert ssn.confidence > 0.6, f"Confidence too low: {ssn.confidence}"
    assert ssn.context_flag == "clean", f"Expected 'clean', got '{ssn.context_flag}'"

    # Span should be within the text
    assert 0 <= ssn.span[0] < ssn.span[1] <= len(text)

    # Type distribution
    assert result.pii_type_distribution.get("SSN", 0) >= 1


# ──────────────────────────────────────────────────────────────
# Test 2: SSN with penalty terms
# ──────────────────────────────────────────────────────────────

def test_ssn_with_penalty_term(engine: Tier0Engine) -> None:
    """SSN near penalty terms: confidence lowered, NOT discarded, flagged."""
    text = "test SSN: 123-45-6789 sample"
    result = engine.extract("doc1", text)

    # SSN MUST still be in pii_features — Tier 2 may need it
    ssn_features = [f for f in result.pii_features if f.entity_type == "SSN"]
    assert len(ssn_features) >= 1, "SSN was incorrectly discarded!"

    ssn = ssn_features[0]
    # Penalty should lower confidence below base (0.70 * 0.5 = 0.35)
    assert ssn.confidence < 0.5, (
        f"Penalty should lower confidence, got {ssn.confidence}"
    )
    assert ssn.context_flag == "penalty_term_present", (
        f"Expected 'penalty_term_present', got '{ssn.context_flag}'"
    )

    # Vector-level penalty flag
    assert result.has_penalty_terms is True


# ──────────────────────────────────────────────────────────────
# Test 3: Credit card — valid Luhn
# ──────────────────────────────────────────────────────────────

def test_credit_card_luhn_valid(engine: Tier0Engine) -> None:
    """Valid Visa number passes Luhn and is detected with high confidence."""
    # 4532015112830366 is a known Luhn-valid Visa test number
    text = "4532015112830366"
    result = engine.extract("doc1", text)

    cc_features = [f for f in result.pii_features if f.entity_type == "CREDIT_CARD"]
    assert len(cc_features) >= 1, "Valid credit card not detected"

    cc = cc_features[0]
    # Base confidence 0.80 * validator(1.0) * clean(1.0) = 0.80
    assert cc.confidence >= 0.75, f"Confidence too low: {cc.confidence}"


# ──────────────────────────────────────────────────────────────
# Test 4: Credit card — invalid Luhn
# ──────────────────────────────────────────────────────────────

def test_credit_card_luhn_invalid(engine: Tier0Engine) -> None:
    """Visa-format number that fails Luhn is rejected by the validator."""
    # 4000000000000000 matches Visa regex but fails Luhn checksum
    text = "4000000000000000"
    result = engine.extract("doc1", text)

    cc_features = [f for f in result.pii_features if f.entity_type == "CREDIT_CARD"]
    assert len(cc_features) == 0, (
        f"Invalid credit card should not appear in features, got {cc_features}"
    )


# ──────────────────────────────────────────────────────────────
# Test 5: Multiple entity types
# ──────────────────────────────────────────────────────────────

def test_multiple_entity_types(engine: Tier0Engine) -> None:
    """Document with SSN, email, and phone produces all three entity types."""
    text = "SSN: 123-45-6789, email: test@example.com, phone: 555-123-4567"
    result = engine.extract("doc1", text)

    detected_types = {f.entity_type for f in result.pii_features}
    assert "SSN" in detected_types, f"SSN not found in {detected_types}"
    assert "EMAIL" in detected_types, f"EMAIL not found in {detected_types}"
    assert "PHONE_US" in detected_types, f"PHONE_US not found in {detected_types}"

    # Distribution should count each type
    dist = result.pii_type_distribution
    for et in ("SSN", "EMAIL", "PHONE_US"):
        assert dist.get(et, 0) >= 1, f"{et} missing from distribution {dist}"


# ──────────────────────────────────────────────────────────────
# Test 6: Empty document
# ──────────────────────────────────────────────────────────────

def test_empty_document(engine: Tier0Engine) -> None:
    """Empty text returns empty feature vector with no crash."""
    result = engine.extract("doc1", "")

    assert result.doc_id == "doc1"
    assert result.pii_features == []
    assert result.pii_type_distribution == {}
    assert result.has_high_conf_pii is False
    assert result.has_penalty_terms is False


# ──────────────────────────────────────────────────────────────
# Test 7: Context window boundary
# ──────────────────────────────────────────────────────────────

def test_context_window_boundary(engine: Tier0Engine) -> None:
    """Penalty term exactly at context_window boundary is correctly flagged.

    Boundary-inclusive: a term at ±100 chars from the match is IN the window.
    A term at ±101 chars is OUT.
    """
    window = engine._config["context_window"]  # 100

    # ── Case A: penalty term exactly at boundary (included) ──
    # We need word boundaries around "test" for \btest\b to match, so we
    # insert a space after "test" and before the SSN.
    # Layout: "test "(5) + "a"*(window-6) + " "(1) + "123-45-6789"(11)
    # "test" at [0,3] (with \b at pos 4 from trailing space).
    # SSN at [window, window+10]. Context [0, window+10+window].
    # "test" IS in context → flagged.
    padding_in = "a" * (window - len("test ") - 1)  # 94 chars
    text_in = "test " + padding_in + " " + "123-45-6789"

    result_in = engine.extract("doc_boundary_in", text_in)
    ssn_in = [f for f in result_in.pii_features if f.entity_type == "SSN"]
    assert len(ssn_in) >= 1, "SSN should be detected (boundary-in)"
    assert ssn_in[0].context_flag == "penalty_term_present", (
        f"Expected penalty at boundary, got '{ssn_in[0].context_flag}'"
    )

    # ── Case B: penalty term just outside boundary (excluded) ──
    # Layout: "test "(5) + "a"*(window-5) + " "(1) + "123-45-6789"(11)
    # "test" at [0,3], SSN at [window+1, window+11].
    # Context [1, window+11+window].
    # "test" at [0,3] is NOT in context [1, ...] → clean.
    padding_out = "a" * (window - len("test "))  # 95 chars
    text_out = "test " + padding_out + " " + "123-45-6789"

    result_out = engine.extract("doc_boundary_out", text_out)
    ssn_out = [f for f in result_out.pii_features if f.entity_type == "SSN"]
    assert len(ssn_out) >= 1, "SSN should be detected (boundary-out)"
    assert ssn_out[0].context_flag == "clean", (
        f"Expected clean outside boundary, got '{ssn_out[0].context_flag}'"
    )


# ──────────────────────────────────────────────────────────────
# Test 8: Batch extraction
# ──────────────────────────────────────────────────────────────

def test_batch_extraction(engine: Tier0Engine) -> None:
    """extract_batch with 3 docs returns 3 PIIFeatureVectors with correct IDs."""
    docs: list[tuple[str, str]] = [
        ("doc1", "SSN: 123-45-6789"),
        ("doc2", "Email: user@company.com"),
        ("doc3", "Credit card: 4532015112830366"),
    ]
    results = engine.extract_batch(docs)

    assert len(results) == 3
    assert all(isinstance(r, PIIFeatureVector) for r in results)
    assert results[0].doc_id == "doc1"
    assert results[1].doc_id == "doc2"
    assert results[2].doc_id == "doc3"

    # Each doc should have at least one feature
    assert len(results[0].pii_features) >= 1, "doc1 should have features"
    assert len(results[1].pii_features) >= 1, "doc2 should have features"
    assert len(results[2].pii_features) >= 1, "doc3 should have features"


# ──────────────────────────────────────────────────────────────
# Additional edge-case tests
# ──────────────────────────────────────────────────────────────

def test_has_high_conf_pii_with_low_threshold(
    low_threshold_engine: Tier0Engine,
) -> None:
    """With threshold=0.4, clean SSN (conf=0.70) sets has_high_conf_pii=True."""
    result = low_threshold_engine.extract("doc1", "The number 123-45-6789 is here.")
    assert result.has_high_conf_pii is True


def test_custom_patterns(engine: Tier0Engine) -> None:
    """Custom patterns are merged with builtin and used for detection."""
    custom = [
        {
            "entity_type": "CUSTOM_ID",
            "regex": r"\bCUST-\d{6}\b",
            "validation": None,
            "context_boost_terms": ["VIP"],
            "context_penalty_terms": ["test"],
            "min_confidence": 0.90,
        }
    ]
    engine_with_custom = Tier0Engine(
        {"context_window": 100, "confidence_threshold": 0.8},
        custom_patterns=custom,
    )
    result = engine_with_custom.extract("doc1", "VIP client: CUST-123456")
    custom_features = [f for f in result.pii_features if f.entity_type == "CUSTOM_ID"]
    assert len(custom_features) == 1
    assert custom_features[0].confidence > 0.9


def test_config_penalty_terms_merged(engine: Tier0Engine) -> None:
    """Config-level penalty terms are applied globally to all patterns."""
    config_with_extra_penalty = {
        "context_window": 100,
        "confidence_threshold": 0.8,
        "context_penalty_terms": ["classified"],
    }
    eng = Tier0Engine(config_with_extra_penalty)
    # "classified" is not in the builtin PENALTY_TERMS, so a clean SSN near it
    # should be flagged only if the config term is merged correctly.
    result = eng.extract("doc1", "classified document: 123-45-6789")
    ssn = [f for f in result.pii_features if f.entity_type == "SSN"]
    assert len(ssn) >= 1
    assert ssn[0].context_flag == "penalty_term_present"


def test_no_match_document(engine: Tier0Engine) -> None:
    """Document with no PII patterns returns empty features.

    Uses short words (< 5 chars) to avoid matching the broad HICN pattern
    (\\b[A-Za-z0-9]{5,12}\\b) which would match common English words.
    """
    result = engine.extract("doc1", "a b c d")
    assert result.pii_features == []
    assert result.pii_type_distribution == {}
    assert result.has_high_conf_pii is False
    assert result.has_penalty_terms is False
    assert result.doc_id == "doc1"
