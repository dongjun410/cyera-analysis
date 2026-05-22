"""Tests for Tier 2: KnownTypeMatcher — 3-signal weighted scoring (TDD).

Test contracts (write these BEFORE implementation):
  1. Exact match + overlapping keywords + similar PII → score >= 0.8, method="known_match"
  2. Completely different cluster → score < 0.5, method="unknown", matched_type=None
  3. Structural match only (different keywords/PII) → score in [0.5, 0.8), method="llm_confirm"
  4. Empty known types → score=0, method="unknown"
  5. Register type → match() finds it
  6. Both PII distributions empty → signal=0.0, not NaN or 1.0
"""

from __future__ import annotations

import math

import pytest

from src.tier2.matching import KnownTypeMatcher
from src.types import ClusterInfo, KnownType, MatchResult


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def financial_report_type() -> KnownType:
    """Known type: Financial Report with structural sig and PII distribution."""
    return KnownType(
        type_id="fin-report-001",
        type_name="Financial Report",
        description="Quarterly/annual financial statements with SSN and revenue data",
        structural_signature="sha256:abc123def456",
        tfidf_keywords=[
            "report", "financial", "annual", "revenue", "earnings",
            "quarter", "fiscal", "balance", "income", "statement",
        ],
        pii_distribution={"SSN": 45, "EMAIL": 35, "PHONE": 20},
    )


@pytest.fixture
def medical_record_type() -> KnownType:
    """Known type: Medical Record — very different from financial."""
    return KnownType(
        type_id="med-record-001",
        type_name="Medical Record",
        description="Patient health records with HIPAA PII",
        structural_signature="sha256:xyz789ghi012",
        tfidf_keywords=[
            "patient", "diagnosis", "treatment", "medication", "doctor",
            "hospital", "surgery", "prescription", "lab", "clinical",
        ],
        pii_distribution={"PATIENT_ID": 80, "DOB": 40, "PHONE": 10},
    )


@pytest.fixture
def financial_cluster() -> ClusterInfo:
    """Cluster that closely matches financial_report_type."""
    return ClusterInfo(
        cluster_id="cluster-001",
        doc_ids=["doc-001", "doc-002", "doc-003"],
        structural_bucket="sha256:abc123def456",  # matches financial_report_type
        cluster_radius=0.15,
        representative_docs=["doc-001"],
        tfidf_keywords=[
            "report", "financial", "annual", "revenue", "q4",
        ],
        pii_distribution={"SSN": 50, "EMAIL": 30, "PHONE": 20},
        language_distribution={"en": 3},
    )


@pytest.fixture
def unknown_cluster() -> ClusterInfo:
    """Cluster that does not match any known type."""
    return ClusterInfo(
        cluster_id="cluster-999",
        doc_ids=["doc-900", "doc-901"],
        structural_bucket="sha256:completely-different-hash-999",
        cluster_radius=0.25,
        representative_docs=["doc-900"],
        tfidf_keywords=["gibberish", "unknown", "random", "noise", "zzz"],
        pii_distribution={"CUSTOM_TYPE": 5},
        language_distribution={"xx": 2},
    )


@pytest.fixture
def partial_match_cluster() -> ClusterInfo:
    """Cluster with matching structure but completely different content."""
    return ClusterInfo(
        cluster_id="cluster-050",
        doc_ids=["doc-050"],
        structural_bucket="sha256:abc123def456",  # matches financial_report_type
        cluster_radius=0.0,
        representative_docs=["doc-050"],
        tfidf_keywords=["recipe", "cooking", "baking", "flour", "sugar"],
        pii_distribution={"INGREDIENT": 10},
        language_distribution={"en": 1},
    )


@pytest.fixture
def empty_pii_type() -> KnownType:
    """Known type with empty PII distribution."""
    return KnownType(
        type_id="empty-pii-001",
        type_name="Empty PII Type",
        description="A type with no PII data",
        structural_signature="sha256:empty-pii-hash",
        tfidf_keywords=["empty", "test"],
        pii_distribution={},
    )


@pytest.fixture
def empty_pii_cluster() -> ClusterInfo:
    """Cluster with empty PII distribution."""
    return ClusterInfo(
        cluster_id="cluster-empty-pii",
        doc_ids=["doc-ep-001"],
        structural_bucket="sha256:empty-pii-hash",
        cluster_radius=0.0,
        representative_docs=["doc-ep-001"],
        tfidf_keywords=["empty", "test", "data"],
        pii_distribution={},
        language_distribution={"en": 1},
    )


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────


def _financial_matcher() -> KnownTypeMatcher:
    """Create a matcher pre-loaded with the financial_report_type."""
    from src.tier2.matching import KnownTypeMatcher
    from tests.test_tier2_matching import financial_report_type

    return KnownTypeMatcher(known_types=[financial_report_type()])


# ──────────────────────────────────────────────────────────────
# Test 1: Exact match → high score
# ──────────────────────────────────────────────────────────────

def test_exact_match_high_score(
    financial_report_type: KnownType,
    financial_cluster: ClusterInfo,
) -> None:
    """Cluster with matching structural_signature + overlapping keywords
    + similar PII distribution → score >= 0.8, method='known_match'."""
    matcher = KnownTypeMatcher(known_types=[financial_report_type])

    result = matcher.match(financial_cluster)

    assert isinstance(result, MatchResult)
    assert result.score >= 0.8, (
        f"Expected score >= 0.8 for matching cluster, got {result.score:.4f}"
    )
    assert result.method == "known_match", (
        f"Expected method='known_match', got {result.method}"
    )
    assert result.matched_type is not None
    assert result.matched_type.type_id == "fin-report-001"

    # Per-signal breakdown present
    assert "structure" in result.match_details
    assert "tfidf" in result.match_details
    assert "pii" in result.match_details
    assert "composite" in result.match_details

    # Structure should be an exact match
    assert result.match_details["structure"] == 1.0, (
        f"Expected structure signal=1.0, got {result.match_details['structure']}"
    )

    # TF-IDF overlap should be > 0.5 (4 common out of 11 total = ~0.36) — actually
    # intersection: report,financial,annual,revenue = 4
    # union = report,financial,annual,revenue,q4,earnings,quarter,fiscal,balance,income,statement = 11
    # Jaccard = 4/11 ≈ 0.364
    # Not very high, but with structure=0.5 and similar PII, composite should still be >= 0.8.
    # Let's verify: structure=0.5*1.0=0.5, pii cosine is very high (~0.99)*0.2≈0.198
    # 0.5 + 0.198 + 0.3*0.364 ≈ 0.5 + 0.198 + 0.109 = 0.807 >= 0.8 ✓
    assert result.match_details["composite"] >= 0.8


# ──────────────────────────────────────────────────────────────
# Test 2: No match → low score
# ──────────────────────────────────────────────────────────────

def test_no_match_low_score(
    financial_report_type: KnownType,
    unknown_cluster: ClusterInfo,
) -> None:
    """Completely different cluster → score < 0.5, method='unknown',
    matched_type=None."""
    matcher = KnownTypeMatcher(known_types=[financial_report_type])

    result = matcher.match(unknown_cluster)

    assert isinstance(result, MatchResult)
    assert result.score < 0.5, (
        f"Expected score < 0.5 for unrelated cluster, got {result.score:.4f}"
    )
    assert result.method == "unknown", (
        f"Expected method='unknown', got {result.method}"
    )
    assert result.matched_type is None, (
        "matched_type should be None when score < 0.5"
    )

    # Per-signal breakdown still present
    assert "structure" in result.match_details
    assert "tfidf" in result.match_details
    assert "pii" in result.match_details
    assert "composite" in result.match_details

    # Structure should not match
    assert result.match_details["structure"] == 0.0


# ──────────────────────────────────────────────────────────────
# Test 3: Partial match → mid score
# ──────────────────────────────────────────────────────────────

def test_partial_match_mid_score(
    financial_report_type: KnownType,
    partial_match_cluster: ClusterInfo,
) -> None:
    """Structural match but different keywords/PII → score in [0.5, 0.8),
    method='llm_confirm'."""
    matcher = KnownTypeMatcher(known_types=[financial_report_type])

    result = matcher.match(partial_match_cluster)

    assert isinstance(result, MatchResult)
    assert 0.5 <= result.score < 0.8, (
        f"Expected score in [0.5, 0.8), got {result.score:.4f}"
    )
    assert result.method == "llm_confirm", (
        f"Expected method='llm_confirm', got {result.method}"
    )
    assert result.matched_type is not None, (
        "matched_type should be set for llm_confirm (best candidate)"
    )
    assert result.matched_type.type_id == "fin-report-001"

    # Structure matches exactly → signal=1.0
    assert result.match_details["structure"] == 1.0

    # TF-IDF should be very low (different keywords)
    assert result.match_details["tfidf"] < 0.2


# ──────────────────────────────────────────────────────────────
# Test 4: Empty known types
# ──────────────────────────────────────────────────────────────

def test_empty_known_types(financial_cluster: ClusterInfo) -> None:
    """No types registered → score=0, method='unknown'."""
    matcher = KnownTypeMatcher(known_types=[])

    result = matcher.match(financial_cluster)

    assert isinstance(result, MatchResult)
    assert result.score == 0.0, (
        f"Expected score=0 with no known types, got {result.score}"
    )
    assert result.method == "unknown"
    assert result.matched_type is None

    # match_details should be an empty dict (no comparisons made)
    assert result.match_details == {}


# ──────────────────────────────────────────────────────────────
# Test 5: Register and match
# ──────────────────────────────────────────────────────────────

def test_register_and_match(financial_cluster: ClusterInfo) -> None:
    """Register a type after construction → match() uses it."""
    matcher = KnownTypeMatcher(known_types=[])

    assert matcher.type_count() == 0

    new_type = KnownType(
        type_id="registered-001",
        type_name="Registered Type",
        description="Registered after construction",
        structural_signature="sha256:abc123def456",  # matches financial_cluster
        tfidf_keywords=["report", "financial", "annual", "revenue", "earnings"],
        pii_distribution={"SSN": 45, "EMAIL": 35, "PHONE": 20},
    )

    matcher.register_type(new_type)

    assert matcher.type_count() == 1
    assert matcher.get_type("registered-001") is new_type
    assert matcher.get_type("nonexistent") is None

    # Now match should find it
    result = matcher.match(financial_cluster)
    assert result.matched_type is not None
    assert result.matched_type.type_id == "registered-001"
    assert result.score >= 0.8
    assert result.method == "known_match"


# ──────────────────────────────────────────────────────────────
# Test 6: Empty PII distributions → signal=0.0
# ──────────────────────────────────────────────────────────────

def test_pii_cosine_empty_distributions(
    empty_pii_type: KnownType,
    empty_pii_cluster: ClusterInfo,
) -> None:
    """Both distributions empty → PII signal=0.0, not NaN or 1.0."""
    matcher = KnownTypeMatcher(known_types=[empty_pii_type])

    result = matcher.match(empty_pii_cluster)

    assert "pii" in result.match_details, "match_details should include 'pii' key"
    pii_signal = result.match_details["pii"]
    assert pii_signal == 0.0, (
        f"Expected PII signal=0.0 for empty distributions, got {pii_signal}"
    )
    assert not math.isnan(pii_signal), "PII signal should not be NaN"

    # Structure matches → 0.5, TF-IDF has some overlap, composite should be >= 0.5
    assert result.match_details["structure"] == 1.0
    assert result.score >= 0.5  # structure alone = 0.5 + some tfidf overlap
