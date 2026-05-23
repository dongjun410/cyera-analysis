"""Tests for Tier 2: Tier2Classifier — cluster-level classification orchestrator (TDD).

Test contracts (write these BEFORE implementation):
  1. test_known_match_skips_llm — Cluster with match score >= 0.8 -> label from matcher,
     LLM.classify NOT called, NER.predict_batch NOT called.
  2. test_unknown_cluster_triggers_llm — No match -> LLM.classify IS called.
  3. test_mid_confidence_triggers_llm — Match score in [0.5, 0.8) -> LLM.classify IS called
     for confirmation.
  4. test_cold_start_no_clustering — cold_start_classify calls LLM per document (no clusters).
  5. test_empty_clusters — Empty cluster list -> empty result list, no crash.
  6. test_all_documents_get_labels — Every document in input gets a ClassificationResult.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.tier2.classifier import Tier2Classifier
from src.types import (
    ClassificationResult,
    ClusterInfo,
    Document,
    KnownType,
    MatchResult,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_documents() -> list[Document]:
    """Three sample documents for testing."""
    return [
        Document(
            doc_id="doc-001",
            text="Q4 2025 Financial Report — Revenue: $12.3M, Net Income: $2.1M. "
                 "Prepared by CFO Jane Smith. SSN: 123-45-6789.",
        ),
        Document(
            doc_id="doc-002",
            text="Annual Earnings Statement FY2025. Total revenue $45.6M. "
                 "Contact: john.doe@example.com, Phone: (555) 123-4567.",
        ),
        Document(
            doc_id="doc-003",
            text="Patient: John Doe, DOB: 1980-05-15. Diagnosis: Type 2 Diabetes. "
                 "Medication: Metformin 500mg. Doctor: Dr. Sarah Chen.",
        ),
    ]


@pytest.fixture
def financial_report_type() -> KnownType:
    """Known type for financial reports."""
    return KnownType(
        type_id="fin-report-001",
        type_name="Financial Report",
        description="Quarterly/annual financial statements",
        structural_signature="sha256:abc123def456",
        tfidf_keywords=["report", "financial", "annual", "revenue", "earnings"],
        pii_distribution={"SSN": 45, "EMAIL": 35, "PHONE": 20},
    )


@pytest.fixture
def medical_record_type() -> KnownType:
    """Known type for medical records."""
    return KnownType(
        type_id="med-record-001",
        type_name="Medical Record",
        description="Patient health records with HIPAA PII",
        structural_signature="sha256:xyz789ghi012",
        tfidf_keywords=["patient", "diagnosis", "treatment", "medication"],
        pii_distribution={"PATIENT_ID": 80, "DOB": 40, "PHONE": 10},
    )


@pytest.fixture
def financial_cluster() -> ClusterInfo:
    """Cluster matching financial reports."""
    return ClusterInfo(
        cluster_id="cluster-001",
        doc_ids=["doc-001", "doc-002"],
        structural_bucket="sha256:abc123def456",
        cluster_radius=0.15,
        representative_docs=["doc-001"],
        tfidf_keywords=["report", "financial", "annual", "revenue", "q4"],
        pii_distribution={"SSN": 50, "EMAIL": 30, "PHONE": 20},
        language_distribution={"en": 2},
    )


@pytest.fixture
def unknown_cluster() -> ClusterInfo:
    """Cluster that does not match any known type."""
    return ClusterInfo(
        cluster_id="cluster-999",
        doc_ids=["doc-003"],
        structural_bucket="sha256:completely-different-hash",
        cluster_radius=0.25,
        representative_docs=["doc-003"],
        tfidf_keywords=["gibberish", "unknown", "random"],
        pii_distribution={"CUSTOM_TYPE": 5},
        language_distribution={"en": 1},
    )


@pytest.fixture
def mid_confidence_cluster() -> ClusterInfo:
    """Cluster with structural match but different content -> mid confidence."""
    return ClusterInfo(
        cluster_id="cluster-050",
        doc_ids=["doc-050"],
        structural_bucket="sha256:abc123def456",  # matches financial
        cluster_radius=0.0,
        representative_docs=["doc-050"],
        tfidf_keywords=["recipe", "cooking", "baking", "flour", "sugar"],
        pii_distribution={"INGREDIENT": 10},
        language_distribution={"en": 1},
    )


# ──────────────────────────────────────────────────────────────
# Helper — builds a mock propagator that returns given results
# ──────────────────────────────────────────────────────────────


def _mock_propagate_results(label: str, confidence: float, doc_ids: list[str]) -> Mock:
    """Create a mock propagator whose propagate() returns results for each doc_id."""
    results = [
        ClassificationResult(
            doc_id=doc_id,
            label=label,
            confidence=confidence,
            method="propagated",
            is_new_type=False,
            needs_manual_review=False,
            rationale=f"Propagated from mock for {doc_id}",
        )
        for doc_id in doc_ids
    ]
    propagator = Mock()
    propagator.propagate.return_value = (results, False)
    return propagator


# ──────────────────────────────────────────────────────────────
# Test 1: Known match skips LLM and NER
# ──────────────────────────────────────────────────────────────


def test_known_match_skips_llm(
    financial_cluster: ClusterInfo,
    financial_report_type: KnownType,
    sample_documents: list[Document],
) -> None:
    """Cluster with match score >= 0.8 -> label from matcher.
    LLM.classify and NER.predict_batch must NOT be called."""
    # ── Setup mocks ──────────────────────────────────────────
    matcher = Mock()
    matcher.match.return_value = MatchResult(
        matched_type=financial_report_type,
        score=0.85,
        method="known_match",
        match_details={"structure": 1.0, "tfidf": 0.5, "pii": 0.9, "composite": 0.85},
    )
    # Attach _types so classifier can extract known type names
    matcher._types = {"fin-report-001": financial_report_type}

    ner = Mock()

    llm = Mock()

    propagator = _mock_propagate_results(
        "Financial Report", 0.85, financial_cluster.doc_ids
    )

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    # ── Execute ──────────────────────────────────────────────
    results = classifier.classify_clusters(
        [financial_cluster], sample_documents
    )

    # ── Assertions ───────────────────────────────────────────
    matcher.match.assert_called_once_with(financial_cluster)

    # NER must NOT be called — skipped for known_match
    ner.predict_batch.assert_not_called()

    # LLM must NOT be called — skipped for known_match
    llm.classify.assert_not_called()

    # Propagator must be called with the matched label
    propagator.propagate.assert_called_once()
    call_args = propagator.propagate.call_args
    assert call_args[0][1] == "Financial Report"
    assert call_args[0][2] == 0.85

    # All documents in the cluster get results
    assert len(results) == len(financial_cluster.doc_ids)
    assert all(r.label == "Financial Report" for r in results)


# ──────────────────────────────────────────────────────────────
# Test 2: Unknown cluster triggers LLM
# ──────────────────────────────────────────────────────────────


def test_unknown_cluster_triggers_llm(
    unknown_cluster: ClusterInfo,
    financial_report_type: KnownType,
    medical_record_type: KnownType,
    sample_documents: list[Document],
) -> None:
    """No match (score < 0.5) -> LLM.classify IS called."""
    # ── Setup mocks ──────────────────────────────────────────
    matcher = Mock()
    matcher.match.return_value = MatchResult(
        matched_type=None,
        score=0.1,
        method="unknown",
        match_details={"structure": 0.0, "tfidf": 0.0, "pii": 0.0, "composite": 0.1},
    )
    matcher._types = {
        "fin-report-001": financial_report_type,
        "med-record-001": medical_record_type,
    }

    ner = Mock()
    ner.predict_batch.return_value = [
        [Mock(entity_type="CUSTOM_TYPE")]  # one entity found
    ]

    llm = Mock()
    llm.classify.return_value = {
        "label": "Medical Record",
        "is_new_type": False,
        "confidence": 0.88,
        "rationale": "Contains patient diagnosis and medication info.",
        "suggested_rules": "",
    }

    propagator = _mock_propagate_results(
        "Medical Record", 0.88, unknown_cluster.doc_ids
    )

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    # ── Execute ──────────────────────────────────────────────
    results = classifier.classify_clusters(
        [unknown_cluster], sample_documents
    )

    # ── Assertions ───────────────────────────────────────────
    matcher.match.assert_called_once_with(unknown_cluster)

    # NER should be called on representative docs
    ner.predict_batch.assert_called_once()
    ner_texts = ner.predict_batch.call_args[0][0]
    assert len(ner_texts) >= 1  # at least one rep doc

    # LLM MUST be called
    llm.classify.assert_called_once()
    llm_args = llm.classify.call_args
    document_text = llm_args[0][0]  # first positional arg: document_text
    known_types = llm_args[0][1]  # second positional arg: known_types
    assert len(document_text) > 0
    assert "Financial Report" in known_types
    assert "Medical Record" in known_types

    # Propagator must be called
    propagator.propagate.assert_called_once()

    # Results should hold the LLM-determined label
    assert len(results) == len(unknown_cluster.doc_ids)
    assert all(r.label == "Medical Record" for r in results)


# ──────────────────────────────────────────────────────────────
# Test 3: Mid-confidence triggers LLM for confirmation
# ──────────────────────────────────────────────────────────────


def test_mid_confidence_triggers_llm(
    mid_confidence_cluster: ClusterInfo,
    financial_report_type: KnownType,
) -> None:
    """Match score in [0.5, 0.8) -> LLM.classify IS called for confirmation."""
    # Documents that match the cluster's doc_ids
    docs = [
        Document(
            doc_id="doc-050",
            text="Flour, sugar, baking powder, vanilla extract. Mix well and bake at 350F.",
        ),
    ]

    # ── Setup mocks ──────────────────────────────────────────
    matcher = Mock()
    matcher.match.return_value = MatchResult(
        matched_type=financial_report_type,
        score=0.62,
        method="llm_confirm",
        match_details={"structure": 1.0, "tfidf": 0.05, "pii": 0.0, "composite": 0.62},
    )
    matcher._types = {"fin-report-001": financial_report_type}

    ner = Mock()
    ner.predict_batch.return_value = [[]]  # no entities found

    llm = Mock()
    llm.classify.return_value = {
        "label": "Recipe Collection",
        "is_new_type": True,
        "confidence": 0.75,
        "rationale": "Contains cooking ingredients, not a financial report.",
        "suggested_rules": r"\b(flour|sugar|recipe)\b",
    }

    propagator = _mock_propagate_results(
        "Recipe Collection", 0.75, mid_confidence_cluster.doc_ids
    )

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    # ── Execute ──────────────────────────────────────────────
    results = classifier.classify_clusters(
        [mid_confidence_cluster], docs
    )

    # ── Assertions ───────────────────────────────────────────
    matcher.match.assert_called_once_with(mid_confidence_cluster)

    # NER should run for non-known_match cases
    ner.predict_batch.assert_called_once()

    # LLM MUST be called — this is llm_confirm path
    llm.classify.assert_called_once()

    # Propagator must be called
    propagator.propagate.assert_called_once()

    # Results match the LLM decision (overrides the mid-confidence candidate)
    assert len(results) == len(mid_confidence_cluster.doc_ids)
    assert all(r.label == "Recipe Collection" for r in results)


# ──────────────────────────────────────────────────────────────
# Test 4: Cold start — LLM per document, no clusters
# ──────────────────────────────────────────────────────────────


def test_cold_start_no_clustering(
    sample_documents: list[Document],
) -> None:
    """cold_start_classify calls LLM.classify once per document, no clusters needed."""
    matcher = Mock()
    ner = Mock()

    llm = Mock()
    llm.classify.side_effect = [
        {
            "label": "Financial Report",
            "is_new_type": False,
            "confidence": 0.82,
            "rationale": "Contains financial figures and SSN.",
            "suggested_rules": "",
        },
        {
            "label": "Financial Report",
            "is_new_type": False,
            "confidence": 0.79,
            "rationale": "Annual earnings data.",
            "suggested_rules": "",
        },
        {
            "label": "Medical Record",
            "is_new_type": True,
            "confidence": 0.91,
            "rationale": "Patient diagnosis and medication.",
            "suggested_rules": r"\bDOB\b",
        },
    ]

    propagator = Mock()  # Not used in cold start

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    # ── Execute ──────────────────────────────────────────────
    results = classifier.cold_start_classify(sample_documents)

    # ── Assertions ───────────────────────────────────────────
    assert len(results) == len(sample_documents)
    assert llm.classify.call_count == len(sample_documents)

    # Each result maps to its document
    assert results[0].doc_id == "doc-001"
    assert results[0].label == "Financial Report"
    assert results[0].confidence == 0.82
    assert results[0].method == "llm_tier2"

    assert results[1].doc_id == "doc-002"
    assert results[1].label == "Financial Report"

    assert results[2].doc_id == "doc-003"
    assert results[2].label == "Medical Record"
    assert results[2].is_new_type is True

    # Matcher not used
    matcher.match.assert_not_called()
    # Propagator not used
    propagator.propagate.assert_not_called()

    # Each LLM call passes empty known_types (cold start = zero-shot)
    for i, call in enumerate(llm.classify.call_args_list):
        known_types = call[0][1]
        assert known_types == [], (
            f"Cold start doc {i} should have empty known_types, got {known_types}"
        )

    # LLM classify receives the document text
    for i, call in enumerate(llm.classify.call_args_list):
        doc_text = call[0][0]
        assert doc_text == sample_documents[i].text


# ──────────────────────────────────────────────────────────────
# Test 5: Empty clusters -> empty results, no crash
# ──────────────────────────────────────────────────────────────


def test_empty_clusters(
    sample_documents: list[Document],
) -> None:
    """Empty cluster list -> empty result list, no crashes."""
    matcher = Mock()
    ner = Mock()
    llm = Mock()
    propagator = Mock()

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    # ── Execute ──────────────────────────────────────────────
    results = classifier.classify_clusters([], sample_documents)

    # ── Assertions ───────────────────────────────────────────
    assert isinstance(results, list)
    assert len(results) == 0

    # None of the dependencies should be called
    matcher.match.assert_not_called()
    ner.predict_batch.assert_not_called()
    llm.classify.assert_not_called()
    propagator.propagate.assert_not_called()


# ──────────────────────────────────────────────────────────────
# Test 6: All documents get labels
# ──────────────────────────────────────────────────────────────


def test_all_documents_get_labels(
    financial_cluster: ClusterInfo,
    unknown_cluster: ClusterInfo,
    financial_report_type: KnownType,
    medical_record_type: KnownType,
    sample_documents: list[Document],
) -> None:
    """Multiple clusters covering all 3 documents -> every document gets a result."""
    # ── Setup mocks ──────────────────────────────────────────
    matcher = Mock()

    # First cluster is a known match; second is unknown
    matcher.match.side_effect = [
        MatchResult(
            matched_type=financial_report_type,
            score=0.88,
            method="known_match",
            match_details={"composite": 0.88},
        ),
        MatchResult(
            matched_type=None,
            score=0.12,
            method="unknown",
            match_details={"composite": 0.12},
        ),
    ]
    matcher._types = {
        "fin-report-001": financial_report_type,
        "med-record-001": medical_record_type,
    }

    ner = Mock()
    ner.predict_batch.return_value = [[]]

    llm = Mock()
    llm.classify.return_value = {
        "label": "Medical Record",
        "is_new_type": False,
        "confidence": 0.90,
        "rationale": "Patient health information.",
        "suggested_rules": "",
    }

    # Propagator for first cluster (known match)
    prop_results_1 = [
        ClassificationResult(
            doc_id="doc-001",
            label="Financial Report",
            confidence=0.88,
            method="propagated",
        ),
        ClassificationResult(
            doc_id="doc-002",
            label="Financial Report",
            confidence=0.88,
            method="propagated",
        ),
    ]
    # Propagator for second cluster (unknown -> LLM)
    prop_results_2 = [
        ClassificationResult(
            doc_id="doc-003",
            label="Medical Record",
            confidence=0.90,
            method="propagated",
        ),
    ]
    propagator = Mock()
    propagator.propagate.side_effect = [
        (prop_results_1, False),
        (prop_results_2, False),
    ]

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    # ── Execute ──────────────────────────────────────────────
    clusters = [financial_cluster, unknown_cluster]
    results = classifier.classify_clusters(clusters, sample_documents)

    # ── Assertions ───────────────────────────────────────────
    # Exactly 3 results (one per document)
    assert len(results) == 3

    # All doc_ids covered
    result_doc_ids = {r.doc_id for r in results}
    assert result_doc_ids == {"doc-001", "doc-002", "doc-003"}

    # Each result is a ClassificationResult
    for r in results:
        assert isinstance(r, ClassificationResult)

    # doc-001 and doc-002 get "Financial Report"
    doc001 = next(r for r in results if r.doc_id == "doc-001")
    doc002 = next(r for r in results if r.doc_id == "doc-002")
    assert doc001.label == "Financial Report"
    assert doc002.label == "Financial Report"

    # doc-003 gets "Medical Record" (from LLM)
    doc003 = next(r for r in results if r.doc_id == "doc-003")
    assert doc003.label == "Medical Record"

    # Propagator called twice (once per cluster)
    assert propagator.propagate.call_count == 2


# ──────────────────────────────────────────────────────────────
# Additional: Test NER runs on representative docs for non-match
# ──────────────────────────────────────────────────────────────


def test_ner_runs_on_representative_docs(
    unknown_cluster: ClusterInfo,
    financial_report_type: KnownType,
    medical_record_type: KnownType,
    sample_documents: list[Document],
) -> None:
    """For unknown clusters, NER should run on representative docs (max 3)."""
    matcher = Mock()
    matcher.match.return_value = MatchResult(
        matched_type=None,
        score=0.1,
        method="unknown",
        match_details={"composite": 0.1},
    )
    matcher._types = {
        "fin-report-001": financial_report_type,
        "med-record-001": medical_record_type,
    }

    ner = Mock()
    ner.predict_batch.return_value = []

    llm = Mock()
    llm.classify.return_value = {
        "label": "Medical Record",
        "is_new_type": False,
        "confidence": 0.88,
        "rationale": "Medical content.",
        "suggested_rules": "",
    }

    propagator = _mock_propagate_results(
        "Medical Record", 0.88, unknown_cluster.doc_ids
    )

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    results = classifier.classify_clusters([unknown_cluster], sample_documents)

    # NER should be called with texts from representative_docs
    ner.predict_batch.assert_called_once()
    batch_texts = ner.predict_batch.call_args[0][0]
    assert isinstance(batch_texts, list)
    # Should be at least 1, at most 3 representative doc texts
    assert 1 <= len(batch_texts) <= 3

    # Known results present
    assert len(results) > 0


# ──────────────────────────────────────────────────────────────
# Additional: Constructor smoke test — all dependencies stored
# ──────────────────────────────────────────────────────────────


def test_constructor_stores_dependencies() -> None:
    """Tier2Classifier stores all injected dependencies."""
    matcher = Mock()
    ner = Mock()
    llm = Mock()
    propagator = Mock()

    classifier = Tier2Classifier(matcher, ner, llm, propagator)

    assert classifier._matcher is matcher
    assert classifier._ner is ner
    assert classifier._llm is llm
    assert classifier._propagator is propagator


def test_constructor_with_config() -> None:
    """Tier2Classifier accepts optional config dict."""
    matcher = Mock()
    ner = Mock()
    llm = Mock()
    propagator = Mock()
    config = {"ner_representative_limit": 5, "llm_enabled": True}

    classifier = Tier2Classifier(matcher, ner, llm, propagator, config=config)

    assert classifier._config == config
    assert classifier._config["ner_representative_limit"] == 5
