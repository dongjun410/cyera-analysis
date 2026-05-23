"""Tests for Tier 3: QualityGate — final precision defense line (TDD).

Test contracts (write these BEFORE implementation):
  1. test_triggers_high_sensitivity     — SSN label → trigger
  2. test_triggers_low_confidence       — confidence 0.55 → trigger
  3. test_no_trigger_normal_doc         — Normal doc, high confidence, non-sensitive → no trigger
  4. test_triggers_semantic_outlier     — distance > 2σ → trigger
  5. test_verify_returns_classification — verify() returns ClassificationResult with method="llm_tier3"
  6. test_verify_low_confidence_flags_review — LLM returns confidence < 0.6 → needs_manual_review=True
  7. test_batch_verify_all_docs_processed — verify_batch returns same number of results as inputs
  8. test_ner_rule_contradiction        — different types from NER vs Tier 0 → trigger
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.tier3.quality_gate import QualityGate
from src.types import ClassificationResult, ClusterInfo, Document, PIIFeature, PIIFeatureVector


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def normal_doc() -> Document:
    """A normal financial document — not sensitive, high confidence."""
    return Document(
        doc_id="doc-001",
        text="Q4 2025 Financial Report — Revenue: $12.3M, Net Income: $2.1M.",
        embedding=np.array([0.1, 0.2, 0.3, 0.4]),
    )


@pytest.fixture
def sensitive_doc() -> Document:
    """A document containing SSN — flagged as high sensitivity."""
    return Document(
        doc_id="doc-002",
        text="Employee record. Name: John Doe. SSN: 123-45-6789.",
        embedding=np.array([0.1, 0.2, 0.3, 0.4]),
    )


@pytest.fixture
def normal_cluster() -> ClusterInfo:
    """A normal cluster with centroid near the normal document embedding."""
    return ClusterInfo(
        cluster_id="cluster-001",
        doc_ids=["doc-001"],
        structural_bucket="sha256:abc123",
        cluster_radius=0.30,
        representative_docs=["doc-001"],
        tfidf_keywords=["report", "financial", "revenue"],
        pii_distribution={"SSN": 10, "EMAIL": 5},
        language_distribution={"en": 1},
        centroid_embedding=np.array([0.1, 0.25, 0.35, 0.45]),
        label="Financial Report",
        label_confidence=0.88,
    )


@pytest.fixture
def mock_llm() -> Mock:
    """Mock MistralClient returning a confident verification result."""
    llm = Mock()
    llm.verify.return_value = {
        "label": "Financial Report",
        "confidence": 0.82,
        "reasoning_chain": "Document contains revenue figures and financial terminology.",
        "needs_manual_review": False,
    }
    return llm


@pytest.fixture
def quality_gate(mock_llm: Mock) -> QualityGate:
    """QualityGate with defaults and mock LLM."""
    return QualityGate(mock_llm)


# ──────────────────────────────────────────────────────────────
# Test 1: High sensitivity label triggers the gate
# ──────────────────────────────────────────────────────────────


def test_triggers_high_sensitivity(
    quality_gate: QualityGate,
    sensitive_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """SSN is in high_sensitivity_types → should_trigger returns True."""
    classification = ClassificationResult(
        doc_id="doc-002",
        label="SSN",
        confidence=0.92,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        sensitive_doc, normal_cluster, classification
    )

    assert result is True, (
        f"SSN label should trigger quality gate, got {result}"
    )


def test_triggers_medical_label(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """MEDICAL is in high_sensitivity_types → should_trigger returns True."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="MEDICAL",
        confidence=0.91,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is True


def test_triggers_credit_card_label(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """CREDIT_CARD is in default high_sensitivity_types → trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="CREDIT_CARD",
        confidence=0.94,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is True


# ──────────────────────────────────────────────────────────────
# Test 2: Low LLM confidence triggers the gate
# ──────────────────────────────────────────────────────────────


def test_triggers_low_confidence(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Confidence 0.55 falls in [0.5, 0.8) → trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.55,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is True, (
        f"Low confidence (0.55) should trigger, got {result}"
    )


def test_triggers_confidence_at_lower_bound(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Confidence 0.5 (exactly at lower bound) → trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.5,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is True, (
        f"Confidence 0.5 (at lower bound) should trigger, got {result}"
    )


def test_triggers_confidence_at_79(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Confidence 0.79 (just below upper bound) → trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.79,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is True, (
        f"Confidence 0.79 (in range) should trigger, got {result}"
    )


# ──────────────────────────────────────────────────────────────
# Test 3: Normal document — no trigger
# ──────────────────────────────────────────────────────────────


def test_no_trigger_normal_doc(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Normal doc, high confidence, non-sensitive, not an outlier → no trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
        needs_manual_review=False,
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is False, (
        f"Normal document should NOT trigger, got {result}"
    )


def test_no_trigger_high_confidence_non_sensitive(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Confidence 0.88 (>= 0.8), non-sensitive label → no trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Invoice",
        confidence=0.88,
        method="llm_tier2",
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification
    )

    assert result is False


# ──────────────────────────────────────────────────────────────
# Test 4: Semantic outlier triggers the gate
# ──────────────────────────────────────────────────────────────


def test_triggers_semantic_outlier(
    mock_llm: Mock,
    normal_cluster: ClusterInfo,
) -> None:
    """Document embedding far from centroid → cosine distance > 2σ → trigger."""
    gate = QualityGate(mock_llm)

    # Document with embedding orthogonal to centroid
    outlier_doc = Document(
        doc_id="doc-outlier",
        text="Cooking recipe: mix flour, sugar, eggs, and vanilla extract.",
        embedding=np.array([-0.5, -0.6, -0.7, -0.8]),
    )

    # Cluster centroid is around [0.1, 0.25, 0.35, 0.45]
    # The cosine distance between these vectors will be large
    cluster = ClusterInfo(
        cluster_id="cluster-001",
        doc_ids=["doc-outlier"],
        structural_bucket="sha256:abc123",
        cluster_radius=0.25,  # 2σ = 0.5
        representative_docs=["doc-outlier"],
        tfidf_keywords=["report", "financial"],
        pii_distribution={},
        language_distribution={"en": 1},
        centroid_embedding=np.array([0.9, 0.85, 0.8, 0.75]),
        label="Financial Report",
    )

    classification = ClassificationResult(
        doc_id="doc-outlier",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
    )

    result = gate.should_trigger(
        outlier_doc, cluster, classification
    )

    assert result is True, (
        f"Semantic outlier should trigger, got {result}"
    )


def test_no_trigger_near_centroid(
    mock_llm: Mock,
) -> None:
    """Document embedding close to centroid → cosine distance < 2σ → no trigger."""
    gate = QualityGate(mock_llm)

    doc = Document(
        doc_id="doc-near",
        text="Financial report Q4 2025.",
        embedding=np.array([0.5, 0.5, 0.5, 0.5]),
    )

    cluster = ClusterInfo(
        cluster_id="cluster-001",
        doc_ids=["doc-near"],
        structural_bucket="sha256:abc",
        cluster_radius=0.50,
        representative_docs=["doc-near"],
        tfidf_keywords=["report"],
        pii_distribution={},
        language_distribution={"en": 1},
        centroid_embedding=np.array([0.5, 0.5, 0.5, 0.5]),
        label="Financial Report",
    )

    classification = ClassificationResult(
        doc_id="doc-near",
        label="Financial Report",
        confidence=0.95,
        method="llm_tier2",
    )

    result = gate.should_trigger(doc, cluster, classification)

    assert result is False, (
        f"Near-centroid document should NOT trigger semantic outlier, got {result}"
    )


# ──────────────────────────────────────────────────────────────
# Test 5: verify() returns ClassificationResult
# ──────────────────────────────────────────────────────────────


def test_verify_returns_classification(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """verify() returns a ClassificationResult with method='llm_tier3'."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.55,
        method="llm_tier2",
    )

    result = quality_gate.verify(normal_doc, normal_cluster, classification)

    assert isinstance(result, ClassificationResult), (
        f"Expected ClassificationResult, got {type(result)}"
    )
    assert result.doc_id == "doc-001"
    assert result.method == "llm_tier3", (
        f"Expected method='llm_tier3', got '{result.method}'"
    )
    assert result.label == "Financial Report"
    assert result.confidence == 0.82
    assert result.needs_manual_review is False


# ──────────────────────────────────────────────────────────────
# Test 6: Low LLM confidence flags manual review
# ──────────────────────────────────────────────────────────────


def test_verify_low_confidence_flags_review(
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """LLM returns confidence < 0.6 → needs_manual_review=True."""
    llm = Mock()
    llm.verify.return_value = {
        "label": "Financial Report",
        "confidence": 0.52,  # < 0.6 → auto-flag
        "reasoning_chain": "Uncertain — document has mixed characteristics.",
        "needs_manual_review": False,  # LLM didn't flag it, but our code should
    }

    gate = QualityGate(llm)

    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.55,
        method="llm_tier2",
    )

    result = gate.verify(normal_doc, normal_cluster, classification)

    assert result.needs_manual_review is True, (
        f"Confidence 0.52 should set needs_manual_review=True, got {result.needs_manual_review}"
    )


def test_verify_llm_flags_review(
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """LLM itself flags needs_manual_review=True → propagated even if confidence >= 0.6."""
    llm = Mock()
    llm.verify.return_value = {
        "label": "Financial Report",
        "confidence": 0.72,
        "reasoning_chain": "Some ambiguity detected.",
        "needs_manual_review": True,
    }

    gate = QualityGate(llm)

    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.60,
        method="llm_tier2",
    )

    result = gate.verify(normal_doc, normal_cluster, classification)

    assert result.needs_manual_review is True, (
        f"LLM flagged review → needs_manual_review should be True, got {result.needs_manual_review}"
    )


# ──────────────────────────────────────────────────────────────
# Test 7: Batch verify processes all documents
# ──────────────────────────────────────────────────────────────


def test_batch_verify_all_docs_processed(
    mock_llm: Mock,
    normal_cluster: ClusterInfo,
) -> None:
    """verify_batch returns same number of ClassificationResults as inputs."""
    gate = QualityGate(mock_llm)

    docs = [
        Document(
            doc_id=f"doc-{i:03d}",
            text=f"Document {i} content.",
        )
        for i in range(5)
    ]

    classifications = [
        ClassificationResult(
            doc_id=f"doc-{i:03d}",
            label="Financial Report",
            confidence=0.55 + i * 0.02,
            method="llm_tier2",
        )
        for i in range(5)
    ]

    results = gate.verify_batch(docs, normal_cluster, classifications)

    assert len(results) == len(docs), (
        f"Expected {len(docs)} results, got {len(results)}"
    )
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all(r.method == "llm_tier3" for r in results)

    # LLM.verify should be called once per document
    assert mock_llm.verify.call_count == len(docs)


def test_batch_verify_empty_list(
    quality_gate: QualityGate,
    normal_cluster: ClusterInfo,
) -> None:
    """verify_batch with empty lists returns empty list, no crash."""
    results = quality_gate.verify_batch([], normal_cluster, [])

    assert isinstance(results, list)
    assert len(results) == 0


# ──────────────────────────────────────────────────────────────
# Test 8: NER-rule contradiction
# ──────────────────────────────────────────────────────────────


def test_ner_rule_contradiction(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """NER finds types that Tier 0 missed AND vice versa → contradiction → trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
    )

    # NER detected: SSN, EMAIL
    ner_results = [
        PIIFeature(
            entity_type="SSN",
            span=(30, 41),
            confidence=0.85,
            context_flag="clean",
        ),
        PIIFeature(
            entity_type="EMAIL",
            span=(60, 75),
            confidence=0.78,
            context_flag="clean",
        ),
    ]

    # Tier 0 detected: SSN, CREDIT_CARD (different from NER's EMAIL)
    tier0_features = {"SSN": 1, "CREDIT_CARD": 2}

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification,
        ner_results=ner_results,
        tier0_features=tier0_features,
    )

    # NER has {SSN, EMAIL}, Tier0 has {SSN, CREDIT_CARD}
    # diff_ner = {EMAIL} (non-empty), diff_tier0 = {CREDIT_CARD} (non-empty)
    # → contradiction → trigger
    assert result is True, (
        f"NER-rule contradiction should trigger, got {result}"
    )


def test_no_ner_contradiction_when_same_types(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """NER and Tier 0 find the same types → no contradiction."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
    )

    ner_results = [
        PIIFeature(
            entity_type="SSN",
            span=(30, 41),
            confidence=0.85,
            context_flag="clean",
        ),
    ]

    tier0_features = {"SSN": 1}

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification,
        ner_results=ner_results,
        tier0_features=tier0_features,
    )

    # Same types → no contradiction → no trigger
    assert result is False


def test_no_ner_contradiction_when_ner_subset(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """NER types are subset of Tier 0 → no contradiction (only diff_ner empty)."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
    )

    ner_results = [
        PIIFeature(
            entity_type="SSN",
            span=(30, 41),
            confidence=0.85,
            context_flag="clean",
        ),
    ]

    tier0_features = {"SSN": 1, "EMAIL": 2}

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification,
        ner_results=ner_results,
        tier0_features=tier0_features,
    )

    assert result is False


def test_no_ner_contradiction_without_ner_or_tier0(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """When ner_results or tier0_features is None/empty → no contradiction."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
    )

    # Neither provided → no trigger
    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification,
    )

    assert result is False


# ──────────────────────────────────────────────────────────────
# Test: needs_manual_review trigger
# ──────────────────────────────────────────────────────────────


def test_triggers_needs_manual_review(
    quality_gate: QualityGate,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """classification.needs_manual_review == True → trigger."""
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
        needs_manual_review=True,
    )

    result = quality_gate.should_trigger(
        normal_doc, normal_cluster, classification,
    )

    assert result is True, (
        f"needs_manual_review=True should trigger, got {result}"
    )


# ──────────────────────────────────────────────────────────────
# Test: Custom config overrides defaults
# ──────────────────────────────────────────────────────────────


def test_custom_high_sensitivity_types(
    mock_llm: Mock,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Custom high_sensitivity_types in config replaces defaults."""
    config = {
        "high_sensitivity_types": ["TOP_SECRET", "CLASSIFIED"],
        "semantic_distance_sigma": 3.0,
    }
    gate = QualityGate(mock_llm, config=config)

    # SSN is NOT in custom list → no trigger from high sensitivity
    classification = ClassificationResult(
        doc_id="doc-001",
        label="SSN",
        confidence=0.95,
        method="llm_tier2",
    )

    result = gate.should_trigger(normal_doc, normal_cluster, classification)
    assert result is False, "SSN should NOT trigger with custom sensitivity list"

    # TOP_SECRET IS in custom list → trigger
    classification2 = ClassificationResult(
        doc_id="doc-001",
        label="TOP_SECRET",
        confidence=0.95,
        method="llm_tier2",
    )

    result2 = gate.should_trigger(normal_doc, normal_cluster, classification2)
    assert result2 is True, "TOP_SECRET SHOULD trigger with custom sensitivity list"


def test_custom_confidence_range(
    mock_llm: Mock,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """Custom llm_confidence_range in config changes trigger threshold."""
    config = {"llm_confidence_range": [0.3, 0.6]}
    gate = QualityGate(mock_llm, config=config)

    # Confidence 0.55 is in [0.3, 0.6) → trigger
    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.55,
        method="llm_tier2",
    )

    result = gate.should_trigger(normal_doc, normal_cluster, classification)
    assert result is True, "0.55 in [0.3, 0.6) should trigger"

    # Confidence 0.65 is >= 0.6 → no trigger from confidence range
    classification2 = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.65,
        method="llm_tier2",
    )

    result2 = gate.should_trigger(normal_doc, normal_cluster, classification2)
    assert result2 is False, "0.65 >= 0.6 should NOT trigger"


def test_verify_passes_cluster_context(
    mock_llm: Mock,
    normal_doc: Document,
    normal_cluster: ClusterInfo,
) -> None:
    """verify() passes cluster context to LLM including label, size, keywords."""
    gate = QualityGate(mock_llm)

    classification = ClassificationResult(
        doc_id="doc-001",
        label="Financial Report",
        confidence=0.55,
        method="llm_tier2",
    )

    gate.verify(normal_doc, normal_cluster, classification)

    # Check that llm.verify was called with cluster_context
    call_args = mock_llm.verify.call_args
    assert call_args is not None

    # Positional args: document_text, current_label, cluster_context
    doc_text = call_args[0][0]
    current_label = call_args[0][1]
    cluster_context = call_args[0][2]

    assert doc_text == normal_doc.text
    assert current_label == "Financial Report"
    assert isinstance(cluster_context, dict)
    assert cluster_context.get("cluster_label") == "Financial Report"
    assert cluster_context.get("cluster_size") == len(normal_cluster.doc_ids)
    assert "keywords" in cluster_context
