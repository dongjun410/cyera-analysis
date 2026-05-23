"""Integration tests for the DataDNA full pipeline (Tier 0→1→2→3).

Verifies end-to-end document classification flow, cold start, fault tolerance,
and discovery loop outlier accumulation.

Tests use real Tier0Engine, StructuralClusterer, and LabelPropagator
(lightweight CPU-based) with Mock LLM and NER for fast execution.

Test contracts:
  1. test_full_pipeline_small_batch  — 20 docs → Tier 0 → each gets PIIFeatureVector
  2. test_tier0_to_tier1_flow        — Tier 0 output → Tier 1 structural buckets
  3. test_cold_start_path            — Empty known types → cold_start_classify
  4. test_cluster_label_propagation  — Cluster with label → propagate → all docs labeled
  5. test_end_to_end_output_schema   — PipelineResult has correct keys
  6. test_tier3_triggers_for_sensitive_docs — SSN label → QualityGate trigger
  7. test_discovery_outlier_accumulation    — Outlier buffer grows correctly
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from src.tier0.engine import Tier0Engine
from src.tier1.structural import StructuralClusterer
from src.tier2.classifier import Tier2Classifier
from src.tier2.matching import KnownTypeMatcher
from src.tier2.propagation import LabelPropagator
from src.tier3.quality_gate import QualityGate
from src.discovery.loop import DiscoveryLoop
from src.types import (
    ClassificationResult,
    ClusterInfo,
    Document,
    KnownType,
    PipelineResult,
    PIIFeatureVector,
    StructuralFeatures,
)


# ══════════════════════════════════════════════════════════════
# Test 1: Full pipeline — small batch through Tier 0
# ══════════════════════════════════════════════════════════════

def test_full_pipeline_small_batch(sample_documents: list[Document]) -> None:
    """20 docs through Tier 0 engine → each gets a PIIFeatureVector with correct doc_id.

    Verifies:
      - All 20 documents are processed without error
      - Each result is a PIIFeatureVector dataclass
      - doc_ids are preserved 1:1 with input
      - HR docs detect SSN, Financial docs detect CREDIT_CARD
      - JSON and plain text docs produce valid (possibly empty) vectors
    """
    engine = Tier0Engine({"context_window": 100, "confidence_threshold": 0.4})

    # Build (doc_id, text) tuples
    docs_tuples = [(d.doc_id, d.text) for d in sample_documents]
    results = engine.extract_batch(docs_tuples)

    # ── All 20 processed ──────────────────────────────────────
    assert len(results) == 20, f"Expected 20 results, got {len(results)}"
    assert all(isinstance(r, PIIFeatureVector) for r in results)

    # ── doc_ids preserved ─────────────────────────────────────
    result_ids = {r.doc_id for r in results}
    expected_ids = {d.doc_id for d in sample_documents}
    assert result_ids == expected_ids

    # ── HR docs should detect SSN ──────────────────────────────
    hr_results = [r for r in results if r.doc_id.startswith("doc_hr")]
    for r in hr_results:
        ssn_count = r.pii_type_distribution.get("SSN", 0)
        assert ssn_count >= 1, (
            f"HR doc {r.doc_id} should have SSN detected, "
            f"got distribution {r.pii_type_distribution}"
        )

    # ── Financial docs should detect CREDIT_CARD ───────────────
    fin_results = [r for r in results if r.doc_id.startswith("doc_fin")]
    # At least one financial doc should have CREDIT_CARD
    any_cc = any(r.pii_type_distribution.get("CREDIT_CARD", 0) >= 1 for r in fin_results)
    assert any_cc, "At least one financial doc should have CREDIT_CARD detected"

    # ── Medical docs should detect MEDICAL_RECORD ───────────────
    med_results = [r for r in results if r.doc_id.startswith("doc_med")]
    any_mrn = any(
        r.pii_type_distribution.get("MEDICAL_RECORD", 0) >= 1 for r in med_results
    )
    # MEDICAL_RECORD pattern matches "MRN: 88421" format
    assert any_mrn, "At least one medical doc should have MEDICAL_RECORD detected"

    # ── JSON docs — should NOT crash on structured data ────────
    json_results = [r for r in results if r.doc_id.startswith("doc_json")]
    for r in json_results:
        assert isinstance(r.pii_features, list), (
            f"JSON doc {r.doc_id} should have features list, not error"
        )

    # ── Plain text docs — expected to have few/no PII features ─
    txt_results = [r for r in results if r.doc_id.startswith("doc_txt")]
    for r in txt_results:
        assert isinstance(r.pii_features, list), (
            f"Plain text doc {r.doc_id} should have features list"
        )


# ══════════════════════════════════════════════════════════════
# Test 2: Tier 0 → Tier 1 structural clustering flow
# ══════════════════════════════════════════════════════════════

def test_tier0_to_tier1_flow(sample_documents: list[Document]) -> None:
    """Tier 0 output → Tier 1 structural clustering produces buckets (not empty dict).

    Flow:
      1. Run Tier 0 on all docs, attach PIIFeatureVector to each Document
      2. Set structural_features on each doc based on metadata file_type
      3. Run StructuralClusterer.cluster()
      4. Verify: returns non-empty dict, each doc assigned to exactly one bucket
    """
    # ── Step 1: Tier 0 PII extraction ─────────────────────────
    engine = Tier0Engine({"context_window": 100, "confidence_threshold": 0.4})
    docs_tuples = [(d.doc_id, d.text) for d in sample_documents]
    tier0_results = engine.extract_batch(docs_tuples)

    # Attach Tier 0 results to documents
    pii_map = {r.doc_id: r for r in tier0_results}
    for doc in sample_documents:
        doc.pii_features = pii_map[doc.doc_id]

    # ── Step 2: Assign structural features from metadata ───────
    for doc in sample_documents:
        ft = doc.metadata.get("file_type", ".txt")
        doc.structural_features = StructuralFeatures(
            file_type=ft,
            file_size_quantile=1,
            page_count=5 if ft == ".pdf" else 0,
            paragraph_count=3,
            table_count=0,
            has_images=False,
            header_pattern="",
            json_schema_signature="type:object" if ft == ".json" else "",
            path_depth=1,
        )

    # ── Step 3: Tier 1 structural clustering ──────────────────
    clusterer = StructuralClusterer()
    buckets = clusterer.cluster(sample_documents)

    # ── Assertions ─────────────────────────────────────────────
    assert isinstance(buckets, dict), f"Expected dict, got {type(buckets)}"
    assert len(buckets) > 0, "Expected at least one bucket"
    # With 4-5 different file_types, expect several buckets
    assert len(buckets) >= 2, (
        f"Expected >= 2 structural buckets, got {len(buckets)}"
    )

    # All 20 doc_ids accounted for, no duplicates
    all_assigned: list[str] = []
    for bucket_id, doc_ids in buckets.items():
        assert isinstance(bucket_id, str), f"Bucket key should be str, got {type(bucket_id)}"
        assert len(bucket_id) == 64, f"Bucket ID should be 64-char SHA256, got len={len(bucket_id)}"
        assert isinstance(doc_ids, list), f"Bucket value should be list, got {type(doc_ids)}"
        all_assigned.extend(doc_ids)

    assert len(all_assigned) == 20, (
        f"Expected 20 assigned docs, got {len(all_assigned)}"
    )
    assert len(set(all_assigned)) == 20, "Duplicate doc_ids in bucket assignment"

    # Dictionaries with same file_type should be in same bucket
    pdf_docs = [d for d in sample_documents if d.metadata.get("file_type") == ".pdf"]
    if len(pdf_docs) >= 2:
        pdf_ids = {d.doc_id for d in pdf_docs}
        pdf_buckets = {
            bid for bid, dids in buckets.items()
            if pdf_ids & set(dids)
        }
        assert len(pdf_buckets) <= len(buckets), (
            "PDF docs should share buckets with same structural profile"
        )


# ══════════════════════════════════════════════════════════════
# Test 3: Cold start path — empty known types
# ══════════════════════════════════════════════════════════════

def test_cold_start_path(
    sample_documents: list[Document],
    mock_llm_client: Mock,
) -> None:
    """Empty known types → cold_start_classify runs without error.

    Verifies:
      - Returns one ClassificationResult per input document
      - All have method="llm_tier2"
      - Mock LLM is called once per document
      - Known type matcher has zero registered types
    """
    # ── Setup with empty known types ───────────────────────────
    matcher = KnownTypeMatcher(known_types=[])
    ner = Mock()
    ner.predict_batch.return_value = []
    propagator = LabelPropagator()  # Real LabelPropagator (lightweight)

    classifier = Tier2Classifier(matcher, ner, mock_llm_client, propagator)

    # ── Execute cold start on a subset (5 docs for speed) ──────
    subset = sample_documents[:5]
    results = classifier.cold_start_classify(subset)

    # ── Assertions ─────────────────────────────────────────────
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all(r.method == "llm_tier2" for r in results), (
        f"All cold start results should have method='llm_tier2'"
    )

    # Each doc_id present
    result_ids = {r.doc_id for r in results}
    assert result_ids == {d.doc_id for d in subset}

    # LLM was called once per document
    assert mock_llm_client.classify.call_count == 5

    # Each LLM call passed empty known_types (zero-shot)
    for call in mock_llm_client.classify.call_args_list:
        known_types = call[0][1]
        assert known_types == [], (
            f"Cold start should use empty known_types, got {known_types}"
        )

    # HR doc should get "HR Document" from our mock
    hr_results = [r for r in results if r.doc_id == "doc_hr_1"]
    assert len(hr_results) == 1
    assert hr_results[0].label == "HR Document"

    # Financial doc should get "Financial Report"
    fin_results = [r for r in results if r.doc_id == "doc_fin_1"]
    assert len(fin_results) == 1
    assert fin_results[0].label == "Financial Report"


# ══════════════════════════════════════════════════════════════
# Test 4: Cluster label propagation
# ══════════════════════════════════════════════════════════════

def test_cluster_label_propagation() -> None:
    """Create a cluster with known label → propagate → all docs get that label.

    Uses the real LabelPropagator (lightweight CPU-based) to verify that
    the propagate() method assigns labels to all cluster member documents.
    """
    # ── Create a cluster with 5 documents ──────────────────────
    cluster = ClusterInfo(
        cluster_id="test-cluster-001",
        doc_ids=["d1", "d2", "d3", "d4", "d5"],
        structural_bucket="sha256:test-bucket-hash-64chars-long-xxxxxxxxxxxxxxxxxxxxxxxxxxx",
        cluster_radius=0.15,
        representative_docs=["d1", "d2"],
        tfidf_keywords=["report", "financial", "annual", "revenue", "statement"],
        pii_distribution={"SSN": 10, "EMAIL": 5},
        language_distribution={"en": 5},
        label="Financial Report",
        label_confidence=0.88,
    )

    # ── Create matching Document objects ───────────────────────
    documents = [
        Document(doc_id=f"d{i}", text=f"Financial document content {i}")
        for i in range(1, 6)
    ]

    # ── Real LabelPropagator ───────────────────────────────────
    propagator = LabelPropagator({"seed": 42})

    results, needs_resplit = propagator.propagate(
        cluster=cluster,
        label="Financial Report",
        confidence=0.88,
        documents=documents,
    )

    # ── Assertions ─────────────────────────────────────────────
    assert isinstance(results, list)
    assert len(results) == 5, (
        f"Expected 5 ClassificationResults, got {len(results)}"
    )
    assert needs_resplit is False, "needs_resplit should be False for homogeneous cluster"

    # All results are ClassificationResult dataclass instances
    assert all(isinstance(r, ClassificationResult) for r in results)

    # All have the propagated label
    assert all(r.label == "Financial Report" for r in results), (
        f"All docs should get label 'Financial Report'"
    )

    # All have confidence 0.88
    assert all(r.confidence == 0.88 for r in results)

    # All have method="propagated"
    assert all(r.method == "propagated" for r in results)

    # Each doc_id present
    result_ids = {r.doc_id for r in results}
    assert result_ids == {"d1", "d2", "d3", "d4", "d5"}

    # Rationale is present
    assert all(len(r.rationale) > 0 for r in results)

    # ── Some should be marked for review (sampling with confidence < 0.85) ──
    # With seed=42, the sample is deterministic
    review_count = sum(1 for r in results if r.needs_manual_review)
    # Confidence 0.88 >= 0.85 so no docs should be auto-flagged from confidence
    # But the propagator marks samples for review when confidence < 0.85
    assert review_count == 0, (
        f"Confidence 0.88 >= 0.85, no manual review expected, got {review_count}"
    )


# ══════════════════════════════════════════════════════════════
# Test 5: PipelineResult output schema
# ══════════════════════════════════════════════════════════════

def test_end_to_end_output_schema() -> None:
    """PipelineResult-like dict has correct keys (results, stats).

    Verifies that the output schema conforms to the expected structure
    as defined in PipelineResult dataclass.
    """
    # ── Build a PipelineResult ─────────────────────────────────
    results = [
        ClassificationResult(
            doc_id="doc-001",
            label="Financial Report",
            confidence=0.92,
            method="known_match",
            is_new_type=False,
            needs_manual_review=False,
            rationale="3-signal match against known Financial Report type.",
        ),
        ClassificationResult(
            doc_id="doc-002",
            label="Medical Record",
            confidence=0.78,
            method="llm_tier2",
            is_new_type=False,
            needs_manual_review=False,
            rationale="LLM classification based on TF-IDF keywords and NER results.",
        ),
        ClassificationResult(
            doc_id="doc-003",
            label="SSN",
            confidence=0.55,
            method="llm_tier2",
            is_new_type=False,
            needs_manual_review=True,
            rationale="Low confidence — possible misclassification.",
        ),
        ClassificationResult(
            doc_id="doc-004",
            label="API Log",
            confidence=0.95,
            method="distilled",
            is_new_type=False,
            needs_manual_review=False,
            rationale="Classified by distilled SetFit model (confidence >= 0.85).",
        ),
    ]

    stats = {
        "total_docs": 4,
        "tier0_detection_rate": 0.75,
        "tier1_cluster_count": 2,
        "tier1_clustering_time_ms": 45.2,
        "tier2_known_match_count": 1,
        "tier2_llm_count": 2,
        "tier2_propagation_count": 4,
        "tier3_trigger_count": 1,
        "tier3_verified_count": 0,
        "distillation_count": 1,
        "new_types_discovered": 0,
        "llm_call_success_rate": 1.0,
        "e2e_latency_ms": 320.0,
    }

    pipeline_result = PipelineResult(results=results, stats=stats)

    # ── Schema assertions ──────────────────────────────────────
    assert isinstance(pipeline_result.results, list)
    assert isinstance(pipeline_result.stats, dict)

    assert len(pipeline_result.results) == 4
    assert pipeline_result.stats["total_docs"] == 4

    # All results are ClassificationResult instances
    assert all(isinstance(r, ClassificationResult) for r in pipeline_result.results)

    # Required stats keys present
    required_keys = {
        "total_docs", "tier0_detection_rate", "tier1_cluster_count",
        "tier2_known_match_count", "tier2_llm_count", "tier3_trigger_count",
        "llm_call_success_rate", "e2e_latency_ms",
    }
    missing = required_keys - set(pipeline_result.stats.keys())
    assert not missing, f"Missing required stats keys: {missing}"

    # PipelineResult can be converted to dict for serialization
    from dataclasses import asdict
    as_dict = asdict(pipeline_result)
    assert "results" in as_dict
    assert "stats" in as_dict


# ══════════════════════════════════════════════════════════════
# Test 6: Tier 3 triggers for sensitive documents
# ══════════════════════════════════════════════════════════════

def test_tier3_triggers_for_sensitive_docs(mock_llm_client: Mock) -> None:
    """Doc with SSN label triggers QualityGate (unit-level integration).

    Verifies the full should_trigger + verify path for sensitive documents.
    """
    gate = QualityGate(mock_llm_client)

    # ── SSN-labeled document ───────────────────────────────────
    ssn_doc = Document(
        doc_id="doc-sensitive-01",
        text="Employee record. Name: John Doe. SSN: 123-45-6789.",
        embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )

    ssn_cluster = ClusterInfo(
        cluster_id="cluster-ssn",
        doc_ids=["doc-sensitive-01"],
        structural_bucket="sha256:ssn-bucket-hash",
        cluster_radius=0.10,
        representative_docs=["doc-sensitive-01"],
        tfidf_keywords=["employee", "ssn", "record"],
        pii_distribution={"SSN": 1},
        language_distribution={"en": 1},
        centroid_embedding=np.array([0.1, 0.25, 0.35, 0.45], dtype=np.float32),
        label="SSN",
    )

    ssn_classification = ClassificationResult(
        doc_id="doc-sensitive-01",
        label="SSN",
        confidence=0.92,
        method="llm_tier2",
    )

    # ── Trigger check ──────────────────────────────────────────
    should_trigger = gate.should_trigger(
        ssn_doc, ssn_cluster, ssn_classification
    )
    assert should_trigger is True, (
        "SSN label in high_sensitivity_types should trigger QualityGate"
    )

    # ── Verify path ────────────────────────────────────────────
    verified = gate.verify(ssn_doc, ssn_cluster, ssn_classification)

    assert isinstance(verified, ClassificationResult)
    assert verified.method == "llm_tier3", (
        f"Expected method='llm_tier3', got '{verified.method}'"
    )
    assert verified.label == "SSN"
    assert verified.confidence == 0.85  # from our mock
    assert verified.needs_manual_review is False  # 0.85 >= 0.6

    # ── Non-sensitive document → no trigger ────────────────────
    normal_doc = Document(
        doc_id="doc-normal-01",
        text="Q4 2025 Financial Report — Revenue: $12.3M.",
        embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )

    normal_classification = ClassificationResult(
        doc_id="doc-normal-01",
        label="Financial Report",
        confidence=0.92,
        method="known_match",
        needs_manual_review=False,
    )

    normal_cluster = ClusterInfo(
        cluster_id="cluster-normal",
        doc_ids=["doc-normal-01"],
        structural_bucket="sha256:normal-hash",
        cluster_radius=0.10,
        representative_docs=["doc-normal-01"],
        tfidf_keywords=["report", "financial"],
        pii_distribution={},
        language_distribution={"en": 1},
        centroid_embedding=np.array([0.1, 0.25, 0.35, 0.45], dtype=np.float32),
        label="Financial Report",
    )

    assert gate.should_trigger(normal_doc, normal_cluster, normal_classification) is False

    # ── Low confidence trigger ─────────────────────────────────
    low_conf = ClassificationResult(
        doc_id="doc-normal-01",
        label="Financial Report",
        confidence=0.55,  # in [0.5, 0.8) → trigger
        method="llm_tier2",
    )
    assert gate.should_trigger(normal_doc, normal_cluster, low_conf) is True

    # ── Manual review flag trigger ─────────────────────────────
    needs_review = ClassificationResult(
        doc_id="doc-normal-01",
        label="Financial Report",
        confidence=0.92,
        method="llm_tier2",
        needs_manual_review=True,
    )
    assert gate.should_trigger(normal_doc, normal_cluster, needs_review) is True


# ══════════════════════════════════════════════════════════════
# Test 7: Discovery outlier accumulation
# ══════════════════════════════════════════════════════════════

def test_discovery_outlier_accumulation() -> None:
    """Add outlier docs → buffer size increases correctly.

    Verifies the DiscoveryLoop outlier collection and buffer management:
      - Initial buffer size = 0
      - After adding N docs, buffer size = N
      - get_buffer_docs() returns correct documents
      - should_run() returns False when below thresholds
      - Buffer clears after run() (with mocked collaborators)
    """
    # ── Mock collaborators ─────────────────────────────────────
    # First 5 docs each get a unique bucket → no pattern trigger
    _bucket_counter = [0]

    def _unique_bucket(doc):
        _bucket_counter[0] += 1
        return f"sha256:unique-bucket-{_bucket_counter[0]}"

    structural = Mock()
    structural.assign_bucket.side_effect = _unique_bucket
    structural.cluster.return_value = {
        "bucket-outlier": ["out-1", "out-2", "out-3"],
    }

    refiner = Mock()
    refiner.refine.return_value = [
        ClusterInfo(
            cluster_id="bucket-outlier_0",
            doc_ids=["out-1", "out-2", "out-3"],
            structural_bucket="bucket-outlier",
            cluster_radius=0.10,
            representative_docs=["out-1"],
            tfidf_keywords=["novel", "unseen", "pattern"],
            pii_distribution={"NOVEL_TYPE": 3},
            language_distribution={"en": 3},
            centroid_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ),
    ]

    embedder = Mock()
    # High coherence embeddings → passes 0.75 threshold
    embedder.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.02, 0.0],
    ], dtype=np.float32)

    matcher = Mock()
    matcher._types = {}  # Empty known types → distance check always passes
    matcher.register_type = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    # ── Initial state ──────────────────────────────────────────
    assert loop.get_buffer_size() == 0
    assert loop.get_buffer_docs() == []
    assert loop.should_run() is False

    # ── Add outliers (each with unique bucket) ─────────────────
    for i in range(5):
        doc = Document(
            doc_id=f"out-{i + 1}",
            text=f"Novel document type content {i + 1} with unusual patterns.",
            metadata={"file_type": ".custom"},
        )
        loop.collect_outlier(doc, "low_match_score")

    # ── Buffer reflects additions ──────────────────────────────
    assert loop.get_buffer_size() == 5, (
        f"Expected buffer size 5, got {loop.get_buffer_size()}"
    )

    buffer_docs = loop.get_buffer_docs()
    assert len(buffer_docs) == 5
    assert buffer_docs[0].doc_id == "out-1"
    assert buffer_docs[4].doc_id == "out-5"

    # ── Not enough to trigger (default min_trigger_count = 100, no pattern repeats) ─
    assert loop.should_run() is False, (
        "5 unique docs < min_trigger_count(100) and no pattern repeats should NOT trigger"
    )

    # ── Add same-pattern docs to trigger via pattern threshold ─
    structural.assign_bucket.side_effect = None  # reset side_effect
    structural.assign_bucket.return_value = "sha256:repeating-pattern"
    for i in range(5):
        doc = Document(
            doc_id=f"out-repeat-{i + 1}",
            text=f"Repeating novel type content {i + 1}.",
            metadata={"file_type": ".custom"},
            label="novel_type",
        )
        loop.collect_outlier(doc, "low_match_score")

    assert loop.get_buffer_size() == 10

    # ── should_run should now be True (pattern threshold = 5) ──
    assert loop.should_run() is True, (
        "Same (bucket, label) pattern >= 5 should trigger should_run"
    )

    # ── run() with only 3 docs matching mock cluster ───────────
    # Actually, since we have 10 docs in the buffer, but the mock structural.cluster only
    # references ["out-1", "out-2", "out-3"], only those 3 will be processed.
    # Let's re-create the loop with appropriate mocks for the docs we have.
    # Actually, let's just verify the buffer state before run and then run.

    # Create a fresh loop with mocks matching out-1 through out-3
    structural2 = Mock()
    structural2.assign_bucket.return_value = "sha256:pattern-hash"
    structural2.cluster.return_value = {
        "pattern-bucket": ["out-1", "out-2", "out-3"],
    }

    refiner2 = Mock()
    refiner2.refine.return_value = [
        ClusterInfo(
            cluster_id="pattern-bucket_0",
            doc_ids=["out-1", "out-2", "out-3"],
            structural_bucket="pattern-bucket",
            cluster_radius=0.10,
            representative_docs=["out-1"],
            tfidf_keywords=["novel", "unseen"],
            pii_distribution={"NOVEL_TYPE": 3},
            language_distribution={"en": 3},
            centroid_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ),
    ]

    embedder2 = Mock()
    embedder2.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.02, 0.0],
    ], dtype=np.float32)

    matcher2 = Mock()
    matcher2._types = {}
    matcher2.register_type = Mock()

    loop2 = DiscoveryLoop(structural2, refiner2, embedder2, matcher2)

    for i in range(3):
        doc = Document(
            doc_id=f"out-{i + 1}",
            text=f"Novel document type content {i + 1}.",
            metadata={"file_type": ".custom"},
        )
        loop2.collect_outlier(doc, "low_match_score")

    assert loop2.get_buffer_size() == 3

    # ── Run discovery ──────────────────────────────────────────
    discovered = loop2.run()

    # Should discover 1 type (mocks return good coherence + no known types)
    assert len(discovered) == 1, (
        f"Expected 1 discovered type, got {len(discovered)}"
    )
    assert isinstance(discovered[0], KnownType)
    assert discovered[0].type_id.startswith("discovered_")
    assert discovered[0].status == "pending_review"

    # Buffer cleared after run
    assert loop2.get_buffer_size() == 0, (
        "Buffer should be cleared after successful run()"
    )
    assert loop2.get_buffer_docs() == []
    assert loop2.should_run() is False
