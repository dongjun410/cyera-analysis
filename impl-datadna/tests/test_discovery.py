"""Tests for DiscoveryLoop — outlier accumulation → periodic re-clustering
→ new type evaluation → registration (TDD).

Test contracts (write these BEFORE implementation):
  1. test_outlier_collection — Add 3 docs → buffer size = 3
  2. test_should_run_count_trigger — Buffer reaches 100 → True
  3. test_should_run_pattern_trigger — Same pattern 5 times → True
  4. test_should_run_below_threshold — 50 outliers, no pattern → False
  5. test_run_produces_types — Mock refiner returns 2 clusters → 2 KnownType objects
  6. test_coherence_filter — Cluster with coherence 0.5 (< 0.75) → rejected
  7. test_distance_filter — Cluster too close to known type (distance 0.1 < 0.3) → rejected
  8. test_min_size_filter — Cluster with 2 docs (< 3) → rejected
  9. test_run_clears_buffer — After run → buffer size = 0 (processed)
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.types import ClusterInfo, Document, KnownType, StructuralFeatures


# ──────────────────────────────────────────────────────────────
# Test 1: Outlier collection increments buffer
# ──────────────────────────────────────────────────────────────

def test_outlier_collection() -> None:
    """Adding 3 documents should set buffer size to 3."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.assign_bucket.return_value = "bucket-default"
    refiner = Mock()
    embedder = Mock()
    matcher = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    assert loop.get_buffer_size() == 0

    for i in range(3):
        doc = Document(doc_id=f"doc-{i}", text=f"text content {i}")
        loop.collect_outlier(doc, "low_match_score")

    assert loop.get_buffer_size() == 3
    assert len(loop.get_buffer_docs()) == 3
    # Verify docs are retrievable
    docs = loop.get_buffer_docs()
    assert docs[0].doc_id == "doc-0"


# ──────────────────────────────────────────────────────────────
# Test 2: should_run triggers on count threshold
# ──────────────────────────────────────────────────────────────

def test_should_run_count_trigger() -> None:
    """Buffer reaching 100 documents should trigger should_run()."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.assign_bucket.return_value = "bucket-default"
    refiner = Mock()
    embedder = Mock()
    matcher = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    assert not loop.should_run()

    for i in range(100):
        doc = Document(doc_id=f"doc-{i}", text=f"text {i}")
        loop.collect_outlier(doc, "low_match_score")

    assert loop.should_run()


# ──────────────────────────────────────────────────────────────
# Test 3: should_run triggers on pattern repetition
# ──────────────────────────────────────────────────────────────

def test_should_run_pattern_trigger() -> None:
    """Same (bucket, label) pattern 5 times should trigger should_run()."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.assign_bucket.return_value = "sha256:abc123"
    refiner = Mock()
    embedder = Mock()
    matcher = Mock()

    loop = DiscoveryLoop(
        structural, refiner, embedder, matcher,
        config={"same_pattern_threshold": 5},
    )

    # Add 3 docs — should not trigger yet (below pattern threshold of 5)
    for i in range(3):
        doc = Document(doc_id=f"doc-{i}", text=f"text {i}", label="financial_report")
        loop.collect_outlier(doc, "low_match_score")

    assert not loop.should_run()

    # Add 2 more with same pattern → should trigger at 5
    for i in range(3, 5):
        doc = Document(doc_id=f"doc-{i}", text=f"text {i}", label="financial_report")
        loop.collect_outlier(doc, "low_match_score")

    assert loop.should_run()


# ──────────────────────────────────────────────────────────────
# Test 4: should_run returns False below thresholds
# ──────────────────────────────────────────────────────────────

def test_should_run_below_threshold() -> None:
    """50 outliers with no repeated pattern → should_run() returns False."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    # Each doc gets a different bucket so no pattern repeats
    structural.assign_bucket.side_effect = lambda doc: f"bucket-{doc.doc_id}"
    refiner = Mock()
    embedder = Mock()
    matcher = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    for i in range(50):
        doc = Document(doc_id=f"doc-{i}", text=f"text {i}", label=f"label-{i}")
        loop.collect_outlier(doc, "low_match_score")

    assert not loop.should_run()


# ──────────────────────────────────────────────────────────────
# Test 5: run() produces KnownType objects
# ──────────────────────────────────────────────────────────────

def test_run_produces_types() -> None:
    """Mock refiner returns 2 clusters → run() returns 2 KnownType objects."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.cluster.return_value = {
        "bucket-1": ["doc-1", "doc-2", "doc-3"],
        "bucket-2": ["doc-4", "doc-5", "doc-6"],
    }

    refiner = Mock()
    centroid1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    centroid2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    refiner.refine.side_effect = [
        [ClusterInfo(
            cluster_id="bucket-1_0",
            doc_ids=["doc-1", "doc-2", "doc-3"],
            structural_bucket="bucket-1",
            cluster_radius=0.15,
            representative_docs=["doc-1"],
            tfidf_keywords=["report", "financial", "annual"],
            pii_distribution={"SSN": 3, "EMAIL": 2},
            language_distribution={"en": 3},
            centroid_embedding=centroid1,
        )],
        [ClusterInfo(
            cluster_id="bucket-2_0",
            doc_ids=["doc-4", "doc-5", "doc-6"],
            structural_bucket="bucket-2",
            cluster_radius=0.12,
            representative_docs=["doc-4"],
            tfidf_keywords=["patient", "medical", "diagnosis"],
            pii_distribution={"PATIENT_ID": 3},
            language_distribution={"en": 3},
            centroid_embedding=centroid2,
        )],
    ]

    embedder = Mock()
    # Return highly similar embeddings → coherence ~1.0 (passes 0.75 threshold)
    embedder.encode.return_value = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.01, 0.0, 0.0],
        [0.98, 0.02, 0.0, 0.0],
    ], dtype=np.float32)

    matcher = Mock()
    matcher._types = {}  # No known types → distance check always passes
    matcher.register_type = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    # Add outlier docs matching the doc_ids referenced by cluster()
    for i in range(6):
        doc = Document(doc_id=f"doc-{i+1}", text=f"document text content {i+1}")
        loop.collect_outlier(doc, "low_match_score")

    result = loop.run()

    assert len(result) == 2
    assert isinstance(result[0], KnownType)
    assert isinstance(result[1], KnownType)

    # Verify KnownType attributes
    for kt in result:
        assert kt.type_id.startswith("discovered_")
        assert kt.type_name.startswith("Unknown-Type-")
        assert kt.status == "pending_review"
        assert kt.semantic_centroid is not None
        assert kt.structural_signature in ("bucket-1", "bucket-2")

    # Verify register_type was called for each discovered type
    assert matcher.register_type.call_count == 2


# ──────────────────────────────────────────────────────────────
# Test 6: Coherence filter rejects low-coherence clusters
# ──────────────────────────────────────────────────────────────

def test_coherence_filter() -> None:
    """Cluster with coherence 0.577 (< 0.75) should be rejected."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.cluster.return_value = {
        "bucket-1": ["doc-1", "doc-2", "doc-3"],
    }

    refiner = Mock()
    refiner.refine.return_value = [ClusterInfo(
        cluster_id="bucket-1_0",
        doc_ids=["doc-1", "doc-2", "doc-3"],
        structural_bucket="bucket-1",
        cluster_radius=0.8,
        representative_docs=["doc-1"],
        tfidf_keywords=["scattered", "unrelated", "random"],
        pii_distribution={},
        language_distribution={"en": 3},
        centroid_embedding=np.array([0.577, 0.577, 0.577], dtype=np.float32),
    )]

    embedder = Mock()
    # Three orthogonal vectors → centroid at [1/3, 1/3, 1/3]
    # cos_sim of each to centroid ≈ 0.577, mean coherence ≈ 0.577 < 0.75
    embedder.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    matcher = Mock()
    matcher._types = {}
    matcher.register_type = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    for i in range(3):
        doc = Document(doc_id=f"doc-{i+1}", text=f"text {i+1}")
        loop.collect_outlier(doc, "low_match_score")

    result = loop.run()

    assert len(result) == 0, "Low-coherence cluster should be rejected"
    assert matcher.register_type.call_count == 0


# ──────────────────────────────────────────────────────────────
# Test 7: Distance filter rejects clusters too close to known type
# ──────────────────────────────────────────────────────────────

def test_distance_filter() -> None:
    """Cluster too close to known type (cosine distance ~0.01 < 0.3) → rejected."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.cluster.return_value = {
        "bucket-1": ["doc-1", "doc-2", "doc-3"],
    }

    refiner = Mock()
    refiner.refine.return_value = [ClusterInfo(
        cluster_id="bucket-1_0",
        doc_ids=["doc-1", "doc-2", "doc-3"],
        structural_bucket="bucket-1",
        cluster_radius=0.05,
        representative_docs=["doc-1"],
        tfidf_keywords=["financial", "report", "annual"],
        pii_distribution={"SSN": 3},
        language_distribution={"en": 3},
        centroid_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )]

    embedder = Mock()
    # All embeddings near [1, 0, 0] → centroid ≈ [1, 0, 0], coherence high
    embedder.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.999, 0.001, 0.0],
        [0.998, 0.002, 0.0],
    ], dtype=np.float32)

    # Known type with centroid very close to the cluster centroid
    known_type = KnownType(
        type_id="known-financial",
        type_name="Financial Report",
        description="Existing financial report type",
        semantic_centroid=np.array([0.999, 0.001, 0.0], dtype=np.float32),
    )

    matcher = Mock()
    matcher._types = {"known-financial": known_type}
    matcher.register_type = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    for i in range(3):
        doc = Document(doc_id=f"doc-{i+1}", text=f"financial text {i+1}")
        loop.collect_outlier(doc, "low_match_score")

    result = loop.run()

    assert len(result) == 0, (
        "Cluster too close to known type should be rejected by distance filter"
    )
    assert matcher.register_type.call_count == 0


# ──────────────────────────────────────────────────────────────
# Test 8: Min cluster size filter rejects small clusters
# ──────────────────────────────────────────────────────────────

def test_min_size_filter() -> None:
    """Cluster with 2 documents (< min_cluster_size=3) → rejected."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.cluster.return_value = {
        "bucket-1": ["doc-1", "doc-2"],
    }

    refiner = Mock()
    refiner.refine.return_value = [ClusterInfo(
        cluster_id="bucket-1_0",
        doc_ids=["doc-1", "doc-2"],
        structural_bucket="bucket-1",
        cluster_radius=0.1,
        representative_docs=["doc-1"],
        tfidf_keywords=["small", "cluster"],
        pii_distribution={},
        language_distribution={"en": 2},
        centroid_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )]

    embedder = Mock()
    embedder.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
    ], dtype=np.float32)

    matcher = Mock()
    matcher._types = {}
    matcher.register_type = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    for i in range(2):
        doc = Document(doc_id=f"doc-{i+1}", text=f"text {i+1}")
        loop.collect_outlier(doc, "low_match_score")

    result = loop.run()

    assert len(result) == 0, "Cluster with < 3 docs should be rejected by min size filter"
    assert matcher.register_type.call_count == 0


# ──────────────────────────────────────────────────────────────
# Test 9: run() clears buffer after processing
# ──────────────────────────────────────────────────────────────

def test_run_clears_buffer() -> None:
    """After run() completes, the outlier buffer should be empty."""
    from src.discovery.loop import DiscoveryLoop

    structural = Mock()
    structural.cluster.return_value = {
        "bucket-1": ["doc-1", "doc-2", "doc-3"],
    }

    refiner = Mock()
    refiner.refine.return_value = [ClusterInfo(
        cluster_id="bucket-1_0",
        doc_ids=["doc-1", "doc-2", "doc-3"],
        structural_bucket="bucket-1",
        cluster_radius=0.1,
        representative_docs=["doc-1"],
        tfidf_keywords=["report", "data"],
        pii_distribution={},
        language_distribution={"en": 3},
        centroid_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )]

    embedder = Mock()
    embedder.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.02, 0.0],
    ], dtype=np.float32)

    matcher = Mock()
    matcher._types = {}
    matcher.register_type = Mock()

    loop = DiscoveryLoop(structural, refiner, embedder, matcher)

    for i in range(3):
        doc = Document(doc_id=f"doc-{i+1}", text=f"text {i+1}")
        loop.collect_outlier(doc, "low_match_score")

    assert loop.get_buffer_size() == 3

    result = loop.run()

    assert len(result) == 1, "Should have discovered 1 new type"
    assert loop.get_buffer_size() == 0, "Buffer should be empty after run()"
    assert loop.get_buffer_docs() == [], "get_buffer_docs() should return empty list"
    assert not loop.should_run(), "should_run() should be False after buffer cleared"
