"""
Unit tests for core.clustering_engine.ClusteringEngine.
"""
import numpy as np
import pytest
from core.clustering_engine import ClusteringEngine


# ── Helper ────────────────────────────────────────────────────

def _make_three_clusters():
    """3 clear 2D clusters of 10 points each at [0,0], [10,10], [20,20]."""
    rng = np.random.RandomState(42)
    parts = []
    for center in [[0, 0], [10, 10], [20, 20]]:
        parts.append(rng.randn(10, 2) * 0.5 + np.array(center))
    return np.vstack(parts)


def _minimal_config(**overrides):
    """Return a config dict with defaults set for stable, predictable clustering."""
    return {
        "method": "kmeans",
        "auto_k": False,
        "k_range": [3, 3],
        "k_step": 1,
        "random_state": 42,
        "split_enabled": False,
        "min_cluster_size": 2,
        "small_cluster_threshold": 1,
        "small_cluster_min_coherence": 0.80,
        **overrides,
    }


# ── fit ───────────────────────────────────────────────────────

def test_fit_basic():
    """3 separable clusters → labels are {0,1,2} with no outliers."""
    embeddings = _make_three_clusters()
    engine = ClusteringEngine(_minimal_config())
    labels = engine.fit(embeddings)

    unique = set(labels)
    assert -1 not in unique, f"expected no -1, got {unique}"
    assert unique == {0, 1, 2}, f"expected {{0,1,2}}, got {unique}"
    assert len(labels) == 30


def test_fit_with_auto_k_disabled():
    """auto_k=False with k_range[0]=3 produces exactly 3 labels."""
    embeddings = _make_three_clusters()
    engine = ClusteringEngine(_minimal_config(auto_k=False))
    labels = engine.fit(embeddings)

    assert len(set(labels)) == 3
    assert -1 not in labels


# ── _compute_coherence ────────────────────────────────────────

def test_compute_coherence_perfect():
    """Identical vectors → coherence ≈ 1.0."""
    vec = np.random.RandomState(42).randn(5, 10)
    identical = np.tile(vec[0:1], (5, 1))  # 5 copies of the same row
    coherence = ClusteringEngine._compute_coherence(identical)
    assert abs(coherence - 1.0) < 0.01, f"expected ~1.0, got {coherence}"


def test_compute_coherence_random():
    """Random vectors → coherence < 1.0."""
    random_vecs = np.random.RandomState(42).randn(100, 10)
    coherence = ClusteringEngine._compute_coherence(random_vecs)
    assert coherence < 1.0, f"random coherence should be < 1.0, got {coherence}"


def test_compute_coherence_single():
    """Single vector → coherence = 1.0."""
    coherence = ClusteringEngine._compute_coherence(np.random.RandomState(42).randn(1, 10))
    assert coherence == 1.0


# ── _renumber_labels ──────────────────────────────────────────

def test_renumber_labels():
    """Non-contiguous labels [5,5,10,10,-1] → [0,0,1,1,-1]."""
    old = np.array([5, 5, 10, 10, -1])
    new = ClusteringEngine._renumber_labels(old)
    expected = np.array([0, 0, 1, 1, -1])
    assert np.array_equal(new, expected), f"expected {expected}, got {new}"


def test_renumber_labels_contiguous():
    """Already-contiguous labels [0,1,2] stay [0,1,2]."""
    old = np.array([0, 1, 2])
    new = ClusteringEngine._renumber_labels(old)
    assert np.array_equal(new, old)


# ── _stage_c_small_clusters ───────────────────────────────────

def test_small_cluster_merge():
    """
    Two big clusters + one high-coherence small cluster (kept) +
    one low-coherence small cluster (merged).
    """
    rng = np.random.RandomState(42)

    # 2 big clusters (≥ small_cluster_threshold)
    big1 = rng.randn(10, 2) * 0.5 + np.array([0.0, 0.0])
    big2 = rng.randn(10, 2) * 0.5 + np.array([10.0, 10.0])

    # High-coherence small cluster: 2 near-identical points
    small_coherent = np.array([[5.0, 5.0], [5.001, 5.001]])

    # Low-coherence small cluster: 2 orthogonal-direction points
    # [5,0] and [0,5] have cosine similarity = 0, well below threshold
    small_incoherent = np.array([[5.0, 0.0], [0.0, 5.0]])

    embeddings = np.vstack([big1, big2, small_coherent, small_incoherent])
    labels = np.array([0]*10 + [1]*10 + [2]*2 + [3]*2)

    config = {
        "small_cluster_threshold": 5,
        "small_cluster_min_coherence": 0.80,
        "split_enabled": False,
    }
    engine = ClusteringEngine(config)
    new_labels = engine._stage_c_small_clusters(embeddings, labels)

    # The high-coherence small cluster (label 2) should be kept
    assert 2 in new_labels, "high-coherence small cluster should be kept"

    # The low-coherence small cluster (label 3) should be merged away
    # (its centroid [2.5, 2.5] is nearer big1 [0,0] than big2 [10,10],
    # so it merges to label 0)
    assert 3 not in new_labels, "low-coherence small cluster should be merged"
