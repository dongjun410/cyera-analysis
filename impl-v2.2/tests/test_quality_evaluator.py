"""
Unit tests for core.quality_evaluator.QualityEvaluator.
"""
import numpy as np
import pytest
from core.quality_evaluator import QualityEvaluator


# ── Helpers ───────────────────────────────────────────────────

def _make_perfect_clusters():
    """3 near-orthogonal 10D clusters → high silhouette with cosine metric."""
    rng = np.random.RandomState(42)
    n = 10
    dims = 10

    c0 = rng.randn(n, dims) * 0.1
    c0[:, :3] += 1.0  # strong signal in dims 0-2

    c1 = rng.randn(n, dims) * 0.1
    c1[:, 3:6] += 1.0  # strong signal in dims 3-5

    c2 = rng.randn(n, dims) * 0.1
    c2[:, 6:9] += 1.0  # strong signal in dims 6-8

    return np.vstack([c0, c1, c2])


# ── evaluate ──────────────────────────────────────────────────

def test_evaluate_perfect_clusters():
    embeddings = _make_perfect_clusters()
    labels = np.array([0]*10 + [1]*10 + [2]*10)
    evaluator = QualityEvaluator({"enabled": True})
    metrics = evaluator.evaluate(embeddings, labels)

    assert metrics["silhouette_score"] > 0.8
    assert metrics["davies_bouldin_index"] < 1.0
    assert "calinski_harabasz_index" in metrics
    assert metrics["num_clusters"] == 3
    assert metrics["num_outliers"] == 0


def test_evaluate_with_outliers():
    embeddings = _make_perfect_clusters()
    # Add 2 outlier points in 10D
    rng = np.random.RandomState(42)
    outliers = rng.randn(2, 10) * 0.5
    embeddings = np.vstack([embeddings, outliers])
    labels = np.array([0]*10 + [1]*10 + [2]*10 + [-1]*2)

    evaluator = QualityEvaluator({"enabled": True})
    metrics = evaluator.evaluate(embeddings, labels)

    assert metrics["num_outliers"] == 2


def test_evaluate_disabled():
    evaluator = QualityEvaluator({"enabled": False})
    metrics = evaluator.evaluate(
        np.random.RandomState(42).randn(10, 5),
        np.array([0]*5 + [1]*5),
    )
    assert metrics == {}


def test_evaluate_fewer_than_2_clusters():
    evaluator = QualityEvaluator({"enabled": True})
    embeddings = np.random.RandomState(42).randn(10, 5)
    labels = np.array([0]*9 + [-1]*1)

    metrics = evaluator.evaluate(embeddings, labels)
    assert metrics == {"error": "fewer_than_2_clusters"}


def test_evaluate_custom_metrics():
    evaluator = QualityEvaluator({"enabled": True, "metrics": ["silhouette"]})
    embeddings = _make_perfect_clusters()
    labels = np.array([0]*10 + [1]*10 + [2]*10)

    metrics = evaluator.evaluate(embeddings, labels)
    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" not in metrics
    assert "calinski_harabasz_index" not in metrics
    # Basic stats are always included
    assert "num_clusters" in metrics


def test_cluster_statistics():
    """3 clusters with sizes [3, 5, 2]."""
    rng = np.random.RandomState(42)
    c0 = rng.randn(3, 5) + np.array([0.0]*5)
    c1 = rng.randn(5, 5) + np.array([10.0]*5)
    c2 = rng.randn(2, 5) + np.array([20.0]*5)
    embeddings = np.vstack([c0, c1, c2])
    labels = np.array([0]*3 + [1]*5 + [2]*2)

    evaluator = QualityEvaluator({"enabled": True})
    metrics = evaluator.evaluate(embeddings, labels)

    assert metrics["num_clusters"] == 3
    # mean = (3+5+2)/3 = 3.333... → round(1) = 3.3
    assert metrics["cluster_size_mean"] == pytest.approx(3.3, abs=0.1)
    # std (ddof=0) = 1.247... → round(1) = 1.2
    assert metrics["cluster_size_std"] == pytest.approx(1.2, abs=0.2)
    assert metrics["cluster_size_min"] == 2
    assert metrics["cluster_size_max"] == 5
