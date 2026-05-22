"""
Unit tests for core.sensitivity_adaptive_scheduler.SensitivityAdaptiveScheduler.
"""
import numpy as np
import pytest
from core.sensitivity_adaptive_scheduler import SensitivityAdaptiveScheduler


# ── Reusable config ───────────────────────────────────────────

SENS_CONFIG = {
    "method": "kmeans",
    "auto_k": False,
    "k_range": [2, 2],
    "k_step": 1,
    "random_state": 42,
    "split_enabled": False,
    "small_cluster_threshold": 1,
    "sensitivity_adaptive": {
        "enabled": True,
        "weights": [2.0, 3.0, 8.0, 10.0, 7.0, 0.5, 1.0, 1.0, 4.0, 1.5],
        "tier_thresholds": [0.3, 0.7],
    },
}


# ── compute_sensitivity_scores ────────────────────────────────

def test_compute_sensitivity_scores_no_pii():
    """All-zero PII columns → all scores = 0."""
    # 19 cols: 10 PII + 7 structure + 2 metadata
    vectors = np.zeros((5, 19), dtype=np.float32)
    scheduler = SensitivityAdaptiveScheduler(SENS_CONFIG)
    scores = scheduler.compute_sensitivity_scores(vectors)

    assert scores.shape == (5,)
    assert np.all(scores == 0.0)


def test_compute_sensitivity_scores_high_pii():
    """High PII counts → scores > 0."""
    vectors = np.zeros((3, 19), dtype=np.float32)
    # High counts in high-weight PII columns (e.g. ssn=col3 has weight 10.0)
    vectors[:, 3] = 5.0  # ssn
    vectors[:, 2] = 3.0  # credit_card

    scheduler = SensitivityAdaptiveScheduler(SENS_CONFIG)
    scores = scheduler.compute_sensitivity_scores(vectors)

    assert scores.shape == (3,)
    assert np.all(scores > 0.0)


def test_compute_sensitivity_scores_normalized():
    """All scores should be in [0, 1] range."""
    rng = np.random.RandomState(42)
    vectors = rng.rand(50, 19).astype(np.float32) * 10

    scheduler = SensitivityAdaptiveScheduler(SENS_CONFIG)
    scores = scheduler.compute_sensitivity_scores(vectors)

    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


# ── partition_into_tiers ──────────────────────────────────────

def test_partition_into_tiers():
    """Scores [0.0, 0.5, 0.8, 1.0] → correct tier masks."""
    scores = np.array([0.0, 0.5, 0.8, 1.0])

    scheduler = SensitivityAdaptiveScheduler(SENS_CONFIG)
    tiers = scheduler.partition_into_tiers(scores)

    # low: < 0.3 → [0.0]
    assert np.array_equal(tiers["low"], [True, False, False, False])
    # medium: >= 0.3 and < 0.7 → [0.5]
    assert np.array_equal(tiers["medium"], [False, True, False, False])
    # high: >= 0.7 → [0.8, 1.0]
    assert np.array_equal(tiers["high"], [False, False, True, True])


def test_partition_all_low():
    """All scores below low_threshold → all in 'low' tier."""
    scores = np.array([0.0, 0.1, 0.2])

    scheduler = SensitivityAdaptiveScheduler(SENS_CONFIG)
    tiers = scheduler.partition_into_tiers(scores)

    assert np.all(tiers["low"])
    assert not np.any(tiers["medium"])
    assert not np.any(tiers["high"])


# ── fit ───────────────────────────────────────────────────────

def test_fit_disabled():
    """When disabled, returns label array of correct length."""
    config = {
        **SENS_CONFIG,
        "sensitivity_adaptive": {"enabled": False, **SENS_CONFIG["sensitivity_adaptive"]},
    }
    rng = np.random.RandomState(42)
    embeddings = rng.randn(30, 64).astype(np.float32)
    structure_vectors = np.zeros((30, 19), dtype=np.float32)

    scheduler = SensitivityAdaptiveScheduler(config)
    labels = scheduler.fit(embeddings, structure_vectors)

    assert len(labels) == 30
    assert labels.dtype == np.int64 or labels.dtype == np.int32


def test_fit_basic():
    """Small 2D embeddings + structure vectors → correct length, >1 clusters."""
    rng = np.random.RandomState(42)
    # Create 2 clusters in 2D
    c0 = rng.randn(15, 2) * 0.5 + np.array([0.0, 0.0])
    c1 = rng.randn(15, 2) * 0.5 + np.array([10.0, 10.0])
    embeddings = np.vstack([c0, c1]).astype(np.float32)

    # Minimal structure vectors (19 cols)
    structure_vectors = np.zeros((30, 19), dtype=np.float32)

    scheduler = SensitivityAdaptiveScheduler(SENS_CONFIG)
    labels = scheduler.fit(embeddings, structure_vectors)

    assert len(labels) == 30
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    assert n_clusters > 1, f"expected >1 clusters, got {n_clusters}"
