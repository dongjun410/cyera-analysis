import pytest
import numpy as np

from core.iterative_optimizer import IterativeOptimizer


DEFAULT_CONFIG = {
    "max_iterations": 3,
    "convergence_threshold": 0.02,
}


class TestIterativeOptimizer:

    def test_compute_centroids(self):
        """3 clusters of 2 points each, verify centroids."""
        optimizer = IterativeOptimizer(DEFAULT_CONFIG)

        embeddings = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [20.0, 20.0],
            [20.1, 20.1],
        ], dtype=np.float64)
        labels = np.array([0, 0, 1, 1, 2, 2])

        centroids = optimizer._compute_centroids(embeddings, labels)

        assert set(centroids.keys()) == {0, 1, 2}
        assert np.allclose(centroids[0], [0.05, 0.05])
        assert np.allclose(centroids[1], [10.05, 10.05])
        assert np.allclose(centroids[2], [20.05, 20.05])

    def test_optimize_converges_immediately(self):
        """Perfectly separated clusters in different directions, labels preserved."""
        optimizer = IterativeOptimizer(DEFAULT_CONFIG)

        # Clusters point in orthogonal directions (not scalar multiples)
        # so cosine similarity won't merge them
        embeddings = np.array([
            [1.0, 0.0], [1.1, 0.1],    # cluster 0: near +x axis
            [0.0, 1.0], [0.1, 1.1],    # cluster 1: near +y axis
            [-1.0, 0.0], [-1.1, 0.1],  # cluster 2: near -x axis
        ], dtype=np.float64)
        labels = np.array([0, 0, 1, 1, 2, 2])

        optimized = optimizer.optimize(embeddings, labels)

        # With well-separated clusters in different directions,
        # labels should be preserved (no merging should occur)
        assert np.array_equal(optimized, labels)

    def test_merge_similar_clusters(self):
        """Two clusters with nearly identical centroids get merged into one."""
        optimizer = IterativeOptimizer(DEFAULT_CONFIG)

        # Two clusters that are very close together
        embeddings = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.001, 1.001],
            [1.001, 1.001],
        ], dtype=np.float64)
        labels = np.array([0, 0, 1, 1])

        centroids = optimizer._compute_centroids(embeddings, labels)
        merged_labels = optimizer._merge_similar_clusters(embeddings, labels, centroids)

        # All documents should end up in the same cluster
        unique_labels = set(merged_labels)
        assert len(unique_labels) == 1

    def test_optimize_preserves_structure(self):
        """Clear cluster structure, labels should be mostly preserved."""
        optimizer = IterativeOptimizer(DEFAULT_CONFIG)

        # Well-separated clusters
        rng = np.random.RandomState(42)
        n_per_cluster = 10

        cluster_0 = rng.randn(n_per_cluster, 10) + np.array([0.0] * 10)
        cluster_1 = rng.randn(n_per_cluster, 10) + np.array([5.0] * 10)
        cluster_2 = rng.randn(n_per_cluster, 10) + np.array([10.0] * 10)

        embeddings = np.vstack([cluster_0, cluster_1, cluster_2])
        labels = np.array(
            [0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster
        )

        optimized = optimizer.optimize(embeddings, labels)

        # The number of unique labels should still be 3 (or at least close)
        n_unique = len(set(optimized))
        assert n_unique >= 2  # at minimum, structure is largely preserved

        # Count label changes
        changes = np.sum(optimized != labels)
        # Most labels should be preserved
        assert changes <= len(labels) // 2  # at most half change

    def test_centroids_ignore_outliers(self):
        """Centroid computation ignores label -1 (outliers)."""
        optimizer = IterativeOptimizer(DEFAULT_CONFIG)

        embeddings = np.array([
            [0.0, 0.0],
            [0.2, 0.2],
            [100.0, 100.0],  # outlier
            [10.0, 10.0],
            [10.2, 10.2],
        ], dtype=np.float64)
        labels = np.array([0, 0, -1, 1, 1])

        centroids = optimizer._compute_centroids(embeddings, labels)

        # Outlier (-1) should not be in centroids
        assert -1 not in centroids
        assert set(centroids.keys()) == {0, 1}
        assert np.allclose(centroids[0], [0.1, 0.1])
        assert np.allclose(centroids[1], [10.1, 10.1])
