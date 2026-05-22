"""
Iterative clustering optimizer

Each iteration performs three operations:
  1. Compute cluster centroids and feature summaries
  2. Merge over-fragmented clusters (centroid similarity > threshold)
  3. Reassign borderline documents (negative silhouette samples)
Convergence: ARI change < threshold or max iterations reached.
"""

import numpy as np
import logging
from typing import List, Dict

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class IterativeOptimizer:

    def __init__(self, config: dict):
        self.max_iterations = config.get("max_iterations", 3)
        self.convergence_threshold = config.get("convergence_threshold", 0.02)
        self.merge_similarity_threshold = 0.85  # 簇间相似度超过此值则合并

    def optimize(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        keywords_per_cluster: Dict[int, List[str]] = None,
    ) -> np.ndarray:
        """
        迭代优化聚类结果。

        参数:
          embeddings: 文档嵌入矩阵
          labels: 初始聚类标签
          keywords_per_cluster: 每个簇的关键词（可选）

        返回:
          优化后的标签
        """
        current_labels = labels.copy()

        for iteration in range(self.max_iterations):
            logger.info(f"迭代优化 Round {iteration + 1}/{self.max_iterations}")

            # Step 1: 计算簇中心
            centroids = self._compute_centroids(embeddings, current_labels)

            # Step 2: 合并过于相似的簇
            new_labels = self._merge_similar_clusters(
                embeddings, current_labels, centroids
            )

            # Step 3: 重新分配边缘文档
            new_labels = self._reassign_borderline_docs(
                embeddings, new_labels, centroids
            )

            # 检查收敛
            ari = adjusted_rand_score(current_labels, new_labels)
            change_rate = 1.0 - ari
            logger.info(f"  变化率: {change_rate:.4f} (ARI={ari:.4f})")

            if change_rate < self.convergence_threshold:
                logger.info(f"  已收敛，停止迭代")
                current_labels = new_labels
                break

            current_labels = new_labels

        return current_labels

    def _compute_centroids(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """计算每个簇的中心"""
        centroids = {}
        for cid in set(labels):
            if cid == -1:
                continue
            mask = labels == cid
            centroids[cid] = embeddings[mask].mean(axis=0)
        return centroids

    def _merge_similar_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """合并中心过于接近的簇"""
        new_labels = labels.copy()
        cids = sorted(centroids.keys())
        merged = set()
        merge_count = 0

        for i, cid_a in enumerate(cids):
            if cid_a in merged:
                continue
            for cid_b in cids[i + 1:]:
                if cid_b in merged:
                    continue
                sim = cosine_similarity(
                    centroids[cid_a].reshape(1, -1),
                    centroids[cid_b].reshape(1, -1),
                )[0][0]
                if sim > self.merge_similarity_threshold:
                    # 将 B 合并到 A
                    new_labels[new_labels == cid_b] = cid_a
                    merged.add(cid_b)
                    merge_count += 1

        if merge_count > 0:
            logger.info(f"  合并了 {merge_count} 对相似簇")
        return new_labels

    def _reassign_borderline_docs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """重新分配边缘文档（silhouette < 0 的文档）"""
        from sklearn.metrics import silhouette_samples

        new_labels = labels.copy()
        valid_mask = labels != -1

        if len(set(labels[valid_mask])) < 2:
            return new_labels

        sample_scores = silhouette_samples(
            embeddings[valid_mask], labels[valid_mask], metric='cosine'
        )

        valid_indices = np.where(valid_mask)[0]
        reassign_count = 0

        for i, idx in enumerate(valid_indices):
            if sample_scores[i] < 0:
                # 该文档在当前簇中不合适，重新分配到最近的簇
                doc_emb = embeddings[idx].reshape(1, -1)
                best_cid = labels[idx]
                best_sim = -1

                for cid, centroid in centroids.items():
                    if cid == -1:
                        continue
                    sim = cosine_similarity(doc_emb, centroid.reshape(1, -1))[0][0]
                    if sim > best_sim:
                        best_sim = sim
                        best_cid = cid

                if best_cid != labels[idx]:
                    new_labels[idx] = best_cid
                    reassign_count += 1

        if reassign_count > 0:
            logger.info(f"  重新分配了 {reassign_count} 篇边缘文档")
        return new_labels
