"""
两阶段聚类引擎（V2 核心模块）
- Stage A: KMeans 粗聚类 + 自动选 K（Silhouette Score）
- Stage B: 簇内细分裂（基于内聚度阈值）
- Stage C: 小簇处理（保留高内聚 / 合并低内聚 / 标记待审核）
- 替代 V1 不稳定的 UMAP + HDBSCAN
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ClusteringEngine:

    def __init__(self, config: dict):
        self.method = config.get("method", "kmeans")
        self.auto_k = config.get("auto_k", True)
        self.k_range = config.get("k_range", [10, 200])
        self.k_step = config.get("k_step", 5)
        self.random_state = config.get("random_state", 42)

        self.split_enabled = config.get("split_enabled", True)
        self.split_threshold = config.get("split_threshold", 0.65)
        self.min_cluster_size = config.get("min_cluster_size", 2)

        self.small_cluster_threshold = config.get("small_cluster_threshold", 5)
        self.small_cluster_min_coherence = config.get("small_cluster_min_coherence", 0.80)

    # ── Stage A: 粗聚类 ──────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        主入口：返回每个文档的 cluster_id 数组
        """
        n_docs = embeddings.shape[0]
        logger.info(f"开始聚类: {n_docs} 篇文档, 维度={embeddings.shape[1]}")

        # Stage A: 粗聚类
        if self.auto_k:
            best_k = self._find_optimal_k(embeddings)
        else:
            best_k = self.k_range[0]

        labels = self._run_kmeans(embeddings, best_k)
        logger.info(f"Stage A 粗聚类完成: K={best_k}")

        # Stage B: 簇内细分裂
        if self.split_enabled:
            labels = self._stage_b_split(embeddings, labels)

        # Stage C: 小簇处理
        labels = self._stage_c_small_clusters(embeddings, labels)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        logger.info(f"聚类完成: {n_clusters} 个簇, {n_outliers} 个离群点")

        return labels

    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        使用 Silhouette Score 自动选择最优 K。
        在 k_range 内搜索，选择得分最高的 K。
        """
        n_docs = embeddings.shape[0]
        k_min = max(self.k_range[0], 2)
        k_max = min(self.k_range[1], n_docs // 2)

        best_k = k_min
        best_score = -1
        scores = {}

        candidates = list(range(k_min, k_max + 1, self.k_step))
        logger.info(f"自动选 K: 搜索范围 [{k_min}, {k_max}], 步长={self.k_step}")

        for k in candidates:
            labels = self._run_kmeans(embeddings, k)
            # 检查是否所有簇都有数据
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue

            score = silhouette_score(embeddings, labels, metric='cosine',
                                     sample_size=min(10000, n_docs))
            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"最优 K={best_k}, Silhouette={best_score:.4f}")

        # 细化搜索：在最优 K 附近以步长 1 精搜
        fine_range = range(max(2, best_k - self.k_step), best_k + self.k_step + 1)
        for k in fine_range:
            if k in scores:
                continue
            labels = self._run_kmeans(embeddings, k)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric='cosine',
                                     sample_size=min(10000, n_docs))
            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"精搜后最优 K={best_k}, Silhouette={best_score:.4f}")
        return best_k

    def _run_kmeans(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        """运行 KMeans"""
        km = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        return km.fit_predict(embeddings)

    # ── Stage B: 簇内细分裂 ──────────────────────────────────

    def _stage_b_split(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        对内聚度低的大簇做层次聚类二次分裂。
        """
        new_labels = labels.copy()
        next_id = labels.max() + 1

        unique_clusters = sorted(set(labels))
        split_count = 0

        for cid in unique_clusters:
            mask = labels == cid
            cluster_size = mask.sum()

            if cluster_size < 2 * self.min_cluster_size:
                continue  # 太小不分裂

            cluster_embs = embeddings[mask]
            coherence = self._compute_coherence(cluster_embs)

            if coherence < self.split_threshold:
                # 用层次聚类分裂
                n_sub = min(cluster_size // self.min_cluster_size, 5)
                if n_sub < 2:
                    continue

                sub_labels = AgglomerativeClustering(
                    n_clusters=n_sub,
                    metric='cosine',
                    linkage='average',
                ).fit_predict(cluster_embs)

                indices = np.where(mask)[0]
                for i, idx in enumerate(indices):
                    if sub_labels[i] == 0:
                        pass  # 保持原 cluster_id
                    else:
                        new_labels[idx] = next_id + sub_labels[i] - 1

                next_id += n_sub - 1
                split_count += 1

        if split_count > 0:
            # 重新编号使 ID 连续
            new_labels = self._renumber_labels(new_labels)

        logger.info(f"Stage B 分裂了 {split_count} 个低内聚簇")
        return new_labels

    # ── Stage C: 小簇处理 ────────────────────────────────────

    def _stage_c_small_clusters(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Small cluster handling:
        - High-coherence small clusters: keep (may contain critical minority docs)
        - Low-coherence small clusters: merge to nearest large cluster by vector distance
        """
        new_labels = labels.copy()
        unique, counts = np.unique(labels, return_counts=True)

        # 计算每个大簇的中心
        large_clusters = {cid: embeddings[labels == cid].mean(axis=0)
                          for cid, cnt in zip(unique, counts)
                          if cnt >= self.small_cluster_threshold}

        merge_count = 0
        keep_count = 0

        for cid, cnt in zip(unique, counts):
            if cnt >= self.small_cluster_threshold:
                continue  # 不是小簇

            mask = labels == cid
            cluster_embs = embeddings[mask]
            coherence = self._compute_coherence(cluster_embs) if cnt >= 2 else 1.0

            if coherence >= self.small_cluster_min_coherence:
                # 高内聚：保留
                keep_count += 1
            elif large_clusters:
                # 低内聚：合并到最近的大簇
                centroid = cluster_embs.mean(axis=0)
                best_cid = min(
                    large_clusters.keys(),
                    key=lambda c: 1 - cosine_similarity(
                        centroid.reshape(1, -1),
                        large_clusters[c].reshape(1, -1)
                    )[0][0]
                )
                new_labels[mask] = best_cid
                merge_count += 1
            else:
                # 没有大簇可合并，标记为离群
                new_labels[mask] = -1

        logger.info(
            f"Stage C 小簇处理: 保留 {keep_count} 个, "
            f"合并 {merge_count} 个到大簇"
        )
        return new_labels

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _compute_coherence(embeddings: np.ndarray) -> float:
        """计算簇内平均余弦相似度"""
        if len(embeddings) < 2:
            return 1.0
        sim_matrix = cosine_similarity(embeddings)
        # 取上三角（不含对角线）的平均值
        n = len(sim_matrix)
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(upper_tri.mean())

    @staticmethod
    def _renumber_labels(labels: np.ndarray) -> np.ndarray:
        """重新编号使 cluster ID 从 0 开始连续"""
        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        mapping[-1] = -1
        return np.array([mapping[l] for l in labels])
