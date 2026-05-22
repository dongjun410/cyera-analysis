"""
聚类质量评估模块
- Silhouette Score（簇间分离度）
- Davies-Bouldin Index（越低越好）
- Calinski-Harabasz Index（越高越好）
- 稳定性评估（多次运行 ARI）
"""

import numpy as np
import logging
from typing import Dict, Any

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

logger = logging.getLogger(__name__)


class QualityEvaluator:

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.metrics = config.get("metrics", ["silhouette", "davies_bouldin", "calinski_harabasz"])

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        评估聚类质量，返回各项指标。
        """
        if not self.enabled:
            return {}

        # 过滤离群点
        valid_mask = labels != -1
        valid_embs = embeddings[valid_mask]
        valid_labels = labels[valid_mask]

        if len(set(valid_labels)) < 2:
            logger.warning("有效簇少于 2 个，无法评估质量")
            return {"error": "fewer_than_2_clusters"}

        results = {}

        if "silhouette" in self.metrics:
            score = silhouette_score(
                valid_embs, valid_labels, metric='cosine',
                sample_size=min(10000, len(valid_embs)),
            )
            results["silhouette_score"] = round(score, 4)
            logger.info(f"  Silhouette Score: {score:.4f} (越接近1越好)")

        if "davies_bouldin" in self.metrics:
            score = davies_bouldin_score(valid_embs, valid_labels)
            results["davies_bouldin_index"] = round(score, 4)
            logger.info(f"  Davies-Bouldin Index: {score:.4f} (越低越好)")

        if "calinski_harabasz" in self.metrics:
            score = calinski_harabasz_score(valid_embs, valid_labels)
            results["calinski_harabasz_index"] = round(score, 2)
            logger.info(f"  Calinski-Harabasz Index: {score:.2f} (越高越好)")

        # 基础统计
        unique, counts = np.unique(valid_labels, return_counts=True)
        results["num_clusters"] = len(unique)
        results["num_outliers"] = int((labels == -1).sum())
        results["cluster_size_mean"] = round(float(counts.mean()), 1)
        results["cluster_size_std"] = round(float(counts.std()), 1)
        results["cluster_size_min"] = int(counts.min())
        results["cluster_size_max"] = int(counts.max())

        return results
