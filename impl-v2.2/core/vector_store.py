"""
Elasticsearch 向量存储
- HNSW kNN（替代 V1 的 bbq_disk）
- 增量 upsert（替代 V1 的 delete + recreate）
"""

import numpy as np
import logging
from typing import List, Dict, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self, config: dict):
        host = config.get("host", "localhost")
        port = config.get("port", 9200)
        self.es = Elasticsearch([f"http://{host}:{port}"])
        self.index_name = config.get("index_name", "doc_clusters_v2")
        self.vector_dims = config.get("vector_dims", 2048)
        self.similarity = config.get("similarity", "cosine")

    def ensure_index(self):
        """创建索引（如果不存在）。不再每次删除重建。"""
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"索引 {self.index_name} 已存在，跳过创建")
            return

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.vector_dims,
                        "index": True,
                        "similarity": self.similarity,
                    },
                    "cluster_id": {"type": "integer"},
                    "cluster_label": {"type": "keyword"},
                    "classification_source": {"type": "keyword"},
                    "pii_types": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"索引 {self.index_name} 创建完成")

    def upsert_documents(
        self,
        documents,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_labels: Dict[int, str] = None,
    ):
        """增量 upsert 文档（替代 V1 的全量 delete + create）"""
        cluster_labels = cluster_labels or {}
        actions = []

        for i, doc in enumerate(documents):
            cid = int(labels[i])
            actions.append({
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {
                    "doc_id": doc.id,
                    "title": doc.title,
                    "content": doc.raw_content[:10000],
                    "embedding": embeddings[i].tolist(),
                    "cluster_id": cid,
                    "cluster_label": cluster_labels.get(cid, ""),
                    "classification_source": doc.classification_source,
                    "pii_types": doc.pii_types_found,
                    "metadata": doc.metadata,
                },
            })

        if actions:
            success, errors = bulk(self.es, actions, stats_only=True)
            logger.info(f"索引完成: 成功 {success} 篇")
            if errors:
                logger.warning(f"索引错误: {errors}")

    def search_similar(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """kNN 相似性搜索"""
        response = self.es.search(
            index=self.index_name,
            body={
                "size": k,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector.tolist(),
                    "k": k,
                    "num_candidates": k * 10,
                },
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    def get_cluster_documents(self, cluster_id: int, size: int = 1000) -> List[Dict]:
        """按簇 ID 查询文档"""
        response = self.es.search(
            index=self.index_name,
            body={"query": {"term": {"cluster_id": cluster_id}}, "size": size},
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
