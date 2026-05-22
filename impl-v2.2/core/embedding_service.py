"""
向量化服务（双通道架构 — Channel 1）
- Channel 1: 原始文本 → 分块嵌入 + 均匀加权聚合 → 语义向量
- Channel 2 的结构特征向量在主流程中与语义向量拼接
- 原始文本不做任何修改
"""

import os
import numpy as np
import logging
from typing import List, Optional

from models.schemas import ProcessedDocument

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "BAAI/bge-m3")
        self.local_path = config.get("local_model_path", "")
        self.device = config.get("device", "cuda")
        self.batch_size = config.get("batch_size", 32)
        self.max_token_length = config.get("max_token_length", 8192)
        self.model = None
        self.dim = None

    def load_model(self):
        """加载嵌入模型"""
        from sentence_transformers import SentenceTransformer

        if self.local_path and os.path.exists(self.local_path):
            self.model = SentenceTransformer(self.local_path, device=self.device)
        else:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"嵌入模型加载完成: dim={self.dim}, device={self.device}")

    def encode_documents(
        self,
        documents: List[ProcessedDocument],
    ) -> np.ndarray:
        """
        对文档列表生成嵌入向量。
        使用分块嵌入 + 均匀加权聚合，解决 V1 的两个问题：
          1. 不再截断到 8192 字符
          2. 不再使用 1/(i+1) 递减权重
        """
        if not self.model:
            self.load_model()

        embeddings = []
        for doc in documents:
            emb = self._encode_single_document(doc)
            embeddings.append(emb)

        return np.vstack(embeddings)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """对纯文本列表做嵌入"""
        if not self.model:
            self.load_model()
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )

    def encode_single_text(self, text: str) -> np.ndarray:
        """对单条文本做嵌入"""
        if not self.model:
            self.load_model()
        return self.model.encode(text, normalize_embeddings=True)

    def _encode_single_document(self, doc: ProcessedDocument) -> np.ndarray:
        """
        Single document embedding (Channel 1: semantic vector).
        Uses the original raw text — no text modification.
        """
        source_text = doc.raw_content
        blocks = doc.content_blocks

        if not blocks or len(blocks) == 0:
            # 如果没有预切分的 blocks，直接对整篇文本编码
            return self.model.encode(
                source_text[:self.max_token_length],
                normalize_embeddings=True,
            )

        # 基于抽象化文本重新分块（如果 content_blocks 是基于 raw_content 的）
        # 这里使用已有的 blocks，但确保不超过模型最大长度
        truncated_blocks = [b[:self.max_token_length] for b in blocks]

        # 批量编码所有块
        block_embeddings = self.model.encode(
            truncated_blocks,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )

        # 均匀加权平均（修复 V1 的 1/(i+1) 问题）
        doc_embedding = np.mean(block_embeddings, axis=0)

        # L2 归一化
        norm = np.linalg.norm(doc_embedding)
        if norm > 0:
            doc_embedding = doc_embedding / norm

        return doc_embedding

    def get_dimension(self) -> int:
        if not self.model:
            self.load_model()
        return self.dim
