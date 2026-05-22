"""
结构特征提取模块 — 双通道架构的 Channel 2

核心思路：
  不修改原始文本，而是从文本中提取结构化数值特征向量。
  该向量与语义嵌入向量拼接后做联合聚类，使聚类同时考虑
  "语义相似性"和"文档结构相似性"。

  两份金额不同但结构相同的合同，它们的结构特征向量会很接近
  （都有高 amount_count、高 date_count、相似的段落结构），
  从而在联合聚类中更容易被分到同一个簇。
"""

import re
import math
import logging
import numpy as np
from typing import List, Dict

from models.schemas import ProcessedDocument

logger = logging.getLogger(__name__)


# ── Pattern registry (for counting, NOT replacing) ────────────

COUNTING_PATTERNS = {
    "email_count": re.compile(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    ),
    "phone_count": re.compile(
        r'(?:\+\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}'
    ),
    "credit_card_count": re.compile(
        r'(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2})|35(?:2[89]|[3-8]\d))'
        r'[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{3,4}'
    ),
    "ssn_count": re.compile(r'\d{3}[\s\-]\d{2}[\s\-]\d{4}'),
    "iban_count": re.compile(
        r'[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){2,7}[\dA-Z]{1,4}'
    ),
    "url_count": re.compile(r'https?://[^\s<>"{}|\\^\[\]`]+'),
    "ip_count": re.compile(
        r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
    ),
    "date_count": re.compile(
        r'\d{4}-\d{2}-\d{2}|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'
    ),
    "amount_count": re.compile(
        r'(?:USD|EUR|GBP|JPY|VND|KRW|\$|€|£|¥|₫|₩|฿|₹)\s*[\d,]+(?:\.\d{1,2})?',
        re.IGNORECASE,
    ),
    "long_number_count": re.compile(r'(?<!\d)\d{6,}(?!\d)'),
}


class StructureFeatureExtractor:
    """
    Extracts a numerical feature vector representing document structure.
    This vector forms Channel 2 of the dual-channel embedding architecture.
    The original text is NEVER modified — features are computed by counting
    and measuring, not by replacing.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.normalize = config.get("normalize", True)

    def extract_features(self, doc: ProcessedDocument) -> np.ndarray:
        """
        Extract structure feature vector from a single document.
        Returns a 1D numpy array of numerical features.
        """
        if not self.enabled:
            return np.zeros(self.feature_dim())

        text = doc.raw_content
        features = []

        # ── PII type distribution (pattern match counts) ─────
        for name, pattern in COUNTING_PATTERNS.items():
            count = len(pattern.findall(text))
            features.append(count)

        # ── Document structure features ──────────────────────
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        sentences = re.split(r'[.!?。！？]\s+', text)

        features.append(len(paragraphs))                     # paragraph_count
        features.append(len(sentences))                       # sentence_count
        features.append(len(text))                            # char_count
        features.append(len(text.split()))                    # word_count
        features.append(                                      # avg_paragraph_length
            np.mean([len(p) for p in paragraphs]) if paragraphs else 0
        )
        features.append(                                      # has_tables (heuristic)
            1.0 if re.search(r'\|.*\|.*\|', text) else 0.0
        )
        features.append(                                      # header_count
            len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        )

        # ── File metadata features ───────────────────────────
        meta = doc.metadata
        features.append(meta.get('file_size_kb', 0))          # file_size
        features.append(meta.get('path_depth', 0))            # path_depth

        # ── Convert to numpy ─────────────────────────────────
        vec = np.array(features, dtype=np.float32)

        if self.normalize and np.max(np.abs(vec)) > 0:
            # Log-scale for counts (avoids large count values dominating)
            vec = np.sign(vec) * np.log1p(np.abs(vec))
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    def extract_batch(self, documents: List[ProcessedDocument]) -> np.ndarray:
        """Extract structure feature vectors for a batch of documents."""
        vectors = [self.extract_features(doc) for doc in documents]
        result = np.vstack(vectors)
        logger.info(
            f"Structure features extracted: {len(documents)} docs, "
            f"dim={result.shape[1]}"
        )
        return result

    @staticmethod
    def feature_dim() -> int:
        """Return the dimensionality of the structure feature vector."""
        return len(COUNTING_PATTERNS) + 7 + 2  # patterns + structure + metadata
