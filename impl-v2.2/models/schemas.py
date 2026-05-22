"""统一数据结构定义"""

import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ClassificationSource(Enum):
    """分类来源"""
    PII_REGEX = "pii_regex"           # 正则预分类
    CLUSTERING = "clustering"          # 聚类分配
    PROPAGATION = "propagation"        # 分类传播
    LLM = "llm"                        # LLM 直接分类
    MANUAL = "manual"                  # 人工标注


class PIIType(Enum):
    """PII 类型（国际化）"""
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"        # Visa/MC/Amex/JCB/UnionPay
    SSN = "ssn"                        # US SSN
    PASSPORT = "passport"              # 多国护照号
    NATIONAL_ID = "national_id"        # 各国国民 ID（日本 My Number 等）
    IBAN = "iban"                       # 国际银行帐号
    IP_ADDRESS = "ip_address"
    URL = "url"
    DATE = "date"
    AMOUNT = "amount"
    PERSON_NAME = "person_name"
    LOCATION = "location"


@dataclass
class PIIDetection:
    """单条 PII 检测结果"""
    pii_type: PIIType
    value: str                         # 原始值（脱敏后存储）
    position: int                      # 在文本中的起始位置
    confidence: float = 1.0


@dataclass
class ProcessedDocument:
    """文档处理结果"""
    id: str
    original_path: str
    title: str

    # 内容
    raw_content: str                   # 原始提取文本
    abstracted_content: str = ""       # (deprecated, kept for compatibility)
    content_blocks: List[str] = field(default_factory=list)

    # PII
    pii_detections: List[PIIDetection] = field(default_factory=list)
    pii_types_found: List[str] = field(default_factory=list)

    # 分类
    preclassified: bool = False        # 是否被预分类器直接标记
    preclassification_label: str = ""

    # 聚类
    cluster_id: int = -1
    cluster_probability: float = 0.0
    classification_source: str = ""

    # 元信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    file_type: str = ""

    @staticmethod
    def generate_id(file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()[:20]


@dataclass
class ClusterInfo:
    """单个簇的信息"""
    cluster_id: int
    size: int
    keywords: List[str]                # TF-IDF 关键词
    llm_label: str = ""                # LLM 生成的业务标签
    llm_description: str = ""          # LLM 生成的描述
    centroid: Any = None               # 簇中心向量
    coherence: float = 0.0             # 簇内一致性
    representative_doc_ids: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)


@dataclass
class ClusteringResult:
    """聚类整体结果"""
    total_documents: int
    preclassified_documents: int       # 被预分类器直接标记的
    clustered_documents: int           # 进入聚类的
    num_clusters: int
    num_outliers: int
    clusters: List[ClusterInfo] = field(default_factory=list)

    # 质量指标
    silhouette_score: float = 0.0
    davies_bouldin_index: float = 0.0
    calinski_harabasz_index: float = 0.0

    # 文档映射
    doc_cluster_map: Dict[str, Dict] = field(default_factory=dict)
