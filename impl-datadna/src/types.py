from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PIIFeature:
    """Single PII entity detected by Tier 0 or DeBERTa."""

    entity_type: str  # "SSN", "EMAIL", "CREDIT_CARD", etc.
    span: tuple[int, int]  # [start, end] character offsets
    confidence: float  # 0.0-1.0
    context_flag: str  # "clean" | "penalty_term_present" | "boost_term_present"


@dataclass
class PIIFeatureVector:
    """Per-document PII feature summary (Tier 0 output, NOT final label)."""

    doc_id: str
    pii_features: list[PIIFeature] = field(default_factory=list)
    pii_type_distribution: dict[str, int] = field(default_factory=dict)
    has_high_conf_pii: bool = False
    has_penalty_terms: bool = False


@dataclass
class StructuralFeatures:
    """Stage A structural fingerprint for a document."""

    file_type: str  # ".pdf", ".docx", ".json", etc.
    file_size_quantile: int  # log-bucketed
    page_count: int = 0
    paragraph_count: int = 0
    table_count: int = 0
    has_images: bool = False
    header_pattern: str = ""  # regex-matched header/footer signature
    json_schema_signature: str = ""  # if JSON/XML
    path_depth: int = 0
    extra: dict = field(default_factory=dict)  # extensible


@dataclass
class Document:
    """Central document representation flowing through all tiers."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    structural_features: StructuralFeatures | None = None
    pii_features: PIIFeatureVector | None = None
    embedding: np.ndarray | None = None
    cluster_id: str | None = None
    label: str | None = None
    label_confidence: float = 0.0
    label_method: str | None = None  # "known_match" | "llm_tier2" | "llm_tier3" | "distilled"


@dataclass
class ClusterInfo:
    """A cluster of documents after Tier 1."""

    cluster_id: str
    doc_ids: list[str]
    structural_bucket: str  # Stage A bucket hash
    cluster_radius: float  # max cosine distance from centroid
    representative_docs: list[str]  # doc_ids closest to centroid
    tfidf_keywords: list[str]  # top-15 terms
    pii_distribution: dict[str, int]
    language_distribution: dict[str, int]
    centroid_embedding: np.ndarray | None = None  # mean of member embeddings
    label: str | None = None
    label_confidence: float = 0.0
    needs_tier3: bool = False


@dataclass
class KnownType:
    """Registered document type in the known type library."""

    type_id: str
    type_name: str
    description: str
    structural_signature: str = ""  # hash of structural pattern
    tfidf_keywords: list[str] = field(default_factory=list)  # representative keywords for TF-IDF overlap
    pii_distribution: dict[str, int] = field(default_factory=dict)  # typical PII type distribution
    semantic_centroid: np.ndarray | None = None  # mean embedding of type exemplars
    detection_rules: list[str] = field(default_factory=list)  # suggested regex for Tier 0 extension
    status: str = "active"  # "active" | "pending_review"
    sample_count: int = 0


@dataclass
class ClassificationResult:
    doc_id: str
    label: str
    confidence: float
    method: str  # "known_match" | "deberta_ner" | "llm_tier2" | "llm_tier3" | "distilled"
    is_new_type: bool = False
    needs_manual_review: bool = False
    rationale: str = ""


@dataclass
class MatchResult:
    """Output of KnownTypeMatcher.match()."""

    matched_type: KnownType | None = None
    score: float = 0.0  # 0.0-1.0 weighted composite
    method: str = "unknown"  # "known_match" | "llm_confirm" | "unknown"
    match_details: dict = field(default_factory=dict)  # per-signal scores


@dataclass
class AssignmentResult:
    """Output of IncrementalAssigner.assign()."""

    doc_id: str
    assigned_cluster_id: str | None = None
    is_outlier: bool = False
    outlier_reason: str = ""  # "intra_bucket_outlier" | "new_structure_candidate" | ""
    needs_reclustering: bool = False


@dataclass
class PipelineResult:
    """Aggregate result for a batch of documents."""

    results: list[ClassificationResult] = field(default_factory=list)
    stats: dict = field(default_factory=dict)  # timing, tier counts, trigger rates
