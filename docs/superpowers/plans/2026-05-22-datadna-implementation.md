# DataDNA Classification Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Tier 0→1→2→3 hierarchical document classification pipeline based on `docs/superpowers/specs/2026-05-22-datadna-optimized-design.md`.

**Architecture:** Four-tier serial pipeline. Tier 0 extracts PII features via regex rules. Tier 1 clusters documents in two stages (structural hash + FAISS semantic refinement). Tier 2 classifies cluster representatives using known-type matching, DeBERTa NER, and Mistral-7B 4-bit LLM. Tier 3 validates high-risk/low-confidence documents with Mistral-7B INT8. Type discovery via outlier accumulation + periodic reclustering. SetFit distillation compresses to ~2ms/doc.

**Tech Stack:** Python 3.10+, BGE-M3, DeBERTa-v3-base, Mistral-7B (Ollama), FAISS, Presidio, SetFit, Elasticsearch

**Methodology:** Strict TDD — every source file has a corresponding test file written first. Red → Green → Commit per task.

---

## File Structure

```
impl-datadna/
├── src/
│   ├── types.py                 [CREATE] All shared dataclasses
│   ├── tier0/
│   │   ├── patterns.py          [CREATE] 50+ PII regex patterns + context terms
│   │   └── engine.py            [CREATE] PII feature extractor (Presidio-based)
│   ├── tier1/
│   │   ├── structural.py        [CREATE] Stage A: structural feature hash bucketing
│   │   ├── semantic.py          [CREATE] Stage B: FAISS IVF + HDBSCAN refinement
│   │   └── incremental.py       [CREATE] O(1) hash + O(log N) FAISS doc assignment
│   ├── tier2/
│   │   ├── matching.py          [CREATE] Known type 3-signal weighted matching
│   │   ├── classifier.py        [CREATE] Cluster-level orchestration (match→NER→LLM)
│   │   └── propagation.py       [CREATE] Label propagation + sampling verification
│   ├── tier3/
│   │   └── quality_gate.py      [CREATE] 5-trigger LLM quality gate (INT8)
│   ├── discovery/
│   │   └── loop.py              [CREATE] Outlier buffer → recluster → type registration
│   ├── distillation/
│   │   └── trainer.py           [CREATE] SetFit teacher→student + incremental retrain
│   ├── embeddings/
│   │   └── bge_m3.py            [CREATE] BGE-M3 batch encoder
│   ├── ner/
│   │   └── deberta.py           [CREATE] DeBERTa-v3-base token classification
│   └── llm/
│       └── client.py            [CREATE] Mistral-7B via Ollama (4-bit + INT8)
├── tests/
│   ├── conftest.py              [CREATE] Shared fixtures (sample docs, mock configs)
│   ├── test_tier0.py            [CREATE] PII detection: patterns, context, confidence
│   ├── test_tier1_structural.py [CREATE] Hash determinism, feature extraction
│   ├── test_tier1_semantic.py   [CREATE] FAISS index, HDBSCAN, homogeneity check
│   ├── test_tier2_matching.py   [CREATE] 3-signal scoring, thresholds
│   ├── test_tier2_classifier.py [CREATE] Orchestrator flow, cold start path
│   ├── test_tier3.py            [CREATE] Trigger logic, sigma thresholds
│   ├── test_discovery.py        [CREATE] Outlier buffer, type registration
│   ├── test_embeddings.py       [CREATE] Embedding shape, batch consistency
│   ├── test_ner.py              [CREATE] Entity extraction, span correctness
│   ├── test_llm_client.py       [CREATE] API call, retry, JSON parsing
│   └── test_integration.py      [CREATE] End-to-end: document → label
├── main.py                      [MODIFY] Full pipeline orchestration
├── incremental.py               [MODIFY] Single-document incremental path
├── config.yaml                  (exists)
└── requirements.txt             (exists)
```

---

## Dependency Graph & Execution Order

```
Group A (parallel, no cross-deps):  Task 1 (types), Task 2 (patterns), Task 4 (embeddings),
                                    Task 5 (ner), Task 6 (llm client), Task 10 (matching),
                                    Task 12 (propagation), Task 15 (distillation)

Group B (depends on A):            Task 3 (tier0 engine, needs 1,2),
                                    Task 7 (structural, needs 1)

Group C (depends on A+B):          Task 8 (semantic, needs 1,4,7),
                                    Task 11 (classifier, needs 1,5,6,10)

Group D (depends on C):            Task 9 (incremental, needs 7,8),
                                    Task 13 (quality gate, needs 1,6),
                                    Task 14 (discovery, needs 1,7,8,11)

Group E (depends on D):            Task 16 (main.py), Task 17 (incremental.py)

Group F (final):                   Task 18 (integration tests), Task 19 (e2e run)
```

---

## Shared Contracts (implemented in Task 1: `src/types.py`)

All modules depend on these dataclasses. Defined first, imported everywhere.

```python
@dataclass
class PIIFeature:
    """Single PII entity detected by Tier 0 or DeBERTa."""
    entity_type: str          # "SSN", "EMAIL", "CREDIT_CARD", etc.
    span: tuple[int, int]     # [start, end] character offsets
    confidence: float         # 0.0-1.0
    context_flag: str         # "clean" | "penalty_term_present" | "boost_term_present"

@dataclass
class PIIFeatureVector:
    """Per-document PII feature summary (Tier 0 output, NOT final label)."""
    doc_id: str
    pii_features: list[PIIFeature]
    pii_type_distribution: dict[str, int]  # {"SSN": 1, "EMAIL": 2}
    has_high_conf_pii: bool
    has_penalty_terms: bool

@dataclass
class StructuralFeatures:
    """Stage A structural fingerprint for a document."""
    file_type: str            # ".pdf", ".docx", ".json", etc.
    file_size_quantile: int   # log-bucketed
    page_count: int
    paragraph_count: int
    table_count: int
    has_images: bool
    header_pattern: str       # regex-matched header/footer signature
    json_schema_signature: str  # if JSON/XML
    path_depth: int
    extra: dict               # extensible

@dataclass
class Document:
    """Central document representation flowing through all tiers."""
    doc_id: str
    text: str
    metadata: dict
    structural_features: StructuralFeatures | None = None
    pii_features: PIIFeatureVector | None = None
    embedding: np.ndarray | None = None
    cluster_id: str | None = None
    label: str | None = None
    label_confidence: float = 0.0
    label_method: str | None = None   # "known_match" | "llm_tier2" | "llm_tier3" | "distilled"

@dataclass
class ClusterInfo:
    """A cluster of documents after Tier 1."""
    cluster_id: str
    doc_ids: list[str]
    structural_bucket: str     # Stage A bucket hash
    centroid_embedding: np.ndarray | None  # mean of member embeddings
    cluster_radius: float      # max cosine distance from centroid
    representative_docs: list[str]  # doc_ids closest to centroid
    tfidf_keywords: list[str]  # top-15 terms
    pii_distribution: dict[str, int]
    language_distribution: dict[str, int]
    label: str | None = None
    label_confidence: float = 0.0
    needs_tier3: bool = False

@dataclass
class KnownType:
    """Registered document type in the known type library."""
    type_id: str
    type_name: str
    description: str
    structural_signature: str     # hash of structural pattern
    semantic_centroid: np.ndarray | None  # mean embedding of type exemplars
    detection_rules: list[str]    # suggested regex for Tier 0 extension
    status: str                   # "active" | "pending_review"
    sample_count: int = 0

@dataclass
class ClassificationResult:
    doc_id: str
    label: str
    confidence: float
    method: str               # "known_match" | "deberta_ner" | "llm_tier2" | "llm_tier3" | "distilled"
    is_new_type: bool = False
    needs_manual_review: bool = False
    rationale: str = ""

@dataclass
class MatchResult:
    """Output of KnownTypeMatcher.match()."""
    matched_type: KnownType | None
    score: float               # 0.0-1.0 weighted composite
    method: str                # "known_match" | "llm_confirm" | "unknown"
    match_details: dict        # per-signal scores

@dataclass
class AssignmentResult:
    """Output of IncrementalAssigner.assign()."""
    doc_id: str
    assigned_cluster_id: str | None
    is_outlier: bool
    outlier_reason: str        # "intra_bucket_outlier" | "new_structure_candidate" | ""
    needs_reclustering: bool

@dataclass
class PipelineResult:
    """Aggregate result for a batch of documents."""
    results: list[ClassificationResult]
    stats: dict                # timing, tier counts, trigger rates
```

---

## Tasks

### Task 1: Shared Types (`src/types.py`)

**Files:** Create `src/types.py`

**Goal:** Define all dataclasses listed in Shared Contracts above. No logic — pure data containers.

**Test:** No separate test file needed — types are validated by every downstream test.

**Key decisions:**
- Use `dataclasses.dataclass` with `field(default_factory=...)` for mutable defaults
- `np.ndarray` fields use `default=None` (embeddings loaded lazily)
- All fields have type hints

- [ ] Create `src/types.py` with all 8 dataclasses
- [ ] Verify module imports cleanly: `python -c "from src.types import Document, ClusterInfo, KnownType"`

---

### Task 2: Tier 0 Patterns (`src/tier0/patterns.py`)

**Files:** Create `src/tier0/patterns.py`

**Goal:** Define 50+ built-in PII detection patterns. Each pattern has: regex, entity_type, validation function name, context_boost_terms, context_penalty_terms.

**Interface contract:**
```python
# patterns.py exports:
BUILTIN_PATTERNS: list[dict]  # list of pattern configs
# Each pattern dict keys:
#   "entity_type": str
#   "regex": str (compiled by engine)
#   "validation": str (function name, looked up in VALIDATORS dict)
#   "context_boost_terms": list[str]
#   "context_penalty_terms": list[str]
#   "min_confidence": float

VALIDATORS: dict[str, callable]  # {"luhn": fn, "checksum": fn, ...}
```
Pattern coverage: SSN, credit card (Visa/MC/Amex/Discover), email, phone (US/CN/UK/DE), IBAN, SWIFT, passport, driver's license, IP address, MAC address, DOB, SSN (China), bank account, medical record number, NPI, DEA number, API key patterns, AWS access key, private key header, JWT token, base64, URL with credentials, etc.

- [ ] Create `src/tier0/patterns.py` with `BUILTIN_PATTERNS` and `VALIDATORS`
- [ ] Verify: `python -c "from src.tier0.patterns import BUILTIN_PATTERNS; print(len(BUILTIN_PATTERNS))"` shows >= 50

---

### Task 3: Tier 0 Engine + Tests

**Files:** Create `src/tier0/engine.py`, Create `tests/test_tier0.py`

**Goal:** PII feature extractor. Consumes document text, applies patterns, checks context windows, outputs `PIIFeatureVector`. This is feature extraction, NOT final classification.

**Interface contract:**
```python
class Tier0Engine:
    def __init__(self, config: dict, custom_patterns: list[dict] | None = None)
    def extract(self, doc_id: str, text: str) -> PIIFeatureVector
    def extract_batch(self, docs: list[tuple[str, str]]) -> list[PIIFeatureVector]
```

**Core logic:**
1. Compile all regexes (builtin + custom) at init
2. For each document, scan text with each pattern
3. For each regex match, check context window (±N chars) for boost/penalty terms
4. If penalty terms present → lower confidence, set `context_flag="penalty_term_present"` (do NOT discard)
5. If boost terms present → raise confidence, set `context_flag="boost_term_present"`
6. Apply validator function if defined (e.g., Luhn check for credit cards)
7. Build `PIIFeatureVector` with all detected entities

**Test contracts (`tests/test_tier0.py`):**
- `test_ssn_detection_clean_context` — SSN in normal text → detected with high confidence
- `test_ssn_with_penalty_term` — SSN near "test"/"sample" → lower confidence, NOT discarded, `context_flag="penalty_term_present"`
- `test_credit_card_luhn_valid` — valid CC number passes Luhn → high confidence
- `test_credit_card_luhn_invalid` — invalid CC number fails Luhn → not reported
- `test_multiple_entity_types` — doc with SSN + email + phone → all three detected
- `test_empty_document` — empty text → empty feature vector, no crash
- `test_context_window_boundary` — penalty term exactly at window boundary → correct flag
- `test_batch_extraction` — multiple docs processed correctly

- [ ] Write failing tests in `tests/test_tier0.py`
- [ ] Run: `pytest tests/test_tier0.py -v` — all 8 tests FAIL
- [ ] Implement `src/tier0/engine.py`
- [ ] Run: `pytest tests/test_tier0.py -v` — all 8 tests PASS
- [ ] Commit: `git add src/tier0/engine.py tests/test_tier0.py && git commit -m "feat: implement Tier 0 PII feature extraction engine"`

---

### Task 4: BGE-M3 Embeddings + Tests

**Files:** Create `src/embeddings/bge_m3.py`, Create `tests/test_embeddings.py`

**Goal:** Thin wrapper around `sentence-transformers` for BGE-M3. Batch encoding, normalize to unit vectors for cosine similarity.

**Interface contract:**
```python
class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda",
                 batch_size: int = 32, max_length: int = 8192)
    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray
    @property
    def dim(self) -> int  # 1024
```

**Test contracts (`tests/test_embeddings.py`):**
- `test_embedding_shape` — N documents → (N, 1024) array
- `test_embedding_normalized` — all output vectors have L2 norm ≈ 1.0 (within 1e-5)
- `test_single_document` — single text works (not just batch)
- `test_cosine_similarity_range` — two similar texts have higher cosine sim than two dissimilar texts
- `test_empty_string` — empty string → non-NaN embedding
- `test_batch_size_respected` — processes in configured batch size
- `test_dim_property` — `.dim` returns 1024

- [ ] Write failing tests in `tests/test_embeddings.py`
- [ ] Run: `pytest tests/test_embeddings.py -v` — all 7 tests FAIL
- [ ] Implement `src/embeddings/bge_m3.py`
- [ ] Run: `pytest tests/test_embeddings.py -v` — all 7 tests PASS
- [ ] Commit: `git add src/embeddings/bge_m3.py tests/test_embeddings.py && git commit -m "feat: implement BGE-M3 embedding service"`

---

### Task 5: DeBERTa NER + Tests

**Files:** Create `src/ner/deberta.py`, Create `tests/test_ner.py`

**Goal:** Encoder-only token classification for PII detection. Consumes Tier 0's `PIIFeatureVector` for low-confidence entities and does deep contextual disambiguation.

**Interface contract:**
```python
class DebertaNER:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", device: str = "cuda")
    def predict(self, text: str, pii_hints: PIIFeatureVector | None = None) -> list[PIIFeature]
    def predict_batch(self, texts: list[str],
                      pii_hints: list[PIIFeatureVector | None] | None = None) -> list[list[PIIFeature]]
```

**Core logic:**
1. Load DeBERTa-v3-base with token classification head fine-tuned for PII
2. If `pii_hints` provided, only run NER on spans where Tier 0 had low confidence — skip clean spans
3. For each predicted entity, extract span, type, confidence
4. Context-aware disambiguation: distinguish real PII from test data patterns

**Test contracts (`tests/test_ner.py`):**
- `test_entity_extraction` — known PII text → correct entity type + span
- `test_no_pii_text` — plain text with no PII → empty list
- `test_multiple_entities_same_type` — doc with 3 emails → 3 entities returned
- `test_with_pii_hints` — providing PIIFeatureVector as hints limits scan scope
- `test_span_boundaries` — entity span [start, end] accurately brackets the PII text
- `test_batch_prediction` — multiple texts processed correctly
- `test_confidence_range` — all confidence values in [0.0, 1.0]

- [ ] Write failing tests in `tests/test_ner.py`
- [ ] Run: `pytest tests/test_ner.py -v` — all 7 tests FAIL
- [ ] Implement `src/ner/deberta.py`
- [ ] Run: `pytest tests/test_ner.py -v` — all 7 tests PASS
- [ ] Commit: `git add src/ner/deberta.py tests/test_ner.py && git commit -m "feat: implement DeBERTa-v3 NER service"`

---

### Task 6: LLM Client + Tests

**Files:** Create `src/llm/client.py`, Create `tests/test_llm_client.py`

**Goal:** Mistral-7B client via Ollama's OpenAI-compatible API. Supports 4-bit (Tier 2, throughput) and INT8 (Tier 3, accuracy) modes. Structured JSON output with schema constraint. Retry with exponential backoff.

**Interface contract:**
```python
@dataclass
class LLMConfig:
    api_base: str              # "http://localhost:11434/v1"
    model: str                 # "mistral:7b"
    quantization: str          # "4bit" | "int8"
    temperature: float = 0.3
    max_tokens: int = 512
    timeout: int = 30
    max_retries: int = 2

class MistralClient:
    def __init__(self, config: LLMConfig)
    def classify(self, prompt: str, document_text: str,
                 known_types: list[str], ner_results: list[PIIFeature] | None = None,
                 output_schema: dict | None = None) -> dict
    def verify(self, prompt: str, document_text: str,
               current_label: str, context: dict | None = None) -> dict
```

**Core logic:**
1. OpenAI-compatible client pointing at Ollama endpoint
2. `classify()`: Construct prompt with system/instruction/document XML tags (prompt injection protection per spec §7.2). Return parsed JSON: `{label, is_new_type, confidence, rationale, suggested_rules}`
3. `verify()`: Tier 3 verification prompt. Return: `{label, confidence, reasoning_chain, needs_manual_review}`
4. Retry: up to 2 retries with 1s, 2s backoff on timeout/500 errors
5. Input sanitization: strip known injection patterns before prompt assembly

**Test contracts (`tests/test_llm_client.py`):**
- `test_classify_known_type` — document matching known type → returns correct label
- `test_classify_new_type` — novel document → `is_new_type=True`, `suggested_rules` non-empty
- `test_verify_confirms_label` — correct label input → high confidence, no manual review
- `test_verify_flags_uncertain` — ambiguous doc → `needs_manual_review=True`
- `test_json_output_parsing` — response is valid JSON matching schema
- `test_retry_on_timeout` — timeout on first call → retries and succeeds
- `test_prompt_injection_sanitization` — doc with "ignore previous instructions" → sanitized before LLM call

- [ ] Write failing tests in `tests/test_llm_client.py`
- [ ] Run: `pytest tests/test_llm_client.py -v` — all 7 tests FAIL
- [ ] Implement `src/llm/client.py`
- [ ] Run: `pytest tests/test_llm_client.py -v` — all 7 tests PASS
- [ ] Commit: `git add src/llm/client.py tests/test_llm_client.py && git commit -m "feat: implement Mistral-7B LLM client with Ollama"`

---

### Task 7: Tier 1 Stage A — Structural Hashing + Tests

**Files:** Create `src/tier1/structural.py`, Create `tests/test_tier1_structural.py`

**Goal:** Deterministic structural feature extraction → hash → bucket assignment. O(N) time, no K parameter. For documents and database columns.

**Interface contract:**
```python
class StructuralClusterer:
    def __init__(self, feature_config: list[str] | None = None)
    def extract_features(self, doc: Document) -> str  # returns structural hash
    def assign_bucket(self, doc: Document) -> str     # returns bucket_id
    def cluster(self, documents: list[Document]) -> dict[str, list[str]]
    # Returns: {bucket_id: [doc_id, ...]}
```

**Core logic:**
1. Extract features per spec §4.2 Stage A: file_type, file_size_quantile, page_count, paragraph_count, table_count, has_images, header_pattern, json_schema_signature, path_depth
2. For DB columns: data_type, value_length_stats, null_ratio, unique_ratio, charset_class
3. Feature vector → deterministic hash (SHA256 of canonical feature string)
4. Same hash → same bucket. No iteration, no distance computation.
5. Unknown file types → hash based on available features, don't crash

**Test contracts (`tests/test_tier1_structural.py`):**
- `test_identical_docs_same_bucket` — two docs with identical structure → same bucket_id
- `test_different_structure_different_bucket` — PDF vs JSON → different buckets
- `test_hash_determinism` — same doc hashed twice → identical hash
- `test_empty_document_handling` — doc with no parseable structure → assigned to bucket, no crash
- `test_cluster_output_format` — cluster() returns dict[str, list[str]] with correct keys
- `test_bucket_count_range` — 100 varied docs → reasonable bucket count (not 1, not 100)
- `test_feature_config_filtering` — only requested features used for hashing

- [ ] Write failing tests in `tests/test_tier1_structural.py`
- [ ] Run: `pytest tests/test_tier1_structural.py -v` — all 7 tests FAIL
- [ ] Implement `src/tier1/structural.py`
- [ ] Run: `pytest tests/test_tier1_structural.py -v` — all 7 tests PASS
- [ ] Commit: `git add src/tier1/structural.py tests/test_tier1_structural.py && git commit -m "feat: implement Tier 1 Stage A structural hashing"`

---

### Task 8: Tier 1 Stage B — FAISS Semantic Refinement + Tests

**Files:** Create `src/tier1/semantic.py`, Create `tests/test_tier1_semantic.py`

**Depends on:** Task 4 (embeddings), Task 7 (structural)

**Goal:** Within each structural bucket, refine by semantic similarity. Only triggers when bucket size > `sem_split_threshold`. Uses FAISS IVF + HDBSCAN for sub-cluster discovery. Handles large clusters (>10K docs) by sampling.

**Interface contract:**
```python
class SemanticRefiner:
    def __init__(self, embedder: BgeM3Embedder, config: dict)
    def should_refine(self, bucket_docs: list[Document]) -> bool
    # Returns True if bucket size > sem_split_threshold AND
    # within-cluster cosine similarity std > variance_threshold

    def refine(self, bucket_id: str, documents: list[Document]) -> list[ClusterInfo]
    # 1. Embed all docs in bucket using embedder
    # 2. Check homogeneity: if mean pairwise cosine sim > 0.85 → return single cluster
    # 3. If std > 0.25: build FAISS IVF index → HDBSCAN → sub-clusters
    # 4. For large clusters (>10K): sample 10K → discover structure → NN assign rest
    # 5. For each sub-cluster: compute centroid, radius, representative docs, TF-IDF keywords
```

**Test contracts (`tests/test_tier1_semantic.py`):**
- `test_should_refine_large_heterogeneous` — large bucket with diverse docs → True
- `test_should_refine_small_bucket` — < sem_split_threshold docs → False
- `test_should_refine_homogeneous` — all docs very similar (cosine > 0.85) → False
- `test_refine_produces_clusters` — heterogeneous bucket → multiple ClusterInfo objects
- `test_cluster_has_representatives` — each ClusterInfo has non-empty representative_docs
- `test_large_cluster_sampling` — >10K docs → uses sampling path, doesn't OOM
- `test_subcluster_count_reasonable` — sub-clusters >= 1, not every doc its own cluster
- `test_centroid_computation` — centroid is valid embedding with correct dim

- [ ] Write failing tests in `tests/test_tier1_semantic.py`
- [ ] Run: `pytest tests/test_tier1_semantic.py -v` — all 8 tests FAIL
- [ ] Implement `src/tier1/semantic.py`
- [ ] Run: `pytest tests/test_tier1_semantic.py -v` — all 8 tests PASS
- [ ] Commit: `git add src/tier1/semantic.py tests/test_tier1_semantic.py && git commit -m "feat: implement Tier 1 Stage B FAISS semantic refinement"`

---

### Task 9: Tier 1 Incremental Document Assignment

**Files:** Create `src/tier1/incremental.py`

**Depends on:** Task 7 (structural), Task 8 (semantic)

**Goal:** O(1) hash + O(log N) FAISS nearest-neighbor assignment for new documents. Detects outliers and triggers re-clustering when needed.

**Interface contract:**
```python
class IncrementalAssigner:
    def __init__(self, structural: StructuralClusterer, refiner: SemanticRefiner,
                 embedder: BgeM3Embedder, config: dict)
    def assign(self, doc: Document,
               known_buckets: dict[str, list[ClusterInfo]]) -> AssignmentResult:
        """
        1. Extract structural features → hash → locate Stage A bucket
        2. If bucket exists:
           a. Embed doc
           b. Find nearest sub-cluster center via FAISS
           c. If distance < 1.5x cluster_radius → assign to cluster
           d. Else → mark as "intra_bucket_outlier"
        3. If bucket doesn't exist → create new bucket, mark as "new_structure_candidate"
        4. Return AssignmentResult with outlier flags
        """
```

**No separate test file** — tested via integration tests (Task 18). This component orchestrates Task 7+8, which are already unit-tested.

- [ ] Implement `src/tier1/incremental.py`
- [ ] Verify import: `python -c "from src.tier1.incremental import IncrementalAssigner"`

---

### Task 10: Tier 2 Known Type Matching + Tests

**Files:** Create `src/tier2/matching.py`, Create `tests/test_tier2_matching.py`

**Goal:** 3-signal weighted scoring against known type library. Determines if a cluster matches an existing type without LLM invocation.

**Interface contract:**
```python
class KnownTypeMatcher:
    def __init__(self, known_types: list[KnownType], config: dict)
    # config keys: structure_signature_weight (0.5), tfidf_overlap_weight (0.3),
    #              pii_distribution_weight (0.2), high_match_threshold (0.8),
    #              low_match_threshold (0.5)

    def match(self, cluster: ClusterInfo) -> MatchResult:
        """
        For each known type:
          1. Structure signature exact match → score += 0.5
          2. TF-IDF keyword Jaccard overlap > 0.6 → score += 0.3
          3. PII distribution cosine similarity > 0.8 → score += 0.2
        Return best match with:
          - score >= 0.8 → method="known_match", no LLM needed
          - score in [0.5, 0.8) → method="llm_confirm", needs LLM
          - score < 0.5 → method="unknown", LLM classification needed
        """

    def register_type(self, known_type: KnownType) -> None
    def get_type(self, type_id: str) -> KnownType | None
```

**Test contracts (`tests/test_tier2_matching.py`):**
- `test_exact_match_high_score` — perfect structural + keyword + PII match → score >= 0.8
- `test_no_match_low_score` — completely different cluster → score < 0.5
- `test_partial_match_mid_score` — structural match but different content → score in [0.5, 0.8)
- `test_empty_known_types` — no types registered → score = 0 for any cluster
- `test_register_and_match` — register type → match finds it
- `test_pii_cosine_similarity_bounds` — PII distribution similarity in [0, 1]

- [ ] Write failing tests in `tests/test_tier2_matching.py`
- [ ] Run: `pytest tests/test_tier2_matching.py -v` — all 6 tests FAIL
- [ ] Implement `src/tier2/matching.py`
- [ ] Run: `pytest tests/test_tier2_matching.py -v` — all 6 tests PASS
- [ ] Commit: `git add src/tier2/matching.py tests/test_tier2_matching.py && git commit -m "feat: implement Tier 2 known type matching"`

---

### Task 11: Tier 2 Classifier + Tests

**Files:** Create `src/tier2/classifier.py`, Create `tests/test_tier2_classifier.py`

**Depends on:** Task 5 (NER), Task 6 (LLM client), Task 10 (matching)

**Goal:** Cluster-level classification orchestrator. The central coordinator of Tier 2 — runs the match→NER→LLM→propagate flow for each cluster.

**Interface contract:**
```python
class Tier2Classifier:
    def __init__(self, matcher: KnownTypeMatcher, ner: DebertaNER,
                 llm: MistralClient, config: dict)
    def classify_clusters(self, clusters: list[ClusterInfo],
                          documents: list[Document]) -> list[ClassificationResult]:
        """
        For each cluster:
          Step 1: Extract cluster features (TF-IDF top-15, PII distribution, language)
          Step 2: Run KnownTypeMatcher.match()
            - score >= 0.8 → adopt match, skip to Step 5
            - score < 0.8 → continue to Step 3-4
          Step 3: Run DeBERTa NER on representative docs
            - Consume Tier 0 PIIFeatureVector for low-conf entities
          Step 4: LLM classification (only unmatched or mid-confidence clusters)
            - Mistral-7B 4-bit, ~300ms/call
            - Input: cluster features + rep doc text (first 2000 chars) + NER results
            - Output: {label, is_new_type, confidence, rationale, suggested_rules}
          Step 5: Propagate labels (delegate to propagation.py)
        Returns list of ClassificationResult per document
        """

    def cold_start_classify(self, documents: list[Document]) -> list[ClassificationResult]:
        """Phase -1: zero-shot LLM per document, no clustering. Temporary labels."""
```

**Test contracts (`tests/test_tier2_classifier.py`):**
- `test_known_match_skips_llm` — cluster with high match score → no LLM call, label from matcher
- `test_unknown_cluster_triggers_llm` — no match → LLM invoked
- `test_mid_confidence_triggers_llm` — score in [0.5, 0.8) → LLM confirmation
- `test_cold_start_no_clustering` — cold_start_classify() uses per-doc LLM
- `test_cluster_features_extracted` — TF-IDF keywords + PII distribution populated
- `test_ner_called_on_representatives` — NER runs on rep docs, not all docs

- [ ] Write failing tests in `tests/test_tier2_classifier.py`
- [ ] Run: `pytest tests/test_tier2_classifier.py -v` — all 6 tests FAIL
- [ ] Implement `src/tier2/classifier.py`
- [ ] Run: `pytest tests/test_tier2_classifier.py -v` — all 6 tests PASS
- [ ] Commit: `git add src/tier2/classifier.py tests/test_tier2_classifier.py && git commit -m "feat: implement Tier 2 cluster classification orchestrator"`

---

### Task 12: Tier 2 Label Propagation

**Files:** Create `src/tier2/propagation.py`

**Goal:** Propagate cluster labels to all member documents. Sampling verification with inconsistency detection.

**Interface contract:**
```python
class LabelPropagator:
    def __init__(self, config: dict)
    # config: sample_strategy, min_samples (3), inconsistency_threshold (0.15)

    def propagate(self, cluster: ClusterInfo, label: str, confidence: float,
                  documents: list[Document], llm: MistralClient | None = None
                  ) -> tuple[list[ClassificationResult], bool]:
        """
        1. Assign cluster label to all cluster documents
        2. Sample documents for verification (inverse cluster size, min 3)
        3. If llm provided: verify sampled docs via LLM
        4. If verification inconsistency > 15% → flag cluster for re-splitting
        Returns: (results, needs_resplit)
        """
```

**No separate test file** — tested via Tier 2 classifier tests and integration tests.

- [ ] Implement `src/tier2/propagation.py`
- [ ] Verify import: `python -c "from src.tier2.propagation import LabelPropagator"`

---

### Task 13: Tier 3 Quality Gate + Tests

**Files:** Create `src/tier3/quality_gate.py`, Create `tests/test_tier3.py`

**Depends on:** Task 6 (LLM client)

**Goal:** Final precision defense line. 5 triggers per spec §4.4. Mistral-7B INT8. Only ~2% of documents reach this tier.

**Interface contract:**
```python
class QualityGate:
    def __init__(self, llm: MistralClient, config: dict)

    def should_trigger(self, doc: Document, cluster: ClusterInfo,
                       classification: ClassificationResult,
                       ner_results: list[PIIFeature] | None = None) -> bool:
        """
        Triggers if ANY of:
          1. Label is high-sensitivity (SSN/CREDIT_CARD/MEDICAL/IBAN)
          2. Semantic distance from cluster centroid > 2σ
          3. DeBERTa NER contradicts Tier 0 rule results
          4. LLM classification confidence in [0.5, 0.8)
          5. Document marked for verification by Tier 2 Step 5
        """

    def verify(self, doc: Document, cluster: ClusterInfo,
               current_classification: ClassificationResult) -> ClassificationResult:
        """
        LLM deep analysis with full doc context + metadata.
        Returns final classification with confidence + reasoning chain.
        If confidence still low → mark needs_manual_review=True
        """

    def verify_batch(self, docs: list[Document], cluster: ClusterInfo,
                     classifications: list[ClassificationResult]
                     ) -> list[ClassificationResult]:
        """Batch verification: same-cluster docs submitted together for context sharing."""
```

**Test contracts (`tests/test_tier3.py`):**
- `test_triggers_high_sensitivity` — SSN-labeled doc → trigger
- `test_triggers_low_confidence` — confidence 0.55 → trigger
- `test_no_trigger_normal_doc` — normal doc, high confidence → no trigger
- `test_triggers_semantic_outlier` — distance > 2σ → trigger
- `test_verify_returns_reasoning` — result includes non-empty rationale
- `test_verify_manual_review_flag` — ambiguous doc → needs_manual_review=True
- `test_batch_verify_context_sharing` — docs in same cluster verified together

- [ ] Write failing tests in `tests/test_tier3.py`
- [ ] Run: `pytest tests/test_tier3.py -v` — all 7 tests FAIL
- [ ] Implement `src/tier3/quality_gate.py`
- [ ] Run: `pytest tests/test_tier3.py -v` — all 7 tests PASS
- [ ] Commit: `git add src/tier3/quality_gate.py tests/test_tier3.py && git commit -m "feat: implement Tier 3 LLM quality gate"`

---

### Task 14: Type Discovery Loop + Tests

**Files:** Create `src/discovery/loop.py`, Create `tests/test_discovery.py`

**Depends on:** Task 7 (structural), Task 8 (semantic), Task 11 (classifier)

**Goal:** Outlier accumulation → periodic re-clustering → new type evaluation → registration. Per spec §4.5.

**Interface contract:**
```python
class DiscoveryLoop:
    def __init__(self, structural: StructuralClusterer, refiner: SemanticRefiner,
                 embedder: BgeM3Embedder, classifier: Tier2Classifier, config: dict)

    def collect_outlier(self, doc: Document, reason: str) -> None
    # Sources: Tier 2 match score < 0.5, LLM is_new_type=True, Tier 3 distance > 3σ

    def should_run(self) -> bool:
    # Triggers: buffer >= min_trigger_count (100), same pattern >= same_pattern_threshold (5),
    #           or time since last run >= time_trigger_hours (24)

    def run(self) -> list[KnownType]:
        """
        1. Get docs from outlier buffer
        2. Run Tier 1 two-stage clustering on outliers
        3. For each candidate cluster:
           a. Check intra-cluster coherence > 0.75
           b. Check distance to nearest known type > 0.3
           c. Check cluster size >= 3
        4. For passing candidates: LLM generates type name, description, detection rules
        5. Register new types → update known type library
        """

    def register_type(self, known_type: KnownType) -> None
```

**Test contracts (`tests/test_discovery.py`):**
- `test_outlier_collection` — documents added to buffer correctly
- `test_should_run_count_trigger` — buffer reaches 100 → True
- `test_should_run_pattern_trigger` — same unknown pattern 5 times → True
- `test_should_run_below_threshold` — buffer < 100, no pattern → False
- `test_run_produces_types` — diverse outliers → new KnownType objects
- `test_coherence_filter` — incoherent cluster (coherence < 0.75) → rejected
- `test_distance_filter` — cluster too close to known type → rejected
- `test_min_size_filter` — cluster with < 3 docs → rejected

- [ ] Write failing tests in `tests/test_discovery.py`
- [ ] Run: `pytest tests/test_discovery.py -v` — all 8 tests FAIL
- [ ] Implement `src/discovery/loop.py`
- [ ] Run: `pytest tests/test_discovery.py -v` — all 8 tests PASS
- [ ] Commit: `git add src/discovery/loop.py tests/test_discovery.py && git commit -m "feat: implement type discovery loop"`

---

### Task 15: Knowledge Distillation Trainer

**Files:** Create `src/distillation/trainer.py`

**Goal:** SetFit-based teacher→student distillation. Compresses Tier 2+3 classification to ~2ms CPU inference. Incremental retrain with F1 degradation guard.

**Interface contract:**
```python
class DistillationTrainer:
    def __init__(self, config: dict)
    # config: teacher, student base_model, training params, inference thresholds

    def build_training_data(self, classified_docs: list[Document]
                            ) -> tuple[list[str], list[str]]:
        """
        1. Filter: confidence > 0.85
        2. Sample per type: 50-200 (inverse frequency)
        3. Hold out 10% for eval
        Returns: (texts, labels)
        """

    def train(self, texts: list[str], labels: list[str]
              ) -> tuple[Any, dict]:  # (model, eval_metrics)
        """Train SetFit model with BGE-M3 base. Return model + per-class F1."""

    def evaluate_against_pipeline(self, model: Any,
                                   eval_docs: list[Document],
                                   pipeline_results: list[ClassificationResult]
                                   ) -> dict:
        """Compare distilled model predictions vs full pipeline. Return agreement rate."""

    def should_retrain(self, new_sample_count: int) -> bool:
        """True if new samples >= retrain_trigger (500)."""

    def check_degradation(self, old_metrics: dict, new_metrics: dict) -> bool:
        """True if any class F1 dropped > 3%."""
```

**No separate test file** — training requires GPU + real models. Logic tested via integration.

- [ ] Implement `src/distillation/trainer.py`
- [ ] Verify import: `python -c "from src.distillation.trainer import DistillationTrainer"`

---

### Task 16: Main Entry Point

**Files:** Modify `main.py`

**Goal:** Full pipeline orchestration. Load config → read documents → run tiers sequentially → output results.

**Core flow:**
```python
def main():
    # 1. Parse args + load config
    # 2. Initialize all components:
    #    - Tier0Engine, StructuralClusterer, BgeM3Embedder, SemanticRefiner
    #    - DebertaNER, MistralClient(4bit), KnownTypeMatcher, Tier2Classifier
    #    - LabelPropagator, MistralClient(int8), QualityGate, DiscoveryLoop
    # 3. Load documents from input directory (support .txt, .pdf, .docx, .json)
    # 4. Cold start check: if known types empty → cold_start path
    # 5. Tier 0: PIIFeatureVector for all docs
    # 6. Tier 1: Stage A structural buckets → Stage B semantic refinement
    # 7. Tier 2: Classify clusters → propagate labels
    # 8. Tier 3: Quality gate on triggered docs
    # 9. Discovery: Collect outliers, check if discovery loop should run
    # 10. Output: JSON results + stats to output directory
    # 11. Log all metrics: timing per tier, trigger rates, cluster counts
```

- [ ] Replace stub `main.py` with full pipeline orchestration
- [ ] Verify: `python main.py --help` shows usage

---

### Task 17: Incremental Entry Point

**Files:** Modify `incremental.py`

**Goal:** Single-document incremental processing path. For new documents arriving after initial clustering.

**Core flow:**
```python
def main():
    # 1. Load config + existing cluster state (from ES or local storage)
    # 2. Initialize IncrementalAssigner with existing buckets and clusters
    # 3. For each new document:
    #    a. Tier 0 feature extraction
    #    b. IncrementalAssigner.assign() → nearest cluster or outlier
    #    c. If assigned → inherit cluster label
    #    d. If outlier → run Tier 2+3 per-doc classification
    # 4. Check if outlier count triggers re-clustering
    # 5. Output: classification result for each doc
```

- [ ] Replace stub `incremental.py` with incremental processing logic
- [ ] Verify: `python incremental.py --help` shows usage

---

### Task 18: Integration Tests

**Files:** Create `tests/test_integration.py`, Create `tests/conftest.py`

**Depends on:** Tasks 3-17

**Goal:** End-to-end pipeline tests with realistic document sets. Verifies Tier 0→1→2→3 flow, cold start, incremental path, and fault tolerance.

**Shared fixtures (`tests/conftest.py`):**
- `sample_documents()` — 20 synthetic documents covering 5 types: HR records (SSN), financial reports (credit card placeholders), medical forms, JSON API logs, plain text
- `sample_config()` — dict with all tier configs from `config.yaml`
- `mock_llm_responses()` — predefined LLM responses for known document patterns

**Integration test contracts (`tests/test_integration.py`):**
- `test_full_pipeline_small_batch` — 20 docs through Tier 0→1→2→3 → each gets a label
- `test_cold_start_path` — empty known types → cold_start_classify triggered
- `test_tier3_triggers_high_sensitivity` — SSN docs flagged for Tier 3
- `test_cluster_label_propagation` — all docs in same cluster get same label
- `test_end_to_end_output_schema` — PipelineResult has correct structure
- `test_pipeline_stats_populated` — stats include timing and trigger counts
- `test_discovery_outlier_accumulation` — novel docs collected in outlier buffer

- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] Write failing integration tests in `tests/test_integration.py`
- [ ] Run: `pytest tests/test_integration.py -v` — all 7 tests FAIL
- [ ] Fix any issues until all tests PASS
- [ ] Commit: `git add tests/conftest.py tests/test_integration.py && git commit -m "test: add integration tests for full pipeline"`

---

### Task 19: End-to-End Verification

**Goal:** Run the full pipeline on real sample documents and verify outputs.

- [ ] Create a sample document set: `mkdir -p output/sample_docs && python -c "..." ` (generate 5 PDF-like .txt files representing HR/finance/medical/API/mixed types)
- [ ] Run: `python main.py --input output/sample_docs --output output/test_run/`
- [ ] Verify output directory contains: `results.json`, `stats.json`
- [ ] Verify `stats.json` metrics: `tier0_pii_detection_rate`, `tier1_cluster_count`, `tier2_llm_trigger_rate`, `tier3_trigger_rate`, `e2e_latency_p95`
- [ ] Verify each document in `results.json` has: `doc_id`, `label`, `confidence`, `method`
- [ ] Run: `pytest tests/ -v` — all tests PASS
- [ ] Commit: `git add -A && git commit -m "feat: complete DataDNA pipeline end-to-end verification"`

---
