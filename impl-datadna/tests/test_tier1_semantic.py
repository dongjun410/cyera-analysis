"""Tests for Tier 1 Stage B: FAISS Semantic Refinement (TDD).

Test contracts (write these BEFORE implementation):
  1. test_should_refine_large_heterogeneous — 100 docs with diverse content → True
  2. test_should_refine_small_bucket — < sem_split_threshold docs → False
  3. test_should_refine_homogeneous — All docs very similar → False (mean cos sim > 0.85)
  4. test_refine_produces_clusters — Heterogeneous bucket → multiple ClusterInfo objects
  5. test_cluster_has_representatives — Each ClusterInfo has non-empty representative_docs
  6. test_subcluster_count_reasonable — Sub-clusters >= 1, not every doc its own cluster
  7. test_centroid_computation — Centroid is valid embedding with correct dimension
  8. test_empty_bucket — Empty document list → returns empty list, no crash
"""

from __future__ import annotations

import numpy as np
import pytest

from src.types import ClusterInfo, Document


# ──────────────────────────────────────────────────────────────
# Mock Embedder — deterministic, no model download needed
# ──────────────────────────────────────────────────────────────

class MockEmbedder:
    """Deterministic mock embedder using word-level random projections.

    Each unique word is mapped to a fixed random unit vector (seeded by
    ``hash(word)``).  A document embedding is the L2-normalized sum of
    its word vectors.

    This gives:
    - Same text → same embedding (deterministic)
    - Similar texts (shared vocabulary) → similar embeddings
    - Diverse texts (different domain words) → different embeddings
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self._word_vectors: dict[str, np.ndarray] = {}
        # Separate RNG for initialisation determinism
        self._init_rng = np.random.RandomState(42)

    def _get_word_vector(self, word: str) -> np.ndarray:
        """Return a deterministic unit vector for `word` (cached)."""
        if word not in self._word_vectors:
            seed = hash(word) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            self._word_vectors[word] = vec
        return self._word_vectors[word]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        results = []
        for text in texts:
            # Split on whitespace and punctuation boundaries
            words = text.lower().split()
            if not words:
                vec = np.zeros(self._dim, dtype=np.float32)
                vec[0] = 1.0
            else:
                vec = np.zeros(self._dim, dtype=np.float32)
                for word in words:
                    vec += self._get_word_vector(word)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                else:
                    vec[0] = 1.0
            results.append(vec)

        return np.stack(results)

    @property
    def dim(self) -> int:
        return self._dim


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_DIVERSE_TEMPLATES = [
    "Annual financial report for fiscal year {i}: revenue grew {pct}% with {amt} in net income.",
    "Employee performance review for {name}: exceeded expectations in {dept} department.",
    "Medical claim form #{i}: patient diagnosed with {condition}, treatment plan includes {med}.",
    "Software architecture document: {component} service uses {tech} for {purpose}.",
    "Legal contract amendment #{i}: Section {section} revised to include {clause}.",
    "Meeting minutes from {date}: discussed {topic}, action items assigned to {person}.",
    "Product specification sheet: {product} model {i}, dimensions {d1}x{d2}x{d3} cm.",
    "Customer support ticket #{i}: user reported {issue} on {platform} version {ver}.",
    "Research paper draft: study on {field} shows {result} with p-value {pval}.",
    "Inventory audit log: warehouse {wh}, item SKU-{sku}, quantity adjusted by {qty}.",
]

_SIMILAR_TEMPLATE = (
    "Quarterly financial summary: revenue ${rev} million, "
    "operating expenses ${exp} million, net profit ${profit} million. "
    "Key metrics include customer growth of {growth}% and "
    "employee headcount of {headcount}."
)


def _make_doc(doc_id: str, text: str) -> Document:
    """Create a minimal Document for testing."""
    return Document(doc_id=doc_id, text=text)


def _make_diverse_docs(count: int, seed: int = 42) -> list[Document]:
    """Generate `count` documents using diverse templates with varied content."""
    rng = np.random.RandomState(seed)
    docs = []
    for i in range(count):
        tmpl = _DIVERSE_TEMPLATES[i % len(_DIVERSE_TEMPLATES)]
        text = tmpl.format(
            i=i,
            pct=rng.randint(5, 50),
            amt=rng.randint(1_000, 1_000_000),
            name=f"Employee_{rng.choice(['A','B','C','D','E'])}",
            dept=rng.choice(["Engineering", "Sales", "Marketing", "Finance"]),
            condition=rng.choice(["flu", "fracture", "migraine", "rash"]),
            med=rng.choice(["aspirin", "ibuprofen", "acetaminophen", "antibiotics"]),
            component=rng.choice(["Auth", "Payment", "Notification", "Search"]),
            tech=rng.choice(["gRPC", "Kafka", "Redis", "Postgres"]),
            purpose=rng.choice(["auth", "messaging", "caching", "storage"]),
            section=rng.randint(1, 20),
            clause=rng.choice(["indemnification", "termination", "liability", "arbitration"]),
            date=f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            topic=rng.choice(["budget", "strategy", "hiring", "product launch"]),
            person=rng.choice(["Alice", "Bob", "Carol", "Dave"]),
            product=rng.choice(["Widget", "Gadget", "Gizmo", "Doohickey"]),
            d1=rng.randint(5, 50), d2=rng.randint(5, 50), d3=rng.randint(5, 50),
            issue=rng.choice(["crash", "slow load", "login fail", "data loss"]),
            platform=rng.choice(["iOS", "Android", "Web", "Desktop"]),
            ver=rng.choice(["2.1", "3.0", "4.2", "5.1"]),
            field=rng.choice(["biology", "physics", "CS", "economics"]),
            result=rng.choice(["positive", "negative", "inconclusive"]),
            pval=f"{rng.random():.3f}",
            wh=rng.choice(["A1", "B2", "C3"]),
            sku=rng.randint(1000, 9999),
            qty=rng.randint(-50, 50),
        )
        docs.append(_make_doc(f"doc-{i:04d}", text))
    return docs


def _make_similar_docs(count: int, seed: int = 42) -> list[Document]:
    """Generate `count` documents using the same template with minor numeric variations."""
    rng = np.random.RandomState(seed)
    docs = []
    for i in range(count):
        text = _SIMILAR_TEMPLATE.format(
            rev=rng.randint(100, 500),
            exp=rng.randint(50, 300),
            profit=rng.randint(10, 200),
            growth=round(rng.uniform(1.0, 20.0), 1),
            headcount=rng.randint(50, 5000),
        )
        docs.append(_make_doc(f"similar-{i:04d}", text))
    return docs


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=128)


@pytest.fixture
def default_config() -> dict:
    return {
        "sem_split_threshold": 50,
        "homogeneity_threshold": 0.85,
        "variance_threshold": 0.25,
        "max_sample_for_large": 10000,
        "faiss_nlist": 100,
        "faiss_nprobe": 10,
    }


@pytest.fixture
def heterogeneous_config() -> dict:
    """Config tuned for the MockEmbedder's statistical profile.

    The mock embedder (128-dim random word projections) produces
    std of cosine similarities around 0.07-0.10 for diverse texts,
    so we lower variance_threshold accordingly.  Real BGE-M3 (1024-dim)
    uses the standard 0.25.
    """
    return {
        "sem_split_threshold": 50,
        "homogeneity_threshold": 0.85,
        "variance_threshold": 0.05,   # lowered for mock embedder
        "max_sample_for_large": 10000,
        "faiss_nlist": 100,
        "faiss_nprobe": 10,
    }


# ──────────────────────────────────────────────────────────────
# Test 1: should_refine — large heterogeneous → True
# ──────────────────────────────────────────────────────────────

def test_should_refine_large_heterogeneous(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """100 diverse docs exceed threshold AND have high variance → True."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)
    docs = _make_diverse_docs(100)

    result = refiner.should_refine(docs)
    assert result is True, (
        f"Large heterogeneous bucket should trigger refinement, got {result}"
    )


# ──────────────────────────────────────────────────────────────
# Test 2: should_refine — small bucket → False
# ──────────────────────────────────────────────────────────────

def test_should_refine_small_bucket(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """Bucket size below sem_split_threshold → False regardless of content."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)

    # 10 diverse docs — still below default threshold of 50
    docs = _make_diverse_docs(10)
    result = refiner.should_refine(docs)
    assert result is False, (
        f"Small bucket (below threshold) should not trigger refinement, got {result}"
    )


# ──────────────────────────────────────────────────────────────
# Test 3: should_refine — homogeneous → False
# ──────────────────────────────────────────────────────────────

def test_should_refine_homogeneous(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """All docs very similar (same template) → high mean cosine sim → False."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)

    # 100 docs from the same template — very similar embeddings
    docs = _make_similar_docs(100)
    result = refiner.should_refine(docs)
    assert result is False, (
        f"Homogeneous bucket should not trigger refinement, got {result}"
    )


# ──────────────────────────────────────────────────────────────
# Test 4: refine — produces multiple clusters for heterogeneous
# ──────────────────────────────────────────────────────────────

def test_refine_produces_clusters(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """Heterogeneous bucket → refine returns multiple ClusterInfo objects."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)
    docs = _make_diverse_docs(100)

    clusters = refiner.refine("bucket-001", docs)

    assert isinstance(clusters, list), f"Expected list, got {type(clusters)}"
    assert len(clusters) >= 2, (
        f"Expected at least 2 sub-clusters for diverse docs, got {len(clusters)}"
    )
    assert all(isinstance(c, ClusterInfo) for c in clusters), (
        "All items must be ClusterInfo instances"
    )


# ──────────────────────────────────────────────────────────────
# Test 5: each cluster has representative docs
# ──────────────────────────────────────────────────────────────

def test_cluster_has_representatives(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """Every ClusterInfo has a non-empty representative_docs list."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)
    docs = _make_diverse_docs(100)

    clusters = refiner.refine("bucket-001", docs)

    for ci in clusters:
        assert len(ci.representative_docs) > 0, (
            f"Cluster {ci.cluster_id} has empty representative_docs"
        )
        assert len(ci.representative_docs) <= 3, (
            f"Expected at most 3 representative docs, got {len(ci.representative_docs)}"
        )
        # Rep docs must be valid doc_ids from the input
        valid_ids = {d.doc_id for d in docs}
        for rep_id in ci.representative_docs:
            assert rep_id in valid_ids, (
                f"Representative doc {rep_id} not in input documents"
            )


# ──────────────────────────────────────────────────────────────
# Test 6: sub-cluster count is reasonable
# ──────────────────────────────────────────────────────────────

def test_subcluster_count_reasonable(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """Sub-cluster count >= 1 and not every doc gets its own cluster."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)

    # 100 diverse docs across 10 templates
    docs = _make_diverse_docs(100)
    clusters = refiner.refine("bucket-001", docs)

    assert len(clusters) >= 1, "Expected at least 1 cluster"
    # With 100 docs, we should NOT get 100 clusters (over-clustering)
    assert len(clusters) < len(docs), (
        f"Expected fewer clusters than documents, got {len(clusters)} == {len(docs)}"
    )

    # All doc_ids should be accounted for across all clusters
    clustered_ids = set()
    for ci in clusters:
        clustered_ids.update(ci.doc_ids)
    input_ids = {d.doc_id for d in docs}
    assert clustered_ids == input_ids, (
        f"Clustered doc_ids must match input doc_ids. "
        f"Missing: {input_ids - clustered_ids}, Extra: {clustered_ids - input_ids}"
    )


# ──────────────────────────────────────────────────────────────
# Test 7: centroid computation
# ──────────────────────────────────────────────────────────────

def test_centroid_computation(
    mock_embedder: MockEmbedder,
    heterogeneous_config: dict,
) -> None:
    """Centroid embedding is a valid numpy array with correct dimension."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, heterogeneous_config)
    docs = _make_diverse_docs(100)

    clusters = refiner.refine("bucket-001", docs)

    for ci in clusters:
        assert ci.centroid_embedding is not None, (
            f"Cluster {ci.cluster_id} missing centroid_embedding"
        )
        assert isinstance(ci.centroid_embedding, np.ndarray), (
            f"Centroid should be np.ndarray, got {type(ci.centroid_embedding)}"
        )
        assert ci.centroid_embedding.shape == (mock_embedder.dim,), (
            f"Expected centroid shape ({mock_embedder.dim},), "
            f"got {ci.centroid_embedding.shape}"
        )
        assert ci.centroid_embedding.dtype == np.float32, (
            f"Expected float32 dtype, got {ci.centroid_embedding.dtype}"
        )
        # Not NaNs
        assert not np.any(np.isnan(ci.centroid_embedding)), "Centroid contains NaN"


# ──────────────────────────────────────────────────────────────
# Test 8: empty bucket
# ──────────────────────────────────────────────────────────────

def test_empty_bucket(
    mock_embedder: MockEmbedder,
    default_config: dict,
) -> None:
    """Empty document list → returns empty list, no crash."""
    from src.tier1.semantic import SemanticRefiner

    refiner = SemanticRefiner(mock_embedder, default_config)

    # refine with empty list
    result = refiner.refine("empty-bucket", [])
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 0, f"Expected empty list, got {len(result)} items"

    # should_refine with empty list
    should = refiner.should_refine([])
    assert should is False, f"Empty list should not trigger refinement, got {should}"
