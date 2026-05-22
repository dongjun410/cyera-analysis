"""
Unit tests for core.label_propagator.LabelPropagator.
"""
import numpy as np
import pytest
from unittest.mock import patch

from core.label_propagator import LabelPropagator
from models.schemas import ProcessedDocument, ClusterInfo


# ── Helpers ───────────────────────────────────────────────────

def make_doc(content="test content"):
    """Create a minimal ProcessedDocument with a unique id."""
    uid = make_doc._counter
    make_doc._counter += 1
    return ProcessedDocument(
        id=f"doc_{uid}",
        original_path=f"/fake/doc_{uid}.txt",
        title=f"Document {uid}",
        raw_content=content,
    )
make_doc._counter = 0


def _make_embeddings_and_docs(n=6, seed=42):
    """Create n documents with 2D embeddings forming 2 clear clusters."""
    rng = np.random.RandomState(seed)
    documents = [make_doc(f"document {i} content text") for i in range(n)]
    c0 = rng.randn(n // 2, 2) * 0.5 + np.array([0.0, 0.0])
    c1 = rng.randn(n // 2, 2) * 0.5 + np.array([10.0, 10.0])
    embeddings = np.vstack([c0, c1])
    return documents, embeddings


# ── _select_representatives ───────────────────────────────────

def test_select_representatives_small():
    """3 embeddings, sample_per_cluster=2 → returns 2 indices."""
    embeddings = np.random.RandomState(42).randn(3, 10)
    lp = LabelPropagator({"sample_per_cluster": 2, "use_llm": False})
    indices = lp._select_representatives(embeddings)
    assert len(indices) == 2
    assert all(isinstance(i, (int, np.integer)) for i in indices)


def test_select_representatives_all():
    """When sample_per_cluster >= n, returns all indices."""
    embeddings = np.random.RandomState(42).randn(4, 10)
    lp = LabelPropagator({"sample_per_cluster": 10, "use_llm": False})
    indices = lp._select_representatives(embeddings)
    assert len(indices) == 4
    assert set(indices) == {0, 1, 2, 3}


def test_select_representatives_returns_ints():
    """Returned indices are integers."""
    embeddings = np.random.RandomState(42).randn(10, 8)
    lp = LabelPropagator({"sample_per_cluster": 5, "use_llm": False})
    indices = lp._select_representatives(embeddings)
    for idx in indices:
        assert isinstance(idx, (int, np.integer)), f"expected int, got {type(idx)}"


# ── _compute_coherence ────────────────────────────────────────

def test_compute_coherence_identical():
    """Identical vectors → coherence ≈ 1.0."""
    vec = np.random.RandomState(42).randn(5, 10)
    identical = np.tile(vec[0:1], (5, 1))
    coherence = LabelPropagator._compute_coherence(identical)
    assert abs(coherence - 1.0) < 0.01


def test_compute_coherence_single():
    """Single vector → coherence = 1.0."""
    coherence = LabelPropagator._compute_coherence(np.random.RandomState(42).randn(1, 10))
    assert coherence == 1.0


# ── _extract_keywords (with spacy mocked) ─────────────────────

def test_extract_keywords_basic():
    """Two meaningful docs → non-empty keyword list (spacy mocked)."""
    texts = [
        "machine learning AI model neural network",
        "deep learning neural network training data",
    ]
    lp = LabelPropagator({"use_llm": False})
    # Mock tokenize_text to pass through (TfidfVectorizer handles tokenization)
    with patch("core.label_propagator.tokenize_text", side_effect=lambda text, lang="en": text):
        keywords = lp._extract_keywords(texts, top_n=10)
    assert isinstance(keywords, list)
    assert len(keywords) > 0, f"expected non-empty keywords, got {keywords}"


def test_extract_keywords_empty():
    """Empty text list → empty keyword list."""
    lp = LabelPropagator({"use_llm": False})
    keywords = lp._extract_keywords([], top_n=10)
    assert keywords == []


# ── process_clusters (with spacy mocked) ───────────────────────

def test_process_clusters_basic():
    """6 documents, 2 clusters → 2 ClusterInfo objects with correct sizes."""
    documents, embeddings = _make_embeddings_and_docs(6)
    labels = np.array([0, 0, 0, 1, 1, 1])

    lp = LabelPropagator({"sample_per_cluster": 3, "use_llm": False})
    with patch("core.label_propagator.tokenize_text", side_effect=lambda text, lang="en": text):
        clusters = lp.process_clusters(documents, embeddings, labels)

    assert len(clusters) == 2
    sizes = {c.cluster_id: c.size for c in clusters}
    assert sizes[0] == 3
    assert sizes[1] == 3


def test_process_clusters_labels_on_docs():
    """After process_clusters, documents have cluster_id and classification_source set."""
    documents, embeddings = _make_embeddings_and_docs(6)
    labels = np.array([0, 0, 0, 1, 1, 1])

    lp = LabelPropagator({"sample_per_cluster": 3, "use_llm": False})
    with patch("core.label_propagator.tokenize_text", side_effect=lambda text, lang="en": text):
        lp.process_clusters(documents, embeddings, labels)

    for doc in documents:
        assert doc.cluster_id in (0, 1)
        assert doc.classification_source == "clustering"


def test_process_clusters_keywords():
    """Each cluster has keywords list."""
    documents, embeddings = _make_embeddings_and_docs(6)
    labels = np.array([0, 0, 0, 1, 1, 1])

    lp = LabelPropagator({"sample_per_cluster": 3, "use_llm": False})
    with patch("core.label_propagator.tokenize_text", side_effect=lambda text, lang="en": text):
        clusters = lp.process_clusters(documents, embeddings, labels)

    for c in clusters:
        assert isinstance(c.keywords, list), f"keywords should be a list, got {type(c.keywords)}"


def test_process_clusters_no_llm():
    """use_llm=False → llm_label is empty, keywords populated."""
    documents, embeddings = _make_embeddings_and_docs(6)
    labels = np.array([0, 0, 0, 1, 1, 1])

    lp = LabelPropagator({"sample_per_cluster": 3, "use_llm": False})
    with patch("core.label_propagator.tokenize_text", side_effect=lambda text, lang="en": text):
        clusters = lp.process_clusters(documents, embeddings, labels)

    for c in clusters:
        assert c.llm_label == ""
        assert c.keywords is not None
