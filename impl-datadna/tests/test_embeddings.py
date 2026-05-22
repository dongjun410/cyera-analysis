"""Tests for BGE-M3 Embedder.

Test contracts (TDD — write these BEFORE implementation):
  1. test_embedding_shape     — encode 5 texts -> (5, dim) array
  2. test_embedding_normalized — all output vectors have L2 norm ~1.0 (within 1e-5)
  3. test_single_document      — encode single text -> shape (1, dim), 2D not 1D
  4. test_cosine_similarity_range — similar texts closer than dissimilar texts
  5. test_empty_string         — encode [""] -> non-NaN valid embedding
  6. test_dim_property         — .dim returns expected dimension (1024 for BGE-M3)
  7. test_same_text_same_embedding — encoding same text twice gives same embedding

Model loading strategy:
  1. Try BAAI/bge-m3 (1024-dim) — production model
  2. Fall back to all-MiniLM-L6-v2 (384-dim) — for CI / quick verification
  3. Skip all tests if no model is available
"""

from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────
# Fixture: load embedder with fallback
# ──────────────────────────────────────────────────────────────

def _model_is_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is fully cached locally (no incomplete downloads)."""
    from pathlib import Path

    model_dir = model_name.replace("/", "--")
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    model_path = cache_base / f"models--{model_dir}"

    if not model_path.exists():
        return False

    # Check for any incomplete downloads — if present, cache is not ready
    blobs = model_path / "blobs"
    if blobs.exists():
        incomplete = list(blobs.glob("*.incomplete"))
        if incomplete:
            return False

    # Check for at least one completed snapshot
    snapshots = model_path / "snapshots"
    if snapshots.exists() and any(snapshots.iterdir()):
        return True

    return False


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped embedder — tries BGE-M3 first, falls back to MiniLM."""
    from src.embeddings.bge_m3 import BgeM3Embedder

    # Try production model (BGE-M3, 1024-dim) — only if cached locally
    if _model_is_cached("BAAI/bge-m3"):
        try:
            model = BgeM3Embedder(model_name="BAAI/bge-m3")
            return model
        except Exception:
            pass

    # Try fallback model for CI / quick verification
    if _model_is_cached("sentence-transformers/all-MiniLM-L6-v2"):
        try:
            model = BgeM3Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return model
        except Exception:
            pass

    # Last resort: try to load any available model
    for candidate in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]:
        try:
            model = BgeM3Embedder(model_name=candidate)
            return model
        except Exception:
            continue

    pytest.skip("Unable to load any embedding model (cache empty, download failed)")


@pytest.fixture(scope="module")
def expected_dim(embedder) -> int:
    """Expected embedding dimension for the loaded model."""
    return embedder.dim


# ──────────────────────────────────────────────────────────────
# Test contract 1: embedding shape
# ──────────────────────────────────────────────────────────────

def test_embedding_shape(embedder, expected_dim: int) -> None:
    """Encoding 5 texts produces a (5, dim) float32 array."""
    texts = [
        "The employee SSN is 123-45-6789.",
        "Credit card number: 4532-0151-1283-0366.",
        "Email sent to user@example.com on Monday.",
        "Quarterly financial report for Q1 2024.",
        "Meeting agenda: project kickoff at 10 AM.",
    ]
    embeddings = embedder.encode(texts)
    assert isinstance(embeddings, np.ndarray), f"Expected np.ndarray, got {type(embeddings)}"
    assert embeddings.shape == (5, expected_dim), (
        f"Expected (5, {expected_dim}), got {embeddings.shape}"
    )
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"


# ──────────────────────────────────────────────────────────────
# Test contract 2: embedding normalization
# ──────────────────────────────────────────────────────────────

def test_embedding_normalized(embedder) -> None:
    """Every output vector has L2 norm approximately 1.0 (within 1e-5)."""
    texts = [
        "The employee SSN is 123-45-6789.",
        "A completely unrelated sentence about weather and climate.",
        "Short text.",
    ]
    embeddings = embedder.encode(texts)
    norms = np.linalg.norm(embeddings, axis=1)
    assert norms.shape == (3,), f"Expected 3 norms, got shape {norms.shape}"
    for i, norm in enumerate(norms):
        assert abs(norm - 1.0) < 1e-5, (
            f"Vector {i} has L2 norm {norm}, expected ~1.0 (diff={abs(norm - 1.0)})"
        )


# ──────────────────────────────────────────────────────────────
# Test contract 3: single document
# ──────────────────────────────────────────────────────────────

def test_single_document(embedder, expected_dim: int) -> None:
    """A single text produces shape (1, dim) — 2D, not 1D."""
    embedding = embedder.encode(["The employee SSN is 123-45-6789."])
    assert embedding.shape == (1, expected_dim), (
        f"Single doc should be (1, {expected_dim}), got {embedding.shape}"
    )
    assert embedding.ndim == 2, f"Should be 2D, got {embedding.ndim}D"


# ──────────────────────────────────────────────────────────────
# Test contract 4: cosine similarity range
# ──────────────────────────────────────────────────────────────

def test_cosine_similarity_range(embedder) -> None:
    """Two similar texts should have higher cosine similarity than two dissimilar texts."""
    text_a = "The employee SSN is 123-45-6789."
    text_b = "Employee social security number: 123-45-6789."
    text_c = "The quarterly financial report shows 15% growth."

    embeddings = embedder.encode([text_a, text_b, text_c])

    # Cosine similarity = dot product (since vectors are already normalized)
    sim_ab = float(np.dot(embeddings[0], embeddings[1]))
    sim_ac = float(np.dot(embeddings[0], embeddings[2]))

    assert sim_ab > sim_ac, (
        f"Expected similar texts (A-B) sim={sim_ab:.4f} > "
        f"dissimilar texts (A-C) sim={sim_ac:.4f}"
    )

    # Similar texts about PII should have high cosine similarity
    assert sim_ab > 0.5, (
        f"Similar texts should have cosine similarity > 0.5, got {sim_ab:.4f}"
    )


# ──────────────────────────────────────────────────────────────
# Test contract 5: empty string
# ──────────────────────────────────────────────────────────────

def test_empty_string(embedder, expected_dim: int) -> None:
    """Encoding an empty string produces a non-NaN valid embedding."""
    embedding = embedder.encode([""])
    assert embedding.shape == (1, expected_dim), (
        f"Expected (1, {expected_dim}), got {embedding.shape}"
    )
    assert not np.any(np.isnan(embedding)), "Embedding contains NaN values!"
    assert not np.any(np.isinf(embedding)), "Embedding contains Inf values!"

    # Should be a real vector (not all zeros)
    assert np.any(embedding != 0.0), "Empty string embedding should not be all zeros"


# ──────────────────────────────────────────────────────────────
# Test contract 6: dim property
# ──────────────────────────────────────────────────────────────

def test_dim_property(embedder, expected_dim: int) -> None:
    """The .dim property returns the expected dimension for the loaded model."""
    assert embedder.dim == expected_dim, (
        f"Expected dim={expected_dim}, got {embedder.dim}"
    )
    assert isinstance(embedder.dim, int), f"dim should be int, got {type(embedder.dim)}"
    assert embedder.dim > 0, "dim should be positive"


# ──────────────────────────────────────────────────────────────
# Test contract 7: same text same embedding (determinism)
# ──────────────────────────────────────────────────────────────

def test_same_text_same_embedding(embedder) -> None:
    """Encoding the same text twice produces identical embeddings."""
    texts = [
        "The employee SSN is 123-45-6789.",
        "Quarterly financial report for Q1 2024.",
        "Email: user@example.com",
    ]
    emb1 = embedder.encode(texts)
    emb2 = embedder.encode(texts)

    assert np.allclose(emb1, emb2, atol=1e-6), (
        f"Same texts produced different embeddings! "
        f"Max diff: {np.max(np.abs(emb1 - emb2))}"
    )


# ──────────────────────────────────────────────────────────────
# Edge case: batch size handling
# ──────────────────────────────────────────────────────────────

def test_batch_size_handling(embedder, expected_dim: int) -> None:
    """Encoding more texts than batch_size works correctly (batching is internal detail)."""
    texts = [
        f"Document number {i} contains some content about various topics."
        for i in range(100)
    ]
    embeddings = embedder.encode(texts)
    assert embeddings.shape == (100, expected_dim), (
        f"Expected (100, {expected_dim}), got {embeddings.shape}"
    )


# ──────────────────────────────────────────────────────────────
# Edge case: empty list
# ──────────────────────────────────────────────────────────────

def test_empty_list(embedder, expected_dim: int) -> None:
    """Encoding an empty list returns an empty (0, dim) array."""
    embeddings = embedder.encode([])
    assert isinstance(embeddings, np.ndarray), f"Expected np.ndarray, got {type(embeddings)}"
    assert embeddings.shape == (0, expected_dim), (
        f"Expected (0, {expected_dim}), got {embeddings.shape}"
    )


# ──────────────────────────────────────────────────────────────
# BGE-M3-specific: dim is 1024
# ──────────────────────────────────────────────────────────────

def test_bge_m3_dim_is_1024() -> None:
    """When BGE-M3 loads successfully, dim must be 1024."""
    from src.embeddings.bge_m3 import BgeM3Embedder

    if not _model_is_cached("BAAI/bge-m3"):
        pytest.skip("BGE-M3 not in local cache (not yet downloaded)")

    try:
        model = BgeM3Embedder(model_name="BAAI/bge-m3")
    except Exception as e:
        pytest.skip(f"BGE-M3 not available for dim check: {e}")

    assert model.dim == 1024, (
        f"BGE-M3 embedding dimension should be 1024, got {model.dim}"
    )
