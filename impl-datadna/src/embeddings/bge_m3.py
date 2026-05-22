"""BGE-M3 embedding service.

Thin wrapper around sentence-transformers for BGE-M3.
Produces normalized unit vectors (L2 norm = 1.0) for cosine similarity.

BGE-M3 is a multilingual embedding model from BAAI that supports:
- Dense embeddings (1024-dim, used here)
- Sparse (lexical) embeddings
- Multi-vector (ColBERT) embeddings

We use only the dense embeddings for document clustering.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class BgeM3Embedder:
    """Thin wrapper around SentenceTransformer for BGE-M3 dense embeddings.

    Usage:
        embedder = BgeM3Embedder()
        embeddings = embedder.encode(["text one", "text two"])
        assert embeddings.shape == (2, 1024)
        assert embedder.dim == 1024
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length

        self._model: SentenceTransformer

        # sentence-transformers with newer transformers may auto-map "cuda"
        # to a CUDA device; but on CPU-only machines, "cuda" will fail.
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception:
            # Fallback to CPU if CUDA is requested but unavailable
            self._model = SentenceTransformer(model_name, device="cpu")

        # Apply max sequence length
        self._model.max_seq_length = max_length

    def encode(
        self, texts: list[str], show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into normalized embedding vectors.

        Args:
            texts: List of text strings to encode.
            show_progress: Display a tqdm progress bar during encoding.

        Returns:
            numpy array of shape (len(texts), dim) with float32 dtype.
            All vectors are L2-normalized (unit norm).
            Empty input list returns (0, dim) array.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize to unit vectors
            convert_to_numpy=True,
        )
        # sentence-transformers returns float32 by default, but be explicit
        return embeddings.astype(np.float32)

    @property
    def dim(self) -> int:
        """Embedding dimension (1024 for BGE-M3 dense embeddings)."""
        # Prefer the newer method name, fall back to the deprecated one
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()
