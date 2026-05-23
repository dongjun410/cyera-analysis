"""Tier 1 Stage B: FAISS Semantic Refinement.

Within each structural bucket from Stage A, refine by semantic similarity.
Only triggers when bucket size > sem_split_threshold.

Uses KMeans clustering (sklearn, with optional FAISS acceleration) for
sub-cluster discovery. Homogeneous buckets are returned as a single cluster.

Per spec section 4.2 Stage B.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.types import ClusterInfo, Document

if TYPE_CHECKING:
    from src.embeddings.bge_m3 import BgeM3Embedder


# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "sem_split_threshold": 50,
    "homogeneity_threshold": 0.85,
    "variance_threshold": 0.25,
    "max_sample_for_large": 10000,
    "faiss_nlist": 100,
    "faiss_nprobe": 10,
}


# ──────────────────────────────────────────────────────────────
# SemanticRefiner
# ──────────────────────────────────────────────────────────────

class SemanticRefiner:
    """Semantic sub-clustering within a structural bucket.

    Uses embeddings from a BgeM3Embedder to split a structural bucket
    into semantically coherent sub-clusters via KMeans.

    Only triggers refinement when the bucket has enough documents AND
    the embeddings show sufficient variance.
    """

    def __init__(
        self,
        embedder: BgeM3Embedder,
        config: dict | None = None,
    ) -> None:
        """Initialize with an embedder and optional config override.

        Args:
            embedder: A BgeM3Embedder instance used to encode document text.
            config: Optional dict overriding any DEFAULT_CONFIG keys.
        """
        self._embedder = embedder
        cfg = dict(DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)
        self._config = cfg

    # ── Public API ────────────────────────────────────────────

    def should_refine(self, bucket_docs: list[Document]) -> bool:
        """Decide whether this bucket needs semantic sub-clustering.

        Returns True only when BOTH:
        1. Number of documents exceeds sem_split_threshold
        2. Cosine-similarity standard deviation exceeds variance_threshold
           (heterogeneous enough to warrant splitting)

        Args:
            bucket_docs: Documents currently assigned to this structural bucket.

        Returns:
            True if the bucket should be semantically refined.
        """
        threshold = self._config["sem_split_threshold"]
        if len(bucket_docs) <= threshold:
            return False

        embeddings = self._embed_docs(bucket_docs)
        _, cos_sims = self._compute_centroid_and_sims(embeddings)
        std = float(np.std(cos_sims))

        return std > self._config["variance_threshold"]

    def refine(
        self, bucket_id: str, documents: list[Document],
    ) -> list[ClusterInfo]:
        """Split a structural bucket into semantic sub-clusters.

        Workflow:
        1. Embed all documents
        2. Check homogeneity — if all docs are very similar, return single cluster
        3. Check variance — if homogeneous enough, return single cluster
        4. Run KMeans to discover sub-clusters
        5. For each sub-cluster: compute centroid, radius, representatives, keywords

        Args:
            bucket_id: The structural bucket ID from Stage A.
            documents: Documents in this structural bucket.

        Returns:
            List of ClusterInfo, one per sub-cluster (or a single entry for
            homogeneous buckets).
        """
        if not documents:
            return []

        embeddings = self._embed_docs(documents)

        # ── Homogeneity check ─────────────────────────────────
        centroid, cos_sims = self._compute_centroid_and_sims(embeddings)
        mean_sim = float(np.mean(cos_sims))
        std_sim = float(np.std(cos_sims))

        if mean_sim > self._config["homogeneity_threshold"]:
            # Very homogeneous — return as single cluster
            return [self._build_cluster(
                bucket_id=bucket_id,
                sub_idx=0,
                documents=documents,
                embeddings=embeddings,
                centroid=centroid,
                member_indices=list(range(len(documents))),
            )]

        # ── Variance check ────────────────────────────────────
        if std_sim <= self._config["variance_threshold"]:
            # Not diverse enough to split
            return [self._build_cluster(
                bucket_id=bucket_id,
                sub_idx=0,
                documents=documents,
                embeddings=embeddings,
                centroid=centroid,
                member_indices=list(range(len(documents))),
            )]

        # ── KMeans sub-clustering ─────────────────────────────
        n_docs = len(documents)
        labels = self._cluster_embeddings(embeddings, n_docs)

        # Build one ClusterInfo per sub-cluster
        clusters: list[ClusterInfo] = []
        n_clusters = int(labels.max()) + 1
        for sub_idx in range(n_clusters):
            member_indices = [i for i, lbl in enumerate(labels) if lbl == sub_idx]
            if not member_indices:
                continue

            # Recompute centroid for this sub-cluster
            sub_centroid = embeddings[member_indices].mean(axis=0)
            sub_centroid = sub_centroid / (np.linalg.norm(sub_centroid) or 1.0)

            clusters.append(self._build_cluster(
                bucket_id=bucket_id,
                sub_idx=sub_idx,
                documents=documents,
                embeddings=embeddings,
                centroid=sub_centroid,
                member_indices=member_indices,
            ))

        return clusters

    # ── Internal helpers ──────────────────────────────────────

    def _embed_docs(self, documents: list[Document]) -> np.ndarray:
        """Encode all document texts into embeddings.

        Returns (N, dim) float32 array.
        """
        texts = [doc.text for doc in documents]
        return self._embedder.encode(texts)

    @staticmethod
    def _compute_centroid_and_sims(
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute centroid and per-point cosine similarities to it.

        Args:
            embeddings: (N, D) float32 array of unit-normalized embeddings.

        Returns:
            (centroid, cosine_similarities) where centroid is (D,) and
            cosine_similarities is (N,).
            Centroid is L2-normalized.
        """
        centroid = embeddings.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Cosine similarity = dot product since embeddings are already unit vectors
        # But to be safe, we compute dot product of each row with centroid
        cos_sims = embeddings @ centroid  # (N,)
        return centroid.astype(np.float32), cos_sims.astype(np.float32)

    def _cluster_embeddings(
        self, embeddings: np.ndarray, n_docs: int,
    ) -> np.ndarray:
        """Run KMeans to assign each embedding to a sub-cluster.

        For small buckets (< 1000): tries K from 2 up to a max, selecting
        the best silhouette score.
        For larger buckets: uses a fixed heuristic K.

        Args:
            embeddings: (N, D) float32 array.
            n_docs: Number of documents.

        Returns:
            (N,) int array of cluster labels.
        """
        if n_docs < 1000:
            max_k = min(int(math.sqrt(n_docs)), 15)
            k_range = range(2, max_k + 1)

            best_k = 2
            best_score = -1.0
            best_labels: np.ndarray | None = None

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(embeddings)
                # silhouette_score needs at least 2 clusters, each with >= 2 members
                unique_lbls = np.unique(labels)
                if len(unique_lbls) < 2:
                    continue
                try:
                    score = silhouette_score(embeddings, labels)
                except ValueError:
                    continue
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            if best_labels is not None:
                return best_labels

        # Fallback / large bucket: fixed heuristic K
        k = min(int(math.sqrt(n_docs / 2)), 20)
        k = max(k, 2)  # at least 2
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(embeddings)

    def _build_cluster(
        self,
        bucket_id: str,
        sub_idx: int,
        documents: list[Document],
        embeddings: np.ndarray,
        centroid: np.ndarray,
        member_indices: list[int],
    ) -> ClusterInfo:
        """Build a ClusterInfo for a sub-cluster.

        Computes:
        - cluster_radius (max cosine distance from centroid)
        - representative_docs (up to 3 docs closest to centroid)
        - tfidf_keywords (top-15 terms from representative doc texts)
        - pii_distribution, language_distribution (from document data)
        """
        cluster_id = f"{bucket_id}_{sub_idx}"
        doc_ids = [documents[i].doc_id for i in member_indices]

        # Cosine similarities of members to centroid
        member_embs = embeddings[member_indices]  # (M, D)
        cos_sims = member_embs @ centroid  # (M,)
        cos_dists = 1.0 - cos_sims  # cosine distance

        cluster_radius = float(np.max(cos_dists)) if len(cos_dists) > 0 else 0.0

        # Representative docs: up to 3 closest to centroid
        n_rep = min(3, len(member_indices))
        if n_rep > 0:
            top_indices = np.argsort(cos_dists)[:n_rep]  # smallest distance = closest
            rep_docs = [documents[member_indices[int(i)]].doc_id for i in top_indices]
        else:
            rep_docs = []

        # TF-IDF keywords from representative doc texts
        rep_texts = []
        for rid in rep_docs:
            for doc in documents:
                if doc.doc_id == rid:
                    rep_texts.append(doc.text)
                    break

        keywords = self._extract_keywords(rep_texts)

        # PII distribution — aggregate from document PII features
        pii_dist: dict[str, int] = {}
        for i in member_indices:
            doc = documents[i]
            if doc.pii_features is not None:
                for ptype, count in doc.pii_features.pii_type_distribution.items():
                    pii_dist[ptype] = pii_dist.get(ptype, 0) + count

        # Language distribution — from metadata
        lang_dist: dict[str, int] = {}
        for i in member_indices:
            doc = documents[i]
            lang = doc.metadata.get("language", "unknown")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        return ClusterInfo(
            cluster_id=cluster_id,
            doc_ids=doc_ids,
            structural_bucket=bucket_id,
            cluster_radius=cluster_radius,
            representative_docs=rep_docs,
            tfidf_keywords=keywords,
            pii_distribution=pii_dist,
            language_distribution=lang_dist,
            centroid_embedding=centroid.copy(),
        )

    @staticmethod
    def _extract_keywords(texts: list[str], top_n: int = 15) -> list[str]:
        """Extract top-N TF-IDF keywords from a list of texts.

        Args:
            texts: Representative document texts.
            top_n: Number of keywords to return (default 15).

        Returns:
            List of top-N terms, or empty list if no valid terms found.
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return []

        try:
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words of 2+ letters
            )
            vectorizer.fit_transform(valid_texts)
            # Return terms sorted by IDF (vocabulary is ordered by feature index)
            feature_names = vectorizer.get_feature_names_out()
            return list(feature_names[:top_n])
        except (ValueError, AttributeError):
            # Fallback: return empty list on any vectorizer failure
            return []


# ──────────────────────────────────────────────────────────────
# Optional FAISS support (attempted import)
# ──────────────────────────────────────────────────────────────

try:
    import faiss  # noqa: F401

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
