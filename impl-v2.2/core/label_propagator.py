"""
Cluster-level classification & LLM semantic naming

Architecture:
  1. Select representative documents per cluster (centroid-nearest + MMR diversity)
  2. Extract TF-IDF keywords from cluster
  3. (Optional) LLM classifies the CLUSTER ITSELF (not a subset of documents)
  4. All documents inherit their cluster's label

NOTE: Classification targets the cluster as a unit (via centroid vector + keywords),
not a subset of documents within the cluster. Documents receive labels through
direct inheritance from their assigned cluster, not through a propagation step.
"""

import numpy as np
import logging
import json
from typing import List, Dict, Optional
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import ProcessedDocument, ClusterInfo

logger = logging.getLogger(__name__)

# ── Multi-language tokenizer registry ────────────────────────

_SPACY_MODELS = {}

def _get_tokenizer(lang: str):
    """Load and cache spaCy model for the detected language."""
    global _SPACY_MODELS
    if lang in _SPACY_MODELS:
        return _SPACY_MODELS[lang]

    import spacy
    MODEL_MAP = {
        "en": "en_core_web_sm",
        "ja": "ja_core_news_sm",
        "zh": "zh_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "ko": "ko_core_news_sm",
        "pt": "pt_core_news_sm",
    }
    model_name = MODEL_MAP.get(lang, "xx_ent_wiki_sm")  # fallback: multilingual
    try:
        nlp = spacy.load(model_name, disable=["ner", "parser"])
        _SPACY_MODELS[lang] = nlp
        return nlp
    except OSError:
        nlp = spacy.load("xx_ent_wiki_sm", disable=["ner", "parser"])
        _SPACY_MODELS[lang] = nlp
        return nlp


def tokenize_text(text: str, lang: str = "en") -> str:
    """Tokenize text using spaCy, return space-separated tokens."""
    if lang == "vi":
        # Vietnamese: use underthesea if available, else whitespace
        try:
            from underthesea import word_tokenize
            tokens = word_tokenize(text)
            return " ".join(tokens)
        except ImportError:
            return text  # Vietnamese is whitespace-delimited
    nlp = _get_tokenizer(lang)
    doc = nlp(text[:100000])
    return " ".join(token.text for token in doc if not token.is_space)


class LabelPropagator:

    def __init__(self, config: dict, llm_config: dict = None):
        self.sample_per_cluster = config.get("sample_per_cluster", 5)
        self.use_llm = config.get("use_llm", False)

        self.llm_config = llm_config or {}
        self.llm_client = None

    # ── 主入口 ────────────────────────────────────────────────

    def process_clusters(
        self,
        documents: List[ProcessedDocument],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> List[ClusterInfo]:
        """
        对聚类结果进行分类传播和语义命名。
        """
        clusters = []
        unique_labels = sorted(set(labels))

        for cid in unique_labels:
            if cid == -1:
                continue

            mask = labels == cid
            cluster_docs = [documents[i] for i in range(len(documents)) if mask[i]]
            cluster_embs = embeddings[mask]

            # 1. 选择代表性文档
            rep_indices = self._select_representatives(cluster_embs)
            rep_docs = [cluster_docs[i] for i in rep_indices]
            rep_doc_ids = [d.id for d in rep_docs]

            # 2. 提取关键词
            all_texts = [d.raw_content for d in cluster_docs]
            keywords = self._extract_keywords(all_texts, top_n=15)

            # 3. 计算内聚度
            coherence = self._compute_coherence(cluster_embs)

            # 4. LLM 语义命名（可选）
            llm_label = ""
            llm_description = ""
            if self.use_llm:
                llm_label, llm_description = self._llm_name_cluster(
                    keywords, [d.raw_content[:500] for d in rep_docs]
                )

            cluster_info = ClusterInfo(
                cluster_id=cid,
                size=len(cluster_docs),
                keywords=keywords,
                llm_label=llm_label,
                llm_description=llm_description,
                coherence=coherence,
                representative_doc_ids=rep_doc_ids,
                document_ids=[d.id for d in cluster_docs],
            )
            clusters.append(cluster_info)

            # 5. 分类传播：将标签写回每篇文档
            label_to_propagate = llm_label or "_".join(keywords[:3])
            for doc in cluster_docs:
                doc.cluster_id = cid
                doc.classification_source = "clustering"

        clusters.sort(key=lambda c: c.size, reverse=True)
        logger.info(f"已处理 {len(clusters)} 个簇的标签传播")
        return clusters

    # ── 代表性文档选择（MMR） ─────────────────────────────────

    def _select_representatives(self, embeddings: np.ndarray) -> List[int]:
        """
        使用 MMR（Maximal Marginal Relevance）选择代表性文档。
        兼顾"靠近中心"和"互相多样"。
        """
        n = len(embeddings)
        k = min(self.sample_per_cluster, n)
        if k >= n:
            return list(range(n))

        centroid = embeddings.mean(axis=0).reshape(1, -1)
        relevance = cosine_similarity(embeddings, centroid).flatten()

        selected = [int(np.argmax(relevance))]  # 第一个选最靠近中心的

        for _ in range(k - 1):
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break

            best_idx = None
            best_score = -float('inf')

            selected_embs = embeddings[selected]

            for idx in remaining:
                rel = relevance[idx]
                # 与已选文档的最大相似度
                sim_to_selected = cosine_similarity(
                    embeddings[idx].reshape(1, -1), selected_embs
                ).max()
                # MMR: 0.7 * relevance - 0.3 * redundancy
                score = 0.7 * rel - 0.3 * sim_to_selected
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)

        return selected

    # ── Keyword extraction (multilingual) ──────────────────────

    def _extract_keywords(self, texts: List[str], top_n: int = 15) -> List[str]:
        """Extract keywords using spaCy tokenization + TF-IDF (language-aware)."""
        # Detect dominant language of the cluster
        lang = self._detect_language(texts[0][:500]) if texts else "en"

        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(tokenize_text(text[:5000], lang))

        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                max_df=0.8,
                min_df=max(1, len(texts) // 10),
                token_pattern=r'(?u)\b\w{2,}\b',
            )
            tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
            feature_names = vectorizer.get_feature_names_out()

            total_tfidf = tfidf_matrix.sum(axis=0).A1
            top_indices = total_tfidf.argsort()[::-1][:top_n * 2]
            candidates = [feature_names[i] for i in top_indices]

            # Filter stopwords (universal) and pure digits
            universal_stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
                'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and',
                'not', 'no', 'but', 'if', 'this', 'that', 'it', 'its',
            }
            keywords = [w for w in candidates
                        if w.lower() not in universal_stopwords
                        and not w.isdigit()
                        and len(w) >= 2]
            return keywords[:top_n]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language of text snippet."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"  # Default to English

    # ── LLM cluster naming (English prompt) ──────────────────

    def _llm_name_cluster(
        self, keywords: List[str], sample_texts: List[str]
    ) -> tuple:
        """Call LLM to generate a business label for the cluster."""
        try:
            if not self.llm_client:
                self._init_llm_client()

            prompt = f"""You are an enterprise document classification expert.
Based on the following information, generate a classification label for a group of documents.

Keywords: {', '.join(keywords[:10])}

Representative document excerpts:
{chr(10).join(f'- {t[:200]}' for t in sample_texts[:3])}

Return JSON format only:
{{"label": "short business label (max 5 words)", "description": "one-sentence description of this document category"}}

Return ONLY the JSON, nothing else."""

            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "qwen2.5-7b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.llm_config.get("max_tokens", 200),
                temperature=self.llm_config.get("temperature", 0.3),
            )

            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            return result.get("label", ""), result.get("description", "")

        except Exception as e:
            logger.warning(f"LLM naming failed: {e}")
            return "_".join(keywords[:3]), ""

    def _init_llm_client(self):
        """初始化 LLM 客户端（兼容 OpenAI API 格式）"""
        from openai import OpenAI
        self.llm_client = OpenAI(
            api_key="not-needed",
            base_url=self.llm_config.get("api_base", "http://localhost:8000/v1"),
        )

    @staticmethod
    def _compute_coherence(embeddings: np.ndarray) -> float:
        if len(embeddings) < 2:
            return 1.0
        sim = cosine_similarity(embeddings)
        n = len(sim)
        return float(sim[np.triu_indices(n, k=1)].mean())
