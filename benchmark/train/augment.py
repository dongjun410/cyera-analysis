# benchmark/train/augment.py
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Entity Pools ---

_FIRST_NAMES = [
    "Michael", "Sarah", "David", "Emily", "Robert", "Jennifer", "William",
    "Lisa", "James", "Maria", "Thomas", "Patricia", "Daniel", "Linda",
    "Richard", "Barbara", "Joseph", "Susan", "Charles", "Jessica",
]

_LAST_NAMES = [
    "Chen", "Kumar", "Martinez", "Johnson", "Williams", "Brown", "Garcia",
    "Miller", "Davis", "Rodriguez", "Wilson", "Anderson", "Taylor", "Thomas",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
]

_COMPANY_SUFFIXES = [
    "Corporation", "Inc.", "LLC", "Group", "Holdings", "Enterprises",
    "Technologies", "Solutions", "Partners", "International",
]

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _random_name() -> str:
    return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"


def _random_company() -> str:
    prefixes = ["Apex", "Meridian", "Nova", "Pinnacle", "Catalyst", "Horizon",
                "Vertex", "Atlas", "Titan", "Omega", "Spectrum", "Fusion",
                "Quantum", "Zenith", "Orion"]
    return f"{random.choice(prefixes)} {random.choice(_COMPANY_SUFFIXES)}"


def _random_date() -> str:
    year = random.randint(2018, 2025)
    month = random.choice(_MONTHS)
    day = random.randint(1, 28)
    return f"{month} {day}, {year}"


def _random_amount() -> str:
    amount = random.uniform(1000, 10000000)
    if amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    return f"${amount:,.0f}"


# --- Entity Substitution ---

_PERSON_PATTERN = re.compile(
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
)
_DATE_PATTERNS = [
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|'
               r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'),
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
]
_AMOUNT_PATTERN = re.compile(
    r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|M|B|K|k))?'
)
_COMPANY_PATTERNS = [
    re.compile(
        r'\b[A-Z][a-z]+ (?:Corporation|Inc\.?|LLC|Group|Holdings|'
        r'Enterprises|Technologies|Solutions|Partners|International|Ltd\.?)\b'
    ),
]


def entity_substitution(text: str) -> str:
    """Replace named entities with synthetic alternatives."""
    result = text

    for pat in _DATE_PATTERNS:
        def _date_repl(m):
            return _random_date()
        result = pat.sub(_date_repl, result)

    def _amount_repl(m):
        return _random_amount()
    result = _AMOUNT_PATTERN.sub(_amount_repl, result)

    for pat in _COMPANY_PATTERNS:
        def _company_repl(m):
            return _random_company()
        result = pat.sub(_company_repl, result)

    def _name_repl(m):
        return _random_name()
    result = _PERSON_PATTERN.sub(_name_repl, result)

    return result


# --- TF-IDF Quality Filter ---

def tfidf_quality_filter(
    candidates: List[str],
    originals: List[str],
    min_sim: float = 0.15,
    max_sim: float = 0.95,
) -> List[str]:
    """Filter augmented documents by TF-IDF cosine similarity."""
    if not candidates:
        return []

    all_texts = originals + candidates
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n_originals = len(originals)
    kept: List[str] = []
    for i, candidate in enumerate(candidates):
        orig_idx = min(i % n_originals, n_originals - 1)
        sim = cosine_similarity(
            tfidf_matrix[orig_idx:orig_idx + 1],
            tfidf_matrix[n_originals + i:n_originals + i + 1],
        )[0][0]
        if min_sim <= sim <= max_sim:
            kept.append(candidate)

    return kept


# --- Back-Translation ---

_MT_MODELS: Dict[str, tuple] = {}


def _get_mt_model(src_lang: str, tgt_lang: str):
    key = f"{src_lang}-{tgt_lang}"
    if key not in _MT_MODELS:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _MT_MODELS[key] = (tokenizer, model)
    return _MT_MODELS[key]


def back_translate(text: str, target_lang: str = "de") -> str:
    """EN -> target_lang -> EN back-translation."""
    tok_fwd, model_fwd = _get_mt_model("en", target_lang)
    inputs = tok_fwd(text, return_tensors="pt", truncation=True, max_length=512)
    translated = model_fwd.generate(**inputs, max_new_tokens=256)
    mid_text = tok_fwd.decode(translated[0], skip_special_tokens=True)

    tok_back, model_back = _get_mt_model(target_lang, "en")
    inputs = tok_back(mid_text, return_tensors="pt", truncation=True, max_length=512)
    back = model_back.generate(**inputs, max_new_tokens=256)
    return tok_back.decode(back[0], skip_special_tokens=True)


# --- LLM Synthesis ---

_LLM_SYNTHESIS_PROMPT = (
    "You are a document generator for a data security classification system. "
    "Generate a realistic, professional document that would be classified as:\n"
    "L1 Category: {l1_label}\n"
    "L2 Subcategory: {l2_label}\n\n"
    "Taxonomy definition:\n{taxonomy_def}\n\n"
    "The document should:\n"
    "- Be 150-400 words long\n"
    "- Include realistic structure, terminology, and formatting typical of this document type\n"
    "- Contain plausible entity names, dates, and numbers\n"
    "- NOT mention the classification label explicitly in the text\n\n"
    "Generated document:"
)


def llm_synthesize(
    l1_label: str,
    l2_label: str,
    taxonomy_def: str = "",
    n: int = 5,
    ollama_url: str = "http://localhost:8003/api/generate",
    model: str = "gemma4:e2b",
) -> List[str]:
    """Use local Gemma4 to generate synthetic DSPM documents."""
    import json
    import urllib.request

    results = []
    for _ in range(n):
        prompt = _LLM_SYNTHESIS_PROMPT.format(
            l1_label=l1_label,
            l2_label=l2_label,
            taxonomy_def=taxonomy_def or f"{l1_label} / {l2_label}",
        )
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 512},
        })
        req = urllib.request.Request(
            ollama_url,
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results.append(data.get("response", "").strip())
        except Exception as e:
            print(f"LLM synthesis failed: {e}")
            continue

    return results


# --- Augmenter (orchestrates all 3 layers) ---

class Augmenter:
    """Orchestrates multi-layer DSPM data augmentation with quality filtering."""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self._bt_langs = ["de", "zh", "fr"]

    def augment(
        self,
        texts: List[str],
        labels: List[Dict[str, str]],
        taxonomy_defs: Dict[str, str] | None = None,
    ) -> tuple[List[str], List[Dict[str, str]]]:
        """Augment a dataset. Returns (augmented_texts, augmented_labels)."""
        all_texts = list(texts)
        all_labels = list(labels)

        # Layer 1: Entity substitution (always works, no external deps)
        for _ in range(self.cfg.augment_entity_sub_count):
            for text, label in zip(texts, labels):
                variant = entity_substitution(text)
                all_texts.append(variant)
                all_labels.append(label.copy())

        # Layer 2: Back-translation (Helsinki NLP, lazy-loaded in back_translate function)
        # Only attempt if importable
        try:
            from benchmark.train.augment import back_translate  # noqa: F811
            for i in range(min(self.cfg.augment_back_translation_count, len(self._bt_langs))):
                lang = self._bt_langs[i]
                for text, label in zip(texts, labels):
                    try:
                        bt_text = back_translate(text, target_lang=lang)
                        if bt_text != text:
                            all_texts.append(bt_text)
                            all_labels.append(label.copy())
                    except Exception:
                        continue
        except ImportError:
            pass  # Helsinki NLP not available

        # Layer 3: LLM synthesis (optional, requires Gemma4 running)
        # Skipped if taxonomy_defs not provided

        # Quality filter on augmented portion only
        n_original = len(texts)
        if n_original < len(all_texts):
            augmented_texts = all_texts[n_original:]
            kept = tfidf_quality_filter(
                augmented_texts,
                texts,
                min_sim=self.cfg.augment_quality_min_similarity,
                max_sim=self.cfg.augment_quality_max_similarity,
            )
            kept_texts: List[str] = []
            kept_labels: List[Dict[str, str]] = []
            kept_set = set(kept)
            for t, l in zip(augmented_texts, all_labels[n_original:]):
                if t in kept_set:
                    kept_texts.append(t)
                    kept_labels.append(l)
            all_texts = texts + kept_texts
            all_labels = labels + kept_labels

        return all_texts, all_labels
