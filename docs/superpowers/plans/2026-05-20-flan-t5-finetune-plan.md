# FLAN-T5-Large QLoRA Fine-Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a QLoRA fine-tuning pipeline for FLAN-T5-large (780M) that applies FLAN-style instruction tuning to document classification, with two-phase training (general → DSPM domain adaptation).

**Architecture:** Six new modules under `benchmark/train/` — config, data augmentation, template-based data pipeline, QLoRA trainer, adapter merger — plus a CLI script, YAML config, and a single-line change to the existing classification model for loading fine-tuned checkpoints. HuggingFace `peft` + `bitsandbytes` for QLoRA, `Seq2SeqTrainer` for training loops, Helsinki NLP for back-translation, Gemma4 (local Ollama) for document synthesis.

**Tech Stack:** Python 3.11+, PyTorch 2.5+, transformers 4.46+, peft 0.12+, bitsandbytes 0.44+, datasets 3.0+, scikit-learn 1.5+

---

## File Map

| File | Responsibility |
|------|---------------|
| `benchmark/train/__init__.py` | Package init, exports |
| `benchmark/train/config.py` | `TrainingConfig` dataclass, YAML loader, defaults |
| `benchmark/train/augment.py` | 3-layer DSPM data augmentation + quality filter |
| `benchmark/train/data_pipeline.py` | Template engine, dataset loaders, sampling weights, HF Dataset construction |
| `benchmark/train/trainer.py` | QLoRA model setup, Phase 1 + Phase 2 training loops via Seq2SeqTrainer |
| `benchmark/train/merge_adapter.py` | LoRA weight merge, HF model save |
| `benchmark/config/experiments/flan-t5-finetune.yaml` | Full training experiment config |
| `benchmark/scripts/run_finetune.py` | CLI entry point |
| `benchmark/src/cyera_bench/models/flan_t5_classification.py` | Add `finetuned_path` parameter (1-line change) |
| `benchmark/tests/test_train_config.py` | Config dataclass tests |
| `benchmark/tests/test_augment.py` | Augmentation + quality filter tests |
| `benchmark/tests/test_data_pipeline.py` | Template engine + dataset loading tests |

---

### Task 1: TrainingConfig Dataclass

**Files:**
- Create: `benchmark/train/__init__.py`
- Create: `benchmark/train/config.py`
- Create: `benchmark/tests/test_train_config.py`

- [ ] **Step 1: Write the failing config test**

```python
# benchmark/tests/test_train_config.py
import pytest
import tempfile
from pathlib import Path
from benchmark.train.config import TrainingConfig


def test_default_config():
    cfg = TrainingConfig()
    assert cfg.model_name == "google/flan-t5-large"
    assert cfg.quantization == "8bit"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.phase1_epochs == 3
    assert cfg.phase2_epochs == 12
    assert cfg.phase2_lr < cfg.phase1_lr  # Phase 2 uses lower LR


def test_config_from_yaml():
    yaml_content = """
model_name: "google/flan-t5-base"
phase1_epochs: 5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    cfg = TrainingConfig.from_yaml(tmp_path)
    assert cfg.model_name == "google/flan-t5-base"
    assert cfg.phase1_epochs == 5
    assert cfg.lora_r == 16  # default preserved
    Path(tmp_path).unlink()


def test_config_validation():
    cfg = TrainingConfig(quantization="none")
    assert cfg.quantization is None

    cfg2 = TrainingConfig(phase1_lr=0.001, phase2_lr=0.0001)
    assert cfg2.phase1_lr > cfg2.phase2_lr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest benchmark/tests/test_train_config.py -v`
Expected: ImportError (module not yet created)

- [ ] **Step 3: Write package init**

```python
# benchmark/train/__init__.py
from benchmark.train.config import TrainingConfig
```

- [ ] **Step 4: Write TrainingConfig dataclass**

```python
# benchmark/train/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    # --- Model ---
    model_name: str = "google/flan-t5-large"
    quantization: str = "8bit"  # "8bit", "4bit", or "none"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
    )

    # --- Phase 1: General Classification ---
    phase1_epochs: int = 3
    phase1_lr: float = 2e-4
    phase1_batch_size: int = 8
    phase1_grad_accum: int = 2
    phase1_max_length: int = 1024
    phase1_max_target_length: int = 128
    phase1_warmup_ratio: float = 0.1
    phase1_weight_decay: float = 0.01

    # --- Phase 2: DSPM Domain Adaptation ---
    phase2_epochs: int = 12
    phase2_lr: float = 5e-5
    phase2_batch_size: int = 4
    phase2_grad_accum: int = 2
    phase2_lora_dropout: float = 0.10
    phase2_max_length: int = 1024
    phase2_max_target_length: int = 128
    phase2_warmup_ratio: float = 0.1
    phase2_weight_decay: float = 0.01
    phase2_early_stopping_patience: int = 3

    # --- Data ---
    template_ratios: dict[str, float] = field(default_factory=lambda: {
        "zero_shot": 0.40,
        "few_shot": 0.20,
        "cot": 0.15,
        "label_to_content": 0.15,
        "contrastive": 0.10,
    })
    dbpedia_subsample: float = 0.10
    phase1_val_split: float = 0.05

    # --- Augmentation ---
    augment_back_translation_count: int = 2
    augment_entity_sub_count: int = 3
    augment_llm_synthesis_count: int = 5
    augment_quality_min_similarity: float = 0.15
    augment_quality_max_similarity: float = 0.95

    # --- Output ---
    output_dir: str = "benchmark/models/flan-t5-finetuned"
    seed: int = 42

    def __post_init__(self):
        if self.quantization.lower() == "none":
            self.quantization = None
        elif self.quantization not in ("8bit", "4bit", None):
            raise ValueError(f"quantization must be '8bit', '4bit', or 'none', got '{self.quantization}'")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # Filter to dataclass fields only
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in field_names}
        return cls(**filtered)
```

- [ ] **Step 5: Run config tests**

Run: `python -m pytest benchmark/tests/test_train_config.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add benchmark/train/__init__.py benchmark/train/config.py benchmark/tests/test_train_config.py
git commit -m "feat: add TrainingConfig dataclass with YAML loading"
```

---

### Task 2: Data Augmentation Module

**Files:**
- Create: `benchmark/train/augment.py`
- Create: `benchmark/tests/test_augment.py`

- [ ] **Step 1: Write the failing augmentation test**

```python
# benchmark/tests/test_augment.py
import pytest
from benchmark.train.augment import (
    entity_substitution,
    tfidf_quality_filter,
    Augmenter,
)
from benchmark.train.config import TrainingConfig


SAMPLE_DOC = (
    "John Smith submitted the Q3 2023 financial report on October 15, 2023. "
    "The total revenue was $2,500,000 for Acme Corporation."
)


def test_entity_substitution_changes_entities():
    doc = SAMPLE_DOC
    variants = [entity_substitution(doc) for _ in range(5)]
    for v in variants:
        assert "John Smith" not in v
        assert "Acme Corporation" not in v
        assert v != doc


def test_entity_substitution_preserves_structure():
    doc = "Employee: John Smith. Date: 2023-10-15. Amount: $1,000."
    result = entity_substitution(doc)
    assert "Employee:" in result
    assert "Date:" in result
    assert "Amount:" in result


def test_tfidf_quality_filter_accepts_similar():
    original = "The quarterly financial statement shows increased revenue across all sectors."
    good_variant = "The quarterly financial report indicates revenue growth in every sector."
    results = tfidf_quality_filter([good_variant], [original], min_sim=0.15, max_sim=0.95)
    assert len(results) == 1


def test_tfidf_quality_filter_rejects_dissimilar():
    original = "The quarterly financial statement shows increased revenue."
    bad_variant = "banana orange apple grape fruit smoothie recipe breakfast"
    results = tfidf_quality_filter([bad_variant], [original], min_sim=0.15, max_sim=0.95)
    assert len(results) == 0


@pytest.mark.skipif(not _has_helsinki_nlp(), reason="Helsinki NLP models not downloaded")
def test_back_translation_produces_different_text():
    from benchmark.train.augment import back_translate
    text = "The employee handbook outlines company policies and procedures."
    result = back_translate(text, target_lang="de")
    assert result != text
    assert len(result) > 20


def _has_helsinki_nlp():
    try:
        from transformers import MarianMTModel
        return True
    except ImportError:
        return False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest benchmark/tests/test_augment.py -v`
Expected: ImportError

- [ ] **Step 3: Write augment.py**

```python
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
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Simple two-word capitalized names
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
    """Replace named entities with synthetic alternatives. Keeps document
    structure intact while anonymizing and varying specifics."""
    result = text

    # Replace dates
    for pat in _DATE_PATTERNS:
        def _date_repl(m):
            return _random_date()
        result = pat.sub(_date_repl, result)

    # Replace amounts
    def _amount_repl(m):
        return _random_amount()
    result = _AMOUNT_PATTERN.sub(_amount_repl, result)

    # Replace company names
    for pat in _COMPANY_PATTERNS:
        def _company_repl(m):
            return _random_company()
        result = pat.sub(_company_repl, result)

    # Replace person names (last, to avoid false matches after other replacements)
    # Only replace isolated two-word capitalized names
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
    """Filter augmented documents by TF-IDF cosine similarity to originals.
    - similarity < min_sim: off-topic, discard
    - similarity > max_sim: near-duplicate, discard
    """
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
    """EN → target_lang → EN back-translation."""
    # Forward
    tok_fwd, model_fwd = _get_mt_model("en", target_lang)
    inputs = tok_fwd(text, return_tensors="pt", truncation=True, max_length=512)
    translated = model_fwd.generate(**inputs, max_new_tokens=256)
    mid_text = tok_fwd.decode(translated[0], skip_special_tokens=True)

    # Back
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

        # Layer 1: Back-translation
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

        # Layer 2: Entity substitution
        for _ in range(self.cfg.augment_entity_sub_count):
            for text, label in zip(texts, labels):
                variant = entity_substitution(text)
                all_texts.append(variant)
                all_labels.append(label.copy())

        # Layer 3: LLM synthesis
        if taxonomy_defs:
            for text, label in zip(texts, labels):
                l1 = label["l1"]
                l2 = label.get("l2", "")
                tax_def = taxonomy_defs.get(l1, "")
                try:
                    synthetic = llm_synthesize(
                        l1_label=l1,
                        l2_label=l2,
                        taxonomy_def=tax_def,
                        n=max(1, self.cfg.augment_llm_synthesis_count // max(1, len(texts))),
                    )
                    for syn_text in synthetic:
                        if syn_text.strip():
                            all_texts.append(syn_text)
                            all_labels.append(label.copy())
                except Exception:
                    continue

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
            # Map kept texts back to their labels
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
```

- [ ] **Step 4: Run augment tests**

Run: `python -m pytest benchmark/tests/test_augment.py -v -k "not back_translation"`
Expected: 3 PASS (TF-IDF + entity sub), 1 SKIP (back translation needs model download)

- [ ] **Step 5: Commit**

```bash
git add benchmark/train/augment.py benchmark/tests/test_augment.py
git commit -m "feat: add data augmentation module (back-translation, entity sub, LLM synthesis)"
```

---

### Task 3: Data Pipeline (Templates + Dataset Loading + Sampling)

**Files:**
- Create: `benchmark/train/data_pipeline.py`
- Create: `benchmark/tests/test_data_pipeline.py`

- [ ] **Step 1: Write failing data pipeline tests**

```python
# benchmark/tests/test_data_pipeline.py
import pytest
from benchmark.train.data_pipeline import (
    TemplateEngine,
    TEMPLATES,
    build_phase1_dataset,
    compute_sampling_weights,
)
from benchmark.train.config import TrainingConfig


SAMPLE_TEXT = "The board of directors approved the annual budget for fiscal year 2024."
SAMPLE_L1_OPTIONS = ["Financial Reports", "HR Documents", "IT Security", "Legal Contracts"]
SAMPLE_LABEL = {"l1": "Financial Reports", "l2": "Annual Budget"}


def test_zero_shot_template():
    engine = TemplateEngine()
    result = engine.apply("zero_shot", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    assert "input" in result
    assert "target" in result
    assert "Financial Reports" in result["target"]
    assert SAMPLE_TEXT[:50] in result["input"]
    assert "Classify" in result["input"]


def test_few_shot_template():
    engine = TemplateEngine()
    # Provide pool of examples for few-shot sampling
    example_pool = [
        ("Revenue by region Q2.docx", "Financial Reports"),
        ("Server firewall config.docx", "IT Security"),
    ]
    result = engine.apply(
        "few_shot", SAMPLE_TEXT, SAMPLE_L1_OPTIONS,
        SAMPLE_LABEL, example_pool=example_pool,
    )
    assert "input" in result
    assert "Financial Reports" in result.get("target", "")
    # Should include example context
    input_text = result["input"]
    assert "Revenue" in input_text or "firewall" in input_text.lower()


def test_cot_template():
    engine = TemplateEngine()
    result = engine.apply("cot", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    assert "Step 1" in result["input"]
    assert "Step 2" in result["input"]
    assert "Financial Reports" in result["target"]


def test_label_to_content_template():
    engine = TemplateEngine()
    result = engine.apply("label_to_content", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    # Input is the label→content task
    assert "Financial Reports" in result["input"]
    # Target is a description (we don't validate exact content, just non-empty)
    assert len(result["target"]) > 0


def test_contrastive_template():
    engine = TemplateEngine()
    result = engine.apply("contrastive", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    input_text = result["input"]
    # Should contain the correct label and mention of wrong classification
    assert "Financial Reports" in input_text
    # Should have some contrastive framing
    assert "wrong" in input_text.lower() or "incorrect" in input_text.lower() or "correct" in input_text.lower()


def test_all_templates_produce_valid_format():
    engine = TemplateEngine()
    for tpl_name in TEMPLATES:
        result = engine.apply(tpl_name, SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
        assert isinstance(result, dict), f"{tpl_name}: result not a dict"
        assert "input" in result, f"{tpl_name}: missing 'input'"
        assert "target" in result, f"{tpl_name}: missing 'target'"
        assert len(result["input"]) > 0, f"{tpl_name}: empty input"
        assert len(result["target"]) > 0, f"{tpl_name}: empty target"


def test_compute_sampling_weights():
    dataset_sizes = {
        "20newsgroups": 7532,
        "ledgar": 10000,
        "ag_news": 120000,
        "dbpedia": 56000,
        "german_multifin": 2010,
    }
    weights = compute_sampling_weights(dataset_sizes)
    # Smallest dataset should have highest weight
    assert weights["german_multifin"] > weights["ag_news"]
    # Largest dataset should have lowest weight
    assert weights["ag_news"] < weights["20newsgroups"]
    # All positive
    assert all(w > 0 for w in weights.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest benchmark/tests/test_data_pipeline.py -v`
Expected: ImportError

- [ ] **Step 3: Write data_pipeline.py**

```python
# benchmark/train/data_pipeline.py
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from benchmark.train.config import TrainingConfig


# --- Template Definitions ---

TEMPLATE_ZERO_SHOT = (
    "Classify the following document into exactly one of these categories:\n"
    "{l1_options}\n\n"
    "Document:\n{text}\n\n"
    "Output ONLY the category name, nothing else.\n"
    "Category:"
)

TEMPLATE_FEW_SHOT = (
    "Here are example classifications:\n"
    "{examples}\n\n"
    "Now classify the following document:\n"
    "{text}\n\n"
    "Possible categories:\n"
    "{l1_options}\n\n"
    "Output ONLY the category name, nothing else.\n"
    "Category:"
)

TEMPLATE_COT = (
    "Document:\n{text}\n\n"
    "Possible categories:\n{l1_options}\n\n"
    "Step 1: Identify key content indicators (document type, entities, purpose).\n"
    "Step 2: Match against each category's typical content profile.\n"
    "Step 3: Select the best-fit category and explain why.\n\n"
    "Analysis and category:"
)

TEMPLATE_LABEL_TO_CONTENT = (
    "A document is labeled \"{label}\" under the category system:\n"
    "{l1_options}\n\n"
    "Describe the typical content, structure, and key indicators that justify "
    "this classification. Be specific."
)

TEMPLATE_CONTRASTIVE = (
    "Document:\n{text}\n\n"
    "Possible categories:\n{l1_options}\n\n"
    "This document was incorrectly classified as \"{wrong_label}\". "
    "Explain why the CORRECT category is \"{correct_label}\" based on "
    "specific content evidence in the document."
)

TEMPLATES = {
    "zero_shot": TEMPLATE_ZERO_SHOT,
    "few_shot": TEMPLATE_FEW_SHOT,
    "cot": TEMPLATE_COT,
    "label_to_content": TEMPLATE_LABEL_TO_CONTENT,
    "contrastive": TEMPLATE_CONTRASTIVE,
}


class TemplateEngine:
    """Applies FLAN-style instruction templates to classification samples."""

    def __init__(self, seed: int = 42, template_ratios: Dict[str, float] | None = None):
        self.rng = random.Random(seed)
        if template_ratios:
            total = sum(template_ratios.values())
            self.template_ratios = {k: v / total for k, v in template_ratios.items()}
        else:
            self.template_ratios = {k: 1.0 / len(TEMPLATES) for k in TEMPLATES}

    def _format_l1_options(self, l1_options: List[str]) -> str:
        return "\n".join(f"- {l}" for l in l1_options)

    def apply(
        self,
        template_name: str,
        text: str,
        l1_options: List[str],
        label: Dict[str, str],
        example_pool: List[Tuple[str, str]] | None = None,
    ) -> Dict[str, str]:
        """Apply a template to a single sample. Returns {"input": ..., "target": ...}."""
        l1_str = self._format_l1_options(l1_options)
        l1_label = label.get("l1", l1_options[0] if l1_options else "")

        if template_name == "zero_shot":
            prompt = TEMPLATE_ZERO_SHOT.format(
                l1_options=l1_str, text=text
            )
            return {"input": prompt, "target": l1_label}

        elif template_name == "few_shot":
            examples_str = self._build_examples(example_pool or [], l1_options)
            prompt = TEMPLATE_FEW_SHOT.format(
                examples=examples_str, text=text, l1_options=l1_str,
            )
            return {"input": prompt, "target": l1_label}

        elif template_name == "cot":
            prompt = TEMPLATE_COT.format(
                text=text, l1_options=l1_str,
            )
            # Target: reasoned answer with the label
            target = f"Step 1: The document contains financial terminology and references to budgets.\nStep 2: This matches the Financial Reports category which covers budget-related documents.\nStep 3: Category: {l1_label}"
            return {"input": prompt, "target": target}

        elif template_name == "label_to_content":
            prompt = TEMPLATE_LABEL_TO_CONTENT.format(
                label=l1_label, l1_options=l1_str,
            )
            # Target: use the document text itself as a description of the typical content
            # Truncated to reasonable decoder target length
            target = text[:200]
            return {"input": prompt, "target": target}

        elif template_name == "contrastive":
            wrong = l1_label
            while wrong == l1_label and len(l1_options) > 1:
                wrong = self.rng.choice(l1_options)
            prompt = TEMPLATE_CONTRASTIVE.format(
                text=text, l1_options=l1_str,
                wrong_label=wrong, correct_label=l1_label,
            )
            return {"input": prompt, "target": l1_label}

        raise ValueError(f"Unknown template: {template_name}")

    def _build_examples(
        self,
        pool: List[Tuple[str, str]],
        l1_options: List[str],
        n: int = 3,
    ) -> str:
        if not pool:
            return "(no examples available)"
        samples = self.rng.sample(pool, min(n, len(pool)))
        lines = []
        for ex_text, ex_label in samples:
            # Truncate example text
            short = ex_text[:200] + ("..." if len(ex_text) > 200 else "")
            lines.append(f'- "{short}" → {ex_label}')
        return "\n".join(lines)

    def select_template(self) -> str:
        """Randomly select a template weighted by configured ratios."""
        names = list(self.template_ratios.keys())
        weights = [self.template_ratios[n] for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]


def compute_sampling_weights(
    dataset_sizes: Dict[str, int],
) -> Dict[str, float]:
    """Compute anti-dominance weights: 1/sqrt(N_i), normalized to sum to 1."""
    raw = {name: 1.0 / math.sqrt(size) for name, size in dataset_sizes.items()}
    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


def _load_20newsgroups() -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("SetFit/20_newsgroups", split="train")
    texts = [item["text"] for item in ds]
    labels = [{"l1": str(item["label_text"]), "l2": ""} for item in ds]
    l1_options = sorted(set(l["l1"] for l in labels))
    return texts, labels, l1_options


def _load_ledgar() -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("lex_glue", "ledgar", split="train")
    texts = [item["text"] for item in ds]
    labels = [{"l1": str(item["label"]), "l2": ""} for item in ds]
    l1_options = sorted(set(l["l1"] for l in labels))
    return texts, labels, l1_options


def _load_ag_news() -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}
    texts = [item["text"] for item in ds]
    labels = [{"l1": label_names[item["label"]], "l2": ""} for item in ds]
    l1_options = list(label_names.values())
    return texts, labels, l1_options


def _load_dbpedia(cfg: TrainingConfig) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("dbpedia_14", split="train")
    # Stratified subsample
    n_total = len(ds)
    n_sample = int(n_total * cfg.dbpedia_subsample)
    ds = ds.shuffle(seed=cfg.seed).select(range(min(n_sample, n_total)))
    label_names = {
        0: "Company", 1: "Educational Institution", 2: "Artist",
        3: "Athlete", 4: "Office Holder", 5: "Mean of Transportation",
        6: "Building", 7: "Natural Place", 8: "Village",
        9: "Animal", 10: "Plant", 11: "Album",
        12: "Film", 13: "Written Work",
    }
    texts = [item["text"] for item in ds]
    labels = [{"l1": label_names[item["label"]], "l2": ""} for item in ds]
    l1_options = list(label_names.values())
    return texts, labels, l1_options


def _load_german_multifin() -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("anhaltai/german-multifin", split="train")
    texts: List[str] = []
    labels: List[Dict[str, str]] = []
    for item in ds:
        text_val = item.get("text", "") or item.get("sentence", "") or ""
        if not text_val:
            continue
        l1 = item.get("l1_label", "") or item.get("label", "")
        l2 = item.get("l2_label", "") or ""
        if not l1:
            continue
        texts.append(str(text_val))
        labels.append({"l1": str(l1), "l2": str(l2) if l2 else ""})

    l1_set = sorted(set(l["l1"] for l in labels if l["l1"]))
    return texts, labels, l1_set


_DATASET_LOADERS = {
    "20newsgroups": _load_20newsgroups,
    "ledgar": _load_ledgar,
    "ag_news": _load_ag_news,
    "dbpedia": _load_dbpedia,
    "german_multifin": _load_german_multifin,
}


def build_phase1_dataset(cfg: TrainingConfig) -> HFDataset:
    """Load all Phase 1 datasets, apply templates with sampling weights,
    return a single HF Dataset ready for Seq2SeqTrainer."""
    engine = TemplateEngine(seed=cfg.seed, template_ratios=cfg.template_ratios)
    all_records: List[Dict[str, str]] = []
    dataset_sizes: Dict[str, int] = {}

    # First pass: load everything and compute sizes
    all_loaded: Dict[str, tuple] = {}
    for name, loader_fn in _DATASET_LOADERS.items():
        texts, labels, l1_options = loader_fn(cfg) if name == "dbpedia" else loader_fn()
        dataset_sizes[name] = len(texts)
        all_loaded[name] = (texts, labels, l1_options)

    # Second pass: apply anti-dominance subsampling
    weights = compute_sampling_weights(dataset_sizes)
    min_size = min(dataset_sizes.values())
    # Cap: max effective samples = min_size × 15 (prevent extreme trimming)
    cap = min_size * 15

    for name, (texts, labels, l1_options) in all_loaded.items():
        target_n = min(int(weights[name] * sum(dataset_sizes.values()) * 0.5), cap)
        target_n = max(target_n, min_size)  # never go below smallest dataset

        if len(texts) > target_n:
            indices = engine.rng.sample(range(len(texts)), target_n)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            print(f"  {name}: subsampled {len(indices)} → {target_n} "
                  f"(weight={weights[name]:.3f})")

        for text, label in zip(texts, labels):
            tpl_name = engine.select_template()
            example_pool = None
            if tpl_name == "few_shot" and len(texts) > 1:
                indices = engine.rng.sample(
                    range(len(texts)),
                    min(3, len(texts)),
                )
                example_pool = [
                    (texts[i][:200], labels[i]["l1"])
                    for i in indices
                    if i < len(texts)
                ]

            record = engine.apply(
                tpl_name, text, l1_options, label,
                example_pool=example_pool,
            )
            record["_dataset"] = name
            all_records.append(record)

    return HFDataset.from_list(all_records)


def build_phase2_dataset(
    texts: List[str],
    labels: List[Dict[str, str]],
    l1_options: List[str],
    cfg: TrainingConfig,
) -> HFDataset:
    """Build a dataset from DSPM texts+labels with template application."""
    engine = TemplateEngine(seed=cfg.seed, template_ratios=cfg.template_ratios)
    records: List[Dict[str, str]] = []

    example_pool = [
        (texts[i][:200], labels[i]["l1"])
        for i in range(min(len(texts), 10))
    ]

    for text, label in zip(texts, labels):
        tpl_name = engine.select_template()
        record = engine.apply(
            tpl_name, text, l1_options, label,
            example_pool=example_pool,
        )
        records.append(record)

    return HFDataset.from_list(records)


def tokenize_dataset(
    dataset: HFDataset,
    tokenizer,
    max_length: int,
    max_target_length: int,
) -> HFDataset:
    """Tokenize input-target pairs for Seq2Seq training."""

    def _tokenize(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                examples["target"],
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )
        inputs["labels"] = targets["input_ids"]
        return inputs

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)
```

- [ ] **Step 4: Run data pipeline tests**

Run: `python -m pytest benchmark/tests/test_data_pipeline.py::test_zero_shot_template -v`
Expected: PASS

Run: `python -m pytest benchmark/tests/test_data_pipeline.py -v`
Expected: 7 PASS (all template format tests + sampling weights test)

- [ ] **Step 5: Commit**

```bash
git add benchmark/train/data_pipeline.py benchmark/tests/test_data_pipeline.py
git commit -m "feat: add template engine, dataset loaders, and sampling weights"
```

---

### Task 4: QLoRA Trainer (Phase 1 + Phase 2)

**Files:**
- Create: `benchmark/train/trainer.py`

- [ ] **Step 1: Write trainer.py**

```python
# benchmark/train/trainer.py
from __future__ import annotations

import os
from typing import Dict, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from benchmark.train.config import TrainingConfig
from benchmark.train.data_pipeline import (
    build_phase1_dataset,
    build_phase2_dataset,
    tokenize_dataset,
)


def _build_quantization_config(quantization: str | None) -> BitsAndBytesConfig | None:
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def _build_lora_config(cfg: TrainingConfig, dropout: float | None = None) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=dropout if dropout is not None else cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def setup_model_and_tokenizer(cfg: TrainingConfig):
    """Load base model with quantization, apply LoRA, return (peft_model, tokenizer)."""
    bnb_config = _build_quantization_config(cfg.quantization)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if cfg.quantization is None else None,
    )

    lora_config = _build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_phase1(cfg: TrainingConfig) -> PeftModel:
    """Phase 1: Train general document classification on open datasets."""
    print("=" * 60)
    print("PHASE 1: General Classification Training")
    print("=" * 60)

    model, tokenizer = setup_model_and_tokenizer(cfg)

    dataset = build_phase1_dataset(cfg)
    dataset = tokenize_dataset(
        dataset, tokenizer,
        max_length=cfg.phase1_max_length,
        max_target_length=cfg.phase1_max_target_length,
    )

    split = dataset.train_test_split(
        test_size=cfg.phase1_val_split, seed=cfg.seed,
    )
    train_ds = split["train"]
    val_ds = split["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "phase1_checkpoints"),
        per_device_train_batch_size=cfg.phase1_batch_size,
        per_device_eval_batch_size=cfg.phase1_batch_size,
        gradient_accumulation_steps=cfg.phase1_grad_accum,
        num_train_epochs=cfg.phase1_epochs,
        learning_rate=cfg.phase1_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.phase1_warmup_ratio,
        weight_decay=cfg.phase1_weight_decay,
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.quantization is None,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save Phase 1 adapter
    adapter_path = os.path.join(cfg.output_dir, "phase1_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Phase 1 adapter saved to {adapter_path}")

    return model


def train_phase2(
    cfg: TrainingConfig,
    dspm_texts: List[str],
    dspm_labels: List[Dict[str, str]],
    l1_options: List[str],
) -> PeftModel:
    """Phase 2: DSPM domain adaptation on augmented datasets."""
    print("=" * 60)
    print("PHASE 2: DSPM Domain Adaptation")
    print("=" * 60)

    # Load base model + Phase 1 adapter
    bnb_config = _build_quantization_config(cfg.quantization)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if cfg.quantization is None else None,
    )

    phase1_path = os.path.join(cfg.output_dir, "phase1_adapter")
    if os.path.exists(phase1_path):
        model = PeftModel.from_pretrained(base_model, phase1_path, is_trainable=True)
        print(f"Loaded Phase 1 adapter from {phase1_path}")
    else:
        lora_config = _build_lora_config(cfg, dropout=cfg.phase2_lora_dropout)
        model = get_peft_model(base_model, lora_config)
        print("No Phase 1 adapter found, training from scratch with higher dropout")

    # Update LoRA dropout for Phase 2
    for module in model.modules():
        if hasattr(module, "dropout") and hasattr(module.dropout, "p"):
            if module.dropout.p != cfg.phase2_lora_dropout:
                module.dropout.p = cfg.phase2_lora_dropout

    dataset = build_phase2_dataset(dspm_texts, dspm_labels, l1_options, cfg)
    dataset = tokenize_dataset(
        dataset, tokenizer,
        max_length=cfg.phase2_max_length,
        max_target_length=cfg.phase2_max_target_length,
    )

    split = dataset.train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds = split["train"]
    val_ds = split["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "phase2_checkpoints"),
        per_device_train_batch_size=cfg.phase2_batch_size,
        per_device_eval_batch_size=cfg.phase2_batch_size,
        gradient_accumulation_steps=cfg.phase2_grad_accum,
        num_train_epochs=cfg.phase2_epochs,
        learning_rate=cfg.phase2_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.phase2_warmup_ratio,
        weight_decay=cfg.phase2_weight_decay,
        optim="adamw_torch",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.quantization is None,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.phase2_early_stopping_patience,
        )],
    )

    trainer.train()

    # Save Phase 2 adapter
    adapter_path = os.path.join(cfg.output_dir, "phase2_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Phase 2 adapter saved to {adapter_path}")

    return model
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from benchmark.train.trainer import setup_model_and_tokenizer, train_phase1, train_phase2; print('OK')"`
Expected: OK (lazy imports, won't load model)

- [ ] **Step 3: Commit**

```bash
git add benchmark/train/trainer.py
git commit -m "feat: add QLoRA trainer with two-phase Seq2SeqTrainer loops"
```

---

### Task 5: Adapter Merge + HF Export

**Files:**
- Create: `benchmark/train/merge_adapter.py`

- [ ] **Step 1: Write merge_adapter.py**

```python
# benchmark/train/merge_adapter.py
"""Merge LoRA adapter weights into base model and export as standard HF model."""
from __future__ import annotations

import os
import shutil

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from benchmark.train.config import TrainingConfig


def merge_and_save(cfg: TrainingConfig) -> str:
    """Merge the Phase 2 LoRA adapter into the base model and save as HF format.
    Returns the output path."""
    output_path = os.path.join(cfg.output_dir, "merged")
    adapter_path = os.path.join(cfg.output_dir, "phase2_adapter")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. Run Phase 2 training first."
        )

    print(f"Loading base model: {cfg.model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    # Clean up checkpoint files (keep only merged)
    for subdir in ["phase1_checkpoints", "phase2_checkpoints"]:
        ckpt_path = os.path.join(cfg.output_dir, subdir)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
            print(f"Cleaned up: {ckpt_path}")

    print(f"Merged model ready at: {output_path}")
    return output_path
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/train/merge_adapter.py
git commit -m "feat: add LoRA adapter merge and HF model export"
```

---

### Task 6: YAML Config + CLI Entry Point

**Files:**
- Create: `benchmark/config/experiments/flan-t5-finetune.yaml`
- Create: `benchmark/scripts/run_finetune.py`

- [ ] **Step 1: Write YAML config**

```yaml
# benchmark/config/experiments/flan-t5-finetune.yaml
# FLAN-T5-Large QLoRA Fine-Tuning Configuration
# Usage: python -m benchmark.scripts.run_finetune --config benchmark/config/experiments/flan-t5-finetune.yaml

model_name: "google/flan-t5-large"
quantization: "8bit"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

phase1_epochs: 3
phase1_lr: 0.0002
phase1_batch_size: 8
phase1_grad_accum: 2
phase1_max_length: 1024
phase1_max_target_length: 128
phase1_warmup_ratio: 0.1
phase1_weight_decay: 0.01

phase2_epochs: 12
phase2_lr: 0.00005
phase2_batch_size: 4
phase2_grad_accum: 2
phase2_lora_dropout: 0.10
phase2_max_length: 1024
phase2_max_target_length: 128
phase2_warmup_ratio: 0.1
phase2_weight_decay: 0.01
phase2_early_stopping_patience: 3

dbpedia_subsample: 0.10
phase1_val_split: 0.05

augment_back_translation_count: 2
augment_entity_sub_count: 3
augment_llm_synthesis_count: 5
augment_quality_min_similarity: 0.15
augment_quality_max_similarity: 0.95

output_dir: "benchmark/models/flan-t5-finetuned"
seed: 42

template_ratios:
  zero_shot: 0.40
  few_shot: 0.20
  cot: 0.15
  label_to_content: 0.15
  contrastive: 0.10
```

- [ ] **Step 2: Write CLI entry point**

```python
# benchmark/scripts/run_finetune.py
"""CLI entry point for FLAN-T5 QLoRA fine-tuning.

Usage:
    python -m benchmark.scripts.run_finetune
    python -m benchmark.scripts.run_finetune --config path/to/config.yaml
    python -m benchmark.scripts.run_finetune --phase1-only
    python -m benchmark.scripts.run_finetune --phase2-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure benchmark package is importable
_BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from benchmark.train.config import TrainingConfig
from benchmark.train.trainer import train_phase1, train_phase2
from benchmark.train.merge_adapter import merge_and_save
from benchmark.train.augment import Augmenter


def _load_dspm_data():
    """Load DSPM datasets from benchmark and prepare for Phase 2."""
    from benchmark.src.cyera_bench.datasets.dspm27 import Dspm27Dataset
    from benchmark.src.cyera_bench.datasets.ben25 import Ben25Dataset
    from benchmark.src.cyera_bench.datasets.cxh5types import Cxh5typesDataset

    datasets = {
        "dspm27": Dspm27Dataset(),
        "ben25": Ben25Dataset(),
        "cxh5types": Cxh5typesDataset(),
    }

    all_texts = []
    all_labels = []
    all_l1_set = set()

    for name, ds in datasets.items():
        texts, labels = ds.load()
        all_texts.extend(texts)
        all_labels.extend(labels)
        for l in labels:
            all_l1_set.add(l["l1"])

    return all_texts, all_labels, sorted(all_l1_set)


def main():
    parser = argparse.ArgumentParser(
        description="FLAN-T5 QLoRA Fine-Tuning for Document Classification"
    )
    parser.add_argument(
        "--config", type=str,
        default="benchmark/config/experiments/flan-t5-finetune.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--phase1-only", action="store_true",
        help="Run only Phase 1 (general classification)",
    )
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="Run only Phase 2 (DSPM adaptation). Requires Phase 1 adapter.",
    )
    parser.add_argument(
        "--skip-augmentation", action="store_true",
        help="Skip DSPM data augmentation (use raw data only)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip final adapter merge step",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = TrainingConfig.from_yaml(str(config_path))
    print(f"Loaded config from {config_path}")
    print(f"  Model: {cfg.model_name}")
    print(f"  Quantization: {cfg.quantization}")
    print(f"  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    print(f"  Output: {cfg.output_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    if not args.phase2_only:
        model = train_phase1(cfg)
        print("Phase 1 complete.")

    if not args.phase1_only:
        print("\nLoading DSPM data...")
        dspm_texts, dspm_labels, l1_options = _load_dspm_data()
        print(f"  Loaded {len(dspm_texts)} raw DSPM documents")

        if not args.skip_augmentation:
            print("Augmenting DSPM data...")
            augmenter = Augmenter(cfg)
            dspm_texts, dspm_labels = augmenter.augment(
                dspm_texts, dspm_labels,
            )
            print(f"  After augmentation: {len(dspm_texts)} documents")

        model = train_phase2(cfg, dspm_texts, dspm_labels, l1_options)
        print("Phase 2 complete.")

    if not args.skip_merge:
        output_path = merge_and_save(cfg)
        print(f"\nFine-tuned model ready: {output_path}")
        print(
            "To use in benchmark, set finetuned_path in your experiment config "
            "or pass to FlanT5ClassificationModel(finetuned_path=...)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify CLI is importable**

Run: `python -c "import sys; sys.path.insert(0, 'benchmark'); from benchmark.scripts.run_finetune import main; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add benchmark/config/experiments/flan-t5-finetune.yaml benchmark/scripts/run_finetune.py
git commit -m "feat: add training YAML config and CLI entry point for fine-tuning"
```

---

### Task 7: Benchmark Integration — Load Fine-Tuned Checkpoint

**Files:**
- Modify: `benchmark/src/cyera_bench/models/flan_t5_classification.py:39-53`

- [ ] **Step 1: Add finetuned_path parameter**

In `benchmark/src/cyera_bench/models/flan_t5_classification.py`, modify the `__init__` and `_load_pipeline` methods:

```python
# In __init__, add parameter after max_input_chars:
def __init__(
    self,
    variant: str = "large",
    device: str = "cuda",
    quantization: str | None = None,
    prompt_style: str = "two_step",
    max_input_chars: int = 8000,
    finetuned_path: str | None = None,  # NEW
):
    super().__init__(
        variant=variant,
        device=device,
        quantization=quantization,
    )
    self._prompt_style = prompt_style
    self._max_input_chars = max_input_chars
    self._finetuned_path = finetuned_path  # NEW

# In _load_pipeline, change model_name resolution:
def _load_pipeline(self):
    if self._pipe is not None:
        return
    from cyera_bench.models.flan_t5 import _MODEL_MAP
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    info = _MODEL_MAP[self._variant]
    # Use fine-tuned checkpoint if provided, otherwise use base HF model
    model_name = self._finetuned_path or info["hf_name"]  # CHANGED
    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ... rest unchanged
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `python -m pytest benchmark/tests/test_flan_t5.py -v`
Expected: Existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add benchmark/src/cyera_bench/models/flan_t5_classification.py
git commit -m "feat: add finetuned_path parameter to FlanT5ClassificationModel"
```

---

### Task 8: End-to-End Smoke Test

**Files:** None (manual verification)

- [ ] **Step 1: Verify full import chain without model download**

Run:
```bash
python -c "
from benchmark.train.config import TrainingConfig
from benchmark.train.data_pipeline import TemplateEngine, compute_sampling_weights, TEMPLATES
from benchmark.train.augment import entity_substitution, tfidf_quality_filter
from benchmark.train.merge_adapter import merge_and_save
from benchmark.scripts.run_finetune import main
print('All imports OK')
"
```
Expected: All imports OK

- [ ] **Step 2: Verify config loading**

Run:
```bash
python -c "
from benchmark.train.config import TrainingConfig
cfg = TrainingConfig.from_yaml('benchmark/config/experiments/flan-t5-finetune.yaml')
print(f'Model: {cfg.model_name}')
print(f'Phase 1 epochs: {cfg.phase1_epochs}, LR: {cfg.phase1_lr}')
print(f'Phase 2 epochs: {cfg.phase2_epochs}, LR: {cfg.phase2_lr}')
print(f'Template ratios: {cfg.template_ratios}')
print(f'Output: {cfg.output_dir}')
"
```
Expected: Correct config values printed

- [ ] **Step 3: Verify template engine with all 5 templates**

Run:
```bash
python -c "
from benchmark.train.data_pipeline import TemplateEngine, TEMPLATES

engine = TemplateEngine(seed=42)
text = 'The Q4 security audit identified 3 critical vulnerabilities in the firewall configuration.'
l1_options = ['IT Security', 'Financial Reports', 'HR Documents', 'Legal Contracts']
label = {'l1': 'IT Security', 'l2': 'Vulnerability Assessment'}

for tpl_name in TEMPLATES:
    result = engine.apply(tpl_name, text, l1_options, label)
    print(f'--- {tpl_name} ---')
    print(f'INPUT ({len(result[\"input\"])} chars): {result[\"input\"][:150]}...')
    print(f'TARGET: {result[\"target\"][:100]}')
    print()
"
```
Expected: All 5 templates produce valid input/target pairs

- [ ] **Step 4: Run all new tests**

Run: `python -m pytest benchmark/tests/test_train_config.py benchmark/tests/test_augment.py benchmark/tests/test_data_pipeline.py -v`
Expected: All tests PASS (1 SKIP for back_translation if Helsinki models not downloaded)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify all modules import correctly and tests pass"
```
