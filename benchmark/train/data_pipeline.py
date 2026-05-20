# benchmark/train/data_pipeline.py
from __future__ import annotations

import math
import random
from typing import List, Tuple

from datasets import Dataset as HFDataset
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

    def __init__(self, seed: int = 42, template_ratios: dict[str, float] | None = None):
        self.rng = random.Random(seed)
        if template_ratios:
            total = sum(template_ratios.values())
            self.template_ratios = {k: v / total for k, v in template_ratios.items()}
        else:
            self.template_ratios = {k: 1.0 / len(TEMPLATES) for k in TEMPLATES}

    def _format_l1_options(self, l1_options: List[str]) -> str:
        return "\n".join(f"- {l}" for l in l1_options)

    def select_template(self) -> str:
        """Randomly select a template weighted by configured ratios."""
        names = list(self.template_ratios.keys())
        weights = [self.template_ratios[n] for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def apply(
        self,
        template_name: str,
        text: str,
        l1_options: List[str],
        label: dict[str, str],
        example_pool: List[Tuple[str, str]] | None = None,
    ) -> dict[str, str]:
        """Apply a template to a single sample. Returns {"input": ..., "target": ...}."""
        l1_str = self._format_l1_options(l1_options)
        l1_label = label.get("l1", l1_options[0] if l1_options else "")

        if template_name == "zero_shot":
            prompt = TEMPLATE_ZERO_SHOT.format(l1_options=l1_str, text=text)
            return {"input": prompt, "target": l1_label}

        elif template_name == "few_shot":
            examples_str = self._build_examples(example_pool or [])
            prompt = TEMPLATE_FEW_SHOT.format(
                examples=examples_str, text=text, l1_options=l1_str,
            )
            return {"input": prompt, "target": l1_label}

        elif template_name == "cot":
            prompt = TEMPLATE_COT.format(text=text, l1_options=l1_str)
            target = (
                f"Step 1: The document contains relevant content indicators.\n"
                f"Step 2: This matches the {l1_label} category.\n"
                f"Step 3: Category: {l1_label}"
            )
            return {"input": prompt, "target": target}

        elif template_name == "label_to_content":
            prompt = TEMPLATE_LABEL_TO_CONTENT.format(label=l1_label, l1_options=l1_str)
            return {"input": prompt, "target": text[:200]}

        elif template_name == "contrastive":
            wrong = l1_label
            while wrong == l1_label and len(l1_options) > 1:
                wrong = self.rng.choice(l1_options)
            if wrong == l1_label:
                wrong = "another category"
            prompt = TEMPLATE_CONTRASTIVE.format(
                text=text, l1_options=l1_str,
                wrong_label=wrong, correct_label=l1_label,
            )
            return {"input": prompt, "target": l1_label}

        raise ValueError(f"Unknown template: {template_name}")

    def _build_examples(self, pool: List[Tuple[str, str]], n: int = 3) -> str:
        if not pool:
            return "(no examples available)"
        samples = self.rng.sample(pool, min(n, len(pool)))
        lines = []
        for ex_text, ex_label in samples:
            short = ex_text[:200] + ("..." if len(ex_text) > 200 else "")
            lines.append(f'- "{short}" → {ex_label}')
        return "\n".join(lines)


def compute_sampling_weights(dataset_sizes: dict[str, int]) -> dict[str, float]:
    """Compute anti-dominance weights: 1/sqrt(N_i), normalized to sum to 1."""
    raw = {name: 1.0 / math.sqrt(size) for name, size in dataset_sizes.items()}
    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


# --- Dataset Loaders ---

def _load_20newsgroups() -> Tuple[List[str], List[dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("SetFit/20_newsgroups", split="train")
    texts = [item["text"] for item in ds]
    labels = [{"l1": str(item["label_text"]), "l2": ""} for item in ds]
    l1_options = sorted(set(l["l1"] for l in labels))
    return texts, labels, l1_options


def _load_ledgar() -> Tuple[List[str], List[dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("lex_glue", "ledgar", split="train")
    texts = [item["text"] for item in ds]
    labels = [{"l1": str(item["label"]), "l2": ""} for item in ds]
    l1_options = sorted(set(l["l1"] for l in labels))
    return texts, labels, l1_options


def _load_ag_news() -> Tuple[List[str], List[dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}
    texts = [item["text"] for item in ds]
    labels = [{"l1": label_names[item["label"]], "l2": ""} for item in ds]
    l1_options = list(label_names.values())
    return texts, labels, l1_options


def _load_dbpedia(cfg: TrainingConfig) -> Tuple[List[str], List[dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("dbpedia_14", split="train")
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
    texts = [item["content"] for item in ds]
    labels = [{"l1": label_names[item["label"]], "l2": ""} for item in ds]
    l1_options = list(label_names.values())
    return texts, labels, l1_options


def _load_german_multifin() -> Tuple[List[str], List[dict[str, str]], List[str]]:
    from datasets import load_dataset
    ds = load_dataset("anhaltai/german-multifin", split="train")
    texts: List[str] = []
    labels: List[dict[str, str]] = []
    for item in ds:
        text_val = item.get("ger_text", "") or ""
        if not text_val:
            continue
        l1 = item.get("highlev_label", "") or ""
        l2_raw = item.get("lowlev_labels", "")
        # lowlev_labels may be a string repr of a list, e.g. "['Accounting']"
        l2 = ""
        if l2_raw:
            try:
                parsed = eval(l2_raw)
                l2 = parsed[0] if isinstance(parsed, list) and parsed else str(l2_raw)
            except Exception:
                l2 = str(l2_raw)
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
    all_records: List[dict[str, str]] = []
    dataset_sizes: dict[str, int] = {}

    # First pass: load everything and compute sizes
    all_loaded: dict[str, tuple] = {}
    for name, loader_fn in _DATASET_LOADERS.items():
        texts, labels, l1_options = loader_fn(cfg) if name == "dbpedia" else loader_fn()
        dataset_sizes[name] = len(texts)
        all_loaded[name] = (texts, labels, l1_options)

    # Second pass: apply anti-dominance subsampling
    weights = compute_sampling_weights(dataset_sizes)
    min_size = min(dataset_sizes.values())
    cap = min_size * 15  # prevent extreme trimming

    for name, (texts, labels, l1_options) in all_loaded.items():
        target_n = min(int(weights[name] * sum(dataset_sizes.values()) * 0.5), cap)
        target_n = max(target_n, min_size)

        if len(texts) > target_n:
            indices = engine.rng.sample(range(len(texts)), target_n)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            print(f"  {name}: subsampled to {target_n} (weight={weights[name]:.3f})")

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
    labels: List[dict[str, str]],
    l1_options: List[str],
    cfg: TrainingConfig,
) -> HFDataset:
    """Build a dataset from DSPM texts+labels with template application."""
    engine = TemplateEngine(seed=cfg.seed, template_ratios=cfg.template_ratios)
    records: List[dict[str, str]] = []

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
