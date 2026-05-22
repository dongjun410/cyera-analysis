"""DeBERTa-v3 NER service — encoder-only token classification for PII detection.

Consumes Tier 0's PIIFeatureVector for low-confidence entities and performs
deep contextual disambiguation. Uses HuggingFace pipeline for token classification.

The base DeBERTa model isn't fine-tuned for PII, so entity predictions will
be generic (PER/ORG/LOC/MISC). This module focuses on structural disambiguation
and provides a foundation that can be swapped for a fine-tuned PII model.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from src.types import PIIFeature, PIIFeatureVector

# Standard CoNLL-2003 NER label set for a generic token classification head.
# When the model is fine-tuned for PII, these labels would be replaced with
# PII-specific types (SSN, EMAIL, CREDIT_CARD, etc.).
_DEFAULT_NER_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]


class DebertaNER:
    """DeBERTa-v3 encoder for token classification (PII detection).

    Uses a HuggingFace token-classification pipeline with DeBERTa-v3-base
    as the backbone. The classification head is randomly initialized if the
    base model hasn't been fine-tuned for NER — swap in a fine-tuned PII model
    for production accuracy.

    Args:
        model_name: HuggingFace model identifier (default: microsoft/deberta-v3-base).
        device: Torch device string (\"cuda\", \"cpu\", or device index).

    Interface:
        predict(text, pii_hints=None) -> list[PIIFeature]
        predict_batch(texts, pii_hints=None) -> list[list[PIIFeature]]
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name

        # ── Resolve device ──────────────────────────────────────
        if device == "cuda":
            if torch.cuda.is_available():
                self._torch_device = 0
            else:
                self._torch_device = -1  # CPU fallback
        elif device == "cpu":
            self._torch_device = -1
        else:
            self._torch_device = device

        # ── Configure token classification head ─────────────────
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = len(_DEFAULT_NER_LABELS)
        config.id2label = {i: label for i, label in enumerate(_DEFAULT_NER_LABELS)}
        config.label2id = {label: i for i, label in enumerate(_DEFAULT_NER_LABELS)}

        # ── Load model with randomly-initialized classification head ──
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ── Build token-classification pipeline ─────────────────
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=self._torch_device,
            aggregation_strategy="simple",
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        pii_hints: PIIFeatureVector | None = None,
    ) -> list[PIIFeature]:
        """Run NER on a single text, returning detected PII entities.

        Args:
            text: Input text to scan for entities.
            pii_hints: Optional Tier 0 PII features (accepted but not used
                       to restrict NER scope in this basic implementation).

        Returns:
            List of PIIFeature objects detected by the NER model.
        """
        # ── Handle empty text ──────────────────────────────────
        if not text or not text.strip():
            return []

        # ── Run pipeline ───────────────────────────────────────
        raw_results = self._pipeline(text)

        # ── Map pipeline output to PIIFeature objects ──────────
        features: list[PIIFeature] = []
        for entity in raw_results:
            # Extract entity type (handle both aggregated and raw output formats)
            entity_type = entity.get("entity_group") or entity.get("entity", "UNKNOWN")

            # If the model returns a numeric label index, resolve to string
            if isinstance(entity_type, int):
                id2label = self._pipeline.model.config.id2label  # type: ignore[union-attr]
                entity_type = id2label.get(entity_type, f"LABEL_{entity_type}")

            # Filter out "O" (outside) labels that may slip through
            if entity_type == "O":
                continue

            feature = PIIFeature(
                entity_type=str(entity_type),
                span=(int(entity["start"]), int(entity["end"])),
                confidence=float(entity["score"]),
                context_flag="clean",  # NER base model doesn't do context analysis
            )
            features.append(feature)

        return features

    def predict_batch(
        self,
        texts: list[str],
        pii_hints: list[PIIFeatureVector | None] | None = None,
    ) -> list[list[PIIFeature]]:
        """Run NER on a batch of texts.

        Args:
            texts: List of input texts.
            pii_hints: Optional per-text Tier 0 features (accepted but not
                       used to restrict NER scope in this basic implementation).

        Returns:
            List of lists, where each inner list contains PIIFeature objects
            for the corresponding input text.
        """
        return [self.predict(text) for text in texts]

    # ──────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The HuggingFace model identifier used for this NER instance."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value
