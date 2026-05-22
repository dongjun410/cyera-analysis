"""
Learned Classification — LLM Auto-Labeling → Lightweight Classifier Distillation

Architecture (mirrors Cyera Layer 5, with a distinct implementation):

  ┌─────────────────────────────────────────────────────────┐
  │                Knowledge Distillation Loop               │
  │                                                         │
  │  Stage 1: LLM Auto-Labeling (Teacher)                  │
  │    Cluster results + representative docs                │
  │    → LLM assigns structured labels with confidence      │
  │    → Produces labeled training dataset                  │
  │                                                         │
  │  Stage 2: Classifier Training (Student)                 │
  │    Labeled dataset → SetFit few-shot fine-tuning        │
  │    → Produces lightweight classifier (~100MB)           │
  │    → 1000x faster than LLM, runs on CPU                │
  │                                                         │
  │  Stage 3: Inference with Fallback                       │
  │    New doc → Classifier (fast, ~2ms/doc)                │
  │    → If confidence >= threshold: accept                 │
  │    → If confidence < threshold: LLM fallback            │
  │    → LLM result added to training buffer                │
  │                                                         │
  │  Stage 4: Incremental Re-training                       │
  │    Buffer reaches N samples → retrain classifier        │
  │    → Classifier continuously improves                   │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

Why SetFit:
  - Needs only 8-16 samples per class (vs hundreds for standard fine-tuning)
  - Based on sentence-transformers (same as our embedding model)
  - Trains in minutes on CPU
  - Inference ~2ms/doc (vs ~500ms for LLM API call)
  - Apache 2.0 license, fully local

Data flow:
  Clustering → LLM labels clusters → Labels become training data →
  Train SetFit → SetFit classifies new docs → Low-confidence → LLM →
  New labels → Buffer → Retrain SetFit → Loop
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from models.schemas import ProcessedDocument, ClusterInfo

logger = logging.getLogger(__name__)


@dataclass
class LabeledSample:
    """A single labeled training sample."""
    doc_id: str
    text: str                          # raw content (truncated for training)
    label: str                         # classification label
    confidence: float                  # LLM confidence or classifier confidence
    source: str                        # "llm" or "human" or "classifier_verified"


class LearnedClassifier:
    """
    Knowledge distillation pipeline:
    LLM (teacher) → labeled data → SetFit (student) → fast inference
    """

    def __init__(self, config: dict, llm_config: dict = None):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.model_dir = config.get("model_dir", "./classifiers/model")
        self.label_map_path = config.get("label_map_path", "./classifiers/label_map.json")

        # Sub-configs
        self.labeling_config = config.get("labeling", {})
        self.training_config = config.get("training", {})
        self.inference_config = config.get("inference", {})
        self.incremental_config = config.get("incremental", {})

        self.llm_config = llm_config or {}
        self.llm_client = None
        self.classifier = None
        self.label_map: Dict[str, int] = {}

        # Training data buffer (for incremental learning)
        self.training_buffer: List[LabeledSample] = []

    # ══════════════════════════════════════════════════════════
    # Stage 1: LLM Auto-Labeling
    # ══════════════════════════════════════════════════════════

    def generate_training_data(
        self,
        clusters: List[ClusterInfo],
        documents: List[ProcessedDocument],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> List[LabeledSample]:
        """
        Use LLM to auto-label representative documents from each cluster.
        This is the 'teacher' step — LLM produces high-quality labels that
        become training data for the lightweight classifier.
        """
        if not self.llm_client:
            self._init_llm_client()

        samples_per_cluster = self.labeling_config.get("samples_per_cluster", 10)
        min_cluster_size = self.labeling_config.get("min_cluster_size", 3)
        confidence_threshold = self.labeling_config.get("confidence_threshold", 0.7)

        all_samples: List[LabeledSample] = []
        doc_map = {d.id: d for d in documents}

        for cluster in clusters:
            if cluster.size < min_cluster_size:
                continue

            # Use cluster label as the target class
            cluster_label = cluster.llm_label or "_".join(cluster.keywords[:3])
            if not cluster_label:
                continue

            # Get representative + random docs from this cluster
            cluster_doc_ids = cluster.document_ids
            sample_ids = self._select_labeling_samples(
                cluster_doc_ids, cluster, embeddings, labels, samples_per_cluster
            )

            # LLM validates each sample's membership in this cluster
            for doc_id in sample_ids:
                doc = doc_map.get(doc_id)
                if not doc:
                    continue

                # Ask LLM: "Does this document belong to category X?"
                # This is validation, not open-ended classification
                confidence = self._llm_validate_label(
                    doc.raw_content[:2000],
                    cluster_label,
                    cluster.keywords[:10],
                )

                if confidence >= confidence_threshold:
                    all_samples.append(LabeledSample(
                        doc_id=doc_id,
                        text=doc.raw_content[:4000],
                        label=cluster_label,
                        confidence=confidence,
                        source="llm",
                    ))

        logger.info(
            f"LLM auto-labeling complete: {len(all_samples)} samples "
            f"across {len(set(s.label for s in all_samples))} classes"
        )
        return all_samples

    def _select_labeling_samples(
        self,
        doc_ids: List[str],
        cluster: ClusterInfo,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n: int,
    ) -> List[str]:
        """Select documents for LLM labeling: mix of representative + random."""
        # Start with representative docs (already selected by MMR)
        selected = list(cluster.representative_doc_ids[:n // 2])

        # Add random samples from the rest
        remaining = [d for d in doc_ids if d not in selected]
        import random
        random.seed(42)
        extra = random.sample(remaining, min(n - len(selected), len(remaining)))
        selected.extend(extra)

        return selected[:n]

    def _llm_validate_label(
        self,
        text_excerpt: str,
        proposed_label: str,
        keywords: List[str],
    ) -> float:
        """
        Ask LLM to validate whether a document matches a proposed label.
        Returns confidence score 0.0-1.0.

        This is more reliable than open-ended classification because the LLM
        only needs to answer "yes/no + confidence" rather than invent a label.
        """
        prompt = f"""You are a document classification validator.

Given a document excerpt and a proposed category, assess whether the document
belongs to this category.

Proposed category: "{proposed_label}"
Category keywords: {', '.join(keywords)}

Document excerpt:
---
{text_excerpt}
---

Respond with ONLY a JSON object:
{{"match": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence"}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "qwen2.5-7b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)

            if result.get("match", False):
                return float(result.get("confidence", 0.0))
            return 0.0

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return 0.0

    # ══════════════════════════════════════════════════════════
    # Stage 2: Classifier Training (SetFit)
    # ══════════════════════════════════════════════════════════

    def train_classifier(self, samples: List[LabeledSample]):
        """
        Train a SetFit classifier using LLM-generated labels.
        SetFit needs only 8-16 samples per class for good performance.
        """
        from setfit import SetFitModel, SetFitTrainer
        from datasets import Dataset

        min_per_class = self.training_config.get("min_samples_per_class", 5)
        max_per_class = self.training_config.get("max_samples_per_class", 200)
        eval_split = self.training_config.get("eval_split", 0.2)
        base_model = self.training_config.get("base_model", "BAAI/bge-m3")
        num_epochs = self.training_config.get("num_epochs", 3)
        batch_size = self.training_config.get("batch_size", 16)

        # Build label map
        label_counts: Dict[str, int] = {}
        for s in samples:
            label_counts[s.label] = label_counts.get(s.label, 0) + 1

        # Filter: only classes with enough samples
        valid_labels = {
            label for label, count in label_counts.items()
            if count >= min_per_class
        }
        filtered = [s for s in samples if s.label in valid_labels]

        if len(valid_labels) < 2:
            logger.warning(
                f"Only {len(valid_labels)} valid classes (need >=2). "
                f"Skipping classifier training."
            )
            return

        # Balance: cap samples per class
        balanced: List[LabeledSample] = []
        per_class: Dict[str, List] = {}
        for s in filtered:
            per_class.setdefault(s.label, []).append(s)
        for label, group in per_class.items():
            balanced.extend(group[:max_per_class])

        # Build label map (string → int)
        unique_labels = sorted(set(s.label for s in balanced))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        inv_label_map = {idx: label for label, idx in self.label_map.items()}

        # Create HuggingFace Dataset
        texts = [s.text for s in balanced]
        int_labels = [self.label_map[s.label] for s in balanced]

        dataset = Dataset.from_dict({"text": texts, "label": int_labels})

        # Train/eval split
        split = dataset.train_test_split(test_size=eval_split, seed=42)

        logger.info(
            f"Training SetFit classifier: {len(unique_labels)} classes, "
            f"{len(split['train'])} train / {len(split['test'])} eval samples"
        )

        # Initialize and train SetFit
        model = SetFitModel.from_pretrained(base_model)

        trainer = SetFitTrainer(
            model=model,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            column_mapping={"text": "text", "label": "label"},
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        trainer.train()

        # Evaluate
        metrics = trainer.evaluate()
        logger.info(f"Classifier evaluation: {metrics}")

        # Save
        os.makedirs(self.model_dir, exist_ok=True)
        model.save_pretrained(self.model_dir)

        # Save label map
        os.makedirs(os.path.dirname(self.label_map_path), exist_ok=True)
        with open(self.label_map_path, 'w') as f:
            json.dump({
                "label_to_id": self.label_map,
                "id_to_label": inv_label_map,
                "metrics": metrics,
                "num_samples": len(balanced),
                "num_classes": len(unique_labels),
            }, f, indent=2, ensure_ascii=False)

        self.classifier = model
        logger.info(f"Classifier saved to {self.model_dir}")

    # ══════════════════════════════════════════════════════════
    # Stage 3: Inference with LLM Fallback
    # ══════════════════════════════════════════════════════════

    def classify_document(
        self, text: str
    ) -> Tuple[str, float, str]:
        """
        Classify a single document using the distilled classifier.
        Falls back to LLM for low-confidence predictions.

        Returns: (label, confidence, source)
          source: "classifier" | "llm_fallback" | "unknown"
        """
        threshold = self.inference_config.get("classifier_confidence_threshold", 0.8)
        use_llm_fallback = self.inference_config.get("llm_fallback", True)

        # Try classifier first
        if self.classifier is not None:
            label, confidence = self._classifier_predict(text)

            if confidence >= threshold:
                return label, confidence, "classifier"

            logger.debug(
                f"Classifier confidence {confidence:.2f} < {threshold}, "
                f"falling back to LLM"
            )

        # LLM fallback
        if use_llm_fallback:
            label, confidence = self._llm_classify(text)
            if label and confidence > 0:
                # Add to training buffer for incremental learning
                self.training_buffer.append(LabeledSample(
                    doc_id="",
                    text=text[:4000],
                    label=label,
                    confidence=confidence,
                    source="llm",
                ))
                self._check_retrain_trigger()
                return label, confidence, "llm_fallback"

        return "unknown", 0.0, "unknown"

    def classify_batch(
        self, texts: List[str]
    ) -> List[Tuple[str, float, str]]:
        """Batch classification with classifier + LLM fallback."""
        results = []
        for text in texts:
            results.append(self.classify_document(text))
        return results

    def _classifier_predict(self, text: str) -> Tuple[str, float]:
        """Run SetFit classifier and return (label_string, confidence)."""
        try:
            # SetFit predict returns class indices
            prediction = self.classifier.predict([text[:4000]])
            probabilities = self.classifier.predict_proba([text[:4000]])

            pred_idx = int(prediction[0])
            confidence = float(probabilities[0].max())

            # Map back to string label
            inv_map = {v: k for k, v in self.label_map.items()}
            label = inv_map.get(pred_idx, "unknown")

            return label, confidence
        except Exception as e:
            logger.warning(f"Classifier prediction failed: {e}")
            return "unknown", 0.0

    def _llm_classify(self, text: str) -> Tuple[str, float]:
        """Open-ended LLM classification for documents the classifier can't handle."""
        if not self.llm_client:
            self._init_llm_client()
        if not self.llm_client:
            return "", 0.0

        # Include known labels as guidance
        known_labels = list(self.label_map.keys()) if self.label_map else []

        prompt = f"""You are a document classification expert.
Classify the following document into a category.

{"Known categories: " + ", ".join(known_labels) if known_labels else "Identify the most appropriate category."}
If the document doesn't fit any known category, propose a new one.

Document excerpt:
---
{text[:2000]}
---

Respond with ONLY a JSON object:
{{"label": "category name (max 5 words)", "confidence": 0.0-1.0, "is_new_category": true/false}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "qwen2.5-7b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.2,
            )
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)
            return result.get("label", ""), float(result.get("confidence", 0.0))
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return "", 0.0

    # ══════════════════════════════════════════════════════════
    # Stage 4: Incremental Re-training
    # ══════════════════════════════════════════════════════════

    def _check_retrain_trigger(self):
        """Check if training buffer has enough samples to trigger retraining."""
        trigger = self.incremental_config.get("retrain_trigger", 500)
        if len(self.training_buffer) >= trigger:
            logger.info(
                f"Training buffer reached {len(self.training_buffer)} samples, "
                f"triggering incremental retraining"
            )
            self.incremental_retrain()

    def incremental_retrain(self):
        """
        Retrain the classifier by merging new LLM-labeled samples
        with the existing training set.
        """
        strategy = self.incremental_config.get("retrain_strategy", "full")

        if not self.training_buffer:
            logger.info("No new samples in buffer, skipping retrain")
            return

        # Load existing training data if available
        existing_samples = self._load_training_data()

        if strategy == "full":
            # Combine existing + new, retrain from scratch
            all_samples = existing_samples + self.training_buffer
            logger.info(
                f"Full retrain: {len(existing_samples)} existing + "
                f"{len(self.training_buffer)} new = {len(all_samples)} total"
            )
            self.train_classifier(all_samples)
        else:
            # Incremental: only train on new data (less stable)
            logger.info(f"Incremental retrain: {len(self.training_buffer)} new samples")
            self.train_classifier(self.training_buffer)

        # Save new samples and clear buffer
        self._save_training_data(existing_samples + self.training_buffer)
        self.training_buffer = []

    def _save_training_data(self, samples: List[LabeledSample]):
        """Persist training data for future retraining."""
        path = os.path.join(os.path.dirname(self.model_dir), "training_data.json")
        data = [
            {"doc_id": s.doc_id, "text": s.text, "label": s.label,
             "confidence": s.confidence, "source": s.source}
            for s in samples
        ]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} training samples to {path}")

    def _load_training_data(self) -> List[LabeledSample]:
        """Load previously saved training data."""
        path = os.path.join(os.path.dirname(self.model_dir), "training_data.json")
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [
            LabeledSample(
                doc_id=d["doc_id"], text=d["text"], label=d["label"],
                confidence=d["confidence"], source=d["source"],
            )
            for d in data
        ]

    # ══════════════════════════════════════════════════════════
    # Model Loading & LLM Client
    # ══════════════════════════════════════════════════════════

    def load_classifier(self) -> bool:
        """Load a previously trained classifier from disk."""
        if not os.path.exists(self.model_dir):
            logger.info("No trained classifier found")
            return False
        try:
            from setfit import SetFitModel
            self.classifier = SetFitModel.from_pretrained(self.model_dir)

            if os.path.exists(self.label_map_path):
                with open(self.label_map_path) as f:
                    data = json.load(f)
                self.label_map = data.get("label_to_id", {})

            logger.info(
                f"Loaded classifier: {len(self.label_map)} classes "
                f"from {self.model_dir}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")
            return False

    def _init_llm_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key="not-needed",
                base_url=self.llm_config.get("api_base", "http://localhost:8000/v1"),
            )
        except Exception as e:
            logger.warning(f"LLM client init failed: {e}")
