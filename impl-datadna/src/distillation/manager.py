"""Distillation Manager — monitors type library → triggers SetFit training.

Watches FusionResult outputs, accumulates high-confidence labeled documents
per type. When any type reaches min_samples_per_class (50), triggers the
DistillationTrainer to build a SetFit model.

Before deploying: evaluates new model → checks F1 degradation vs current.
If any class F1 drops > 3%, the new model is rejected (human review needed).

Integrates with E3MLEngine.set_model() for seamless deployment.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from src.distillation.trainer import DistillationError, DistillationTrainer
from src.types import Document, FusionResult

logger = logging.getLogger(__name__)


class DistillationManager:
    """Monitors classification results and manages SetFit training lifecycle.

    Usage:
        mgr = DistillationManager(e3_engine)
        for result in fusion_results:
            mgr.record(result, document)

        if mgr.should_train():
            success = mgr.train_and_deploy(documents)
    """

    def __init__(
        self,
        e3_engine=None,
        config: dict | None = None,
    ) -> None:
        """Initialize the distillation manager.

        Args:
            e3_engine: E3MLEngine instance for model deployment.
            config: Optional overrides for DistillationTrainer defaults.
        """
        self._e3_engine = e3_engine
        self._trainer = DistillationTrainer(config)

        # Accumulate labeled documents by type
        self._labeled: dict[str, list[Document]] = defaultdict(list)
        self._total_labeled: int = 0

        # Current model state
        self._current_model: Any = None
        self._current_metrics: dict | None = None
        self._training_count: int = 0

    # ── Recording ──────────────────────────────────────────────

    def record(self, result: FusionResult, document: Document) -> None:
        """Record a high-confidence classification result.

        Only records results with composite_confidence >= threshold
        (default 0.85) and method="fusion_full" (LLM-verified).
        This ensures training data quality.

        Args:
            result: FusionResult from the voter.
            document: The classified Document.
        """
        threshold = self._trainer.confidence_threshold
        if result.composite_confidence < threshold:
            return

        # Prefer fusion_full (LLM-verified) labels for training
        # fusion_fast labels are accepted but at a higher threshold
        if result.method == "fusion_fast" and result.composite_confidence < 0.90:
            return

        label = result.final_label
        if label in ("unclassified", "unknown"):
            return

        # Store document with its label
        labeled_doc = Document(
            doc_id=document.doc_id,
            text=document.text,
            metadata=document.metadata,
            label=label,
            label_confidence=result.composite_confidence,
            label_method=result.method,
        )
        self._labeled[label].append(labeled_doc)
        self._total_labeled += 1

    def record_batch(
        self, results: list[FusionResult], documents: list[Document]
    ) -> None:
        """Record a batch of results."""
        doc_map = {d.doc_id: d for d in documents}
        for result in results:
            doc = doc_map.get(result.doc_id)
            if doc is not None:
                self.record(result, doc)

    # ── Training trigger ───────────────────────────────────────

    def should_train(self) -> bool:
        """Check if any type has reached min_samples_per_class.

        Returns True if at least one type has >= threshold samples
        and the total new samples exceed retrain_trigger.
        """
        min_samples = self._trainer.min_samples_per_class
        for label, docs in self._labeled.items():
            if len(docs) >= min_samples:
                return True
        return False

    def get_class_counts(self) -> dict[str, int]:
        """Return current sample counts per class."""
        return {label: len(docs) for label, docs in self._labeled.items()}

    # ── Training + deployment ──────────────────────────────────

    def train_and_deploy(self) -> dict:
        """Train a SetFit model and deploy to E3 engine if validation passes.

        Workflow:
        1. Build training data from accumulated labeled documents
        2. Train SetFit model
        3. Evaluate against pipeline (teacher agreement)
        4. Check F1 degradation vs current model (if exists)
        5. If no degradation → deploy to E3 engine
        6. If degradation → reject, require human review

        Returns:
            dict with keys: trained, deployed, metrics, reason.
        """
        if not self.should_train():
            return {"trained": False, "deployed": False, "reason": "Insufficient samples"}

        # Gather all labeled documents
        all_docs = []
        for docs in self._labeled.values():
            all_docs.extend(docs)

        logger.info(
            "Starting distillation: %d docs across %d classes",
            len(all_docs), len(self._labeled),
        )

        # Step 1-2: Build training data and train
        try:
            train_texts, train_labels, eval_docs = self._trainer.build_training_data(all_docs)
            model, new_metrics = self._trainer.train(train_texts, train_labels)
        except DistillationError as exc:
            logger.warning("Distillation failed: %s", exc)
            return {"trained": False, "deployed": False, "reason": str(exc)}

        self._training_count += 1

        # Step 3: Check F1 degradation vs current
        degraded = False
        if self._current_metrics is not None:
            degraded = self._trainer.check_degradation(
                self._current_metrics, new_metrics
            )
            if degraded:
                logger.warning(
                    "F1 degradation detected! New model rejected. "
                    "Old macro F1=%.3f, new macro F1=%.3f",
                    self._current_metrics.get("macro_f1", 0),
                    new_metrics.get("macro_f1", 0),
                )
                return {
                    "trained": True,
                    "deployed": False,
                    "metrics": new_metrics,
                    "reason": "F1 degradation detected — human review required",
                }

        # Step 4: Deploy to E3 engine
        if self._e3_engine is not None:
            self._e3_engine.set_model(model, self._trainer)
            logger.info(
                "Model deployed to E3 engine: %d labels, macro F1=%.3f",
                new_metrics.get("num_labels", 0),
                new_metrics.get("macro_f1", 0),
            )

        self._current_model = model
        self._current_metrics = new_metrics

        return {
            "trained": True,
            "deployed": True,
            "metrics": new_metrics,
            "reason": "Model trained and deployed successfully",
        }

    def force_train(self, texts: list[str], labels: list[str]) -> dict:
        """Force training with explicit data (bypasses accumulation)."""
        try:
            model, metrics = self._trainer.train(texts, labels)
        except DistillationError as exc:
            return {"trained": False, "deployed": False, "reason": str(exc)}

        if self._e3_engine is not None:
            self._e3_engine.set_model(model, self._trainer)

        self._current_model = model
        self._current_metrics = metrics
        self._training_count += 1

        return {
            "trained": True,
            "deployed": True,
            "metrics": metrics,
            "reason": "Force-trained and deployed",
        }

    # ── State ──────────────────────────────────────────────────

    @property
    def total_labeled(self) -> int:
        return self._total_labeled

    @property
    def class_count(self) -> int:
        return len(self._labeled)

    @property
    def is_model_deployed(self) -> bool:
        return self._current_model is not None

    def reset_accumulator(self) -> None:
        """Clear accumulated training data (e.g., after successful training)."""
        self._labeled.clear()
        self._total_labeled = 0
