"""
Knowledge Distillation Trainer — LLM (teacher) → SetFit (student).

Compresses Tier 2 + Tier 3 classification knowledge into a ~2 ms CPU inference
model using SetFit with sentence-transformers embeddings. The distilled model
replaces the slow LLM path for the majority of documents while preserving the
quality-gate guarantees of the full pipeline.

Core workflow:
    1. build_training_data  — filter high-confidence pipeline results, balance per class
    2. train                — fine-tune a SetFit model on the curated training set
    3. evaluate_against_pipeline — validate agreement between distilled and full pipeline
    4. predict              — fast CPU inference wrapper
    5. should_retrain / check_degradation — triggers for keeping the model current
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from src.types import ClassificationResult, Document, KnownType

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "min_samples_per_class": 50,
    "max_samples_per_class": 200,
    "confidence_threshold": 0.85,
    "human_review_ratio": 0.1,
    "eval_split": 0.1,
    "retrain_trigger": 500,
    "f1_degradation_threshold": 0.03,
    "confidence_threshold_accept": 0.85,
    "confidence_threshold_fallback": 0.6,
    "seed": 42,
    "setfit_model_id": "BAAI/bge-m3",
}


class DistillationError(Exception):
    """Raised when distillation prerequisites are not met."""


class DistillationTrainer:
    """SetFit-based teacher→student distillation.

    Compresses Tier 2+3 classification to ~2 ms CPU inference.  The trainer
    collects high-confidence pipeline outputs, balances training data per
    label via inverse-frequency sampling, and fine-tunes a SetFit model
    that can serve as a fast fallback for the full LLM pipeline.

    Parameters
    ----------
    config : dict | None
        Overrides for any of the default configuration keys.  Merged shallowly
        on top of DEFAULT_CONFIG.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)

        self.min_samples_per_class: int = cfg["min_samples_per_class"]
        self.max_samples_per_class: int = cfg["max_samples_per_class"]
        self.confidence_threshold: float = cfg["confidence_threshold"]
        self.human_review_ratio: float = cfg["human_review_ratio"]
        self.eval_split: float = cfg["eval_split"]
        self.retrain_trigger: int = cfg["retrain_trigger"]
        self.f1_degradation_threshold: float = cfg["f1_degradation_threshold"]
        self.confidence_threshold_accept: float = cfg["confidence_threshold_accept"]
        self.confidence_threshold_fallback: float = cfg["confidence_threshold_fallback"]
        self.seed: int = cfg["seed"]
        self.setfit_model_id: str = cfg["setfit_model_id"]

        self.model: Any = None  # populated by train()

    # ------------------------------------------------------------------
    # build_training_data
    # ------------------------------------------------------------------

    def build_training_data(
        self,
        classified_docs: list[Document],
    ) -> tuple[list[str], list[str], list[Document]]:
        """Filter, balance, and split high-confidence pipeline outputs into a training set.

        Steps
        -----
        1. Keep only documents whose ``label_confidence > confidence_threshold``
           (default 0.85) and whose ``label`` is not None.
        2. Group retained documents by their ``label`` value.
        3. For each label, determine the target sample count using *inverse
           frequency*: rare labels get up to ``max_samples_per_class``, common
           labels get ``min_samples_per_class``.  If fewer than the target are
           available, all available documents are used.
        4. Randomly hold out ``eval_split`` (default 10 %) of each label's
           sampled documents for downstream validation.

        Parameters
        ----------
        classified_docs : list[Document]
            Documents that have been through the full pipeline (Tier 0–3) and
            carry a ``label`` and ``label_confidence``.

        Returns
        -------
        train_texts : list[str]
            Document texts for fine-tuning.
        train_labels : list[str]
            Corresponding labels.
        eval_docs : list[Document]
            Held-out documents that can be passed to
            :meth:`evaluate_against_pipeline`.
        """
        rng = random.Random(self.seed)

        # ---- 1. Filter by confidence -------------------------------------------------
        high_conf: list[Document] = [
            d
            for d in classified_docs
            if d.label is not None and d.label_confidence >= self.confidence_threshold
        ]

        if not high_conf:
            raise DistillationError(
                "No documents pass the confidence threshold "
                f"({self.confidence_threshold}).  "
                "Check pipeline output quality or lower the threshold."
            )

        # ---- 2. Group by label -------------------------------------------------------
        by_label: dict[str, list[Document]] = defaultdict(list)
        for doc in high_conf:
            assert doc.label is not None  # guarded above
            by_label[doc.label].append(doc)

        # ---- 3. Inverse-frequency sample size per label ------------------------------
        total = len(high_conf)
        label_frequencies: dict[str, float] = {
            lbl: len(docs) / total for lbl, docs in by_label.items()
        }

        # Inverse frequency: rare → high weight, common → low weight
        min_freq = min(label_frequencies.values())
        max_freq = max(label_frequencies.values())
        freq_range = max_freq - min_freq if max_freq > min_freq else 1.0

        def _sample_size(label: str) -> int:
            available = len(by_label[label])
            if available <= self.min_samples_per_class:
                return available

            freq = label_frequencies[label]
            # Normalise to [0, 1] where 0 = rarest, 1 = commonest
            t = (freq - min_freq) / freq_range  # 0=rare … 1=common
            # Invert: rare wants max_samples, common wants min_samples
            target = int(
                round(self.max_samples_per_class - t * (self.max_samples_per_class - self.min_samples_per_class))
            )
            return min(target, available)

        # ---- 4. Sample and split train / eval ---------------------------------------
        train_texts: list[str] = []
        train_labels: list[str] = []
        eval_docs: list[Document] = []

        for label, docs in by_label.items():
            n = _sample_size(label)
            sampled = rng.sample(docs, min(n, len(docs)))
            rng.shuffle(sampled)

            split_at = max(1, int(len(sampled) * (1.0 - self.eval_split)))
            train_docs = sampled[:split_at]
            eval_docs.extend(sampled[split_at:])

            for d in train_docs:
                train_texts.append(d.text)
                train_labels.append(d.label)

        return train_texts, train_labels, eval_docs

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(
        self, texts: list[str], labels: list[str]
    ) -> tuple[Any, dict]:
        """Train a SetFit model on the provided (text, label) pairs.

        Uses ``BAAI/bge-m3`` (or the model set via ``setfit_model_id`` in
        config) as the sentence-transformers backbone.

        If the ``setfit`` package is not installed this method raises
        :class:`NotImplementedError` with clear installation instructions
        rather than failing with a cryptic import error.

        Parameters
        ----------
        texts : list[str]
            Training document texts.
        labels : list[str]
            Integer-compatible or string labels (one per text).

        Returns
        -------
        model : Any
            The trained SetFit model object.
        metrics : dict
            Evaluation metrics dictionary containing at least:
            * ``per_class_f1`` — ``dict[str, float]``
            * ``macro_f1`` — ``float``
        """
        if not texts:
            raise DistillationError("Training data is empty – cannot train.")

        try:
            from setfit import SetFitModel, Trainer, TrainingArguments  # type: ignore[import-untyped]
            from datasets import Dataset  # type: ignore[import-untyped]
        except ImportError:
            raise NotImplementedError(
                "setfit and datasets are required for knowledge distillation.\n"
                "Install with:  pip install setfit datasets\n"
                "This provides the SetFit (Sentence Transformer Fine-Tuning) "
                "package for few-shot text classification."
            )

        # Label → integer encoding (SetFit expects integer labels)
        unique_labels = sorted(set(labels))
        label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        int_labels = [label2id[lbl] for lbl in labels]

        # Build HF Dataset
        train_dataset = Dataset.from_dict({"text": texts, "label": int_labels})
        train_dataset = train_dataset.shuffle(seed=self.seed)

        # Load SetFit model
        model = SetFitModel.from_pretrained(self.setfit_model_id)

        # Training arguments (few-shot friendly defaults)
        args = TrainingArguments(
            output_dir="/tmp/setfit-checkpoints",
            num_epochs=3,
            batch_size=16,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="no",
            load_best_model_at_end=False,
            seed=self.seed,
        )

        # We need an eval split for per-class F1
        train_test = train_dataset.train_test_split(
            test_size=self.eval_split, seed=self.seed
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_test["train"],
            eval_dataset=train_test["test"],
        )
        trainer.train()

        # Evaluate on the eval split
        eval_results = trainer.evaluate()
        eval_dataset = train_test["test"]
        y_true = [id2label[ex["label"]] for ex in eval_dataset]
        y_pred_raw = model.predict([ex["text"] for ex in eval_dataset])

        # SetFit model.predict returns ndarray of int labels
        y_pred: list[str]
        if hasattr(y_pred_raw, "tolist"):
            y_pred = [id2label[int(p)] for p in y_pred_raw.tolist()]
        else:
            y_pred = [id2label[int(p)] for p in y_pred_raw]

        # Per-class F1
        per_class_f1 = self._compute_per_class_f1(y_true, y_pred, unique_labels)
        macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0

        metrics: dict = {
            "per_class_f1": per_class_f1,
            "macro_f1": macro_f1,
            "eval_loss": eval_results.get("eval_loss", None),
            "num_labels": len(unique_labels),
            "train_samples": len(texts),
        }

        self.model = model
        return model, metrics

    # ------------------------------------------------------------------
    # evaluate_against_pipeline
    # ------------------------------------------------------------------

    def evaluate_against_pipeline(
        self,
        model: Any,
        eval_docs: list[Document],
        pipeline_results: list[ClassificationResult],
    ) -> dict:
        """Compare distilled model predictions against the full pipeline.

        For each evaluation document the distilled model is invoked and its
        top-1 prediction is compared with the label assigned by the full
        pipeline (teacher).

        Parameters
        ----------
        model : Any
            A trained SetFit model (returned by :meth:`train`).
        eval_docs : list[Document]
            Held-out documents (produced by :meth:`build_training_data`).
        pipeline_results : list[ClassificationResult]
            Reference labels from the full pipeline.  Must be aligned with
            ``eval_docs`` by ``doc_id``.

        Returns
        -------
        dict
            * ``agreement_rate`` (``float``) — overall fraction of matches
            * ``per_class_agreement`` (``dict[str, float]``)
            * ``total_evaluated`` (``int``)
        """
        if model is None:
            raise DistillationError(
                "model is None.  Call train() first or load a previously "
                "trained model."
            )

        if not eval_docs:
            return {
                "agreement_rate": 1.0,
                "per_class_agreement": {},
                "total_evaluated": 0,
            }

        # Index pipeline results by doc_id for O(1) lookup
        ref: dict[str, ClassificationResult] = {
            r.doc_id: r for r in pipeline_results
        }

        total = 0
        agreements = 0
        per_class: dict[str, dict[str, int]] = defaultdict(
            lambda: {"agree": 0, "total": 0}
        )

        for doc in eval_docs:
            ref_result = ref.get(doc.doc_id)
            if ref_result is None:
                continue  # no pipeline result → skip

            pred_label, _pred_conf = self.predict(model, doc.text)

            per_class[ref_result.label]["total"] += 1
            total += 1
            if pred_label == ref_result.label:
                agreements += 1
                per_class[ref_result.label]["agree"] += 1

        if total == 0:
            return {
                "agreement_rate": 1.0,
                "per_class_agreement": {},
                "total_evaluated": 0,
            }

        per_class_agreement: dict[str, float] = {}
        for lbl, counts in per_class.items():
            if counts["total"] > 0:
                per_class_agreement[lbl] = counts["agree"] / counts["total"]

        return {
            "agreement_rate": agreements / total,
            "per_class_agreement": per_class_agreement,
            "total_evaluated": total,
        }

    # ------------------------------------------------------------------
    # should_retrain
    # ------------------------------------------------------------------

    def should_retrain(self, new_sample_count: int) -> bool:
        """Return ``True`` when the accumulated new samples meet or exceed
        the retrain trigger threshold.

        Parameters
        ----------
        new_sample_count : int
            Number of new high-confidence samples accumulated since the
            last training run.

        Returns
        -------
        bool
        """
        return new_sample_count >= self.retrain_trigger

    # ------------------------------------------------------------------
    # check_degradation
    # ------------------------------------------------------------------

    def check_degradation(self, old_metrics: dict, new_metrics: dict) -> bool:
        """Return ``True`` if **any** class F1 dropped by more than
        ``f1_degradation_threshold`` (default 0.03).

        Parameters
        ----------
        old_metrics : dict
            Metrics dict from a previous :meth:`train` call.  Must contain
            ``per_class_f1``.
        new_metrics : dict
            Metrics dict from the most recent :meth:`train` call.

        Returns
        -------
        bool
            ``True`` → degradation detected; candidate model should **not**
            replace the production model without manual review.
        """
        old_f1: dict[str, float] = old_metrics.get("per_class_f1", {})
        new_f1: dict[str, float] = new_metrics.get("per_class_f1", {})

        # A new class appearing in old but missing in new also counts as
        # degradation (F1 effectively dropped to 0).
        for class_name, old_score in old_f1.items():
            new_score = new_f1.get(class_name, 0.0)
            if old_score - new_score > self.f1_degradation_threshold:
                return True

        return False

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, model: Any, text: str) -> tuple[str, float]:
        """Run inference with the distilled model on a single document.

        Wraps ``SetFitModel.predict()`` and returns the top-1 label
        together with a confidence score.

        Parameters
        ----------
        model : Any
            A trained SetFit model.
        text : str
            Document text to classify.

        Returns
        -------
        label : str
            Predicted document type label.
        confidence : float
            Confidence score in [0, 1].

        Raises
        ------
        DistillationError
            If ``model`` is ``None``.
        """
        if model is None:
            raise DistillationError(
                "model is None.  Call train() first or load a previously "
                "trained model."
            )

        raw = model.predict([text])

        # SetFit predict_proba returns (n_samples, n_classes) ndarray when
        # available; fall back to hard label otherwise.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])
            if hasattr(proba, "tolist"):
                proba_list = proba.tolist()[0]
            else:
                proba_list = list(proba[0])
            best_idx = max(range(len(proba_list)), key=lambda i: proba_list[i])
            confidence = float(proba_list[best_idx])
        else:
            # Hard-label only — no confidence available
            best_idx = int(raw[0]) if hasattr(raw, "__getitem__") else int(raw)
            confidence = 1.0

        # Map integer index back to label string
        if hasattr(model, "model") and hasattr(model.model, "config"):
            id2label = getattr(model.model.config, "id2label", {})
        else:
            id2label = {}

        label = id2label.get(best_idx, str(best_idx))
        return label, confidence

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_per_class_f1(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
    ) -> dict[str, float]:
        """Compute per-class F1 scores without external dependencies.

        Uses the standard formula::

            P = TP / (TP + FP)
            R = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)

        Returns ``{label: f1}`` for every label in *labels*.
        """
        f1_scores: dict[str, float] = {}
        for lbl in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1_scores[lbl] = f1

        return f1_scores
