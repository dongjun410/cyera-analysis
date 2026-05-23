"""E3: ML classifier engine — wraps SetFit distilled model.

This engine is UNAVAILABLE until a SetFit model has been trained
(sample_count >= 50 per class). Once trained, it provides ~2ms
inference with confidence from predict_proba.

Delegates to distillation/trainer.py for the actual model.
"""

from __future__ import annotations

from typing import Any

from src.engines.base import BaseEngine
from src.types import Document, EngineOutput


class E3MLEngine(BaseEngine):
    """SetFit ML classifier engine.

    Wraps a trained SetFit model for fast (~2ms) CPU inference.
    Unavailable until training has occurred.

    Attributes:
        engine_id: "E3_ml"
        weight: 1.5 (statistical model, broad coverage)
    """

    engine_id = "E3_ml"

    def __init__(self) -> None:
        self._model: Any = None
        self._trainer: Any = None

    def set_model(self, model: Any, trainer: Any = None) -> None:
        """Set the trained SetFit model, making this engine available."""
        self._model = model
        self._trainer = trainer

    @property
    def weight(self) -> float:
        return 1.5

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def analyze(self, doc: Document) -> EngineOutput:
        """Run SetFit inference on the document.

        Returns:
            EngineOutput with predicted label and confidence, or
            status="unavailable" if model not trained.
        """
        if not self.is_available or self._trainer is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )

        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        try:
            label, confidence = self._trainer.predict(self._model, text)
            return EngineOutput(
                engine_id=self.engine_id,
                label=label,
                confidence=round(confidence, 4),
                status="matched",
                metadata={},
            )
        except Exception:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )
