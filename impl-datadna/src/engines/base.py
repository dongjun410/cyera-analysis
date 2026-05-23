"""Base class for all classification engines.

Every engine has the same minimal contract:
  - analyze(doc) → EngineOutput
  - weight: float (pre-set based on validation accuracy)
  - is_available: bool (runtime health check)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Document, EngineOutput


class BaseEngine(ABC):
    """Abstract base for all 6 classification engines.

    Subclasses must implement analyze(), weight, and is_available.
    The fusion voter calls analyze() on every engine and uses weight
    in the weighted voting calculation.
    """

    @abstractmethod
    def analyze(self, doc: Document) -> EngineOutput:
        """Run this engine on a document and return its output.

        Args:
            doc: Document with text and metadata.

        Returns:
            EngineOutput with engine_id, label, confidence, status, metadata.
            If the engine cannot produce output, status must be "unavailable"
            or "no_match" (not an exception).
        """
        ...

    @property
    @abstractmethod
    def weight(self) -> float:
        """Pre-set engine weight for fusion voting.

        Weights per spec section 2.3:
          E1 regex: 1.0, E2 template: 1.0, E3 ML: 1.5,
          E4 kNN: 1.0, E5 structural: 0.8, E6 LLM: 2.0
        """
        ...

    @property
    def is_available(self) -> bool:
        """Whether this engine is ready to produce output.

        Default True. Override if the engine has runtime dependencies
        that may be unavailable (e.g. model not loaded, service down).
        """
        return True
