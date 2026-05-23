"""E6: LLM classification engine.

Wraps Mistral-7B via llm/client.py. Activated only when E1-E5 preliminary
fusion confidence < 0.85 (gate controlled by fusion/voter.py).

Highest weight (2.0) — when LLM has an opinion, it dominates the vote.
Highest latency (~1.4s) — the reason for the preliminary consensus gate.

Degradation: if Ollama is down, this engine returns "unavailable".
System continues with E1-E5 fusion (degraded=true).
"""

from __future__ import annotations

from src.engines.base import BaseEngine
from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, EngineOutput


class E6LLMEngine(BaseEngine):
    """LLM classification engine — Mistral-7B via Ollama.

    Attributes:
        engine_id: "E6_llm"
        weight: 2.0 (highest accuracy, dominates when available)
    """

    engine_id = "E6_llm"

    def __init__(
        self,
        llm_client=None,
        type_library: TypeLibrary | None = None,
    ) -> None:
        """Initialize the LLM engine.

        Args:
            llm_client: MistralClient instance (or None -> unavailable).
            type_library: TypeLibrary for known type names.
        """
        self._client = llm_client
        self._type_library = type_library or get_type_library()

    @property
    def weight(self) -> float:
        return 2.0

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def analyze(self, doc: Document) -> EngineOutput:
        """Classify document using Mistral-7B.

        Returns:
            EngineOutput with LLM-assigned label and confidence, or
            status="unavailable" if LLM client not configured.
        """
        if not self.is_available or self._client is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )

        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        try:
            known_types = [
                t.type_name for t in self._type_library.list_active()
            ]
            response = self._client.classify(text, known_types)
            label = response.get("label", "unknown")
            confidence = float(response.get("confidence", 0.0))
            is_new = bool(response.get("is_new_type", False))

            return EngineOutput(
                engine_id=self.engine_id,
                label=label,
                confidence=round(confidence, 4),
                status="matched",
                metadata={
                    "is_new_type": is_new,
                    "rationale": response.get("rationale", ""),
                },
            )
        except Exception:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )
