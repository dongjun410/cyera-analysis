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
        """Classify document using Mistral-7B, constrained to known types only.

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

            # Build constrained prompt that forces known type selection
            response = self._classify_constrained(text, known_types)
            label = response.get("label", "unknown")
            confidence = float(response.get("confidence", 0.0))
            is_new = bool(response.get("is_new_type", False))

            # If LLM returned a label not in known types, find closest match
            if label not in known_types and label != "unknown":
                closest = self._closest_known_type(label, known_types)
                if closest:
                    label = closest
                    confidence = max(0.0, confidence - 0.1)  # penalty for mismatch

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

    def _classify_constrained(
        self, text: str, known_types: list[str]
    ) -> dict:
        """Call LLM with a prompt that forces selection from known types."""
        types_list = "\n".join(f"- {t}" for t in known_types)

        system = (
            "You are a document classifier. You MUST choose exactly one type "
            "from the known types list below. Do NOT invent new types. "
            "If no type matches perfectly, pick the closest one. "
            "Always respond with valid JSON."
        )

        truncated = text[:2000]
        user = (
            "<instruction>Classify this document into EXACTLY ONE of the "
            "known types listed below. Do not create new types. Choose the "
            "closest match even if imperfect.</instruction>\n"
            f"<known_types>\n{types_list}\n</known_types>\n"
            f"<document>{truncated}</document>\n"
            "<output_schema>"
            '{"label": "<one of the known types>", "confidence": 0.0-1.0, '
            '"is_new_type": false, "rationale": "..."}'
            "</output_schema>"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._client._call_llm(messages)

    def _closest_known_type(
        self, label: str, known_types: list[str]
    ) -> str | None:
        """Find the closest known type to an unknown label using simple word overlap."""
        if not known_types:
            return None

        label_words = set(label.lower().split())
        best_type = None
        best_score = 0

        for kt in known_types:
            kt_words = set(kt.lower().split())
            overlap = len(label_words & kt_words)
            # Also check if one is substring of another
            bonus = 2.0 if label.lower() in kt.lower() or kt.lower() in label.lower() else 0
            score = overlap + bonus
            if score > best_score:
                best_score = score
                best_type = kt

        return best_type if best_score >= 1 else None
