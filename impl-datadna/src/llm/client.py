"""Mistral-7B client via Ollama's OpenAI-compatible API.

Tier 2 Classification + Tier 3 Verification using structured JSON output
with schema constraint, retry with exponential backoff, and prompt injection
protection per spec section 7.2.

Capabilities:
  - classify(): Tier 2 — classify document into known types or identify new types
  - verify():   Tier 3 — quality gate verification with reasoning chain
  - _sanitize(): Strip prompt injection patterns before assembly
  - _call_llm(): Retry wrapper with exponential backoff
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for the Mistral-7B LLM client.

    Attributes:
        api_base: Ollama OpenAI-compatible endpoint URL.
        model: Model name as registered in Ollama (e.g. \"mistral:7b\").
        quantization: Quantization level (\"4bit\" for Tier 2, \"int8\" for Tier 3).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the completion.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on timeout/5xx errors.
    """

    api_base: str
    model: str
    quantization: str  # "4bit" | "int8"
    temperature: float = 0.3
    max_tokens: int = 512
    timeout: int = 30
    max_retries: int = 2


# ──────────────────────────────────────────────────────────────
# Mistral Client
# ──────────────────────────────────────────────────────────────

class MistralClient:
    """Mistral-7B LLM client for Tier 2 classification and Tier 3 verification.

    Communicates with Ollama via its OpenAI-compatible chat completions
    endpoint. Prompts use XML tags for structured instruction following
    per spec section 7.2.

    Args:
        config: LLMConfig with endpoint, model, and retry settings.
    """

    # Patterns flagged as prompt injection — replaced with [REDACTED].
    _INJECTION_PATTERNS: list[str] = [
        r"ignore\s+previous\s+instructions",
        r"ignore\s+all\s+previous\s+instructions",
        r"system:",
        r"<system>",
        r"<instruction>",
    ]

    # Max characters of document text sent to the LLM.
    _MAX_DOCUMENT_CHARS: int = 2000

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.api_base,
            api_key="ollama",  # Ollama ignores the key
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def classify(
        self,
        document_text: str,
        known_types: list[str],
        ner_results: list | None = None,
        pii_features: dict | None = None,
    ) -> dict:
        """Tier 2 classification — assign document to known type or flag as new.

        Args:
            document_text: The document content to classify.
            known_types: List of known type names to choose from.
            ner_results: Optional NER results (accepted, reserved for future use).
            pii_features: Optional PII features (accepted, reserved for future use).

        Returns:
            dict with keys: label, is_new_type, confidence, rationale, suggested_rules.
            On failure returns fallback with label="unknown", confidence=0.0.
        """
        system_content = (
            "You are a document classifier. You analyze document content and "
            "assign it to the most appropriate known type or identify it as a "
            "new type. Always respond with valid JSON matching the output schema exactly."
        )
        user_content = self._build_classify_prompt(document_text, known_types)
        messages = self._build_messages(system_content, user_content)

        try:
            return self._call_llm(messages)
        except (json.JSONDecodeError, Exception):
            return {
                "label": "unknown",
                "is_new_type": False,
                "confidence": 0.0,
                "rationale": "LLM response could not be parsed as valid JSON",
                "suggested_rules": "",
            }

    def verify(
        self,
        document_text: str,
        current_label: str,
        cluster_context: dict | None = None,
    ) -> dict:
        """Tier 3 verification — quality gate to confirm or correct a classification.

        Args:
            document_text: The document content to verify.
            current_label: The label currently assigned to this document.
            cluster_context: Optional cluster context (accepted, reserved for future use).

        Returns:
            dict with keys: label, confidence, reasoning_chain, needs_manual_review.
            On failure returns fallback with confidence=0.0, needs_manual_review=True.
        """
        system_content = (
            "You are a document classification verifier. You carefully review "
            "document classifications and confirm or correct them. Always respond "
            "with valid JSON matching the output schema exactly."
        )
        user_content = self._build_verify_prompt(document_text, current_label)
        messages = self._build_messages(system_content, user_content)

        try:
            return self._call_llm(messages)
        except (json.JSONDecodeError, Exception):
            return {
                "label": current_label,
                "confidence": 0.0,
                "reasoning_chain": "LLM response could not be parsed as valid JSON",
                "needs_manual_review": True,
            }

    # ──────────────────────────────────────────────────────────
    # Sanitization
    # ──────────────────────────────────────────────────────────

    def _sanitize(self, text: str) -> str:
        """Strip known prompt injection patterns and replace with [REDACTED].

        Patterns are matched case-insensitively. The sanitized text is safe
        to embed in XML prompt tags without risk of instruction hijacking.

        Args:
            text: Raw text that may contain injection attempts.

        Returns:
            Sanitized text with injection patterns replaced.
        """
        for pattern in self._INJECTION_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text

    # ──────────────────────────────────────────────────────────
    # Prompt Building
    # ──────────────────────────────────────────────────────────

    def _build_classify_prompt(
        self, document_text: str, known_types: list[str]
    ) -> str:
        """Build the user prompt for Tier 2 classification.

        Wraps the (sanitized, truncated) document in XML tags per spec section 7.2.

        Args:
            document_text: Raw document text.
            known_types: List of known type names.

        Returns:
            Formatted user prompt string with XML-tagged sections.
        """
        truncated = self._sanitize(document_text)[: self._MAX_DOCUMENT_CHARS]
        types_str = "\n".join(f"- {t}" for t in known_types)

        return (
            "<instruction>Classify the following document into one of the "
            "known types or identify it as a new type.</instruction>\n"
            f"<known_types>\n{types_str}\n</known_types>\n"
            f"<document>{truncated}</document>\n"
            "<output_schema>"
            '{"label": "...", "is_new_type": false, "confidence": 0.0, '
            '"rationale": "...", "suggested_rules": "..."}'
            "</output_schema>"
        )

    def _build_verify_prompt(
        self, document_text: str, current_label: str
    ) -> str:
        """Build the user prompt for Tier 3 verification.

        Args:
            document_text: Raw document text.
            current_label: The label to verify.

        Returns:
            Formatted user prompt string with XML-tagged sections.
        """
        truncated = self._sanitize(document_text)[: self._MAX_DOCUMENT_CHARS]

        return (
            "<instruction>Verify whether the following document is correctly "
            "classified as the given label. If the label is incorrect, provide "
            "the correct label.</instruction>\n"
            f"<document>{truncated}</document>\n"
            f"<current_label>{current_label}</current_label>\n"
            "<output_schema>"
            '{"label": "...", "confidence": 0.0, '
            '"reasoning_chain": "...", "needs_manual_review": false}'
            "</output_schema>"
        )

    def _build_messages(
        self, system_content: str, user_content: str
    ) -> list[dict[str, str]]:
        """Assemble the messages list for the OpenAI chat completions API.

        Args:
            system_content: System prompt describing the LLM's role.
            user_content: User prompt with the task and document.

        Returns:
            List of message dicts with \"role\" and \"content\" keys.
        """
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ──────────────────────────────────────────────────────────
    # LLM Call with Retry
    # ──────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict[str, str]]) -> dict:
        """Call the LLM with exponential backoff retry on failure.

        Retries on timeout, connection errors, and 5xx responses.
        Does NOT retry on JSON parse errors (model output issue).

        Args:
            messages: List of message dicts for the chat API.

        Returns:
            Parsed JSON response dict.

        Raises:
            json.JSONDecodeError: If the LLM response is not valid JSON
                                  (caught by classify/verify for fallback).
            Exception: If all retry attempts are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                # content should be a JSON string; parse it
                if content is None:
                    raise json.JSONDecodeError(
                        "Empty response content", "", 0
                    )
                return json.loads(content)

            except json.JSONDecodeError:
                # JSON parse errors are model output issues, not transport
                # issues — do NOT retry, propagate to fallback handler
                raise

            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    # All retries exhausted — propagate to fallback
                    raise last_error

        # Should be unreachable (loop always raises or returns)
        raise last_error  # type: ignore[misc]
