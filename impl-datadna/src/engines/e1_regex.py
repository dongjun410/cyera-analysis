"""E1: Regex rule engine.

Deterministic document type inference via 55+ pre-built regex rules.
Each rule matches document text patterns and associated PII types.
Confidence = base_confidence + PII boost (max 1.0).

Independent of all other engines. Only dependency is the rule library.
"""

from __future__ import annotations

from src.engines.base import BaseEngine
from src.knowledge.rules import BUILTIN_RULES, PII_PATTERNS, DocTypeRule
from src.types import Document, EngineOutput


class E1RegexEngine(BaseEngine):
    """Regex rule engine — matches document text against 55+ type rules.

    For each rule that fires, computes:
      base_confidence + PII boost (matching associated_pii types × 0.3)

    Returns the highest-confidence match. If multiple rules match,
    only the top-scoring (label, confidence) is returned.

    Attributes:
        engine_id: "E1_regex"
        weight: 1.0 (deterministic, low false-positive rate)
    """

    engine_id = "E1_regex"

    def __init__(self) -> None:
        self._rules: list[DocTypeRule] = list(BUILTIN_RULES)

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, doc: Document) -> EngineOutput:
        """Run all regex rules against the document text.

        Returns:
            EngineOutput with the highest-confidence match, or
            status="no_match" if no rule fires.
        """
        text = doc.text or ""
        if not text:
            return EngineOutput(
                engine_id=self.engine_id,
                status="no_match",
            )

        best_label = None
        best_confidence = 0.0
        best_rule_id = None

        for rule in self._rules:
            if not rule.pattern.search(text):
                continue

            # Base confidence from rule
            confidence = rule.base_confidence

            # PII boost: check how many associated_pii types are present
            if rule.associated_pii:
                pii_matches = 0
                for pii_type in rule.associated_pii:
                    pii_re = PII_PATTERNS.get(pii_type)
                    if pii_re is not None and pii_re.search(text):
                        pii_matches += 1
                boost = min(pii_matches / len(rule.associated_pii), 1.0) * 0.3
                confidence += boost

            confidence = min(confidence, 1.0)

            if confidence > best_confidence:
                best_confidence = confidence
                best_label = rule.label
                best_rule_id = rule.rule_id

        if best_label is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="no_match",
            )

        return EngineOutput(
            engine_id=self.engine_id,
            label=best_label,
            confidence=round(best_confidence, 4),
            status="matched",
            metadata={"rule_id": best_rule_id},
        )
