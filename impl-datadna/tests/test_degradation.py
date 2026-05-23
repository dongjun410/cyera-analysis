"""Degradation path tests -- fault injection for each engine.

Per spec section 6: when any engine fails, the system continues
with remaining engines. Verifies:
  - Each engine can be turned off independently
  - System still produces valid classifications
  - degraded=True is set when engines are unavailable
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.engines.e1_regex import E1RegexEngine
from src.fusion.voter import FusionVoter
from src.types import Document, EngineOutput


@pytest.fixture
def hr_doc() -> Document:
    return Document(
        doc_id="doc_hr_1",
        text="Employee SSN: 123-45-6789, Salary: $95,000, Payroll ID: PR-001",
        metadata={"file_type": ".docx"},
    )


def _make_mock_engine(engine_id: str, weight: float, label: str, confidence: float):
    """Create a mock engine that always returns the same output."""
    engine = Mock()
    engine.engine_id = engine_id
    engine.weight = weight
    engine.is_available = True
    engine.analyze.return_value = EngineOutput(
        engine_id=engine_id,
        label=label,
        confidence=confidence,
        status="matched",
    )
    return engine


def _make_failing_engine(engine_id: str, weight: float):
    """Create a mock engine that always returns unavailable."""
    engine = Mock()
    engine.engine_id = engine_id
    engine.weight = weight
    engine.is_available = False
    engine.analyze.return_value = EngineOutput(
        engine_id=engine_id,
        status="unavailable",
    )
    return engine


class TestDegradationPaths:
    """Verify each engine can fail without crashing the system."""

    def test_all_engines_available_normal(self, hr_doc):
        """Normal operation with all engines working."""
        engines = [
            _make_mock_engine("E1_regex", 1.0, "HR & Payroll", 0.85),
            _make_mock_engine("E2_template", 1.0, "HR Record", 1.0),
            _make_mock_engine("E5_structural", 0.8, "unknown", 0.0),
        ]
        voter = FusionVoter(engines=engines)
        result = voter.classify(hr_doc)
        assert result.final_label != "unclassified"
        assert not result.degraded

    def test_e1_regex_fails_others_continue(self, hr_doc):
        """E1 failure -> E2+E5 still classify."""
        engines = [
            _make_failing_engine("E1_regex", 1.0),
            _make_mock_engine("E2_template", 1.0, "HR Record", 1.0),
            _make_mock_engine("E5_structural", 0.8, "unknown", 0.0),
        ]
        voter = FusionVoter(engines=engines)
        result = voter.classify(hr_doc)
        assert result.final_label != "unclassified"
        assert result.degraded

    def test_e2_template_fails_others_continue(self, hr_doc):
        """E2 failure -> E1+E5 still classify."""
        engines = [
            _make_mock_engine("E1_regex", 1.0, "HR & Payroll", 0.85),
            _make_failing_engine("E2_template", 1.0),
            _make_mock_engine("E5_structural", 0.8, "unknown", 0.0),
        ]
        voter = FusionVoter(engines=engines)
        result = voter.classify(hr_doc)
        assert result.final_label != "unclassified"
        assert result.degraded

    def test_e5_structural_fails_others_continue(self, hr_doc):
        """E5 failure -> E1+E2 still classify."""
        engines = [
            _make_mock_engine("E1_regex", 1.0, "HR & Payroll", 0.85),
            _make_mock_engine("E2_template", 1.0, "HR Record", 1.0),
            _make_failing_engine("E5_structural", 0.8),
        ]
        voter = FusionVoter(engines=engines)
        result = voter.classify(hr_doc)
        assert result.final_label != "unclassified"
        assert result.degraded

    def test_e6_llm_unavailable_still_works(self, hr_doc):
        """E6 LLM unavailable -> system degrades but still works.

        Uses a mock E1 with low confidence to prevent the 0.85 consensus
        threshold from being met, which forces E6 to be invoked. E6 then
        returns unavailable, setting degraded=True but the system still
        produces a classification from E1's output.
        """
        # Mock E1 with low confidence so prelim consensus < 0.85,
        # forcing E6 invocation
        e1 = Mock()
        e1.engine_id = "E1_regex"
        e1.weight = 1.0
        e1.is_available = True
        e1.analyze.return_value = EngineOutput(
            engine_id="E1_regex",
            label="HR & Payroll",
            confidence=0.5,
            status="matched",
        )
        e6 = Mock()
        e6.engine_id = "E6_llm"
        e6.weight = 2.0
        e6.is_available = False
        e6.analyze.return_value = EngineOutput(
            engine_id="E6_llm",
            status="unavailable",
        )
        voter = FusionVoter(engines=[e1, e6])
        result = voter.classify(hr_doc)
        assert result.final_label != "unclassified"
        assert result.degraded
        # method="fusion_full" because E6 was invoked (the full pipeline
        # was attempted, even though E6 returned unavailable)

    def test_all_engines_fail_unclassified(self, hr_doc):
        """All engines fail -> unclassified + manual_review."""
        engines = [
            _make_failing_engine("E1_regex", 1.0),
            _make_failing_engine("E2_template", 1.0),
            _make_failing_engine("E5_structural", 0.8),
        ]
        voter = FusionVoter(engines=engines)
        result = voter.classify(hr_doc)
        assert result.final_label == "unclassified"
        assert result.degraded
        assert result.manual_review

    def test_two_of_three_engines_fail_still_works(self, hr_doc):
        """2 out of 3 fail -> remaining 1 carries the vote."""
        engines = [
            _make_failing_engine("E1_regex", 1.0),
            _make_mock_engine("E2_template", 1.0, "HR Record", 1.0),
            _make_failing_engine("E5_structural", 0.8),
        ]
        voter = FusionVoter(engines=engines)
        result = voter.classify(hr_doc)
        assert result.final_label == "HR Record"
        assert result.degraded

    def test_discovery_collects_low_confidence(self):
        """DiscoveryLoop collects low-confidence outliers via collect_outlier."""
        from unittest.mock import MagicMock

        from src.discovery.loop import DiscoveryLoop
        from src.types import FusionResult

        # DiscoveryLoop requires 4 injected dependencies; mock them all
        structural = MagicMock()
        refiner = MagicMock()
        embedder = MagicMock()
        matcher = MagicMock()

        loop = DiscoveryLoop(structural, refiner, embedder, matcher)
        doc = Document(doc_id="test", text="test", metadata={})
        result = FusionResult(
            doc_id="test",
            final_label="unknown",
            composite_confidence=0.25,
            degraded=False,
            manual_review=True,
        )
        # Low confidence -> collect as outlier with the reason
        loop.collect_outlier(doc, reason="fusion_confidence_below_threshold")
        assert loop.get_buffer_size() == 1
