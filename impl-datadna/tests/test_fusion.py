"""Unit tests for the fusion voter."""

from __future__ import annotations

import pytest

from src.engines.e1_regex import E1RegexEngine
from src.engines.e5_structural import E5StructuralEngine
from src.fusion.voter import FusionVoter
from src.types import Document


@pytest.fixture
def hr_doc() -> Document:
    return Document(
        doc_id="doc_hr_1",
        text="Employee SSN: 123-45-6789, Salary: $95,000, Payroll ID: PR-001",
        metadata={"file_type": ".docx"},
    )


@pytest.fixture
def generic_doc() -> Document:
    return Document(
        doc_id="doc_gen_1",
        text="The weather is nice today.",
        metadata={"file_type": ".txt"},
    )


@pytest.fixture
def voter_with_e1() -> FusionVoter:
    return FusionVoter(engines=[E1RegexEngine()])


class TestFusionVoter:
    def test_single_engine_fusion(self, voter_with_e1, hr_doc):
        result = voter_with_e1.classify(hr_doc)
        assert result.doc_id == "doc_hr_1"
        assert result.final_label is not None
        assert result.final_label != "unclassified"
        assert 0.0 <= result.composite_confidence <= 1.0

    def test_method_is_fusion_fast_without_llm(self, voter_with_e1, hr_doc):
        result = voter_with_e1.classify(hr_doc)
        assert result.method == "fusion_fast"
        assert result.engine_outputs["E6_llm"].status == "skipped"

    def test_engine_outputs_recorded(self, voter_with_e1, hr_doc):
        result = voter_with_e1.classify(hr_doc)
        assert "E1_regex" in result.engine_outputs

    def test_label_scores_dict(self, voter_with_e1, hr_doc):
        result = voter_with_e1.classify(hr_doc)
        assert isinstance(result.label_scores, dict)
        assert len(result.label_scores) > 0

    def test_all_same_label_high_consensus(self, hr_doc):
        e1 = E1RegexEngine()
        e5 = E5StructuralEngine()
        voter = FusionVoter(engines=[e1, e5])
        result = voter.classify(hr_doc)
        assert result.final_label is not None
        assert result.final_label != "unclassified"

    def test_generic_document_all_no_match(self, generic_doc):
        e1 = E1RegexEngine()
        e5 = E5StructuralEngine()
        voter = FusionVoter(engines=[e1, e5])
        result = voter.classify(generic_doc)
        assert result.final_label == "unclassified"
        assert result.manual_review is True

    def test_manual_review_flagged_low_confidence(self):
        voter = FusionVoter(engines=[])
        doc = Document(doc_id="d", text="test", metadata={})
        result = voter.classify(doc)
        assert result.manual_review is True
        assert result.composite_confidence == 0.0
