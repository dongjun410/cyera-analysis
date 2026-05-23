"""Unit tests for each of the 6 classification engines."""

from __future__ import annotations

import pytest

from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine
from src.types import Document


# ── Test document fixtures ──

@pytest.fixture
def hr_doc() -> Document:
    return Document(
        doc_id="doc_hr_1",
        text="Employee SSN: 123-45-6789, Name: John Smith, Salary: $95,000, "
             "Department: Engineering, Payroll ID: PR-001",
        metadata={"file_type": ".docx", "department": "HR"},
    )


@pytest.fixture
def fin_doc() -> Document:
    return Document(
        doc_id="doc_fin_1",
        text="Quarterly Revenue: $1.2M, Credit card: Visa 4532015112830366, "
             "Net income: $340K, Invoice #INV-2024-089, IBAN CH9300762011623852957",
        metadata={"file_type": ".pdf"},
    )


@pytest.fixture
def med_doc() -> Document:
    return Document(
        doc_id="doc_med_1",
        text="Patient ID: MRN: 88421, Diagnosis: Hypertension, "
             "NPI: 1234567890, Prescribed: Lisinopril 10mg daily",
        metadata={"file_type": ".pdf"},
    )


@pytest.fixture
def generic_doc() -> Document:
    return Document(
        doc_id="doc_gen_1",
        text="The weather is nice today and the sun is shining brightly.",
        metadata={"file_type": ".txt"},
    )


@pytest.fixture
def empty_doc() -> Document:
    return Document(doc_id="doc_empty", text="", metadata={})


# ── E1: Regex Engine ──

class TestE1RegexEngine:
    def test_weight_is_1_0(self):
        engine = E1RegexEngine()
        assert engine.weight == 1.0

    def test_is_available_by_default(self):
        engine = E1RegexEngine()
        assert engine.is_available is True

    def test_matches_hr_document(self, hr_doc):
        engine = E1RegexEngine()
        output = engine.analyze(hr_doc)
        assert output.status == "matched"
        assert output.engine_id == "E1_regex"
        assert "HR" in output.label

    def test_matches_financial_document(self, fin_doc):
        engine = E1RegexEngine()
        output = engine.analyze(fin_doc)
        assert output.status == "matched"
        assert output.label == "Financial Report"

    def test_matches_medical_document(self, med_doc):
        engine = E1RegexEngine()
        output = engine.analyze(med_doc)
        assert output.status == "matched"
        assert output.label == "Medical Record"

    def test_no_match_generic_document(self, generic_doc):
        engine = E1RegexEngine()
        output = engine.analyze(generic_doc)
        assert output.status == "no_match"
        assert output.label is None

    def test_empty_document_no_match(self, empty_doc):
        engine = E1RegexEngine()
        output = engine.analyze(empty_doc)
        assert output.status == "no_match"

    def test_confidence_in_range(self, hr_doc):
        engine = E1RegexEngine()
        output = engine.analyze(hr_doc)
        assert 0.0 <= output.confidence <= 1.0


# ── E2: Template Hash Engine ──

class TestE2TemplateEngine:
    def test_weight_is_1_0(self):
        engine = E2TemplateEngine()
        assert engine.weight == 1.0

    def test_pii_rich_document_processed(self, hr_doc):
        engine = E2TemplateEngine()
        output = engine.analyze(hr_doc)
        assert output.engine_id == "E2_template"
        assert output.status in ("matched", "no_match")

    def test_empty_document_no_match(self, empty_doc):
        engine = E2TemplateEngine()
        output = engine.analyze(empty_doc)
        assert output.status == "no_match"

    def test_pii_replacement_produces_hash(self, hr_doc):
        engine = E2TemplateEngine()
        replaced_text, count = engine._replace_pii(hr_doc.text)
        assert "[SSN]" in replaced_text
        assert count >= 1


# ── E3: ML Engine ──

class TestE3MLEngine:
    def test_weight_is_1_5(self):
        engine = E3MLEngine()
        assert engine.weight == 1.5

    def test_unavailable_when_no_model(self, hr_doc):
        engine = E3MLEngine()
        assert engine.is_available is False
        output = engine.analyze(hr_doc)
        assert output.status == "unavailable"

    def test_becomes_available_after_set_model(self):
        engine = E3MLEngine()
        engine.set_model(object(), object())
        assert engine.is_available is True


# ── E4: kNN Engine ──

class TestE4kNNEngine:
    def test_weight_is_1_0(self):
        engine = E4kNNEngine()
        assert engine.weight == 1.0

    def test_unavailable_without_embedder(self, hr_doc):
        engine = E4kNNEngine(embedder=None)
        assert engine.is_available is False
        output = engine.analyze(hr_doc)
        assert output.status == "unavailable"


# ── E5: Structural Engine ──

class TestE5StructuralEngine:
    def test_weight_is_0_8(self):
        engine = E5StructuralEngine()
        assert engine.weight == pytest.approx(0.8)

    def test_no_match_without_metadata(self):
        engine = E5StructuralEngine()
        doc = Document(doc_id="d", text="irrelevant", metadata={})
        output = engine.analyze(doc)
        assert output.status == "no_match"

    def test_returns_signature_hash(self, hr_doc):
        engine = E5StructuralEngine()
        output = engine.analyze(hr_doc)
        assert "signature_hash" in output.metadata
        assert len(output.metadata["signature_hash"]) == 64


# ── E6: LLM Engine ──

class TestE6LLMEngine:
    def test_weight_is_2_0(self):
        engine = E6LLMEngine()
        assert engine.weight == 2.0

    def test_unavailable_without_client(self, hr_doc):
        engine = E6LLMEngine(llm_client=None)
        assert engine.is_available is False
        output = engine.analyze(hr_doc)
        assert output.status == "unavailable"
