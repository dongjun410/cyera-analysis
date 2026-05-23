"""Shared pytest fixtures for 6-engine fusion architecture tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.types import Document


@pytest.fixture(scope="module")
def sample_documents() -> list[Document]:
    """20 synthetic documents covering HR, Finance, Medical, API, General."""
    hr_texts = [
        "Employee SSN: 123-45-6789, Name: John Smith, Start date: 2020-03-15, Department: Engineering, Salary: $95,000",
        "Employee SSN: 456-78-9012, Name: Sarah Chen, Start date: 2019-07-01, Department: Marketing, Salary: $110,000",
        "Employee SSN: 789-01-2345, Name: Michael Brown, Start date: 2021-11-01, Department: Finance, Title: Senior Analyst",
        "Employee SSN: 321-65-9870, Name: Emily Davis, Start date: 2023-01-15, Department: HR, Benefits: Full medical + dental",
    ]
    fin_texts = [
        "Quarterly revenue: $1.2M, Credit card payment: Visa 4532015112830366, Net income: $340K, CFO approval: required",
        "Invoice #INV-2024-089: Total $45,230. Payment method: Mastercard 5500000000000004, Due: 30 days net",
        "Expense report March 2024: Travel $3,450, Meals $890, Credit card: 4532015112830366, Submitted by: CFO",
        "Bank statement Q1 2024: Account #****4321, Balance: $1,450,000. Wire transfer: $250,000 to IBAN CH9300762011623852957",
    ]
    med_texts = [
        "Patient ID: MRN: 88421, Diagnosis: Hypertension, NPI: 1234567890, Prescribed: Lisinopril 10mg daily, Follow-up: 3 months",
        "Patient ID: MRN: 55102, Diagnosis: Type 2 Diabetes, NPI: 9876543210, A1C: 7.2%, Medication: Metformin 500mg BID",
        "Patient ID: MRN: 77634, Diagnosis: Anxiety Disorder, NPI: 4567890123, Referred to: Dr. James Wilson, Psychiatry Dept",
        "Patient ID: MRN: 99201, Diagnosis: COPD, NPI: 2345678901, Pulmonary function test scheduled, Smoking cessation advised",
    ]
    api_texts = [
        '{"timestamp": "2024-01-15T08:23:45Z", "endpoint": "/api/users", "status": 200, "response_time_ms": 45}',
        '{"timestamp": "2024-01-15T09:10:12Z", "endpoint": "/api/orders", "status": 201, "response_time_ms": 120}',
        '{"timestamp": "2024-01-15T10:45:33Z", "endpoint": "/api/auth/login", "status": 401, "response_time_ms": 15}',
        '{"timestamp": "2024-01-15T11:00:01Z", "endpoint": "/api/reports/generate", "status": 202, "response_time_ms": 350}',
    ]
    gen_texts = [
        "Meeting notes: Discuss Q1 goals and team building activities. Action items: finalize budget by Friday.",
        "Project roadmap 2024: Phase 1 infrastructure upgrade, Phase 2 feature rollout, Phase 3 performance optimization.",
        "Quarterly all-hands agenda: Welcome new hires, team updates, Q&A session. Catering: sandwiches and salad.",
        "Team offsite planning: Location TBD, budget $5,000, activities: hiking + brainstorming. Date: April 15-16.",
    ]

    documents: list[Document] = []
    for i, text in enumerate(hr_texts, 1):
        documents.append(Document(doc_id=f"doc_hr_{i}", text=text, metadata={"file_type": ".docx"}))
    for i, text in enumerate(fin_texts, 1):
        documents.append(Document(doc_id=f"doc_fin_{i}", text=text, metadata={"file_type": ".pdf"}))
    for i, text in enumerate(med_texts, 1):
        documents.append(Document(doc_id=f"doc_med_{i}", text=text, metadata={"file_type": ".pdf"}))
    for i, text in enumerate(api_texts, 1):
        documents.append(Document(doc_id=f"doc_api_{i}", text=text, metadata={"file_type": ".json"}))
    for i, text in enumerate(gen_texts, 1):
        documents.append(Document(doc_id=f"doc_gen_{i}", text=text, metadata={"file_type": ".txt"}))
    return documents


# ──────────────────────────────────────────────────────────────
# sample_config — simplified tier configuration
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_config() -> dict:
    """Return a dict with all tier configs needed for initialization."""
    return {
        "tier0": {
            "context_window": 100,
            "confidence_threshold": 0.8,
            "context_penalty_terms": ["test", "sample", "example"],
            "context_boost_terms": [],
        },
        "tier1": {
            "stage_a": {
                "structural_features": [
                    "file_type", "file_size_quantile", "page_count",
                    "paragraph_count", "table_count", "has_images",
                    "header_pattern", "json_schema_signature", "path_depth",
                ],
            },
            "stage_b": {
                "sem_split_threshold": 50,
                "homogeneity_threshold": 0.85,
                "variance_threshold": 0.25,
                "max_sample_for_large": 10000,
            },
            "incremental": {
                "outlier_radius_multiplier": 1.5,
                "outlier_trigger_ratio": 0.2,
                "new_bucket_trigger": 50,
            },
        },
        "embedding": {
            "model_name": "BAAI/bge-m3",
            "dim": 1024,
        },
        "tier2": {
            "ner_representative_limit": 3,
            "known_type_matching": {
                "structure_signature_weight": 0.5,
                "tfidf_overlap_weight": 0.3,
                "pii_distribution_weight": 0.2,
                "high_match_threshold": 0.8,
                "low_match_threshold": 0.5,
            },
            "propagation": {
                "sample_strategy": "inverse_cluster_size",
                "min_samples": 3,
                "inconsistency_threshold": 0.15,
            },
        },
        "tier3": {
            "high_sensitivity_types": ["SSN", "CREDIT_CARD", "MEDICAL", "IBAN"],
            "semantic_distance_sigma": 2.0,
            "outlier_sigma": 3.0,
            "ner_rule_contradiction": True,
            "llm_confidence_range": [0.5, 0.8],
        },
        "discovery": {
            "min_trigger_count": 100,
            "same_pattern_threshold": 5,
            "time_trigger_hours": 24,
            "min_coherence": 0.75,
            "min_distance_to_known": 0.3,
            "min_cluster_size": 3,
        },
    }


# ──────────────────────────────────────────────────────────────
# mock_llm_client — returns predefined responses per doc content
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client() -> Mock:
    """Return a Mock MistralClient that returns predefined responses
    based on document content keywords.
    """
    def _classify_side_effect(
        document_text: str,
        known_types: list[str],
        ner_results: list | None = None,
        pii_features: dict | None = None,
    ) -> dict:
        text_lower = document_text.lower()

        if "ssn" in text_lower or "employee" in text_lower:
            label = "HR Document"
        elif "revenue" in text_lower or "credit card" in text_lower or "invoice" in text_lower or "iban" in text_lower:
            label = "Financial Report"
        elif "patient" in text_lower or "diagnosis" in text_lower or "npi" in text_lower:
            label = "Medical Record"
        elif "timestamp" in text_lower and "endpoint" in text_lower:
            label = "API Log"
        elif "meeting" in text_lower or "agenda" in text_lower:
            label = "Meeting Notes"
        else:
            label = "General Document"

        return {
            "label": label,
            "is_new_type": False,
            "confidence": 0.82,
            "rationale": f"Classified as '{label}' based on document content keywords.",
            "suggested_rules": "",
        }

    def _verify_side_effect(
        document_text: str,
        current_label: str,
        cluster_context: dict | None = None,
    ) -> dict:
        if current_label == "unknown":
            return {
                "label": current_label,
                "confidence": 0.40,
                "reasoning_chain": "Cannot verify unknown label with confidence.",
                "needs_manual_review": True,
            }
        return {
            "label": current_label,
            "confidence": 0.85,
            "reasoning_chain": f"Verified: document content aligns with '{current_label}'.",
            "needs_manual_review": False,
        }

    llm = Mock()
    llm.classify.side_effect = _classify_side_effect
    llm.verify.side_effect = _verify_side_effect
    return llm
