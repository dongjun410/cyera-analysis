"""Tests for Tier 1 Stage A: Structural Hashing (TDD).

Test contracts (write these BEFORE implementation):
  1. Identical StructuralFeatures → same bucket_id
  2. Different structure (PDF vs JSON) → different bucket_ids
  3. Hash determinism — same doc hashed twice → identical hash string
  4. Empty document (no structural features at all) → no crash, assigned to some bucket
  5. cluster() output format — dict[str, list[str]] with correct types
  6. 100 varied docs → reasonable bucket count (not 1, not 100)
  7. feature_config filtering — only configured features used for hashing
"""

from __future__ import annotations

import hashlib

import pytest

from src.tier1.structural import StructuralClusterer
from src.types import Document, StructuralFeatures


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def clusterer() -> StructuralClusterer:
    """Default clusterer with all features enabled."""
    return StructuralClusterer()


@pytest.fixture
def pdf_features() -> StructuralFeatures:
    """Typical PDF document structure."""
    return StructuralFeatures(
        file_type=".pdf",
        file_size_quantile=3,       # 100KB-1MB
        page_count=12,
        paragraph_count=45,
        table_count=3,
        has_images=True,
        header_pattern="ACME CORP",
        json_schema_signature="",
        path_depth=2,
    )


@pytest.fixture
def json_features() -> StructuralFeatures:
    """Typical JSON document structure."""
    return StructuralFeatures(
        file_type=".json",
        file_size_quantile=2,       # 10KB-100KB
        page_count=0,
        paragraph_count=0,
        table_count=0,
        has_images=False,
        header_pattern="",
        json_schema_signature="type:object;keys:id,name,email",
        path_depth=3,
    )


# ──────────────────────────────────────────────────────────────
# Test 1: Identical docs → same bucket
# ──────────────────────────────────────────────────────────────

def test_identical_docs_same_bucket(
    clusterer: StructuralClusterer,
    pdf_features: StructuralFeatures,
) -> None:
    """Two documents with identical StructuralFeatures get the same bucket_id."""
    doc1 = Document(
        doc_id="doc-001",
        text="Annual financial report for Q4 2025...",
        structural_features=pdf_features,
    )
    doc2 = Document(
        doc_id="doc-002",
        text="Another financial document with different content...",
        structural_features=pdf_features,
    )

    bucket1 = clusterer.assign_bucket(doc1)
    bucket2 = clusterer.assign_bucket(doc2)

    assert bucket1 == bucket2, (
        f"Identical structure should produce same bucket, got {bucket1} vs {bucket2}"
    )
    assert isinstance(bucket1, str)
    assert len(bucket1) == 64  # SHA256 hex digest length


# ──────────────────────────────────────────────────────────────
# Test 2: Different structure → different bucket
# ──────────────────────────────────────────────────────────────

def test_different_structure_different_bucket(
    clusterer: StructuralClusterer,
    pdf_features: StructuralFeatures,
    json_features: StructuralFeatures,
) -> None:
    """PDF doc and JSON doc have fundamentally different structure → different buckets."""
    pdf_doc = Document(
        doc_id="pdf-001",
        text="Quarterly report...",
        structural_features=pdf_features,
    )
    json_doc = Document(
        doc_id="json-001",
        text='{"id": 1, "name": "Alice"}',
        structural_features=json_features,
    )

    pdf_bucket = clusterer.assign_bucket(pdf_doc)
    json_bucket = clusterer.assign_bucket(json_doc)

    assert pdf_bucket != json_bucket, (
        f"PDF and JSON docs should get different buckets, "
        f"got {pdf_bucket} and {json_bucket}"
    )


# ──────────────────────────────────────────────────────────────
# Test 3: Hash determinism
# ──────────────────────────────────────────────────────────────

def test_hash_determinism(
    clusterer: StructuralClusterer,
    pdf_features: StructuralFeatures,
) -> None:
    """Same document hashed twice produces identical hash string."""
    doc = Document(
        doc_id="doc-001",
        text="Some text here...",
        structural_features=pdf_features,
    )

    hash1 = clusterer.extract_features(doc)
    hash2 = clusterer.extract_features(doc)

    assert hash1 == hash2, (
        f"Same doc hashed twice should produce identical hash, "
        f"got {hash1} vs {hash2}"
    )
    assert isinstance(hash1, str)
    assert len(hash1) == 64


# ──────────────────────────────────────────────────────────────
# Test 4: Empty document handling
# ──────────────────────────────────────────────────────────────

def test_empty_document_handling(clusterer: StructuralClusterer) -> None:
    """Document with no structural_features set at all → no crash, assigned to some bucket."""
    # Doc with structural_features=None and empty metadata
    doc1 = Document(doc_id="empty-1", text="", structural_features=None)

    # Should not crash
    bucket1 = clusterer.assign_bucket(doc1)
    assert isinstance(bucket1, str)
    assert len(bucket1) == 64

    # Doc with structural_features=None but some metadata
    doc2 = Document(
        doc_id="empty-2",
        text="",
        structural_features=None,
        metadata={"file_type": ".xlsx", "has_images": True},
    )

    bucket2 = clusterer.assign_bucket(doc2)
    assert isinstance(bucket2, str)
    assert len(bucket2) == 64

    # Both should produce valid (non-crashing) bucket assignments
    # Two truly empty docs should get the same bucket
    doc3 = Document(doc_id="empty-3", text="", structural_features=None)
    bucket3 = clusterer.assign_bucket(doc3)
    assert bucket1 == bucket3, (
        "Two docs with no structural features should map to same bucket"
    )


# ──────────────────────────────────────────────────────────────
# Test 5: cluster() output format
# ──────────────────────────────────────────────────────────────

def test_cluster_output_format(
    clusterer: StructuralClusterer,
    pdf_features: StructuralFeatures,
    json_features: StructuralFeatures,
) -> None:
    """cluster() returns dict[str, list[str]] with correct types and contents."""
    docs = [
        Document(
            doc_id="pdf-1",
            text="Annual report...",
            structural_features=pdf_features,
        ),
        Document(
            doc_id="pdf-2",
            text="Q4 summary...",
            structural_features=pdf_features,
        ),
        Document(
            doc_id="json-1",
            text='{"users": []}',
            structural_features=json_features,
        ),
    ]

    result = clusterer.cluster(docs)

    # Correct return type
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, list) for v in result.values())
    assert all(isinstance(doc_id, str) for v in result.values() for doc_id in v)

    # All doc_ids present
    all_doc_ids = set()
    for ids in result.values():
        all_doc_ids.update(ids)
    assert all_doc_ids == {"pdf-1", "pdf-2", "json-1"}

    # pdf-1 and pdf-2 should be in the same bucket
    pdf_buckets = {k for k, v in result.items() if "pdf-1" in v}
    assert len(pdf_buckets) == 1
    assert "pdf-2" in result[pdf_buckets.pop()]


# ──────────────────────────────────────────────────────────────
# Test 6: Bucket count range
# ──────────────────────────────────────────────────────────────

def test_bucket_count_range() -> None:
    """100 documents spread across ~12 structural profiles → bucket count > 1 and < 100.

    We define a small set of structural profiles and spread 100 docs across them.
    This guarantees that structurally-similar docs land in the same bucket and the
    bucket count is far below the document count.
    """
    clusterer = StructuralClusterer()

    # Define a small set of structural profiles
    profiles: list[StructuralFeatures] = [
        StructuralFeatures(       # Profile 0: PDF report with images
            file_type=".pdf", file_size_quantile=3, page_count=20,
            paragraph_count=80, table_count=4, has_images=True,
            header_pattern="REPORT", json_schema_signature="", path_depth=2,
        ),
        StructuralFeatures(       # Profile 1: PDF report without images
            file_type=".pdf", file_size_quantile=3, page_count=15,
            paragraph_count=60, table_count=2, has_images=False,
            header_pattern="REPORT", json_schema_signature="", path_depth=2,
        ),
        StructuralFeatures(       # Profile 2: Small PDF
            file_type=".pdf", file_size_quantile=1, page_count=2,
            paragraph_count=8, table_count=0, has_images=False,
            header_pattern="", json_schema_signature="", path_depth=1,
        ),
        StructuralFeatures(       # Profile 3: JSON API response
            file_type=".json", file_size_quantile=2, page_count=0,
            paragraph_count=0, table_count=0, has_images=False,
            header_pattern="", json_schema_signature="type:object;keys:id,name", path_depth=1,
        ),
        StructuralFeatures(       # Profile 4: JSON config file
            file_type=".json", file_size_quantile=1, page_count=0,
            paragraph_count=0, table_count=0, has_images=False,
            header_pattern="", json_schema_signature="type:object;keys:settings,env", path_depth=2,
        ),
        StructuralFeatures(       # Profile 5: DOCX memorandum
            file_type=".docx", file_size_quantile=2, page_count=5,
            paragraph_count=25, table_count=1, has_images=False,
            header_pattern="MEMO", json_schema_signature="", path_depth=1,
        ),
        StructuralFeatures(       # Profile 6: DOCX with images
            file_type=".docx", file_size_quantile=3, page_count=10,
            paragraph_count=40, table_count=2, has_images=True,
            header_pattern="CONFIDENTIAL", json_schema_signature="", path_depth=2,
        ),
        StructuralFeatures(       # Profile 7: Large XLSX
            file_type=".xlsx", file_size_quantile=4, page_count=0,
            paragraph_count=0, table_count=8, has_images=False,
            header_pattern="", json_schema_signature="", path_depth=1,
        ),
        StructuralFeatures(       # Profile 8: CSV export
            file_type=".csv", file_size_quantile=3, page_count=0,
            paragraph_count=0, table_count=1, has_images=False,
            header_pattern="", json_schema_signature="", path_depth=1,
        ),
        StructuralFeatures(       # Profile 9: TXT log
            file_type=".txt", file_size_quantile=4, page_count=0,
            paragraph_count=0, table_count=0, has_images=False,
            header_pattern="", json_schema_signature="", path_depth=0,
        ),
        StructuralFeatures(       # Profile 10: XML data
            file_type=".xml", file_size_quantile=2, page_count=0,
            paragraph_count=0, table_count=0, has_images=False,
            header_pattern="", json_schema_signature="type:element;root:dataset", path_depth=1,
        ),
        StructuralFeatures(       # Profile 11: Rich DOCX report
            file_type=".docx", file_size_quantile=5, page_count=50,
            paragraph_count=200, table_count=12, has_images=True,
            header_pattern="ANNUAL_REPORT", json_schema_signature="", path_depth=3,
        ),
    ]

    # Spread 100 docs across profiles (round-robin)
    docs: list[Document] = []
    for i in range(100):
        profile = profiles[i % len(profiles)]
        docs.append(Document(
            doc_id=f"doc-{i:03d}",
            text=f"Document content {i}...",
            structural_features=profile,
        ))

    result = clusterer.cluster(docs)

    bucket_count = len(result)

    # Should have multiple buckets (not just 1)
    assert bucket_count > 1, (
        f"Expected more than 1 bucket for varied docs, got {bucket_count}"
    )

    # Should have fewer buckets than documents (structural coarsening)
    assert bucket_count < 100, (
        f"Expected fewer than 100 buckets, got {bucket_count}"
    )

    # All doc_ids should be accounted for
    all_ids = {doc_id for ids in result.values() for doc_id in ids}
    assert len(all_ids) == 100

    # Each doc_id appears exactly once
    all_list = [doc_id for ids in result.values() for doc_id in ids]
    assert len(all_list) == len(set(all_list)), "Duplicate doc_ids found"


# ──────────────────────────────────────────────────────────────
# Test 7: Feature config filtering
# ──────────────────────────────────────────────────────────────

def test_feature_config_filtering(pdf_features: StructuralFeatures) -> None:
    """Only configured features are used for hashing; others are ignored.

    Two docs that differ ONLY in non-configured features should get the
    same bucket when those features are excluded from the config.
    """
    # Clusterer that only uses file_type and page_count
    limited = StructuralClusterer(feature_config=["file_type", "page_count"])

    sf_a = StructuralFeatures(
        file_type=".pdf",
        file_size_quantile=1,
        page_count=12,
        paragraph_count=45,        # NOT in config
        table_count=3,             # NOT in config
        has_images=True,           # NOT in config
        header_pattern="ACME",     # NOT in config
        json_schema_signature="",   # NOT in config
        path_depth=2,              # NOT in config
    )
    sf_b = StructuralFeatures(
        file_type=".pdf",
        file_size_quantile=5,      # DIFFERENT but NOT in config
        page_count=12,
        paragraph_count=99,        # DIFFERENT but NOT in config
        table_count=0,             # DIFFERENT but NOT in config
        has_images=False,          # DIFFERENT but NOT in config
        header_pattern="OTHER",    # DIFFERENT but NOT in config
        json_schema_signature="x",  # DIFFERENT but NOT in config
        path_depth=9,              # DIFFERENT but NOT in config
    )

    doc_a = Document(doc_id="a", text="...", structural_features=sf_a)
    doc_b = Document(doc_id="b", text="...", structural_features=sf_b)

    bucket_a = limited.assign_bucket(doc_a)
    bucket_b = limited.assign_bucket(doc_b)

    assert bucket_a == bucket_b, (
        f"Only file_type and page_count should matter for hashing, "
        f"got {bucket_a} vs {bucket_b}"
    )

    # Sanity check: if we change a configured feature, buckets differ
    sf_c = StructuralFeatures(
        file_type=".pdf",
        file_size_quantile=1,
        page_count=99,  # DIFFERENT page_count — in config
        paragraph_count=0,
        table_count=0,
        has_images=False,
        header_pattern="",
        json_schema_signature="",
        path_depth=0,
    )
    doc_c = Document(doc_id="c", text="...", structural_features=sf_c)
    bucket_c = limited.assign_bucket(doc_c)

    assert bucket_a != bucket_c, (
        "Different page_count (in config) should yield different buckets"
    )


# ──────────────────────────────────────────────────────────────
# Additional test: metadata fallback extraction
# ──────────────────────────────────────────────────────────────

def test_metadata_fallback_extraction(clusterer: StructuralClusterer) -> None:
    """When structural_features is None, features are extracted from metadata dict."""
    doc = Document(
        doc_id="md-doc",
        text="document body...",
        structural_features=None,
        metadata={
            "file_type": ".docx",
            "file_size_quantile": 4,
            "page_count": 8,
            "paragraph_count": 32,
            "table_count": 1,
            "has_images": True,
            "header_pattern": "CONFIDENTIAL",
            "json_schema_signature": "",
            "path_depth": 1,
        },
    )

    bucket = clusterer.assign_bucket(doc)
    assert isinstance(bucket, str)
    assert len(bucket) == 64

    # Same metadata → same bucket
    doc2 = Document(
        doc_id="md-doc-2",
        text="different body...",
        structural_features=None,
        metadata={
            "file_type": ".docx",
            "file_size_quantile": 4,
            "page_count": 8,
            "paragraph_count": 32,
            "table_count": 1,
            "has_images": True,
            "header_pattern": "CONFIDENTIAL",
            "json_schema_signature": "",
            "path_depth": 1,
        },
    )
    bucket2 = clusterer.assign_bucket(doc2)
    assert bucket == bucket2, (
        "Same metadata should produce same bucket via fallback"
    )


# ──────────────────────────────────────────────────────────────
# Additional test: partial metadata handling
# ──────────────────────────────────────────────────────────────

def test_partial_metadata_graceful(clusterer: StructuralClusterer) -> None:
    """Missing keys in metadata don't cause crashes — unavailable features are skipped."""
    doc = Document(
        doc_id="partial",
        text="some text",
        structural_features=None,
        metadata={
            "file_type": ".txt",
            # Missing: file_size_quantile, page_count, etc.
        },
    )
    # Should not crash
    bucket = clusterer.assign_bucket(doc)
    assert isinstance(bucket, str)
    assert len(bucket) == 64


# ──────────────────────────────────────────────────────────────
# Additional test: extract_features returns hash, assign_bucket is alias
# ──────────────────────────────────────────────────────────────

def test_extract_features_is_stable_hash(
    clusterer: StructuralClusterer,
    pdf_features: StructuralFeatures,
) -> None:
    """extract_features returns a valid SHA256 hex string."""
    doc = Document(
        doc_id="doc-1",
        text="test",
        structural_features=pdf_features,
    )
    result = clusterer.extract_features(doc)

    assert isinstance(result, str)
    assert len(result) == 64

    # Verify it's valid hex
    int(result, 16)

    # Verify it matches the SHA256 of the canonical feature string
    expected = hashlib.sha256(
        clusterer._build_canonical_string(doc).encode("utf-8")
    ).hexdigest()
    assert result == expected
