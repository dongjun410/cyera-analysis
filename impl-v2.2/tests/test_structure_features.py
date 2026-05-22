import pytest
import numpy as np

from models.schemas import ProcessedDocument
from core.structure_feature_extractor import StructureFeatureExtractor


def make_doc(raw_content, **kwargs):
    """Create a minimal ProcessedDocument for testing."""
    defaults = {
        "id": "test-doc-1",
        "original_path": "/test/doc.txt",
        "title": "Test Document",
        "raw_content": raw_content,
    }
    defaults.update(kwargs)
    return ProcessedDocument(**defaults)


class TestStructureFeatureExtractor:

    def test_feature_dim(self):
        """Feature dimension should be 19."""
        extractor = StructureFeatureExtractor({})
        assert extractor.feature_dim() == 19

    def test_extract_features_empty_doc(self):
        """Doc with empty raw_content returns vector of length 19 with all PII counts zero."""
        extractor = StructureFeatureExtractor({"normalize": False})
        doc = make_doc("")
        vec = extractor.extract_features(doc)
        assert len(vec) == 19
        assert vec.dtype == np.float32
        # All 10 PII pattern counts should be zero
        assert np.allclose(vec[:10], np.zeros(10))

    def test_extract_features_with_email(self):
        """Doc containing an email has email_count > 0."""
        extractor = StructureFeatureExtractor({"normalize": False})
        doc = make_doc("contact john@example.com for info")
        vec = extractor.extract_features(doc)
        # email_count is at index 0
        assert vec[0] > 0

    def test_extract_features_with_phone(self):
        """Doc with a phone number has phone_count > 0."""
        extractor = StructureFeatureExtractor({"normalize": False})
        doc = make_doc("Call me at +1-555-123-4567 for details")
        vec = extractor.extract_features(doc)
        # phone_count is at index 1
        assert vec[1] > 0

    def test_extract_features_with_url(self):
        """Doc with a URL has url_count > 0."""
        extractor = StructureFeatureExtractor({"normalize": False})
        doc = make_doc("Visit https://example.com/page for more")
        vec = extractor.extract_features(doc)
        # url_count is at index 5
        assert vec[5] > 0

    def test_extract_features_with_date(self):
        """Doc with a date string has date_count > 0."""
        extractor = StructureFeatureExtractor({"normalize": False})
        doc = make_doc("The report is dated 2024-01-15 and approved")
        vec = extractor.extract_features(doc)
        # date_count is at index 7
        assert vec[7] > 0

    def test_extract_features_disabled(self):
        """When disabled, returns zeros regardless of content."""
        extractor = StructureFeatureExtractor({"enabled": False, "normalize": False})
        doc = make_doc("contact john@example.com for info")
        vec = extractor.extract_features(doc)
        assert np.allclose(vec, np.zeros(19))

    def test_extract_batch(self):
        """Batch of 3 docs returns shape (3, 19)."""
        extractor = StructureFeatureExtractor({"normalize": False})
        docs = [
            make_doc("Doc one content"),
            make_doc("Doc two is different"),
            make_doc("Doc three here"),
        ]
        batch = extractor.extract_batch(docs)
        assert batch.shape == (3, 19)
        assert batch.dtype == np.float32

    def test_normalized_output(self):
        """Normalized feature vector has L2 norm ~1.0 when features are non-zero."""
        extractor = StructureFeatureExtractor({"normalize": True})
        doc = make_doc(
            "Contact john@example.com or call +1-555-123-4567. "
            "Visit https://example.com/page. Dated 2024-01-15."
        )
        vec = extractor.extract_features(doc)
        norm = np.linalg.norm(vec)
        # Normalized vector should have L2 norm very close to 1.0
        assert abs(norm - 1.0) < 0.01

    def test_structure_features_paragraph_count(self):
        """Doc with 3 paragraphs separated by double newlines."""
        extractor = StructureFeatureExtractor({"normalize": False})
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = make_doc(text)
        vec = extractor.extract_features(doc)
        # paragraph_count is at index 10
        assert vec[10] >= 3
