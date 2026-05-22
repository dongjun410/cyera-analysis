import pytest
from models.schemas import (
    ClassificationSource,
    PIIType,
    PIIDetection,
    ProcessedDocument,
    ClusterInfo,
    ClusteringResult,
)


class TestGenerateId:
    def test_generate_id_deterministic(self):
        """Same path gives same ID, different paths give different IDs."""
        id1 = ProcessedDocument.generate_id("/path/to/doc1.pdf")
        id2 = ProcessedDocument.generate_id("/path/to/doc1.pdf")
        id3 = ProcessedDocument.generate_id("/path/to/doc2.pdf")

        assert id1 == id2
        assert id1 != id3

    def test_generate_id_length(self):
        """ID is 20 characters."""
        doc_id = ProcessedDocument.generate_id("/some/file.txt")
        assert len(doc_id) == 20
        assert all(c in "0123456789abcdef" for c in doc_id)


class TestPIIDetection:
    def test_pii_detection_default_confidence(self):
        """Default confidence is 1.0."""
        detection = PIIDetection(
            pii_type=PIIType.EMAIL,
            value="test@example.com",
            position=0,
        )
        assert detection.confidence == 1.0

    def test_pii_detection_creation(self):
        """All fields set correctly."""
        detection = PIIDetection(
            pii_type=PIIType.PHONE,
            value="+1-555-123-4567",
            position=42,
            confidence=0.95,
        )
        assert detection.pii_type == PIIType.PHONE
        assert detection.value == "+1-555-123-4567"
        assert detection.position == 42
        assert detection.confidence == 0.95


class TestProcessedDocument:
    def test_processed_document_creation(self):
        """Minimal fields work, defaults are correct."""
        doc = ProcessedDocument(
            id="abc123",
            original_path="/data/report.pdf",
            title="Annual Report",
            raw_content="Some content here.",
        )
        assert doc.id == "abc123"
        assert doc.original_path == "/data/report.pdf"
        assert doc.title == "Annual Report"
        assert doc.raw_content == "Some content here."

    def test_processed_document_defaults(self):
        """pii_detections, content_blocks are empty lists, cluster_id=-1, preclassified=False."""
        doc = ProcessedDocument(
            id="def456",
            original_path="/data/doc.txt",
            title="Test",
            raw_content="Test content",
        )
        assert doc.abstracted_content == ""
        assert doc.content_blocks == []
        assert doc.pii_detections == []
        assert doc.pii_types_found == []
        assert doc.preclassified is False
        assert doc.preclassification_label == ""
        assert doc.cluster_id == -1
        assert doc.cluster_probability == 0.0
        assert doc.classification_source == ""
        assert doc.metadata == {}
        assert doc.file_size == 0
        assert doc.file_type == ""


class TestClusterInfo:
    def test_cluster_info_creation(self):
        """All fields set correctly, defaults for list fields."""
        cluster = ClusterInfo(
            cluster_id=3,
            size=150,
            keywords=["financial", "quarterly", "report"],
            llm_label="Financial Reports",
            llm_description="Quarterly and annual financial statements",
        )
        assert cluster.cluster_id == 3
        assert cluster.size == 150
        assert cluster.keywords == ["financial", "quarterly", "report"]
        assert cluster.llm_label == "Financial Reports"
        assert cluster.llm_description == "Quarterly and annual financial statements"
        assert cluster.representative_doc_ids == []
        assert cluster.document_ids == []
        assert cluster.coherence == 0.0
        assert cluster.centroid is None


class TestClusteringResult:
    def test_clustering_result_creation(self):
        """All fields, defaults for clusters/doc_cluster_map."""
        result = ClusteringResult(
            total_documents=1000,
            preclassified_documents=50,
            clustered_documents=950,
            num_clusters=12,
            num_outliers=15,
            silhouette_score=0.72,
            davies_bouldin_index=0.45,
            calinski_harabasz_index=120.5,
        )
        assert result.total_documents == 1000
        assert result.preclassified_documents == 50
        assert result.clustered_documents == 950
        assert result.num_clusters == 12
        assert result.num_outliers == 15
        assert result.silhouette_score == 0.72
        assert result.davies_bouldin_index == 0.45
        assert result.calinski_harabasz_index == 120.5
        assert result.clusters == []
        assert result.doc_cluster_map == {}


class TestEnums:
    def test_pii_type_enum_values(self):
        """Verify key enum values."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.CREDIT_CARD.value == "credit_card"
        assert PIIType.SSN.value == "ssn"
        assert PIIType.IP_ADDRESS.value == "ip_address"
        assert PIIType.URL.value == "url"
        assert PIIType.DATE.value == "date"

    def test_classification_source_enum_values(self):
        """Verify key enum values."""
        assert ClassificationSource.PII_REGEX.value == "pii_regex"
        assert ClassificationSource.CLUSTERING.value == "clustering"
        assert ClassificationSource.PROPAGATION.value == "propagation"
        assert ClassificationSource.LLM.value == "llm"
        assert ClassificationSource.MANUAL.value == "manual"
