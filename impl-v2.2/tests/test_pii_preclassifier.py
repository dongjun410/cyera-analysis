import pytest

from models.schemas import ProcessedDocument, PIIType
from core.pii_preclassifier import PIIPreclassifier


def make_doc(raw_content, doc_id="test-doc-1"):
    """Create a minimal ProcessedDocument for testing."""
    return ProcessedDocument(
        id=doc_id,
        original_path="/test/doc.txt",
        title="Test Document",
        raw_content=raw_content,
    )


DEFAULT_CONFIG = {
    "enabled": True,
    "engine": "presidio",
    "confidence_threshold": 0.8,
    "supported_entities": [],
}


class TestPIIPreclassifier:

    def test_scan_document_detects_email(self):
        """Doc with an email address detects 1 EMAIL."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc("Please contact us at contact@example.com for support.")
        result = classifier.scan_document(doc)

        emails = [d for d in result.pii_detections if d.pii_type == PIIType.EMAIL]
        assert len(emails) == 1
        assert "contact@example.com" not in emails[0].value  # value is masked

    def test_scan_document_detects_phone(self):
        """Doc with a phone number detects 1 PHONE."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc("Call me at +1-555-123-4567 for details.")
        result = classifier.scan_document(doc)

        phones = [d for d in result.pii_detections if d.pii_type == PIIType.PHONE]
        assert len(phones) == 1

    def test_scan_document_detects_credit_card(self):
        """Doc with a credit card number detects 1 CREDIT_CARD."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc("Payment by card 4111-1111-1111-1111 is accepted.")
        result = classifier.scan_document(doc)

        cards = [d for d in result.pii_detections if d.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1

    def test_scan_document_detects_ssn(self):
        """Doc with an SSN detects 1 SSN."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc("The SSN is 123-45-6789 for this record.")
        result = classifier.scan_document(doc)

        ssns = [d for d in result.pii_detections if d.pii_type == PIIType.SSN]
        assert len(ssns) == 1

    def test_scan_document_detects_ip(self):
        """Doc with an IP address detects 1 IP_ADDRESS."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc("Connected from 192.168.1.1 at midnight.")
        result = classifier.scan_document(doc)

        ips = [d for d in result.pii_detections if d.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 1

    def test_scan_document_no_pii(self):
        """Doc with no PII detects nothing."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc("Hello world, no PII here. Just a regular sentence.")
        result = classifier.scan_document(doc)

        assert len(result.pii_detections) == 0

    def test_scan_document_disabled(self):
        """When disabled, the doc is returned unchanged."""
        config = {"enabled": False}
        classifier = PIIPreclassifier(config)
        doc = make_doc("contact@example.com and +1-555-123-4567")
        original_detections = len(doc.pii_detections)
        result = classifier.scan_document(doc)

        # Doc should be unchanged (no new detections)
        assert len(result.pii_detections) == original_detections

    def test_scan_batch_splits(self):
        """One doc with PII (enough to preclassify), one without.

        The doc with 5+ emails AND 5+ phones triggers the
        "PII-heavy: Contact/Directory" preclassification rule."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)

        # Doc with enough PII to trigger preclassification
        # Phone numbers need 3 digit groups (area + exchange + subscriber)
        pii_lines = []
        for i in range(5):
            pii_lines.append(f"Contact person{i}@example.com")
            pii_lines.append(f"Phone: +1-555-{100+i:03d}-{2000+i:04d}")
        doc_with_pii = make_doc("\n".join(pii_lines), doc_id="doc-pii")

        # Doc with no PII
        doc_no_pii = make_doc("This is a regular document with no PII.", doc_id="doc-clean")

        preclassified, unmatched = classifier.scan_batch([doc_with_pii, doc_no_pii])

        assert len(preclassified) == 1
        assert len(unmatched) == 1
        assert preclassified[0].id == "doc-pii"
        assert preclassified[0].preclassified is True
        assert unmatched[0].id == "doc-clean"
        assert unmatched[0].preclassified is False

    def test_preclassification_contact(self):
        """Doc with 5+ emails AND 5+ phones gets preclassified."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)

        lines = []
        for i in range(5):
            lines.append(f"Contact user{i}@example.com")
            lines.append(f"Phone: +1-555-{100+i:03d}-{2000+i:04d}")
        doc = make_doc("\n".join(lines))
        result = classifier.scan_document(doc)

        assert result.preclassified is True
        assert "Contact/Directory" in result.preclassification_label

    def test_mask_email(self):
        """Email masking: first 2 chars + ***@ + domain."""
        masked = PIIPreclassifier._mask_value("john@example.com", PIIType.EMAIL)
        assert masked == "jo***@example.com"

    def test_mask_credit_card(self):
        """Credit card masking: first4 + **** + last4."""
        masked = PIIPreclassifier._mask_value("4111111111111111", PIIType.CREDIT_CARD)
        assert masked == "4111****1111"

    def test_mask_short_value(self):
        """Values <= 4 chars become '****'."""
        masked = PIIPreclassifier._mask_value("abc", PIIType.EMAIL)
        assert masked == "****"

    def test_pii_types_found(self):
        """Doc with email+phone has both in pii_types_found list."""
        classifier = PIIPreclassifier(DEFAULT_CONFIG)
        doc = make_doc(
            "Email: test@example.com and phone: +1-555-123-4567"
        )
        result = classifier.scan_document(doc)

        assert "email" in result.pii_types_found
        assert "phone" in result.pii_types_found
