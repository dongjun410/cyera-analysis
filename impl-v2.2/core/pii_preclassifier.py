"""
PII 预分类器（国际化版本）
- 使用 Microsoft Presidio 检测 50+ 国家的 PII 模式
- 支持 email / phone / credit card / SSN / passport / IBAN 等
- 高置信度匹配直接标记分类，不进入聚类 pipeline
- 回退到正则（当 Presidio 不可用时）
"""

import re
import logging
from typing import List, Tuple, Dict

from models.schemas import ProcessedDocument, PIIDetection, PIIType

logger = logging.getLogger(__name__)


# ── Presidio 实体类型 → 内部 PIIType 映射 ─────────────────────

PRESIDIO_TYPE_MAP = {
    "EMAIL_ADDRESS": PIIType.EMAIL,
    "PHONE_NUMBER": PIIType.PHONE,
    "CREDIT_CARD": PIIType.CREDIT_CARD,
    "IBAN_CODE": PIIType.IBAN,
    "US_SSN": PIIType.SSN,
    "US_PASSPORT": PIIType.PASSPORT,
    "UK_NHS": PIIType.NATIONAL_ID,
    "SG_NRIC_FIN": PIIType.NATIONAL_ID,
    "AU_TFN": PIIType.NATIONAL_ID,
    "IP_ADDRESS": PIIType.IP_ADDRESS,
    "PERSON": PIIType.PERSON_NAME,
    "LOCATION": PIIType.LOCATION,
    "DATE_TIME": PIIType.DATE,
    "URL": PIIType.URL,
}

# 文档级分类规则（国际化）
PRECLASSIFICATION_RULES = {
    "PII-heavy: Contact/Directory": {PIIType.PHONE: 5, PIIType.EMAIL: 5},
    "PII-heavy: Financial/Payment": {PIIType.CREDIT_CARD: 2},
    "PII-heavy: Identity Documents": {PIIType.SSN: 2},
    "PII-heavy: Banking Records": {PIIType.IBAN: 3},
}

# ── 回退正则（当 Presidio 不可用时） ──────────────────────────

FALLBACK_PATTERNS = {
    PIIType.EMAIL: re.compile(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    ),
    PIIType.PHONE: re.compile(
        r'(?<!\d)'
        r'(?:\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}'
        r'(?!\d)',
    ),
    PIIType.CREDIT_CARD: re.compile(
        r'(?<!\d)'
        r'(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))'  # Visa/MC/Amex/Discover
        r'[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{3,4}'
        r'(?!\d)',
    ),
    PIIType.IP_ADDRESS: re.compile(
        r'(?<!\d)'
        r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
        r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
        r'(?!\d)',
    ),
    PIIType.SSN: re.compile(
        r'(?<!\d)\d{3}[\s\-]\d{2}[\s\-]\d{4}(?!\d)',  # US SSN: XXX-XX-XXXX
    ),
}


class PIIPreclassifier:

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.engine = config.get("engine", "presidio")
        self.supported_entities = config.get("supported_entities", [])
        self.analyzer = None

    def _init_presidio(self):
        """Initialize Microsoft Presidio analyzer"""
        try:
            from presidio_analyzer import AnalyzerEngine
            self.analyzer = AnalyzerEngine()
            logger.info("Presidio analyzer initialized")
        except ImportError:
            logger.warning(
                "Presidio not installed, falling back to regex. "
                "Install: pip install presidio-analyzer presidio-anonymizer"
            )
            self.engine = "regex"

    def scan_document(self, doc: ProcessedDocument) -> ProcessedDocument:
        """Scan document for PII entities"""
        if not self.enabled:
            return doc

        text = doc.raw_content
        detections: List[PIIDetection] = []

        if self.engine == "presidio":
            detections = self._scan_presidio(text)
        else:
            detections = self._scan_regex(text)

        doc.pii_detections = detections
        doc.pii_types_found = list(set(d.pii_type.value for d in detections))

        # Try preclassification
        label = self._try_preclassify(detections)
        if label:
            doc.preclassified = True
            doc.preclassification_label = label
            doc.classification_source = "pii_presidio" if self.engine == "presidio" else "pii_regex"

        return doc

    def scan_batch(self, documents: List[ProcessedDocument]) -> Tuple[
        List[ProcessedDocument],
        List[ProcessedDocument],
    ]:
        """Batch scan, split into preclassified vs unmatched"""
        if self.engine == "presidio" and self.analyzer is None:
            self._init_presidio()

        preclassified = []
        unmatched = []

        for doc in documents:
            doc = self.scan_document(doc)
            if doc.preclassified:
                preclassified.append(doc)
            else:
                unmatched.append(doc)

        logger.info(
            f"PII pre-classification done: {len(preclassified)} labeled, "
            f"{len(unmatched)} to cluster"
        )
        return preclassified, unmatched

    def _scan_presidio(self, text: str) -> List[PIIDetection]:
        """Use Presidio for multi-language PII detection"""
        if self.analyzer is None:
            self._init_presidio()
        if self.analyzer is None:
            return self._scan_regex(text)

        entities = self.supported_entities or list(PRESIDIO_TYPE_MAP.keys())

        try:
            results = self.analyzer.analyze(
                text=text[:50000],  # Presidio performance cap
                language="en",     # Presidio auto-detects patterns regardless
                entities=entities,
                score_threshold=self.confidence_threshold,
            )
        except Exception as e:
            logger.warning(f"Presidio analysis failed: {e}")
            return self._scan_regex(text)

        detections = []
        for result in results:
            pii_type = PRESIDIO_TYPE_MAP.get(result.entity_type)
            if pii_type:
                raw_value = text[result.start:result.end]
                detections.append(PIIDetection(
                    pii_type=pii_type,
                    value=self._mask_value(raw_value, pii_type),
                    position=result.start,
                    confidence=result.score,
                ))
        return detections

    def _scan_regex(self, text: str) -> List[PIIDetection]:
        """Fallback regex-based PII detection"""
        detections = []
        for pii_type, pattern in FALLBACK_PATTERNS.items():
            for match in pattern.finditer(text):
                detections.append(PIIDetection(
                    pii_type=pii_type,
                    value=self._mask_value(match.group(), pii_type),
                    position=match.start(),
                    confidence=0.9,
                ))
        return detections

    def _try_preclassify(self, detections: List[PIIDetection]) -> str:
        """Classify document based on PII density"""
        type_counts: Dict[PIIType, int] = {}
        for d in detections:
            type_counts[d.pii_type] = type_counts.get(d.pii_type, 0) + 1

        for label, rules in PRECLASSIFICATION_RULES.items():
            if all(type_counts.get(t, 0) >= n for t, n in rules.items()):
                return label
        return ""

    @staticmethod
    def _mask_value(value: str, pii_type: PIIType) -> str:
        """Mask PII value for storage"""
        if len(value) <= 4:
            return "****"
        if pii_type == PIIType.EMAIL:
            parts = value.split('@')
            return parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else "****"
        if pii_type in (PIIType.CREDIT_CARD, PIIType.SSN, PIIType.IBAN):
            digits_only = re.sub(r'\D', '', value)
            return digits_only[:4] + "****" + digits_only[-4:] if len(digits_only) >= 8 else "****"
        if pii_type == PIIType.PHONE:
            return value[:3] + "****" + value[-4:] if len(value) > 7 else "****"
        return value[:4] + "****"
