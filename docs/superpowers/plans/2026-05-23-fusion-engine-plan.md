# 6-Engine Fusion Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `impl-datadna/` from serial tiered architecture (Tier 0→1→2→3) to 6-engine parallel fusion architecture per `docs/superpowers/specs/2026-05-23-optimal-architecture.md`.

**Architecture:** 6 independent engines (E1 regex, E2 template hash, E3 ML/SetFit, E4 semantic kNN, E5 structural signature, E6 LLM) run in parallel → weighted voting fusion → composite confidence + audit log. E1-E5 consensus ≥ 0.85 skips LLM for efficiency; any engine can fail without crashing the system.

**Tech Stack:** Python 3.11+, sentence-transformers (BGE-M3), SetFit, Mistral-7B via Ollama, numpy, scikit-learn

---

## File Structure Map

| File | Responsibility |
|------|---------------|
| `src/types.py` | `EngineOutput`, `FusionResult`, `AuditRecord` dataclasses + existing `Document`, `KnownType` |
| `src/engines/__init__.py` | Re-exports all 6 engines |
| `src/engines/base.py` | `BaseEngine` ABC with `analyze()`, `weight`, `is_available` |
| `src/engines/e1_regex.py` | E1: regex rule matching with PII boosting |
| `src/engines/e2_template.py` | E2: PII detection → entity replacement → SHA256 → template match |
| `src/engines/e3_ml.py` | E3: SetFit model wrapper (delegates to distillation/trainer.py) |
| `src/engines/e4_knn.py` | E4: BGE-M3 embedding → centroid cosine similarity |
| `src/engines/e5_structural.py` | E5: structural feature extraction → SHA256 hash → match |
| `src/engines/e6_llm.py` | E6: Mistral-7B classification via llm/client.py |
| `src/fusion/__init__.py` | Re-exports `FusionVoter` |
| `src/fusion/voter.py` | Weighted voting, preliminary consensus check, LLM gating |
| `src/knowledge/__init__.py` | Re-exports knowledge components |
| `src/knowledge/rules.py` | 50+ document type regex rules (extends tier0/patterns.py PII patterns) |
| `src/knowledge/templates.py` | Pre-computed template hashes for E2 |
| `src/knowledge/type_library.py` | `TypeLibrary` — known type registry with centroids, keywords, PII distributions |
| `src/monitoring/__init__.py` | Re-exports audit + metrics |
| `src/monitoring/audit.py` | Per-decision JSON audit log writer |
| `src/monitoring/metrics.py` | 7 monitoring metrics + alert threshold checks |
| `src/embeddings/bge_m3.py` | KEEP unchanged |
| `src/llm/client.py` | KEEP unchanged |
| `src/distillation/trainer.py` | KEEP unchanged |
| `src/discovery/loop.py` | KEEP (adapt imports later) |
| `main.py` | REWRITE: parallel engine dispatch → fusion → output |
| `config.yaml` | REWRITE: engine weights, consensus threshold, LLM config |
| `tests/test_engines.py` | Unit tests per engine |
| `tests/test_fusion.py` | Fusion voter tests |
| `tests/test_integration.py` | REWRITE: end-to-end fusion pipeline tests |
| `tests/conftest.py` | REWRITE: fixtures for new architecture |

**REMOVE:**
- `src/tier0/` (entire directory)
- `src/tier1/` (entire directory)
- `src/tier2/` (entire directory)
- `src/tier3/` (entire directory)
- `src/ner/` (entire directory)
- `tests/test_tier0.py`
- `tests/test_tier1_structural.py`
- `tests/test_tier1_semantic.py`
- `tests/test_tier2_matching.py`
- `tests/test_tier2_classifier.py`
- `tests/test_tier3.py`
- `tests/test_ner.py`
- `tests/test_discovery.py` (keep file, rewrite content)
- `tests/test_embeddings.py` (keep)
- `tests/test_llm_client.py` (keep)

---

### Task 1: Revise `src/types.py` — Add Engine Types

**Files:**
- Modify: `impl-datadna/src/types.py`

- [ ] **Step 1: Add EngineOutput, FusionResult, AuditRecord dataclasses**

Add to end of `impl-datadna/src/types.py` (keep all existing dataclasses):

```python
@dataclass
class EngineOutput:
    """Output from a single classification engine.

    Attributes:
        engine_id: "E1_regex", "E2_template", etc.
        label: Predicted document type, or None if engine had no output.
        confidence: Engine's self-assessed confidence in [0, 1].
        status: "matched" | "no_match" | "unavailable" | "skipped".
        metadata: Engine-specific trace (rule name, hash, distance, etc.).
    """

    engine_id: str
    label: str | None = None
    confidence: float = 0.0
    status: str = "unavailable"
    metadata: dict = field(default_factory=dict)


@dataclass
class FusionResult:
    """Output of the fusion voter for a single document.

    Attributes:
        doc_id: Document identifier.
        final_label: The winning label after weighted voting.
        composite_confidence: Normalized score in [0, 1].
        method: "fusion_fast" (no LLM) or "fusion_full" (with LLM).
        degraded: True if any engine was unavailable.
        manual_review: True if confidence < 0.4 threshold.
        engine_outputs: Per-engine outputs for audit trail.
        label_scores: Score per label from fusion calculation.
    """

    doc_id: str
    final_label: str
    composite_confidence: float
    method: str = "fusion_fast"
    degraded: bool = False
    manual_review: bool = False
    engine_outputs: dict[str, EngineOutput] = field(default_factory=dict)
    label_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class AuditRecord:
    """Full audit log entry for a single document classification.

    JSON-serializable. Records every engine's output and the fusion decision.
    Per spec section 8.
    """

    doc_id: str
    timestamp: str
    final_label: str
    composite_confidence: float
    method: str
    degraded: bool
    manual_review: bool
    engines: dict = field(default_factory=dict)
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/types.py
git commit -m "feat: add EngineOutput, FusionResult, AuditRecord dataclasses"
```

---

### Task 2: Create `src/engines/base.py` — BaseEngine ABC

**Files:**
- Create: `impl-datadna/src/engines/__init__.py`
- Create: `impl-datadna/src/engines/base.py`

- [ ] **Step 1: Create `__init__.py`**

```python
"""Six parallel classification engines with uniform interface."""
```

- [ ] **Step 2: Write `base.py`**

```python
"""Base class for all classification engines.

Every engine has the same minimal contract:
  - analyze(doc) → EngineOutput
  - weight: float (pre-set based on validation accuracy)
  - is_available: bool (runtime health check)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Document, EngineOutput


class BaseEngine(ABC):
    """Abstract base for all 6 classification engines.

    Subclasses must implement analyze(), weight, and is_available.
    The fusion voter calls analyze() on every engine and uses weight
    in the weighted voting calculation.
    """

    @abstractmethod
    def analyze(self, doc: Document) -> EngineOutput:
        """Run this engine on a document and return its output.

        Args:
            doc: Document with text and metadata.

        Returns:
            EngineOutput with engine_id, label, confidence, status, metadata.
            If the engine cannot produce output, status must be "unavailable"
            or "no_match" (not an exception).
        """
        ...

    @property
    @abstractmethod
    def weight(self) -> float:
        """Pre-set engine weight for fusion voting.

        Weights per spec section 2.3:
          E1 regex: 1.0, E2 template: 1.0, E3 ML: 1.5,
          E4 kNN: 1.0, E5 structural: 0.8, E6 LLM: 2.0
        """
        ...

    @property
    def is_available(self) -> bool:
        """Whether this engine is ready to produce output.

        Default True. Override if the engine has runtime dependencies
        that may be unavailable (e.g. model not loaded, service down).
        """
        return True
```

- [ ] **Step 3: Commit**

```bash
git add impl-datadna/src/engines/__init__.py impl-datadna/src/engines/base.py
git commit -m "feat: add BaseEngine ABC with uniform engine interface"
```

---

### Task 3: Create `src/knowledge/rules.py` — 50+ Document Type Regex Rules

**Files:**
- Create: `impl-datadna/src/knowledge/__init__.py`
- Create: `impl-datadna/src/knowledge/rules.py`

- [ ] **Step 1: Create `__init__.py`**

```python
"""Pre-built knowledge: rules, templates, and type library."""
```

- [ ] **Step 2: Write `rules.py` with 50+ document type inference rules**

```python
"""50+ document type inference rules for E1 regex engine.

Each rule has:
  - rule_id: unique identifier
  - pattern: regex matching document text keywords/patterns
  - associated_pii: PII types that boost confidence when present
  - label: target document type
  - base_confidence: starting confidence (before PII boost)

Rules are self-contained: they match both document text AND PII patterns
inline. E1 does not depend on a separate PII detection module.

Extends the 66 PII patterns from the original tier0/patterns.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DocTypeRule:
    """A single document type inference rule."""

    rule_id: str
    pattern: re.Pattern
    associated_pii: list[str] = field(default_factory=list)
    label: str = ""
    base_confidence: float = 0.7


# ── PII helper patterns (shared by rules for associated_pii matching) ──

PII_PATTERNS: dict[str, re.Pattern] = {
    "SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "PHONE": re.compile(r"\b\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b"),
    "MONEY": re.compile(r"\$\s?\d{1,3}(?:,?\d{3})*(?:\.\d{2})?\b"),
    "DATE_OF_BIRTH": re.compile(r"\b(?:DOB|Date\s*of\s*Birth)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    "NPI": re.compile(r"\bNPI[:\s]+\d{10}\b", re.IGNORECASE),
    "MEDICAL_RECORD": re.compile(r"\b(?:MRN|Medical\s*Record\s*Number)[:\s]+\d+\b", re.IGNORECASE),
    "EMPLOYER_ID": re.compile(r"\b(?:EIN|Employer\s*ID)[:\s]+\d{2}[-\s]?\d{7}\b", re.IGNORECASE),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "API_KEY": re.compile(r"\b(?:api[_-]?key|apikey|API_KEY)[:\s]*[\w-]{20,}\b", re.IGNORECASE),
    "PASSPORT": re.compile(r"\b(?:Passport|PASSPORT)[:\s#]*[A-Z0-9]{6,12}\b", re.IGNORECASE),
    "DRIVER_LICENSE": re.compile(r"\b(?:DL|Driver'?s?\s*License)[:\s#]*[A-Z0-9]{5,20}\b", re.IGNORECASE),
}


# ── 50+ Document Type Rules ──

def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


BUILTIN_RULES: list[DocTypeRule] = [
    # ── HR & Payroll (6 rules) ──
    DocTypeRule("HR_PAYROLL_001", _compile(r"\b(?:W-2|W2|payroll|pay\s*stub|compensation\s*statement)\b"),
                ["SSN", "MONEY", "EMPLOYER_ID"], "HR & Payroll", 0.70),
    DocTypeRule("HR_OFFER_001", _compile(r"\b(?:offer\s*letter|employment\s*offer|start\s*date.*salary)\b"),
                ["SSN", "MONEY", "DATE_OF_BIRTH"], "HR & Payroll", 0.65),
    DocTypeRule("HR_BENEFIT_001", _compile(r"\b(?:benefits?\s*enrollment|401\s*k|health\s*plan|dental|vision\s*plan)\b"),
                ["SSN", "DATE_OF_BIRTH"], "HR & Payroll", 0.65),
    DocTypeRule("HR_REVIEW_001", _compile(r"\b(?:performance\s*review|annual\s*evaluation|performance\s*assessment)\b"),
                ["EMPLOYER_ID"], "HR & Payroll", 0.60),
    DocTypeRule("HR_TERM_001", _compile(r"\b(?:termination\s*letter|separation\s*agreement|final\s*paycheck)\b"),
                ["SSN", "MONEY", "EMPLOYER_ID"], "HR & Payroll", 0.70),
    DocTypeRule("HR_TAX_001", _compile(r"\b(?:W-4|W4|I-9|I9|tax\s*withholding|withholding\s*certificate)\b"),
                ["SSN", "EMPLOYER_ID"], "HR & Payroll", 0.70),

    # ── Financial Reports (7 rules) ──
    DocTypeRule("FIN_REPORT_001", _compile(r"\b(?:quarterly\s*report|annual\s*report|10-K|10-Q|earnings\s*release)\b"),
                ["MONEY"], "Financial Report", 0.70),
    DocTypeRule("FIN_INVOICE_001", _compile(r"\b(?:invoice|INV[-\s]?\d+|bill\s*to|due\s*date.*amount)\b"),
                ["MONEY", "CREDIT_CARD"], "Financial Report", 0.70),
    DocTypeRule("FIN_BANK_001", _compile(r"\b(?:bank\s*statement|account\s*statement|transaction\s*history|ending\s*balance)\b"),
                ["MONEY", "IBAN"], "Financial Report", 0.70),
    DocTypeRule("FIN_TAX_001", _compile(r"\b(?:tax\s*return|Form\s*1040|Schedule\s*[A-Z]|tax\s*filing)\b"),
                ["SSN", "MONEY", "EMPLOYER_ID"], "Financial Report", 0.75),
    DocTypeRule("FIN_EXPENSE_001", _compile(r"\b(?:expense\s*report|reimbursement|travel\s*expense|out-of-pocket)\b"),
                ["MONEY", "CREDIT_CARD"], "Financial Report", 0.65),
    DocTypeRule("FIN_BUDGET_001", _compile(r"\b(?:budget\s*proposal|budget\s*plan|fiscal\s*year.*budget|budget\s*forecast)\b"),
                ["MONEY"], "Financial Report", 0.60),
    DocTypeRule("FIN_WIRE_001", _compile(r"\b(?:wire\s*transfer|SWIFT|routing\s*number|account\s*number.*transfer)\b"),
                ["MONEY", "IBAN"], "Financial Report", 0.70),

    # ── Medical Records (6 rules) ──
    DocTypeRule("MED_RECORD_001", _compile(r"\b(?:diagnosis|patient.*history|prescribed|medical\s*record|HIPAA)\b"),
                ["MEDICAL_RECORD", "NPI", "DATE_OF_BIRTH"], "Medical Record", 0.70),
    DocTypeRule("MED_LAB_001", _compile(r"\b(?:lab\s*results?|blood\s*test|urinalysis|CBC|metabolic\s*panel)\b"),
                ["MEDICAL_RECORD", "DATE_OF_BIRTH"], "Medical Record", 0.70),
    DocTypeRule("MED_SCRIPT_001", _compile(r"\b(?:prescription|Rx|dispense|refill|dosage.*mg)\b"),
                ["MEDICAL_RECORD", "NPI", "DATE_OF_BIRTH"], "Medical Record", 0.70),
    DocTypeRule("MED_INSURANCE_001", _compile(r"\b(?:insurance\s*claim|EOB|explanation\s*of\s*benefits|claim\s*#)\b"),
                ["SSN", "MEDICAL_RECORD", "NPI"], "Medical Record", 0.75),
    DocTypeRule("MED_REFERRAL_001", _compile(r"\b(?:referral|referred\s*to|consult|specialist.*referral)\b"),
                ["MEDICAL_RECORD", "NPI"], "Medical Record", 0.65),
    DocTypeRule("MED_CONSENT_001", _compile(r"\b(?:informed\s*consent|HIPAA\s*authorization|privacy\s*notice|patient\s*rights)\b"),
                ["MEDICAL_RECORD", "DATE_OF_BIRTH"], "Medical Record", 0.65),

    # ── Legal Documents (6 rules) ──
    DocTypeRule("LEGAL_CONTRACT_001", _compile(r"\b(?:contract|agreement.*between|parties.*agree|terms?\s*and\s*conditions?)\b"),
                ["MONEY"], "Legal Document", 0.60),
    DocTypeRule("LEGAL_NDA_001", _compile(r"\b(?:non[-\s]?disclosure|NDA|confidentiality\s*agreement|proprietary\s*information)\b"),
                [], "Legal Document", 0.75),
    DocTypeRule("LEGAL_COMPLIANCE_001", _compile(r"\b(?:compliance|regulatory|GDPR|CCPA|SOX|PCI[-\s]DSS|data\s*protection)\b"),
                ["SSN", "CREDIT_CARD"], "Legal Document", 0.65),
    DocTypeRule("LEGAL_LITIGATION_001", _compile(r"\b(?:lawsuit|litigation|cease\s*and\s*desist|settlement\s*offer|deposition)\b"),
                [], "Legal Document", 0.70),
    DocTypeRule("LEGAL_IP_001", _compile(r"\b(?:patent|trademark|copyright|intellectual\s*property|licensing\s*agreement)\b"),
                [], "Legal Document", 0.70),
    DocTypeRule("LEGAL_M_A_001", _compile(r"\b(?:merger|acquisition|due\s*diligence|letter\s*of\s*intent|term\s*sheet)\b"),
                ["MONEY"], "Legal Document", 0.65),

    # ── API Logs / Technical (5 rules) ──
    DocTypeRule("TECH_API_001", _compile(r"\b(?:timestamp.*endpoint|API\s*response|HTTP.*\d{3}|request.*response.*ms)\b"),
                ["IP_ADDRESS", "API_KEY"], "API Log", 0.75),
    DocTypeRule("TECH_LOG_001", _compile(r"\b(?:ERROR|WARN|INFO|DEBUG|stack\s*trace|exception.*at)\b"),
                ["IP_ADDRESS"], "API Log", 0.55),
    DocTypeRule("TECH_CONFIG_001", _compile(r"\b(?:server.*config|database.*config|connection.*string|environment.*variable)\b"),
                ["IP_ADDRESS", "API_KEY"], "Technical Document", 0.60),
    DocTypeRule("TECH_CODE_001", _compile(r"\b(?:function|class|import|def\s+\w+\s*\(|package\s+\w+|module\s+\w+)\b"),
                [], "Technical Document", 0.40),
    DocTypeRule("TECH_README_001", _compile(r"\b(?:README|installation|quick\s*start|getting\s*started|documentation)\b"),
                [], "Technical Document", 0.45),

    # ── Identity / Personal Documents (5 rules) ──
    DocTypeRule("ID_PASSPORT_001", _compile(r"\b(?:passport\s*(?:number|#|no)|nationality|issuing\s*country|date\s*of\s*expiry)\b"),
                ["PASSPORT", "DATE_OF_BIRTH"], "Identity Document", 0.75),
    DocTypeRule("ID_DRIVER_001", _compile(r"\b(?:driver'?s?\s*license|driving\s*license|DL\s*(?:number|#|no))\b"),
                ["DRIVER_LICENSE", "DATE_OF_BIRTH"], "Identity Document", 0.75),
    DocTypeRule("ID_BCERT_001", _compile(r"\b(?:birth\s*certificate|certificate\s*of\s*birth|place\s*of\s*birth)\b"),
                ["DATE_OF_BIRTH", "SSN"], "Identity Document", 0.75),
    DocTypeRule("ID_RESUME_001", _compile(r"\b(?:resume|CV|curriculum\s*vitae|work\s*experience|education.*degree)\b"),
                ["EMAIL", "PHONE"], "Identity Document", 0.60),
    DocTypeRule("ID_APPLICATION_001", _compile(r"\b(?:application\s*form|personal\s*information|applicant.*details)\b"),
                ["SSN", "EMAIL", "PHONE", "DATE_OF_BIRTH"], "Identity Document", 0.65),

    # ── Email / Communication (4 rules) ──
    DocTypeRule("COMM_EMAIL_001", _compile(r"\b(?:From:|To:|Subject:|CC:|BCC:|Sent:|forwarded\s*message)\b"),
                ["EMAIL"], "Email / Communication", 0.75),
    DocTypeRule("COMM_MEETING_001", _compile(r"\b(?:meeting\s*(?:notes|minutes|agenda)|attendees|action\s*items?)\b"),
                ["EMAIL"], "Email / Communication", 0.60),
    DocTypeRule("COMM_MEMO_001", _compile(r"\b(?:memorandum|memo|internal\s*communication|all[-\s]hands)\b"),
                [], "Email / Communication", 0.55),
    DocTypeRule("COMM_SLACK_001", _compile(r"\b(?:slack|channel|thread|message.*sent|reacted\s*to)\b"),
                [], "Email / Communication", 0.40),

    # ── Government / Regulatory (4 rules) ──
    DocTypeRule("GOV_FORM_001", _compile(r"\b(?:government.*form|federal.*form|OMB\s*No\.|Paperwork\s*Reduction)\b"),
                ["SSN", "EMPLOYER_ID"], "Government Form", 0.70),
    DocTypeRule("GOV_GRANT_001", _compile(r"\b(?:grant\s*proposal|grant\s*application|funding\s*request|NIH|NSF)\b"),
                ["MONEY"], "Government Form", 0.65),
    DocTypeRule("GOV_PERMIT_001", _compile(r"\b(?:permit\s*application|building\s*permit|zoning|environmental\s*assessment)\b"),
                [], "Government Form", 0.60),
    DocTypeRule("GOV_CENSUS_001", _compile(r"\b(?:census|survey.*response|demographic.*data|household.*survey)\b"),
                ["SSN", "DATE_OF_BIRTH"], "Government Form", 0.65),

    # ── Education (3 rules) ──
    DocTypeRule("EDU_TRANSCRIPT_001", _compile(r"\b(?:transcript|GPA|grade\s*point|semester|academic\s*record)\b"),
                ["DATE_OF_BIRTH"], "Education Record", 0.70),
    DocTypeRule("EDU_DIPLOMA_001", _compile(r"\b(?:diploma|degree\s*of|conferred|graduation.*date|honors)\b"),
                [], "Education Record", 0.65),
    DocTypeRule("EDU_ENROLLMENT_001", _compile(r"\b(?:enrollment|registration|student\s*ID|class\s*schedule|tuition)\b"),
                ["MONEY"], "Education Record", 0.65),

    # ── Real Estate / Property (3 rules) ──
    DocTypeRule("REAL_DEED_001", _compile(r"\b(?:deed|title\s*transfer|property\s*description|parcel\s*(?:number|#)|lot\s*#)\b"),
                ["MONEY"], "Real Estate Document", 0.70),
    DocTypeRule("REAL_MORTGAGE_001", _compile(r"\b(?:mortgage|loan\s*agreement|amortization|escrow|closing\s*statement)\b"),
                ["SSN", "MONEY"], "Real Estate Document", 0.75),
    DocTypeRule("REAL_LEASE_001", _compile(r"\b(?:lease\s*agreement|rental\s*agreement|tenant|landlord|security\s*deposit)\b"),
                ["MONEY", "SSN"], "Real Estate Document", 0.70),

    # ── Marketing / Sales (3 rules) ──
    DocTypeRule("MKTG_CAMPAIGN_001", _compile(r"\b(?:marketing\s*campaign|ad\s*budget|impressions|CTR|conversion\s*rate)\b"),
                ["MONEY"], "Marketing Document", 0.60),
    DocTypeRule("MKTG_ANALYTICS_001", _compile(r"\b(?:analytics\s*report|traffic\s*report|user\s*acquisition|churn\s*rate)\b"),
                [], "Marketing Document", 0.55),
    DocTypeRule("MKTG_PITCH_001", _compile(r"\b(?:sales\s*pitch|sales\s*deck|proposal|ROI|pricing\s*model)\b"),
                ["MONEY"], "Marketing Document", 0.55),

    # ── Scientific / Research (3 rules) ──
    DocTypeRule("SCI_PAPER_001", _compile(r"\b(?:abstract|methodology|results|discussion|citation|et\s*al\.|journal)\b"),
                [], "Scientific Paper", 0.55),
    DocTypeRule("SCI_CLINICAL_001", _compile(r"\b(?:clinical\s*trial|placebo|double[-\s]blind|informed\s*consent|IRB)\b"),
                ["MEDICAL_RECORD"], "Scientific Paper", 0.70),
    DocTypeRule("SCI_PATENT_001", _compile(r"\b(?:claims?\s*(?:1|what\s*is\s*claimed)|embodiment|prior\s*art|filing\s*date|inventor)\b"),
                [], "Scientific Paper", 0.65),
]

# Total: 55 rules covering ~40-60% of common enterprise document types.
```

- [ ] **Step 3: Commit**

```bash
git add impl-datadna/src/knowledge/
git commit -m "feat: add 55 document type regex rules for E1 engine"
```

---

### Task 4: Create `src/knowledge/templates.py` — Pre-computed Template Hashes

**Files:**
- Create: `impl-datadna/src/knowledge/templates.py`

- [ ] **Step 1: Write `templates.py`**

```python
"""Pre-computed template hashes for E2 template engine.

Each template is a SHA256 hash of a PII-replaced document sample.
When E2 processes a document, it replaces PII entities with type
placeholders, hashes the result, and checks against this library.

Templates are organized by document type for O(1) lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TemplateEntry:
    """A single template hash entry in the library.

    Attributes:
        template_hash: SHA256 hex digest of PII-replaced sample text.
        label: Document type this template represents.
        sample_count: How many samples contributed to this template.
        source: "builtin" (pre-computed) or "learned" (from customer data).
    """

    template_hash: str
    label: str
    sample_count: int = 1
    source: str = "builtin"


class TemplateLibrary:
    """In-memory template hash library with O(1) lookup.

    Usage:
        lib = TemplateLibrary()
        lib.add("abc123...", "HR & Payroll")
        result = lib.lookup("abc123...")  # → TemplateEntry or None
    """

    def __init__(self) -> None:
        self._templates: dict[str, TemplateEntry] = {}
        self._by_label: dict[str, list[str]] = {}

    def add(
        self, template_hash: str, label: str, source: str = "builtin"
    ) -> None:
        """Add or update a template hash entry.

        If the hash already exists with the same label, increment sample_count.
        """
        if template_hash in self._templates:
            existing = self._templates[template_hash]
            if existing.label == label:
                existing.sample_count += 1
            return

        entry = TemplateEntry(
            template_hash=template_hash,
            label=label,
            sample_count=1,
            source=source,
        )
        self._templates[template_hash] = entry
        self._by_label.setdefault(label, []).append(template_hash)

    def lookup(self, template_hash: str) -> TemplateEntry | None:
        """Look up a template hash. Returns None if not found."""
        return self._templates.get(template_hash)

    def partial_match(
        self, template_hash: str, prefix_len: int = 16
    ) -> TemplateEntry | None:
        """Look up by hash prefix (partial match fallback)."""
        for h, entry in self._templates.items():
            if h[:prefix_len] == template_hash[:prefix_len]:
                return entry
        return None

    def hashes_for_label(self, label: str) -> list[str]:
        """Return all template hashes for a given label."""
        return list(self._by_label.get(label, []))

    @property
    def count(self) -> int:
        return len(self._templates)

    @property
    def labels(self) -> list[str]:
        return sorted(self._by_label.keys())


# ── Builtin template library (empty at factory — populated from samples) ──

BUILTIN_TEMPLATES = TemplateLibrary()

# Pre-populate with placeholder entries for common types.
# Real hashes will be computed from actual sample documents during
# Phase 0 knowledge building. These labels ensure the template library
# is aware of all types even before real templates are loaded.
_BUILTIN_LABELS = [
    "HR & Payroll",
    "Financial Report",
    "Medical Record",
    "Legal Document",
    "API Log",
    "Technical Document",
    "Identity Document",
    "Email / Communication",
    "Government Form",
    "Education Record",
    "Real Estate Document",
    "Marketing Document",
    "Scientific Paper",
]

for _lbl in _BUILTIN_LABELS:
    # Register label without any hashes yet — real hashes come from
    # sample document processing during deployment.
    BUILTIN_TEMPLATES._by_label.setdefault(_lbl, [])
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/knowledge/templates.py
git commit -m "feat: add template hash library for E2 engine"
```

---

### Task 5: Create `src/knowledge/type_library.py` — KnownType Registry

**Files:**
- Create: `impl-datadna/src/knowledge/type_library.py`

- [ ] **Step 1: Write `type_library.py`**

```python
"""Type library — central registry of known document types.

Manages the lifecycle of document types: registration, centroid updates,
keyword management, PII distribution tracking, and expiration.

Per spec section 5, types have a source (builtin/llm/discovery), status
(active/deprecated), and optional expiration for dynamic types.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class TypeInfo:
    """Metadata for a single known document type.

    Attributes:
        type_id: Unique identifier (snake_case).
        type_name: Human-readable display name.
        source: "builtin", "llm", or "discovery".
        status: "active" or "deprecated".
        centroid: Mean BGE-M3 embedding of type exemplars (1024-dim).
        keywords: Representative keywords for this type.
        pii_distribution: Typical PII type frequency distribution.
        structural_signatures: Structural hashes seen for this type.
        sample_count: Number of documents labeled as this type.
        created_at: Unix timestamp of first registration.
        last_seen_at: Unix timestamp of most recent sample.
        rules: E1 regex rule IDs associated with this type.
        template_hashes: E2 template hashes for this type.
    """

    __slots__ = (
        "type_id", "type_name", "source", "status", "centroid",
        "keywords", "pii_distribution", "structural_signatures",
        "sample_count", "created_at", "last_seen_at",
        "rules", "template_hashes",
    )

    def __init__(
        self,
        type_id: str,
        type_name: str,
        source: str = "builtin",
        centroid: np.ndarray | None = None,
        keywords: list[str] | None = None,
        pii_distribution: dict[str, float] | None = None,
        structural_signatures: list[str] | None = None,
        rules: list[str] | None = None,
        template_hashes: list[str] | None = None,
    ) -> None:
        self.type_id = type_id
        self.type_name = type_name
        self.source = source
        self.status = "active"
        self.centroid = centroid
        self.keywords = keywords or []
        self.pii_distribution = pii_distribution or {}
        self.structural_signatures = structural_signatures or []
        self.sample_count = 0
        self.created_at = time.time()
        self.last_seen_at = self.created_at
        self.rules = rules or []
        self.template_hashes = template_hashes or []

    def is_deprecated(self, max_idle_days: int = 180) -> bool:
        """Check if type should be deprecated (no new samples)."""
        if self.source == "builtin":
            return False
        idle = time.time() - self.last_seen_at
        return idle > max_idle_days * 86400

    def is_expired(self, max_idle_days: int = 365) -> bool:
        """Check if type should be deleted."""
        if self.source == "builtin":
            return False
        return time.time() - self.last_seen_at > max_idle_days * 86400

    def update_centroid(self, new_embedding: np.ndarray) -> None:
        """Incremental centroid update with a new sample embedding."""
        if self.centroid is None:
            self.centroid = np.asarray(new_embedding, dtype=np.float32)
        else:
            n = float(self.sample_count)
            self.centroid = (self.centroid * n + np.asarray(new_embedding, dtype=np.float32)) / (n + 1.0)
        self.sample_count += 1
        self.last_seen_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (centroid omitted)."""
        return {
            "type_id": self.type_id,
            "type_name": self.type_name,
            "source": self.source,
            "status": self.status,
            "keywords": self.keywords,
            "pii_distribution": self.pii_distribution,
            "structural_signatures": self.structural_signatures,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
        }


class TypeLibrary:
    """Central registry of known document types.

    Provides registration, lookup, centroid management, and type
    lifecycle operations (deprecation, expiration, dedup).

    Usage:
        lib = TypeLibrary()
        lib.register("hr_payroll", "HR & Payroll", source="builtin")
        info = lib.get("hr_payroll")
        all_types = lib.list_active()
    """

    def __init__(self) -> None:
        self._types: dict[str, TypeInfo] = {}

    def register(
        self,
        type_id: str,
        type_name: str,
        source: str = "builtin",
        centroid: np.ndarray | None = None,
        keywords: list[str] | None = None,
        pii_distribution: dict[str, float] | None = None,
        structural_signatures: list[str] | None = None,
        rules: list[str] | None = None,
        template_hashes: list[str] | None = None,
    ) -> TypeInfo:
        """Register a new type or return existing one.

        If a type with the same type_id already exists, it is returned
        unchanged (no overwrite). Use update_centroid() to add samples.
        """
        if type_id in self._types:
            return self._types[type_id]

        info = TypeInfo(
            type_id=type_id,
            type_name=type_name,
            source=source,
            centroid=centroid,
            keywords=keywords,
            pii_distribution=pii_distribution,
            structural_signatures=structural_signatures,
            rules=rules,
            template_hashes=template_hashes,
        )
        self._types[type_id] = info
        return info

    def get(self, type_id: str) -> TypeInfo | None:
        """Look up a type by ID."""
        return self._types.get(type_id)

    def get_by_name(self, type_name: str) -> TypeInfo | None:
        """Look up a type by display name."""
        for info in self._types.values():
            if info.type_name == type_name:
                return info
        return None

    def list_active(self) -> list[TypeInfo]:
        """Return all active (non-deprecated) types."""
        return [t for t in self._types.values() if t.status == "active"]

    def list_centroids(self) -> list[tuple[str, np.ndarray | None]]:
        """Return (type_name, centroid) for all active types with centroids."""
        return [
            (t.type_name, t.centroid)
            for t in self._types.values()
            if t.status == "active" and t.centroid is not None
        ]

    def check_deprecation(self) -> list[str]:
        """Return type_ids that should be deprecated (>180 days idle)."""
        return [tid for tid, info in self._types.items() if info.is_deprecated()]

    def check_expiration(self) -> list[str]:
        """Return type_ids that should be deleted (>365 days idle)."""
        return [tid for tid, info in self._types.items() if info.is_expired()]

    def deprecate(self, type_id: str) -> None:
        """Mark a type as deprecated."""
        info = self._types.get(type_id)
        if info:
            info.status = "deprecated"

    def remove(self, type_id: str) -> None:
        """Remove a type from the library."""
        self._types.pop(type_id, None)

    @property
    def count(self) -> int:
        return len(self._types)

    @property
    def active_count(self) -> int:
        return len(self.list_active())


# ── Builtin types — 13 document types registered at factory ──

BUILTIN_TYPE_NAMES: list[tuple[str, str, list[str]]] = [
    ("hr_payroll", "HR & Payroll", ["payroll", "salary", "W2", "SSN", "benefits", "compensation"]),
    ("financial_report", "Financial Report", ["revenue", "invoice", "balance", "expense", "budget", "tax"]),
    ("medical_record", "Medical Record", ["diagnosis", "patient", "prescribed", "HIPAA", "lab", "NPI"]),
    ("legal_document", "Legal Document", ["contract", "agreement", "NDA", "compliance", "litigation"]),
    ("api_log", "API Log", ["timestamp", "endpoint", "status", "request", "response"]),
    ("technical_document", "Technical Document", ["config", "server", "database", "function", "README"]),
    ("identity_document", "Identity Document", ["passport", "license", "birth", "resume", "application"]),
    ("email_communication", "Email / Communication", ["From:", "Subject:", "meeting", "memo", "agenda"]),
    ("government_form", "Government Form", ["form", "grant", "permit", "census", "federal"]),
    ("education_record", "Education Record", ["transcript", "diploma", "GPA", "enrollment", "semester"]),
    ("real_estate", "Real Estate Document", ["deed", "mortgage", "lease", "property", "escrow"]),
    ("marketing", "Marketing Document", ["campaign", "analytics", "pitch", "CTR", "conversion"]),
    ("scientific_paper", "Scientific Paper", ["abstract", "methodology", "clinical", "patent", "journal"]),
]

# Global singleton — initialized once at startup
_default_library: TypeLibrary | None = None


def get_type_library() -> TypeLibrary:
    """Return the global TypeLibrary singleton, initializing if needed."""
    global _default_library
    if _default_library is None:
        _default_library = TypeLibrary()
        for tid, tname, keywords in BUILTIN_TYPE_NAMES:
            _default_library.register(
                type_id=tid,
                type_name=tname,
                source="builtin",
                keywords=keywords,
            )
    return _default_library
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/knowledge/type_library.py
git commit -m "feat: add TypeLibrary with 13 builtin document types"
```

---

### Task 6: Create E1 Regex Engine

**Files:**
- Create: `impl-datadna/src/engines/e1_regex.py`

- [ ] **Step 1: Write `e1_regex.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/e1_regex.py
git commit -m "feat: add E1 regex rule engine with 55 document type rules"
```

---

### Task 7: Create E2 Template Hash Engine

**Files:**
- Create: `impl-datadna/src/engines/e2_template.py`

- [ ] **Step 1: Write `e2_template.py`**

```python
"""E2: Template hash engine.

PII detection → entity type placeholder replacement → SHA256 hash →
template library lookup.

Inspired by Cyera's metadata replacement scheme (patent US12026123B2).
If the document has < 3 PII entities, template match is unlikely and
this engine naturally produces no output — the fusion voter handles it.

Dependencies:
  - PII_PATTERNS from knowledge/rules.py (shared regex patterns, not runtime)
  - TemplateLibrary from knowledge/templates.py
"""

from __future__ import annotations

import hashlib

from src.engines.base import BaseEngine
from src.knowledge.rules import PII_PATTERNS
from src.knowledge.templates import BUILTIN_TEMPLATES, TemplateLibrary
from src.types import Document, EngineOutput


class E2TemplateEngine(BaseEngine):
    """Template hash engine — PII replacement + SHA256 + template lookup.

    Independent of E1 regex engine. PII detection here uses the same
    pattern definitions but runs independently — if PII_PATTERNS is
    corrupted, only E2 is affected, not E1.

    Attributes:
        engine_id: "E2_template"
        weight: 1.0 (deterministic, match = high confidence)
    """

    engine_id = "E2_template"

    def __init__(self, template_library: TemplateLibrary | None = None) -> None:
        self._library = template_library or BUILTIN_TEMPLATES
        self._pii_patterns = PII_PATTERNS

    @property
    def weight(self) -> float:
        return 1.0

    def analyze(self, doc: Document) -> EngineOutput:
        """Replace PII entities, hash, and look up in template library.

        Returns:
            EngineOutput with matched label and confidence=1.0, or
            status="no_match" if hash not found in library.
        """
        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        # Step 1: Detect and replace PII entities with type placeholders
        replaced_text, pii_count = self._replace_pii(text)

        # Step 2: SHA256 hash of the replaced text
        template_hash = hashlib.sha256(
            replaced_text.encode("utf-8")
        ).hexdigest()

        # Step 3: Look up in template library
        entry = self._library.lookup(template_hash)
        if entry is not None:
            return EngineOutput(
                engine_id=self.engine_id,
                label=entry.label,
                confidence=1.0,
                status="matched",
                metadata={
                    "template_hash": template_hash,
                    "pii_count": pii_count,
                    "source": entry.source,
                },
            )

        # Try partial match (first 16 hex chars) as fallback
        partial = self._library.partial_match(template_hash, prefix_len=16)
        if partial is not None:
            return EngineOutput(
                engine_id=self.engine_id,
                label=partial.label,
                confidence=0.5,
                status="matched",
                metadata={
                    "template_hash": template_hash,
                    "pii_count": pii_count,
                    "match_type": "partial",
                },
            )

        return EngineOutput(
            engine_id=self.engine_id,
            status="no_match",
            metadata={"pii_count": pii_count},
        )

    def _replace_pii(self, text: str) -> tuple[str, int]:
        """Replace detected PII entities with type placeholders.

        Returns (replaced_text, entity_count).
        """
        count = 0
        # Sort matches by start position for deterministic replacement
        matches: list[tuple[int, int, str]] = []
        for pii_type, pattern in self._pii_patterns.items():
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), pii_type))

        if not matches:
            return text, 0

        # Sort by start, then by end descending (longest match first)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Build replaced text, skipping overlapping matches
        result: list[str] = []
        pos = 0
        last_end = 0
        for start, end, pii_type in matches:
            if start < last_end:
                continue  # Skip overlapping
            result.append(text[pos:start])
            result.append(f"[{pii_type}]")
            pos = end
            last_end = end
            count += 1

        result.append(text[pos:])
        return "".join(result), count
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/e2_template.py
git commit -m "feat: add E2 template hash engine with PII replacement"
```

---

### Task 8: Create E3 ML (SetFit) Engine

**Files:**
- Create: `impl-datadna/src/engines/e3_ml.py`

- [ ] **Step 1: Write `e3_ml.py`**

```python
"""E3: ML classifier engine — wraps SetFit distilled model.

This engine is UNAVAILABLE until a SetFit model has been trained
(sample_count >= 50 per class). Once trained, it provides ~2ms
inference with confidence from predict_proba.

Delegates to distillation/trainer.py for the actual model.
"""

from __future__ import annotations

from typing import Any

from src.engines.base import BaseEngine
from src.types import Document, EngineOutput


class E3MLEngine(BaseEngine):
    """SetFit ML classifier engine.

    Wraps a trained SetFit model for fast (~2ms) CPU inference.
    Unavailable until training has occurred.

    Attributes:
        engine_id: "E3_ml"
        weight: 1.5 (statistical model, broad coverage)
    """

    engine_id = "E3_ml"

    def __init__(self) -> None:
        self._model: Any = None
        self._trainer: Any = None

    def set_model(self, model: Any, trainer: Any = None) -> None:
        """Set the trained SetFit model, making this engine available."""
        self._model = model
        self._trainer = trainer

    @property
    def weight(self) -> float:
        return 1.5

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def analyze(self, doc: Document) -> EngineOutput:
        """Run SetFit inference on the document.

        Returns:
            EngineOutput with predicted label and confidence, or
            status="unavailable" if model not trained.
        """
        if not self.is_available or self._trainer is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )

        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        try:
            label, confidence = self._trainer.predict(self._model, text)
            return EngineOutput(
                engine_id=self.engine_id,
                label=label,
                confidence=round(confidence, 4),
                status="matched",
                metadata={},
            )
        except Exception:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/e3_ml.py
git commit -m "feat: add E3 SetFit ML classifier engine"
```

---

### Task 9: Create E4 Semantic kNN Engine

**Files:**
- Create: `impl-datadna/src/engines/e4_knn.py`

- [ ] **Step 1: Write `e4_knn.py`**

```python
"""E4: Semantic kNN engine.

BGE-M3 embedding → cosine similarity to all known type centroids →
nearest centroid label + distance-based confidence.

Activation: type library must have ≥ 5 active types with centroids.
Coverage: broadest engine — can match any document type with a centroid.

Degradation: if BGE-M3 is unavailable, this engine returns "unavailable".
Only E4 is affected — all other engines continue independently.
"""

from __future__ import annotations

import numpy as np

from src.engines.base import BaseEngine
from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, EngineOutput


class E4kNNEngine(BaseEngine):
    """Semantic kNN engine — BGE-M3 embedding + centroid similarity.

    Attributes:
        engine_id: "E4_knn"
        weight: 1.0 (semantic similarity, broad coverage)
    """

    engine_id = "E4_knn"

    def __init__(
        self,
        embedder=None,
        type_library: TypeLibrary | None = None,
        min_types: int = 5,
    ) -> None:
        """Initialize the kNN engine.

        Args:
            embedder: BgeM3Embedder instance (or any object with encode()).
            type_library: TypeLibrary for centroid lookup.
            min_types: Minimum active types with centroids to activate.
        """
        self._embedder = embedder
        self._type_library = type_library or get_type_library()
        self._min_types = min_types

    @property
    def weight(self) -> float:
        return 1.0

    @property
    def is_available(self) -> bool:
        if self._embedder is None:
            return False
        centroids = self._type_library.list_centroids()
        return len(centroids) >= self._min_types

    def analyze(self, doc: Document) -> EngineOutput:
        """Embed document + find nearest type centroid.

        Returns:
            EngineOutput with nearest type label and cosine similarity
            as confidence, or "unavailable"/"no_match".
        """
        if not self.is_available or self._embedder is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )

        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        centroids = self._type_library.list_centroids()
        if not centroids:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        try:
            # Reuse existing embedding if available on doc
            if doc.embedding is not None:
                embedding = doc.embedding
            else:
                embedding = self._embedder.encode([text])[0]

            best_label = None
            best_similarity = -1.0

            for label, centroid in centroids:
                if centroid is None:
                    continue
                sim = float(np.dot(embedding, centroid))
                if sim > best_similarity:
                    best_similarity = sim
                    best_label = label

            if best_label is None:
                return EngineOutput(
                    engine_id=self.engine_id,
                    status="no_match",
                )

            return EngineOutput(
                engine_id=self.engine_id,
                label=best_label,
                confidence=round(max(0.0, best_similarity), 4),
                status="matched",
                metadata={"cosine_similarity": round(best_similarity, 4)},
            )
        except Exception:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/e4_knn.py
git commit -m "feat: add E4 semantic kNN engine with BGE-M3 centroids"
```

---

### Task 10: Create E5 Structural Signature Engine

**Files:**
- Create: `impl-datadna/src/engines/e5_structural.py`

- [ ] **Step 1: Write `e5_structural.py`**

```python
"""E5: Structural signature engine.

Extracts document structural features (file type, size quantile, page/para
counts, etc.), hashes them with SHA256, and matches against known structural
signatures in the type library.

Coverage: narrow — only distinguishes file FORMAT, not content type.
Weight: lowest (0.8) — auxiliary signal, not primary classifier.
"""

from __future__ import annotations

import hashlib
import math

from src.engines.base import BaseEngine
from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, EngineOutput


class E5StructuralEngine(BaseEngine):
    """Structural signature engine — file metadata hash matching.

    Attributes:
        engine_id: "E5_structural"
        weight: 0.8 (lowest — format-level only, not content)
    """

    engine_id = "E5_structural"

    _FEATURE_KEYS = [
        "file_type", "file_size_quantile", "page_count",
        "paragraph_count", "table_count", "has_images",
        "header_pattern", "json_schema_signature", "path_depth",
    ]

    def __init__(self, type_library: TypeLibrary | None = None) -> None:
        self._type_library = type_library or get_type_library()

    @property
    def weight(self) -> float:
        return 0.8

    def analyze(self, doc: Document) -> EngineOutput:
        """Extract structural signature, hash, and match against type library.

        Returns:
            EngineOutput with matched type or status="no_match".
        """
        meta = doc.metadata or {}
        if not meta:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        # Build structural feature string
        signature = self._build_signature(meta)
        sig_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()

        # Check against known structural signatures in type library
        for info in self._type_library.list_active():
            if sig_hash in info.structural_signatures:
                return EngineOutput(
                    engine_id=self.engine_id,
                    label=info.type_name,
                    confidence=1.0,
                    status="matched",
                    metadata={"signature_hash": sig_hash},
                )

        return EngineOutput(
            engine_id=self.engine_id,
            status="no_match",
            metadata={"signature_hash": sig_hash},
        )

    def _build_signature(self, metadata: dict) -> str:
        """Build canonical structural feature string for hashing."""
        parts = []
        for key in sorted(self._FEATURE_KEYS):
            val = metadata.get(key)
            if val is None:
                val = self._default_for_key(key)
            parts.append(f"{key}:{self._format_value(val)}")
        return "|".join(parts)

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    @staticmethod
    def _default_for_key(key: str):
        defaults = {
            "file_type": "",
            "file_size_quantile": 0,
            "page_count": 0,
            "paragraph_count": 0,
            "table_count": 0,
            "has_images": False,
            "header_pattern": "",
            "json_schema_signature": "",
            "path_depth": 0,
        }
        return defaults.get(key, "")
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/e5_structural.py
git commit -m "feat: add E5 structural signature engine"
```

---

### Task 11: Create E6 LLM Engine

**Files:**
- Create: `impl-datadna/src/engines/e6_llm.py`

- [ ] **Step 1: Write `e6_llm.py`**

```python
"""E6: LLM classification engine.

Wraps Mistral-7B via llm/client.py. Activated only when E1-E5 preliminary
fusion confidence < 0.85 (gate controlled by fusion/voter.py).

Highest weight (2.0) — when LLM has an opinion, it dominates the vote.
Highest latency (~1.4s) — the reason for the preliminary consensus gate.

Degradation: if Ollama is down, this engine returns "unavailable".
System continues with E1-E5 fusion (degraded=true).
"""

from __future__ import annotations

from src.engines.base import BaseEngine
from src.knowledge.type_library import TypeLibrary, get_type_library
from src.types import Document, EngineOutput


class E6LLMEngine(BaseEngine):
    """LLM classification engine — Mistral-7B via Ollama.

    Attributes:
        engine_id: "E6_llm"
        weight: 2.0 (highest accuracy, dominates when available)
    """

    engine_id = "E6_llm"

    def __init__(
        self,
        llm_client=None,
        type_library: TypeLibrary | None = None,
    ) -> None:
        """Initialize the LLM engine.

        Args:
            llm_client: MistralClient instance (or None → unavailable).
            type_library: TypeLibrary for known type names.
        """
        self._client = llm_client
        self._type_library = type_library or get_type_library()

    @property
    def weight(self) -> float:
        return 2.0

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def analyze(self, doc: Document) -> EngineOutput:
        """Classify document using Mistral-7B.

        Returns:
            EngineOutput with LLM-assigned label and confidence, or
            status="unavailable" if LLM client not configured.
        """
        if not self.is_available or self._client is None:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )

        text = doc.text or ""
        if not text:
            return EngineOutput(engine_id=self.engine_id, status="no_match")

        try:
            known_types = [
                t.type_name for t in self._type_library.list_active()
            ]
            response = self._client.classify(text, known_types)
            label = response.get("label", "unknown")
            confidence = float(response.get("confidence", 0.0))
            is_new = bool(response.get("is_new_type", False))

            return EngineOutput(
                engine_id=self.engine_id,
                label=label,
                confidence=round(confidence, 4),
                status="matched",
                metadata={
                    "is_new_type": is_new,
                    "rationale": response.get("rationale", ""),
                },
            )
        except Exception:
            return EngineOutput(
                engine_id=self.engine_id,
                status="unavailable",
            )
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/e6_llm.py
git commit -m "feat: add E6 LLM classification engine"
```

---

### Task 12: Create Fusion Voter

**Files:**
- Create: `impl-datadna/src/fusion/__init__.py`
- Create: `impl-datadna/src/fusion/voter.py`

- [ ] **Step 1: Create `__init__.py`**

```python
"""Weighted voting fusion engine."""
```

- [ ] **Step 2: Write `voter.py`**

```python
"""Fusion voter — weighted voting across 6 engines.

Core algorithm per spec section 2.3:
  1. Group engine outputs by label
  2. score(label) = Σ(engine.weight × confidence × is_available)
  3. final_label = argmax(score)
  4. composite_confidence = max_score / Σ(all engine weights)

LLM gating per spec section 2.4:
  1. Run E1-E5 first
  2. Preliminary fusion → if max_score / Σ(E1..E5 weights) >= 0.85
     → skip E6, method="fusion_fast"
  3. Else → run E6, full 6-engine fusion, method="fusion_full"

All engines unavailable → label="unclassified", manual_review=true.
"""

from __future__ import annotations

from src.types import Document, EngineOutput, FusionResult


# ── Engine weights per spec section 2.3 ──
ENGINE_WEIGHTS: dict[str, float] = {
    "E1_regex": 1.0,
    "E2_template": 1.0,
    "E3_ml": 1.5,
    "E4_knn": 1.0,
    "E5_structural": 0.8,
    "E6_llm": 2.0,
}

# Threshold for skipping LLM (E1-E5 consensus)
PRELIMINARY_CONSENSUS_THRESHOLD = 0.85

# Threshold below which manual review is required
MANUAL_REVIEW_THRESHOLD = 0.4


class FusionVoter:
    """Weighted voting fusion across 6 parallel classification engines.

    Usage:
        voter = FusionVoter(engines=[e1, e2, e3, e4, e5, e6])
        result = voter.classify(document)
        # result.method is "fusion_fast" or "fusion_full"
    """

    def __init__(self, engines: list | None = None) -> None:
        """Initialize with a list of BaseEngine instances.

        Args:
            engines: List of engine instances. If None, no engines registered.
        """
        self._engines: dict[str, object] = {}
        self._fast_engines: list[str] = []  # Engine IDs for E1-E5
        self._all_engines: list[str] = []   # All engine IDs

        if engines:
            for engine in engines:
                self._engines[engine.engine_id] = engine
                self._all_engines.append(engine.engine_id)
                if engine.engine_id != "E6_llm":
                    self._fast_engines.append(engine.engine_id)

    def classify(self, doc: Document) -> FusionResult:
        """Classify a document through the fusion pipeline.

        1. Run E1-E5 in sequence (total < 5ms)
        2. Preliminary fusion → check consensus
        3. If consensus < threshold → run E6
        4. Final fusion with all available engine outputs

        Args:
            doc: Document to classify.

        Returns:
            FusionResult with final_label, composite_confidence, method, etc.
        """
        engine_outputs: dict[str, EngineOutput] = {}
        degraded = False

        # ── Step 1: Run E1-E5 ──────────────────────────────────
        for eid in self._fast_engines:
            engine = self._engines.get(eid)
            if engine is None:
                continue
            try:
                output = engine.analyze(doc)
            except Exception:
                output = EngineOutput(
                    engine_id=eid,
                    status="unavailable",
                )
                degraded = True
            engine_outputs[eid] = output
            if output.status == "unavailable":
                degraded = True

        # ── Step 2: Preliminary fusion (E1-E5 only) ────────────
        fast_scores = self._compute_scores(engine_outputs)
        fast_max = max(fast_scores.values()) if fast_scores else 0.0
        fast_total_weight = sum(
            ENGINE_WEIGHTS.get(eid, 0.0) for eid in self._fast_engines
        )
        prelim_confidence = fast_max / fast_total_weight if fast_total_weight > 0 else 0.0

        # ── Step 3: LLM gating ─────────────────────────────────
        method = "fusion_fast"
        e6_output = None

        if prelim_confidence < PRELIMINARY_CONSENSUS_THRESHOLD:
            # Need LLM — run E6
            e6_engine = self._engines.get("E6_llm")
            if e6_engine is not None:
                try:
                    e6_output = e6_engine.analyze(doc)
                except Exception:
                    e6_output = EngineOutput(
                        engine_id="E6_llm",
                        status="unavailable",
                    )
                    degraded = True
                engine_outputs["E6_llm"] = e6_output
                if e6_output.status == "unavailable":
                    degraded = True
                method = "fusion_full"
            else:
                method = "fusion_fast"
        else:
            # Skip LLM — mark as skipped in output
            engine_outputs["E6_llm"] = EngineOutput(
                engine_id="E6_llm",
                status="skipped",
            )

        # ── Step 4: Full fusion (all available engines) ────────
        final_scores = self._compute_scores(engine_outputs)

        if not final_scores:
            # All engines unavailable → unclassified
            return FusionResult(
                doc_id=doc.doc_id,
                final_label="unclassified",
                composite_confidence=0.0,
                method=method,
                degraded=True,
                manual_review=True,
                engine_outputs=engine_outputs,
                label_scores={},
            )

        max_label = max(final_scores, key=final_scores.get)
        max_score = final_scores[max_label]
        total_weight = sum(
            ENGINE_WEIGHTS.get(eid, 0.0)
            for eid in self._all_engines
            if eid in engine_outputs
            and engine_outputs[eid].status not in ("unavailable", "skipped")
        )
        composite_confidence = (
            max_score / total_weight if total_weight > 0 else 0.0
        )

        manual_review = composite_confidence < MANUAL_REVIEW_THRESHOLD

        return FusionResult(
            doc_id=doc.doc_id,
            final_label=max_label,
            composite_confidence=round(composite_confidence, 4),
            method=method,
            degraded=degraded,
            manual_review=manual_review,
            engine_outputs=engine_outputs,
            label_scores={
                lbl: round(s, 4) for lbl, s in final_scores.items()
            },
        )

    def _compute_scores(
        self, engine_outputs: dict[str, EngineOutput]
    ) -> dict[str, float]:
        """Compute weighted scores for each label.

        score(label) = Σ(weight × confidence) for engines voting for that label.
        Engines with status "unavailable" or "skipped" contribute 0.
        """
        scores: dict[str, float] = {}
        for eid, output in engine_outputs.items():
            if output.status in ("unavailable", "skipped"):
                continue
            if output.label is None:
                continue
            weight = ENGINE_WEIGHTS.get(eid, 0.0)
            scores[output.label] = (
                scores.get(output.label, 0.0) + weight * output.confidence
            )
        return scores
```

- [ ] **Step 3: Commit**

```bash
git add impl-datadna/src/fusion/
git commit -m "feat: add weighted voting fusion voter with LLM gating"
```

---

### Task 13: Create Monitoring Module

**Files:**
- Create: `impl-datadna/src/monitoring/__init__.py`
- Create: `impl-datadna/src/monitoring/audit.py`
- Create: `impl-datadna/src/monitoring/metrics.py`

- [ ] **Step 1: Create `__init__.py`**

```python
"""Quality monitoring and audit logging."""
```

- [ ] **Step 2: Write `audit.py`**

```python
"""Per-decision JSON audit log writer.

Every document classification produces an AuditRecord with all 6 engine
outputs, the fusion decision, and metadata. Per spec section 8.

Records are written as JSON Lines for append-only streaming.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.types import FusionResult


class AuditLogger:
    """JSON Lines audit log for every classification decision.

    Usage:
        logger = AuditLogger(Path("./output/audit.jsonl"))
        logger.log(result)
        # Each call appends one JSON line to the file.
    """

    def __init__(self, output_path: Path) -> None:
        self._path = output_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def log(self, result: FusionResult) -> None:
        """Append one audit record to the log file.

        Args:
            result: FusionResult from the fusion voter.
        """
        engines = {}
        for eid, output in result.engine_outputs.items():
            engines[eid] = {
                "status": output.status,
                "label": output.label,
                "confidence": output.confidence,
                "metadata": output.metadata,
            }

        record = {
            "doc_id": result.doc_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "final_label": result.final_label,
            "composite_confidence": result.composite_confidence,
            "method": result.method,
            "degraded": result.degraded,
            "manual_review": result.manual_review,
            "label_scores": result.label_scores,
            "engines": engines,
        }

        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._count += 1

    @property
    def count(self) -> int:
        return self._count
```

- [ ] **Step 3: Write `metrics.py`**

```python
"""Quality monitoring metrics and alert threshold checks.

7 monitoring metrics per spec section 7:
  1. Fusion confidence distribution → alert: P50 < 0.3
  2. Per-engine output rate → alert: any engine rate drop > 30%
  3. LLM call rate → alert: > 50% or = 0%
  4. manual_review backlog → alert: > 10%
  5. New type registration rate → alert: > 10/hour
  6. Label distribution KL divergence vs baseline → alert: > 0.3
  7. fusion_fast validation inconsistency rate → alert: > 5%

Does NOT auto-remediate. Alerts → human operator decision.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from src.types import FusionResult


@dataclass
class MetricsSnapshot:
    """Aggregated metrics over a monitoring window (default 1 hour)."""

    total_documents: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)
    manual_review_count: int = 0
    degraded_count: int = 0
    confidence_values: list[float] = field(default_factory=list)
    engine_output_rates: dict[str, float] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    @property
    def llm_call_rate(self) -> float:
        total = self.total_documents
        if total == 0:
            return 0.0
        fusion_full = self.method_counts.get("fusion_full", 0)
        return fusion_full / total

    @property
    def manual_review_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.manual_review_count / self.total_documents

    @property
    def p50_confidence(self) -> float:
        if not self.confidence_values:
            return 0.0
        sorted_vals = sorted(self.confidence_values)
        mid = len(sorted_vals) // 2
        return sorted_vals[mid]


class MetricsCollector:
    """Collects per-document metrics and checks alert thresholds.

    Usage:
        collector = MetricsCollector()
        collector.record(result)
        snapshot = collector.snapshot()
        for alert in snapshot.alerts:
            logger.warning("ALERT: %s", alert)
    """

    def __init__(self) -> None:
        self._results: list[FusionResult] = []
        self._baseline_labels: dict[str, float] | None = None

    def set_baseline(self, label_distribution: dict[str, float]) -> None:
        """Set baseline label distribution for KL divergence comparison."""
        self._baseline_labels = label_distribution

    def record(self, result: FusionResult) -> None:
        """Record one classification result."""
        self._results.append(result)

    def snapshot(self) -> MetricsSnapshot:
        """Compute current metrics snapshot and check thresholds."""
        snap = MetricsSnapshot()
        snap.total_documents = len(self._results)

        engine_output_counts: dict[str, int] = {}

        for r in self._results:
            # Method
            snap.method_counts[r.method] = (
                snap.method_counts.get(r.method, 0) + 1
            )
            # Confidence
            snap.confidence_values.append(r.composite_confidence)
            # Manual review
            if r.manual_review:
                snap.manual_review_count += 1
            # Degraded
            if r.degraded:
                snap.degraded_count += 1
            # Label distribution
            lbl = r.final_label
            snap.label_distribution[lbl] = snap.label_distribution.get(lbl, 0) + 1
            # Engine outputs
            for eid, eout in r.engine_outputs.items():
                if eout.status == "matched":
                    engine_output_counts[eid] = (
                        engine_output_counts.get(eid, 0) + 1
                    )

        # Compute engine output rates
        total = max(snap.total_documents, 1)
        for eid in ["E1_regex", "E2_template", "E3_ml", "E4_knn", "E5_structural", "E6_llm"]:
            snap.engine_output_rates[eid] = engine_output_counts.get(eid, 0) / total

        # ── Alert checks ──
        # 1. Confidence P50 < 0.3
        if snap.p50_confidence < 0.3:
            snap.alerts.append(
                f"Low confidence: P50={snap.p50_confidence:.3f} < 0.3"
            )

        # 2. LLM call rate > 50% or = 0%
        llm_rate = snap.llm_call_rate
        if llm_rate > 0.5:
            snap.alerts.append(
                f"High LLM call rate: {llm_rate:.1%} > 50%"
            )
        if llm_rate == 0.0 and total >= 50:
            snap.alerts.append("LLM call rate is 0% — LLM may be down")

        # 3. Manual review > 10%
        mr_rate = snap.manual_review_rate
        if mr_rate > 0.1:
            snap.alerts.append(
                f"High manual review rate: {mr_rate:.1%} > 10%"
            )

        # 4. Label distribution KL divergence vs baseline
        if self._baseline_labels is not None and snap.label_distribution:
            kl = self._kl_divergence(
                snap.label_distribution, self._baseline_labels
            )
            if kl > 0.3:
                snap.alerts.append(
                    f"Label drift detected: KL={kl:.3f} > 0.3"
                )

        return snap

    def reset(self) -> None:
        """Clear accumulated results for the next window."""
        self._results.clear()

    @staticmethod
    def _kl_divergence(
        current: dict[str, int],
        baseline: dict[str, float],
    ) -> float:
        """Compute KL divergence of current distribution vs baseline."""
        total = sum(current.values())
        if total == 0:
            return 0.0

        kl = 0.0
        for label, baseline_p in baseline.items():
            current_p = current.get(label, 0) / total
            if current_p > 0 and baseline_p > 0:
                kl += current_p * math.log(current_p / baseline_p)
        return kl
```

- [ ] **Step 4: Commit**

```bash
git add impl-datadna/src/monitoring/
git commit -m "feat: add audit logger and 7-metric quality monitoring"
```

---

### Task 14: Update `src/engines/__init__.py` — Re-export All Engines

**Files:**
- Modify: `impl-datadna/src/engines/__init__.py`

- [ ] **Step 1: Rewrite `__init__.py`**

```python
"""Six parallel classification engines with uniform interface."""

from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine

__all__ = [
    "E1RegexEngine",
    "E2TemplateEngine",
    "E3MLEngine",
    "E4kNNEngine",
    "E5StructuralEngine",
    "E6LLMEngine",
]
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/src/engines/__init__.py
git commit -m "feat: re-export all 6 engines from engines package"
```

---

### Task 15: Rewrite `config.yaml` for Fusion Architecture

**Files:**
- Modify: `impl-datadna/config.yaml`

- [ ] **Step 1: Rewrite `config.yaml`**

```yaml
# 6-Engine Fusion Architecture Configuration

# ── Engine Weights ──
engines:
  E1_regex:
    weight: 1.0
  E2_template:
    weight: 1.0
  E3_ml:
    weight: 1.5
  E4_knn:
    weight: 1.0
  E5_structural:
    weight: 0.8
  E6_llm:
    weight: 2.0

# ── Fusion ──
fusion:
  preliminary_consensus_threshold: 0.85  # E1-E5 consensus → skip LLM
  manual_review_threshold: 0.4           # composite_confidence < 0.4 → review

# ── E4 kNN ──
knn:
  min_types_for_activation: 5

# ── E6 LLM ──
llm:
  api_base: "http://localhost:11434/v1"
  model: "mistral:7b"
  quantization: "4bit"
  temperature: 0.3
  max_input_chars: 2000

# ── Embedding ──
embedding:
  model_name: "BAAI/bge-m3"
  device: "cuda"
  dim: 1024
  batch_size: 32
  max_token_length: 8192

# ── Knowledge Distillation (E3 ML) ──
distillation:
  min_samples_per_class: 50
  max_samples_per_class: 200
  confidence_threshold: 0.85
  retrain_trigger: 500
  f1_degradation_threshold: 0.03

# ── Type Discovery ──
discovery:
  outlier_buffer:
    min_trigger_count: 100
    same_pattern_threshold: 5
  candidate_evaluation:
    min_coherence: 0.75
    min_distance_to_known: 0.3
    min_cluster_size: 3

# ── Monitoring ──
monitoring:
  window_hours: 1
  alerts:
    confidence_p50_min: 0.3
    llm_call_rate_max: 0.5
    manual_review_rate_max: 0.1
    label_drift_kl_max: 0.3
    fusion_fast_inconsistency_max: 0.05
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/config.yaml
git commit -m "feat: rewrite config.yaml for 6-engine fusion architecture"
```

---

### Task 16: Write Tests — `tests/test_engines.py`

**Files:**
- Create: `impl-datadna/tests/test_engines.py`

- [ ] **Step 1: Write engine unit tests**

```python
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
        # May match or not match depending on template library
        # but should always return a valid EngineOutput
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
        engine.set_model(object(), object())  # dummy model + trainer
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
        assert len(output.metadata["signature_hash"]) == 64  # SHA256


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
```

- [ ] **Step 2: Run tests**

```bash
cd impl-datadna && python -m pytest tests/test_engines.py -v
```

- [ ] **Step 3: Commit**

```bash
git add impl-datadna/tests/test_engines.py
git commit -m "test: add unit tests for all 6 engines"
```

---

### Task 17: Write Tests — `tests/test_fusion.py`

**Files:**
- Create: `impl-datadna/tests/test_fusion.py`

- [ ] **Step 1: Write fusion voter tests**

```python
"""Unit tests for the fusion voter."""

from __future__ import annotations

import pytest

from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
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
        """When E1 and E5 agree on same label, consensus should be high."""
        e1 = E1RegexEngine()
        e5 = E5StructuralEngine()
        voter = FusionVoter(engines=[e1, e5])
        result = voter.classify(hr_doc)
        # E5 won't match (no structural sig), E1 should match HR
        assert result.final_label is not None
        assert result.final_label != "unclassified"

    def test_generic_document_all_no_match(self, generic_doc):
        """When no engine matches, result should be unclassified."""
        e1 = E1RegexEngine()
        e5 = E5StructuralEngine()
        voter = FusionVoter(engines=[e1, e5])
        result = voter.classify(generic_doc)
        # With no engines matching, we should get unclassified
        # (E1 no_match + E5 no_match)
        assert result.final_label == "unclassified"
        assert result.manual_review is True

    def test_manual_review_flagged_low_confidence(self):
        """If confidence is below 0.4, manual_review should be True."""
        # With no engines, confidence=0 → must be manual_review
        voter = FusionVoter(engines=[])
        doc = Document(doc_id="d", text="test", metadata={})
        result = voter.classify(doc)
        assert result.manual_review is True
        assert result.composite_confidence == 0.0
```

- [ ] **Step 2: Run tests**

```bash
cd impl-datadna && python -m pytest tests/test_fusion.py -v
```

- [ ] **Step 3: Commit**

```bash
git add impl-datadna/tests/test_fusion.py
git commit -m "test: add fusion voter unit tests"
```

---

### Task 18: Rewrite `tests/conftest.py` for Fusion Architecture

**Files:**
- Modify: `impl-datadna/tests/conftest.py`

- [ ] **Step 1: Rewrite `conftest.py`**

```python
"""Shared pytest fixtures for 6-engine fusion architecture tests."""

from __future__ import annotations

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
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/tests/conftest.py
git commit -m "test: update conftest for fusion architecture fixtures"
```

---

### Task 19: Rewrite `main.py` — Parallel Fusion Pipeline

**Files:**
- Modify: `impl-datadna/main.py`

- [ ] **Step 1: Rewrite `main.py`**

```python
#!/usr/bin/env python
"""DataDNA 6-Engine Fusion Classification — Main Entry Point

Parallel engine dispatch → weighted voting fusion → audit output.

Usage:
    python main.py --input ./docs/ --output ./output/ --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from src.embeddings.bge_m3 import BgeM3Embedder
from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine
from src.fusion.voter import FusionVoter
from src.knowledge.type_library import get_type_library
from src.llm.client import LLMConfig, MistralClient
from src.monitoring.audit import AuditLogger
from src.monitoring.metrics import MetricsCollector
from src.types import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Document loading (unchanged from original main.py)
# ═══════════════════════════════════════════════════════════════

def load_documents(input_dir: str) -> list[Document]:
    documents: list[Document] = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in (".txt", ".pdf", ".docx", ".json"):
            continue
        doc_id = file_path.stem
        rel_path = file_path.relative_to(input_path)
        metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_type": suffix,
            "file_size": file_path.stat().st_size,
            "path_depth": max(len(rel_path.parts) - 1, 0),
        }
        text = ""
        try:
            if suffix == ".pdf":
                text = _read_pdf(file_path, metadata)
            elif suffix == ".docx":
                text = _read_docx(file_path, metadata)
            elif suffix == ".json":
                text = _read_json(file_path)
            else:
                text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            continue
        if not text.strip():
            continue
        documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))
    logger.info("Loaded %d documents from %s", len(documents), input_dir)
    return documents


def _read_pdf(file_path: Path, metadata: dict[str, Any]) -> str:
    try:
        import fitz
    except ImportError:
        return file_path.read_text(encoding="utf-8", errors="replace")
    doc = fitz.open(str(file_path))
    try:
        pages = [page.get_text() for page in doc]
        metadata["page_count"] = len(pages)
        return "\n".join(pages)
    finally:
        doc.close()


def _read_docx(file_path: Path, metadata: dict[str, Any]) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError:
        return file_path.read_text(encoding="utf-8", errors="replace")
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs]
    metadata["paragraph_count"] = len(paragraphs)
    return "\n".join(paragraphs)


def _read_json(file_path: Path) -> str:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return file_path.read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════
# Component initialization
# ═══════════════════════════════════════════════════════════════

def _init_components(config: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, Any] = {}

    # Type library (always available)
    type_lib = get_type_library()
    components["type_library"] = type_lib
    logger.info("TypeLibrary initialized (%d types)", type_lib.count)

    # E1: Regex engine (always available)
    components["e1"] = E1RegexEngine()
    logger.info("E1 Regex engine initialized")

    # E2: Template engine (always available)
    components["e2"] = E2TemplateEngine()
    logger.info("E2 Template engine initialized")

    # E3: ML engine (available after training)
    components["e3"] = E3MLEngine()
    logger.info("E3 ML engine initialized (model not yet trained)")

    # BGE-M3 embedder for E4
    try:
        emb = config.get("embedding", {})
        embedder = BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cuda"),
            batch_size=emb.get("batch_size", 32),
            max_length=emb.get("max_token_length", 8192),
        )
        components["embedder"] = embedder
        logger.info("BgeM3Embedder initialized (dim=%d)", embedder.dim)
    except Exception as exc:
        logger.warning("BgeM3Embedder unavailable: %s — E4 will be disabled", exc)
        components["embedder"] = None

    # E4: kNN engine
    knn_cfg = config.get("knn", {})
    components["e4"] = E4kNNEngine(
        embedder=components["embedder"],
        type_library=type_lib,
        min_types=knn_cfg.get("min_types_for_activation", 5),
    )
    available = "available" if components["e4"].is_available else "unavailable"
    logger.info("E4 kNN engine initialized (%s)", available)

    # E5: Structural engine (always available)
    components["e5"] = E5StructuralEngine(type_library=type_lib)
    logger.info("E5 Structural engine initialized")

    # E6: LLM engine
    try:
        llm_cfg = config.get("llm", {})
        llm_client = MistralClient(LLMConfig(
            api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "mistral:7b"),
            quantization=llm_cfg.get("quantization", "4bit"),
            temperature=llm_cfg.get("temperature", 0.3),
        ))
        components["e6"] = E6LLMEngine(llm_client=llm_client, type_library=type_lib)
        logger.info("E6 LLM engine initialized")
    except Exception as exc:
        logger.warning("E6 LLM engine unavailable: %s", exc)
        components["e6"] = E6LLMEngine(llm_client=None, type_library=type_lib)

    # Fusion voter
    all_engines = [
        components["e1"], components["e2"], components["e3"],
        components["e4"], components["e5"], components["e6"],
    ]
    components["voter"] = FusionVoter(engines=all_engines)
    logger.info("FusionVoter initialized with %d engines", len(all_engines))

    return components


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="DataDNA 6-Engine Fusion Classifier")
    parser.add_argument("--input", required=True, help="Document directory path")
    parser.add_argument("--output", default="./output/", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    output_dir = Path(args.output)

    with open(args.config, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    logger.info("DataDNA Fusion: input=%s, output=%s", args.input, args.output)

    stats: dict[str, Any] = {}
    overall_start = time.perf_counter()

    # Init
    init_start = time.perf_counter()
    comp = _init_components(config)
    stats["init_time_s"] = round(time.perf_counter() - init_start, 3)

    # Load documents
    doc_start = time.perf_counter()
    documents = load_documents(args.input)
    stats["doc_load_time_s"] = round(time.perf_counter() - doc_start, 3)
    stats["doc_count"] = len(documents)

    if not documents:
        logger.warning("No documents found")
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"results": [], "stats": stats}, open(output_dir / "results.json", "w"))
        return 0

    # Classify
    voter: FusionVoter = comp["voter"]
    audit = AuditLogger(output_dir / "audit.jsonl")
    metrics = MetricsCollector()

    classify_start = time.perf_counter()
    results = []
    method_counts: dict[str, int] = {}
    degraded_count = 0
    manual_review_count = 0

    for doc in documents:
        result = voter.classify(doc)
        audit.log(result)
        metrics.record(result)
        results.append(result)

        method_counts[result.method] = method_counts.get(result.method, 0) + 1
        if result.degraded:
            degraded_count += 1
        if result.manual_review:
            manual_review_count += 1

    classify_time = round(time.perf_counter() - classify_start, 3)
    stats["classify_time_s"] = classify_time
    stats["docs"] = len(results)
    stats["method_counts"] = method_counts
    stats["degraded_count"] = degraded_count
    stats["manual_review_count"] = manual_review_count
    stats["avg_time_per_doc_ms"] = round(
        (classify_time / len(results)) * 1000, 1
    ) if results else 0

    # Metrics snapshot
    snap = metrics.snapshot()
    stats["metrics"] = {
        "p50_confidence": round(snap.p50_confidence, 4),
        "llm_call_rate": round(snap.llm_call_rate, 4),
        "manual_review_rate": round(snap.manual_review_rate, 4),
        "alerts": snap.alerts,
    }

    if snap.alerts:
        for alert in snap.alerts:
            logger.warning("ALERT: %s", alert)

    # Output
    total_time = round(time.perf_counter() - overall_start, 3)
    stats["total_time_s"] = total_time

    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = []
    for r in results:
        results_json.append({
            "doc_id": r.doc_id,
            "final_label": r.final_label,
            "composite_confidence": r.composite_confidence,
            "method": r.method,
            "degraded": r.degraded,
            "manual_review": r.manual_review,
        })

    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results_json, "stats": stats}, fh, ensure_ascii=False, indent=2)

    logger.info("Complete: %d docs in %.3fs | fusion_fast=%d fusion_full=%d | avg=%.1fms",
                len(results), total_time,
                method_counts.get("fusion_fast", 0),
                method_counts.get("fusion_full", 0),
                stats["avg_time_per_doc_ms"])
    logger.info("Audit log: %s (%d records)", output_dir / "audit.jsonl", audit.count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit**

```bash
git add impl-datadna/main.py
git commit -m "feat: rewrite main.py for 6-engine parallel fusion pipeline"
```

---

### Task 20: Remove Old Tier/NER Modules

**Files:**
- Remove directories and test files

- [ ] **Step 1: Remove old modules**

```bash
rm -rf impl-datadna/src/tier0
rm -rf impl-datadna/src/tier1
rm -rf impl-datadna/src/tier2
rm -rf impl-datadna/src/tier3
rm -rf impl-datadna/src/ner
rm -f impl-datadna/tests/test_tier0.py
rm -f impl-datadna/tests/test_tier1_structural.py
rm -f impl-datadna/tests/test_tier1_semantic.py
rm -f impl-datadna/tests/test_tier2_matching.py
rm -f impl-datadna/tests/test_tier2_classifier.py
rm -f impl-datadna/tests/test_tier3.py
rm -f impl-datadna/tests/test_ner.py
rm -f impl-datadna/tests/test_discovery.py
```

- [ ] **Step 2: Commit**

```bash
git add -A impl-datadna/
git commit -m "refactor: remove old tier0-3 and NER modules"
```

---

### Task 21: Run Full Test Suite and Verify

- [ ] **Step 1: Run all remaining tests**

```bash
cd impl-datadna && python -m pytest tests/ -v --tb=short
```

Expected: All tests in `test_engines.py`, `test_fusion.py`, `test_embeddings.py`, `test_llm_client.py` pass.

- [ ] **Step 2: Verify import chain**

```bash
cd impl-datadna && python -c "
from src.engines import E1RegexEngine, E2TemplateEngine, E3MLEngine, E4kNNEngine, E5StructuralEngine, E6LLMEngine
from src.fusion.voter import FusionVoter
from src.knowledge.rules import BUILTIN_RULES
from src.knowledge.type_library import get_type_library
from src.monitoring.audit import AuditLogger
from src.monitoring.metrics import MetricsCollector
print('All imports OK')
print(f'Rules: {len(BUILTIN_RULES)}')
print(f'Types: {get_type_library().count}')
"
```

Expected: "All imports OK", "Rules: 55", "Types: 13"

- [ ] **Step 3: Commit if any fixes needed**

```bash
git add -A impl-datadna/
git commit -m "chore: finalize fusion architecture migration"
```
```

---

## Self-Review

**1. Spec coverage check against `2026-05-23-optimal-architecture.md`:**

| Spec Section | Covered By |
|---|---|
| §2.1 Why not serial gating | Task 12 (fusion voter design) |
| §2.2 Six engines | Tasks 6-11 (E1 through E6) |
| §2.3 Fusion mechanism | Task 12 (weighted voting, confidence calc) |
| §2.4 LLM call optimization | Task 12 (preliminary consensus gate) |
| §3.1 E1 Regex design | Task 6 |
| §3.2 E2 Template design | Task 7 |
| §3.3 E3 ML design | Task 8 |
| §3.4 E4 kNN design | Task 9 |
| §3.5 E5 Structural design | Task 10 |
| §3.6 E6 LLM design | Task 11 |
| §5 Type library lifecycle | Task 5 |
| §6 Degradation paths | Task 12 (unavailable → weight=0), each engine's is_available |
| §7 Quality monitoring | Task 13 (7 metrics + alerts) |
| §8 Audit log | Task 13 (JSON Lines audit logger) |
| §12 Phase 0+1 | Tasks 3-5 (Phase 0), Tasks 6-20 (Phase 1) |

**2. Placeholder scan:** No TBDs, TODOs, or incomplete sections.

**3. Type consistency:**
- `EngineOutput` defined in Task 1, used consistently in Tasks 6-12
- `FusionResult` defined in Task 1, used in Tasks 12-13, 19
- `Document` from existing types.py, used throughout
- `TypeLibrary` from Task 5, used in Tasks 9-11, 19
- All engine class names match between individual files and `__init__.py` re-exports
