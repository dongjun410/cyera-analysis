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
    DocTypeRule("FIN_EXPENSE_002", _compile(r"\b(?:expense\s*report.*employee|employee.*expense\s*report)\b"),
                ["MONEY", "CREDIT_CARD"], "Financial Report", 0.80),
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
                ["SSN", "MEDICAL_RECORD", "NPI"], "Medical Record", 0.85),
    DocTypeRule("MED_CLAIM_001", _compile(r"\b(?:Claim\s*#.*Patient|Patient.*Claim\s*#|Amount\s*Billed.*Diagnosis)\b"),
                ["MEDICAL_RECORD", "NPI", "MONEY"], "Medical Record", 0.85),
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
    DocTypeRule("TECH_CHANGELOG_001", _compile(r"\b(?:changelog|release\s*notes?|what'?s\s*new|version\s*\d+\.\d+|fixed\s*(?:a\s+)?bug)\b"),
                [], "Technical Document", 0.65),
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
