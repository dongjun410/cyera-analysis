"""Tier 0: Built-in PII detection patterns and validators.

Provides 50+ regex-based PII detection patterns with context boost/penalty
terms, and validator functions for checksum/luhn-based verification.
"""

from __future__ import annotations

import re


# ──────────────────────────────────────────────────────────────
# Validator Functions
# ──────────────────────────────────────────────────────────────

def luhn(value: str) -> bool:
    """Luhn algorithm (mod 10) for credit card / IMEI validation.

    Args:
        value: Numeric string (digits only).

    Returns:
        True if the value passes the Luhn checksum.
    """
    digits = value.replace("-", "").replace(" ", "")
    if not digits.isdigit():
        return False
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def is_valid_ssn(value: str) -> bool:
    """Validate US Social Security Number format.

    Rules:
        - Format: ###-##-#### or #########
        - Area (first 3) not in 000, 666, 900-999
        - Group (middle 2) not 00
        - Serial (last 4) not 0000
        - ITIN rule: not 9xx-xx-xxxx

    Args:
        value: SSN string (with or without dashes).

    Returns:
        True if the SSN passes content validation rules.
    """
    flat = value.replace("-", "").strip()
    if len(flat) != 9 or not flat.isdigit():
        return False
    area = flat[:3]
    group = flat[3:5]
    serial = flat[5:9]
    if area in ("000", "666"):
        return False
    if 900 <= int(area) <= 999:
        return False
    if group == "00":
        return False
    if serial == "0000":
        return False
    return True


def is_valid_imei(value: str) -> bool:
    """Validate IMEI number using Luhn checksum.

    IMEI numbers are 15 digits (including check digit).
    IMEISV are 16 digits — the first 14 are the TAC+SNR, the last 2 are SV.

    Args:
        value: IMEI string (digits only, possibly with dashes/spaces).

    Returns:
        True if the value is a valid 15-digit IMEI passing Luhn.
    """
    flat = value.replace("-", "").replace(" ", "")
    if len(flat) != 15 or not flat.isdigit():
        return False
    return luhn(flat)


# ──────────────────────────────────────────────────────────────
# Validator Registry
# ──────────────────────────────────────────────────────────────

VALIDATORS: dict[str, callable] = {
    "luhn": luhn,
    "is_valid_ssn": is_valid_ssn,
    "is_valid_imei": is_valid_imei,
}


# ──────────────────────────────────────────────────────────────
# Shared Context Terms
# ──────────────────────────────────────────────────────────────

PENALTY_TERMS: list[str] = [
    "test", "sample", "example", "TODO", "dummy",
    "placeholder", "xxxx", "foo", "bar",
]

BOOST_TERMS: list[str] = [
    "SSN:", "社会安全号", "信用卡号", "passport",
    "account number", "customer ID", "patient", "employee",
    "confidential", "sensitive", "PII", "secret",
    "credential", "token", "authorization",
]


# ──────────────────────────────────────────────────────────────
# Built-in Patterns (50+)
# ──────────────────────────────────────────────────────────────

BUILTIN_PATTERNS: list[dict] = [
    # ── US Social Security Number ──
    {
        "entity_type": "SSN",
        "regex": r"\b(?!000|666|9\d{2})(\d{3})[- ]?(?!00)(\d{2})[- ]?(?!0000)(\d{4})\b",
        "validation": "is_valid_ssn",
        "context_boost_terms": ["SSN", "Social Security", "社会安全号"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── Credit Cards ──
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "信用卡号", "Visa"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "信用卡号", "MasterCard", "Mastercard"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b2(?:2[2-9]\d|2[3-6]\d{2}|27[01]\d|2720)[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "信用卡号", "MasterCard", "Mastercard"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b3[47]\d{2}[- ]?\d{6}[- ]?\d{5}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "信用卡号", "Amex", "American Express"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b(?:6011|65\d{2}|64[4-9]\d|622(?:1[2-9]\d|[2-8]\d{2}|9[01]\d|92[0-5]))[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "Discover"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b3(?:0[0-5]|[68]\d)\d{2}[- ]?\d{4}[- ]?\d{4}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "Diners Club"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },
    {
        "entity_type": "CREDIT_CARD",
        "regex": r"\b(?:2131|1800)[- ]?\d{4}[- ]?\d{4}[- ]?\d{3}\b",
        "validation": "luhn",
        "context_boost_terms": ["credit card", "card number", "JCB"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Email Address ──
    {
        "entity_type": "EMAIL",
        "regex": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "validation": None,
        "context_boost_terms": ["email", "e-mail", "contact"],
        "context_penalty_terms": PENALTY_TERMS + ["@example", "@test", "@sample"],
        "min_confidence": 0.85,
    },

    # ── Phone Numbers ──
    {
        "entity_type": "PHONE_US",
        "regex": r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "validation": None,
        "context_boost_terms": ["phone", "tel", "mobile", "contact number", "call"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },
    {
        "entity_type": "PHONE_CN",
        "regex": r"\b(?:\+?86)?[-.\s]?1[3-9]\d{9}\b",
        "validation": None,
        "context_boost_terms": ["手机", "电话", "联系方式", "phone", "mobile"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },
    {
        "entity_type": "PHONE_UK",
        "regex": r"(?<!\w)(?:\+44[-.\s]?|0)[1-9]\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
        "validation": None,
        "context_boost_terms": ["phone", "tel", "mobile", "contact"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },
    {
        "entity_type": "PHONE_DE",
        "regex": r"(?<!\w)(?:\+49[-.\s]?|0)(?:1[567]\d{1,2}|[2-9]\d{1,4})[-.\s]?\d{3,8}\b",
        "validation": None,
        "context_boost_terms": ["Telefon", "Handy", "Mobil", "phone", "Rufnummer"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── IBAN ──
    {
        "entity_type": "IBAN",
        "regex": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        "validation": None,
        "context_boost_terms": ["IBAN", "International Bank Account Number", "account", "银行账户"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── SWIFT/BIC ──
    {
        "entity_type": "SWIFT_BIC",
        "regex": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
        "validation": None,
        "context_boost_terms": ["SWIFT", "BIC", "routing", "wire transfer", "银行代码"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Passport Numbers ──
    {
        "entity_type": "PASSPORT_US",
        "regex": r"\b[A-Z0-9]{6,9}\b",
        "validation": None,
        "context_boost_terms": ["passport", "travel document", "护照", "US passport"],
        "context_penalty_terms": PENALTY_TERMS + ["reference", "tracking"],
        "min_confidence": 0.60,
    },
    {
        "entity_type": "PASSPORT_CN",
        "regex": r"\b[EGP]\d{8}\b",
        "validation": None,
        "context_boost_terms": ["护照", "passport", "中国护照", "旅行证件"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── Driver's License ──
    {
        "entity_type": "DRIVERS_LICENSE_US",
        "regex": r"\b[A-Z]{1,2}\d{6,8}\b",
        "validation": None,
        "context_boost_terms": ["driver license", "driver's license", "DL", "驾照", "driving"],
        "context_penalty_terms": PENALTY_TERMS + ["invoice", "order", "tracking"],
        "min_confidence": 0.60,
    },
    {
        "entity_type": "DRIVERS_LICENSE_UK",
        "regex": r"\b[A-Z]{5}\d{6}[A-Z0-9]{2}\d[A-Z]{2}\b",
        "validation": None,
        "context_boost_terms": ["driver", "driving licence", "DVLA", "驾照"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── IP Addresses ──
    {
        "entity_type": "IPV4",
        "regex": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "validation": None,
        "context_boost_terms": ["IP", "address", "host", "server"],
        "context_penalty_terms": PENALTY_TERMS + ["version", "0.0.0.0", "127.0.0.1", "255.255", "192.168"],
        "min_confidence": 0.75,
    },
    {
        "entity_type": "IPV6",
        "regex": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        "validation": None,
        "context_boost_terms": ["IP", "address", "IPv6", "host"],
        "context_penalty_terms": PENALTY_TERMS + ["::1", "loopback"],
        "min_confidence": 0.80,
    },

    # ── MAC Address ──
    {
        "entity_type": "MAC_ADDRESS",
        "regex": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        "validation": None,
        "context_boost_terms": ["MAC", "hardware address", "physical address", "network"],
        "context_penalty_terms": PENALTY_TERMS + ["00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"],
        "min_confidence": 0.70,
    },

    # ── Date of Birth ──
    {
        "entity_type": "DATE_OF_BIRTH",
        "regex": r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b",
        "validation": None,
        "context_boost_terms": ["DOB", "birth date", "出生日期", "date of birth", "birthday", "born"],
        "context_penalty_terms": PENALTY_TERMS + ["created", "updated", "modified", "published", "expires", "valid until"],
        "min_confidence": 0.65,
    },
    {
        "entity_type": "DATE_OF_BIRTH",
        "regex": r"\b(?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])[-/.](?:19|20)\d{2}\b",
        "validation": None,
        "context_boost_terms": ["DOB", "birth date", "出生日期", "date of birth", "birthday", "born"],
        "context_penalty_terms": PENALTY_TERMS + ["created", "updated", "modified", "published", "expires", "valid until"],
        "min_confidence": 0.65,
    },

    # ── China ID Card Number (18-digit) ──
    {
        "entity_type": "CHINA_ID_CARD",
        "regex": r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
        "validation": None,
        "context_boost_terms": ["身份证", "身份证号", "公民身份号码", "ID card", "Chinese ID", "居民身份证"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Bank Account (US Routing + Account) ──
    {
        "entity_type": "US_ROUTING_NUMBER",
        "regex": r"\b(?:0\d|1[0-2]|2[1-9]|3[0-2]|6[1-9]|7[0-2]|80)\d{7}\b",
        "validation": None,
        "context_boost_terms": ["routing number", "ABA", "transit", "wire routing"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },
    {
        "entity_type": "US_BANK_ACCOUNT",
        "regex": r"\b\d{8,17}\b",
        "validation": None,
        "context_boost_terms": ["account number", "bank account", "checking", "savings", "账号"],
        "context_penalty_terms": PENALTY_TERMS + ["order", "invoice", "reference", "tracking", "phone"],
        "min_confidence": 0.55,
    },

    # ── Medical Record Number ──
    {
        "entity_type": "MEDICAL_RECORD",
        "regex": r"\bMRN[:\-]?\s*\d{5,12}\b",
        "validation": None,
        "context_boost_terms": ["MRN", "medical record", "病历号", "patient ID", "medical", "clinical", "hospital"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── NPI (National Provider Identifier) ──
    {
        "entity_type": "NPI",
        "regex": r"\b[1-9]\d{9}\b",
        "validation": None,
        "context_boost_terms": ["NPI", "National Provider Identifier", "provider", "physician", "clinician"],
        "context_penalty_terms": PENALTY_TERMS + ["phone", "order", "invoice"],
        "min_confidence": 0.60,
    },

    # ── DEA Number ──
    {
        "entity_type": "DEA_NUMBER",
        "regex": r"\b[A-Z]{2}\d{7}\b",
        "validation": None,
        "context_boost_terms": ["DEA", "Drug Enforcement", "controlled substance", "prescriber"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── IMEI ──
    {
        "entity_type": "IMEI",
        "regex": r"\b\d{15}\b",
        "validation": "is_valid_imei",
        "context_boost_terms": ["IMEI", "device ID", "手机串号", "serial"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── API Key (generic) ──
    {
        "entity_type": "API_KEY",
        "regex": r"\b[aA][pP][iI][-_]?[kK][eE][yY][\s:=]+['\"]?([A-Za-z0-9+/=_\-]{20,60})['\"]?",
        "validation": None,
        "context_boost_terms": ["api", "key", "token", "auth", "credential"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── AWS Access Key ID ──
    {
        "entity_type": "AWS_ACCESS_KEY",
        "regex": r"\bAKIA[0-9A-Z]{16}\b",
        "validation": None,
        "context_boost_terms": ["AWS", "access key", "ACCESS_KEY_ID", "amazon"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.90,
    },

    # ── AWS Secret Key ──
    {
        "entity_type": "AWS_SECRET_KEY",
        "regex": r"\b(?:aws[_\-]?(?:secret|private)[_\-]?key)[\s:=]+['\"]?([A-Za-z0-9+/]{40})['\"]?",
        "validation": None,
        "context_boost_terms": ["AWS", "secret", "SECRET_ACCESS_KEY", "credential"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Private Key Header ──
    {
        "entity_type": "PRIVATE_KEY",
        "regex": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "validation": None,
        "context_boost_terms": ["private key", "certificate", "TLS", "SSL", "秘钥"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.95,
    },

    # ── JWT Token ──
    {
        "entity_type": "JWT_TOKEN",
        "regex": r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b",
        "validation": None,
        "context_boost_terms": ["JWT", "token", "bearer", "Authorization", "authentication"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Base64 Encoded Data (long sequences, potential secrets) ──
    {
        "entity_type": "BASE64_LONG",
        "regex": r"\b[A-Za-z0-9+/]{60,}={0,2}\b",
        "validation": None,
        "context_boost_terms": ["encoded", "base64", "payload", "data"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.50,
    },

    # ── URL with Embedded Credentials ──
    {
        "entity_type": "URL_CREDENTIALS",
        "regex": r"\b[a-zA-Z][a-zA-Z0-9+\-.]*://[^:/\s]+:[^@\s]+@[^\s]+\b",
        "validation": None,
        "context_boost_terms": ["connection", "database", "url", "credential"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Generic Credentials in Config ──
    {
        "entity_type": "CONFIG_CREDENTIAL",
        "regex": r"(?:password|passwd|pwd|secret|credential)[\s:=]+['\"]?(?![*\s]+$)([^\s'\"]{4,})['\"]?",
        "validation": None,
        "context_boost_terms": ["config", "credential", "password", "secret", "env"],
        "context_penalty_terms": PENALTY_TERMS + ["****", "changeme", "yourpassword"],
        "min_confidence": 0.70,
    },

    # ── Google API Key ──
    {
        "entity_type": "GOOGLE_API_KEY",
        "regex": r"\bAIza[0-9A-Za-z\-_]{35}\b",
        "validation": None,
        "context_boost_terms": ["google", "api", "GCP", "cloud", "firebase"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.90,
    },

    # ── GitHub Token ──
    {
        "entity_type": "GITHUB_TOKEN",
        "regex": r"\b(?:gh[pousr]_[A-Za-z0-9_]{36,}|github[_\-]?token[\s:=]+['\"]?([A-Za-z0-9_]{40})['\"]?)",
        "validation": None,
        "context_boost_terms": ["github", "token", "git", "repository", "repo"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Slack Token ──
    {
        "entity_type": "SLACK_TOKEN",
        "regex": r"\bxox[bpsar]\-[A-Za-z0-9\-]{10,60}\b",
        "validation": None,
        "context_boost_terms": ["slack", "token", "webhook", "bot"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Stripe API Key ──
    {
        "entity_type": "STRIPE_KEY",
        "regex": r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b",
        "validation": None,
        "context_boost_terms": ["stripe", "payment", "api", "key"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Twilio Auth Token ──
    {
        "entity_type": "TWILIO_TOKEN",
        "regex": r"\bSK[0-9a-fA-F]{32}\b",
        "validation": None,
        "context_boost_terms": ["twilio", "auth", "token", "SMS", "phone"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Generic OAuth Token ──
    {
        "entity_type": "OAUTH_TOKEN",
        "regex": r"\b(?:ya29\.[A-Za-z0-9\-_]+|ya29\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+)\b",
        "validation": None,
        "context_boost_terms": ["OAuth", "token", "refresh", "access", "bearer"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Bitcoin Address ──
    {
        "entity_type": "BITCOIN_ADDRESS",
        "regex": r"\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59})\b",
        "validation": None,
        "context_boost_terms": ["bitcoin", "BTC", "wallet", "crypto", "加密"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Ethereum Address ──
    {
        "entity_type": "ETHEREUM_ADDRESS",
        "regex": r"\b0x[a-fA-F0-9]{40}\b",
        "validation": None,
        "context_boost_terms": ["ethereum", "ETH", "wallet", "crypto", "blockchain"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── GPS Coordinates ──
    {
        "entity_type": "GPS_COORDINATES",
        "regex": r"\b[-+]?(?:180(?:\.0+)?|(?:\d{1,2}|1[0-7]\d)(?:\.\d+)?)[,;\s]+[-+]?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)\b",
        "validation": None,
        "context_boost_terms": ["GPS", "latitude", "longitude", "coordinates", "location", "geolocation"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.65,
    },

    # ── Vehicle VIN ──
    {
        "entity_type": "VIN",
        "regex": r"\b[A-HJ-NPR-Z0-9]{17}\b",
        "validation": None,
        "context_boost_terms": ["VIN", "vehicle identification", "chassis number", "车架号"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── ISBN ──
    {
        "entity_type": "ISBN",
        "regex": r"\b(?:ISBN[-: ]?)?(?:97[89][- ]?)?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?\d\b",
        "validation": None,
        "context_boost_terms": ["ISBN", "book", "publication", "publisher"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── Database Connection String ──
    {
        "entity_type": "DB_CONNECTION_STRING",
        "regex": r"\b(?:jdbc|mongodb|mysql|postgres|postgresql|mssql|oracle|sqlite|redis)://[^\s]+\b",
        "validation": None,
        "context_boost_terms": ["connection", "database", "db", "datasource"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Azure Storage Key ──
    {
        "entity_type": "AZURE_STORAGE_KEY",
        "regex": r"\b(?:DefaultEndpointsProtocol|AccountName|AccountKey)[\s:=]+['\"]?([^\s'\"]{5,})['\"]?",
        "validation": None,
        "context_boost_terms": ["Azure", "storage", "blob", "connection string", "account"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── Generic Secret / Token ──
    {
        "entity_type": "SECRET_TOKEN",
        "regex": r"\b(?:secret|token|key)[\s:=]+['\"]?([A-Za-z0-9+/=_-]{20,60})['\"]?",
        "validation": None,
        "context_boost_terms": ["secret", "token", "key", "auth", "credential"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.70,
    },

    # ── CPF (Brazil) ──
    {
        "entity_type": "CPF",
        "regex": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
        "validation": None,
        "context_boost_terms": ["CPF", "Brazil", "tax ID", "cadastro"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── UK National Insurance Number (NINO) ──
    {
        "entity_type": "NINO",
        "regex": r"\b[A-CEGHJ-PR-TW-Z]{2}[0-9]{2}[0-9]{2}[0-9]{2}[A-D]\b",
        "validation": None,
        "context_boost_terms": ["National Insurance", "NINO", "NI number", "HMRC"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── Canadian SIN ──
    {
        "entity_type": "SIN_CANADA",
        "regex": r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b",
        "validation": None,
        "context_boost_terms": ["SIN", "Social Insurance Number", "Canada", "Canadian"],
        "context_penalty_terms": PENALTY_TERMS + ["phone", "fax"],
        "min_confidence": 0.60,
    },

    # ── Generic Certificate ──
    {
        "entity_type": "CERTIFICATE",
        "regex": r"-----BEGIN CERTIFICATE-----",
        "validation": None,
        "context_boost_terms": ["certificate", "TLS", "SSL", "X509", "证书"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.95,
    },

    # ── RSA Public Key ──
    {
        "entity_type": "PUBLIC_KEY",
        "regex": r"-----BEGIN (?:RSA |EC )?PUBLIC KEY-----",
        "validation": None,
        "context_boost_terms": ["public key", "RSA", "encryption", "certificate"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.95,
    },

    # ── PGP Message / Key ──
    {
        "entity_type": "PGP_KEY",
        "regex": r"-----BEGIN PGP (?:PUBLIC|PRIVATE|MESSAGE|SIGNED MESSAGE)-----",
        "validation": None,
        "context_boost_terms": ["PGP", "GPG", "encryption", "key", "签名"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.95,
    },

    # ── AWS ARN ──
    {
        "entity_type": "AWS_ARN",
        "regex": r"\barn:(?:aws|aws-cn|aws-us-gov):[a-zA-Z0-9\-]+:[a-zA-Z0-9\-]*:\d{12}:[^\s]+\b",
        "validation": None,
        "context_boost_terms": ["ARN", "AWS", "resource", "amazon"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.80,
    },

    # ── SSH Private Key ──
    {
        "entity_type": "SSH_PRIVATE_KEY",
        "regex": r"-----BEGIN OPENSSH PRIVATE KEY-----",
        "validation": None,
        "context_boost_terms": ["SSH", "private key", "key pair", "authorized_keys"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.95,
    },

    # ── Heroku API Key ──
    {
        "entity_type": "HEROKU_KEY",
        "regex": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "validation": None,
        "context_boost_terms": ["heroku", "api", "key", "token"],
        "context_penalty_terms": PENALTY_TERMS + ["uuid", "guid", "identifier"],
        "min_confidence": 0.60,
    },

    # ── LinkedIn Token ──
    {
        "entity_type": "LINKEDIN_TOKEN",
        "regex": r"\b(?:linkedin|LI)[_\-]?(?:api[_\-]?key|token|secret)[\s:=]+['\"]?([A-Za-z0-9]{20,})['\"]?",
        "validation": None,
        "context_boost_terms": ["LinkedIn", "api", "token", "OAuth"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.75,
    },

    # ── PayPal/Braintree Token ──
    {
        "entity_type": "PAYPAL_TOKEN",
        "regex": r"\b(?:access_token\$production\$[a-z0-9]{16}\$[a-f0-9]{32})\b",
        "validation": None,
        "context_boost_terms": ["PayPal", "Braintree", "payment", "token", "merchant"],
        "context_penalty_terms": PENALTY_TERMS,
        "min_confidence": 0.85,
    },

    # ── Age ──
    {
        "entity_type": "AGE",
        "regex": r"\b[Aa]ge[:\s=]+(\d{1,3})\b",
        "validation": None,
        "context_boost_terms": ["age", "patient age", "年龄", "years old", "DOB"],
        "context_penalty_terms": PENALTY_TERMS + ["contract", "warranty"],
        "min_confidence": 0.55,
    },

    # ── Health Insurance Claim Number ──
    {
        "entity_type": "HICN",
        "regex": r"\b[A-Za-z0-9]{5,12}\b",
        "validation": None,
        "context_boost_terms": ["HICN", "Health Insurance Claim", "Medicare", "insurance", "claim", "医保"],
        "context_penalty_terms": PENALTY_TERMS + ["order", "tracking", "invoice"],
        "min_confidence": 0.55,
    },
]


# ──────────────────────────────────────────────────────────────
# Module-level verification
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"BUILTIN_PATTERNS: {len(BUILTIN_PATTERNS)} patterns")
    print(f"VALIDATORS: {len(VALIDATORS)} validators")
    for name in sorted(VALIDATORS):
        print(f"  - {name}")
    entity_types = sorted(set(p["entity_type"] for p in BUILTIN_PATTERNS))
    print(f"Unique entity types: {len(entity_types)}")
    for et in entity_types:
        count = sum(1 for p in BUILTIN_PATTERNS if p["entity_type"] == et)
        print(f"  - {et}: {count}")
