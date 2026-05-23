#!/usr/bin/env python
"""Bootstrap E2 template hashes from sample documents.

Pre-computes template hashes for known document types so the E2
engine has matches from day 1, not just an empty library.

Uses the EXACT same PII replacement logic as E2TemplateEngine._replace_pii()
to guarantee hash consistency between bootstrap and runtime.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.knowledge.rules import PII_PATTERNS  # noqa: E402


def replace_pii(text: str) -> tuple[str, int]:
    """Replace PII entities with type placeholders.

    EXACT copy of E2TemplateEngine._replace_pii() logic.
    Must stay in sync with src/engines/e2_template.py.
    """
    count = 0
    matches: list[tuple[int, int, str]] = []
    for pii_type, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), pii_type))

    if not matches:
        return text, 0

    # Sort by start, then by end descending (longest match first)
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

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


# Known sample doc -> correct label mapping
SAMPLE_LABELS: dict[str, str] = {
    "hr_01": "HR & Payroll",
    "hr_02": "HR & Payroll",
    "finance_01": "Financial Report",
    "finance_02": "Financial Report",
    "medical_01": "Medical Record",
    "json_log_01": "API Log",
    "plain_01": "Email / Communication",
}


def main() -> None:
    sample_dir = Path("output/sample_docs")
    if not sample_dir.is_dir():
        print(f"ERROR: {sample_dir} not found. Run from impl-datadna/ directory.")
        sys.exit(1)

    templates: dict[str, dict] = {}

    print("=" * 72)
    print("  Bootstrap E2 Template Hashes")
    print("=" * 72)

    for file_path in sorted(sample_dir.glob("*")):
        doc_id = file_path.stem
        label = SAMPLE_LABELS.get(doc_id)
        if label is None:
            print(f"  SKIP {doc_id:15s} — no label mapping")
            continue

        text = file_path.read_text(encoding="utf-8", errors="replace")
        replaced, pii_count = replace_pii(text)
        template_hash = hashlib.sha256(
            replaced.encode("utf-8")
        ).hexdigest()

        templates[template_hash] = {
            "label": label,
            "doc_id": doc_id,
            "pii_count": pii_count,
        }

        # Show before/after for verification
        print(f"\n  [{doc_id}] -> {label}")
        print(f"    Original : {text.strip()}")
        print(f"    Replaced : {replaced.strip()}")
        print(f"    PII count: {pii_count}")
        print(f"    SHA256   : {template_hash}")

    # Output as JSON
    out_path = Path("output/bootstrap_templates.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(templates, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(templates)} hashes to {out_path}")

    # Generate Python code to paste into templates.py
    print("\n" + "=" * 72)
    print("  Add to _bootstrap_builtin_templates() in")
    print("  src/knowledge/templates.py:")
    print("=" * 72)
    for h, info in templates.items():
        print(
            f'    BUILTIN_TEMPLATES.add(\n'
            f'        "{h}",\n'
            f'        "{info["label"]}", source="builtin",\n'
            f'    )  # {info["doc_id"]}'
        )


if __name__ == "__main__":
    main()
