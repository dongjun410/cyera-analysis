import json
import re
from typing import Dict


def _to_snake(text: str) -> str:
    """Convert a display label to snake_case."""
    t = text.lower()
    t = t.replace("&", "and")
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = t.strip("_")
    t = re.sub(r"_+", "_", t)
    return t


def build_label_mapping(taxonomy_path: str) -> Dict[str, str]:
    """Read file_labels_20_10.json and return {snake_key: display_name} for L1+L2."""
    mapping: Dict[str, str] = {}

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    for category in taxonomy.get("categories", []):
        l1_display = category["name"]
        l1_snake = _to_snake(l1_display)
        mapping[l1_snake] = l1_display

        for file_type in category.get("file_types", []):
            l2_snake = _to_snake(file_type)
            mapping[l2_snake] = file_type

    return mapping


def normalize_label(label: str, mapping: Dict[str, str]) -> str:
    """Normalize a label (snake_case or display) to display name."""
    if not label:
        return ""
    snake = _to_snake(label)
    if snake in mapping:
        return mapping[snake]
    # Try matching without trailing suffixes like _hse, _hr, _it
    parts = snake.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) <= 4 and parts[0] in mapping:
        return mapping[parts[0]]
    return label
