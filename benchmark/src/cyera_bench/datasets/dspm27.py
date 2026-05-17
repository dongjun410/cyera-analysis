import json
import os
from typing import Dict, List, Tuple

from cyera_bench.datasets.doc_label_base import BaseDocLabelDataset
from cyera_bench.utils.label_mapping import build_label_mapping, normalize_label

_TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    "ZerosOne", "gemma-doc-label", "config", "file_labels_20_10.json",
)

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")


def _extract_pdf_text(pdf_path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except ImportError:
        raise ImportError(
            "pdfplumber not installed. Install with: pip install pdfplumber"
        )


class Dspm27Dataset(BaseDocLabelDataset):
    def __init__(self, data_root: str = ""):
        if not data_root:
            data_root = os.path.join(
                _BASE, "ZerosOne", "gemma-doc-label", "testdata"
            )
        super().__init__(data_root)
        self._label_map = build_label_mapping(_TAXONOMY_PATH)

    @property
    def l1_labels(self) -> List[str]:
        _, labels = self.load()
        return sorted(set(l["l1"] for l in labels if l["l1"]))

    @property
    def l2_labels(self) -> List[str]:
        _, labels = self.load()
        return sorted(set(l["l2"] for l in labels if l["l2"]))

    def load(self) -> Tuple[List[str], List[Dict[str, str]]]:
        if self._cache is not None:
            return self._cache

        labels_path = os.path.join(self._data_root, "dspm27", "dspm_gpt52_labels.json")
        pdfs_dir = os.path.join(self._data_root, "dspm27", "pdfs")
        texts_cache_path = os.path.join(self._data_root, "dspm27", "dspm27_texts.jsonl")

        with open(labels_path, "r", encoding="utf-8") as f:
            labels_raw = json.load(f)

        # Check for pre-extracted text cache
        texts_cache: Dict[str, str] = {}
        if os.path.exists(texts_cache_path):
            with open(texts_cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    texts_cache[item["filename"]] = item["text"]

        texts_list: List[str] = []
        labels_list: List[Dict[str, str]] = []

        for filename, entry in labels_raw.items():
            # Skip entries with missing labels
            if entry.get("l1") is None or entry.get("l2") is None:
                continue

            if filename in texts_cache:
                text = texts_cache[filename]
            else:
                pdf_path = os.path.join(pdfs_dir, filename)
                if not os.path.exists(pdf_path):
                    continue
                text = _extract_pdf_text(pdf_path)

            if not text or not text.strip():
                continue

            texts_list.append(text)
            labels_list.append({
                "l1": normalize_label(entry["l1"], self._label_map),
                "l2": normalize_label(entry["l2"], self._label_map),
            })

        self._cache = (texts_list, labels_list)
        return self._cache
