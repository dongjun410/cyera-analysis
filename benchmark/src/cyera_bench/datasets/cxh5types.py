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


class Cxh5typesDataset(BaseDocLabelDataset):
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

        texts_path = os.path.join(self._data_root, "cxh5types", "cxh5types_texts.jsonl")
        labels_path = os.path.join(self._data_root, "cxh5types", "cxh5types_human_labels.json")

        with open(labels_path, "r", encoding="utf-8") as f:
            labels_raw = json.load(f)

        texts_list: List[str] = []
        labels_list: List[Dict[str, str]] = []

        with open(texts_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                filename = item["filename"]
                label_entry = labels_raw.get(filename)
                if label_entry and label_entry.get("l1") and label_entry.get("l2"):
                    texts_list.append(item["text"])
                    labels_list.append({
                        "l1": normalize_label(label_entry["l1"], self._label_map),
                        "l2": normalize_label(label_entry["l2"], self._label_map),
                    })

        self._cache = (texts_list, labels_list)
        return self._cache
