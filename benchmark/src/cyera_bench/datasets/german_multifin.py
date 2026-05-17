from typing import Dict, List, Tuple

from cyera_bench.datasets.doc_label_base import BaseDocLabelDataset


class GermanMultiFinDataset(BaseDocLabelDataset):
    """German-MultiFin — hierarchical financial document classification.

    Data source: anhaltai/german-multifin on HuggingFace.
    5 L1 categories, 23 L2 subcategories. German-language financial text.
    Multi-label at L2 level; uses the first label for single-label evaluation.
    """

    def __init__(self, data_root: str = "", split: str = "test"):
        super().__init__(data_root)
        self._split = split

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

        from datasets import load_dataset

        ds = load_dataset("anhaltai/german-multifin", split=self._split)
        texts: List[str] = []
        labels: List[Dict[str, str]] = []

        for item in ds:
            texts.append(item["ger_text"])
            l2_list = item["lowlev_labels"]
            labels.append({
                "l1": item["highlev_label"],
                "l2": l2_list[0] if l2_list else "",
            })

        self._cache = (texts, labels)
        return self._cache
