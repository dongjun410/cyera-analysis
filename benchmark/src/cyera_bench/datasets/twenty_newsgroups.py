from typing import Dict, List, Tuple

from cyera_bench.datasets.doc_label_base import BaseDocLabelDataset


class TwentyNewsgroupsDataset(BaseDocLabelDataset):
    """20 Newsgroups — classic 20-class news document classification benchmark.

    Data source: SetFit/20_newsgroups on HuggingFace.
    Flat labels (no L2).
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
        return []

    def load(self) -> Tuple[List[str], List[Dict[str, str]]]:
        if self._cache is not None:
            return self._cache

        from datasets import load_dataset

        ds = load_dataset("SetFit/20_newsgroups", split=self._split)
        texts: List[str] = []
        labels: List[Dict[str, str]] = []

        for item in ds:
            texts.append(item["text"])
            labels.append({"l1": item["label_text"], "l2": ""})

        self._cache = (texts, labels)
        return self._cache
