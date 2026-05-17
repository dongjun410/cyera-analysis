from typing import Dict, List, Tuple

from cyera_bench.datasets.doc_label_base import BaseDocLabelDataset


class LedgarDataset(BaseDocLabelDataset):
    """Ledgar — 100-class legal contract clause classification benchmark.

    Data source: lex_glue/ledgar on HuggingFace.
    Flat labels (no L2). 60K train / 10K test.
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

        ds = load_dataset("lex_glue", "ledgar", split=self._split)
        # Ledgar ClassLabel has .names for int -> str conversion
        label_names = ds.features["label"].names
        texts: List[str] = []
        labels: List[Dict[str, str]] = []

        for item in ds:
            texts.append(item["text"])
            labels.append({"l1": label_names[item["label"]], "l2": ""})

        self._cache = (texts, labels)
        return self._cache
