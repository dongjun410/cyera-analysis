from typing import List
from datasets import Dataset, load_dataset
from cyera_bench.datasets.base import BaseDataset

class PiiMaskingDataset(BaseDataset):
    """Loader for ai4privacy/pii-masking-300k — real-world PII detection benchmark."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._cache: dict[str, Dataset] = {}

    @property
    def entity_types(self) -> List[str]:
        return [
            "PERSON", "EMAIL", "PHONE", "STREET_ADDRESS", "CITY", "STATE",
            "ZIP_CODE", "DATE_OF_BIRTH", "AGE", "ID_CARD", "PASSPORT",
            "DRIVERS_LICENSE", "SSN", "CREDIT_CARD", "BANK_ACCOUNT",
            "IP_ADDRESS", "URL",
        ]

    def load(self, split: str = "test") -> Dataset:
        if split not in self._cache:
            ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
            split_ds = ds.train_test_split(test_size=0.2, seed=self._seed)
            self._cache["train"] = split_ds["train"]
            self._cache["test"] = split_ds["test"]
        return self._cache[split]

    def texts(self, split: str = "test") -> List[str]:
        ds = self.load(split)
        return ds["source_text"]
