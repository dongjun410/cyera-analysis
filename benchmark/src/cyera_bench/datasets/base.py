from abc import ABC, abstractmethod
from typing import List
from datasets import Dataset

class BaseDataset(ABC):
    @property
    @abstractmethod
    def entity_types(self) -> List[str]:
        """Entity types present in this dataset, e.g. ['PER', 'ORG', 'LOC', 'MISC']"""
        ...

    @abstractmethod
    def load(self, split: str = "test") -> Dataset:
        """Load a named split, return a HuggingFace Dataset."""
        ...

    def texts(self, split: str = "test") -> List[str]:
        """Convenience: return raw text list for the split."""
        ds = self.load(split)
        return ds["tokens"]

    def bio_tags(self, split: str = "test") -> List[List[str]]:
        """Return BIO tags for the split."""
        ds = self.load(split)
        return ds["ner_tags"]
