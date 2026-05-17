from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseDocLabelDataset(ABC):
    """Abstract base for document classification datasets.

    Returns (texts, labels) pairs where labels = [{'l1': ..., 'l2': ...}, ...].
    """

    def __init__(self, data_root: str):
        self._data_root = data_root
        self._cache: Tuple[List[str], List[Dict[str, str]]] | None = None

    @property
    @abstractmethod
    def l1_labels(self) -> List[str]:
        """Unique L1 category display names."""
        ...

    @property
    @abstractmethod
    def l2_labels(self) -> List[str]:
        """Unique L2 category display names."""
        ...

    @abstractmethod
    def load(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """Returns (texts, labels). Labels: [{'l1': ..., 'l2': ...}, ...]."""
        ...

    def texts(self) -> List[str]:
        texts, _ = self.load()
        return texts

    def labels(self) -> List[Dict[str, str]]:
        _, labels = self.load()
        return labels
