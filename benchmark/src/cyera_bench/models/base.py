from abc import ABC, abstractmethod
from typing import List
from cyera_bench.types import Entity

class BaseModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Full model identifier, e.g. 'google/flan-t5-large'"""
        ...

    @property
    @abstractmethod
    def param_count(self) -> int:
        """Number of parameters"""
        ...

    @abstractmethod
    def predict(self, texts: List[str]) -> List[List[Entity]]:
        """Batch inference: list of texts -> list of entity lists"""
        ...

    def warmup(self, n: int = 10) -> None:
        """Run dummy inference to warm GPU caches."""
        dummy = ["Warmup sentence."] * min(n, 4)
        self.predict(dummy)
