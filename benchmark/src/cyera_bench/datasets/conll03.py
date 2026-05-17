from typing import List
from datasets import Dataset, load_dataset
from cyera_bench.datasets.base import BaseDataset

_ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


class Conll03Dataset(BaseDataset):
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._cache: dict[str, Dataset] = {}

    @property
    def entity_types(self) -> List[str]:
        return ["PER", "ORG", "LOC", "MISC"]

    def load(self, split: str = "test") -> Dataset:
        if split not in self._cache:
            ds = load_dataset("conllpp", split=split, trust_remote_code=True)
            self._cache[split] = ds
        return self._cache[split]

    def bio_tags(self, split: str = "test") -> List[List[str]]:
        ds = self.load(split)
        return [[_ID2LABEL[tid] for tid in tag_ids] for tag_ids in ds["ner_tags"]]
