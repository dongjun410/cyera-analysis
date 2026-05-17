# FLAN-T5 NER/PII Benchmark Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable CLI + YAML-driven benchmark framework that evaluates FLAN-T5 variants on NER/PII classification tasks with precision/recall/F1 + throughput/latency metrics, outputting terminal tables, markdown, and JSON reports.

**Architecture:** Three-layer plugin architecture — Models (BaseModel → FlanT5Model), Datasets (BaseDataset → Conll03/PiiMasking/SyntheticPii), Metrics (MetricsCalculator). An Orchestrator reads YAML experiment configs, wires layers together, runs two-phase evaluation (accuracy then throughput sweep), and hands results to a multi-format Reporter.

**Tech Stack:** Python 3.11, PyTorch 2.5+, HuggingFace transformers/datasets/evaluate, seqeval, bitsandbytes, PyYAML, rich (terminal tables)

---

### Task 1: Project scaffolding

**Files:**
- Create: `benchmark/pyproject.toml`
- Create: `benchmark/src/cyera_bench/__init__.py`
- Create: `benchmark/src/cyera_bench/models/__init__.py`
- Create: `benchmark/src/cyera_bench/datasets/__init__.py`
- Create: `benchmark/src/cyera_bench/metrics/__init__.py`
- Create: `benchmark/results/.gitkeep`
- Create: `benchmark/tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p benchmark/src/cyera_bench/models
mkdir -p benchmark/src/cyera_bench/datasets
mkdir -p benchmark/src/cyera_bench/metrics
mkdir -p benchmark/config/experiments
mkdir -p benchmark/results
mkdir -p benchmark/tests
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "cyera-bench"
version = "0.1.0"
description = "FLAN-T5 NER/PII benchmark framework — validates Cyera DataDNA technical claims"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.5.0",
    "transformers>=4.46.0",
    "datasets>=3.0.0",
    "evaluate>=0.4.0",
    "seqeval>=1.2.0",
    "accelerate>=1.0.0",
    "bitsandbytes>=0.44.0",
    "pyyaml>=6.0",
    "rich>=13.0",
    "numpy>=1.26",
    "scikit-learn>=1.5",
]

[project.scripts]
cyera-bench = "cyera_bench.__main__:main"

[tool.setuptools.package-dir]
"" = "src"
```

- [ ] **Step 3: Create empty __init__.py files**

```bash
touch benchmark/src/cyera_bench/__init__.py
touch benchmark/src/cyera_bench/models/__init__.py
touch benchmark/src/cyera_bench/datasets/__init__.py
touch benchmark/src/cyera_bench/metrics/__init__.py
touch benchmark/tests/__init__.py
touch benchmark/results/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/
git commit -m "feat: scaffold benchmark project structure"
```

---

### Task 2: Core types — Entity and BenchmarkResult

**Files:**
- Create: `benchmark/src/cyera_bench/types.py`
- Create: `benchmark/tests/test_types.py`

- [ ] **Step 1: Write failing tests for Entity and BenchmarkResult**

```python
# benchmark/tests/test_types.py
import pytest
from cyera_bench.types import Entity, BenchmarkResult

def test_entity_creation():
    e = Entity(type="PER", text="John", start=0, end=4, confidence=0.95)
    assert e.type == "PER"
    assert e.text == "John"
    assert e.start == 0
    assert e.end == 4
    assert e.confidence == 0.95

def test_entity_default_confidence():
    e = Entity(type="ORG", text="Acme", start=10, end=14)
    assert e.confidence == 1.0

def test_benchmark_result_creation():
    br = BenchmarkResult(
        experiment_name="test-exp",
        model_name="flan-t5",
        model_variant="large",
        dataset_name="conll03",
        per_entity_metrics={"PER": {"precision": 0.95, "recall": 0.93, "f1": 0.94}},
        macro_f1=0.94,
        throughput_tokens_per_sec=342.7,
        latency_p50_ms=23.0,
        latency_p95_ms=45.0,
        latency_p99_ms=78.0,
        gpu_memory_peak_gb=4.2,
        total_samples=3453,
        total_time_sec=12.5,
    )
    assert br.macro_f1 == 0.94
    assert br.per_entity_metrics["PER"]["f1"] == 0.94
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd benchmark && python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cyera_bench'`

- [ ] **Step 3: Write types.py**

```python
# benchmark/src/cyera_bench/types.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Entity:
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0

@dataclass
class BenchmarkResult:
    experiment_name: str
    model_name: str
    model_variant: str
    dataset_name: str
    per_entity_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    gpu_memory_peak_gb: float = 0.0
    total_samples: int = 0
    total_time_sec: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd benchmark && python -m pytest tests/test_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmark/src/cyera_bench/types.py benchmark/tests/test_types.py
git commit -m "feat: add Entity and BenchmarkResult core types"
```

---

### Task 3: BaseModel abstract interface

**Files:**
- Create: `benchmark/src/cyera_bench/models/base.py`

- [ ] **Step 1: Write BaseModel**

```python
# benchmark/src/cyera_bench/models/base.py
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
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/src/cyera_bench/models/base.py
git commit -m "feat: add BaseModel abstract interface"
```

---

### Task 4: FlanT5Model implementation

**Files:**
- Create: `benchmark/src/cyera_bench/models/flan_t5.py`
- Create: `benchmark/tests/test_flan_t5.py`

- [ ] **Step 1: Write failing test**

```python
# benchmark/tests/test_flan_t5.py
import pytest
from cyera_bench.models.flan_t5 import FlanT5Model
from cyera_bench.types import Entity

@pytest.fixture
def model():
    return FlanT5Model(variant="base", device="cpu")

def test_model_name(model):
    assert model.name == "google/flan-t5-base"
    assert model.param_count == 250_000_000

def test_model_variants():
    variants = {
        "small": ("google/flan-t5-small", 77_000_000),
        "base": ("google/flan-t5-base", 250_000_000),
        "large": ("google/flan-t5-large", 780_000_000),
        "xl": ("google/flan-t5-xl", 3_000_000_000),
    }
    for variant, (expected_name, expected_params) in variants.items():
        m = FlanT5Model(variant=variant, device="cpu")
        assert m.name == expected_name
        assert m.param_count == expected_params

def test_predict_returns_list_of_entity_lists(model):
    texts = ["John works at Google in New York.", "Mary visited Paris."]
    results = model.predict(texts)
    assert len(results) == 2
    assert all(isinstance(e, Entity) for e in results[0])

def test_predict_empty_texts(model):
    results = model.predict([])
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd benchmark && python -m pytest tests/test_flan_t5.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write FlanT5Model**

```python
# benchmark/src/cyera_bench/models/flan_t5.py
from typing import List, Dict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from cyera_bench.models.base import BaseModel
from cyera_bench.types import Entity

_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "small": {"hf_name": "google/flan-t5-small",  "ner_checkpoint": "pepegiallo/flan-t5-small_ner"},
    "base":  {"hf_name": "google/flan-t5-base",   "ner_checkpoint": "pepegiallo/flan-t5-base_ner"},
    "large": {"hf_name": "google/flan-t5-large",  "ner_checkpoint": None},
    "xl":    {"hf_name": "google/flan-t5-xl",     "ner_checkpoint": None},
}

_PARAM_COUNTS: Dict[str, int] = {
    "small": 77_000_000,
    "base":  250_000_000,
    "large": 780_000_000,
    "xl":    3_000_000_000,
}

class FlanT5Model(BaseModel):
    def __init__(self, variant: str = "large", device: str = "cuda",
                 quantization: str | None = None, ner_checkpoint: str | None = None):
        if variant not in _MODEL_MAP:
            raise ValueError(f"Unknown variant '{variant}'. Choose: {list(_MODEL_MAP.keys())}")
        self._variant = variant
        self._device = device
        self._quantization = quantization
        info = _MODEL_MAP[variant]

        checkpoint = ner_checkpoint or info["ner_checkpoint"]

        if checkpoint:
            model_kwargs = {"device_map": device} if device == "cuda" else {}
            if quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            self._pipe = pipeline(
                "token-classification",
                model=checkpoint,
                aggregation_strategy="simple",
                device=0 if device == "cuda" else -1,
                **model_kwargs,
            )
            self._mode = "token-classification"
        else:
            self._pipe = pipeline(
                "text2text-generation",
                model=info["hf_name"],
                device=0 if device == "cuda" else -1,
            )
            self._mode = "text2text-generation"

    @property
    def name(self) -> str:
        return _MODEL_MAP[self._variant]["hf_name"]

    @property
    def param_count(self) -> int:
        return _PARAM_COUNTS[self._variant]

    def predict(self, texts: List[str]) -> List[List[Entity]]:
        if not texts:
            return []

        if self._mode == "token-classification":
            return self._predict_token_classification(texts)
        else:
            return self._predict_text2text(texts)

    def _predict_token_classification(self, texts: List[str]) -> List[List[Entity]]:
        results = self._pipe(texts)
        entity_lists: List[List[Entity]] = []

        for per_text_result in results:
            entities = []
            for item in per_text_result:
                entities.append(Entity(
                    type=item["entity_group"],
                    text=item["word"],
                    start=item["start"],
                    end=item["end"],
                    confidence=item["score"],
                ))
            entity_lists.append(entities)

        return entity_lists

    def _predict_text2text(self, texts: List[str]) -> List[List[Entity]]:
        prompts = [f"Extract named entities (person, organization, location, miscellaneous) from the text:\n{t}"
                   for t in texts]
        outputs = self._pipe(prompts, max_new_tokens=128)
        entity_lists: List[List[Entity]] = []

        for output in outputs:
            entities = self._parse_text2text_output(output["generated_text"])
            entity_lists.append(entities)

        return entity_lists

    def _parse_text2text_output(self, text: str) -> List[Entity]:
        entities = []
        tag_map = {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}
        for line in text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                tag, _, value = line.partition(":")
                tag = tag.strip().upper()
                value = value.strip()
                if tag in tag_map and value:
                    entities.append(Entity(type=tag_map[tag], text=value, start=0, end=0, confidence=0.8))
        return entities
```

- [ ] **Step 4: Run tests to verify**

Due to model download requirements, skip CPU-only full model test for now. Run only the variant name test:

```bash
cd benchmark && python -m pytest tests/test_flan_t5.py::test_model_variants -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/src/cyera_bench/models/flan_t5.py benchmark/tests/test_flan_t5.py
git commit -m "feat: add FlanT5Model with token-classification and text2text modes"
```

---

### Task 5: BaseDataset abstract interface

**Files:**
- Create: `benchmark/src/cyera_bench/datasets/base.py`

- [ ] **Step 1: Write BaseDataset**

```python
# benchmark/src/cyera_bench/datasets/base.py
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
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/src/cyera_bench/datasets/base.py
git commit -m "feat: add BaseDataset abstract interface"
```

---

### Task 6: Conll03Dataset

**Files:**
- Create: `benchmark/src/cyera_bench/datasets/conll03.py`
- Create: `benchmark/tests/test_conll03.py`

- [ ] **Step 1: Write failing test**

```python
# benchmark/tests/test_conll03.py
import pytest
from cyera_bench.datasets.conll03 import Conll03Dataset

def test_entity_types():
    ds = Conll03Dataset()
    assert set(ds.entity_types) == {"PER", "ORG", "LOC", "MISC"}

def test_load_test_split():
    ds = Conll03Dataset()
    dataset = ds.load("test")
    assert len(dataset) > 0
    assert "tokens" in dataset.features
    assert "ner_tags" in dataset.features

def test_load_validation_split():
    ds = Conll03Dataset()
    dataset = ds.load("validation")
    assert len(dataset) > 0

def test_texts_method():
    ds = Conll03Dataset()
    texts = ds.texts("test")
    assert isinstance(texts, list)
    assert len(texts) > 0
    assert isinstance(texts[0], list)

def test_bio_tags_method():
    ds = Conll03Dataset()
    tags = ds.bio_tags("test")
    assert isinstance(tags, list)
    assert len(tags) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd benchmark && python -m pytest tests/test_conll03.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write Conll03Dataset**

```python
# benchmark/src/cyera_bench/datasets/conll03.py
from typing import List
from datasets import Dataset, load_dataset

from cyera_bench.datasets.base import BaseDataset

_ID2LABEL = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
             5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

class Conll03Dataset(BaseDataset):
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._cache: dict[str, Dataset] = {}

    @property
    def entity_types(self) -> List[str]:
        return ["PER", "ORG", "LOC", "MISC"]

    def load(self, split: str = "test") -> Dataset:
        if split not in self._cache:
            ds = load_dataset("conll2003", trust_remote_code=True, split=split)
            self._cache[split] = ds
        return self._cache[split]

    def bio_tags(self, split: str = "test") -> List[List[str]]:
        ds = self.load(split)
        return [[_ID2LABEL[tid] for tid in tag_ids] for tag_ids in ds["ner_tags"]]
```

- [ ] **Step 4: Run tests**

```bash
cd benchmark && python -m pytest tests/test_conll03.py -v
```
Expected: PASS (5 tests). Note: first run downloads `conll2003` from HuggingFace (~5 MB).

- [ ] **Step 5: Commit**

```bash
git add benchmark/src/cyera_bench/datasets/conll03.py benchmark/tests/test_conll03.py
git commit -m "feat: add Conll03Dataset loader"
```

---

### Task 7: PiiMaskingDataset

**Files:**
- Create: `benchmark/src/cyera_bench/datasets/pii_masking.py`

- [ ] **Step 1: Write PiiMaskingDataset**

```python
# benchmark/src/cyera_bench/datasets/pii_masking.py
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
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/src/cyera_bench/datasets/pii_masking.py
git commit -m "feat: add PiiMaskingDataset loader (ai4privacy/pii-masking-300k)"
```

---

### Task 8: SyntheticPiiDataset with built-in generator

**Files:**
- Create: `benchmark/src/cyera_bench/datasets/synthetic_pii.py`
- Create: `benchmark/tests/test_synthetic_pii.py`

- [ ] **Step 1: Write failing tests**

```python
# benchmark/tests/test_synthetic_pii.py
import pytest
from cyera_bench.datasets.synthetic_pii import SyntheticPiiDataset
from cyera_bench.types import Entity

@pytest.fixture
def ds():
    return SyntheticPiiDataset(size=100, seed=42)

def test_entity_types(ds):
    expected = {
        "CREDIT_CARD", "SSN", "EMAIL", "PHONE", "API_KEY",
        "PASSWORD", "BANK_ACCOUNT", "IP_ADDRESS", "URL",
        "DATE_OF_BIRTH", "DRIVERS_LICENSE", "PASSPORT",
    }
    assert set(ds.entity_types) == expected

def test_load_returns_dataset(ds):
    dataset = ds.load("test")
    assert len(dataset) > 0
    assert "tokens" in dataset.features
    assert "ner_tags" in dataset.features

def test_load_train_split_differs(ds):
    train = ds.load("train")
    test = ds.load("test")
    assert len(train) == 80
    assert len(test) == 20

def test_output_has_bio_format(ds):
    dataset = ds.load("test")
    for tag_seq in dataset["ner_tags"]:
        for tag in tag_seq:
            assert tag in (0, 1, 2)  # O, B-X, I-X
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd benchmark && python -m pytest tests/test_synthetic_pii.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write SyntheticPiiDataset**

```python
# benchmark/src/cyera_bench/datasets/synthetic_pii.py
import random
import re
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict

from cyera_bench.datasets.base import BaseDataset

_GENERATORS: Dict[str, callable] = {}

def _register(entity_type: str):
    def decorator(fn):
        _GENERATORS[entity_type] = fn
        return fn
    return decorator

@_register("CREDIT_CARD")
def _gen_cc():
    digits = [str(random.randint(0, 9)) for _ in range(16)]
    return "".join(digits)

@_register("SSN")
def _gen_ssn():
    return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"

@_register("EMAIL")
def _gen_email():
    users = ["john.doe", "jane.smith", "admin", "support", "info"]
    domains = ["company.com", "example.org", "acme.co", "corp.net"]
    return f"{random.choice(users)}@{random.choice(domains)}"

@_register("PHONE")
def _gen_phone():
    return f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

@_register("API_KEY")
def _gen_api_key():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    key = "".join(random.choice(chars) for _ in range(40))
    return f"sk-{key}"

@_register("PASSWORD")
def _gen_password():
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*") for _ in range(12))

@_register("BANK_ACCOUNT")
def _gen_bank():
    return f"{random.randint(10000000, 99999999)}"

@_register("IP_ADDRESS")
def _gen_ip():
    return f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

@_register("URL")
def _gen_url():
    paths = ["/api/v1/users", "/admin/login", "/docs/internal", "/health"]
    return f"https://{random.choice(['internal', 'staging', 'prod'])}.company.com{random.choice(paths)}"

@_register("DATE_OF_BIRTH")
def _gen_dob():
    return f"{random.randint(1900, 2010)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"

@_register("DRIVERS_LICENSE")
def _gen_dl():
    return f"DL-{''.join(str(random.randint(0,9)) for _ in range(10))}"

@_register("PASSPORT")
def _gen_passport():
    return f"{chr(random.randint(65, 90))}{random.randint(1000000, 9999999)}"

_TEMPLATES = [
    "The {entity_type} is {value}.",
    "Please update my {entity_type} from {value}.",
    "{value} is the {entity_type} for account #12345.",
    "Record: {entity_type}={value}",
    "Contact info: {value} ({entity_type}).",
    "Customer's {entity_type} changed to {value}.",
    "ID verification: {entity_type} {value}.",
    "This is a test {entity_type}: {value}.",
    "employee record: {value} ({entity_type})",
    "The system detected {entity_type}: {value}",
    "Input field {entity_type} contains: {value}",
    "Legacy data: {entity_type} was {value}",
]

_EDGE_TEMPLATES = [
    ("This is a test {entity_type} {value}.", True),
    ("The sample {entity_type} is {value}.", True),
    ("{value} is a valid {entity_type}.", True),
    ("The number {value} looks like a {entity_type}.", False),
    ("Invalid {entity_type}: {value}.", True),
]

class SyntheticPiiDataset(BaseDataset):
    def __init__(self, size: int = 1000, seed: int = 42, include_edge_cases: bool = True):
        self._size = size
        self._seed = seed
        self._include_edge_cases = include_edge_cases
        random.seed(seed)
        self._cache: Dict[str, Dataset] = {}

    @property
    def entity_types(self) -> List[str]:
        return sorted(_GENERATORS.keys())

    def load(self, split: str = "test") -> Dataset:
        if split not in self._cache:
            data = self._generate_all()
            ds = Dataset.from_list(data)
            split_ds = ds.train_test_split(test_size=0.2, seed=self._seed)
            self._cache["train"] = split_ds["train"]
            self._cache["test"] = split_ds["test"]
        return self._cache[split]

    def _generate_all(self) -> List[dict]:
        samples: List[dict] = []
        random.seed(self._seed)

        for _ in range(self._size):
            entity_type = random.choice(list(_GENERATORS.keys()))
            value = _GENERATORS[entity_type]()
            template = random.choice(_TEMPLATES)
            text = template.format(entity_type=entity_type, value=value)
            tokens, tags = self._bio_annotate(text, entity_type, value)
            samples.append({"tokens": tokens, "ner_tags": tags})

        if self._include_edge_cases:
            for template, is_pii in _EDGE_TEMPLATES:
                for etype in random.sample(list(_GENERATORS.keys()), min(5, len(_GENERATORS))):
                    value = _GENERATORS[etype]()
                    text = template.format(entity_type=etype, value=value)
                    if is_pii:
                        tokens, tags = self._bio_annotate(text, etype, value)
                    else:
                        tokens = text.split()
                        tags = [0] * len(tokens)
                    samples.append({"tokens": tokens, "ner_tags": tags})

        return samples

    def _bio_annotate(self, text: str, entity_type: str, value: str) -> Tuple[List[str], List[int]]:
        words = text.split()
        tags: List[int] = [0] * len(words)
        entity_type_id = self._entity_type_to_id(entity_type)

        value_words = value.split()
        for i in range(len(words) - len(value_words) + 1):
            if words[i:i + len(value_words)] == value_words:
                tags[i] = entity_type_id * 2 - 1  # B-entity
                for j in range(1, len(value_words)):
                    tags[i + j] = entity_type_id * 2  # I-entity
                break

        return words, tags

    def _entity_type_to_id(self, etype: str) -> int:
        types = sorted(_GENERATORS.keys())
        return types.index(etype) + 1  # 0 is O
```

- [ ] **Step 4: Run tests**

```bash
cd benchmark && python -m pytest tests/test_synthetic_pii.py -v
```
Expected: PASS (4 tests). No network required.

- [ ] **Step 5: Commit**

```bash
git add benchmark/src/cyera_bench/datasets/synthetic_pii.py benchmark/tests/test_synthetic_pii.py
git commit -m "feat: add SyntheticPiiDataset with 12 PII type generators and edge cases"
```

---

### Task 9: MetricsCalculator

**Files:**
- Create: `benchmark/src/cyera_bench/metrics/calculator.py`
- Create: `benchmark/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# benchmark/tests/test_metrics.py
import pytest
import numpy as np
from cyera_bench.types import Entity, BenchmarkResult
from cyera_bench.metrics.calculator import MetricsCalculator

def make_entity(text, etype="PER", start=0, end=0):
    return Entity(type=etype, text=text, start=start, end=end)

def test_ner_f1_perfect():
    predictions = [[make_entity("John", "PER")]]
    ground_truth = [[make_entity("John", "PER")]]
    calc = MetricsCalculator()
    per_entity, macro_f1 = calc.compute_ner_metrics(predictions, ground_truth)
    assert per_entity["PER"]["f1"] == 1.0
    assert macro_f1 == 1.0

def test_ner_f1_empty():
    predictions: list = [[]]
    ground_truth: list = [[]]
    calc = MetricsCalculator()
    per_entity, macro_f1 = calc.compute_ner_metrics(predictions, ground_truth)
    assert macro_f1 == 0.0

def test_throughput():
    calc = MetricsCalculator()
    throughput = calc.compute_throughput(total_tokens=1000, total_time_sec=5.0)
    assert throughput == 200.0

def test_latency_percentiles():
    calc = MetricsCalculator()
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50, p95, p99 = calc.compute_latency_percentiles(latencies)
    assert p50 == 30.0
    assert p95 >= 40.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd benchmark && python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write MetricsCalculator**

```python
# benchmark/src/cyera_bench/metrics/calculator.py
from typing import List, Dict, Tuple
import numpy as np
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from cyera_bench.types import Entity, BenchmarkResult

class MetricsCalculator:
    def compute_ner_metrics(
        self,
        predictions: List[List[Entity]],
        ground_truth: List[List[Entity]],
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        y_true, y_pred = self._to_bio(predictions, ground_truth)

        if all(len(seq) == 0 for seq in y_true) and all(len(seq) == 0 for seq in y_pred):
            return {}, 0.0

        try:
            report = seqeval_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            return {}, 0.0

        per_entity: Dict[str, Dict[str, float]] = {}
        for key, metrics in report.items():
            if key not in ("micro avg", "macro avg", "weighted avg"):
                per_entity[key] = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                }

        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        return per_entity, macro_f1

    def compute_throughput(self, total_tokens: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return total_tokens / total_time_sec

    def compute_latency_percentiles(self, latencies_ms: List[float]) -> Tuple[float, float, float]:
        if not latencies_ms:
            return 0.0, 0.0, 0.0
        arr = np.array(latencies_ms)
        return (
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
            float(np.percentile(arr, 99)),
        )

    def _to_bio(
        self,
        predictions: List[List[Entity]],
        ground_truth: List[List[Entity]],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        y_true: List[List[str]] = []
        y_pred: List[List[str]] = []

        for gt_entities, pred_entities in zip(ground_truth, predictions):
            gt_sorted = sorted(gt_entities, key=lambda e: e.start)
            pred_sorted = sorted(pred_entities, key=lambda e: e.start)

            gt_strs = [f"{e.type}:{e.text}" for e in gt_sorted]
            pred_strs = [f"{e.type}:{e.text}" for e in pred_sorted]

            y_true.append(gt_strs)
            y_pred.append(pred_strs)

        return y_true, y_pred
```

- [ ] **Step 4: Run tests**

```bash
cd benchmark && python -m pytest tests/test_metrics.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmark/src/cyera_bench/metrics/calculator.py benchmark/tests/test_metrics.py
git commit -m "feat: add MetricsCalculator — NER F1, throughput, latency percentiles"
```

---

### Task 10: Reporter — terminal, markdown, JSON output

**Files:**
- Create: `benchmark/src/cyera_bench/reporter.py`

- [ ] **Step 1: Write Reporter**

```python
# benchmark/src/cyera_bench/reporter.py
import json
import os
from datetime import datetime
from typing import List, Dict
from cyera_bench.types import BenchmarkResult

class Reporter:
    def __init__(self, output_formats: List[str] | None = None, output_path: str = "./results/"):
        self.formats = output_formats or ["terminal", "markdown", "json"]
        self.output_path = output_path

    def report(self, result: BenchmarkResult) -> None:
        for fmt in self.formats:
            if fmt == "terminal":
                self._report_terminal(result)
            elif fmt == "markdown":
                self._report_markdown(result)
            elif fmt == "json":
                self._report_json(result)

    def _report_terminal(self, r: BenchmarkResult) -> None:
        print()
        print("=" * 58)
        print(f"  Benchmark: {r.experiment_name}")
        print(f"  Model: {r.model_name} ({r.model_variant}, {r.param_count/1e6:.0f}M params)")
        print(f"  Dataset: {r.dataset_name} ({r.total_samples} samples)")
        print(f"  Device: {'CUDA' if r.gpu_memory_peak_gb > 0 else 'CPU'}")
        print("=" * 58)

        if r.per_entity_metrics:
            print(f"  {'Entity':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
            for etype, m in sorted(r.per_entity_metrics.items()):
                print(f"  {etype:<16} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
            print(f"  {'-'*46}")
            print(f"  {'Macro F1':<16} {r.macro_f1:>30.4f}")

        print()
        print(f"  Throughput:     {r.throughput_tokens_per_sec:>8.1f} tokens/sec")
        print(f"  Latency P50:    {r.latency_p50_ms:>8.1f} ms")
        print(f"  Latency P95:    {r.latency_p95_ms:>8.1f} ms")
        print(f"  Latency P99:    {r.latency_p99_ms:>8.1f} ms")
        if r.gpu_memory_peak_gb > 0:
            print(f"  GPU Memory Peak:{r.gpu_memory_peak_gb:>8.1f} GB")
        print(f"  Total Time:     {r.total_time_sec:>8.1f} sec")
        print("=" * 58)
        print()

    def _report_markdown(self, r: BenchmarkResult) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{r.experiment_name}_{date_str}.md"
        filepath = os.path.join(self.output_path, filename)

        lines = [
            f"# Benchmark: {r.experiment_name}",
            "",
            f"- **Model:** {r.model_name} ({r.param_count/1e6:.0f}M params, {r.model_variant})",
            f"- **Dataset:** {r.dataset_name} ({r.total_samples} samples)",
            f"- **Date:** {date_str}",
            "",
        ]

        if r.per_entity_metrics:
            lines.append("## Entity-Level Metrics")
            lines.append("")
            lines.append("| Entity | Precision | Recall | F1 |")
            lines.append("|--------|-----------|--------|-----|")
            for etype, m in sorted(r.per_entity_metrics.items()):
                lines.append(f"| {etype} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
            lines.append(f"| **Macro Avg** | - | - | **{r.macro_f1:.4f}** |")
            lines.append("")

        lines.extend([
            "## Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Throughput | {r.throughput_tokens_per_sec:.1f} tokens/sec |",
            f"| Latency P50 | {r.latency_p50_ms:.1f} ms |",
            f"| Latency P95 | {r.latency_p95_ms:.1f} ms |",
            f"| Latency P99 | {r.latency_p99_ms:.1f} ms |",
            f"| GPU Memory Peak | {r.gpu_memory_peak_gb:.1f} GB |",
            f"| Total Time | {r.total_time_sec:.1f} sec |",
            "",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  [Markdown report saved to {filepath}]")

    def _report_json(self, r: BenchmarkResult) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{r.experiment_name}_{date_str}.json"
        filepath = os.path.join(self.output_path, filename)

        data = {
            "experiment_name": r.experiment_name,
            "model_name": r.model_name,
            "model_variant": r.model_variant,
            "model_param_count": r.param_count,
            "dataset_name": r.dataset_name,
            "per_entity_metrics": r.per_entity_metrics,
            "macro_f1": r.macro_f1,
            "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_p99_ms": r.latency_p99_ms,
            "gpu_memory_peak_gb": r.gpu_memory_peak_gb,
            "total_samples": r.total_samples,
            "total_time_sec": r.total_time_sec,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"  [JSON report saved to {filepath}]")

    def compare(self, result_paths: List[str]) -> None:
        """Load multiple JSON results and print a comparison table."""
        results: List[BenchmarkResult] = []
        for path in result_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(BenchmarkResult(**data))

        if not results:
            print("No results to compare.")
            return

        print()
        print("=" * 80)
        print("  Cross-Experiment Comparison")
        print("=" * 80)
        print(f"  {'Experiment':<30} {'Macro F1':>10} {'Throughput':>12} {'P50 Lat':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10}")
        for r in results:
            print(f"  {r.experiment_name:<30} {r.macro_f1:>10.4f} {r.throughput_tokens_per_sec:>10.1f} t/s {r.latency_p50_ms:>8.1f} ms")
        print("=" * 80)
        print()
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/src/cyera_bench/reporter.py
git commit -m "feat: add Reporter — terminal, markdown, JSON output + cross-experiment comparison"
```

---

### Task 11: Orchestrator — wires config, model, dataset, metrics, reporter

**Files:**
- Create: `benchmark/src/cyera_bench/orchestrator.py`

- [ ] **Step 1: Write Orchestrator**

```python
# benchmark/src/cyera_bench/orchestrator.py
import time
import torch
from typing import List, Dict, Any
from cyera_bench.types import Entity, BenchmarkResult
from cyera_bench.models.base import BaseModel
from cyera_bench.models.flan_t5 import FlanT5Model
from cyera_bench.datasets.base import BaseDataset
from cyera_bench.datasets.conll03 import Conll03Dataset
from cyera_bench.datasets.pii_masking import PiiMaskingDataset
from cyera_bench.datasets.synthetic_pii import SyntheticPiiDataset
from cyera_bench.metrics.calculator import MetricsCalculator
from cyera_bench.reporter import Reporter

_MODEL_REGISTRY = {"flan-t5": FlanT5Model}
_DATASET_REGISTRY = {
    "conll03": Conll03Dataset,
    "pii-masking": PiiMaskingDataset,
    "synthetic-pii": SyntheticPiiDataset,
}

class BenchmarkOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_calc = MetricsCalculator()

        exp = config["experiment"]
        self.experiment_name = exp["name"]

        model_cfg = config["model"]
        model_cls = _MODEL_REGISTRY[model_cfg["type"]]
        self.model: BaseModel = model_cls(
            variant=model_cfg.get("variant", "large"),
            device=model_cfg.get("device", "cuda"),
            quantization=model_cfg.get("quantization"),
        )

        dataset_cfg = config["dataset"]
        dataset_cls = _DATASET_REGISTRY[dataset_cfg["type"]]
        kwargs = dataset_cfg.get("kwargs", {})
        self.dataset: BaseDataset = dataset_cls(**kwargs)
        self.dataset_split = dataset_cfg.get("split", "test")

        output_cfg = config.get("output", {})
        self.reporter = Reporter(
            output_formats=output_cfg.get("formats", ["terminal", "markdown", "json"]),
            output_path=output_cfg.get("path", "./results/"),
        )

        self.metric_names = config.get("metrics", [])
        self.batch_sizes = model_cfg.get("batch_sizes", [1, 4, 8, 16, 32])

    def run(self) -> BenchmarkResult:
        print(f"\nLoading dataset: {self.dataset.__class__.__name__} ({self.dataset_split} split)...")
        texts_raw = self.dataset.texts(self.dataset_split)
        texts = [" ".join(t) if isinstance(t, list) else t for t in texts_raw]

        try:
            ground_truth = self._load_ground_truth(texts_raw)
        except (KeyError, AttributeError):
            print("  [WARN] No ground truth labels found. Skipping accuracy metrics.")
            ground_truth = None

        print(f"Warming up model: {self.model.name}...")
        self.model.warmup(n=10)

        # Phase 1: Accuracy
        if ground_truth is not None and "ner_f1" in self.metric_names:
            print("Phase 1/2: Accuracy evaluation...")
            predictions = self._run_inference(texts, batch_size=self.batch_sizes[0])
            per_entity, macro_f1 = self.metrics_calc.compute_ner_metrics(predictions, ground_truth)
        else:
            predictions = []
            per_entity, macro_f1 = {}, 0.0

        # Phase 2: Throughput sweep
        print("Phase 2/2: Throughput sweep...")
        best_throughput = 0.0
        all_latencies: List[float] = []
        total_tokens_all = 0

        eval_texts = texts[:min(len(texts), 1000)]

        for bs in self.batch_sizes:
            latencies, total_tokens, total_time = self._benchmark_throughput(eval_texts, batch_size=bs)
            throughput = total_tokens / total_time if total_time > 0 else 0
            all_latencies.extend(latencies)
            total_tokens_all += total_tokens

            if throughput > best_throughput:
                best_throughput = throughput
                print(f"  batch_size={bs:>2}: {throughput:>8.1f} tokens/sec, P50={self.metrics_calc.compute_latency_percentiles(latencies)[0]:.0f}ms")

        p50, p95, p99 = self.metrics_calc.compute_latency_percentiles(all_latencies)

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9

        result = BenchmarkResult(
            experiment_name=self.experiment_name,
            model_name=self.model.name,
            model_variant=self.config["model"].get("variant", "large"),
            dataset_name=self.config["dataset"]["type"],
            per_entity_metrics=per_entity,
            macro_f1=macro_f1,
            throughput_tokens_per_sec=best_throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            gpu_memory_peak_gb=gpu_mem,
            total_samples=len(texts),
            total_time_sec=sum(all_latencies) / 1000.0,
        )

        self.reporter.report(result)
        return result

    def _load_ground_truth(self, texts_raw: list) -> List[List[Entity]] | None:
        tag_seqs = self.dataset.bio_tags(self.dataset_split)
        entities: List[List[Entity]] = []

        for tokens, tag_seq in zip(texts_raw, tag_seqs):
            tokens_l = list(tokens) if isinstance(tokens, (list, tuple)) else tokens.split()
            sample_entities: List[Entity] = []
            current_entity: str | None = None
            current_start = 0
            current_tokens: List[str] = []

            for i, tag in enumerate(tag_seq):
                tag_str = tag if isinstance(tag, str) else f"TAG-{tag}"
                if tag_str.startswith("B-"):
                    if current_entity:
                        sample_entities.append(Entity(
                            type=current_entity,
                            text=" ".join(current_tokens),
                            start=current_start,
                            end=current_start + len(" ".join(current_tokens)),
                        ))
                    current_entity = tag_str[2:]
                    current_start = i
                    current_tokens = [tokens_l[i] if i < len(tokens_l) else ""]
                elif tag_str.startswith("I-") and current_entity:
                    current_tokens.append(tokens_l[i] if i < len(tokens_l) else "")
                else:
                    if current_entity:
                        sample_entities.append(Entity(
                            type=current_entity,
                            text=" ".join(current_tokens),
                            start=current_start,
                            end=current_start + len(" ".join(current_tokens)),
                        ))
                        current_entity = None
                        current_tokens = []

            if current_entity:
                sample_entities.append(Entity(
                    type=current_entity,
                    text=" ".join(current_tokens),
                    start=current_start,
                    end=current_start + len(" ".join(current_tokens)),
                ))

            entities.append(sample_entities)

        return entities

    def _run_inference(self, texts: List[str], batch_size: int) -> List[List[Entity]]:
        all_results: List[List[Entity]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_results.extend(self.model.predict(batch))
        return all_results

    def _benchmark_throughput(self, texts: List[str], batch_size: int):
        latencies: List[float] = []
        total_tokens = 0

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            total_tokens += sum(len(t.split()) for t in batch)

            t_start = time.perf_counter()
            self.model.predict(batch)
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) * 1000)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t0

        return latencies, total_tokens, total_time
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/src/cyera_bench/orchestrator.py
git commit -m "feat: add BenchmarkOrchestrator — YAML-driven experiment runner"
```

---

### Task 12: CLI entry point (`__main__.py`)

**Files:**
- Create: `benchmark/src/cyera_bench/__main__.py`

- [ ] **Step 1: Write CLI entry point**

```python
# benchmark/src/cyera_bench/__main__.py
import argparse
import sys
import yaml
from pathlib import Path
from cyera_bench.orchestrator import BenchmarkOrchestrator
from cyera_bench.reporter import Reporter

def main():
    parser = argparse.ArgumentParser(description="Cyera FLAN-T5 NER/PII Benchmark")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=False,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Compare multiple result JSON files",
    )
    parser.add_argument(
        "--defaults",
        type=str,
        default=None,
        help="Path to defaults YAML file to merge with experiment config",
    )

    args = parser.parse_args()

    if args.compare:
        reporter = Reporter()
        reporter.compare(args.compare)
        return

    if not args.config:
        print("Usage: python -m cyera_bench --config config/experiments/<name>.yaml")
        print("   or: python -m cyera_bench --compare results/*.json")
        sys.exit(1)

    config = _load_config(args.config, args.defaults)
    orch = BenchmarkOrchestrator(config)
    result = orch.run()

    return result

def _load_config(config_path: str, defaults_path: str | None = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if defaults_path:
        with open(defaults_path, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f)
        config = _merge_configs(defaults, config)

    return config

def _merge_configs(defaults: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
            defaults[key] = _merge_configs(defaults[key], value)
        else:
            defaults[key] = value
    return defaults

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/src/cyera_bench/__main__.py
git commit -m "feat: add CLI entry point (cyera-bench --config + --compare)"
```

---

### Task 13: YAML experiment configs

**Files:**
- Create: `benchmark/config/experiments/flan-t5-defaults.yaml`
- Create: `benchmark/config/experiments/flan-t5-base-conll03.yaml`
- Create: `benchmark/config/experiments/flan-t5-large-conll03.yaml`
- Create: `benchmark/config/experiments/flan-t5-large-pii.yaml`
- Create: `benchmark/config/experiments/flan-t5-xl-conll03.yaml`

- [ ] **Step 1: Write defaults config**

```yaml
# benchmark/config/experiments/flan-t5-defaults.yaml
model:
  type: "flan-t5"
  variant: "large"
  device: "cuda"
  quantization: null
  batch_sizes: [1, 4, 8, 16, 32]

metrics:
  - ner_f1
  - throughput_tokens_per_sec
  - latency_p50_p95_p99

output:
  formats: [terminal, markdown, json]
  path: "./results/"
```

- [ ] **Step 2: Write experiment configs**

```yaml
# benchmark/config/experiments/flan-t5-base-conll03.yaml
experiment:
  name: "flan-t5-base-conll03"
  description: "FLAN-T5-Base on CoNLL-03 NER benchmark"

model:
  variant: "base"

dataset:
  type: "conll03"
  split: "test"
```

```yaml
# benchmark/config/experiments/flan-t5-large-conll03.yaml
experiment:
  name: "flan-t5-large-conll03"
  description: "FLAN-T5-Large on CoNLL-03 NER benchmark"

model:
  variant: "large"

dataset:
  type: "conll03"
  split: "test"
```

```yaml
# benchmark/config/experiments/flan-t5-large-pii.yaml
experiment:
  name: "flan-t5-large-pii"
  description: "FLAN-T5-Large on PII-Masking-300k real-world PII benchmark"

model:
  variant: "large"

dataset:
  type: "pii-masking"
  split: "test"
```

```yaml
# benchmark/config/experiments/flan-t5-xl-conll03.yaml
experiment:
  name: "flan-t5-xl-conll03"
  description: "FLAN-T5-XL (3B) on CoNLL-03 NER benchmark"

model:
  variant: "xl"
  quantization: "8bit"

dataset:
  type: "conll03"
  split: "test"
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/config/
git commit -m "feat: add YAML experiment configs for base/large/xl on CoNLL-03 and PII"
```

---

### Task 14: Environment setup scripts

**Files:**
- Create: `benchmark/setup.sh`
- Create: `benchmark/setup.bat`

- [ ] **Step 1: Write setup.sh**

```bash
#!/bin/bash
# FLAN-T5 Benchmark — Linux/macOS environment setup

set -e

echo "=== Cyera FLAN-T5 Benchmark Setup ==="

# Check for Conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found. Please install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "[WARN] No NVIDIA GPU detected. Will run on CPU (slow)."
fi

# Create conda environment if not exists
ENV_NAME="cyera-bench"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[OK] Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -y -n ${ENV_NAME} python=3.11
fi

# Activate and install
echo "Installing PyTorch and dependencies..."
conda run -n ${ENV_NAME} pip install --upgrade pip

# Detect CUDA version for PyTorch
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" | cut -d. -f1)
    if [ "$CUDA_VERSION" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    conda run -n ${ENV_NAME} pip install torch torchvision --index-url ${TORCH_INDEX}
else
    conda run -n ${ENV_NAME} pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing benchmark dependencies..."
conda run -n ${ENV_NAME} pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate: conda activate ${ENV_NAME}"
echo "Run:      python -m cyera_bench --config config/experiments/flan-t5-base-conll03.yaml"
```

- [ ] **Step 2: Write setup.bat**

```batch
@echo off
REM FLAN-T5 Benchmark — Windows environment setup
echo === Cyera FLAN-T5 Benchmark Setup ===

where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Conda not found. Please install Miniconda first.
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
) else (
    echo [WARN] No NVIDIA GPU detected. Will run on CPU.
)

set ENV_NAME=cyera-bench
conda env list | findstr /c:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Conda environment '%ENV_NAME%' already exists.
) else (
    echo Creating conda environment '%ENV_NAME%' with Python 3.11...
    conda create -y -n %ENV_NAME% python=3.11
)

echo Installing PyTorch and dependencies...
conda run -n %ENV_NAME% pip install --upgrade pip
conda run -n %ENV_NAME% pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
conda run -n %ENV_NAME% pip install -e .

echo.
echo === Setup complete ===
echo Activate: conda activate %ENV_NAME%
echo Run:      python -m cyera_bench --config config/experiments/flan-t5-base-conll03.yaml
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/setup.sh benchmark/setup.bat
git commit -m "feat: add environment setup scripts for Linux and Windows"
```

---

## Execution Order

Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14

Tasks 6, 7, 8 can run in parallel after Task 5. Tasks 9, 10 can run in parallel after Task 8.
