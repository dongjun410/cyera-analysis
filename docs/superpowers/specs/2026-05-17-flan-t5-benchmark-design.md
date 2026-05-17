# FLAN-T5 NER/PII Benchmark Framework — Design Spec

**Date:** 2026-05-17
**Status:** Draft
**Goal:** Build a reusable benchmarking framework for FLAN-T5 NER/PII classification, validating technical claims in `overall_analysis_report.md` section 3.3.1.

---

## 1. Scope

- Phase 1 (this spec): FLAN-T5 Small/Base/Large/XL across public NER benchmarks + PII detection datasets.
- Phase 2 (future): Mistral model integration via the same `BaseModel` interface, dual-model pipeline.

## 2. Architecture

```
┌──────────────────────────────────┐
│         YAML Config              │
├──────────────────────────────────┤
│     BenchmarkOrchestrator        │
├──────────┬──────────┬────────────┤
│  Models  │ Datasets │  Metrics   │
├──────────┴──────────┴────────────┤
│         Reporter                 │  (terminal / markdown / json)
└──────────────────────────────────┘
```

### 2.1 Core Interfaces

```python
class BaseModel(ABC):
    @property
    def name(self) -> str: ...
    @property
    def param_count(self) -> int: ...

    @abstractmethod
    def predict(self, texts: List[str]) -> List[List[Entity]]: ...

class BaseDataset(ABC):
    @property
    def entity_types(self) -> List[str]: ...

    @abstractmethod
    def load(self, split: str) -> Dataset: ...

class MetricsCalculator:
    def compute(
        self,
        predictions: List[List[Entity]],
        ground_truth: List[List[Entity]],
        latencies_ms: List[float]
    ) -> BenchmarkResult: ...
```

### 2.2 Implementation & Extension Points

| Layer | Phase 1 | Phase 2+ |
|-------|---------|----------|
| Models | `FlanT5Model` (Small/Base/Large/XL) | `MistralModel` |
| Datasets | `Conll03Dataset`, `PiiMaskingDataset`, `SyntheticPiiDataset` | `OntonotesDataset`, `CustomCSVDataset` |
| Metrics | NER F1 (seqeval), throughput, P50/P95/P99 latency | Per-type confusion matrix, calibration |

## 3. Environment

- Python 3.11, virtual env via Miniconda
- PyTorch 2.5+ with CUDA 12.5+ (RTX 5070 Blackwell sm_120)
- HuggingFace: `transformers`, `datasets`, `evaluate`, `seqeval`, `accelerate`
- `bitsandbytes` for quantization experiments
- `setup.sh` / `setup.bat` automates the full install

## 4. Datasets

| Dataset | Source | Entity Types | Volume | Purpose |
|---------|--------|-------------|--------|---------|
| CoNLL-03 | HF `conll2003` | PER/ORG/LOC/MISC (4) | ~3.5K test sentences | NER SOTA benchmark |
| PII-Masking-300k | HF `ai4privacy/pii-masking-300k` | 17 types (PERSON/EMAIL/PHONE/ID/URL...) | ~30K test | Real PII detection benchmark |
| Synthetic PII | Built-in generator | CREDIT_CARD/SSN/API_KEY/BANK_ACCOUNT etc. (12 types) | Configurable | Sensitive types that can't appear in public data |
| Synthetic Edge | Built-in generator | Ambiguity samples, false positives/negatives | ~500 | Boundary case stress testing |

All datasets output BIO tagging format for uniform evaluation.

## 5. Experiment Config (YAML)

```yaml
experiment:
  name: "flan-t5-large-conll03"
  description: "FLAN-T5-Large on CoNLL-03 NER benchmark"

model:
  type: "flan-t5"
  variant: "large"           # small | base | large | xl
  quantization: null          # null | 4bit | 8bit
  device: "cuda"

dataset:
  type: "conll03"
  entity_types: ["PER", "ORG", "LOC", "MISC"]

metrics:
  - ner_f1
  - throughput_tokens_per_sec
  - latency_p50_p95_p99

output:
  formats: [terminal, markdown, json]
  path: "./results/"
```

## 6. Execution Flow

```
python -m cyera_bench --config config/experiments/flan-t5-large-conll03.yaml
```

1. Parse YAML config
2. ModelFactory → load model with requested quantization
3. DatasetFactory → load test split
4. Warmup (10 inferences, not measured)
5. Two-phase evaluation:
   - Phase A: NER accuracy (precision/recall/F1 per entity type + macro)
   - Phase B: Throughput/latency sweep across batch sizes [1,4,8,16,32]
6. Reporter outputs terminal table + markdown + JSON

Cross-experiment comparison:
```
python -m cyera_bench --compare results/*.json
```

## 7. Output Formats

- **Terminal**: Rich table with entity-level F1, throughput, latency, GPU memory
- **Markdown**: `results/<name>_<date>.md` — concise tables, embeddable in analysis reports
- **JSON**: `results/<name>_<date>.json` — full structured results for programmatic consumption

## 8. Directory Structure

```
benchmark/
├── pyproject.toml
├── setup.sh
├── setup.bat
├── config/
│   └── experiments/
│       ├── flan-t5-defaults.yaml      # Shared defaults (batch sizes, warmup)
│       ├── flan-t5-base-conll03.yaml
│       ├── flan-t5-large-conll03.yaml
│       ├── flan-t5-large-pii.yaml
│       └── flan-t5-xl-conll03.yaml
├── src/cyera_bench/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── flan_t5.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── conll03.py
│   │   ├── pii_masking.py
│   │   └── synthetic_pii.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── calculator.py
│   └── reporter.py
├── results/
│   └── .gitkeep
└── tests/
    └── test_synthetic_generator.py
```

## 9. Key Metrics Definitions

- **NER F1 (seqeval)**: Entity-level exact span match, BIO tagging strict mode
- **Throughput**: `total_tokens / total_time_seconds`, excluding warmup
- **Latency P50/P95/P99**: Per-sample end-to-end inference time (tokenize → generate → decode)
- **GPU Memory Peak**: `torch.cuda.max_memory_allocated() / 1e9` GB
