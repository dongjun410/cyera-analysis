# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository has two workstreams:

1. **Technology research and analysis** covering **Cyera's data security platform** (DSPM, Omni DLP, AI Guardian). Includes technical architecture analysis, patent portfolio deep-dive, competitive landscape, and product evolution tracking. Work product is independent analysis reports.
2. **Benchmark framework** (`benchmark/`) — a Python package (`cyera-bench`) for benchmarking NLP models on NER and document classification tasks. Supports FLAN-T5, sklearn, and Gemma models across multiple datasets.

## Key Files

### Analysis Reports
- `overall_analysis_report.md` — Independent analysis of Cyera's technical architecture, classification engine (DataDNA, FLAN-T5/Mistral), data discovery, DataGraph, Omni DLP, AI Guardian, and patent portfolio overview. Uses platform-capability lens (what/why).
- `patents/technical_analysis_report.md` — Per-patent deep-dive with scenario-based unified implementation plans. Uses engineering-implementation lens (how). Authoritative for patent technical details. Cross-validated with `overall_analysis_report.md`.
- `patents/` — All 8 Cyera patent PDFs (5 granted US patents + 3 published applications).

### Benchmark Framework
- `benchmark/src/cyera_bench/__main__.py` — CLI entry point: `cyera-bench --config <yaml> [--defaults <yaml>]` or `--compare <json...>`
- `benchmark/src/cyera_bench/orchestrator.py` — Core loop: model/dataset dispatch, accuracy + throughput phases
- `benchmark/src/cyera_bench/reporter.py` — Terminal, markdown, JSON output + cross-experiment comparison
- `benchmark/src/cyera_bench/types.py` — `Entity`, `BenchmarkResult` (NER), `ClassificationBenchmarkResult` dataclasses
- `benchmark/config/experiments/` — YAML configs with deep-merge inheritance (defaults + experiment overrides)
- `benchmark/results/` — Output directory for markdown + JSON reports
- `docs/huggingface-setup.md` — HF mirror and model download setup guide

## Patent Inventory (8 documents, 6 sections)

| # | Patent | Status | Family |
|---|--------|--------|--------|
| 1 | US12026123B2 | Granted 2024-07-02 | Data Discovery |
| 2 | US12499083B2 | Granted 2025-12-16 | Data Discovery (Cont.) |
| 3 | US12566567B2 | Granted 2026-03-03 | Data Discovery (CIP) |
| 4 | US12299167B2 | Granted 2025-05-13 | Data Classification |
| 5 | US12316686B1 | Granted 2025-05-27 | Security Policy (Trail Security) |
| 6 | US20240362301A1 | Pending | Clustering Classification |
| 7 | US20250068701A1 | Pending | Clustering Classification (Cont.) |
| 8 | WO2024224367A1 | PCT National Phase | Clustering Classification (PCT) |

## Benchmark Framework Structure

### Task Types
- **NER** (`task_type="ner"`): BIO-tagging via `seqeval`. Throughput in tokens/sec.
- **Classification** (`task_type="classification"`): L1/L2 document labeling via `sklearn`. Throughput in chars/sec.

### Models (`benchmark/src/cyera_bench/models/`)
| Model | Type Key | Task | Description |
|-------|----------|------|-------------|
| `FlanT5Model` | `flan-t5` | NER | HF pipeline: token-classification (small/base) or text2text (large) |
| `FlanT5ClassificationModel` | `flan-t5-classification` | Classification | Two-step/single-step L1→L2 prompt + fuzzy label matching |
| `DocClassifierSklearnModel` | `doc-classifier-sklearn` | Classification | TF-IDF + LogisticRegression from `ZerosOne/doc-classifier` |
| `GemmaDocLabelModel` | `gemma-doc-label` | Classification | HTTP to Gemma4:e2b Ollama at port 8003 |

All models extend `BaseModel` (NER) or implement `predict_labels()` (classification).

### Datasets (`benchmark/src/cyera_bench/datasets/`)
| Dataset | Type Key | Task | Source |
|---------|----------|------|--------|
| `Conll03Dataset` | `conll03` | NER | HF `conllpp` (PER/ORG/LOC/MISC) |
| `PiiMaskingDataset` | `pii-masking` | NER | HF `ai4privacy/pii-masking-300k` (17 types) |
| `SyntheticPiiDataset` | `synthetic-pii` | NER | Programmatic generation (12 types) |
| `Dspm27Dataset` | `dspm27` | Classification | 27 PDFs, GPT-labeled L1/L2 |
| `Ben25Dataset` | `ben25` | Classification | 25 docs, GPT-labeled L1/L2 |
| `Cxh5typesDataset` | `cxh5types` | Classification | 258 docs, human-annotated L1/L2 |

Classification datasets extend `BaseDocLabelDataset`; NER datasets extend `BaseDataset`.

### Metrics (`benchmark/src/cyera_bench/metrics/`)
- `calculator.py` — NER metrics via `seqeval` (per-entity P/R/F1, macro F1), throughput, latency P50/P95/P99
- `classification.py` — L1/L2 accuracy, L2@correct-L1, macro F1 per level, sklearn `classification_report`

### Orchestrator flow
1. Parse config → determine `task_type`
2. Lazy-import model/dataset classes via string-name dispatch
3. Phase 1: Accuracy — full dataset at batch_size=1
4. Phase 2: Throughput — sweep batch sizes, measure tokens/chars per sec
5. Capture GPU peak memory via `torch.cuda.max_memory_allocated()`
6. Construct result dataclass → pass to `Reporter`

### Test files (`benchmark/tests/`)
- `test_types.py` — Entity and BenchmarkResult dataclass validation
- `test_conll03.py` — Conll03Dataset: entity types, splits, BIO tags
- `test_flan_t5.py` — FlanT5Model: variant metadata, errors, skip-model-download
- `test_metrics.py` — NER metrics: perfect F1, empty, throughput, percentiles
- `test_synthetic_pii.py` — SyntheticPiiDataset: types, 80/20 split, BIO format

### Dependencies
Python >= 3.11. Key packages: `torch>=2.5`, `transformers>=4.46`, `datasets>=3.0,<4.0`, `seqeval>=1.2`, `scikit-learn>=1.5`, `pyyaml>=6.0`, `bitsandbytes>=0.44`, `accelerate>=1.0`, `numpy>=1.26`. Runtime: `pdfplumber` (dspm27), `joblib` (sklearn model).

## Research Methodology

When conducting analysis in this repo:

1. **Source grading**: Classify every factual claim by source reliability (A=Cyera official, B=third-party verified like Forrester/Gartner, C=reasonable inference, D=competitor/indirect).
2. **Distinguish facts from inference**: Never present reasonable inference as confirmed fact. Mark speculative claims explicitly.
3. **Verify sources**: Cross-reference claims against primary sources (Cyera blog, USPTO patents, BusinessWire press releases, analyst reports) before citing.
4. **Use WebSearch + WebFetch**: Gather information from multiple angles before writing. Prefer official Cyera sources and third-party analyst reports over competitor content.
5. **Cross-validate between reports**: `overall_analysis_report.md` and `patents/technical_analysis_report.md` describe the same platform from different lenses. When updating one, verify the other remains consistent (patent counts, timelines, technical claims, inventor names). The patent report is authoritative on patent-level detail; the platform report is authoritative on product-level capability.

## Benchmark Coding Standards

- Run tests with `pytest benchmark/tests/` from the repo root
- Experiment configs use deep-merge: defaults file + experiment file (experiment overrides win)
- Models and datasets register via string-name dispatch in orchestrator — no import-time side effects
- New models implement `predict(texts)` for NER or `predict_labels(texts)` for classification
- New datasets extend `BaseDataset` (NER) or `BaseDocLabelDataset` (classification)
- Results saved to `benchmark/results/` with dated filenames

## Analysis Standards

- Be independently critical — do not cater to prior analyses or preferences
- Note what is MISSING from any reference analysis being compared against
- Include a timeline dimension — Cyera's platform evolves rapidly (major releases every 2-3 months)
- When a reference analysis makes specific technical claims (throughput numbers, model types, etc.), verify each independently
- Cite sources with URLs; group sources by reliability tier at the end of reports
- Competitor claims (patent counts, technical approaches) must carry source grades — do not present them as unqualified facts
