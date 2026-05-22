# Enterprise Document Intelligent Clustering System V2.2 — Implementation Spec

> Date: 2026-05-22
> Source: `企业文档智能聚类系统V2.2_实施方案_含知识蒸馏.md` (3606-line design doc)
> Target: `impl-v2.2/`

## Decisions

- **Implementation strategy**: Follow design doc exactly — copy specified code as written, wire together
- **Execution approach**: Parallel subagents (3 groups of independent modules)
- **Dependencies**: Python 3.10+, Elasticsearch 8.x (Docker), BGE-M3 model (~2GB), optional LLM (Ollama/vLLM)

## Scope — 14 files

### 10 Core Modules (`core/`)
1. `document_processor.py` — PDF/DOCX/TXT/MD/XLSX/PPTX parsing with smart chunking
2. `pii_preclassifier.py` — Presidio + regex fallback PII detection and pre-classification
3. `structure_feature_extractor.py` — Channel 2: PII density + document structure numerical features
4. `embedding_service.py` — Channel 1: BGE-M3 semantic embedding with chunk aggregation
5. `clustering_engine.py` — Two-stage: KMeans (auto-K via Silhouette) + agglomerative splitting
6. `sensitivity_adaptive_scheduler.py` — Three-tier partitioning (high/medium/low) with per-tier params
7. `iterative_optimizer.py` — ARI-convergent merge + reassign loop
8. `label_propagator.py` — MMR representative selection + TF-IDF keywords + LLM cluster naming
9. `quality_evaluator.py` — Silhouette / DBI / CHI metrics
10. `learned_classifier.py` — LLM teacher → SetFit student distillation + incremental retraining

### 4 Support Files
- `models/schemas.py` — ProcessedDocument, ClusterInfo, PIIDetection, LabeledSample dataclasses
- `main.py` — Full 6-phase pipeline CLI
- `incremental.py` — Classifier-first incremental update with kNN fallback
- `distill.py` — Offline distillation training entry point
- `docker-compose.yml` — ES 8.18 + Kibana

### 5 Existing Files (no changes needed)
- `config.yaml`, `requirements.txt`, `README.md`, `core/__init__.py`, `models/__init__.py`

## Parallel Implementation Groups

**Group 1** (data layer): `models/schemas.py` + `document_processor.py` + `pii_preclassifier.py` + `structure_feature_extractor.py`

**Group 2** (processing layer): `embedding_service.py` + `clustering_engine.py` + `sensitivity_adaptive_scheduler.py` + `iterative_optimizer.py` + `quality_evaluator.py`

**Group 3** (output layer): `label_propagator.py` + `vector_store.py` + `learned_classifier.py` + `main.py` + `incremental.py` + `distill.py` + `docker-compose.yml`

## Verification
- All 11 imports in `core/__init__.py` resolve to existing files
- `python -c "import core"` succeeds
- `python -c "import models.schemas"` succeeds
- `main.py --help` parses without error
