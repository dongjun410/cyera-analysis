# FLAN-T5-Large QLoRA Fine-Tuning for Document Classification — Design Spec

**Date:** 2026-05-20
**Status:** Approved
**Goal:** Domain-specialize FLAN-T5-large (780M) for document classification (L1/L2), achieving significant accuracy improvements over current zero-shot baseline across benchmark datasets, with DSPM data as the primary domain target.

---

## 1. Problem Statement

Current FLAN-T5-large zero-shot classification performance (post-fix):
- 20newsgroups: 48.1% L1 (20 classes)
- Ledgar: 13.1% single-shot / 30.0% D&C (100 classes)
- German-MultiFin: 44.5% L1 (6 L1 + 23 L2)
- Cxh5types: 78.8% L1 (3 classes)

Gap to Gemma4 (2B) is 11–16pp. The benchmark report identifies model capacity (780M vs 2B) and instruction-following ability as binding constraints. Fine-tuning targets the instruction-following gap — teaching FLAN-T5-large to follow classification instructions reliably, using domain-relevant data.

RTX 5070 (12GB VRAM) rules out full fine-tuning. 8-bit QLoRA is chosen: 780M model with 8-bit quantization leaves ample VRAM headroom with zero measurable accuracy degradation vs full precision.

## 2. Architecture Overview

```
Data Preparation Layer
├── Template Engine: 5 prompt formats (zero-shot, few-shot, CoT, label→desc, contrastive)
├── Data Augmenter: back-translation + entity substitution + LLM synthesis for DSPM datasets
└── Sampling Controller: anti-dominance weighting per dataset

Training Layer
├── Phase 1: General classification (20news, Ledgar, German-MultiFin, AG News, DBpedia-14)
│   ~195K samples, 3 epochs, QLoRA r=16 α=32
└── Phase 2: DSPM domain adaptation (augmented Dspm27, Ben25, Cxh5types)
    ~900 samples, 12 epochs with early stopping, lower LR

Output Layer
└── Merged LoRA adapter → standard HF model → zero-code-change integration with benchmark
```

## 3. Template System (FLAN Core)

Five templates replace FLAN's "input reversal" with classification-appropriate bidirectional tasks.

### 3.1 Template Mix

| # | Template | Ratio | Purpose |
|:--:|----------|:-----:|---------|
| 1 | Zero-shot classification | 40% | Standard: `"Classify: {text} → {label}"` |
| 2 | Few-shot classification | 20% | 2–3 dynamic in-dataset examples before target |
| 3 | Chain-of-Thought | 15% | Force explicit reasoning: identify indicators → match categories → select |
| 4 | Label→Content generation | 15% | Replace input reversal: `"Given label '{label}', describe typical content structure and indicators"` |
| 5 | Discriminative contrast | 10% | `"Classified as {wrong} but correct is {correct}. Explain why."` — uses highest-perplexity wrong label |

### 3.2 L1/L2 Handling

For datasets with L2 labels (German-MultiFin, Cxh5types, Dspm27, Ben25):
- L2 templates follow L1 templates as a secondary generation target
- L2 prompts formatted as: `"Document (L1={l1}) → Subcategory: {l2}"`
- L2 contributes ~15% of total training samples

### 3.3 Sampling Weights

Anti-dominance weighting via `1/sqrt(N_i)` smoothing:

| Dataset | Samples | Raw Ratio | Weight | Effective Mix |
|---------|:-------:|:---------:|:------:|:-------------:|
| 20newsgroups | 7,532 | 51% | 1.00× | ~22% |
| Ledgar | 10,000 | 68% | 0.87× | ~19% |
| AG News | 120,000 | 816% | 0.25× | ~11% |
| DBpedia-14 (10% subset) | 56,000 | 381% | 0.37× | ~16% |
| German-MultiFin | 2,010 | 14% | 1.93× | ~32% |

Total Phase 1: ~195K training samples after template application (~150K unique documents × templates).

## 4. Data Augmentation for DSPM Datasets

Dspm27 (~21 train) and Ben25 (~20 train) are too small for fine-tuning. Three-layer augmentation targets 10 variants per original sample.

### 4.1 Augmentation Layers

| Layer | Method | Variants/sample | Tool |
|:-----:|--------|:---------------:|------|
| 1 | Back-translation (EN→DE→EN, EN→ZH→EN) | 2 | Helsinki NLP OPUS-MT (local, lightweight) |
| 2 | Entity substitution (names, dates, amounts, orgs) | 3 | regex + fake entity pool per type |
| 3 | LLM label-conditioned synthesis | 5 | Gemma4 (local Ollama): generates realistic documents matching label + taxonomy definition |

### 4.2 Quality Control

- TF-IDF cosine similarity between original and synthetic documents
- Discard: similarity < 0.15 (off-topic) or > 0.95 (near-duplicate)
- Target: ~200 augmented documents per DSPM dataset

### 4.3 Augmented Dataset Composition

| Dataset | Original | Augmented | Total (Phase 2) |
|---------|:--------:|:---------:|:---------------:|
| Dspm27 | 21 | ~200 | ~221 |
| Ben25 | 20 | ~200 | ~220 |
| Cxh5types | 258 | — | 258 |
| **Total** | **299** | **~400** | **~700** |

## 5. Training Configuration

### 5.1 Phase 1 — General Classification Foundation

```
Model:       google/flan-t5-large (780M)
Quantization: 8-bit (bitsandbytes load_in_8bit)
LoRA:         r=16, alpha=32, dropout=0.05
              target_modules: [q, v, k, o, wi_0, wi_1, wo]
Optimizer:    AdamW (lr=2e-4, weight_decay=0.01)
Schedule:     cosine, warmup_ratio=0.1
Batch:        8 (gradient_accumulation=2, effective=16)
Epochs:       3
Max length:   1024 (encoder), 128 (decoder)
Train data:   ~195K documents (7.5K 20news + 10K Ledgar + 120K AG News + 56K DBpedia + 2K German-MultiFin)
Est. time:    ~4–5 hours (RTX 5070)
```

### 5.2 Phase 2 — DSPM Domain Adaptation

```
Quantization: 8-bit (same)
LoRA:         r=16, alpha=32, dropout=0.10 (increased for small-data regularization)
              Load Phase 1 adapter weights as initialization
Optimizer:    AdamW (lr=5e-5, weight_decay=0.01) — 4× lower than Phase 1
Schedule:     cosine, warmup_ratio=0.1
Batch:        4 (gradient_accumulation=2, effective=8)
Epochs:       12 max, early_stopping patience=3 on val_loss
Max length:   1024 (encoder), 128 (decoder)
Train data:   ~700 docs × template application → ~910 training samples
Validation:   10% holdout from augmented DSPM data
Est. time:    ~1–1.5 hours (RTX 5070)
```

## 6. Recommended Open Datasets

### 6.1 Existing (in benchmark)

| Dataset | Samples | L1 Classes | L2 | Domain |
|---------|:-------:|:----------:|:--:|--------|
| 20 Newsgroups | 7,532 | 20 | — | News/discussion |
| Ledgar | 10,000 | 100 | — | Legal contracts |
| German-MultiFin | 2,010 | 6 | 23 | Financial (DE) |

### 6.2 New Additions

| Dataset | Total | Used | L1 Classes | Domain | Rationale |
|---------|:-----:|:----:|:----------:|--------|-----------|
| AG News | 120K | 120K | 4 | News | High sample count → template diversity; broad-domain baseline |
| DBpedia-14 | 560K | 56K (10%) | 14 | Encyclopedia ontology | 14-class structure close to DSPM multi-class; ontology-based labeling |

Both available from HuggingFace (`ag_news`, `dbpedia_14`). DBpedia subsampled via stratified 10% random split to keep training time reasonable while maintaining 14-class balance.

## 7. Evaluation Protocol

### 7.1 Primary Metrics (80/20 split, shared across all models)

| Dataset | Metrics | Current FLAN-T5 Zero-shot | Target Improvement |
|---------|---------|:------------------------:|:------------------:|
| 20newsgroups | L1 Acc, Macro F1 | 48.1% | +10–15pp |
| Ledgar (D&C) | L1 Acc | 30.0% | +15–20pp |
| Ledgar (single-shot) | L1 Acc | 13.1% | +15–25pp |
| German-MultiFin | L1 Acc, L2 Acc | 44.5% / — | +10–15pp |
| AG News | L1 Acc | TBD (new baseline) | Establish baseline |
| DBpedia-14 | L1 Acc | TBD (new baseline) | Establish baseline |

### 7.2 DSPM Sanity Check

| Dataset | N (test) | Purpose | Criterion |
|---------|:--------:|---------|-----------|
| Cxh5types | 52 | Only statistically reliable DSPM test set | 78.8% → target 90%+ |
| Dspm27 | 6 | Qualitative check only | No regression |
| Ben25 | 5 | Qualitative check only | No regression |

### 7.3 Comparison Baselines

- FLAN-T5-large zero-shot (current, unmodified)
- Gemma4 zero-shot (2B, strongest current zero-shot)
- sklearn native (TF-IDF+LR, trained per-dataset on train split)

### 7.4 Ablation (optional, if time permits)

- Phase 1 only (no DSPM adaptation) — measures general classification gain
- Phase 2 only (DSPM data directly, no Phase 1 pretraining) — measures data augmentation value
- Template ablation: zero-shot-only vs full template mix — measures template diversity contribution

## 8. File Structure

```
benchmark/
├── train/                              # NEW: training module
│   ├── __init__.py
│   ├── config.py                       # TrainingConfig dataclass, YAML loading
│   ├── data_pipeline.py                # Template application, sampling weights, DataLoader
│   ├── augment.py                      # DSPM data augmentation (3 layers)
│   ├── trainer.py                      # Phase 1 + Phase 2 QLoRA training loops
│   └── merge_adapter.py                # LoRA merge + HF model export
├── config/
│   └── experiments/
│       └── flan-t5-finetune.yaml       # Full training experiment config
├── models/
│   ├── flan_t5.py                      # Existing: add optional finetuned checkpoint path
│   ├── flan_t5_classification.py       # Existing: no changes needed
│   └── flan-t5-finetuned/             # Output dir (git-ignored)
└── scripts/
    └── run_finetune.py                 # CLI entry: python -m benchmark.scripts.run_finetune
```

## 9. Integration with Existing Benchmark

Zero changes required to the existing benchmark evaluation pipeline:
1. Fine-tuned model saved as standard HF `AutoModelForSeq2SeqLM` format
2. `FlanT5ClassificationModel._load_pipeline()` loads any HF model path
3. Add optional `finetuned_path` parameter to `FlanT5ClassificationModel.__init__()` — when provided, loads fine-tuned checkpoint instead of `google/flan-t5-large`
4. Benchmark YAML configs add one optional field: `finetuned_path`

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Augmented DSPM data quality low | Medium | High | TF-IDF similarity filter + manual spot-check 10% of synthetic docs |
| Phase 1 overfits to prompt format | Medium | Medium | 5-template diversity; evaluate on unseen template variants |
| Catastrophic forgetting in Phase 2 | Low | Medium | Lower LR (5e-5), early stopping, validate on Phase 1 datasets after Phase 2 |
| AG News too easy (4 classes, broad) | Low | Low | Weighted lower in sampling; its value is template diversity, not classification difficulty |
| VRAM OOM | Low | High | 8-bit 780M ~800MB; worst-case reduce batch to 4 or max_length to 768 |
