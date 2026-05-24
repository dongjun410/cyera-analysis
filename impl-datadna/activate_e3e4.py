#!/usr/bin/env python
"""Activate E3 (SetFit) + E4 (real centroids) and measure improvement.

Uses cxh5types as labeled source:
  - Train E3 SetFit on 200 docs (stratified 70/30 split)
  - Compute E4 centroids from training docs
  - Evaluate on held-out test set (~58 docs)
  - Compare LLM call rate and accuracy vs cold-start baseline
"""

from __future__ import annotations

import json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "src"))
from cyera_bench.datasets.cxh5types import Cxh5typesDataset

from src.distillation.manager import DistillationManager
from src.embeddings.bge_m3 import BgeM3Embedder
from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine
from src.fusion.voter import FusionVoter
from src.knowledge.type_library import get_type_library
from src.llm.client import LLMConfig, MistralClient
from src.types import Document

# ── Load cxh5types ──
ds = Cxh5typesDataset()
texts, label_dicts = ds.load()
labels = [ld["l1"] for ld in label_dicts]
print(f"Loaded cxh5types: {len(texts)} docs, {len(set(labels))} classes")
for lbl, cnt in Counter(labels).most_common():
    print(f"  {lbl}: {cnt}")

# ── Stratified train/test split ──
rng = np.random.RandomState(42)
by_class = defaultdict(list)
for i, (t, l) in enumerate(zip(texts, labels)):
    by_class[l].append((t, l))

train_docs, test_docs = [], []
for lbl, items in by_class.items():
    rng.shuffle(items)
    split = max(1, int(len(items) * 0.2))  # 20% test
    test_docs.extend(items[:split])
    train_docs.extend(items[split:])

print(f"\nTrain: {len(train_docs)} docs, Test: {len(test_docs)} docs")
for lbl, cnt in Counter(l for _, l in train_docs).most_common():
    print(f"  Train {lbl}: {cnt}")
for lbl, cnt in Counter(l for _, l in test_docs).most_common():
    print(f"  Test  {lbl}: {cnt}")

# ── Init components ──
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

type_lib = get_type_library()
# Replace with cxh5types labels
for info in list(type_lib._types.values()):
    type_lib.remove(info.type_id)
for lbl in sorted(set(labels)):
    tid = lbl.lower().replace(" ", "_").replace("/", "_")[:50]
    type_lib.register(tid, lbl, source="builtin")

e1 = E1RegexEngine()
e2 = E2TemplateEngine()
e3 = E3MLEngine()

emb_cfg = config.get("embedding", {})
embedder = BgeM3Embedder(
    model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
    device=emb_cfg.get("device", "cuda"), batch_size=32,
)
e4 = E4kNNEngine(embedder=embedder, type_library=type_lib, min_types=1)
e5 = E5StructuralEngine(type_library=type_lib)

llm_cfg = config.get("llm", {})
llm = MistralClient(LLMConfig(
    api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
    model=llm_cfg.get("model", "qwen2.5:7b"),
    quantization="4bit", temperature=0.3,
))
e6 = E6LLMEngine(llm_client=llm, type_library=type_lib)

# ── BASELINE: Cold-start (E3 unavailable, E4 keyword centroids) ──
print("\n" + "=" * 60)
print("BASELINE: Cold-start (E3 unavailable, E4 keyword centroids)")
print("=" * 60)
voter_baseline = FusionVoter(engines=[e1, e2, e3, e4, e5, e6])
e4.bootstrap_centroids()
print(f"  E3: {e3.is_available}, E4: {e4.is_available}")

def evaluate(voter, docs, name):
    y_true, y_pred = [], []
    fast_n, full_n = 0, 0
    t0 = time.perf_counter()
    for text, label in docs:
        doc = Document(doc_id=f"{name}_{label}", text=text, metadata={"file_type": ".txt"})
        result = voter.classify(doc)
        y_true.append(label)
        y_pred.append(result.final_label)
        if result.method == "fusion_fast": fast_n += 1
        else: full_n += 1
    elapsed = time.perf_counter() - t0

    # Per-class F1
    all_cls = sorted(set(y_true))
    per_class = {}
    for cls in all_cls:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[cls] = {"f1": round(f1, 4), "prec": round(prec, 4),
                          "rec": round(rec, 4), "n": sum(1 for t in y_true if t == cls)}
    macro_f1 = sum(per_class[c]["f1"] for c in all_cls) / len(all_cls) if all_cls else 0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return {
        "macro_f1": round(macro_f1, 4), "accuracy": round(correct / len(y_true), 4),
        "correct": correct, "total": len(y_true),
        "llm_rate": round(full_n / len(docs), 4) if docs else 0,
        "fast": fast_n, "full": full_n,
        "avg_ms": round(elapsed / len(docs) * 1000, 0) if docs else 0,
        "per_class": per_class,
    }

baseline_train = evaluate(voter_baseline, train_docs, "train")
baseline_test = evaluate(voter_baseline, test_docs, "test")
print(f"  Test: Acc={baseline_test['accuracy']*100:.1f}%, MacroF1={baseline_test['macro_f1']:.4f}, "
      f"LLM={baseline_test['llm_rate']*100:.0f}%, fusion_fast={baseline_test['fast']}, "
      f"avg={baseline_test['avg_ms']}ms")
for cls, m in baseline_test["per_class"].items():
    print(f"    {cls:35s} F1={m['f1']:.4f}")

# ── ACTIVATE E4: Real centroids from training data ──
print("\n" + "=" * 60)
print("ACTIVATE E4: Computing real centroids from training data")
print("=" * 60)
# Move BGE-M3 to CPU to avoid OOM (GPU occupied by Ollama qwen2.5:7b)
embedder._model.to("cpu")
for lbl, items in by_class.items():
    train_texts_for_lbl = [t for t, l in train_docs if l == lbl]
    embs = embedder.encode(train_texts_for_lbl)
    centroid = embs.mean(axis=0).astype("float32")
    centroid = centroid / (float(np.linalg.norm(centroid)) or 1.0)
    info = type_lib.get_by_name(lbl)
    if info:
        info.centroid = centroid
        info.sample_count = len(train_texts_for_lbl)
    print(f"  {lbl:35s} centroid from {len(train_texts_for_lbl)} docs")
# Move back to GPU for evaluation
embedder._model.to("cuda")

e4_activated = E4kNNEngine(embedder=embedder, type_library=type_lib, min_types=1)
print(f"  E4: {e4_activated.is_available} ({len(type_lib.list_centroids())} centroids)")

# ── ACTIVATE E3: Train SetFit on training data ──
print("\n" + "=" * 60)
print("ACTIVATE E3: Training SetFit on training data")
print("=" * 60)

# Free GPU memory: release BGE-M3 embedder before SetFit training
# BGE-M3 (~4GB) + SetFit training (loads its own BGE-M3 ~4GB + training ~4GB)
# = ~12GB, OOM on 12GB GPU. Release first, then reload after training.
print("  Releasing BGE-M3 embedder to free GPU memory...")
import gc, torch
del embedder
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"  GPU memory after release: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, "
      f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved")

train_texts_list = [t for t, l in train_docs]
train_labels_list = [l for t, l in train_docs]
print(f"  Training on {len(train_texts_list)} docs")

e3_activated = E3MLEngine()
# Release BGE-M3 from GPU, train SetFit, then reload
# Ollama qwen2.5:7b stays (5GB), BGE-M3 released, SetFit trains in its own space
print("  Releasing BGE-M3 again for SetFit training...")
del embedder
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

mgr = DistillationManager(e3_engine=e3_activated, min_deploy_f1=0.50)
result = mgr.force_train(train_texts_list, train_labels_list)
gc.collect()
torch.cuda.empty_cache()

# Reload BGE-M3 on CPU for E4 (GPU is full: qwen2.5 + SetFit)
# E4 only does single cosine similarity per doc, CPU is fast enough
print("  Reloading BGE-M3 on CPU for E4 evaluation...")
embedder = BgeM3Embedder(
    model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
    device="cpu", batch_size=1,
)
print(f"  Deployed: {result['deployed']}, F1: {result.get('metrics', {}).get('macro_f1', 'N/A')}")
if not result["deployed"]:
    print(f"  Reason: {result['reason']}")

# ── FULL FUSION with E3+E4 activated ──
print("\n" + "=" * 60)
print("FULL FUSION (E3+E4 activated)")
print("=" * 60)
voter_full = FusionVoter(engines=[e1, e2, e3_activated, e4_activated, e5, e6])
print(f"  E1: available, E2: available, E3: {e3_activated.is_available}, "
      f"E4: {e4_activated.is_available}, E5: available, E6: available")

full_test = evaluate(voter_full, test_docs, "test")

print(f"\n  Test set ({len(test_docs)} docs):")
print(f"  Acc={full_test['accuracy']*100:.1f}%, MacroF1={full_test['macro_f1']:.4f}")
print(f"  LLM rate={full_test['llm_rate']*100:.0f}%, fusion_fast={full_test['fast']}, "
      f"fusion_full={full_test['full']}, avg={full_test['avg_ms']}ms")

for cls, m in full_test["per_class"].items():
    print(f"    {cls:35s} F1={m['f1']:.4f}")

# ── COMPARISON ──
print("\n" + "=" * 60)
print("COMPARISON: Cold-start vs E3+E4 Activated")
print("=" * 60)
for metric in ["macro_f1", "accuracy", "llm_rate", "fast", "full", "avg_ms"]:
    b = baseline_test[metric]
    f = full_test[metric]
    delta = f - b if isinstance(b, (int, float)) else 0
    sign = "+" if delta > 0 else ""
    print(f"  {metric:12s}: baseline={b}, activated={f} ({sign}{delta})")

# Save report
Path("./eval_output").mkdir(exist_ok=True)
report = {
    "dataset": "cxh5types",
    "train_size": len(train_docs), "test_size": len(test_docs),
    "baseline": {k: v for k, v in baseline_test.items() if k != "per_class"},
    "activated": {k: v for k, v in full_test.items() if k != "per_class"},
    "baseline_per_class": baseline_test["per_class"],
    "activated_per_class": full_test["per_class"],
    "e3_deployed": e3_activated.is_available,
    "e4_active": e4_activated.is_available,
}
with open("./eval_output/activate_e3e4.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nReport: eval_output/activate_e3e4.json")
