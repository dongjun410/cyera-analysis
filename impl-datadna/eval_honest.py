#!/usr/bin/env python
"""Correct SetFit evaluation with train/test split.

CRITICAL: Training data and test data MUST be disjoint.
Previously all 30 docs were used for both training and evaluation,
which is memorization, not generalization. This script:

1. Stratified 70/30 split preserving per-class proportions
2. Train SetFit ONLY on the training split
3. Evaluate ONLY on the held-out test split
4. Report honest generalization accuracy
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

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
from eval_accuracy import SCENARIO_A, SCENARIO_B, compute_metrics, init_components


def stratified_split(
    labeled_docs: list[tuple], test_ratio: float = 0.3, seed: int = 42,
) -> tuple[list, list]:
    """Split labeled docs into train/test, preserving per-class ratios.

    Returns (train_docs, test_docs).
    """
    rng = random.Random(seed)
    by_class: dict[str, list] = defaultdict(list)
    for item in labeled_docs:
        label = item[2]
        by_class[label].append(item)

    train, test = [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_texts_labels(docs: list[tuple]) -> tuple[list[str], list[str]]:
    texts = [item[1] for item in docs]
    labels = [item[2] for item in docs]
    return texts, labels


def evaluate_on_docs(comp: dict, labeled_docs: list[tuple], name: str = "") -> dict:
    """Evaluate fusion voter on a set of labeled documents."""
    voter = comp["voter"]
    y_true, y_pred = [], []
    fast_count, full_count = 0, 0
    details = []

    t0 = time.perf_counter()
    for doc_id, text, label, metadata in labeled_docs:
        doc = Document(doc_id=doc_id, text=text, metadata=metadata)
        result = voter.classify(doc)
        y_true.append(label)
        y_pred.append(result.final_label)
        if result.method == "fusion_fast":
            fast_count += 1
        else:
            full_count += 1
        details.append({
            "doc_id": doc_id, "true": label, "pred": result.final_label,
            "correct": label == result.final_label,
            "confidence": result.composite_confidence, "method": result.method,
        })

    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(y_true, y_pred)
    metrics["method_distribution"] = {"fusion_fast": fast_count, "fusion_full": full_count}
    metrics["llm_call_rate"] = round(full_count / len(labeled_docs), 4) if labeled_docs else 0
    metrics["total_time_s"] = round(elapsed, 1)
    metrics["avg_ms_per_doc"] = round(elapsed / len(labeled_docs) * 1000, 0) if labeled_docs else 0
    metrics["details"] = details
    metrics["name"] = name
    return metrics


def main() -> int:
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print("=" * 70)
    print("CORRECT SetFit Evaluation — Disjoint Train/Test Split")
    print("=" * 70)

    # ── Step 1: Split ──
    all_labeled = list(SCENARIO_A) + list(SCENARIO_B)

    # Show class distribution
    by_class = defaultdict(list)
    for item in all_labeled:
        by_class[item[2]].append(item[0])
    print(f"\nDataset: {len(all_labeled)} documents, {len(by_class)} classes")
    for lbl, ids in sorted(by_class.items()):
        print(f"  {lbl:25s}: {len(ids)} docs  ({', '.join(ids[:3])}{'...' if len(ids)>3 else ''})")

    train_docs, test_docs = stratified_split(all_labeled, test_ratio=0.3, seed=42)

    train_by_class = defaultdict(list)
    for item in train_docs:
        train_by_class[item[2]].append(item[0])
    test_by_class = defaultdict(list)
    for item in test_docs:
        test_by_class[item[2]].append(item[0])

    print(f"\nStratified 70/30 split (seed=42):")
    print(f"  Train: {len(train_docs)} docs")
    for lbl in sorted(train_by_class):
        print(f"    {lbl:25s}: {len(train_by_class[lbl])} docs")
    print(f"  Test:  {len(test_docs)} docs")
    for lbl in sorted(test_by_class):
        print(f"    {lbl:25s}: {len(test_by_class[lbl])} docs")

    # ── Step 2: Baseline (cold-start, no E3) ──
    print(f"\n{'─' * 70}")
    print("BASELINE: Cold-start (E3 untrained, E4 keyword centroids)")
    print(f"{'─' * 70}")
    t0 = time.perf_counter()
    comp_baseline = init_components(config)
    if comp_baseline["embedder"] is not None:
        n = comp_baseline["e4"].bootstrap_centroids()
        print(f"  E4 bootstrapped {n} centroids from keywords")

    baseline_train = evaluate_on_docs(comp_baseline, train_docs, "baseline_train")
    baseline_test = evaluate_on_docs(comp_baseline, test_docs, "baseline_test")

    print(f"\n  Train set ({len(train_docs)} docs): "
          f"Acc={baseline_train['correct']}/{baseline_train['total']} "
          f"({baseline_train['correct']/baseline_train['total']*100:.1f}%), "
          f"Macro F1={baseline_train['macro_f1']:.4f}")
    print(f"  Test set  ({len(test_docs)} docs):  "
          f"Acc={baseline_test['correct']}/{baseline_test['total']} "
          f"({baseline_test['correct']/baseline_test['total']*100:.1f}%), "
          f"Macro F1={baseline_test['macro_f1']:.4f}")

    if baseline_test['correct'] < baseline_test['total']:
        print(f"  Test errors:")
        for d in baseline_test["details"]:
            if not d["correct"]:
                print(f"    X {d['doc_id']}: true='{d['true']}' -> pred='{d['pred']}' "
                      f"(conf={d['confidence']:.2f})")

    # ── Step 3: Compute real E4 centroids from TRAINING data only ──
    print(f"\n{'─' * 70}")
    print("Computing E4 centroids from TRAINING data only (not test data)")
    print(f"{'─' * 70}")
    type_lib = get_type_library()
    embedder = comp_baseline.get("embedder")

    if embedder is not None:
        by_type: dict[str, list[str]] = defaultdict(list)
        for doc_id, text, label, metadata in train_docs:  # TRAIN ONLY
            by_type[label].append(text)

        for label, texts in by_type.items():
            info = type_lib.get_by_name(label)
            if info is None:
                tid = label.lower().replace(" & ", "_").replace(" ", "_").replace("/", "_")
                info = type_lib.register(tid, label, source="builtin")
            import numpy as np
            embs = embedder.encode(texts)
            centroid = embs.mean(axis=0).astype("float32")
            centroid = centroid / (float(np.linalg.norm(centroid)) or 1.0)
            info.centroid = centroid
            info.sample_count = len(texts)
            print(f"  {label:25s} centroid from {len(texts)} train docs")

    # ── Step 4: Train SetFit on TRAINING data only ──
    print(f"\n{'─' * 70}")
    print("Training SetFit on TRAINING data only")
    print(f"{'─' * 70}")

    train_texts, train_labels = build_texts_labels(train_docs)
    comp_trained = init_components(config)
    if comp_trained["embedder"] is not None:
        n = comp_trained["e4"].bootstrap_centroids()
        print(f"  E4 bootstrapped {n} centroids from keywords")

    mgr = DistillationManager(e3_engine=comp_trained["e3"])
    result = mgr.force_train(train_texts, train_labels)

    if not result["deployed"]:
        print(f"  E3 REJECTED: {result.get('reason')}")
        print(f"  Training Macro F1: {result['metrics'].get('macro_f1', 'N/A'):.4f}")
        print(f"  (System continues without E3 — cold-start mode)")
        # Continue evaluation without E3 — this is the correct degraded behavior
    else:
        print(f"  Trained on {len(train_texts)} docs, {result['metrics'].get('num_labels', '?')} labels")
        print(f"  Training Macro F1: {result['metrics'].get('macro_f1', 'N/A'):.4f}")

    # Re-init voter (E3 may or may not be deployed based on quality gate)
    e3_status = "available" if comp_trained["e3"].is_available else "REJECTED (F1 too low)"
    all_engines = [comp_trained["e1"], comp_trained["e2"], comp_trained["e3"],
                   comp_trained["e4"], comp_trained["e5"], comp_trained["e6"]]
    comp_trained["voter"] = FusionVoter(engines=all_engines)

    # ── Step 5: Evaluate on HELD-OUT test set ONLY ──
    print(f"\n{'═' * 70}")
    print("FINAL: Evaluation on HELD-OUT test set (NOT seen during training)")
    print(f"{'═' * 70}")
    print(f"  E1 Regex:     available")
    print(f"  E2 Template:  available")
    print(f"  E3 ML:        {e3_status}")
    print(f"  E4 kNN:       available ({len(type_lib.list_centroids())} centroids from train)")
    print(f"  E5 Structural: available")
    print(f"  E6 LLM:       available")

    trained_test = evaluate_on_docs(comp_trained, test_docs, "trained_test")
    # Also evaluate on train set to show memorization level
    trained_train = evaluate_on_docs(comp_trained, train_docs, "trained_train")

    print(f"\n  {'='*50}")
    print(f"  Train set [{len(train_docs)} docs] — (seen during training)")
    print(f"    Accuracy:    {trained_train['correct']}/{trained_train['total']} "
          f"({trained_train['correct']/trained_train['total']*100:.1f}%)")
    print(f"    Macro F1:    {trained_train['macro_f1']:.4f}")
    print(f"    LLM rate:    {trained_train['llm_call_rate']*100:.0f}%")
    print(f"    Avg latency: {trained_train['avg_ms_per_doc']:.0f}ms")

    print(f"\n  Test set  [{len(test_docs)} docs]  — (HELD OUT, unseen)")
    print(f"    Accuracy:    {trained_test['correct']}/{trained_test['total']} "
          f"({trained_test['correct']/trained_test['total']*100:.1f}%)")
    print(f"    Macro F1:    {trained_test['macro_f1']:.4f}")
    print(f"    LLM rate:    {trained_test['llm_call_rate']*100:.0f}%")
    print(f"    Avg latency: {trained_test['avg_ms_per_doc']:.0f}ms")

    print(f"\n  Per-class F1 on TEST set:")
    for lbl in sorted(trained_test["per_class"]):
        p = trained_test["per_class"][lbl]
        marker = " (not in test)" if p["support"] == 0 else ""
        print(f"    {lbl:25s}  F1={p['f1']:.4f}  "
              f"P={p['precision']:.4f}  R={p['recall']:.4f}  n={p['support']}{marker}")

    if trained_test["correct"] < trained_test["total"]:
        print(f"\n  Test errors:")
        for d in trained_test["details"]:
            if not d["correct"]:
                e3_out = None
                print(f"    X {d['doc_id']}: true='{d['true']}' -> pred='{d['pred']}' "
                      f"(conf={d['confidence']:.2f}, method={d['method']})")
    else:
        print(f"\n  No errors on test set.")

    # ── Requirement checks (on held-out test only!) ──
    print(f"\n{'═' * 70}")
    print("R1-R4 Checks (on HELD-OUT test data only)")
    print(f"{'═' * 70}")

    # Split test docs into scenario A and B for R1
    test_a_ids = {item[0] for item in SCENARIO_A} & {d["doc_id"] for d in trained_test["details"]}
    test_b_ids = {item[0] for item in SCENARIO_B} & {d["doc_id"] for d in trained_test["details"]}
    a_details = [d for d in trained_test["details"] if d["doc_id"] in test_a_ids]
    b_details = [d for d in trained_test["details"] if d["doc_id"] in test_b_ids]

    a_true = [d["true"] for d in a_details]
    a_pred = [d["pred"] for d in a_details]
    b_true = [d["true"] for d in b_details]
    b_pred = [d["pred"] for d in b_details]
    a_m = compute_metrics(a_true, a_pred)
    b_m = compute_metrics(b_true, b_pred)

    delta = abs(a_m["macro_f1"] - b_m["macro_f1"]) if a_m["total"] > 0 and b_m["total"] > 0 else 0
    print(f"  R1 |dF1|: {delta:.4f} (threshold: 0.05)  {'PASS' if delta < 0.05 else 'FAIL'}")

    min_f1 = min(a_m.get("macro_f1", 0), b_m.get("macro_f1", 0))
    print(f"  R2 min(F1): {min_f1:.4f} (threshold: 0.90)  {'PASS' if min_f1 >= 0.90 else 'NOT YET'}")

    llm_rate = trained_test["llm_call_rate"]
    avg_ms = trained_test["avg_ms_per_doc"]
    r4_ok = llm_rate < 0.20 and avg_ms < 300
    print(f"  R4 LLM={llm_rate*100:.0f}%, Lat={avg_ms:.0f}ms  {'PASS' if r4_ok else 'NOT YET'}")

    # ── Save report ──
    report = {
        "method": "disjoint_train_test_split",
        "train_size": len(train_docs),
        "test_size": len(test_docs),
        "seed": 42,
        "baseline_coldstart": {
            "train_macro_f1": baseline_train["macro_f1"],
            "test_macro_f1": baseline_test["macro_f1"],
            "test_correct": f"{baseline_test['correct']}/{baseline_test['total']}",
        },
        "e3_trained": {
            "train_macro_f1": trained_train["macro_f1"],
            "test_macro_f1": trained_test["macro_f1"],
            "test_correct": f"{trained_test['correct']}/{trained_test['total']}",
            "test_llm_rate": trained_test["llm_call_rate"],
            "test_avg_ms": trained_test["avg_ms_per_doc"],
        },
        "r1_delta": round(delta, 4),
        "r2_min_f1": round(min_f1, 4),
        "test_details": trained_test["details"],
        "train_details": trained_train["details"],
        "note": "Test data was NEVER seen during training. These are honest generalization results.",
    }
    Path("./eval_output").mkdir(exist_ok=True)
    with open("./eval_output/honest_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport: eval_output/honest_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
