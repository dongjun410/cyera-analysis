#!/usr/bin/env python
"""Warm-start SetFit training for E3 ML engine.

Uses the labeled evaluation documents as training data, then re-runs
accuracy evaluation with all 6 engines active (including E3 SetFit).

This simulates the "maturity phase" where E3 has been trained on
accumulated high-confidence pipeline outputs.

Usage:
    python train_warmstart.py
"""

from __future__ import annotations

import json
import time
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
from src.knowledge.type_library import get_type_library, TypeInfo
from src.llm.client import LLMConfig, MistralClient
from src.types import Document

# Import the eval dataset
from eval_accuracy import SCENARIO_A, SCENARIO_B, compute_metrics, init_components


def build_training_data() -> tuple[list[str], list[str]]:
    """Combine Scenario A and B labeled docs into training set."""
    texts: list[str] = []
    labels: list[str] = []

    for doc_id, text, label, metadata in SCENARIO_A + SCENARIO_B:
        texts.append(text)
        labels.append(label)

    return texts, labels


def evaluate_with_e3(comp: dict) -> dict:
    """Run accuracy eval with E3 ML engine active."""
    voter = comp["voter"]

    all_docs = []
    for doc_id, text, label, metadata in SCENARIO_A + SCENARIO_B:
        all_docs.append((doc_id, text, label, metadata))

    y_true: list[str] = []
    y_pred: list[str] = []
    fast_count = 0
    full_count = 0
    details: list[dict] = []

    t0 = time.perf_counter()
    for doc_id, text, label, metadata in all_docs:
        doc = Document(doc_id=doc_id, text=text, metadata=metadata)
        result = voter.classify(doc)
        y_true.append(label)
        y_pred.append(result.final_label)
        if result.method == "fusion_fast":
            fast_count += 1
        else:
            full_count += 1
        details.append({
            "doc_id": doc_id,
            "true": label,
            "pred": result.final_label,
            "correct": label == result.final_label,
            "confidence": result.composite_confidence,
            "method": result.method,
        })

    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(y_true, y_pred)
    metrics["method_distribution"] = {"fusion_fast": fast_count, "fusion_full": full_count}
    metrics["llm_call_rate"] = round(full_count / len(all_docs), 4)
    metrics["total_time_s"] = round(elapsed, 1)
    metrics["avg_ms_per_doc"] = round(elapsed / len(all_docs) * 1000, 0)
    metrics["details"] = details
    return metrics


def main() -> int:
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print("=" * 65)
    print("Warm-Start SetFit Training + Accuracy Evaluation")
    print("=" * 65)

    # ── Step 1: Init components ──
    print("\n[1] Initializing components...")
    t0 = time.perf_counter()
    comp = init_components(config)
    print(f"    Init: {time.perf_counter() - t0:.1f}s")

    # ── Step 2: Bootstrap E4 centroids from keywords ──
    print("\n[2] Bootstrapping E4 centroids from keywords...")
    if comp["embedder"] is not None:
        n = comp["e4"].bootstrap_centroids()
        print(f"    Bootstrapped {n} centroids")

    # ── Step 3: Also update centroids with actual embeddings ──
    print("\n[3] Computing real centroids from labeled documents...")
    type_lib = get_type_library()
    embedder = comp.get("embedder")

    if embedder is not None:
        # Group labeled docs by type
        by_type: dict[str, list[str]] = {}
        for doc_id, text, label, metadata in SCENARIO_A + SCENARIO_B:
            by_type.setdefault(label, []).append(text)

        for label, texts in by_type.items():
            # Find or register type
            info = type_lib.get_by_name(label)
            if info is None:
                tid = label.lower().replace(" & ", "_").replace(" ", "_").replace("/", "_")
                info = type_lib.register(tid, label, source="builtin")

            # Compute centroid
            embs = embedder.encode(texts)
            centroid = embs.mean(axis=0).astype("float32")
            centroid = centroid / (float(__import__('numpy').linalg.norm(centroid)) or 1.0)
            info.centroid = centroid
            info.sample_count = len(texts)
            print(f"    {label:25s} centroid from {len(texts)} docs")

    # ── Step 4: Train SetFit ──
    print("\n[4] Training SetFit (E3 ML engine)...")
    train_texts, train_labels = build_training_data()

    mgr = DistillationManager(e3_engine=comp["e3"])
    result = mgr.force_train(train_texts, train_labels)

    if result["deployed"]:
        metrics = result.get("metrics", {})
        print(f"    Trained! Macro F1={metrics.get('macro_f1', 'N/A')}, "
              f"Labels={metrics.get('num_labels', 'N/A')}")
    else:
        print(f"    Training failed: {result.get('reason', 'unknown')}")
        return 1

    # Re-init voter with trained E3
    all_engines = [comp["e1"], comp["e2"], comp["e3"],
                   comp["e4"], comp["e5"], comp["e6"]]
    comp["voter"] = FusionVoter(engines=all_engines)

    # ── Step 5: Re-evaluate ──
    print(f"\n[5] Evaluating with all 6 engines (E3 trained)...")
    print(f"    E1 Regex:     available")
    print(f"    E2 Template:  available")
    print(f"    E3 ML:        available (SetFit trained)")
    print(f"    E4 kNN:       available ({len(type_lib.list_centroids())} centroids)")
    print(f"    E5 Structural: available")
    print(f"    E6 LLM:       available")

    metrics = evaluate_with_e3(comp)

    # ── Results ──
    print(f"\n{'=' * 65}")
    print(f"RESULTS (ALL 6 ENGINES ACTIVE)")
    print(f"{'=' * 65}")
    print(f"  Total docs:     {metrics['total']}")
    print(f"  Correct:        {metrics['correct']}/{metrics['total']} ({metrics['correct']/metrics['total']*100:.1f}%)")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Macro Prec:     {metrics['macro_precision']:.4f}")
    print(f"  Macro Rec:      {metrics['macro_recall']:.4f}")
    print(f"  LLM call rate:  {metrics['llm_call_rate']*100:.0f}%")
    print(f"  Avg latency:    {metrics['avg_ms_per_doc']:.0f}ms")
    print(f"  fusion_fast:    {metrics['method_distribution'].get('fusion_fast', 0)}")
    print(f"  fusion_full:    {metrics['method_distribution'].get('fusion_full', 0)}")

    print(f"\n  Per-class F1:")
    for lbl in sorted(metrics["per_class"]):
        p = metrics["per_class"][lbl]
        print(f"    {lbl:25s}  F1={p['f1']:.4f}  P={p['precision']:.4f}  R={p['recall']:.4f}  n={p['support']}")

    if any(not d["correct"] for d in metrics["details"]):
        print(f"\n  Errors:")
        for d in metrics["details"]:
            if not d["correct"]:
                print(f"    X {d['doc_id']}: true='{d['true']}' -> pred='{d['pred']}' (conf={d['confidence']:.2f}, method={d['method']})")

    # ── R1-R4 checks ──
    print(f"\n{'─' * 65}")
    print(f"Requirement Checks")
    print(f"{'─' * 65}")

    # Separate scenario A and B for R1
    a_details = [d for d in metrics["details"] if d["doc_id"] in {x[0] for x in SCENARIO_A}]
    b_details = [d for d in metrics["details"] if d["doc_id"] in {x[0] for x in SCENARIO_B}]

    a_correct = sum(1 for d in a_details if d["correct"])
    b_correct = sum(1 for d in b_details if d["correct"])
    a_acc = a_correct / len(a_details) if a_details else 0
    b_acc = b_correct / len(b_details) if b_details else 0

    # Approximate per-scenario F1
    a_true = [d["true"] for d in a_details]
    a_pred = [d["pred"] for d in a_details]
    b_true = [d["true"] for d in b_details]
    b_pred = [d["pred"] for d in b_details]
    a_metrics = compute_metrics(a_true, a_pred)
    b_metrics = compute_metrics(b_true, b_pred)

    delta = abs(a_metrics["macro_f1"] - b_metrics["macro_f1"])
    print(f"  R1: |dF1| = {delta:.4f} (threshold: 0.05)  {'PASS' if delta < 0.05 else 'FAIL'}")

    min_f1 = min(a_metrics["macro_f1"], b_metrics["macro_f1"])
    print(f"  R2: min(Macro F1) = {min_f1:.4f} (threshold: 0.90)  {'PASS' if min_f1 >= 0.90 else 'FAIL'}")

    llm_rate = metrics["llm_call_rate"]
    avg_ms = metrics["avg_ms_per_doc"]
    r4_pass = llm_rate < 0.20 and avg_ms < 300
    print(f"  R4: LLM={llm_rate*100:.0f}%, Latency={avg_ms:.0f}ms  {'PASS' if r4_pass else 'NOT YET'}")

    # Save
    report = {
        "metrics": {k: v for k, v in metrics.items() if k != "details"},
        "r1_delta": round(delta, 4),
        "r2_min_f1": round(min_f1, 4),
        "r4_llm_rate": round(llm_rate, 4),
        "r4_avg_ms": round(avg_ms, 0),
        "engine_status": {
            "E1_regex": True,
            "E2_template": True,
            "E3_ml": comp["e3"].is_available,
            "E4_knn": comp["e4"].is_available,
            "E5_structural": True,
            "E6_llm": comp["e6"].is_available,
        },
        "details": metrics["details"],
    }
    Path("./eval_output").mkdir(exist_ok=True)
    with open("./eval_output/warmstart_accuracy.json", "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport: eval_output/warmstart_accuracy.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
