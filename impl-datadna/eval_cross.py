#!/usr/bin/env python
"""Cross-framework evaluation: benchmark datasets → 6-engine fusion.

Loads benchmark classification datasets, runs them through the fusion
voter, and compares against ground truth labels.

Usage:
    python eval_cross.py --dataset twenty_newsgroups --size 1000
    python eval_cross.py --dataset cxh5types
    python eval_cross.py --all   # run both, compute R1 delta
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add benchmark package to path
_benchmark_src = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmark", "src",
)
if _benchmark_src not in sys.path:
    sys.path.insert(0, _benchmark_src)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Dataset loaders ────────────────────────────────────────────

def _load_twenty_newsgroups(size: int, seed: int) -> tuple[list[str], list[str]]:
    from cyera_bench.datasets.twenty_newsgroups import TwentyNewsgroupsDataset
    ds = TwentyNewsgroupsDataset()
    texts, label_dicts = ds.load()
    labels = [ld["l1"] for ld in label_dicts]
    texts, labels = _sample(texts, labels, size, seed)
    return texts, labels

def _load_cxh5types() -> tuple[list[str], list[str]]:
    from cyera_bench.datasets.cxh5types import Cxh5typesDataset
    ds = Cxh5typesDataset()
    texts, label_dicts = ds.load()
    labels = [ld["l1"] for ld in label_dicts]
    return texts, labels

def _sample(texts, labels, size, seed):
    if len(texts) <= size:
        return texts, labels
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(texts), size=size, replace=False)
    return [texts[i] for i in indices], [labels[i] for i in indices]

# ── Fusion pipeline init ───────────────────────────────────────

def init_components(config: dict) -> dict:
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

    comp: dict[str, Any] = {}
    type_lib = get_type_library()

    comp["e1"] = E1RegexEngine()
    comp["e2"] = E2TemplateEngine()
    comp["e3"] = E3MLEngine()

    try:
        emb = config.get("embedding", {})
        embedder = BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cuda"),
            batch_size=emb.get("batch_size", 32),
        )
        comp["embedder"] = embedder
        comp["e4"] = E4kNNEngine(embedder=embedder, type_library=type_lib, min_types=1)
    except Exception:
        comp["embedder"] = None
        comp["e4"] = E4kNNEngine(embedder=None, type_library=type_lib)

    comp["e5"] = E5StructuralEngine(type_library=type_lib)

    try:
        llm_cfg = config.get("llm", {})
        llm = MistralClient(LLMConfig(
            api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "mistral:7b"),
            quantization="4bit", temperature=0.3,
        ))
        comp["e6"] = E6LLMEngine(llm_client=llm, type_library=type_lib)
    except Exception:
        comp["e6"] = E6LLMEngine(llm_client=None, type_library=type_lib)

    # Bootstrap E4
    if comp["embedder"] is not None:
        comp["e4"].bootstrap_centroids()

    engines = [comp["e1"], comp["e2"], comp["e3"], comp["e4"], comp["e5"], comp["e6"]]
    comp["voter"] = FusionVoter(engines=engines)
    return comp

# ── Metrics ─────────────────────────────────────────────────────

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true))
    per_class = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4), "support": sum(1 for t in y_true if t == lbl)}

    n = len(labels)
    macro_p = sum(p["precision"] for p in per_class.values()) / n if n else 0
    macro_r = sum(p["recall"] for p in per_class.values()) / n if n else 0
    macro_f1 = sum(p["f1"] for p in per_class.values()) / n if n else 0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    return {
        "per_class": per_class,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "total": len(y_true),
        "correct": correct,
        "accuracy": round(correct / len(y_true), 4) if y_true else 0,
    }

# ── Run evaluation ──────────────────────────────────────────────

def run_eval(comp: dict, texts: list[str], labels: list[str], name: str) -> dict:
    from src.types import Document
    from src.knowledge.type_library import get_type_library
    voter = comp["voter"]
    type_lib = get_type_library()

    # ── Register dataset labels into TypeLibrary ──
    # This simulates configuring the system with the customer's actual
    # document taxonomy before running classification.
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        tid = lbl.lower().replace(" ", "_").replace(".", "_").replace("/", "_")[:50]
        existing = type_lib.get(tid)
        if existing is None:
            type_lib.register(tid, lbl, source="builtin")
    logger.info("  Registered %d labels into TypeLibrary (now %d total types)",
                len(unique_labels), type_lib.count)

    y_true, y_pred = [], []
    fast_n, full_n, degraded_n = 0, 0, 0
    details = []
    latencies = []

    t_start = time.perf_counter()
    for i, (text, label) in enumerate(zip(texts, labels)):
        doc = Document(
            doc_id=f"{name}_{i:05d}",
            text=text,
            metadata={"file_type": ".txt", "file_size": len(text)},
        )
        t0 = time.perf_counter()
        result = voter.classify(doc)
        latencies.append((time.perf_counter() - t0) * 1000)

        y_true.append(label)
        y_pred.append(result.final_label)

        if result.method == "fusion_fast":
            fast_n += 1
        else:
            full_n += 1
        if result.degraded:
            degraded_n += 1

        details.append({
            "doc_id": f"{name}_{i:05d}",
            "true": label,
            "pred": result.final_label,
            "correct": label == result.final_label,
            "confidence": result.composite_confidence,
            "method": result.method,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            acc_sofar = sum(1 for d in details if d["correct"]) / len(details)
            logger.info("  [%s] %d/%d (%.1f%% acc, %.0f ms/doc)",
                        name, i + 1, len(texts), acc_sofar * 100,
                        elapsed / (i + 1) * 1000)

    elapsed = time.perf_counter() - t_start
    metrics = compute_metrics(y_true, y_pred)
    metrics["name"] = name
    metrics["method_distribution"] = {"fusion_fast": fast_n, "fusion_full": full_n}
    metrics["llm_call_rate"] = round(full_n / len(texts), 4) if texts else 0
    metrics["degraded_rate"] = round(degraded_n / len(texts), 4) if texts else 0
    metrics["total_time_s"] = round(elapsed, 1)
    metrics["avg_ms_per_doc"] = round(elapsed / len(texts) * 1000, 0) if texts else 0
    metrics["p50_ms"] = round(float(np.percentile(latencies, 50)), 1) if latencies else 0
    metrics["p95_ms"] = round(float(np.percentile(latencies, 95)), 1) if latencies else 0
    metrics["details"] = details
    return metrics

# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-framework fusion eval")
    parser.add_argument("--dataset", choices=["twenty_newsgroups", "cxh5types"],
                        help="Single dataset to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Run both datasets, compute R1")
    parser.add_argument("--size", type=int, default=1000,
                        help="Sample size for 20newsgroups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./eval_output/")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print("=" * 70)
    print("Cross-Framework Fusion Evaluation")
    print("=" * 70)

    # Init
    t0 = time.perf_counter()
    comp = init_components(config)
    init_s = time.perf_counter() - t0
    print(f"\nInit: {init_s:.1f}s")
    print(f"  E1 Regex:     {'available' if comp['e1'].is_available else 'DOWN'}")
    print(f"  E2 Template:  {'available' if comp['e2'].is_available else 'DOWN'}")
    print(f"  E3 ML:        {'available' if comp['e3'].is_available else 'unavailable (not trained)'}")
    print(f"  E4 kNN:       {'available' if comp['e4'].is_available else 'unavailable'}")
    print(f"  E5 Structural:{'available' if comp['e5'].is_available else 'DOWN'}")
    print(f"  E6 LLM:       {'available' if comp['e6'].is_available else 'DOWN'}")

    results = {}
    datasets_to_run = []

    if args.all or args.dataset == "twenty_newsgroups":
        datasets_to_run.append(("twenty_newsgroups", args.size, args.seed))
    if args.all or args.dataset == "cxh5types":
        datasets_to_run.append(("cxh5types", 0, args.seed))

    for ds_name, size, seed in datasets_to_run:
        print(f"\n{'─' * 70}")
        if ds_name == "twenty_newsgroups":
            print(f"Dataset A: 20 Newsgroups (homogeneous text)")
            texts, labels = _load_twenty_newsgroups(size, seed)
        else:
            print(f"Dataset B: Cxh5types (enterprise documents, human-labeled)")
            texts, labels = _load_cxh5types()

        print(f"  {len(texts)} docs, {len(set(labels))} classes")
        class_counts = Counter(labels)
        for lbl, n in class_counts.most_common(5):
            print(f"    {lbl[:50]:50s} {n:5d}")
        if len(class_counts) > 5:
            print(f"    ... and {len(class_counts) - 5} more classes")

        metrics = run_eval(comp, texts, labels, ds_name)
        results[ds_name] = metrics

        print(f"\n  Accuracy:    {metrics['correct']}/{metrics['total']} "
              f"({metrics['accuracy']*100:.1f}%)")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Macro Prec:  {metrics['macro_precision']:.4f}")
        print(f"  Macro Rec:   {metrics['macro_recall']:.4f}")
        print(f"  LLM rate:    {metrics['llm_call_rate']*100:.0f}%")
        print(f"  Avg latency: {metrics['avg_ms_per_doc']:.0f}ms "
              f"(P50={metrics['p50_ms']:.0f}ms, P95={metrics['p95_ms']:.0f}ms)")

        # Top/bottom classes
        per_class = metrics["per_class"]
        sorted_f1 = sorted(per_class.items(), key=lambda x: x[1]["f1"])
        print(f"\n  Best classes (F1):")
        for lbl, m in sorted_f1[-5:]:
            print(f"    {lbl[:45]:45s} F1={m['f1']:.4f} n={m['support']}")
        print(f"  Worst classes (F1):")
        for lbl, m in sorted_f1[:5]:
            print(f"    {lbl[:45]:45s} F1={m['f1']:.4f} n={m['support']}")

    # ── R1 check ──
    if len(results) == 2:
        a = results.get("twenty_newsgroups", {})
        b = results.get("cxh5types", {})
        a_f1 = a.get("macro_f1", 0)
        b_f1 = b.get("macro_f1", 0)
        delta = abs(a_f1 - b_f1)

        print(f"\n{'═' * 70}")
        print(f"R1 Multi-Scenario Stability")
        print(f"{'═' * 70}")
        print(f"  Scenario A (20news, homogeneous text):    Macro F1 = {a_f1:.4f}")
        print(f"  Scenario B (cxh5types, enterprise docs):  Macro F1 = {b_f1:.4f}")
        print(f"  |ΔF1| = {delta:.4f}  (threshold: 0.05)")
        if delta < 0.05:
            print(f"  R1 PASS")
        else:
            print(f"  R1 FAIL — cross-scenario stability not met")

        min_f1 = min(a_f1, b_f1)
        print(f"\nR2 Accuracy")
        print(f"  min(Macro F1) = {min_f1:.4f}  (threshold: 0.90)")
        if min_f1 >= 0.90:
            print(f"  R2 PASS")
        else:
            print(f"  R2 NOT YET — {0.90 - min_f1:.4f} below threshold")

        llm_a = a.get("llm_call_rate", 0)
        llm_b = b.get("llm_call_rate", 0)
        avg_llm = (llm_a * a.get("total", 0) + llm_b * b.get("total", 0)) / \
                  (a.get("total", 0) + b.get("total", 0)) if (a.get("total", 0) + b.get("total", 0)) > 0 else 0
        avg_ms = (a.get("avg_ms_per_doc", 0) + b.get("avg_ms_per_doc", 0)) / 2
        print(f"\nR4 Efficiency (maturity target)")
        print(f"  LLM call rate: {avg_llm*100:.0f}%  (target: <20%)")
        print(f"  Avg latency:   {avg_ms:.0f}ms  (target: <300ms)")
        if avg_llm < 0.20 and avg_ms < 300:
            print(f"  R4 PASS")
        else:
            print(f"  R4 NOT YET — requires E3 trained + E4 centroids populated")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    for ds_name, metrics in results.items():
        # Strip details for the summary file
        summary = {k: v for k, v in metrics.items() if k != "details"}
        report = {
            "dataset": ds_name,
            "metrics": summary,
            "engine_status": {
                "E1_regex": comp["e1"].is_available,
                "E2_template": comp["e2"].is_available,
                "E3_ml": comp["e3"].is_available,
                "E4_knn": comp["e4"].is_available,
                "E5_structural": comp["e5"].is_available,
                "E6_llm": comp["e6"].is_available,
            },
            "details": metrics.get("details", []),
        }
        fname = f"cross_eval_{ds_name}.json"
        with open(output_dir / fname, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nReports saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
