#!/usr/bin/env python
"""Cross-framework eval for V2.2 — via subprocess to main.py.

Usage:
    python eval_cross.py --dataset twenty_newsgroups --size 300
    python eval_cross.py --dataset cxh5types
    python eval_cross.py --all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "src"))

def _load_twenty_newsgroups(size: int, seed: int) -> tuple[list[str], list[str]]:
    from cyera_bench.datasets.twenty_newsgroups import TwentyNewsgroupsDataset
    ds = TwentyNewsgroupsDataset()
    texts, label_dicts = ds.load()
    labels = [ld["l1"] for ld in label_dicts]
    if len(texts) > size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(texts), size=size, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    return texts, labels

def _load_cxh5types() -> tuple[list[str], list[str]]:
    from cyera_bench.datasets.cxh5types import Cxh5typesDataset
    ds = Cxh5typesDataset()
    texts, label_dicts = ds.load()
    return texts, [ld["l1"] for ld in label_dicts]

def write_temp_docs(texts: list[str], labels: list[str], tmpdir: str) -> tuple[str, dict[str, str]]:
    import hashlib
    input_dir = os.path.join(tmpdir, "input_docs")
    os.makedirs(input_dir, exist_ok=True)
    ground_truth: dict[str, str] = {}
    for i, text in enumerate(texts):
        fname = f"doc_{i:05d}.txt"
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(text)
        # V2.2 generates doc IDs via SHA256(full_path)[:20]
        doc_id = hashlib.sha256(os.path.abspath(fpath).encode()).hexdigest()[:20]
        ground_truth[doc_id] = labels[i]
    return input_dir, ground_truth

def run_v22_subprocess(input_dir: str, output_dir: str, v22_dir: str) -> dict:
    """Run V2.2 main.py as subprocess and parse cluster output."""
    print("  Running V2.2 main.py...")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "main.py", "--input", input_dir, "--output", output_dir],
        cwd=v22_dir,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s (exit={result.returncode})")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        return {"clusters": [], "total_documents": 0, "error": result.stderr[:200]}

    # Find cluster JSON
    json_files = sorted(glob.glob(os.path.join(output_dir, "clusters_*.json")))
    if not json_files:
        # Check stderr for any output about clusters
        for line in result.stderr.split("\n"):
            if "cluster" in line.lower():
                print(f"  V2.2 log: {line[:120]}")
        print(f"  WARNING: No clusters_*.json found in {output_dir}")
        return {"clusters": [], "total_documents": 0, "error": "no_cluster_output"}

    with open(json_files[-1], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"  Loaded {json_files[-1]}: {data.get('num_clusters', len(data.get('clusters', [])))} clusters")
    return data

def evaluate(results: dict, ground_truth: dict[str, str]) -> dict:
    doc_labels: dict[str, str] = {}
    clusters_list = results.get("clusters", [])
    if isinstance(clusters_list, dict):
        clusters_list = list(clusters_list.values())

    for c in clusters_list:
        if isinstance(c, dict):
            label = c.get("llm_label") or c.get("keywords_label") or f"cluster_{c.get('cluster_id', '?')}"
            doc_ids = c.get("doc_ids", [])
        else:
            label = getattr(c, "llm_label", None) or getattr(c, "keywords_label", None) or f"cluster_{getattr(c, 'cluster_id', '?')}"
            doc_ids = getattr(c, "doc_ids", [])
        for did in doc_ids:
            doc_labels[did] = label

    common_ids = sorted(set(doc_labels.keys()) & set(ground_truth.keys()))
    if not common_ids:
        return {"error": "no_common_ids", "num_docs_evaluated": 0, "macro_f1": 0.0}

    y_true = [ground_truth[did] for did in common_ids]
    y_pred = [doc_labels[did] for did in common_ids]

    # Cluster purity
    cluster_docs_map: dict[str, list[str]] = defaultdict(list)
    for did, lbl in doc_labels.items():
        cluster_docs_map[lbl].append(did)
    purities, sizes = [], []
    for lbl, dids in cluster_docs_map.items():
        true_lbls = [ground_truth[d] for d in dids if d in ground_truth]
        if not true_lbls: continue
        mc = Counter(true_lbls).most_common(1)[0][1]
        purities.append(mc / len(true_lbls))
        sizes.append(len(true_lbls))
    weighted_purity = sum(p * s for p, s in zip(purities, sizes)) / sum(sizes) if sizes else 0.0

    # ARI / NMI
    cluster_ids = [doc_labels.get(did, "no_cluster") for did in common_ids]
    true_codes, cmap = [], {}
    for s in y_true:
        if s not in cmap: cmap[s] = len(cmap)
        true_codes.append(cmap[s])
    cluster_codes, cmap2 = [], {}
    for s in cluster_ids:
        if s not in cmap2: cmap2[s] = len(cmap2)
        cluster_codes.append(cmap2[s])
    ari = adjusted_rand_score(true_codes, cluster_codes)
    nmi = normalized_mutual_info_score(true_codes, cluster_codes)

    # Majority-vote mapping
    mapping: dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        mapping[p][t] += 1
    pred_to_true = {p: c.most_common(1)[0][0] for p, c in mapping.items()}

    all_classes = sorted(set(y_true))
    per_class = {}
    for cls in all_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and pred_to_true.get(p) == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and pred_to_true.get(p) == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and pred_to_true.get(p) != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                          "support": sum(1 for t in y_true if t == cls)}

    n = len(all_classes)
    macro_f1 = sum(per_class[c]["f1"] for c in all_classes) / n if n else 0
    correct = sum(1 for t, p in zip(y_true, y_pred) if pred_to_true.get(p) == t)

    return {
        "num_docs_evaluated": len(common_ids),
        "num_clusters": len(cluster_docs_map),
        "cluster_purity_weighted": round(weighted_purity, 4),
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {c: per_class[c]["f1"] for c in all_classes},
        "accuracy": round(correct / len(common_ids), 4) if common_ids else 0,
        "correct": correct,
        "total": len(common_ids),
        "detail": per_class,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["twenty_newsgroups", "cxh5types"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./eval_output/")
    args = parser.parse_args()

    v22_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output, exist_ok=True)

    datasets_to_run = []
    if args.all or args.dataset == "twenty_newsgroups":
        datasets_to_run.append(("twenty_newsgroups", args.size, args.seed))
    if args.all or args.dataset == "cxh5types":
        datasets_to_run.append(("cxh5types", 0, args.seed))

    results_all = {}
    for ds_name, size, seed in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"V2.2 Eval: {ds_name}")
        print(f"{'='*60}")

        if ds_name == "twenty_newsgroups":
            texts, labels = _load_twenty_newsgroups(size, seed)
        else:
            texts, labels = _load_cxh5types()
        print(f"  {len(texts)} docs, {len(set(labels))} classes")

        tmpdir = tempfile.mkdtemp(prefix="v22eval_")
        try:
            input_dir, ground_truth = write_temp_docs(texts, labels, tmpdir)
            output_dir = os.path.join(tmpdir, "output")

            results = run_v22_subprocess(input_dir, output_dir, v22_dir)
            metrics = evaluate(results, ground_truth)
            results_all[ds_name] = metrics

            if "error" in metrics:
                print(f"\n  ERROR: {metrics['error']}")
                results_all[ds_name] = {"macro_f1": 0.0, "error": metrics["error"]}
                continue
            print(f"\n  Docs evaluated: {metrics['num_docs_evaluated']}")
            print(f"  Clusters:       {metrics['num_clusters']}")
            print(f"  Purity (w):     {metrics['cluster_purity_weighted']:.4f}")
            print(f"  ARI:            {metrics['adjusted_rand_index']:.4f}")
            print(f"  NMI:            {metrics['normalized_mutual_info']:.4f}")
            print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
            print(f"  Accuracy (maj): {metrics['accuracy']*100:.1f}%")

            if metrics.get("detail"):
                print(f"\n  Per-class F1:")
                for cls in sorted(metrics["detail"]):
                    d = metrics["detail"][cls]
                    print(f"    {cls[:50]:50s} F1={d['f1']:.4f} n={d['support']}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if len(results_all) == 2:
        a = results_all.get("twenty_newsgroups", {})
        b = results_all.get("cxh5types", {})
        delta = abs(a.get("macro_f1", 0) - b.get("macro_f1", 0))
        print(f"\n{'='*60}")
        print(f"R1: |dF1| = {delta:.4f}  {'PASS' if delta < 0.05 else 'FAIL'}")
        print(f"R2: min(F1) = {min(a.get('macro_f1',0), b.get('macro_f1',0)):.4f}  {'PASS' if min(a.get('macro_f1',0), b.get('macro_f1',0)) >= 0.90 else 'NOT YET'}")

    with open(os.path.join(args.output, "v22_eval.json"), "w") as fh:
        json.dump(results_all, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {args.output}/v22_eval.json")

if __name__ == "__main__":
    main()
