"""Comprehensive benchmark: flan-t5-large vs gemma-doc-label on GPU.

Datasets: 20 Newsgroups, Ledgar, German-MultiFin
Metrics: L1 accuracy, latency, GPU memory, GPU utilization
"""
from __future__ import annotations

import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, "benchmark/src")
sys.path.insert(0, ".")

from cyera_bench.datasets.german_multifin import GermanMultiFinDataset
from cyera_bench.datasets.ledgar import LedgarDataset
from cyera_bench.datasets.twenty_newsgroups import TwentyNewsgroupsDataset
from cyera_bench.models.flan_t5_classification import FlanT5ClassificationModel
from cyera_bench.models.gemma_doc_label import GemmaDocLabelModel
from benchmark.test_performance import build_label_options, compute_accuracy

import torch

SAMPLES_PER_DATASET = 1000
SEED = 42

DATASETS = {
    "20newsgroups": TwentyNewsgroupsDataset,
    "ledgar": LedgarDataset,
    "german-multifin": GermanMultiFinDataset,
}

MODELS = {
    "flan-t5-large": lambda: FlanT5ClassificationModel(variant="large", device="cuda"),
    "gemma-doc-label": lambda: GemmaDocLabelModel(device="cuda"),
}


def gpu_utilization() -> float:
    """Query GPU utilization via nvidia-smi (0-100)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return -1.0


def gpu_memory_mb() -> tuple:
    """Return (allocated_mb, peak_mb)."""
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.max_memory_allocated() / 1024**2,
    )


def sample_dataset(texts, labels, n):
    """Stratified sample of n documents."""
    if len(texts) <= n:
        return texts, labels

    by_class = defaultdict(list)
    for i, l in enumerate(labels):
        by_class[l["l1"]].append(i)

    per_class = max(1, n // len(by_class))
    indices = []
    for cls, idxs in by_class.items():
        indices.extend(random.sample(idxs, min(per_class, len(idxs))))

    # Top up if needed
    remaining = n - len(indices)
    if remaining > 0:
        all_idx = set(range(len(texts))) - set(indices)
        indices.extend(random.sample(sorted(all_idx), remaining))

    random.shuffle(indices)
    return [texts[i] for i in indices], [labels[i] for i in indices]


def run_benchmark(model, model_name, texts, labels, l1_opts, l2_opts):
    """Run inference and collect all metrics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    yt_l1 = [l["l1"] for l in labels]
    yt_l2 = [l["l2"] for l in labels]

    pred_l1, pred_l2, latencies = [], [], []
    util_samples: List[float] = []

    mem_start, _ = gpu_memory_mb()

    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        p = model.predict_labels([text], l1_opts, l2_opts)
        latencies.append(time.perf_counter() - t0)
        pred_l1.append(p[0]["l1"])
        pred_l2.append(p[0]["l2"])

        # Sample GPU util every 10 docs
        if i % 10 == 0:
            util_samples.append(gpu_utilization())

    torch.cuda.synchronize()
    mem_end, mem_peak = gpu_memory_mb()

    acc = compute_accuracy(yt_l1, pred_l1, yt_l2, pred_l2)

    return {
        **acc,
        "latency_med": statistics.median(latencies),
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_mean": statistics.mean(latencies),
        "gpu_mem_start_mb": mem_start,
        "gpu_mem_end_mb": mem_end,
        "gpu_mem_peak_mb": mem_peak,
        "gpu_util_avg": statistics.mean(util_samples) if util_samples else -1,
    }


def main():
    random.seed(SEED)

    all_results = []

    for ds_name, ds_cls in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        ds = ds_cls()
        texts, labels = ds.load()
        l1_opts, l2_opts = build_label_options(labels)
        print(f"  Full: {len(texts)} docs, {len(l1_opts)} L1 classes")

        s_texts, s_labels = sample_dataset(texts, labels, SAMPLES_PER_DATASET)
        print(f"  Sample: {len(s_texts)} docs")

        for model_name, model_factory in MODELS.items():
            print(f"  [{model_name}] Loading + running {len(s_texts)} docs...", flush=True)
            model = model_factory()

            t_start = time.perf_counter()
            result = run_benchmark(model, model_name, s_texts, s_labels, l1_opts, l2_opts)
            elapsed = time.perf_counter() - t_start

            result["dataset"] = ds_name
            result["model"] = model_name
            result["n_docs"] = len(s_texts)
            result["total_time_s"] = elapsed
            all_results.append(result)

            print(f"    L1={result['l1_acc']:.4f}  F1={result['macro_l1_f1']:.4f}  "
                  f"med={result['latency_med']:.3f}s  p95={result['latency_p95']:.3f}s  "
                  f"mem={result['gpu_mem_peak_mb']:.0f}MB  util={result['gpu_util_avg']:.1f}%")

    # Summary table
    print(f"\n{'='*100}")
    print(f"  SUMMARY — {SAMPLES_PER_DATASET} stratified samples per dataset")
    print(f"{'='*100}")
    header = (f"  {'Dataset':<20} {'Model':<20} {'L1 Acc':>8} {'MacroF1':>8} "
              f"{'Med(s)':>8} {'P95(s)':>8} {'PeakMem':>8} {'GPU%':>6}")
    print(header)
    print(f"  {'-'*90}")
    prev_ds = ""
    for r in all_results:
        ds_label = r["dataset"] if r["dataset"] != prev_ds else ""
        prev_ds = r["dataset"]
        print(f"  {ds_label:<20} {r['model']:<20} {r['l1_acc']:>8.4f} {r['macro_l1_f1']:>8.4f} "
              f"{r['latency_med']:>8.3f} {r['latency_p95']:>8.3f} {r['gpu_mem_peak_mb']:>7.0f}MB {r['gpu_util_avg']:>5.1f}%")

    # Winner per dataset
    print(f"\n  --- Per-Dataset Winner ---")
    for ds_name in DATASETS:
        rs = [r for r in all_results if r["dataset"] == ds_name]
        if len(rs) == 2:
            better = rs[0] if rs[0]["l1_acc"] > rs[1]["l1_acc"] else rs[1]
            worse = rs[1] if better is rs[0] else rs[0]
            print(f"  {ds_name:<20} {better['model']} leads by {better['l1_acc']-worse['l1_acc']:+.4f} L1")


if __name__ == "__main__":
    main()
