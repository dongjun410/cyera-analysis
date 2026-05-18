"""Performance-only benchmark: latency and throughput across models, datasets, devices.

Usage:
    python benchmark/test_performance.py --model gemma-doc-label --dataset ben25 --device cuda
    python benchmark/test_performance.py --all  # runs all combinations
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Tuple

from cyera_bench.datasets.ben25 import Ben25Dataset
from cyera_bench.datasets.cxh5types import Cxh5typesDataset
from cyera_bench.datasets.dspm27 import Dspm27Dataset
from cyera_bench.models.gemma_doc_label import GemmaDocLabelModel
from cyera_bench.models.doc_classifier_sklearn import DocClassifierSklearnModel
from cyera_bench.models.flan_t5_classification import FlanT5ClassificationModel


DATASETS = {
    "ben25": Ben25Dataset,
    "cxh5types": Cxh5typesDataset,
    "dspm27": Dspm27Dataset,
}

MODELS = {
    "gemma-doc-label": GemmaDocLabelModel,
    "doc-classifier-sklearn": DocClassifierSklearnModel,
    "flan-t5-classification": FlanT5ClassificationModel,
}

RESULTS: List[dict] = []


def build_label_options(labels: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    l1_set = sorted(set(l["l1"] for l in labels if l["l1"]))
    l2_options: Dict[str, List[str]] = {}
    for l1_name in l1_set:
        l2_set = sorted(set(
            l["l2"] for l in labels if l["l1"] == l1_name and l["l2"]
        ))
        l2_options[l1_name] = l2_set
    return l1_set, l2_options


def measure_latency(model, texts: List[str], l1_opts: List[str],
                    l2_opts: Dict[str, List[str]], warmup_n: int = 3) -> List[float]:
    latencies: List[float] = []
    n = len(texts)

    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        try:
            model.predict_labels([text], l1_opts, l2_opts)
        except Exception as e:
            print(f"  [WARN] doc {i} error: {e}")
        latencies.append(time.perf_counter() - t0)

        if i < warmup_n:
            continue  # don't print warmup
        if (i + 1) % 5 == 0 or i == n - 1:
            recent = latencies[max(warmup_n, i - 4):i + 1]
            avg = statistics.mean(recent)
            eta = avg * (n - i - 1)
            print(f"  [{i + 1}/{n}] avg {avg:.2f}s/doc | ETA {eta:.0f}s", end="\r")

    print()
    return latencies


def run_one(model_key: str, dataset_key: str, device: str, model_variant: str | None = None):
    print(f"\n{'='*60}")
    print(f"  {model_key} @ {dataset_key} | device={device}")
    print(f"{'='*60}")

    # Load dataset
    ds_cls = DATASETS[dataset_key]
    ds = ds_cls()
    texts, labels = ds.load()
    l1_opts, l2_opts = build_label_options(labels)
    print(f"  Documents: {len(texts)}")

    # Load model
    model_cls = MODELS[model_key]
    kwargs = {"device": device}
    if model_variant:
        kwargs["variant"] = model_variant
    print(f"  Loading model...")
    model = model_cls(**kwargs)
    print(f"  Model: {model.name}")

    # Warmup
    print(f"  Warming up ({min(3, len(texts))} docs)...")
    warmup_n = 3
    warmup_start = time.perf_counter()
    for i in range(min(warmup_n, len(texts))):
        t0 = time.perf_counter()
        try:
            model.predict_labels([texts[i]], l1_opts, l2_opts)
        except Exception as e:
            print(f"  [WARN] warmup doc {i} error: {e}")
        elapsed = time.perf_counter() - t0
        print(f"  warmup [{i + 1}/{warmup_n}] {elapsed:.1f}s")

    warmup_total = time.perf_counter() - warmup_start
    print(f"  Warmup total: {warmup_total:.1f}s")

    # Measure
    print(f"  Measuring ({len(texts)} docs)...")
    all_latencies = measure_latency(model, texts, l1_opts, l2_opts, warmup_n=0)

    # Stats (exclude warmup)
    measure_lat = all_latencies[warmup_n:] if len(all_latencies) > warmup_n else all_latencies
    if not measure_lat:
        print("  No measurements (all warmup).")
        return

    avg = statistics.mean(measure_lat)
    med = statistics.median(measure_lat)
    mn = min(measure_lat)
    mx = max(measure_lat)
    p95 = sorted(measure_lat)[int(len(measure_lat) * 0.95)]
    total_chars = sum(len(t) for t in texts[warmup_n:])
    throughput = total_chars / sum(measure_lat) if sum(measure_lat) > 0 else 0

    print(f"\n  --- Results ---")
    print(f"  Docs measured:   {len(measure_lat)}")
    print(f"  Total time:      {sum(measure_lat):.1f}s")
    print(f"  Avg latency:     {avg:.2f}s/doc")
    print(f"  Median latency:  {med:.2f}s/doc")
    print(f"  Min / Max:       {mn:.2f}s / {mx:.2f}s")
    print(f"  P95 latency:     {p95:.2f}s")
    print(f"  Throughput:      {throughput:.1f} chars/sec")

    RESULTS.append({
        "model": model_key + (f"({model_variant})" if model_variant else ""),
        "dataset": dataset_key,
        "device": device,
        "docs": len(measure_lat),
        "total_s": sum(measure_lat),
        "avg_s": avg,
        "median_s": med,
        "p95_s": p95,
        "chars_per_sec": throughput,
    })


def print_summary():
    if not RESULTS:
        return
    print(f"\n{'='*80}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    header = f"  {'Model':<35s} {'Dataset':<12s} {'Dev':<5s} {'Docs':>5s} {'Avg(s)':>8s} {'Med(s)':>8s} {'P95(s)':>8s} {'Chars/s':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in RESULTS:
        print(f"  {r['model']:<35s} {r['dataset']:<12s} {r['device']:<5s} {str(r['docs']):>5s} {r['avg_s']:>8.2f} {r['median_s']:>8.2f} {r['p95_s']:>8.2f} {r['chars_per_sec']:>12.1f}")

    # GPU vs CPU comparison
    print(f"\n  --- GPU vs CPU Speedup ---")
    for model_key in sorted(set(r["model"] for r in RESULTS)):
        for dataset_key in sorted(set(r["dataset"] for r in RESULTS)):
            gpu_r = [r for r in RESULTS if r["model"] == model_key and r["dataset"] == dataset_key and r["device"] == "cuda"]
            cpu_r = [r for r in RESULTS if r["model"] == model_key and r["dataset"] == dataset_key and r["device"] == "cpu"]
            if gpu_r and cpu_r:
                speedup = cpu_r[0]["avg_s"] / gpu_r[0]["avg_s"]
                print(f"  {model_key:<35s} {dataset_key:<12s} {speedup:>5.1f}x")


def main():
    ap = argparse.ArgumentParser(description="Performance benchmark")
    ap.add_argument("--model", choices=sorted(MODELS.keys()))
    ap.add_argument("--dataset", choices=sorted(DATASETS.keys()))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--variant", help="Model variant (e.g. base, large)")
    ap.add_argument("--all", action="store_true", help="Run all combinations")
    args = ap.parse_args()

    if args.all:
        combinations = [
            # (model, variant, dataset, devices)
            ("gemma-doc-label", None, "ben25", ["cuda", "cpu"]),
            ("gemma-doc-label", None, "cxh5types", ["cuda", "cpu"]),
            ("gemma-doc-label", None, "dspm27", ["cuda", "cpu"]),
            ("doc-classifier-sklearn", None, "ben25", ["cpu"]),
            ("doc-classifier-sklearn", None, "cxh5types", ["cpu"]),
            ("doc-classifier-sklearn", None, "dspm27", ["cpu"]),
            ("flan-t5-classification", "base", "ben25", ["cuda", "cpu"]),
            ("flan-t5-classification", "base", "cxh5types", ["cuda", "cpu"]),
            ("flan-t5-classification", "base", "dspm27", ["cuda", "cpu"]),
            ("flan-t5-classification", "large", "ben25", ["cuda", "cpu"]),
            ("flan-t5-classification", "large", "cxh5types", ["cuda", "cpu"]),
            ("flan-t5-classification", "large", "dspm27", ["cuda", "cpu"]),
        ]
        for model_key, variant, dataset_key, devices in combinations:
            for device in devices:
                run_one(model_key, dataset_key, device, model_variant=variant)
        print_summary()
    else:
        if not args.model or not args.dataset:
            ap.error("--model and --dataset required (or use --all)")
        run_one(args.model, args.dataset, args.device, model_variant=args.variant)
        print_summary()


if __name__ == "__main__":
    main()
