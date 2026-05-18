"""Unified benchmark: latency + accuracy across models, datasets, devices.

Usage:
    python benchmark/test_performance.py --model gemma-doc-label --dataset ben25 --device cuda
    python benchmark/test_performance.py --all  # runs all combinations
"""
from __future__ import annotations

import argparse
import datetime
import json
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
REPORT_LINES: List[str] = []
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
REPORT_PATH = f"benchmark/results/unified-perf-{TIMESTAMP}.md"


def build_label_options(labels: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    l1_set = sorted(set(l["l1"] for l in labels if l["l1"]))
    l2_options: Dict[str, List[str]] = {}
    for l1_name in l1_set:
        l2_set = sorted(set(
            l["l2"] for l in labels if l["l1"] == l1_name and l["l2"]
        ))
        l2_options[l1_name] = l2_set
    return l1_set, l2_options


def compute_accuracy(y_true_l1, y_pred_l1, y_true_l2, y_pred_l2):
    """Compute classification accuracy metrics."""
    n = len(y_true_l1)
    if n == 0:
        return {}

    l1_correct = sum(1 for t, p in zip(y_true_l1, y_pred_l1) if t == p)
    l2_correct = sum(1 for t, p in zip(y_true_l2, y_pred_l2) if t == p)
    correct_l1_idx = [i for i in range(n) if y_true_l1[i] == y_pred_l1[i]]
    l2_given = (sum(1 for i in correct_l1_idx if y_true_l2[i] == y_pred_l2[i]) / len(correct_l1_idx)
                if correct_l1_idx else 0.0)

    try:
        from sklearn.metrics import classification_report
        l1_report = classification_report(y_true_l1, y_pred_l1, output_dict=True, zero_division=0)
        l2_report = classification_report(y_true_l2, y_pred_l2, output_dict=True, zero_division=0)
    except Exception:
        l1_report, l2_report = {}, {}

    per_l1 = {}
    for k, m in l1_report.items():
        if k not in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            per_l1[k] = {"prec": m["precision"], "rec": m["recall"], "f1": m["f1-score"], "sup": int(m["support"])}

    per_l2 = {}
    for k, m in l2_report.items():
        if k not in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            per_l2[k] = {"prec": m["precision"], "rec": m["recall"], "f1": m["f1-score"], "sup": int(m["support"])}

    return {
        "l1_acc": l1_correct / n,
        "l2_acc": l2_correct / n,
        "l2_given_correct_l1": l2_given,
        "macro_l1_f1": l1_report.get("macro avg", {}).get("f1-score", 0.0),
        "macro_l2_f1": l2_report.get("macro avg", {}).get("f1-score", 0.0),
        "per_l1": per_l1,
        "per_l2": per_l2,
    }


def run_one(model_key: str, dataset_key: str, device: str, model_variant: str | None = None):
    print(f"\n{'='*60}")
    print(f"  {model_key} @ {dataset_key} | device={device}")
    print(f"{'='*60}")

    # Load dataset
    ds_cls = DATASETS[dataset_key]
    ds = ds_cls()
    texts, labels = ds.load()
    l1_opts, l2_opts = build_label_options(labels)
    y_true_l1 = [l["l1"] for l in labels]
    y_true_l2 = [l["l2"] for l in labels]
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
    warmup_n = min(3, len(texts))
    print(f"  Warming up ({warmup_n} docs)...")
    warmup_start = time.perf_counter()
    for i in range(warmup_n):
        t0 = time.perf_counter()
        try:
            model.predict_labels([texts[i]], l1_opts, l2_opts)
        except Exception as e:
            print(f"  [WARN] warmup doc {i} error: {e}")
        print(f"  warmup [{i + 1}/{warmup_n}] {time.perf_counter() - t0:.1f}s")
    warmup_total = time.perf_counter() - warmup_start
    print(f"  Warmup total: {warmup_total:.1f}s")

    # Measure latency + collect predictions
    n = len(texts)
    all_latencies: List[float] = []
    y_pred_l1: List[str] = []
    y_pred_l2: List[str] = []

    print(f"  Measuring ({n} docs)...")
    for i in range(n):
        t0 = time.perf_counter()
        try:
            preds = model.predict_labels([texts[i]], l1_opts, l2_opts)
            y_pred_l1.append(preds[0]["l1"])
            y_pred_l2.append(preds[0]["l2"])
        except Exception as e:
            print(f"  [WARN] doc {i} error: {e}")
            y_pred_l1.append("error")
            y_pred_l2.append("error")
        all_latencies.append(time.perf_counter() - t0)

        if i < warmup_n:
            continue
        if (i + 1) % 10 == 0 or i == n - 1:
            recent = all_latencies[max(warmup_n, i - 4):i + 1]
            avg = statistics.mean(recent) if recent else 0
            eta = avg * (n - i - 1)
            print(f"  [{i + 1}/{n}] avg {avg:.2f}s/doc | ETA {eta:.0f}s", end="\r")
    print()

    # Latency stats (exclude warmup)
    measure_lat = all_latencies[warmup_n:]
    if not measure_lat:
        print("  No measurements.")
        return

    avg = statistics.mean(measure_lat)
    med = statistics.median(measure_lat)
    mn = min(measure_lat)
    mx = max(measure_lat)
    p95 = sorted(measure_lat)[int(len(measure_lat) * 0.95)] if len(measure_lat) > 1 else measure_lat[0]
    total_chars = sum(len(t) for t in texts[warmup_n:])
    throughput = total_chars / sum(measure_lat) if sum(measure_lat) > 0 else 0

    # Accuracy (all docs)
    acc = compute_accuracy(y_true_l1, y_pred_l1, y_true_l2, y_pred_l2)
    l1_acc = acc.get("l1_acc", 0)
    l2_acc = acc.get("l2_acc", 0)
    l2_given = acc.get("l2_given_correct_l1", 0)
    macro_l1_f1 = acc.get("macro_l1_f1", 0)
    macro_l2_f1 = acc.get("macro_l2_f1", 0)

    print(f"\n  --- Latency ---")
    print(f"  Docs measured:   {len(measure_lat)}")
    print(f"  Avg / Med:       {avg:.2f}s / {med:.2f}s")
    print(f"  Min–Max / P95:   {mn:.2f}s–{mx:.2f}s / {p95:.2f}s")
    print(f"  Throughput:      {throughput:.1f} chars/sec")
    print(f"\n  --- Accuracy ---")
    print(f"  L1: {l1_acc:.4f}  L2: {l2_acc:.4f}  L2@L1: {l2_given:.4f}  F1(L1): {macro_l1_f1:.4f}  F1(L2): {macro_l2_f1:.4f}")

    model_label = model_key + (f"({model_variant})" if model_variant else "")
    RESULTS.append({
        "model": model_label,
        "dataset": dataset_key,
        "device": device,
        "docs": len(measure_lat),
        "avg_s": avg,
        "median_s": med,
        "min_s": mn,
        "max_s": mx,
        "p95_s": p95,
        "chars_per_sec": throughput,
        "l1_acc": l1_acc,
        "l2_acc": l2_acc,
        "l2_given_correct_l1": l2_given,
        "macro_l1_f1": macro_l1_f1,
        "macro_l2_f1": macro_l2_f1,
        "per_l1": acc.get("per_l1", {}),
        "per_l2": acc.get("per_l2", {}),
    })


def generate_report():
    if not RESULTS:
        return

    lines: List[str] = []
    H = lines.append

    H(f"# Unified Benchmark Report — Latency + Accuracy")
    H(f"")
    H(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC+8')}")
    H(f"**Report ID:** unified-perf-{TIMESTAMP}")
    H(f"")
    H(f"## 1. Test Environment")
    H(f"")
    H(f"| Component | Detail |")
    H(f"|-----------|--------|")
    H(f"| CPU | AMD Ryzen 7 9700X (8 Cores / 16 Threads, 3.8 GHz base, ~5.5 GHz boost) |")
    H(f"| RAM | 32 GB |")
    H(f"| GPU | NVIDIA GeForce RTX 5070 (12 GB VRAM, Blackwell sm_120) |")
    H(f"| CUDA | 13.1 |")
    H(f"| PyTorch | 2.13.0.dev20260517+cu130 (nightly) |")
    H(f"| Ollama Model | gemma4:e2b (7.2 GB) |")
    H(f"| Methodology | Direct Ollama /api/generate with display-name prompts, sklearn TF-IDF+LR, FLAN-T5 model.generate() |")
    H(f"")

    # Models under test
    H(f"## 2. Models Under Test")
    H(f"")
    H(f"| # | Model | Params | Backend |")
    H(f"|---|-------|--------|---------|")
    H(f"| 1 | GemmaDocLabelModel | 2B | Ollama gemma4:e2b (HTTP) |")
    H(f"| 2 | DocClassifierSklearnModel | — | TF-IDF + LogisticRegression |")
    H(f"| 3 | FlanT5ClassificationModel base | 250M | google/flan-t5-base |")
    H(f"| 4 | FlanT5ClassificationModel large | 780M | google/flan-t5-large |")
    H(f"")

    # Datasets
    H(f"## 3. Datasets")
    H(f"")
    H(f"| Dataset | Size | Format | Label Source | L1 Count |")
    H(f"|---------|:----:|--------|-------------|:--------:|")
    H(f"| Ben25 | 25 | JSONL text | GPT-5.2 | 9 |")
    H(f"| Dspm27 | 27 | PDF (pdfplumber) | GPT-5.2 | 14 |")
    H(f"| Cxh5types | 258 | JSONL text | Human | 3 |")
    H(f"")

    # Latency table
    H(f"## 4. Latency — Median (seconds per document)")
    H(f"")
    H(f"| Model | Dev | Ben25 | Dspm27 | Cxh5types |")
    H(f"|-------|-----|:-----:|:------:|:---------:|")
    for model_label in sorted(set(r["model"] for r in RESULTS)):
        for dev in ["cuda", "cpu"]:
            row = f"| {model_label} | {dev.upper()} |"
            for ds in ["ben25", "dspm27", "cxh5types"]:
                matches = [r for r in RESULTS if r["model"] == model_label and r["dataset"] == ds and r["device"] == dev]
                if matches:
                    row += f" **{matches[0]['median_s']:.2f}** |"
                else:
                    row += " — |"
            # Only output row if at least one value
            if "—" not in row or row.count("—") < 3:
                H(row)
    H(f"")

    # Accuracy table
    H(f"## 5. Accuracy — L1 / L2 / L2@CorrectL1 / Macro-F1")
    H(f"")
    H(f"| Model | Dev | Ben25 | Dspm27 | Cxh5types |")
    H(f"|-------|-----|-------|--------|-----------|")
    for model_label in sorted(set(r["model"] for r in RESULTS)):
        for dev in ["cuda", "cpu"]:
            row = f"| {model_label} | {dev.upper()} |"
            for ds in ["ben25", "dspm27", "cxh5types"]:
                matches = [r for r in RESULTS if r["model"] == model_label and r["dataset"] == ds and r["device"] == dev]
                if matches:
                    r = matches[0]
                    row += f" {r['l1_acc']:.1%}/{r['l2_acc']:.1%}/{r['l2_given_correct_l1']:.1%}/{r['macro_l1_f1']:.3f} |"
                else:
                    row += " — |"
            if "—" not in row or row.count("—") < 3:
                H(row)
    H(f"")
    H(f"> Format: L1 Accuracy / L2 Accuracy / L2@Correct-L1 / Macro L1 F1")
    H(f"")

    # GPU Speedup
    H(f"## 6. GPU Speedup (CPU median / GPU median)")
    H(f"")
    H(f"| Model | Ben25 | Dspm27 | Cxh5types |")
    H(f"|-------|:-----:|:------:|:---------:|")
    for model_label in sorted(set(r["model"] for r in RESULTS)):
        row = f"| {model_label} |"
        for ds in ["ben25", "dspm27", "cxh5types"]:
            gpu = [r for r in RESULTS if r["model"] == model_label and r["dataset"] == ds and r["device"] == "cuda"]
            cpu = [r for r in RESULTS if r["model"] == model_label and r["dataset"] == ds and r["device"] == "cpu"]
            if gpu and cpu:
                speedup = cpu[0]["median_s"] / gpu[0]["median_s"] if gpu[0]["median_s"] > 0 else 0
                row += f" **{speedup:.1f}x** |"
            else:
                row += " — |"
        if "—" not in row or row.count("—") < 3:
            H(row)
    H(f"")

    # Detailed latency
    H(f"## 7. Detailed Latency Data")
    H(f"")
    H(f"| Model | Dataset | Dev | Docs | Avg(s) | Med(s) | Min(s) | Max(s) | P95(s) |")
    H(f"|-------|---------|-----|:----:|:------:|:------:|:------:|:------:|:------:|")
    for r in sorted(RESULTS, key=lambda x: (x["model"], x["dataset"], x["device"])):
        H(f"| {r['model']} | {r['dataset']} | {r['device']} | {r['docs']} | {r['avg_s']:.2f} | {r['median_s']:.2f} | {r['min_s']:.2f} | {r['max_s']:.2f} | {r['p95_s']:.2f} |")
    H(f"")

    # Detailed accuracy
    H(f"## 8. Detailed Accuracy Data")
    H(f"")
    H(f"| Model | Dataset | Dev | L1 Acc | L2 Acc | L2@L1 | Macro L1 F1 | Macro L2 F1 |")
    H(f"|-------|---------|-----|:------:|:------:|:-----:|:-----------:|:-----------:|")
    for r in sorted(RESULTS, key=lambda x: (x["model"], x["dataset"], x["device"])):
        H(f"| {r['model']} | {r['dataset']} | {r['device']} | {r['l1_acc']:.4f} | {r['l2_acc']:.4f} | {r['l2_given_correct_l1']:.4f} | {r['macro_l1_f1']:.4f} | {r['macro_l2_f1']:.4f} |")
    H(f"")

    # Per-category accuracy for Gemma only (most interesting)
    H(f"## 9. GemmaDocLabelModel — Per-Category Accuracy")
    H(f"")
    for r in RESULTS:
        if not r["model"].startswith("gemma"):
            continue
        per_l1 = r.get("per_l1", {})
        if not per_l1:
            continue
        H(f"### {r['model']} @ {r['dataset']} ({r['device']})")
        H(f"")
        H(f"| L1 Category | Prec | Rec | F1 | Support |")
        H(f"|-------------|:----:|:---:|:---:|:-------:|")
        for cat, m in sorted(per_l1.items(), key=lambda x: -x[1]["f1"]):
            H(f"| {cat} | {m['prec']:.3f} | {m['rec']:.3f} | {m['f1']:.3f} | {int(m['sup'])} |")
        H(f"")

    # Cross-model comparison
    H(f"## 10. Cross-Model Comparison (GPU, Ben25)")
    H(f"")
    H(f"| Model | Latency (med) | L1 Acc | L2 Acc | Trade-off |")
    H(f"|-------|:------------:|:------:|:------:|-----------|")
    for model_label in ["doc-classifier-sklearn", "flan-t5-classification(base)", "flan-t5-classification(large)", "gemma-doc-label"]:
        matches = [r for r in RESULTS if r["model"] == model_label and r["dataset"] == "ben25" and r["device"] == "cuda"]
        if not matches:
            matches = [r for r in RESULTS if r["model"] == model_label and r["dataset"] == "ben25"]
        if matches:
            r = matches[0]
            H(f"| {model_label} | {r['median_s']:.2f}s | {r['l1_acc']:.1%} | {r['l2_acc']:.1%} | speed={r['median_s']:.2f}s, acc={r['l1_acc']:.1%} |")

    H(f"")
    H(f"---")
    H(f"")
    H(f"*Generated by benchmark/test_performance.py | Report: {REPORT_PATH}*")

    # Write to file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [Report saved to {REPORT_PATH}]")


def main():
    ap = argparse.ArgumentParser(description="Unified latency + accuracy benchmark")
    ap.add_argument("--model", choices=sorted(MODELS.keys()))
    ap.add_argument("--dataset", choices=sorted(DATASETS.keys()))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--variant", help="Model variant (e.g. base, large)")
    ap.add_argument("--all", action="store_true", help="Run all combinations")
    args = ap.parse_args()

    if args.all:
        combinations = [
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
        generate_report()
    else:
        if not args.model or not args.dataset:
            ap.error("--model and --dataset required (or use --all)")
        run_one(args.model, args.dataset, args.device, model_variant=args.variant)
        generate_report()


if __name__ == "__main__":
    main()
