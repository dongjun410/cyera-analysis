"""
Held-out dataset generalization experiment.

Compares zero-shot FLAN-T5-large vs fine-tuned FLAN-T5-large on datasets
NOT seen in Phase 1 (20newsgroups, Ledgar, AG News, DBpedia-14, German-MultiFin)
or Phase 2 (Dspm27, Ben25, Cxh5types) training.

Purpose: determine whether fine-tuning improvement transfers to unseen domains,
or is limited to domains seen during training.

Usage:
  python benchmark/eval_heldout.py
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Dict, List, Tuple

from datasets import load_dataset

sys.path.insert(0, "benchmark/src")
from cyera_bench.models.flan_t5_classification import FlanT5ClassificationModel


# ── Held-out datasets ────────────────────────────────────────────────────────
#
# Selection criteria:
#   1. NOT in any Phase 1 or Phase 2 training data
#   2. Different domain, text style, or label structure from training
#   3. Single-label classification (L1 only for simplicity)
#   4. >= 500 test samples for statistical significance
#
# Phase 1 domains: news, legal contracts, German finance, encyclopedia
# Phase 2 domains: enterprise DSPM docs (BIO label: mixed enterprise content)
#
# So these datasets are UNSEEN:
#   - emotion: tweets about emotions (social media, informal)
#   - banking77: banking customer intents (transactional queries, 77 classes)

HELDOUT_DATASETS = {
    "emotion": {
        "hf_id": "dair-ai/emotion",
        "description": "Tweet emotion classification (6 classes)",
        "test_size": 2000,
        "sample_n": 800,  # sample for speed
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "domain": "social media / informal",
        "reason_held_out": "No tweets or emotional content in Phase 1/2 training",
    },
    "banking77": {
        "hf_id": "banking77",
        "description": "Banking customer intent classification (77 classes)",
        "test_size": 3080,
        "sample_n": 1000,
        "labels": None,  # will load from dataset
        "domain": "banking customer service / transactional",
        "reason_held_out": "No customer service queries or banking apps in training. "
                           "Ledgar = legal contract clauses, not banking queries. "
                           "German-MultiFin = German financial docs, not English intents.",
    },
}

RANDOM_SEED = 42
FINETUNED_PATH = "benchmark/models/flan-t5-finetuned/merged"
MAX_SAMPLES_PER_CLASS = 200  # cap to avoid 1-class dominance in stratified sampling


def load_emotion_test() -> Tuple[List[str], List[str], List[str]]:
    """Load emotion dataset: return (texts, true_labels, label_options)."""
    ds = load_dataset("dair-ai/emotion", trust_remote_code=False)
    label_names = ds["train"].features["label"].names
    test = ds["test"]
    texts = test["text"]
    true_labels = [label_names[l] for l in test["label"]]
    return list(texts), true_labels, label_names


def load_banking77_test(n_samples: int = 1000) -> Tuple[List[str], List[str], List[str]]:
    """Load banking77: stratified subsample + return (texts, true_labels, label_options)."""
    ds = load_dataset("banking77", trust_remote_code=False)
    label_names = ds["train"].features["label"].names
    test = ds["test"]

    # Stratified subsample: at most MAX_SAMPLES_PER_CLASS per class
    import random
    random.seed(RANDOM_SEED)

    class_indices: Dict[int, List[int]] = {}
    for i, label_id in enumerate(test["label"]):
        class_indices.setdefault(label_id, []).append(i)

    selected_indices = []
    for label_id, indices in class_indices.items():
        n = min(len(indices), MAX_SAMPLES_PER_CLASS)
        selected_indices.extend(random.sample(indices, n))

    # Cap total to n_samples
    if len(selected_indices) > n_samples:
        selected_indices = random.sample(selected_indices, n_samples)

    selected_indices.sort()
    texts = [test["text"][i] for i in selected_indices]
    true_labels = [label_names[test["label"][i]] for i in selected_indices]

    return texts, true_labels, label_names


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import classification_report

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        "n": n,
    }


def evaluate_model(
    model, texts: List[str], label_options: List[str], label: str
) -> Tuple[List[str], List[float], dict]:
    """Evaluate a single model on test texts. Returns (predictions, latencies, metrics)."""
    print(f"  [{label}] Evaluating {len(texts)} documents...")

    predictions = []
    latencies = []

    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        try:
            preds = model.predict_labels([text], label_options, {})
            predictions.append(preds[0]["l1"])
        except Exception as e:
            predictions.append(f"ERROR:{e}"[:30])
        latencies.append(time.perf_counter() - t0)

        if (i + 1) % 200 == 0:
            print(f"    [{label}] {i+1}/{len(texts)}...")

    return predictions, latencies


def main():
    print("=" * 70)
    print("  HELD-OUT DATASET GENERALIZATION EXPERIMENT")
    print("  Zero-shot FLAN-T5-large vs Fine-tuned FLAN-T5-large")
    print("=" * 70)

    # ── Load models ──────────────────────────────────────────────
    print("\n[1] Loading models...")
    print(f"  Zero-shot: google/flan-t5-large")
    print(f"  Fine-tuned: {FINETUNED_PATH}")

    model_zero = FlanT5ClassificationModel(
        variant="large", device="cuda", max_input_chars=8000,
    )
    model_finetuned = FlanT5ClassificationModel(
        variant="large", device="cuda", max_input_chars=8000,
        finetuned_path=FINETUNED_PATH,
    )
    print("  Models loaded.\n")

    # ── Evaluate each dataset ───────────────────────────────────
    all_results = []

    for ds_key, ds_info in HELDOUT_DATASETS.items():
        print(f"[2] Dataset: {ds_key} — {ds_info['description']}")
        print(f"    Domain: {ds_info['domain']}")
        print(f"    Reason held-out: {ds_info['reason_held_out']}")

        # Load dataset
        if ds_key == "emotion":
            texts, true_labels, label_options = load_emotion_test()
            sample_n = ds_info["sample_n"]
            # Random subsample
            import random
            random.seed(RANDOM_SEED)
            indices = random.sample(range(len(texts)), min(sample_n, len(texts)))
            texts = [texts[i] for i in indices]
            true_labels = [true_labels[i] for i in indices]
        elif ds_key == "banking77":
            texts, true_labels, label_options = load_banking77_test(
                n_samples=ds_info["sample_n"]
            )
        else:
            raise ValueError(f"Unknown dataset: {ds_key}")

        print(f"    Test samples: {len(texts)}")
        print(f"    Label count: {len(label_options)}")

        # Show label distribution
        from collections import Counter
        dist = Counter(true_labels)
        print(f"    Label distribution (top 10):")
        for label, count in dist.most_common(10):
            print(f"      {label}: {count}")

        # Evaluate zero-shot
        print(f"\n    --- Zero-shot FLAN-T5-large ---")
        pred_zero, lat_zero = evaluate_model(
            model_zero, texts, label_options, "zero-shot"
        )
        metrics_zero = compute_metrics(true_labels, pred_zero, label_options)

        # Evaluate fine-tuned
        print(f"\n    --- Fine-tuned FLAN-T5-large ---")
        pred_ft, lat_ft = evaluate_model(
            model_finetuned, texts, label_options, "fine-tuned"
        )
        metrics_ft = compute_metrics(true_labels, pred_ft, label_options)

        # Print per-dataset comparison
        print(f"\n    {'='*50}")
        print(f"    RESULTS: {ds_key}")
        print(f"    {'='*50}")
        print(f"    {'Metric':<20} {'Zero-shot':>12} {'Fine-tuned':>12} {'Delta':>12}")
        print(f"    {'-'*56}")
        for metric_name in ["accuracy", "macro_f1", "weighted_f1"]:
            z = metrics_zero[metric_name]
            f = metrics_ft[metric_name]
            delta = f - z
            sign = "+" if delta >= 0 else ""
            print(f"    {metric_name:<20} {z:>12.4f} {f:>12.4f} {sign}{delta:>11.4f}")

        med_z = statistics.median(lat_zero) if lat_zero else 0
        med_f = statistics.median(lat_ft) if lat_ft else 0
        delta_lat = med_f - med_z
        sign_lat = "+" if delta_lat >= 0 else ""
        print(f"    {'median_latency(s)':<20} {med_z:>12.4f} {med_f:>12.4f} {sign_lat}{delta_lat:>11.4f}")

        all_results.append({
            "dataset": ds_key,
            "description": ds_info["description"],
            "domain": ds_info["domain"],
            "n_samples": len(texts),
            "n_classes": len(label_options),
            "zero_shot": metrics_zero,
            "fine_tuned": metrics_ft,
            "zero_latency_median": med_z,
            "ft_latency_median": med_f,
        })
        print()

    # ── Cross-dataset summary ───────────────────────────────────
    print("=" * 90)
    print("  FINAL SUMMARY — Out-of-Distribution Generalization")
    print("=" * 90)
    print()
    print(f"  {'Dataset':<15} {'Domain':<22} {'Classes':>8} {'N':>6} "
          f"{'Zero L1':>10} {'FT L1':>10} {'Delta':>10} {'Zero F1':>10} {'FT F1':>10} "
          f"{'Zero ms':>10} {'FT ms':>10}")
    print(f"  {'-'*125}")

    for r in all_results:
        z_acc = r["zero_shot"]["accuracy"]
        f_acc = r["fine_tuned"]["accuracy"]
        z_f1 = r["zero_shot"]["macro_f1"]
        f_f1 = r["fine_tuned"]["macro_f1"]
        z_lat = r["zero_latency_median"] * 1000
        f_lat = r["ft_latency_median"] * 1000
        delta = f_acc - z_acc
        sign = "+" if delta >= 0 else ""

        print(f"  {r['dataset']:<15} {r['domain']:<22} {r['n_classes']:>8} {r['n_samples']:>6} "
              f"{z_acc:>10.4f} {f_acc:>10.4f} {sign}{delta:>9.4f} "
              f"{z_f1:>10.4f} {f_f1:>10.4f} "
              f"{z_lat:>10.1f} {f_lat:>10.1f}")

    print()
    print("  Interpretation:")
    print("  - If fine-tuned L1 >= zero-shot L1 on held-out datasets → improvement")
    print("    is TRANSFERABLE (model learned to classify, not just memorize).")
    print("  - If fine-tuned L1 << zero-shot L1 on held-out datasets → improvement")
    print("    is DOMAIN-SPECIFIC (catastrophic forgetting of general ability).")
    print("  - In the latter case, V2.2's clustering+distillation pipeline is the")
    print("    right approach — fine-tuned FLAN-T5 is not a reliable cold-start")
    print("    model for unseen customer domains.")
    print()


if __name__ == "__main__":
    main()
