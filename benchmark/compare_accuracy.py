"""Accuracy-only comparison on a shared 80/20 train/test split.

All models evaluated on the SAME hold-out test set = fair comparison.
"""
from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from cyera_bench.datasets.ben25 import Ben25Dataset
from cyera_bench.datasets.cxh5types import Cxh5typesDataset
from cyera_bench.datasets.dspm27 import Dspm27Dataset
from cyera_bench.models.doc_classifier_sklearn import DocClassifierSklearnModel
from cyera_bench.models.gemma_doc_label import GemmaDocLabelModel
from cyera_bench.models.flan_t5_classification import FlanT5ClassificationModel

DATASETS = {
    "ben25": Ben25Dataset,
    "cxh5types": Cxh5typesDataset,
    "dspm27": Dspm27Dataset,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2


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
    n = len(y_true_l1)
    if n == 0:
        return {}
    l1_correct = sum(1 for t, p in zip(y_true_l1, y_pred_l1) if t == p)
    l2_correct = sum(1 for t, p in zip(y_true_l2, y_pred_l2) if t == p)
    correct_l1_idx = [i for i in range(n) if y_true_l1[i] == y_pred_l1[i]]
    l2_given = (sum(1 for i in correct_l1_idx if y_true_l2[i] == y_pred_l2[i]) / len(correct_l1_idx)
                if correct_l1_idx else 0.0)

    from sklearn.metrics import classification_report
    l1_report = classification_report(y_true_l1, y_pred_l1, output_dict=True, zero_division=0)
    return {
        "l1_acc": l1_correct / n,
        "l2_acc": l2_correct / n,
        "l2_given_correct_l1": l2_given,
        "macro_l1_f1": l1_report.get("macro avg", {}).get("f1-score", 0.0),
    }


def evaluate_model(model, eval_texts, l1_opts, l2_opts, label=""):
    """Collect predictions and latencies for a model on eval_texts."""
    y_pred_l1: List[str] = []
    y_pred_l2: List[str] = []
    latencies: List[float] = []

    for text in eval_texts:
        t0 = time.perf_counter()
        try:
            preds = model.predict_labels([text], l1_opts, l2_opts)
            y_pred_l1.append(preds[0]["l1"])
            y_pred_l2.append(preds[0]["l2"])
        except Exception:
            y_pred_l1.append("error")
            y_pred_l2.append("error")
        latencies.append(time.perf_counter() - t0)

    return y_pred_l1, y_pred_l2, latencies


def main():
    results: List[dict] = []

    for ds_name, ds_cls in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load + split
        ds = ds_cls()
        texts, labels = ds.load()
        y_l1_all = [l["l1"] for l in labels]

        try:
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                stratify=y_l1_all,
            )
        except ValueError:
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            )
        print(f"  Full: {len(texts)} | Train: {len(train_texts)} | Test: {len(test_texts)}")

        # Build label options from FULL dataset (needed for Gemma/FlanT5 prompts)
        l1_opts, l2_opts = build_label_options(labels)
        y_true_l1 = [l["l1"] for l in test_labels]
        y_true_l2 = [l["l2"] for l in test_labels]
        print(f"  L1 categories: {len(l1_opts)}")

        # ---- sklearn (CPU, trained on train set) ----
        print(f"\n  [sklearn] Training on {len(train_texts)} docs...")
        sk = DocClassifierSklearnModel(device="cpu")
        t0 = time.perf_counter()
        sk.fit(train_texts, train_labels)
        fit_time = time.perf_counter() - t0
        print(f"  [sklearn] Trained in {fit_time:.1f}s")
        sk_pred_l1, sk_pred_l2, sk_lat = evaluate_model(sk, test_texts, l1_opts, l2_opts)
        sk_acc = compute_accuracy(y_true_l1, sk_pred_l1, y_true_l2, sk_pred_l2)

        # ---- FlanT5 large (GPU) ----
        print(f"  [flan-t5-large] Evaluating on {len(test_texts)} docs...")
        flan = FlanT5ClassificationModel(variant="large", device="cuda")
        flan_pred_l1, flan_pred_l2, flan_lat = evaluate_model(flan, test_texts, l1_opts, l2_opts)
        flan_acc = compute_accuracy(y_true_l1, flan_pred_l1, y_true_l2, flan_pred_l2)

        # ---- Gemma4 (GPU) ----
        print(f"  [gemma-doc-label] Evaluating on {len(test_texts)} docs...")
        gemma = GemmaDocLabelModel(device="cuda")
        gemma_pred_l1, gemma_pred_l2, gemma_lat = evaluate_model(gemma, test_texts, l1_opts, l2_opts)
        gemma_acc = compute_accuracy(y_true_l1, gemma_pred_l1, y_true_l2, gemma_pred_l2)

        results.append({
            "dataset": ds_name,
            "train_n": len(train_texts),
            "test_n": len(test_texts),
            "sklearn": sk_acc,
            "flan_t5_large": flan_acc,
            "gemma": gemma_acc,
        })

        # Per-dataset summary
        print(f"\n  --- {ds_name} Results (same {len(test_texts)} test docs) ---")
        print(f"  {'Model':<25} {'L1 Acc':>8} {'L2 Acc':>8} {'L2@L1':>8} {'Macro L1 F1':>12} {'Latency(med)':>14}")
        print(f"  {'-'*75}")
        for name, acc, lat in [
            ("sklearn (trained)", sk_acc, sk_lat),
            ("flan-t5-large", flan_acc, flan_lat),
            ("gemma-doc-label", gemma_acc, gemma_lat),
        ]:
            med = statistics.median(lat) if lat else 0
            print(f"  {name:<25} {acc['l1_acc']:>8.4f} {acc['l2_acc']:>8.4f} {acc['l2_given_correct_l1']:>8.4f} {acc['macro_l1_f1']:>12.4f} {med:>13.3f}s")

    # Cross-dataset summary
    print(f"\n{'='*90}")
    print(f"  FINAL SUMMARY — All models on same test split (80/20, random_state={RANDOM_STATE})")
    print(f"{'='*90}")
    print(f"  {'Dataset':<15} {'Train':>6} {'Test':>6} {'sklearn L1':>11} {'FlanT5-L L1':>12} {'Gemma4 L1':>11}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['dataset']:<15} {r['train_n']:>6} {r['test_n']:>6} "
              f"{r['sklearn']['l1_acc']:>11.4f} {r['flan_t5_large']['l1_acc']:>12.4f} {r['gemma']['l1_acc']:>11.4f}")

    print(f"\n  {'Dataset':<15} {'Train':>6} {'Test':>6} {'sklearn L2':>11} {'FlanT5-L L2':>12} {'Gemma4 L2':>11}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['dataset']:<15} {r['train_n']:>6} {r['test_n']:>6} "
              f"{r['sklearn']['l2_acc']:>11.4f} {r['flan_t5_large']['l2_acc']:>12.4f} {r['gemma']['l2_acc']:>11.4f}")


if __name__ == "__main__":
    main()
