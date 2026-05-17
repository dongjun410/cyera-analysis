from typing import Dict, List

from sklearn.metrics import accuracy_score, classification_report


def compute_classification_metrics(
    y_true_l1: List[str],
    y_pred_l1: List[str],
    y_true_l2: List[str],
    y_pred_l2: List[str],
) -> Dict:
    """Compute document classification metrics.

    Returns dict with l1_accuracy, l2_accuracy, l2_given_correct_l1,
    macro_l1_f1, macro_l2_f1, per_l1_metrics, per_l2_metrics.
    """
    n = len(y_true_l1)

    if n == 0:
        return _empty_result()

    l1_correct = sum(1 for t, p in zip(y_true_l1, y_pred_l1) if t == p)
    l1_acc = l1_correct / n

    l2_correct = sum(1 for t, p in zip(y_true_l2, y_pred_l2) if t == p)
    l2_acc = l2_correct / n

    correct_l1_indices = [i for i in range(n) if y_true_l1[i] == y_pred_l1[i]]
    if correct_l1_indices:
        l2_correct_given = sum(
            1 for i in correct_l1_indices
            if y_true_l2[i] == y_pred_l2[i]
        )
        l2_given_correct = l2_correct_given / len(correct_l1_indices)
    else:
        l2_given_correct = 0.0

    try:
        l1_report = classification_report(
            y_true_l1, y_pred_l1, output_dict=True, zero_division=0
        )
    except Exception:
        l1_report = {}

    try:
        l2_report = classification_report(
            y_true_l2, y_pred_l2, output_dict=True, zero_division=0
        )
    except Exception:
        l2_report = {}

    # Extract per-class metrics, skip aggregator keys
    per_l1: Dict[str, Dict[str, float]] = {}
    for key, m in l1_report.items():
        if key not in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            per_l1[key] = {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1-score"],
                "support": int(m["support"]),
            }

    per_l2: Dict[str, Dict[str, float]] = {}
    for key, m in l2_report.items():
        if key not in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            per_l2[key] = {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1-score"],
                "support": int(m["support"]),
            }

    macro_l1_f1 = l1_report.get("macro avg", {}).get("f1-score", 0.0)
    macro_l2_f1 = l2_report.get("macro avg", {}).get("f1-score", 0.0)

    return {
        "l1_accuracy": l1_acc,
        "l2_accuracy": l2_acc,
        "l2_accuracy_given_correct_l1": l2_given_correct,
        "macro_l1_f1": macro_l1_f1,
        "macro_l2_f1": macro_l2_f1,
        "per_l1_metrics": per_l1,
        "per_l2_metrics": per_l2,
    }


def _empty_result() -> Dict:
    return {
        "l1_accuracy": 0.0,
        "l2_accuracy": 0.0,
        "l2_accuracy_given_correct_l1": 0.0,
        "macro_l1_f1": 0.0,
        "macro_l2_f1": 0.0,
        "per_l1_metrics": {},
        "per_l2_metrics": {},
    }
