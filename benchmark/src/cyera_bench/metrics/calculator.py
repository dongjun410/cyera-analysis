from typing import List, Dict, Tuple
import numpy as np
from cyera_bench.types import Entity


def _get_seqeval_report():
    from seqeval.metrics import classification_report as cr
    return cr


class MetricsCalculator:
    def compute_ner_metrics(
        self,
        predictions: List[List[Entity]],
        ground_truth: List[List[Entity]],
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        y_true, y_pred = self._to_bio(predictions, ground_truth)

        if all(len(seq) == 0 for seq in y_true) and all(len(seq) == 0 for seq in y_pred):
            return {}, 0.0

        try:
            report = _get_seqeval_report()(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            return {}, 0.0

        per_entity: Dict[str, Dict[str, float]] = {}
        for key, metrics in report.items():
            if key not in ("micro avg", "macro avg", "weighted avg"):
                per_entity[key] = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                }

        macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
        return per_entity, macro_f1

    def compute_throughput(self, total_tokens: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return total_tokens / total_time_sec

    def compute_latency_percentiles(self, latencies_ms: List[float]) -> Tuple[float, float, float]:
        if not latencies_ms:
            return 0.0, 0.0, 0.0
        arr = np.array(latencies_ms)
        return (
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
            float(np.percentile(arr, 99)),
        )

    def _to_bio(
        self,
        predictions: List[List[Entity]],
        ground_truth: List[List[Entity]],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        y_true: List[List[str]] = []
        y_pred: List[List[str]] = []

        for gt_entities, pred_entities in zip(ground_truth, predictions):
            gt_sorted = sorted(gt_entities, key=lambda e: e.start)
            pred_sorted = sorted(pred_entities, key=lambda e: e.start)

            gt_tags = [f"B-{e.type}" for e in gt_sorted]
            pred_tags = [f"B-{e.type}" for e in pred_sorted]

            y_true.append(gt_tags)
            y_pred.append(pred_tags)

        return y_true, y_pred
