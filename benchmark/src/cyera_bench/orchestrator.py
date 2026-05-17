import time
import numpy as np
from typing import List, Dict, Any, Tuple
from cyera_bench.types import Entity, BenchmarkResult, ClassificationBenchmarkResult
from cyera_bench.reporter import Reporter


def _import_classification_metrics():
    from cyera_bench.metrics.classification import compute_classification_metrics
    return compute_classification_metrics


def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _get_model_class(name: str):
    """Lazy import model class to avoid loading heavy deps when not needed."""
    if name == "flan-t5":
        from cyera_bench.models.flan_t5 import FlanT5Model
        return FlanT5Model
    elif name == "flan-t5-classification":
        from cyera_bench.models.flan_t5_classification import FlanT5ClassificationModel
        return FlanT5ClassificationModel
    elif name == "doc-classifier-sklearn":
        from cyera_bench.models.doc_classifier_sklearn import DocClassifierSklearnModel
        return DocClassifierSklearnModel
    elif name == "gemma-doc-label":
        from cyera_bench.models.gemma_doc_label import GemmaDocLabelModel
        return GemmaDocLabelModel
    raise ValueError(f"Unknown model type: {name}")


def _get_dataset_class(name: str):
    """Lazy import dataset class."""
    if name == "conll03":
        from cyera_bench.datasets.conll03 import Conll03Dataset
        return Conll03Dataset
    elif name == "pii-masking":
        from cyera_bench.datasets.pii_masking import PiiMaskingDataset
        return PiiMaskingDataset
    elif name == "synthetic-pii":
        from cyera_bench.datasets.synthetic_pii import SyntheticPiiDataset
        return SyntheticPiiDataset
    elif name == "dspm27":
        from cyera_bench.datasets.dspm27 import Dspm27Dataset
        return Dspm27Dataset
    elif name == "ben25":
        from cyera_bench.datasets.ben25 import Ben25Dataset
        return Ben25Dataset
    elif name == "cxh5types":
        from cyera_bench.datasets.cxh5types import Cxh5typesDataset
        return Cxh5typesDataset
    elif name == "20newsgroups":
        from cyera_bench.datasets.twenty_newsgroups import TwentyNewsgroupsDataset
        return TwentyNewsgroupsDataset
    elif name == "ledgar":
        from cyera_bench.datasets.ledgar import LedgarDataset
        return LedgarDataset
    elif name == "german-multifin":
        from cyera_bench.datasets.german_multifin import GermanMultiFinDataset
        return GermanMultiFinDataset
    raise ValueError(f"Unknown dataset type: {name}")

class BenchmarkOrchestrator:
    @property
    def metrics_calc(self):
        if self._metrics_calc is None:
            from cyera_bench.metrics.calculator import MetricsCalculator
            self._metrics_calc = MetricsCalculator()
        return self._metrics_calc

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._metrics_calc = None

        exp = config["experiment"]
        self.experiment_name = exp["name"]

        self.task_type = config.get("task_type", "ner")

        model_cfg = config["model"]
        model_cls = _get_model_class(model_cfg["type"])

        self.model: Any = model_cls(
            variant=model_cfg.get("variant", "large"),
            device=model_cfg.get("device", "cuda"),
            quantization=model_cfg.get("quantization"),
        )

        dataset_cfg = config["dataset"]
        dataset_cls = _get_dataset_class(dataset_cfg["type"])
        if self.task_type == "classification":
            data_root = dataset_cfg.get("data_root", "")
            self.dataset: Any = dataset_cls(data_root=data_root)
        else:
            kwargs = dataset_cfg.get("kwargs", {})
            self.dataset: Any = dataset_cls(**kwargs)
            self.dataset_split = dataset_cfg.get("split", "test")

        output_cfg = config.get("output", {})
        self.reporter = Reporter(
            output_formats=output_cfg.get("formats", ["terminal", "markdown", "json"]),
            output_path=output_cfg.get("path", "./results/"),
        )

        self.metric_names = config.get("metrics", [])
        self.batch_sizes = model_cfg.get("batch_sizes", [1, 4, 8, 16, 32])

    def run(self):
        if self.task_type == "classification":
            return self._run_classification()
        return self._run_ner()

    def _run_ner(self) -> BenchmarkResult:
        print(f"\nLoading dataset: {self.dataset.__class__.__name__} ({self.dataset_split} split)...")
        texts_raw = self.dataset.texts(self.dataset_split)
        texts = [" ".join(t) if isinstance(t, list) else t for t in texts_raw]

        try:
            ground_truth = self._load_ground_truth(texts_raw)
        except (KeyError, AttributeError):
            print("  [WARN] No ground truth labels found. Skipping accuracy metrics.")
            ground_truth = None

        print(f"Warming up model: {self.model.name}...")
        self.model.warmup(n=10)

        # Phase 1: Accuracy
        if ground_truth is not None and "ner_f1" in self.metric_names:
            print("Phase 1/2: Accuracy evaluation...")
            predictions = self._run_inference(texts, batch_size=self.batch_sizes[0])
            per_entity, macro_f1 = self.metrics_calc.compute_ner_metrics(predictions, ground_truth)
        else:
            predictions = []
            per_entity, macro_f1 = {}, 0.0

        # Phase 2: Throughput sweep
        print("Phase 2/2: Throughput sweep...")
        best_throughput = 0.0
        all_latencies: List[float] = []

        eval_texts = texts[:min(len(texts), 1000)]

        for bs in self.batch_sizes:
            latencies, total_tokens, total_time = self._benchmark_throughput(eval_texts, batch_size=bs)
            throughput = total_tokens / total_time if total_time > 0 else 0
            all_latencies.extend(latencies)

            if throughput > best_throughput:
                best_throughput = throughput
                print(f"  batch_size={bs:>2}: {throughput:>8.1f} tokens/sec")

        p50, p95, p99 = self.metrics_calc.compute_latency_percentiles(all_latencies)

        gpu_mem = 0.0
        gpu_mem = 0.0
        if (t := _get_torch()) and t.cuda.is_available():
            gpu_mem = t.cuda.max_memory_allocated() / 1e9

        result = BenchmarkResult(
            experiment_name=self.experiment_name,
            model_name=self.model.name,
            model_variant=self.config["model"].get("variant", "large"),
            dataset_name=self.config["dataset"]["type"],
            param_count=self.model.param_count,
            per_entity_metrics=per_entity,
            macro_f1=macro_f1,
            throughput_tokens_per_sec=best_throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            gpu_memory_peak_gb=gpu_mem,
            total_samples=len(texts),
            total_time_sec=sum(all_latencies) / 1000.0,
        )

        self.reporter.report(result)
        return result

    def _load_ground_truth(self, texts_raw: list) -> List[List[Entity]]:
        tag_seqs = self.dataset.bio_tags(self.dataset_split)
        entities: List[List[Entity]] = []

        for tokens, tag_seq in zip(texts_raw, tag_seqs):
            tokens_l = list(tokens) if isinstance(tokens, (list, tuple)) else tokens.split()
            sample_entities: List[Entity] = []
            current_entity: str | None = None
            current_start = 0
            current_tokens: List[str] = []

            for i, tag in enumerate(tag_seq):
                tag_str = tag if isinstance(tag, str) else f"TAG-{tag}"
                if tag_str.startswith("B-"):
                    if current_entity:
                        sample_entities.append(Entity(
                            type=current_entity,
                            text=" ".join(current_tokens),
                            start=current_start,
                            end=current_start + len(" ".join(current_tokens)),
                        ))
                    current_entity = tag_str[2:]
                    current_start = i
                    current_tokens = [tokens_l[i] if i < len(tokens_l) else ""]
                elif tag_str.startswith("I-") and current_entity:
                    current_tokens.append(tokens_l[i] if i < len(tokens_l) else "")
                else:
                    if current_entity:
                        sample_entities.append(Entity(
                            type=current_entity,
                            text=" ".join(current_tokens),
                            start=current_start,
                            end=current_start + len(" ".join(current_tokens)),
                        ))
                        current_entity = None
                        current_tokens = []

            if current_entity:
                sample_entities.append(Entity(
                    type=current_entity,
                    text=" ".join(current_tokens),
                    start=current_start,
                    end=current_start + len(" ".join(current_tokens)),
                ))

            entities.append(sample_entities)

        return entities

    def _run_inference(self, texts: List[str], batch_size: int) -> List[List[Entity]]:
        all_results: List[List[Entity]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_results.extend(self.model.predict(batch))
        return all_results

    def _benchmark_throughput(self, texts: List[str], batch_size: int):
        latencies: List[float] = []
        total_tokens = 0

        ((t := _get_torch()) and t.cuda.is_available() and t.cuda.synchronize())
        t0 = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            total_tokens += sum(len(t.split()) for t in batch)

            t_start = time.perf_counter()
            self.model.predict(batch)
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) * 1000)

        ((t := _get_torch()) and t.cuda.is_available() and t.cuda.synchronize())
        total_time = time.perf_counter() - t0

        return latencies, total_tokens, total_time

    def _run_classification(self) -> ClassificationBenchmarkResult:
        print(f"\nLoading dataset: {self.dataset.__class__.__name__}...")
        texts, labels = self.dataset.load()
        y_true_l1 = [l["l1"] for l in labels]
        y_true_l2 = [l["l2"] for l in labels]
        n_samples = len(texts)

        # Build L2 options map from ground truth
        l1_set = sorted(set(y_true_l1))
        l2_options: Dict[str, List[str]] = {}
        for l1_name in l1_set:
            l2_set = sorted(set(
                l2 for l1, l2 in zip(y_true_l1, y_true_l2) if l1 == l1_name
            ))
            l2_options[l1_name] = l2_set

        print(f"Warming up model: {self.model.name}...")
        self.model.warmup(n=3)

        # Phase 1: Accuracy
        y_pred_l1: List[str] = []
        y_pred_l2: List[str] = []
        if "classification_accuracy" in self.metric_names:
            print(f"Phase 1/2: Accuracy evaluation ({n_samples} samples)...")
            batch_size = self.batch_sizes[0]
            for i in range(0, n_samples, batch_size):
                batch_texts = texts[i:i + batch_size]
                preds = self.model.predict_labels(batch_texts, l1_set, l2_options)
                y_pred_l1.extend(p["l1"] for p in preds)
                y_pred_l2.extend(p["l2"] for p in preds)

            compute_classification_metrics = _import_classification_metrics()
            metrics = compute_classification_metrics(
                y_true_l1, y_pred_l1, y_true_l2, y_pred_l2,
            )
        else:
            metrics = {
                "l1_accuracy": 0.0, "l2_accuracy": 0.0,
                "l2_accuracy_given_correct_l1": 0.0,
                "macro_l1_f1": 0.0, "macro_l2_f1": 0.0,
                "per_l1_metrics": {}, "per_l2_metrics": {},
            }

        # Phase 2: Throughput sweep
        print("Phase 2/2: Throughput sweep...")
        best_throughput = 0.0
        all_latencies: List[float] = []
        eval_texts = texts[:min(n_samples, 1000)]

        for bs in self.batch_sizes:
            latencies, total_chars, total_time = self._benchmark_classification(eval_texts, batch_size=bs)
            throughput = total_chars / total_time if total_time > 0 else 0
            all_latencies.extend(latencies)

            if throughput > best_throughput:
                best_throughput = throughput
                print(f"  batch_size={bs:>2}: {throughput:>8.1f} chars/sec")

        p50, p95, p99 = self.metrics_calc.compute_latency_percentiles(all_latencies)

        gpu_mem = 0.0
        gpu_mem = 0.0
        if (t := _get_torch()) and t.cuda.is_available():
            gpu_mem = t.cuda.max_memory_allocated() / 1e9

        result = ClassificationBenchmarkResult(
            experiment_name=self.experiment_name,
            model_name=self.model.name,
            model_variant=self.config["model"].get("variant", "large"),
            dataset_name=self.config["dataset"]["type"],
            param_count=self.model.param_count,
            l1_accuracy=metrics["l1_accuracy"],
            l2_accuracy=metrics["l2_accuracy"],
            l2_accuracy_given_correct_l1=metrics["l2_accuracy_given_correct_l1"],
            macro_l1_f1=metrics["macro_l1_f1"],
            macro_l2_f1=metrics["macro_l2_f1"],
            per_l1_metrics=metrics["per_l1_metrics"],
            per_l2_metrics=metrics["per_l2_metrics"],
            throughput_chars_per_sec=best_throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            gpu_memory_peak_gb=gpu_mem,
            total_samples=n_samples,
            total_time_sec=sum(all_latencies) / 1000.0,
        )

        self.reporter.report(result)
        return result

    def _benchmark_classification(self, texts: List[str], batch_size: int):
        latencies: List[float] = []
        total_chars = 0

        ((t := _get_torch()) and t.cuda.is_available() and t.cuda.synchronize())
        t0 = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            total_chars += sum(len(t) for t in batch)

            t_start = time.perf_counter()
            # Run dummy predict for throughput measurement
            _ = self.model.predict(batch)
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) * 1000)

        ((t := _get_torch()) and t.cuda.is_available() and t.cuda.synchronize())
        total_time = time.perf_counter() - t0

        return latencies, total_chars, total_time
