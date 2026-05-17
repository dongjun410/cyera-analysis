import time
import torch
from typing import List, Dict, Any
from cyera_bench.types import Entity, BenchmarkResult
from cyera_bench.models.base import BaseModel
from cyera_bench.models.flan_t5 import FlanT5Model
from cyera_bench.datasets.base import BaseDataset
from cyera_bench.datasets.conll03 import Conll03Dataset
from cyera_bench.datasets.pii_masking import PiiMaskingDataset
from cyera_bench.datasets.synthetic_pii import SyntheticPiiDataset
from cyera_bench.metrics.calculator import MetricsCalculator
from cyera_bench.reporter import Reporter

_MODEL_REGISTRY = {"flan-t5": FlanT5Model}
_DATASET_REGISTRY = {
    "conll03": Conll03Dataset,
    "pii-masking": PiiMaskingDataset,
    "synthetic-pii": SyntheticPiiDataset,
}

class BenchmarkOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_calc = MetricsCalculator()

        exp = config["experiment"]
        self.experiment_name = exp["name"]

        model_cfg = config["model"]
        model_cls = _MODEL_REGISTRY[model_cfg["type"]]
        self.model: BaseModel = model_cls(
            variant=model_cfg.get("variant", "large"),
            device=model_cfg.get("device", "cuda"),
            quantization=model_cfg.get("quantization"),
        )

        dataset_cfg = config["dataset"]
        dataset_cls = _DATASET_REGISTRY[dataset_cfg["type"]]
        kwargs = dataset_cfg.get("kwargs", {})
        self.dataset: BaseDataset = dataset_cls(**kwargs)
        self.dataset_split = dataset_cfg.get("split", "test")

        output_cfg = config.get("output", {})
        self.reporter = Reporter(
            output_formats=output_cfg.get("formats", ["terminal", "markdown", "json"]),
            output_path=output_cfg.get("path", "./results/"),
        )

        self.metric_names = config.get("metrics", [])
        self.batch_sizes = model_cfg.get("batch_sizes", [1, 4, 8, 16, 32])

    def run(self) -> BenchmarkResult:
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
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9

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

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            total_tokens += sum(len(t.split()) for t in batch)

            t_start = time.perf_counter()
            self.model.predict(batch)
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) * 1000)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t0

        return latencies, total_tokens, total_time
