from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Entity:
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class BenchmarkResult:
    experiment_name: str
    model_name: str
    model_variant: str
    dataset_name: str
    per_entity_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    gpu_memory_peak_gb: float = 0.0
    param_count: int = 0
    total_samples: int = 0
    total_time_sec: float = 0.0
