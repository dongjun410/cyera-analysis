import pytest
from cyera_bench.types import Entity, BenchmarkResult


def test_entity_creation():
    e = Entity(type="PER", text="John", start=0, end=4, confidence=0.95)
    assert e.type == "PER"
    assert e.text == "John"
    assert e.start == 0
    assert e.end == 4
    assert e.confidence == 0.95


def test_entity_default_confidence():
    e = Entity(type="ORG", text="Acme", start=10, end=14)
    assert e.confidence == 1.0


def test_benchmark_result_creation():
    br = BenchmarkResult(
        experiment_name="test-exp",
        model_name="flan-t5",
        model_variant="large",
        dataset_name="conll03",
        per_entity_metrics={"PER": {"precision": 0.95, "recall": 0.93, "f1": 0.94}},
        macro_f1=0.94,
        throughput_tokens_per_sec=342.7,
        latency_p50_ms=23.0,
        latency_p95_ms=45.0,
        latency_p99_ms=78.0,
        gpu_memory_peak_gb=4.2,
        total_samples=3453,
        total_time_sec=12.5,
    )
    assert br.macro_f1 == 0.94
    assert br.per_entity_metrics["PER"]["f1"] == 0.94
