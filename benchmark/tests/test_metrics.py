import pytest
from cyera_bench.types import Entity
from cyera_bench.metrics.calculator import MetricsCalculator


def make_entity(text, etype="PER", start=0, end=0):
    return Entity(type=etype, text=text, start=start, end=end)


def test_ner_f1_perfect():
    predictions = [[make_entity("John", "PER")]]
    ground_truth = [[make_entity("John", "PER")]]
    calc = MetricsCalculator()
    per_entity, macro_f1 = calc.compute_ner_metrics(predictions, ground_truth)
    assert per_entity["PER"]["f1"] == 1.0
    assert macro_f1 == 1.0


def test_ner_f1_empty():
    predictions = [[]]
    ground_truth = [[]]
    calc = MetricsCalculator()
    per_entity, macro_f1 = calc.compute_ner_metrics(predictions, ground_truth)
    assert macro_f1 == 0.0


def test_throughput():
    calc = MetricsCalculator()
    throughput = calc.compute_throughput(total_tokens=1000, total_time_sec=5.0)
    assert throughput == 200.0


def test_latency_percentiles():
    calc = MetricsCalculator()
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50, p95, p99 = calc.compute_latency_percentiles(latencies)
    assert p50 == 30.0
    assert p95 >= 40.0
