import json
import os
from datetime import datetime
from typing import List
from cyera_bench.types import BenchmarkResult

class Reporter:
    def __init__(self, output_formats: List[str] | None = None, output_path: str = "./results/"):
        self.formats = output_formats or ["terminal", "markdown", "json"]
        self.output_path = output_path

    def report(self, result: BenchmarkResult) -> None:
        for fmt in self.formats:
            if fmt == "terminal":
                self._report_terminal(result)
            elif fmt == "markdown":
                self._report_markdown(result)
            elif fmt == "json":
                self._report_json(result)

    def _report_terminal(self, r: BenchmarkResult) -> None:
        print()
        print("=" * 58)
        print(f"  Benchmark: {r.experiment_name}")
        print(f"  Model: {r.model_name} ({r.model_variant}, {r.param_count/1e6:.0f}M params)")
        print(f"  Dataset: {r.dataset_name} ({r.total_samples} samples)")
        print(f"  Device: {'CUDA' if r.gpu_memory_peak_gb > 0 else 'CPU'}")
        print("=" * 58)

        if r.per_entity_metrics:
            print(f"  {'Entity':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
            for etype, m in sorted(r.per_entity_metrics.items()):
                print(f"  {etype:<16} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
            print(f"  {'-'*46}")
            print(f"  {'Macro F1':<16} {r.macro_f1:>30.4f}")

        print()
        print(f"  Throughput:     {r.throughput_tokens_per_sec:>8.1f} tokens/sec")
        print(f"  Latency P50:    {r.latency_p50_ms:>8.1f} ms")
        print(f"  Latency P95:    {r.latency_p95_ms:>8.1f} ms")
        print(f"  Latency P99:    {r.latency_p99_ms:>8.1f} ms")
        if r.gpu_memory_peak_gb > 0:
            print(f"  GPU Memory Peak:{r.gpu_memory_peak_gb:>8.1f} GB")
        print(f"  Total Time:     {r.total_time_sec:>8.1f} sec")
        print("=" * 58)
        print()

    def _report_markdown(self, r: BenchmarkResult) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{r.experiment_name}_{date_str}.md"
        filepath = os.path.join(self.output_path, filename)

        lines = [
            f"# Benchmark: {r.experiment_name}",
            "",
            f"- **Model:** {r.model_name} ({r.param_count/1e6:.0f}M params, {r.model_variant})",
            f"- **Dataset:** {r.dataset_name} ({r.total_samples} samples)",
            f"- **Date:** {date_str}",
            "",
        ]

        if r.per_entity_metrics:
            lines.append("## Entity-Level Metrics")
            lines.append("")
            lines.append("| Entity | Precision | Recall | F1 |")
            lines.append("|--------|-----------|--------|-----|")
            for etype, m in sorted(r.per_entity_metrics.items()):
                lines.append(f"| {etype} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
            lines.append(f"| **Macro Avg** | - | - | **{r.macro_f1:.4f}** |")
            lines.append("")

        lines.extend([
            "## Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Throughput | {r.throughput_tokens_per_sec:.1f} tokens/sec |",
            f"| Latency P50 | {r.latency_p50_ms:.1f} ms |",
            f"| Latency P95 | {r.latency_p95_ms:.1f} ms |",
            f"| Latency P99 | {r.latency_p99_ms:.1f} ms |",
            f"| GPU Memory Peak | {r.gpu_memory_peak_gb:.1f} GB |",
            f"| Total Time | {r.total_time_sec:.1f} sec |",
            "",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  [Markdown report saved to {filepath}]")

    def _report_json(self, r: BenchmarkResult) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{r.experiment_name}_{date_str}.json"
        filepath = os.path.join(self.output_path, filename)

        data = {
            "experiment_name": r.experiment_name,
            "model_name": r.model_name,
            "model_variant": r.model_variant,
            "model_param_count": r.param_count,
            "dataset_name": r.dataset_name,
            "per_entity_metrics": r.per_entity_metrics,
            "macro_f1": r.macro_f1,
            "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_p99_ms": r.latency_p99_ms,
            "gpu_memory_peak_gb": r.gpu_memory_peak_gb,
            "total_samples": r.total_samples,
            "total_time_sec": r.total_time_sec,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"  [JSON report saved to {filepath}]")

    def compare(self, result_paths: List[str]) -> None:
        """Load multiple JSON results and print a comparison table."""
        results: List[BenchmarkResult] = []
        for path in result_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(BenchmarkResult(**data))

        if not results:
            print("No results to compare.")
            return

        print()
        print("=" * 80)
        print("  Cross-Experiment Comparison")
        print("=" * 80)
        print(f"  {'Experiment':<30} {'Macro F1':>10} {'Throughput':>12} {'P50 Lat':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10}")
        for r in results:
            print(f"  {r.experiment_name:<30} {r.macro_f1:>10.4f} {r.throughput_tokens_per_sec:>10.1f} t/s {r.latency_p50_ms:>8.1f} ms")
        print("=" * 80)
        print()
