#!/usr/bin/env python
"""Latency benchmark for 6-engine fusion pipeline.

Measures fusion_fast vs fusion_full timing to verify the R4 requirement:
  - Weighted average latency < 300ms (target)
  - fusion_fast: E1-E5 only, ~5ms
  - fusion_full: E1-E5 + E6 LLM, ~1.4s

Usage:
    python benchmark.py --config config.yaml --num-docs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from src.embeddings.bge_m3 import BgeM3Embedder
from src.engines.e1_regex import E1RegexEngine
from src.engines.e2_template import E2TemplateEngine
from src.engines.e3_ml import E3MLEngine
from src.engines.e4_knn import E4kNNEngine
from src.engines.e5_structural import E5StructuralEngine
from src.engines.e6_llm import E6LLMEngine
from src.fusion.voter import FusionVoter
from src.knowledge.type_library import get_type_library
from src.llm.client import LLMConfig, MistralClient
from src.types import Document

logging.basicConfig(
    level=logging.WARNING,  # quiet during benchmark
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Sample documents for benchmarking ──

SAMPLE_DOCS: list[tuple[str, str, dict]] = [
    # HR docs (PII-rich -> E1+E2 strong signal)
    ("hr_01", "Employee SSN: 123-45-6789, Name: John Smith, Start date: 2020-03-15, "
     "Department: Engineering, Salary: $95,000, Payroll ID: PR-001, "
     "Benefits: Full medical, dental, vision. 401k contribution: 6% match.",
     {"file_type": ".docx"}),
    ("hr_02", "Employee SSN: 456-78-9012, Name: Sarah Chen, Start date: 2019-07-01, "
     "Department: Marketing, Salary: $110,000. Performance review: Exceeds expectations. "
     "Promotion effective: 2024-01-01. New title: Senior Marketing Manager.",
     {"file_type": ".docx"}),
    ("hr_03", "W-2 Form 2024: Employer EIN: 12-3456789, Employee SSN: 789-01-2345, "
     "Wages: $85,000, Federal tax withheld: $12,750. State tax: $4,250.",
     {"file_type": ".pdf"}),

    # Financial docs (PII-rich -> E1+E2+E4)
    ("fin_01", "Quarterly Revenue Report Q1 2024: Total Revenue: $1,200,000. "
     "Net Income: $340,000. Credit card payments processed: Visa 4532015112830366 "
     "for $45,230. Wire transfer: IBAN CH9300762011623852957 for $250,000.",
     {"file_type": ".pdf"}),
    ("fin_02", "Invoice #INV-2024-089: Bill To: Acme Corp, Amount: $45,230. "
     "Payment: Mastercard 5500000000000004. Due: Net 30. "
     "Line items: Consulting $30,000, Software licenses $15,230.",
     {"file_type": ".pdf"}),
    ("fin_03", "Tax Return 2024: Form 1040, SSN: 321-65-9870, AGI: $145,000, "
     "Schedule A deductions: $28,500, Schedule C business income: $35,000. "
     "Tax due: $18,200. Filing status: Married filing jointly.",
     {"file_type": ".pdf"}),

    # Medical docs
    ("med_01", "Patient ID: MRN: 88421, DOB: 1978-05-15, Diagnosis: Hypertension Stage 2, "
     "NPI: 1234567890, Prescribed: Lisinopril 10mg daily. Blood pressure: 145/92. "
     "Follow-up: 3 months. Physician: Dr. Sarah Wilson.",
     {"file_type": ".pdf"}),
    ("med_02", "Lab Results: Patient MRN: 55102, CBC: WBC 7.2, RBC 4.8, HGB 14.5, "
     "Glucose: 142 mg/dL (HIGH), A1C: 7.2%. NPI: 9876543210. "
     "Diagnosis: Type 2 Diabetes. Medication: Metformin 500mg BID.",
     {"file_type": ".pdf"}),

    # Generic docs (no PII -> likely need LLM)
    ("gen_01", "The weather forecast for today indicates partly cloudy skies with "
     "a high of 72 degrees Fahrenheit. Winds will be light and variable at 5-10 mph. "
     "There is a 20% chance of afternoon showers.",
     {"file_type": ".txt"}),
    ("gen_02", "Project roadmap 2024: Phase 1 infrastructure upgrade (Q1-Q2), "
     "Phase 2 feature rollout (Q3), Phase 3 performance optimization (Q4). "
     "Key milestones include database migration and API gateway deployment.",
     {"file_type": ".txt"}),

    # API logs
    ("api_01", '{"timestamp": "2024-01-15T08:23:45Z", "endpoint": "/api/users", '
     '"status": 200, "response_time_ms": 45, "method": "GET", '
     '"user_agent": "Mozilla/5.0", "ip": "192.168.1.100"}',
     {"file_type": ".json"}),
    ("api_02", '{"timestamp": "2024-01-15T09:10:12Z", "endpoint": "/api/orders", '
     '"status": 201, "response_time_ms": 120, "method": "POST", '
     '"payload_size_bytes": 2048, "api_key": "sk-proj-abc123xyz"}',
     {"file_type": ".json"}),
]


def init_components(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize all 6 engines and the fusion voter.

    Mirrors main.py _init_components() to use real Ollama LLM. Engines that
    fail to initialize (BGE-M3 not found, Ollama down) degrade gracefully and
    the benchmark records their unavailability.
    """
    components: dict[str, Any] = {}
    type_lib = get_type_library()

    components["e1"] = E1RegexEngine()
    components["e2"] = E2TemplateEngine()
    components["e3"] = E3MLEngine()

    # Embedding model (may fail if BAAI/bge-m3 not downloaded)
    try:
        emb = config.get("embedding", {})
        embedder = BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cuda"),
            batch_size=emb.get("batch_size", 32),
            max_length=emb.get("max_token_length", 8192),
        )
        components["embedder"] = embedder
    except Exception:
        components["embedder"] = None

    knn_cfg = config.get("knn", {})
    components["e4"] = E4kNNEngine(
        embedder=components["embedder"],
        type_library=type_lib,
        min_types=knn_cfg.get("min_types_for_activation", 5),
    )
    components["e5"] = E5StructuralEngine(type_library=type_lib)

    # LLM client (may fail if Ollama is not running)
    try:
        llm_cfg = config.get("llm", {})
        llm_client = MistralClient(LLMConfig(
            api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "mistral:7b"),
            quantization=llm_cfg.get("quantization", "4bit"),
            temperature=llm_cfg.get("temperature", 0.3),
        ))
        components["e6"] = E6LLMEngine(llm_client=llm_client, type_library=type_lib)
    except Exception:
        components["e6"] = E6LLMEngine(llm_client=None, type_library=type_lib)

    all_engines = [
        components["e1"], components["e2"], components["e3"],
        components["e4"], components["e5"], components["e6"],
    ]
    components["voter"] = FusionVoter(engines=all_engines)
    return components


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (linear interpolation)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fusion latency benchmark")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--num-docs", type=int, default=50,
                        help="Number of documents to classify (default: 50)")
    parser.add_argument("--output", default="./bench_output/", help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    print("=" * 60)
    print("6-Engine Fusion Latency Benchmark")
    print("=" * 60)

    # ── Init ──
    print("\n[1] Initializing components...")
    init_start = time.perf_counter()
    comp = init_components(config)
    init_time = time.perf_counter() - init_start
    print(f"    Init time: {init_time:.3f}s")
    print(f"    E1 Regex:     available")
    print(f"    E2 Template:  available")
    print(f"    E3 ML:        {'available' if comp['e3'].is_available else 'unavailable (not trained)'}")
    print(f"    E4 kNN:       {'available' if comp['e4'].is_available else 'unavailable (no embedder/centroids)'}")
    print(f"    E5 Structural: available")
    print(f"    E6 LLM:       {'available' if comp['e6'].is_available else 'unavailable (Ollama not running)'}")

    # ── Build document list ──
    voter: FusionVoter = comp["voter"]
    documents: list[Document] = []
    repeats = max(1, args.num_docs // len(SAMPLE_DOCS) + 1)
    for i in range(repeats):
        for j, (doc_id_prefix, text, metadata) in enumerate(SAMPLE_DOCS):
            if len(documents) >= args.num_docs:
                break
            doc_id = f"{doc_id_prefix}_{i}"
            documents.append(Document(
                doc_id=doc_id,
                text=text,
                metadata=dict(metadata),
            ))
        if len(documents) >= args.num_docs:
            break
    documents = documents[:args.num_docs]

    print(f"\n[2] Benchmarking {len(documents)} documents...")

    # ── Warmup: 2 docs to prime caches ──
    for doc in documents[:min(2, len(documents))]:
        voter.classify(doc)

    # ── Benchmark ──
    fast_times: list[float] = []
    full_times: list[float] = []
    fast_count = 0
    full_count = 0
    total_start = time.perf_counter()

    for doc in documents:
        t0 = time.perf_counter()
        result = voter.classify(doc)
        elapsed = time.perf_counter() - t0

        if result.method == "fusion_fast":
            fast_times.append(elapsed * 1000)  # ms
            fast_count += 1
        else:
            full_times.append(elapsed * 1000)  # ms
            full_count += 1

    total_elapsed = time.perf_counter() - total_start

    # ── Report ──
    print(f"\n[3] Results ({args.num_docs} docs in {total_elapsed:.3f}s)")
    print("-" * 60)
    print(f"  fusion_fast: {fast_count} docs ({fast_count / args.num_docs * 100:.1f}%)")
    print(f"  fusion_full: {full_count} docs ({full_count / args.num_docs * 100:.1f}%)")

    if fast_times:
        print(f"\n  fusion_fast latency (ms):")
        print(f"    P50: {percentile(fast_times, 50):.2f}")
        print(f"    P95: {percentile(fast_times, 95):.2f}")
        print(f"    P99: {percentile(fast_times, 99):.2f}")
        print(f"    Mean: {statistics.mean(fast_times):.2f}")
        print(f"    Min:  {min(fast_times):.2f}")
        print(f"    Max:  {max(fast_times):.2f}")

    if full_times:
        print(f"\n  fusion_full latency (ms):")
        print(f"    P50: {percentile(full_times, 50):.2f}")
        print(f"    P95: {percentile(full_times, 95):.2f}")
        print(f"    P99: {percentile(full_times, 99):.2f}")
        print(f"    Mean: {statistics.mean(full_times):.2f}")
        print(f"    Min:  {min(full_times):.2f}")
        print(f"    Max:  {max(full_times):.2f}")

    # ── Weighted average (per spec R4) ──
    all_times = fast_times + full_times
    if all_times:
        weighted_avg = statistics.mean(all_times)
        print(f"\n  Weighted average latency: {weighted_avg:.2f} ms")
        r4_target = 300.0
        if weighted_avg < r4_target:
            print(f"  PASS: R4 target met (< {r4_target:.0f} ms)")
        else:
            print(f"  FAIL: R4 target exceeded (> {r4_target:.0f} ms)")
            print(f"    Delta: {weighted_avg - r4_target:.2f} ms")

    # ── Engine output stats ──
    e1_matches = 0
    for doc in documents[:args.num_docs]:
        out = comp["e1"].analyze(doc)
        if out.status == "matched":
            e1_matches += 1
    print(f"\n  E1 regex hit rate: {e1_matches}/{args.num_docs} ({e1_matches / args.num_docs * 100:.1f}%)")

    # ── Save report JSON ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "num_docs": args.num_docs,
            "e1_available": True,
            "e2_available": True,
            "e3_available": comp["e3"].is_available,
            "e4_available": comp["e4"].is_available,
            "e5_available": True,
            "e6_available": comp["e6"].is_available,
        },
        "results": {
            "fusion_fast_count": fast_count,
            "fusion_full_count": full_count,
            "llm_call_rate": round(full_count / args.num_docs, 4) if args.num_docs else 0,
            "weighted_avg_ms": round(weighted_avg, 2) if all_times else 0,
            "r4_target_ms": 300.0,
            "r4_pass": weighted_avg < 300.0 if all_times else None,
            "fusion_fast": {
                "p50_ms": round(percentile(fast_times, 50), 2) if fast_times else None,
                "p95_ms": round(percentile(fast_times, 95), 2) if fast_times else None,
                "p99_ms": round(percentile(fast_times, 99), 2) if fast_times else None,
                "mean_ms": round(statistics.mean(fast_times), 2) if fast_times else None,
                "min_ms": round(min(fast_times), 2) if fast_times else None,
                "max_ms": round(max(fast_times), 2) if fast_times else None,
            },
            "fusion_full": {
                "p50_ms": round(percentile(full_times, 50), 2) if full_times else None,
                "p95_ms": round(percentile(full_times, 95), 2) if full_times else None,
                "p99_ms": round(percentile(full_times, 99), 2) if full_times else None,
                "mean_ms": round(statistics.mean(full_times), 2) if full_times else None,
                "min_ms": round(min(full_times), 2) if full_times else None,
                "max_ms": round(max(full_times), 2) if full_times else None,
            },
        },
    }
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
