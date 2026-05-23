#!/usr/bin/env python
"""DataDNA 6-Engine Fusion Classification — Main Entry Point

Parallel engine dispatch → weighted voting fusion → audit output.

Usage:
    python main.py --input ./docs/ --output ./output/ --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
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
from src.monitoring.audit import AuditLogger
from src.monitoring.metrics import MetricsCollector
from src.types import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Document loading
# ═══════════════════════════════════════════════════════════════

def load_documents(input_dir: str) -> list[Document]:
    documents: list[Document] = []
    input_path = Path(input_dir)
    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in (".txt", ".pdf", ".docx", ".json"):
            continue
        doc_id = file_path.stem
        rel_path = file_path.relative_to(input_path)
        metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_type": suffix,
            "file_size": file_path.stat().st_size,
            "path_depth": max(len(rel_path.parts) - 1, 0),
        }
        text = ""
        try:
            if suffix == ".pdf":
                text = _read_pdf(file_path, metadata)
            elif suffix == ".docx":
                text = _read_docx(file_path, metadata)
            elif suffix == ".json":
                text = _read_json(file_path)
            else:
                text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            continue
        if not text.strip():
            continue
        documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))
    logger.info("Loaded %d documents from %s", len(documents), input_dir)
    return documents


def _read_pdf(file_path: Path, metadata: dict[str, Any]) -> str:
    try:
        import fitz
    except ImportError:
        return file_path.read_text(encoding="utf-8", errors="replace")
    doc = fitz.open(str(file_path))
    try:
        pages = [page.get_text() for page in doc]
        metadata["page_count"] = len(pages)
        return "\n".join(pages)
    finally:
        doc.close()


def _read_docx(file_path: Path, metadata: dict[str, Any]) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError:
        return file_path.read_text(encoding="utf-8", errors="replace")
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs]
    metadata["paragraph_count"] = len(paragraphs)
    return "\n".join(paragraphs)


def _read_json(file_path: Path) -> str:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return file_path.read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════
# Component initialization
# ═══════════════════════════════════════════════════════════════

def _init_components(config: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, Any] = {}

    # Type library (always available)
    type_lib = get_type_library()
    components["type_library"] = type_lib
    logger.info("TypeLibrary initialized (%d types)", type_lib.count)

    # E1: Regex engine (always available)
    components["e1"] = E1RegexEngine()
    logger.info("E1 Regex engine initialized")

    # E2: Template engine (always available)
    components["e2"] = E2TemplateEngine()
    logger.info("E2 Template engine initialized")

    # E3: ML engine (available after training)
    components["e3"] = E3MLEngine()
    logger.info("E3 ML engine initialized (model not yet trained)")

    # BGE-M3 embedder for E4
    try:
        emb = config.get("embedding", {})
        embedder = BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cuda"),
            batch_size=emb.get("batch_size", 32),
            max_length=emb.get("max_token_length", 8192),
        )
        components["embedder"] = embedder
        logger.info("BgeM3Embedder initialized (dim=%d)", embedder.dim)
    except Exception as exc:
        logger.warning("BgeM3Embedder unavailable: %s — E4 will be disabled", exc)
        components["embedder"] = None

    # E4: kNN engine
    knn_cfg = config.get("knn", {})
    components["e4"] = E4kNNEngine(
        embedder=components["embedder"],
        type_library=type_lib,
        min_types=knn_cfg.get("min_types_for_activation", 5),
    )
    available = "available" if components["e4"].is_available else "unavailable"
    logger.info("E4 kNN engine initialized (%s)", available)

    # E5: Structural engine (always available)
    components["e5"] = E5StructuralEngine(type_library=type_lib)
    logger.info("E5 Structural engine initialized")

    # E6: LLM engine
    try:
        llm_cfg = config.get("llm", {})
        llm_client = MistralClient(LLMConfig(
            api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "mistral:7b"),
            quantization=llm_cfg.get("quantization", "4bit"),
            temperature=llm_cfg.get("temperature", 0.3),
        ))
        components["e6"] = E6LLMEngine(llm_client=llm_client, type_library=type_lib)
        logger.info("E6 LLM engine initialized")
    except Exception as exc:
        logger.warning("E6 LLM engine unavailable: %s", exc)
        components["e6"] = E6LLMEngine(llm_client=None, type_library=type_lib)

    # Fusion voter
    all_engines = [
        components["e1"], components["e2"], components["e3"],
        components["e4"], components["e5"], components["e6"],
    ]
    components["voter"] = FusionVoter(engines=all_engines)
    logger.info("FusionVoter initialized with %d engines", len(all_engines))

    return components


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="DataDNA 6-Engine Fusion Classifier")
    parser.add_argument("--input", required=True, help="Document directory path")
    parser.add_argument("--output", default="./output/", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    output_dir = Path(args.output)

    with open(args.config, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    logger.info("DataDNA Fusion: input=%s, output=%s", args.input, args.output)

    stats: dict[str, Any] = {}
    overall_start = time.perf_counter()

    # Init
    init_start = time.perf_counter()
    comp = _init_components(config)
    stats["init_time_s"] = round(time.perf_counter() - init_start, 3)

    # Load documents
    doc_start = time.perf_counter()
    documents = load_documents(args.input)
    stats["doc_load_time_s"] = round(time.perf_counter() - doc_start, 3)
    stats["doc_count"] = len(documents)

    if not documents:
        logger.warning("No documents found")
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"results": [], "stats": stats}, open(output_dir / "results.json", "w"))
        return 0

    # Classify
    voter: FusionVoter = comp["voter"]
    audit = AuditLogger(output_dir / "audit.jsonl")
    metrics = MetricsCollector()

    classify_start = time.perf_counter()
    results = []
    method_counts: dict[str, int] = {}
    degraded_count = 0
    manual_review_count = 0

    for doc in documents:
        result = voter.classify(doc)
        audit.log(result)
        metrics.record(result)
        results.append(result)

        method_counts[result.method] = method_counts.get(result.method, 0) + 1
        if result.degraded:
            degraded_count += 1
        if result.manual_review:
            manual_review_count += 1

    classify_time = round(time.perf_counter() - classify_start, 3)
    stats["classify_time_s"] = classify_time
    stats["docs"] = len(results)
    stats["method_counts"] = method_counts
    stats["degraded_count"] = degraded_count
    stats["manual_review_count"] = manual_review_count
    stats["avg_time_per_doc_ms"] = round(
        (classify_time / len(results)) * 1000, 1
    ) if results else 0

    # Metrics snapshot
    snap = metrics.snapshot()
    stats["metrics"] = {
        "p50_confidence": round(snap.p50_confidence, 4),
        "llm_call_rate": round(snap.llm_call_rate, 4),
        "manual_review_rate": round(snap.manual_review_rate, 4),
        "alerts": snap.alerts,
    }

    if snap.alerts:
        for alert in snap.alerts:
            logger.warning("ALERT: %s", alert)

    # Output
    total_time = round(time.perf_counter() - overall_start, 3)
    stats["total_time_s"] = total_time

    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = []
    for r in results:
        results_json.append({
            "doc_id": r.doc_id,
            "final_label": r.final_label,
            "composite_confidence": r.composite_confidence,
            "method": r.method,
            "degraded": r.degraded,
            "manual_review": r.manual_review,
        })

    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results_json, "stats": stats}, fh, ensure_ascii=False, indent=2)

    logger.info("Complete: %d docs in %.3fs | fusion_fast=%d fusion_full=%d | avg=%.1fms",
                len(results), total_time,
                method_counts.get("fusion_fast", 0),
                method_counts.get("fusion_full", 0),
                stats["avg_time_per_doc_ms"])
    logger.info("Audit log: %s (%d records)", output_dir / "audit.jsonl", audit.count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
