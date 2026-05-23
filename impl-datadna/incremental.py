#!/usr/bin/env python
"""DataDNA Fusion — Incremental document classification entry point.

Single-document or batch incremental classification using the 6-engine
fusion voter. For new documents arriving after initial deployment.

Usage:
    python incremental.py --input ./new_docs/ --output ./inc_output/ --config config.yaml
    python incremental.py --input single_doc.txt --output ./inc_output/
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

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx", ".json"}


# ═══════════════════════════════════════════════════════════════
# Document loading
# ═══════════════════════════════════════════════════════════════

def load_single_document(file_path: str) -> Document | None:
    path = Path(file_path)
    if not path.is_file():
        logger.error("File not found: %s", file_path)
        return None
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.error("Unsupported type: %s (supported: %s)", suffix, SUPPORTED_SUFFIXES)
        return None
    doc_id = path.stem
    metadata: dict[str, Any] = {
        "file_path": str(path),
        "file_type": suffix,
        "file_size": path.stat().st_size,
        "path_depth": 0,
    }
    text = _read_file(path, suffix, metadata)
    if not text or not text.strip():
        logger.warning("Empty document, skipping: %s", doc_id)
        return None
    return Document(doc_id=doc_id, text=text, metadata=metadata)


def load_documents_from_dir(input_dir: str) -> list[Document]:
    documents: list[Document] = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error("Directory not found: %s", input_dir)
        return documents
    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        doc_id = file_path.stem
        rel_path = file_path.relative_to(input_path)
        metadata: dict[str, Any] = {
            "file_path": str(file_path),
            "file_type": suffix,
            "file_size": file_path.stat().st_size,
            "path_depth": max(len(rel_path.parts) - 1, 0),
        }
        try:
            text = _read_file(file_path, suffix, metadata)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            continue
        if not text or not text.strip():
            continue
        documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))
    logger.info("Loaded %d documents from %s", len(documents), input_dir)
    return documents


def _read_file(file_path: Path, suffix: str, metadata: dict[str, Any]) -> str:
    if suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(file_path))
            try:
                pages = [page.get_text() for page in doc]
                metadata["page_count"] = len(pages)
                return "\n".join(pages)
            finally:
                doc.close()
        except ImportError:
            return file_path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".docx":
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs]
            metadata["paragraph_count"] = len(paragraphs)
            return "\n".join(paragraphs)
        except ImportError:
            return file_path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, str):
                return data
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return file_path.read_text(encoding="utf-8", errors="replace")
    else:
        return file_path.read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════
# Component initialization
# ═══════════════════════════════════════════════════════════════

def _init_components(config: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    type_lib = get_type_library()

    components["e1"] = E1RegexEngine()
    components["e2"] = E2TemplateEngine()
    components["e3"] = E3MLEngine()

    try:
        emb = config.get("embedding", {})
        embedder = BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cuda"),
            batch_size=emb.get("batch_size", 32),
            max_length=emb.get("max_token_length", 8192),
        )
        components["embedder"] = embedder
    except Exception as exc:
        logger.warning("BGE-M3 unavailable: %s", exc)
        components["embedder"] = None

    knn_cfg = config.get("knn", {})
    components["e4"] = E4kNNEngine(
        embedder=components["embedder"],
        type_library=type_lib,
        min_types=knn_cfg.get("min_types_for_activation", 5),
    )
    components["e5"] = E5StructuralEngine(type_library=type_lib)

    try:
        llm_cfg = config.get("llm", {})
        llm_client = MistralClient(LLMConfig(
            api_base=llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=llm_cfg.get("model", "mistral:7b"),
            quantization=llm_cfg.get("quantization", "4bit"),
            temperature=llm_cfg.get("temperature", 0.3),
        ))
        components["e6"] = E6LLMEngine(llm_client=llm_client, type_library=type_lib)
    except Exception as exc:
        logger.warning("E6 LLM unavailable: %s", exc)
        components["e6"] = E6LLMEngine(llm_client=None, type_library=type_lib)

    all_engines = [
        components["e1"], components["e2"], components["e3"],
        components["e4"], components["e5"], components["e6"],
    ]
    components["voter"] = FusionVoter(engines=all_engines)
    return components


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="DataDNA Fusion Incremental Classifier")
    parser.add_argument("--input", required=True, help="File or directory path")
    parser.add_argument("--output", default="./output_inc/", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    output_dir = Path(args.output)

    with open(args.config, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    logger.info("DataDNA Fusion Incremental: input=%s, output=%s", args.input, args.output)

    stats: dict[str, Any] = {}
    overall_start = time.perf_counter()

    init_start = time.perf_counter()
    comp = _init_components(config)
    stats["init_time_s"] = round(time.perf_counter() - init_start, 3)

    input_path = Path(args.input)
    doc_start = time.perf_counter()
    if input_path.is_file():
        doc = load_single_document(args.input)
        documents = [doc] if doc is not None else []
    elif input_path.is_dir():
        documents = load_documents_from_dir(args.input)
    else:
        logger.error("Input path not found: %s", args.input)
        return 1

    stats["doc_load_time_s"] = round(time.perf_counter() - doc_start, 3)
    stats["doc_count"] = len(documents)

    if not documents:
        logger.warning("No documents found")
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump({"results": [], "stats": stats}, open(output_dir / "results.json", "w"))
        return 0

    voter: FusionVoter = comp["voter"]
    audit = AuditLogger(output_dir / "audit.jsonl")
    metrics = MetricsCollector()

    classify_start = time.perf_counter()
    results = []
    method_counts: dict[str, int] = {}

    for doc in documents:
        result = voter.classify(doc)
        audit.log(result)
        metrics.record(result)
        results.append(result)
        method_counts[result.method] = method_counts.get(result.method, 0) + 1

    classify_time = round(time.perf_counter() - classify_start, 3)
    stats["classify_time_s"] = classify_time
    stats["method_counts"] = method_counts
    stats["avg_time_per_doc_ms"] = round(
        (classify_time / len(results)) * 1000, 1
    ) if results else 0

    snap = metrics.snapshot()
    if snap.alerts:
        for alert in snap.alerts:
            logger.warning("ALERT: %s", alert)

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

    logger.info("Complete: %d docs in %.3fs | fast=%d full=%d | avg=%.1fms",
                len(results), total_time,
                method_counts.get("fusion_fast", 0),
                method_counts.get("fusion_full", 0),
                stats["avg_time_per_doc_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
