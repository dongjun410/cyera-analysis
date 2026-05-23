#!/usr/bin/env python
"""DataDNA real-data evaluation using dspm27 (PDFs) + cxh5types + ben25.

Loads real enterprise documents from the ZerosOne benchmark dataset:
  - dspm27: 29 real PDFs with GPT labels (10+ L1 categories)
  - cxh5types: 258 text docs with HUMAN labels (3 L1 categories)
  - ben25: 25 text docs with GPT labels

Writes documents to temp dir with their native formats, runs the DataDNA
pipeline, and evaluates clustering + classification accuracy against
ground truth labels.

Usage:
    python eval_realdata.py
    python eval_realdata.py --clustering-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ZEROSONE_BASE = r"C:\Users\31531\Desktop\AI\ZerosOne\gemma-doc-label\testdata"


def load_all_data() -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    """Load all three datasets, returning documents and ground truth.

    Each doc is (doc_id, text, file_type, source_dataset).
    Also writes ground truth as {doc_id: l1_label}.
    """
    docs: list[tuple[str, str, str, str]] = []
    ground_truth: dict[str, str] = {}

    # ── dspm27: real PDFs ──────────────────────────────────────
    pdf_dir = os.path.join(ZEROSONE_BASE, "dspm27", "pdfs")
    labels_path = os.path.join(ZEROSONE_BASE, "dspm27", "dspm_gpt52_labels.json")

    with open(labels_path, encoding="utf-8") as f:
        dspm_labels = json.load(f)

    for filename, label_dict in dspm_labels.items():
        pdf_path = os.path.join(pdf_dir, filename)
        if not os.path.exists(pdf_path):
            logger.warning("dspm27 PDF missing: %s", filename)
            continue

        doc_id = f"dspm_{Path(filename).stem.replace(' ', '_')}"
        # Store original PDF path — DataDNA will read it
        docs.append((doc_id, pdf_path, ".pdf", "dspm27"))
        ground_truth[doc_id] = label_dict.get("l1", "unknown")

    # ── cxh5types: text files ──────────────────────────────────
    texts_path = os.path.join(ZEROSONE_BASE, "cxh5types", "cxh5types_texts.jsonl")
    labels_path = os.path.join(ZEROSONE_BASE, "cxh5types", "cxh5types_human_labels.json")

    with open(labels_path, encoding="utf-8") as f:
        cxh_labels = json.load(f)

    with open(texts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            file_id = entry.get("filename", entry.get("file", ""))
            text = entry.get("text", "")
            if not text.strip():
                continue

            doc_id = f"cxh_{file_id.replace('/', '_').replace('.', '_')[:60]}"
            label_entry = cxh_labels.get(file_id)
            if label_entry and label_entry.get("l1"):
                l1 = label_entry["l1"]
            else:
                l1 = "unknown"

            docs.append((doc_id, text, ".md", "cxh5types"))
            ground_truth[doc_id] = l1

    # ── ben25: text files ──────────────────────────────────────
    texts_path = os.path.join(ZEROSONE_BASE, "ben25", "ben25_texts.jsonl")
    labels_path = os.path.join(ZEROSONE_BASE, "ben25", "ben25_gpt52_labels.json")

    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            ben_labels = json.load(f)

        if os.path.exists(texts_path):
            with open(texts_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    file_id = entry.get("filename", entry.get("file", ""))
                    text = entry.get("text", "")
                    if not text.strip():
                        continue

                    doc_id = f"ben_{file_id.replace('/', '_').replace('.', '_')[:60]}"
                    label_entry = ben_labels.get(file_id)
                    if label_entry and label_entry.get("l1"):
                        l1 = label_entry["l1"]
                    else:
                        l1 = "unknown"

                    docs.append((doc_id, text, ".md", "ben25"))
                    ground_truth[doc_id] = l1

    logger.info("Loaded %d documents: %d PDF + %d TXT",
                len(docs),
                sum(1 for d in docs if d[2] == ".pdf"),
                sum(1 for d in docs if d[2] != ".pdf"))
    return docs, ground_truth


def write_temp_docs(docs: list[tuple[str, str, str, str]], tmpdir: str) -> str:
    """Write all documents to a temp directory.

    PDFs are copied as-is. Text docs are written to .txt files with
    .md extension preserved in metadata.

    Returns the input directory path.
    """
    input_dir = os.path.join(tmpdir, "input_docs")
    os.makedirs(input_dir, exist_ok=True)

    pdf_dir = os.path.join(input_dir, "pdfs")
    txt_dir = os.path.join(input_dir, "texts")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    for doc_id, content, file_type, source in docs:
        if file_type == ".pdf":
            # Copy the actual PDF file
            ext = ".pdf"
            dest = os.path.join(pdf_dir, f"{doc_id}{ext}")
            if os.path.exists(content):
                shutil.copy2(content, dest)
        else:
            ext = ".txt"
            dest = os.path.join(txt_dir, f"{doc_id}{ext}")
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(content)

    logger.info("Wrote %d docs to %s", len(docs), input_dir)
    return input_dir


def run_datadna(
    input_dir: str, config_path: str, clustering_only: bool = False,
) -> tuple[dict, dict, list]:
    """Run DataDNA pipeline and return (result_lookup, stats, clusters)."""
    impl_dir = os.path.dirname(os.path.abspath(__file__))
    if impl_dir not in sys.path:
        sys.path.insert(0, impl_dir)

    import yaml
    from src.embeddings.bge_m3 import BgeM3Embedder
    from src.tier0.engine import Tier0Engine
    from src.tier1.semantic import SemanticRefiner
    from src.tier1.structural import StructuralClusterer
    from src.types import ClassificationResult, ClusterInfo, Document

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Load docs — handle PDFs and TXTs separately
    documents: list[Document] = []
    for file_path in sorted(Path(input_dir).rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in (".txt", ".pdf"):
            continue

        doc_id = file_path.stem
        text = ""
        metadata: dict = {
            "file_path": str(file_path),
            "file_type": suffix,
            "file_size": file_path.stat().st_size,
            "path_depth": len(file_path.relative_to(input_dir).parts) - 1,
        }

        if suffix == ".pdf":
            try:
                import fitz
                pdf_doc = fitz.open(str(file_path))
                try:
                    pages = [page.get_text() for page in pdf_doc]
                    metadata["page_count"] = len(pages)
                    text = "\n".join(pages)
                finally:
                    pdf_doc.close()
            except ImportError:
                text = file_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            continue

        documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))

    logger.info("Loaded %d documents (%d PDF, %d TXT)",
                len(documents),
                sum(1 for d in documents if d.metadata.get("file_type") == ".pdf"),
                sum(1 for d in documents if d.metadata.get("file_type") != ".pdf"))

    # ── Common components ──────────────────────────────────────
    engine = Tier0Engine(config.get("tier0", {}))
    structural = StructuralClusterer(
        feature_config=config.get("tier1", {}).get("stage_a", {}).get("structural_features")
    )
    emb_cfg = config.get("embedding", {})
    embedder = BgeM3Embedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "cuda"),
        batch_size=emb_cfg.get("batch_size", 32),
        max_length=emb_cfg.get("max_token_length", 8192),
    )
    stage_b_cfg = config.get("tier1", {}).get("stage_b", {})
    refiner = SemanticRefiner(embedder, stage_b_cfg)

    stats: dict = {}
    overall_start = time.perf_counter()

    # ── Tier 0 ─────────────────────────────────────────────────
    t0_start = time.perf_counter()
    doc_tuples = [(d.doc_id, d.text) for d in documents]
    pii_vectors = engine.extract_batch(doc_tuples)
    for doc, pii_vec in zip(documents, pii_vectors):
        doc.pii_features = pii_vec
    stats["tier0_time_s"] = round(time.perf_counter() - t0_start, 3)
    pii_count = sum(1 for v in pii_vectors if v.pii_features)
    stats["docs_with_pii"] = pii_count
    logger.info("Tier 0: %.3fs, %d/%d docs with PII",
                stats["tier0_time_s"], pii_count, len(documents))

    # ── Tier 1: Structural Clustering ──────────────────────────
    t1_start = time.perf_counter()
    buckets = structural.cluster(documents)
    doc_lookup = {d.doc_id: d for d in documents}
    all_clusters: list[ClusterInfo] = []
    sem_threshold = stage_b_cfg.get("sem_split_threshold", 50)

    for bucket_id, doc_ids in buckets.items():
        bucket_docs = [doc_lookup[did] for did in doc_ids if did in doc_lookup]
        if not bucket_docs:
            continue

        if len(bucket_docs) >= sem_threshold:
            try:
                sub_clusters = refiner.refine(bucket_id, bucket_docs)
                all_clusters.extend(sub_clusters)
            except Exception as e:
                logger.warning("BGE-M3 degraded for bucket %s: %s", bucket_id[:16], e)
                all_clusters.append(ClusterInfo(
                    cluster_id=bucket_id,
                    doc_ids=sorted([d.doc_id for d in bucket_docs]),
                    structural_bucket=bucket_id,
                    cluster_radius=0.0,
                    representative_docs=[d.doc_id for d in bucket_docs[:3]],
                    tfidf_keywords=[],
                    pii_distribution={},
                    language_distribution={},
                ))
        else:
            all_clusters.append(ClusterInfo(
                cluster_id=bucket_id,
                doc_ids=sorted([d.doc_id for d in bucket_docs]),
                structural_bucket=bucket_id,
                cluster_radius=0.0,
                representative_docs=[d.doc_id for d in bucket_docs[:3]],
                tfidf_keywords=[],
                pii_distribution={},
                language_distribution={},
            ))

    stats["tier1_time_s"] = round(time.perf_counter() - t1_start, 3)
    stats["cluster_count"] = len(all_clusters)
    stats["stage_a_buckets"] = len(buckets)
    logger.info("Tier 1: %d structural buckets → %d sub-clusters, %.3fs",
                len(buckets), len(all_clusters), stats["tier1_time_s"])

    # ── Tier 2 / Classification ────────────────────────────────
    if clustering_only:
        results = []
        for cluster in all_clusters:
            for doc_id in cluster.doc_ids:
                results.append(ClassificationResult(
                    doc_id=doc_id,
                    label=cluster.cluster_id,
                    confidence=1.0,
                    method="clustering_only",
                    rationale=f"Cluster {cluster.cluster_id}",
                ))
        stats["method"] = "clustering_only"
    else:
        # Full pipeline with LLM
        from src.llm.client import LLMConfig, MistralClient
        from src.ner.deberta import DebertaNER
        from src.tier2.classifier import Tier2Classifier
        from src.tier2.matching import KnownTypeMatcher
        from src.tier2.propagation import LabelPropagator

        t2 = config.get("tier2", {})
        ner = DebertaNER(
            model_name=t2.get("ner_model", "microsoft/deberta-v3-base"),
            device=t2.get("ner_device", "cuda"),
        )
        t2_llm_cfg = t2.get("llm", {})
        llm_tier2 = MistralClient(LLMConfig(
            api_base=t2_llm_cfg.get("api_base", "http://localhost:11434/v1"),
            model=t2_llm_cfg.get("model", "mistral:7b"),
            quantization=t2_llm_cfg.get("quantization", "4bit"),
            temperature=t2_llm_cfg.get("temperature", 0.3),
        ))
        match_cfg = t2.get("known_type_matching", {})
        matcher = KnownTypeMatcher(known_types=[], config=match_cfg)
        prop_cfg = t2.get("propagation", {})
        propagator = LabelPropagator(prop_cfg)
        classifier = Tier2Classifier(matcher, ner, llm_tier2, propagator)

        if matcher.type_count() == 0:
            logger.info("Cold start: zero-shot LLM per document (%d docs)", len(documents))
            results = classifier.cold_start_classify(documents)
            stats["method"] = "cold_start"
        else:
            t2_start = time.perf_counter()
            results = classifier.classify_clusters(all_clusters, documents)
            stats["tier2_time_s"] = round(time.perf_counter() - t2_start, 3)
            methods = {}
            for r in results:
                methods[r.method] = methods.get(r.method, 0) + 1
            stats["tier2_methods"] = methods
            stats["method"] = "full_pipeline"

    stats["total_time_s"] = round(time.perf_counter() - overall_start, 3)

    # ── Build result lookup ────────────────────────────────────
    result_lookup: dict[str, dict] = {}
    for r in results:
        result_lookup[r.doc_id] = {
            "label": r.label,
            "confidence": r.confidence,
            "method": r.method,
        }
    for cluster in all_clusters:
        for doc_id in cluster.doc_ids:
            if doc_id in result_lookup:
                result_lookup[doc_id]["cluster_id"] = cluster.cluster_id

    # ── Per-bucket file type distribution ──────────────────────
    file_type_stats: dict[str, dict[str, int]] = {}
    for bucket_id, doc_ids in buckets.items():
        pdf_count = 0
        txt_count = 0
        for did in doc_ids:
            d = doc_lookup.get(did)
            if d and d.metadata.get("file_type") == ".pdf":
                pdf_count += 1
            else:
                txt_count += 1
        file_type_stats[bucket_id[:16]] = {"pdf": pdf_count, "txt": txt_count}
    stats["file_type_distribution"] = file_type_stats

    return result_lookup, stats, all_clusters


def evaluate(result_lookup: dict, ground_truth: dict, clusters: list) -> dict:
    """Compute clustering and classification metrics."""
    common_ids = sorted(set(result_lookup.keys()) & set(ground_truth.keys()))
    if not common_ids:
        return {"error": "no_common_ids"}

    y_true = [ground_truth[did] for did in common_ids]
    y_pred = [result_lookup[did]["label"] for did in common_ids]

    # Cluster assignments
    cluster_ids = [result_lookup[did].get("cluster_id", "no_cluster") for did in common_ids]

    # ── Cluster Purity ─────────────────────────────────────────
    cluster_docs: dict[str, list[str]] = defaultdict(list)
    for did, cid in zip(common_ids, cluster_ids):
        cluster_docs[cid].append(did)

    purities = []
    cluster_sizes = []
    for cid, doc_ids_in_cluster in cluster_docs.items():
        if cid == "no_cluster" or not doc_ids_in_cluster:
            continue
        true_labels_in = [ground_truth[did] for did in doc_ids_in_cluster]
        majority_count = Counter(true_labels_in).most_common(1)[0][1]
        purities.append(majority_count / len(doc_ids_in_cluster))
        cluster_sizes.append(len(doc_ids_in_cluster))

    total_in = sum(cluster_sizes)
    weighted_purity = (
        sum(p * s for p, s in zip(purities, cluster_sizes)) / total_in
        if total_in > 0 else 0.0
    )
    macro_purity = float(np.mean(purities)) if purities else 0.0

    # ── ARI / NMI ──────────────────────────────────────────────
    def str_to_codes(strings: list[str]) -> list[int]:
        code_map = {}
        codes = []
        for s in strings:
            if s not in code_map:
                code_map[s] = len(code_map)
            codes.append(code_map[s])
        return codes

    ari = adjusted_rand_score(str_to_codes(y_true), str_to_codes(cluster_ids))
    nmi = normalized_mutual_info_score(str_to_codes(y_true), str_to_codes(cluster_ids))

    # ── Per-class P/R/F1 ───────────────────────────────────────
    mapping: dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        mapping[p][t] += 1

    pred_to_true = {}
    for pred_label, true_counts in mapping.items():
        pred_to_true[pred_label] = true_counts.most_common(1)[0][0]

    all_classes = sorted(set(y_true))
    per_class = {}
    for cls in all_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and pred_to_true.get(p) == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and pred_to_true.get(p) == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and pred_to_true.get(p) != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    macro_prec = np.mean([v["precision"] for v in per_class.values()]) if per_class else 0.0
    macro_rec = np.mean([v["recall"] for v in per_class.values()]) if per_class else 0.0
    macro_f1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0.0

    coverage = sum(1 for lbl in y_pred if lbl not in ("unknown", "unclassified", "")) / len(y_pred)

    method_counts = Counter(result_lookup[did]["method"] for did in common_ids)

    return {
        "num_docs_evaluated": len(common_ids),
        "num_clusters": len(cluster_docs),
        "cluster_purity_weighted": round(weighted_purity, 4),
        "cluster_purity_macro": round(macro_purity, 4),
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "coverage": round(coverage, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {cls: per_class[cls]["f1"] for cls in all_classes},
        "method_distribution": dict(method_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DataDNA real-data evaluation")
    parser.add_argument("--clustering-only", action="store_true",
                       help="Only run Tier 0 + Tier 1 (no LLM)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="./eval_output/")
    parser.add_argument("--no-llm", action="store_true",
                       help="Skip LLM (cluster ID as label)")
    args = parser.parse_args()

    docs, ground_truth = load_all_data()
    logger.info("Ground truth: %d documents, %d unique L1 labels",
                len(ground_truth), len(set(ground_truth.values())))

    tmpdir = tempfile.mkdtemp(prefix="datadna_real_")
    try:
        input_dir = write_temp_docs(docs, tmpdir)
        result_lookup, pipeline_stats, clusters = run_datadna(
            input_dir, args.config,
            clustering_only=args.clustering_only or args.no_llm,
        )

        metrics = evaluate(result_lookup, ground_truth, clusters)

        print("\n" + "=" * 70)
        print("  DataDNA REAL-DATA Evaluation")
        print("  Sources: dspm27 (29 PDFs) + cxh5types (258 TXT) + ben25")
        print("=" * 70)
        print(f"  Documents evaluated:      {metrics['num_docs_evaluated']}")
        print(f"  Clusters discovered:      {metrics['num_clusters']}")
        print(f"  Pipeline total time:      {pipeline_stats.get('total_time_s', 0):.1f}s")
        print(f"  Method:                   {pipeline_stats.get('method', 'unknown')}")
        print(f"  Tier 0 PII detection:     {pipeline_stats.get('docs_with_pii', 0)} docs")
        print(f"  Stage A buckets:          {pipeline_stats.get('stage_a_buckets', '?')}")
        if "tier2_methods" in pipeline_stats:
            print(f"  Tier 2 methods:           {pipeline_stats['tier2_methods']}")
        print("-" * 70)
        print(f"  Cluster Purity (weighted): {metrics['cluster_purity_weighted']:.4f}")
        print(f"  Cluster Purity (macro):    {metrics['cluster_purity_macro']:.4f}")
        print(f"  Adjusted Rand Index:       {metrics['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info:    {metrics['normalized_mutual_info']:.4f}")
        print(f"  Coverage:                  {metrics['coverage']:.4f}")
        print("-" * 70)
        print(f"  Macro Precision:           {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:              {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:                  {metrics['macro_f1']:.4f}")
        print("-" * 70)
        print(f"  Per-class F1:")
        for cls, f1 in sorted(metrics['per_class_f1'].items(), key=lambda x: -x[1])[:10]:
            print(f"    {cls:45s} {f1:.4f}")
        print("-" * 70)
        print(f"  File type distribution (per bucket):")
        for bucket, counts in pipeline_stats.get("file_type_distribution", {}).items():
            print(f"    {bucket}: PDF={counts['pdf']}, TXT={counts['txt']}")
        print("=" * 70)

        purity = metrics['cluster_purity_weighted']
        if purity >= 0.80:
            print("  [GOOD] Cluster purity >= 80% - structural hashing works!")
        elif purity >= 0.50:
            print("  [FAIR] Cluster purity 50-80% - mixed but meaningful")
        else:
            print("  [POOR] Cluster purity < 50%")

        ari = metrics['adjusted_rand_index']
        if ari >= 0.50:
            print("  [GOOD] ARI >= 0.5 - strong label alignment")
        elif ari >= 0.20:
            print("  [FAIR] ARI 0.2-0.5 - moderate alignment")
        else:
            print("  [POOR] ARI < 0.2")

        f1 = metrics['macro_f1']
        if f1 >= 0.85:
            print("  [GOOD] Macro F1 >= 85% - production-ready classification")
        elif f1 >= 0.60:
            print("  [FAIR] Macro F1 60-85%")
        else:
            print("  [POOR] Macro F1 < 60%")

        print()
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, "eval_realdata.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"metrics": metrics, "pipeline_stats": pipeline_stats},
                      fh, ensure_ascii=False, indent=2)
        logger.info("Report saved to %s", report_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
