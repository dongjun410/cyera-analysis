#!/usr/bin/env python
"""DataDNA ↔ Benchmark evaluation bridge.

Loads a benchmark classification dataset, writes documents to a temp
directory, runs the DataDNA pipeline, and compares cluster labels against
ground truth using clustering metrics (purity, ARI, NMI) and per-class
precision/recall/F1.

Usage:
    python eval_bridge.py --dataset twenty_newsgroups --size 500
    python eval_bridge.py --dataset ledgar --size 300 --categories 10
    python eval_bridge.py --dataset german_multifin --size 200

Metrics computed:
    - Cluster Purity: fraction of docs in each cluster that share the
      majority ground-truth label. >0.80 = good clustering.
    - Adjusted Rand Index (ARI): similarity between clustering and
      ground truth, corrected for chance. >0.5 = meaningful structure.
    - NMI: Normalized Mutual Information. >0.5 = strong alignment.
    - Per-class Precision/Recall/F1 (after label propagation via
      majority vote mapping).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Step 1: Load benchmark dataset → temp files
# ═══════════════════════════════════════════════════════════════════

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "twenty_newsgroups": {
        "module": "cyera_bench.datasets.twenty_newsgroups",
        "class": "TwentyNewsgroupsDataset",
        "kwargs": {},
        "label_level": "l1",
    },
    "ledgar": {
        "module": "cyera_bench.datasets.ledgar",
        "class": "LedgarDataset",
        "kwargs": {},
        "label_level": "l1",
    },
    "german_multifin": {
        "module": "cyera_bench.datasets.german_multifin",
        "class": "GermanMultiFinDataset",
        "kwargs": {},
        "label_level": "l1",
    },
}


def load_dataset(name: str) -> tuple[list[str], list[str]]:
    """Load a benchmark dataset and return (texts, ground_truth_labels)."""
    # Ensure benchmark package is importable
    benchmark_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark", "src",
    )
    if benchmark_src not in sys.path:
        sys.path.insert(0, benchmark_src)

    cfg = DATASET_CONFIGS[name]
    mod = __import__(cfg["module"], fromlist=[cfg["class"]])
    dataset_cls = getattr(mod, cfg["class"])
    dataset = dataset_cls(**cfg["kwargs"])
    texts, label_dicts = dataset.load()

    # Extract labels at configured level
    level = cfg["label_level"]
    if level == "l1":
        labels = [ld["l1"] for ld in label_dicts]
    else:
        labels = [f"{ld.get('l1', '')}/{ld.get('l2', '')}" for ld in label_dicts]

    return texts, labels


def write_temp_docs(texts: list[str], labels: list[str], tmpdir: str) -> tuple[str, dict[str, str]]:
    """Write document texts to temp files and save ground truth.

    Returns (input_dir, ground_truth_dict) where ground_truth_dict
    maps doc_id (filename stem) → label.
    """
    input_dir = os.path.join(tmpdir, "input_docs")
    os.makedirs(input_dir, exist_ok=True)

    ground_truth: dict[str, str] = {}

    for i, text in enumerate(texts):
        # Sanitize label for filename
        safe_label = labels[i].replace("/", "_").replace(" ", "_")[:30]
        doc_id = f"doc_{i:05d}"
        filename = f"{doc_id}.txt"
        filepath = os.path.join(input_dir, filename)

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(text)

        ground_truth[doc_id] = labels[i]

    logger.info("Wrote %d documents to %s", len(texts), input_dir)
    return input_dir, ground_truth


# ═══════════════════════════════════════════════════════════════════
# Step 2: Run DataDNA pipeline
# ═══════════════════════════════════════════════════════════════════

def run_datadna(
    input_dir: str, output_dir: str, config_path: str,
    clustering_only: bool = False,
) -> dict[str, Any]:
    """Import and run the DataDNA main() function, capturing results.

    Parameters
    ----------
    clustering_only : bool
        If True, only run Tier 0 + Tier 1 (clustering). Skip LLM/NER.
        Labels are assigned as cluster IDs (for purity/ARI/NMI evaluation).

    Returns:
        (result_lookup, stats, all_clusters)
    """
    # Add impl-datadna to path
    impl_dir = os.path.dirname(os.path.abspath(__file__))
    if impl_dir not in sys.path:
        sys.path.insert(0, impl_dir)

    # We can't call main() directly (it calls sys.exit), so we
    # replicate the core pipeline flow inline.
    import yaml

    from src.discovery.loop import DiscoveryLoop
    from src.embeddings.bge_m3 import BgeM3Embedder
    from src.llm.client import LLMConfig, MistralClient
    from src.ner.deberta import DebertaNER
    from src.tier0.engine import Tier0Engine
    from src.tier1.semantic import SemanticRefiner
    from src.tier1.structural import StructuralClusterer
    from src.tier2.classifier import Tier2Classifier
    from src.tier2.matching import KnownTypeMatcher
    from src.tier2.propagation import LabelPropagator
    from src.tier3.quality_gate import QualityGate
    from src.types import ClassificationResult, ClusterInfo, Document

    # Load config
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Load documents
    documents: list[Document] = []
    for file_path in sorted(Path(input_dir).rglob("*.txt")):
        doc_id = file_path.stem
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            documents.append(Document(
                doc_id=doc_id,
                text=text,
                metadata={
                    "file_path": str(file_path),
                    "file_type": ".txt",
                    "file_size": file_path.stat().st_size,
                },
            ))

    logger.info("Loaded %d documents", len(documents))

    # ── Common components (always needed) ──────────────────────
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

    stats: dict[str, Any] = {}
    overall_start = time.perf_counter()

    if not clustering_only:
        # ── Full pipeline: init LLM + NER components ──────────────
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

    # ── Tier 0: PII Feature Extraction ─────────────────────────
    t0_start = time.perf_counter()
    doc_tuples = [(d.doc_id, d.text) for d in documents]
    pii_vectors = engine.extract_batch(doc_tuples)
    for doc, pii_vec in zip(documents, pii_vectors):
        doc.pii_features = pii_vec
    t0_time = round(time.perf_counter() - t0_start, 3)
    stats["tier0_time_s"] = t0_time
    pii_detected = sum(1 for v in pii_vectors if v.pii_features)
    stats["docs_with_pii"] = pii_detected
    logger.info("Tier 0: %.3fs, %d/%d docs with PII",
                t0_time, pii_detected, len(documents))

    # ── Tier 1: Two-Stage Clustering ───────────────────────────
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
            except Exception:
                # BGE-M3 degraded → structural-only cluster
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
    t1_time = round(time.perf_counter() - t1_start, 3)
    stats["tier1_time_s"] = t1_time
    stats["cluster_count"] = len(all_clusters)
    logger.info("Tier 1: %d clusters, %.3fs", len(all_clusters), t1_time)

    # ── Tier 2 / Label Assignment ──────────────────────────────
    if clustering_only:
        # Assign cluster_id as label → evaluate clustering quality directly
        results: list[ClassificationResult] = []
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
    elif matcher.type_count() == 0:
        logger.info("Cold start: running zero-shot LLM per document")
        results = classifier.cold_start_classify(documents)
        stats["method"] = "cold_start"
    else:
        t2_start = time.perf_counter()
        results = classifier.classify_clusters(all_clusters, documents)
        t2_time = round(time.perf_counter() - t2_start, 3)
        stats["tier2_time_s"] = t2_time
        methods = {}
        for r in results:
            methods[r.method] = methods.get(r.method, 0) + 1
        stats["tier2_methods"] = methods
        stats["method"] = "full_pipeline"
        logger.info("Tier 2: %d results, methods=%s, %.3fs", len(results), methods, t2_time)

    stats["total_time_s"] = round(time.perf_counter() - overall_start, 3)

    # Build result lookup: doc_id → {label, confidence, method, cluster_id}
    result_lookup: dict[str, dict[str, Any]] = {}
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

    return result_lookup, stats, all_clusters


# ═══════════════════════════════════════════════════════════════════
# Step 3: Evaluate against ground truth
# ═══════════════════════════════════════════════════════════════════

def evaluate(
    result_lookup: dict[str, dict[str, Any]],
    ground_truth: dict[str, str],
    clusters: list,
) -> dict[str, Any]:
    """Compare DataDNA results against ground truth labels.

    Metrics:
      - Cluster Purity: per-cluster majority fraction, averaged
      - ARI: Adjusted Rand Index (cluster labels vs ground truth)
      - NMI: Normalized Mutual Information
      - Per-class P/R/F1 after majority-vote label mapping
      - Coverage: fraction of documents that received a non-"unknown" label
    """
    # Build aligned arrays of doc_ids present in BOTH results and ground truth
    common_ids = sorted(set(result_lookup.keys()) & set(ground_truth.keys()))

    if not common_ids:
        logger.error("No common doc_ids between results and ground truth!")
        return {"error": "no_common_ids"}

    y_true = [ground_truth[did] for did in common_ids]
    y_pred = [result_lookup[did]["label"] for did in common_ids]

    # Build cluster_id arrays for ARI/NMI
    cluster_ids: list[str] = []
    for did in common_ids:
        cid = result_lookup[did].get("cluster_id", "no_cluster")
        cluster_ids.append(cid)

    # ── Cluster Purity ───────────────────────────────────────────
    # Group by cluster_id
    cluster_docs: dict[str, list[str]] = defaultdict(list)
    for did, cid in zip(common_ids, cluster_ids):
        cluster_docs[cid].append(did)

    purities: list[float] = []
    cluster_sizes: list[int] = []
    for cid, doc_ids_in_cluster in cluster_docs.items():
        if cid == "no_cluster":
            continue
        true_labels_in_cluster = [ground_truth[did] for did in doc_ids_in_cluster]
        # Majority label count / total in cluster
        from collections import Counter
        majority_count = Counter(true_labels_in_cluster).most_common(1)[0][1]
        purity = majority_count / len(doc_ids_in_cluster)
        purities.append(purity)
        cluster_sizes.append(len(doc_ids_in_cluster))

    # Weighted average purity (weighted by cluster size)
    total_in_clusters = sum(cluster_sizes)
    weighted_purity = (
        sum(p * s for p, s in zip(purities, cluster_sizes)) / total_in_clusters
        if total_in_clusters > 0 else 0.0
    )
    macro_purity = np.mean(purities) if purities else 0.0

    # ── ARI / NMI ─────────────────────────────────────────────────
    # Map cluster_ids and true labels to integer codes
    true_codes = _str_to_int_codes(y_true)
    cluster_codes = _str_to_int_codes(cluster_ids)

    ari = adjusted_rand_score(true_codes, cluster_codes)
    nmi = normalized_mutual_info_score(true_codes, cluster_codes)

    # ── Per-class P/R/F1 (majority vote mapping) ───────────────────
    per_class_metrics = _compute_majority_vote_metrics(y_true, y_pred)

    # ── Coverage ──────────────────────────────────────────────────
    labeled = sum(1 for lbl in y_pred if lbl not in ("unknown", "unclassified", ""))
    coverage = labeled / len(y_pred) if y_pred else 0.0

    # ── Method distribution ───────────────────────────────────────
    method_counts: dict[str, int] = defaultdict(int)
    for did in common_ids:
        method = result_lookup[did].get("method", "unknown")
        method_counts[method] += 1

    return {
        "num_docs_evaluated": len(common_ids),
        "num_clusters": len(cluster_docs),
        "cluster_purity_weighted": round(weighted_purity, 4),
        "cluster_purity_macro": round(macro_purity, 4),
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "coverage": round(coverage, 4),
        "macro_precision": round(per_class_metrics["macro_precision"], 4),
        "macro_recall": round(per_class_metrics["macro_recall"], 4),
        "macro_f1": round(per_class_metrics["macro_f1"], 4),
        "per_class_f1": per_class_metrics["per_class_f1"],
        "method_distribution": dict(method_counts),
    }


def _str_to_int_codes(strings: list[str]) -> list[int]:
    """Map string labels to integer codes for sklearn metrics."""
    code_map: dict[str, int] = {}
    codes: list[int] = []
    for s in strings:
        if s not in code_map:
            code_map[s] = len(code_map)
        codes.append(code_map[s])
    return codes


def _compute_majority_vote_metrics(
    y_true: list[str], y_pred: list[str],
) -> dict[str, Any]:
    """Compute per-class P/R/F1 using majority vote label mapping.

    For each predicted label, find the most common true label that it
    maps to. This is the standard evaluation for clustering→classification.
    """
    from collections import Counter

    # Build mapping: predicted_label → Counter of true_labels
    mapping: dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        mapping[p][t] += 1

    # Majority vote mapping: each predicted label → best true label
    pred_to_true: dict[str, str] = {}
    for pred_label, true_counts in mapping.items():
        pred_to_true[pred_label] = true_counts.most_common(1)[0][0]

    # Compute per-class metrics
    all_true_classes = sorted(set(y_true))
    per_class: dict[str, dict[str, float]] = {}

    for cls in all_true_classes:
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

    return {
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class_f1": {cls: per_class[cls]["f1"] for cls in all_true_classes},
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DataDNA pipeline against benchmark datasets"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Which benchmark dataset to use",
    )
    parser.add_argument(
        "--size", type=int, default=200,
        help="Number of documents to sample (default: 200)",
    )
    parser.add_argument(
        "--categories", type=int, default=0,
        help="Limit to top-N most frequent categories (0=all)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="DataDNA config file path",
    )
    parser.add_argument(
        "--output", default="./eval_output/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--clustering-only", action="store_true",
        help="Only run Tier 0 + Tier 1 (no LLM). Evaluate cluster purity/ARI/NMI.",
    )
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────
    logger.info("Loading dataset: %s", args.dataset)
    texts, labels = load_dataset(args.dataset)
    logger.info("Loaded %d total documents with %d unique labels",
                len(texts), len(set(labels)))

    # ── Sample ─────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)

    if args.categories > 0:
        # Keep only top-N most frequent categories
        from collections import Counter
        top_cats = {cat for cat, _ in Counter(labels).most_common(args.categories)}
        filtered = [(t, l) for t, l in zip(texts, labels) if l in top_cats]
        texts, labels = zip(*filtered) if filtered else ([], [])
        texts, labels = list(texts), list(labels)
        logger.info("Filtered to %d categories: %s", args.categories, sorted(top_cats))

    if len(texts) > args.size:
        indices = rng.choice(len(texts), size=args.size, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    logger.info("Sampled %d documents with %d unique labels",
                len(texts), len(set(labels)))

    # ── Write temp files ───────────────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="datadna_eval_")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        input_dir, ground_truth = write_temp_docs(texts, labels, tmpdir)

        # ── Run DataDNA ────────────────────────────────────────────
        mode = "clustering-only" if args.clustering_only else "full pipeline"
        logger.info("Running DataDNA %s on %d documents...", mode, len(texts))
        result_lookup, pipeline_stats, clusters = run_datadna(
            input_dir, output_dir, args.config,
            clustering_only=args.clustering_only,
        )

        # ── Evaluate ───────────────────────────────────────────────
        logger.info("Evaluating against ground truth...")
        metrics = evaluate(result_lookup, ground_truth, clusters)

        # ── Report ─────────────────────────────────────────────────
        print("\n" + "=" * 68)
        print(f"  DataDNA Evaluation Report — {args.dataset}")
        print("=" * 68)
        print(f"  Documents evaluated:      {metrics['num_docs_evaluated']}")
        print(f"  Clusters discovered:      {metrics['num_clusters']}")
        print(f"  Pipeline total time:      {pipeline_stats.get('total_time_s', 0):.2f}s")
        print(f"  Method:                   {pipeline_stats.get('method', 'full_pipeline')}")
        if 'tier2_methods' in pipeline_stats:
            print(f"  Tier 2 methods:           {pipeline_stats['tier2_methods']}")
        print("-" * 68)
        print(f"  Cluster Purity (weighted): {metrics['cluster_purity_weighted']:.4f}")
        print(f"  Cluster Purity (macro):    {metrics['cluster_purity_macro']:.4f}")
        print(f"  Adjusted Rand Index:       {metrics['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info:    {metrics['normalized_mutual_info']:.4f}")
        print(f"  Coverage (labeled docs):   {metrics['coverage']:.4f}")
        print("-" * 68)
        print(f"  Macro Precision:           {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:              {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:                  {metrics['macro_f1']:.4f}")
        print("-" * 68)
        print(f"  Method distribution:")
        for method, count in sorted(metrics['method_distribution'].items()):
            pct = count / metrics['num_docs_evaluated'] * 100
            print(f"    {method:25s} {count:5d} ({pct:5.1f}%)")
        print("=" * 68)

        # ── Save detailed report ───────────────────────────────────
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, f"eval_{args.dataset}_{args.size}.json")
        report = {
            "dataset": args.dataset,
            "size": args.size,
            "seed": args.seed,
            "metrics": metrics,
            "pipeline_stats": pipeline_stats,
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        logger.info("Detailed report saved to %s", report_path)

        # ── Interpret ──────────────────────────────────────────────
        print("\n  Interpretation:")
        purity = metrics['cluster_purity_weighted']
        if purity >= 0.80:
            print(f"  [GOOD] Cluster purity {purity:.2%} — GOOD. Clusters map clearly to true categories.")
        elif purity >= 0.60:
            print(f"  [FAIR]  Cluster purity {purity:.2%} — FAIR. Some clusters are mixed-category.")
        else:
            print(f"  [POOR] Cluster purity {purity:.2%} — POOR. Clusters don't align with true labels.")

        ari = metrics['adjusted_rand_index']
        if ari >= 0.50:
            print(f"  [GOOD] ARI {ari:.3f} — GOOD. Clustering structure matches ground truth.")
        elif ari >= 0.25:
            print(f"  [FAIR]  ARI {ari:.3f} — FAIR. Some structural alignment with labels.")
        else:
            print(f"  [POOR] ARI {ari:.3f} — POOR. Little correlation with ground truth labels.")

        f1 = metrics['macro_f1']
        if f1 >= 0.80:
            print(f"  [GOOD] Macro F1 {f1:.2%} — GOOD classification performance.")
        elif f1 >= 0.50:
            print(f"  [FAIR]  Macro F1 {f1:.2%} — FAIR classification performance.")
        else:
            print(f"  [POOR] Macro F1 {f1:.2%} — POOR. Majority-vote mapping doesn't work well.")
        print()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.debug("Cleaned up temp directory: %s", tmpdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
