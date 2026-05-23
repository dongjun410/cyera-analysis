#!/usr/bin/env python
"""Two-phase DataDNA evaluation: cold start → state → normal pipeline.

Phase 1: Cold start — zero-shot LLM per document, register discovered types
         into known type library, save state JSON.
Phase 2: Normal pipeline — load state, Tier 0→1→2 with known type matching,
         evaluate clustering purity and known_match precision.

Usage:
    python eval_twopass.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ZEROSONE_BASE = r"C:\Users\31531\Desktop\AI\ZerosOne\gemma-doc-label\testdata"
IMPL_DIR = os.path.dirname(os.path.abspath(__file__))
if IMPL_DIR not in sys.path:
    sys.path.insert(0, IMPL_DIR)


def _extract_keywords(texts: list[str], top_n: int = 15) -> list[str]:
    """Extract top-N TF-IDF keywords from representative texts."""
    if not texts:
        return []
    valid = [t for t in texts if t.strip()]
    if not valid:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(max_features=top_n, stop_words="english",
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        v.fit_transform(valid)
        return list(v.get_feature_names_out()[:top_n])
    except Exception:
        return []


def load_all_data() -> tuple[list, dict[str, str]]:
    """Load dspm27 + cxh5types + ben25. Returns (docs, ground_truth)."""
    docs: list[tuple[str, str, str, str]] = []  # (doc_id, text_or_path, file_type, source)
    ground_truth: dict[str, str] = {}

    # dspm27 PDFs
    pdf_dir = os.path.join(ZEROSONE_BASE, "dspm27", "pdfs")
    with open(os.path.join(ZEROSONE_BASE, "dspm27", "dspm_gpt52_labels.json"), encoding="utf-8") as f:
        dspm_labels = json.load(f)
    for filename, ld in dspm_labels.items():
        pdf_path = os.path.join(pdf_dir, filename)
        if not os.path.exists(pdf_path):
            continue
        doc_id = f"dspm_{Path(filename).stem.replace(' ', '_')}"
        docs.append((doc_id, pdf_path, ".pdf", "dspm27"))
        ground_truth[doc_id] = ld.get("l1", "unknown")

    # cxh5types
    for ds_name in ["cxh5types", "ben25"]:
        texts_path = os.path.join(ZEROSONE_BASE, ds_name, f"{ds_name}_texts.jsonl")
        labels_path = os.path.join(ZEROSONE_BASE, ds_name,
                                   f"{ds_name}_human_labels.json" if ds_name == "cxh5types"
                                   else f"{ds_name}_gpt52_labels.json")
        if not os.path.exists(texts_path) or not os.path.exists(labels_path):
            continue
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
        with open(texts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                entry = json.loads(line)
                text = entry.get("text", "")
                if not text.strip(): continue
                file_id = entry.get("filename", entry.get("file", ""))
                doc_id = f"{ds_name[:3]}_{file_id.replace('/', '_').replace('.', '_')[:60]}"
                le = labels.get(file_id)
                l1 = le.get("l1", "unknown") if le else "unknown"
                docs.append((doc_id, text, ".md", ds_name))
                ground_truth[doc_id] = l1

    return docs, ground_truth


def write_temp_docs(docs: list, tmpdir: str) -> str:
    input_dir = os.path.join(tmpdir, "input_docs")
    os.makedirs(os.path.join(input_dir, "pdfs"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "texts"), exist_ok=True)
    for doc_id, content, file_type, source in docs:
        if file_type == ".pdf":
            dest = os.path.join(input_dir, "pdfs", f"{doc_id}.pdf")
            if os.path.exists(content):
                shutil.copy2(content, dest)
        else:
            dest = os.path.join(input_dir, "texts", f"{doc_id}.txt")
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(content)
    logger.info("Wrote %d docs to %s", len(docs), input_dir)
    return input_dir


def load_documents_from_dir(input_dir: str) -> list:
    """Load documents from temp dir into DataDNA Document objects."""
    from src.types import Document
    documents: list[Document] = []
    for file_path in sorted(Path(input_dir).rglob("*")):
        if not file_path.is_file(): continue
        suffix = file_path.suffix.lower()
        if suffix not in (".txt", ".pdf"): continue
        doc_id = file_path.stem
        text = ""
        metadata = {
            "file_path": str(file_path), "file_type": suffix,
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
        if not text.strip(): continue
        documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))
    return documents


def init_components(config: dict, clustering_only: bool = False) -> dict:
    """Initialize all DataDNA pipeline components."""
    from src.embeddings.bge_m3 import BgeM3Embedder
    from src.tier0.engine import Tier0Engine
    from src.tier1.semantic import SemanticRefiner
    from src.tier1.structural import StructuralClusterer

    comp = {}
    comp["engine"] = Tier0Engine(config.get("tier0", {}))
    emb_cfg = config.get("embedding", {})
    comp["embedder"] = BgeM3Embedder(
        model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
        device=emb_cfg.get("device", "cuda"),
        batch_size=emb_cfg.get("batch_size", 32),
        max_length=emb_cfg.get("max_token_length", 8192),
    )
    stage_a = config.get("tier1", {}).get("stage_a", {})
    comp["structural"] = StructuralClusterer(
        feature_config=stage_a.get("structural_features"))
    stage_b = config.get("tier1", {}).get("stage_b", {})
    comp["refiner"] = SemanticRefiner(comp["embedder"], stage_b)

    if not clustering_only:
        from src.llm.client import LLMConfig, MistralClient
        from src.ner.deberta import DebertaNER
        from src.tier2.classifier import Tier2Classifier
        from src.tier2.matching import KnownTypeMatcher
        from src.tier2.propagation import LabelPropagator

        t2 = config.get("tier2", {})
        comp["ner"] = DebertaNER(
            model_name=t2.get("ner_model", "microsoft/deberta-v3-base"),
            device=t2.get("ner_device", "cuda"),
        )
        t2_llm = t2.get("llm", {})
        comp["llm_tier2"] = MistralClient(LLMConfig(
            api_base=t2_llm.get("api_base", "http://localhost:11434/v1"),
            model=t2_llm.get("model", "mistral:7b"),
            quantization=t2_llm.get("quantization", "4bit"),
            temperature=t2_llm.get("temperature", 0.3),
        ))
        match_cfg = t2.get("known_type_matching", {})
        comp["matcher"] = KnownTypeMatcher(known_types=[], config=match_cfg)
        prop_cfg = t2.get("propagation", {})
        comp["propagator"] = LabelPropagator(prop_cfg)
        comp["classifier"] = Tier2Classifier(
            comp["matcher"], comp["ner"], comp["llm_tier2"], comp["propagator"])

    return comp


def run_tier01(documents: list, config: dict, comp: dict) -> tuple[list, dict]:
    """Run Tier 0 + Tier 1, return (clusters, stats)."""
    from src.tier1.semantic import SemanticRefiner
    from src.types import ClusterInfo

    stats = {}
    t0_start = time.perf_counter()
    doc_tuples = [(d.doc_id, d.text) for d in documents]
    pii_vectors = comp["engine"].extract_batch(doc_tuples)
    for doc, pii_vec in zip(documents, pii_vectors):
        doc.pii_features = pii_vec
    stats["tier0_time_s"] = round(time.perf_counter() - t0_start, 3)
    pii_count = sum(1 for v in pii_vectors if v.pii_features)
    stats["docs_with_pii"] = pii_count

    t1_start = time.perf_counter()
    buckets = comp["structural"].cluster(documents)
    doc_lookup = {d.doc_id: d for d in documents}
    all_clusters: list[ClusterInfo] = []
    sem_threshold = config.get("tier1", {}).get("stage_b", {}).get("sem_split_threshold", 50)

    for bucket_id, doc_ids in buckets.items():
        bucket_docs = [doc_lookup[did] for did in doc_ids if did in doc_lookup]
        if not bucket_docs:
            continue
        if len(bucket_docs) >= sem_threshold:
            try:
                sub_clusters = comp["refiner"].refine(bucket_id, bucket_docs)
                all_clusters.extend(sub_clusters)
            except Exception:
                all_clusters.append(ClusterInfo(
                    cluster_id=bucket_id,
                    doc_ids=sorted([d.doc_id for d in bucket_docs]),
                    structural_bucket=bucket_id, cluster_radius=0.0,
                    representative_docs=[d.doc_id for d in bucket_docs[:3]],
                    tfidf_keywords=[], pii_distribution={}, language_distribution={},
                ))
        else:
            all_clusters.append(ClusterInfo(
                cluster_id=bucket_id,
                doc_ids=sorted([d.doc_id for d in bucket_docs]),
                structural_bucket=bucket_id, cluster_radius=0.0,
                representative_docs=[d.doc_id for d in bucket_docs[:3]],
                tfidf_keywords=[], pii_distribution={}, language_distribution={},
            ))

    stats["tier1_time_s"] = round(time.perf_counter() - t1_start, 3)
    stats["cluster_count"] = len(all_clusters)
    stats["stage_a_buckets"] = len(buckets)
    logger.info("Tier 0+1: %d PII docs, %d buckets → %d clusters (%.3fs + %.3fs)",
                pii_count, len(buckets), len(all_clusters),
                stats["tier0_time_s"], stats["tier1_time_s"])
    return all_clusters, stats, doc_lookup, buckets


def evaluate(result_lookup: dict, ground_truth: dict, clusters: list) -> dict:
    """Compute all evaluation metrics."""
    common_ids = sorted(set(result_lookup.keys()) & set(ground_truth.keys()))
    if not common_ids:
        return {"error": "no_common_ids"}

    y_true = [ground_truth[did] for did in common_ids]
    y_pred = [result_lookup[did]["label"] for did in common_ids]
    cluster_ids = [result_lookup[did].get("cluster_id", "nc") for did in common_ids]

    # Cluster purity
    cd: dict[str, list] = defaultdict(list)
    for did, cid in zip(common_ids, cluster_ids):
        cd[cid].append(did)
    purities, sizes = [], []
    for cid, docs in cd.items():
        if cid == "nc" or not docs: continue
        mc = Counter(ground_truth[d] for d in docs).most_common(1)[0][1]
        purities.append(mc / len(docs))
        sizes.append(len(docs))
    total_in = sum(sizes)
    w_purity = sum(p * s for p, s in zip(purities, sizes)) / total_in if total_in else 0
    m_purity = float(np.mean(purities)) if purities else 0

    # ARI/NMI
    def to_codes(x):
        m = {}; r = []
        for v in x:
            if v not in m: m[v] = len(m)
            r.append(m[v])
        return r
    ari = adjusted_rand_score(to_codes(y_true), to_codes(cluster_ids))
    nmi = normalized_mutual_info_score(to_codes(y_true), to_codes(cluster_ids))

    # Per-class metrics via majority-vote mapping
    mapping = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        mapping[p][t] += 1
    p2t = {p: c.most_common(1)[0][0] for p, c in mapping.items()}
    classes = sorted(set(y_true))
    per_class = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p2t.get(p) == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p2t.get(p) == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p2t.get(p) != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    mp = np.mean([v["precision"] for v in per_class.values()]) if per_class else 0
    mr = np.mean([v["recall"] for v in per_class.values()]) if per_class else 0
    mf1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0
    coverage = sum(1 for l in y_pred if l not in ("unknown", "unclassified", "")) / len(y_pred)
    method_counts = dict(Counter(result_lookup[did]["method"] for did in common_ids))

    return {
        "num_docs": len(common_ids), "num_clusters": len(cd),
        "cluster_purity_weighted": round(w_purity, 4),
        "cluster_purity_macro": round(m_purity, 4),
        "ari": round(ari, 4), "nmi": round(nmi, 4), "coverage": round(coverage, 4),
        "macro_precision": round(mp, 4), "macro_recall": round(mr, 4), "macro_f1": round(mf1, 4),
        "per_class_f1": {c: per_class[c]["f1"] for c in classes},
        "method_distribution": method_counts,
    }


def print_report(metrics: dict, stats: dict, phase: str):
    print(f"\n{'='*65}")
    print(f"  Phase {phase} Results")
    print(f"{'='*65}")
    print(f"  Documents:       {metrics['num_docs']}")
    print(f"  Clusters:        {metrics['num_clusters']}")
    print(f"  Pipeline time:   {stats.get('total_time_s', 0):.1f}s")
    print(f"  Method:          {stats.get('method', '?')}")
    if 'tier2_methods' in stats:
        print(f"  Tier 2 methods:  {stats['tier2_methods']}")
    print(f"  Method dist:     {metrics['method_distribution']}")
    print(f"{'-'*65}")
    print(f"  Cluster Purity (w):  {metrics['cluster_purity_weighted']:.4f}")
    print(f"  Cluster Purity (m):  {metrics['cluster_purity_macro']:.4f}")
    print(f"  ARI:                 {metrics['ari']:.4f}")
    print(f"  NMI:                 {metrics['nmi']:.4f}")
    print(f"  Coverage:            {metrics['coverage']:.4f}")
    print(f"{'-'*65}")
    print(f"  Macro Precision:     {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:        {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"{'-'*65}")
    print(f"  Top-10 Per-class F1:")
    for cls, f1 in sorted(metrics['per_class_f1'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {cls[:50]:50s} {f1:.4f}")
    print(f"{'='*65}")


def main():
    import yaml

    # Load config + data
    with open(os.path.join(IMPL_DIR, "config.yaml"), "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    docs_data, ground_truth = load_all_data()
    logger.info("Loaded %d docs, %d unique labels", len(docs_data), len(set(ground_truth.values())))

    tmpdir = tempfile.mkdtemp(prefix="datadna_2phase_")

    try:
        # ════════════════════════════════════════════════════════
        # PHASE 1: Cold Start — build known types from LLM
        # ════════════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("PHASE 1: Cold Start — zero-shot LLM per document")
        logger.info("=" * 50)

        input_dir = write_temp_docs(docs_data, tmpdir)
        comp = init_components(config)
        documents = load_documents_from_dir(input_dir)

        clusters, stats, doc_lookup, buckets = run_tier01(documents, config, comp)

        # Cold start: per-doc LLM
        overall_start = time.perf_counter()
        results = comp["classifier"].cold_start_classify(documents)
        stats["total_time_s"] = round(time.perf_counter() - overall_start, 3)
        stats["method"] = "cold_start"

        # Build known types from LLM results
        from src.types import KnownType, ClusterInfo
        label_docs: dict[str, list] = defaultdict(list)
        for r in results:
            if r.label and r.label != "unknown":
                label_docs[r.label].append(r.doc_id)

        # Build doc_id → cluster mapping for structural_signature
        doc_to_cluster = {}
        for c in clusters:
            for did in c.doc_ids:
                doc_to_cluster[did] = c

        from collections import Counter as _Ctr
        for label, doc_ids in label_docs.items():
            if len(doc_ids) < 2:
                continue

            # Find most common structural bucket for this label
            bucket_counts = _Ctr()
            for did in doc_ids:
                cl = doc_to_cluster.get(did)
                if cl: bucket_counts[cl.structural_bucket] += 1
            top_bucket = bucket_counts.most_common(1)[0][0] if bucket_counts else ""

            # Compute centroid from member embeddings
            member_embs = []
            for did in doc_ids:
                d = doc_lookup.get(did)
                if d and d.embedding is not None:
                    member_embs.append(d.embedding)
            centroid = None
            if member_embs:
                centroid = np.mean(member_embs, axis=0).astype(np.float32)
                centroid = centroid / (np.linalg.norm(centroid) or 1.0)

            # Get representative keywords (top-15 TF-IDF)
            rep_texts = []
            for did in doc_ids[:10]:
                d = doc_lookup.get(did)
                if d: rep_texts.append(d.text[:2000])

            # Simple keyword extraction
            keywords = _extract_keywords(rep_texts)

            # PII distribution from a sample
            pii_dist: dict[str, int] = {}
            for did in doc_ids[:20]:
                d = doc_lookup.get(did)
                if d and d.pii_features:
                    for pt, cnt in d.pii_features.pii_type_distribution.items():
                        pii_dist[pt] = pii_dist.get(pt, 0) + cnt

            kt = KnownType(
                type_id=f"type_{label.replace(' ', '_').replace('/', '_')[:50]}",
                type_name=label,
                description=f"Discovered via cold start ({len(doc_ids)} docs)",
                structural_signature=top_bucket,
                tfidf_keywords=keywords,
                pii_distribution=pii_dist,
                semantic_centroid=centroid,
                status="active",
                sample_count=len(doc_ids),
            )
            comp["matcher"].register_type(kt)

        logger.info("Registered %d known types from cold start", comp["matcher"].type_count())

        # Phase 1 result_lookup
        p1_lookup = {}
        for r in results:
            p1_lookup[r.doc_id] = {"label": r.label, "confidence": r.confidence, "method": r.method}
        for c in clusters:
            for did in c.doc_ids:
                if did in p1_lookup:
                    p1_lookup[did]["cluster_id"] = c.cluster_id

        p1_metrics = evaluate(p1_lookup, ground_truth, clusters)
        print_report(p1_metrics, stats, "1: Cold Start")

        # Free GPU memory between phases
        import torch
        torch.cuda.empty_cache()

        # ════════════════════════════════════════════════════════
        # PHASE 2: Normal Pipeline — with known types
        # ════════════════════════════════════════════════════════
        # Skip NER for Phase 2 — known type matching doesn't need it
        # Representative docs can be very long, causing CPU OOM in DeBERTa
        comp["ner"].predict_batch = lambda texts, **kw: [[] for _ in texts]
        logger.info("=" * 50)
        logger.info("PHASE 2: Normal Pipeline — %d known types loaded", comp["matcher"].type_count())
        logger.info("=" * 50)

        # Re-run Tier 0+1 (same data, fresh clusters)
        # Reset embeddings
        for d in documents:
            d.embedding = None
            d.pii_features = None
            d.cluster_id = None
            d.label = None

        clusters2, stats2, doc_lookup2, buckets2 = run_tier01(documents, config, comp)

        overall_start2 = time.perf_counter()
        results2 = comp["classifier"].classify_clusters(clusters2, documents)
        stats2["total_time_s"] = round(time.perf_counter() - overall_start2, 3)

        methods2 = {}
        for r in results2:
            methods2[r.method] = methods2.get(r.method, 0) + 1
        stats2["tier2_methods"] = methods2
        stats2["method"] = "normal_pipeline"

        p2_lookup = {}
        for r in results2:
            p2_lookup[r.doc_id] = {"label": r.label, "confidence": r.confidence, "method": r.method}
        for c in clusters2:
            for did in c.doc_ids:
                if did in p2_lookup:
                    p2_lookup[did]["cluster_id"] = c.cluster_id

        p2_metrics = evaluate(p2_lookup, ground_truth, clusters2)
        print_report(p2_metrics, stats2, "2: Normal Pipeline")

        # ════════════════════════════════════════════════════════
        # Comparison Summary
        # ════════════════════════════════════════════════════════
        print(f"\n{'='*65}")
        print(f"  Phase Comparison")
        print(f"{'='*65}")
        print(f"  {'Metric':<30s} {'Phase 1 (Cold)':>15s} {'Phase 2 (Normal)':>15s} {'Target':>8s}")
        print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*8}")
        print(f"  {'Cluster Purity (weighted)':<30s} {p1_metrics['cluster_purity_weighted']:>15.4f} {p2_metrics['cluster_purity_weighted']:>15.4f} {'>0.80':>8s}")
        print(f"  {'Macro F1':<30s} {p1_metrics['macro_f1']:>15.4f} {p2_metrics['macro_f1']:>15.4f} {'>0.85':>8s}")
        print(f"  {'Macro Precision':<30s} {p1_metrics['macro_precision']:>15.4f} {p2_metrics['macro_precision']:>15.4f}")
        print(f"  {'Macro Recall':<30s} {p1_metrics['macro_recall']:>15.4f} {p2_metrics['macro_recall']:>15.4f}")
        print(f"  {'ARI':<30s} {p1_metrics['ari']:>15.4f} {p2_metrics['ari']:>15.4f}")
        print(f"  {'NMI':<30s} {p1_metrics['nmi']:>15.4f} {p2_metrics['nmi']:>15.4f}")
        print(f"  {'Coverage':<30s} {p1_metrics['coverage']:>15.4f} {p2_metrics['coverage']:>15.4f}")
        print(f"{'='*65}")

        # Method distribution shift
        p1_methods = p1_metrics.get("method_distribution", {})
        p2_methods = p2_metrics.get("method_distribution", {})
        if p2_methods:
            known_match_pct = p2_methods.get("known_match", 0) / sum(p2_methods.values()) * 100
            llm_pct = p2_methods.get("llm_tier2", 0) / sum(p2_methods.values()) * 100
            print(f"\n  Phase 2 LLM reduction: {llm_pct:.1f}% via LLM, {known_match_pct:.1f}% via known_match")
            if known_match_pct > 50:
                print(f"  [GOOD] known_match > 50% — state accumulation is effective")

        # Verdict
        print(f"\n  Production Readiness:")
        checks = []
        checks.append(("End-to-end F1 > 85%", p2_metrics['macro_f1'] >= 0.85))
        checks.append(("Cluster Purity > 80%", p2_metrics['cluster_purity_weighted'] >= 0.80))
        known_match_ok = p2_methods.get("known_match", 0) / max(sum(p2_methods.values()), 1) > 0.5 if p2_methods else False
        checks.append(("known_match > 50% of docs", known_match_ok))
        for name, ok in checks:
            print(f"    {'[PASS]' if ok else '[FAIL]'} {name}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
