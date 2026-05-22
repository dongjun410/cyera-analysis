"""
企业文档智能聚类系统 V2 — 主入口
全流程：解析 → 预分类 → 抽象化 → 向量化 → 聚类 → 迭代优化 → 传播 → 评估 → 存储
"""

import os
import json
import yaml
import logging
import argparse
import numpy as np
from datetime import datetime

from core.document_processor import DocumentProcessor
from core.pii_preclassifier import PIIPreclassifier
from core.structure_feature_extractor import StructureFeatureExtractor
from core.embedding_service import EmbeddingService
from core.iterative_optimizer import IterativeOptimizer
from core.label_propagator import LabelPropagator
from core.quality_evaluator import QualityEvaluator
from core.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger("main")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_dir: str):
    """保存聚类结果 JSON"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"clusters_{ts}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"结果已保存: {path}")
    return path


def generate_report(results: dict, quality: dict, output_dir: str):
    """生成 Markdown 分析报告"""
    report = []
    report.append("# 文档聚类分析报告 (V2)\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("\n## 数据概览\n")
    report.append(f"| 指标 | 值 |\n|------|------|\n")
    report.append(f"| 总文档数 | {results['total_documents']} |\n")
    report.append(f"| 预分类文档 | {results['preclassified_documents']} |\n")
    report.append(f"| 聚类文档 | {results['clustered_documents']} |\n")
    report.append(f"| 聚类数量 | {results['num_clusters']} |\n")
    report.append(f"| 离群文档 | {results['num_outliers']} |\n")

    if quality:
        report.append("\n## 质量评估\n")
        report.append(f"| 指标 | 值 | 说明 |\n|------|------|------|\n")
        if "silhouette_score" in quality:
            s = quality["silhouette_score"]
            grade = "优" if s > 0.5 else "良" if s > 0.3 else "中" if s > 0.1 else "差"
            report.append(f"| Silhouette Score | {s} | {grade}（越接近1越好） |\n")
        if "davies_bouldin_index" in quality:
            d = quality["davies_bouldin_index"]
            grade = "优" if d < 1.0 else "良" if d < 2.0 else "中" if d < 3.0 else "差"
            report.append(f"| Davies-Bouldin Index | {d} | {grade}（越低越好） |\n")
        if "calinski_harabasz_index" in quality:
            report.append(f"| Calinski-Harabasz Index | {quality['calinski_harabasz_index']} | 越高越好 |\n")

    report.append("\n## 主要聚类\n")
    for i, cluster in enumerate(results.get("clusters", [])[:30]):
        label = cluster.get("llm_label") or cluster.get("keywords_label", "")
        report.append(f"\n### 聚类 {i+1}: {label}\n")
        report.append(f"- 文档数: {cluster['size']}\n")
        report.append(f"- 内聚度: {cluster.get('coherence', 'N/A')}\n")
        report.append(f"- 关键词: {', '.join(cluster.get('keywords', [])[:10])}\n")
        if cluster.get("llm_description"):
            report.append(f"- 描述: {cluster['llm_description']}\n")

    path = os.path.join(output_dir, "clustering_report.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    logger.info(f"报告已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description='企业文档智能聚类系统 V2')
    parser.add_argument('--input', '-i', required=True, help='输入文档目录')
    parser.add_argument('--output', '-o', default='./output', help='输出目录')
    parser.add_argument('--config', '-c', default='config.yaml', help='配置文件')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)

    # ═══════════════════════════════════════════════════════════
    # Phase 0: 文档解析
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 0: 文档解析")
    processor = DocumentProcessor(config["document"])
    all_documents = processor.process_directory(args.input)
    logger.info(f"共解析 {len(all_documents)} 篇文档")

    if not all_documents:
        logger.error("没有解析到任何文档，退出")
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 0.5: PII 预分类
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 0.5: PII 预分类")
    pii_classifier = PIIPreclassifier(config["pii_preclassifier"])
    preclassified_docs, clustering_docs = pii_classifier.scan_batch(all_documents)

    if not clustering_docs:
        logger.warning("所有文档均被预分类器标记，无需聚类")
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Structure Feature Extraction (Channel 2)
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 1: Structure Feature Extraction (Channel 2)")
    extractor = StructureFeatureExtractor(config["structure_features"])
    structure_vectors = extractor.extract_batch(clustering_docs)

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 向量化
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 2: Dual-Channel Embedding (Semantic + Structure)")
    embedding_service = EmbeddingService(config["embedding"])
    semantic_vectors = embedding_service.encode_documents(clustering_docs)
    logger.info(f"Channel 1 semantic vectors: shape={semantic_vectors.shape}")

    # Fuse Channel 1 (semantic) + Channel 2 (structure)
    structure_weight = 0.3
    weighted_structure = structure_vectors * structure_weight
    embeddings = np.hstack([semantic_vectors, weighted_structure])
    logger.info(f"Fused embedding: shape={embeddings.shape}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Sensitivity-Adaptive Clustering
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 3: Sensitivity-Adaptive Clustering")

    from core.sensitivity_adaptive_scheduler import SensitivityAdaptiveScheduler
    scheduler = SensitivityAdaptiveScheduler(config["clustering"])
    labels = scheduler.fit(embeddings, structure_vectors)

    # ═══════════════════════════════════════════════════════════
    # Phase 3.5: 迭代优化
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 3.5: Iterative Optimization")
    optimizer = IterativeOptimizer(config["clustering"])
    labels = optimizer.optimize(embeddings, labels)

    # ═══════════════════════════════════════════════════════════
    # Phase 4: 分类传播 & 语义命名
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 4: 分类传播 & 语义命名")
    propagator = LabelPropagator(
        config["label_propagation"],
        config.get("llm", {}),
    )
    clusters = propagator.process_clusters(clustering_docs, embeddings, labels)

    # ═══════════════════════════════════════════════════════════
    # Phase 5: 质量评估
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 5: 质量评估")
    evaluator = QualityEvaluator(config["quality"])
    quality_metrics = evaluator.evaluate(embeddings, labels)

    # ═══════════════════════════════════════════════════════════
    # Phase 5.5: Knowledge Distillation (LLM → Classifier)
    # ═══════════════════════════════════════════════════════════
    lc_config = config.get("learned_classifier", {})
    if lc_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("Phase 5.5: Knowledge Distillation")
        from core.learned_classifier import LearnedClassifier

        lc = LearnedClassifier(lc_config, config.get("llm", {}))

        has_classifier = lc.load_classifier()

        if not has_classifier:
            logger.info("No existing classifier found. Running LLM auto-labeling...")
            training_samples = lc.generate_training_data(
                clusters, clustering_docs, embeddings, labels
            )
            if len(training_samples) >= 10:
                logger.info(f"Training classifier with {len(training_samples)} samples...")
                lc.train_classifier(training_samples)
            else:
                logger.warning(
                    f"Only {len(training_samples)} samples, need >=10. "
                    f"Skipping distillation. Run distill.py manually later."
                )
        else:
            logger.info("Loaded existing classifier. Skipping training.")

    # ═══════════════════════════════════════════════════════════
    # Phase 6: 存储 & 输出
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Phase 6: 存储 & 输出")

    # Elasticsearch
    try:
        store = VectorStore(config["elasticsearch"])
        store.ensure_index()
        cluster_label_map = {c.cluster_id: c.llm_label for c in clusters}
        store.upsert_documents(clustering_docs, embeddings, labels, cluster_label_map)
    except Exception as e:
        logger.warning(f"Elasticsearch 索引失败（可继续）: {e}")

    # 构建结果
    results = {
        "total_documents": len(all_documents),
        "preclassified_documents": len(preclassified_docs),
        "clustered_documents": len(clustering_docs),
        "num_clusters": len(clusters),
        "num_outliers": int((labels == -1).sum()),
        "quality_metrics": quality_metrics,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "keywords": c.keywords,
                "keywords_label": "_".join(c.keywords[:3]),
                "llm_label": c.llm_label,
                "llm_description": c.llm_description,
                "coherence": round(c.coherence, 4),
                "representative_doc_ids": c.representative_doc_ids,
                "document_ids": c.document_ids,
            }
            for c in clusters
        ],
        "preclassified": [
            {
                "doc_id": d.id,
                "title": d.title,
                "label": d.preclassification_label,
                "pii_types": d.pii_types_found,
            }
            for d in preclassified_docs
        ],
    }

    save_results(results, args.output)
    generate_report(results, quality_metrics, args.output)

    # 质量报告单独保存
    quality_path = os.path.join(args.output, "quality_report.json")
    with open(quality_path, 'w') as f:
        json.dump(quality_metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("全部完成！")
    logger.info(f"  总文档: {len(all_documents)}")
    logger.info(f"  预分类: {len(preclassified_docs)}")
    logger.info(f"  聚类数: {len(clusters)}")
    logger.info(f"  Silhouette: {quality_metrics.get('silhouette_score', 'N/A')}")


if __name__ == "__main__":
    main()
