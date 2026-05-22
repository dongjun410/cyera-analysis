"""
Knowledge Distillation entry point.
Run after initial clustering to generate training data and train the classifier.

Usage:
  # After running main.py at least once:
  python distill.py --clusters output/clusters_*.json --config config.yaml

  # Or as part of the full pipeline:
  python main.py --input /docs --output ./output
  python distill.py --clusters ./output/clusters_latest.json
"""

import os
import json
import yaml
import logging
import argparse
import numpy as np

from core.learned_classifier import LearnedClassifier
from models.schemas import ProcessedDocument, ClusterInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distill")


def load_cluster_results(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation: LLM → Classifier')
    parser.add_argument('--clusters', '-r', required=True, help='Clustering results JSON')
    parser.add_argument('--config', '-c', default='config.yaml')
    parser.add_argument('--output', '-o', default='./classifiers')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding='utf-8'))
    results = load_cluster_results(args.clusters)

    # Reconstruct ClusterInfo objects
    clusters = []
    for c in results.get("clusters", []):
        clusters.append(ClusterInfo(
            cluster_id=c["cluster_id"],
            size=c["size"],
            keywords=c.get("keywords", []),
            llm_label=c.get("llm_label", ""),
            llm_description=c.get("llm_description", ""),
            coherence=c.get("coherence", 0),
            representative_doc_ids=c.get("representative_doc_ids", []),
            document_ids=c.get("document_ids", []),
        ))

    logger.info(f"Loaded {len(clusters)} clusters from {args.clusters}")

    # Reconstruct documents from Elasticsearch
    logger.info("Loading documents from Elasticsearch...")
    from core.vector_store import VectorStore
    store = VectorStore(config["elasticsearch"])

    documents = []
    embeddings_list = []
    labels_list = []
    for cluster in clusters:
        es_docs = store.get_cluster_documents(cluster.cluster_id)
        for es_doc in es_docs:
            doc = ProcessedDocument(
                id=es_doc["doc_id"],
                original_path="",
                title=es_doc.get("title", ""),
                raw_content=es_doc.get("content", ""),
            )
            documents.append(doc)
            embeddings_list.append(es_doc.get("embedding", []))
            labels_list.append(cluster.cluster_id)

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)

    logger.info(f"Loaded {len(documents)} documents")

    # Stage 1: LLM auto-labeling
    lc = LearnedClassifier(
        config.get("learned_classifier", {}),
        config.get("llm", {}),
    )
    training_samples = lc.generate_training_data(
        clusters, documents, embeddings, labels
    )

    if len(training_samples) < 10:
        logger.error(f"Only {len(training_samples)} samples generated. Need at least 10.")
        return

    # Stage 2: Train classifier
    lc.train_classifier(training_samples)

    # Stage 3: Quick validation
    correct = 0
    total = 0
    for sample in training_samples[:100]:
        pred_label, pred_conf, pred_source = lc.classify_document(sample.text)
        if pred_label == sample.label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    logger.info(f"Self-validation accuracy: {accuracy:.1%} ({correct}/{total})")
    logger.info("Distillation complete!")


if __name__ == "__main__":
    main()
