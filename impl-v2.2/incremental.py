"""
Incremental document processing with classifier-first architecture:
  1. Parse + PII pre-classify
  2. Try distilled classifier (fast, ~2ms/doc)
  3. If classifier confidence high → accept label
  4. If low → try kNN cluster assignment as fallback
  5. If still low → LLM fallback (slow but accurate)
  6. LLM results feed back into training buffer for retraining
"""

import os
import json
import yaml
import numpy as np
import logging
import argparse

from sklearn.metrics.pairwise import cosine_similarity

from core.document_processor import DocumentProcessor
from core.pii_preclassifier import PIIPreclassifier
from core.structure_feature_extractor import StructureFeatureExtractor
from core.embedding_service import EmbeddingService
from core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("incremental")


def main():
    parser = argparse.ArgumentParser(description='Incremental update with classifier-first')
    parser.add_argument('--input', '-i', required=True, help='New documents directory')
    parser.add_argument('--config', '-c', default='config.yaml')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='kNN cosine similarity threshold')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding='utf-8'))

    # 1. Parse new documents
    processor = DocumentProcessor(config["document"])
    new_docs = processor.process_directory(args.input)

    # 2. PII pre-classify
    pii = PIIPreclassifier(config["pii_preclassifier"])
    preclassified, to_process = pii.scan_batch(new_docs)

    # 3. Structure features + embedding
    extractor = StructureFeatureExtractor(config["structure_features"])
    structure_vecs = extractor.extract_batch(to_process)

    emb_service = EmbeddingService(config["embedding"])
    new_embeddings = emb_service.encode_documents(to_process)

    # 4. Try distilled classifier first (if available)
    lc_config = config.get("learned_classifier", {})
    classifier_assigned = 0
    knn_assigned = 0
    llm_fallback_count = 0
    buffered = 0

    lc = None
    if lc_config.get("enabled", False):
        from core.learned_classifier import LearnedClassifier
        lc = LearnedClassifier(lc_config, config.get("llm", {}))
        has_classifier = lc.load_classifier()
        if has_classifier:
            logger.info("Distilled classifier loaded. Using classifier-first strategy.")
        else:
            logger.info("No classifier found. Using kNN-only strategy.")
            lc = None

    store = VectorStore(config["elasticsearch"])

    for i, doc in enumerate(to_process):
        label = None
        source = None

        # Strategy A: Distilled classifier (fast)
        if lc is not None:
            pred_label, pred_conf, pred_source = lc.classify_document(doc.raw_content)
            if pred_source == "classifier" and pred_conf >= 0.8:
                label = pred_label
                source = "classifier"
                classifier_assigned += 1
            elif pred_source == "llm_fallback":
                label = pred_label
                source = "llm_fallback"
                llm_fallback_count += 1

        # Strategy B: kNN cluster assignment (if classifier didn't handle it)
        if label is None:
            results = store.search_similar(new_embeddings[i], k=1)
            if results:
                top = results[0]
                sim = cosine_similarity(
                    new_embeddings[i].reshape(1, -1),
                    np.array(top["embedding"]).reshape(1, -1),
                )[0][0]
                if sim >= args.threshold:
                    doc.cluster_id = top["cluster_id"]
                    doc.classification_source = "knn_assign"
                    knn_assigned += 1
                    label = top.get("cluster_label", "")
                    source = "knn"

        # Strategy C: Buffer (nothing worked)
        if label is None:
            doc.cluster_id = -1
            doc.classification_source = "buffered"
            buffered += 1

    logger.info(
        f"Incremental processing complete: "
        f"classifier={classifier_assigned}, "
        f"knn={knn_assigned}, "
        f"llm_fallback={llm_fallback_count}, "
        f"buffered={buffered}"
    )

    # 5. Index all documents
    labels = np.array([d.cluster_id for d in to_process])
    store.upsert_documents(to_process, new_embeddings, labels)

    if buffered > 0:
        logger.info(
            f"{buffered} docs buffered. Run full re-clustering when buffer is large enough."
        )


if __name__ == "__main__":
    main()
