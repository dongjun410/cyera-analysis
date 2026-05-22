"""企业文档智能聚类系统 V2.2 — 核心模块"""

import logging

_logger = logging.getLogger(__name__)

_imports = [
    ("document_processor", "DocumentProcessor"),
    ("pii_preclassifier", "PIIPreclassifier"),
    ("structure_feature_extractor", "StructureFeatureExtractor"),
    ("embedding_service", "EmbeddingService"),
    ("clustering_engine", "ClusteringEngine"),
    ("sensitivity_adaptive_scheduler", "SensitivityAdaptiveScheduler"),
    ("iterative_optimizer", "IterativeOptimizer"),
    ("label_propagator", "LabelPropagator"),
    ("quality_evaluator", "QualityEvaluator"),
    ("vector_store", "VectorStore"),
    ("learned_classifier", "LearnedClassifier"),
]

for _mod_name, _class_name in _imports:
    try:
        _mod = __import__(f"core.{_mod_name}", fromlist=[_class_name])
        globals()[_class_name] = getattr(_mod, _class_name)
    except ImportError as _e:
        _logger.warning(f"Skipping core.{_mod_name}: {_e}")
    except Exception as _e:
        _logger.warning(f"Skipping core.{_mod_name}: {_e}")
