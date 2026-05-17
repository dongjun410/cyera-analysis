import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib

_DOC_CLASSIFIER_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "ZerosOne" / "doc-classifier"
_MODELS_DIR = _DOC_CLASSIFIER_ROOT / "model"
_PER_L1_ROOT = _MODELS_DIR / "l2_per_l1_min30"
_PREDICT_PY = _DOC_CLASSIFIER_ROOT / "tier_classifier" / "dspm_best" / "scripts" / "predict_documents.py"
_TIER_CLASSIFIER_ROOT = _PREDICT_PY.parent.parent.parent


def _ensure_search_path():
    if str(_TIER_CLASSIFIER_ROOT) not in sys.path:
        sys.path.insert(0, str(_TIER_CLASSIFIER_ROOT))

    # Mock optional heavy imports that aren't needed for per_l1 sklearn mode
    for mod_name in ("torch", "transformers"):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                sys.modules[mod_name] = _FakeModule(mod_name)


class _FakeModule:
    """Placeholder for optional heavy modules not needed in per_l1 mode."""
    def __init__(self, name):
        self.__name__ = name

    def __getattr__(self, name):
        return None


def _load_predict_module():
    """Dynamically import predict_documents.py from doc-classifier."""
    _ensure_search_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict_documents_service", str(_PREDICT_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DocClassifierSklearnModel:
    """doc-classifier sklearn pipeline as a benchmark model.

    Wraps the L1 hybrid enhanced + per-L1 L2 models from doc-classifier
    as a local sklearn inference engine (no GPU, no network).
    """

    def __init__(self, variant: str = "sklearn", device: str = "cpu",
                 quantization: str | None = None):
        if not _MODELS_DIR.exists():
            raise FileNotFoundError(
                f"doc-classifier not found at {_DOC_CLASSIFIER_ROOT}. "
                "Ensure ZerosOne/doc-classifier exists."
            )

        self._mod = _load_predict_module()

        # Load L1 model
        self._l1_model, self._l1_map = self._mod.load_l1_model(
            _MODELS_DIR, use_enhanced=True
        )

        # Load per-L1 L2 models
        self._per_l1_models: Dict[str, tuple] = {}
        for d in sorted(_PER_L1_ROOT.iterdir()):
            if not d.is_dir():
                continue
            mf = d / "l2_model.joblib"
            if mf.exists():
                self._per_l1_models[d.name] = (joblib.load(mf), {})

    @property
    def name(self) -> str:
        return "doc-classifier (sklearn TF-IDF+LR)"

    @property
    def param_count(self) -> int:
        return 0  # sklearn model - param count not meaningful

    def warmup(self, n: int = 3) -> None:
        dummy = ["Warmup document."]
        for _ in range(min(n, 2)):
            _ = self.predict(dummy)

    def predict(self, texts: List[str]) -> List[str]:
        """Forward pass — returns empty list (placeholder for throughput measurement)."""
        return ["" for _ in texts]

    def predict_labels(
        self,
        texts: List[str],
        l1_options: List[str],
        l2_options: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        """Classify documents using doc-classifier's L1→L2 pipeline."""
        results: List[Dict[str, str]] = []
        for text in texts:
            result = self._classify_one(text)
            results.append(result)

        return results

    def _classify_one(self, text: str) -> Dict[str, str]:
        if not text or not text.strip():
            return {"l1": "unknown", "l2": "unknown"}

        try:
            raw = self._mod.classify_document(
                text,
                self._l1_model,
                self._l1_map,
                {},
                {},
                {},
                None,
                filename="inline.txt",
                l2_mode="per_l1",
                per_l1_l2_models=self._per_l1_models,
            )
        except Exception:
            return {"l1": "unknown", "l2": "unknown"}

        l1_label = raw.get("l1", {}).get("label", "unknown")
        l2_label = raw.get("l2", {}).get("label", "unknown")
        return {"l1": l1_label, "l2": l2_label}
