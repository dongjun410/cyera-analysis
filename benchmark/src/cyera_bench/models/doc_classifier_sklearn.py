import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- External doc-classifier paths (optional, for backward compat) ---
_DOC_CLASSIFIER_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "ZerosOne" / "doc-classifier"
_MODELS_DIR = _DOC_CLASSIFIER_ROOT / "model"
_PER_L1_ROOT = _MODELS_DIR / "l2_per_l1_min30"
_PREDICT_PY = _DOC_CLASSIFIER_ROOT / "tier_classifier" / "dspm_best" / "scripts" / "predict_documents.py"
_TIER_CLASSIFIER_ROOT = _PREDICT_PY.parent.parent.parent


def _ensure_search_path():
    if str(_TIER_CLASSIFIER_ROOT) not in sys.path:
        sys.path.insert(0, str(_TIER_CLASSIFIER_ROOT))

    for mod_name in ("torch", "transformers"):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                sys.modules[mod_name] = _FakeModule(mod_name)


class _FakeModule:
    def __init__(self, name):
        self.__name__ = name

    def __getattr__(self, name):
        return None


def _load_external_predict_module():
    _ensure_search_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict_documents_service", str(_PREDICT_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DocClassifierSklearnModel:
    """TF-IDF + LogisticRegression document classifier.

    Two modes:
    - Native (fit): train on provided dataset labels → fair per-dataset benchmark
    - External (fallback): load pre-trained doc-classifier model from ZerosOne/
    """

    def __init__(self, variant: str = "sklearn", device: str = "cpu",
                 quantization: str | None = None):
        self._is_fit = False

        # L1
        self._l1_tfidf: TfidfVectorizer | None = None
        self._l1_clf: LogisticRegression | None = None

        # Per-L1 L2
        self._l2_tfidfs: Dict[str, TfidfVectorizer] = {}
        self._l2_clfs: Dict[str, LogisticRegression] = {}

        # External model (loaded lazily only when needed)
        self._ext_mod = None
        self._ext_l1_model = None
        self._ext_l1_map = None
        self._ext_per_l1: Dict[str, tuple] = {}

    @property
    def name(self) -> str:
        if self._is_fit:
            return "doc-classifier-sklearn (native TF-IDF+LR, fitted)"
        return "doc-classifier-sklearn (external Dspm model)"

    @property
    def param_count(self) -> int:
        return 0

    def warmup(self, n: int = 3) -> None:
        dummy = ["Warmup document."]
        for _ in range(min(n, 2)):
            _ = self.predict(dummy)

    def predict(self, texts: List[str]) -> List[str]:
        return ["" for _ in texts]

    # ------------------------------------------------------------------
    # Native training path
    # ------------------------------------------------------------------

    def fit(self, texts: List[str], labels: List[Dict[str, str]]) -> None:
        """Train TF-IDF + LogisticRegression on the provided (text, label) pairs.

        Called once per dataset before predict_labels().
        """
        y_l1 = [l["l1"] for l in labels]

        # L1 classifier
        self._l1_tfidf = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            sublinear_tf=True, max_df=0.9, min_df=1,
        )
        X_l1 = self._l1_tfidf.fit_transform(texts)
        self._l1_clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
        )
        self._l1_clf.fit(X_l1, y_l1)

        # Per-L1 L2 classifiers
        self._l2_tfidfs = {}
        self._l2_clfs = {}
        unique_l1 = sorted(set(y_l1))
        for l1_name in unique_l1:
            indices = [i for i, l in enumerate(labels) if l["l1"] == l1_name]
            y_l2 = [labels[i]["l2"] for i in indices]
            unique_l2 = set(y_l2)

            # Need at least 2 classes and 3 samples to train a classifier
            if len(unique_l2) < 2 or len(indices) < 3:
                continue

            sub_texts = [texts[i] for i in indices]
            tfidf = TfidfVectorizer(
                max_features=3000, ngram_range=(1, 2),
                sublinear_tf=True, max_df=0.9, min_df=1,
            )
            X_sub = tfidf.fit_transform(sub_texts)
            clf = LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
            )
            try:
                clf.fit(X_sub, y_l2)
            except ValueError:
                continue
            self._l2_tfidfs[l1_name] = tfidf
            self._l2_clfs[l1_name] = clf

        self._is_fit = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_labels(
        self,
        texts: List[str],
        l1_options: List[str],
        l2_options: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        if self._is_fit:
            return self._predict_native(texts)
        return self._predict_external(texts)

    def _predict_native(self, texts: List[str]) -> List[Dict[str, str]]:
        X = self._l1_tfidf.transform(texts)
        l1_preds = self._l1_clf.predict(X)

        results: List[Dict[str, str]] = []
        for i, text in enumerate(texts):
            l1 = l1_preds[i]
            l2 = "unknown"
            if l1 in self._l2_clfs:
                X_l2 = self._l2_tfidfs[l1].transform([text])
                l2 = self._l2_clfs[l1].predict(X_l2)[0]
            results.append({"l1": l1, "l2": l2})
        return results

    # ------------------------------------------------------------------
    # External model fallback (backward compat)
    # ------------------------------------------------------------------

    def _load_external(self):
        if self._ext_mod is not None:
            return
        if not _MODELS_DIR.exists():
            raise FileNotFoundError(
                f"External doc-classifier not found at {_DOC_CLASSIFIER_ROOT}. "
                "Call fit() to train natively instead."
            )
        self._ext_mod = _load_external_predict_module()
        self._ext_l1_model, self._ext_l1_map = self._ext_mod.load_l1_model(
            _MODELS_DIR, use_enhanced=True
        )
        self._ext_per_l1 = {}
        for d in sorted(_PER_L1_ROOT.iterdir()):
            if not d.is_dir():
                continue
            mf = d / "l2_model.joblib"
            if mf.exists():
                self._ext_per_l1[d.name] = (joblib.load(mf), {})

    def _predict_external(self, texts: List[str]) -> List[Dict[str, str]]:
        self._load_external()
        results: List[Dict[str, str]] = []
        for text in texts:
            results.append(self._classify_one_external(text))
        return results

    def _classify_one_external(self, text: str) -> Dict[str, str]:
        if not text or not text.strip():
            return {"l1": "unknown", "l2": "unknown"}
        try:
            raw = self._ext_mod.classify_document(
                text,
                self._ext_l1_model,
                self._ext_l1_map,
                {}, {}, {}, None,
                filename="inline.txt",
                l2_mode="per_l1",
                per_l1_l2_models=self._ext_per_l1,
            )
        except Exception:
            return {"l1": "unknown", "l2": "unknown"}

        return {
            "l1": raw.get("l1", {}).get("label", "unknown"),
            "l2": raw.get("l2", {}).get("label", "unknown"),
        }
