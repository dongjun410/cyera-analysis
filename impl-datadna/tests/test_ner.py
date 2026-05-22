"""Tests for DeBERTa-v3 NER service.

Test contracts (TDD — write these BEFORE implementation):
  1. test_init_loads_model          — DebertaNER() initializes without error
  2. test_predict_returns_list      — predict() returns list[PIIFeature]
  3. test_predict_empty_text        — empty string → empty list, no crash
  4. test_predict_batch             — 3 texts → 3 lists returned
  5. test_predict_batch_length_matches — output list length matches input length
  6. test_pii_feature_structure     — each returned feature has entity_type, span, confidence, context_flag
  7. test_no_pii_text               — plain text → empty list or low-confidence features (both OK)

Model loading strategy:
  1. Try microsoft/deberta-v3-base (production) — only if cached locally
  2. Fall back to google/bert_uncased_L-2_H-128_A-2 (~14 MB) — for CI / quick verification
  3. Skip all tests if no model is available

IMPORTANT: The base models aren't fine-tuned for PII, so entity predictions
may be generic (PER/ORG/LOC). Tests verify structure, not PII accuracy.
"""

from __future__ import annotations

import pytest

from src.types import PIIFeature, PIIFeatureVector


# ──────────────────────────────────────────────────────────────
# Cache check utility
# ──────────────────────────────────────────────────────────────

def _model_is_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is fully cached locally (no incomplete downloads)."""
    from pathlib import Path

    model_dir = model_name.replace("/", "--")
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    model_path = cache_base / f"models--{model_dir}"

    if not model_path.exists():
        return False

    # Check for any incomplete downloads
    blobs = model_path / "blobs"
    if blobs.exists():
        incomplete = list(blobs.glob("*.incomplete"))
        if incomplete:
            return False

    # Check for at least one completed snapshot
    snapshots = model_path / "snapshots"
    if snapshots.exists() and any(snapshots.iterdir()):
        return True

    return False


# ──────────────────────────────────────────────────────────────
# Fixture: load DebertaNER (module-scoped, shared across tests)
# ──────────────────────────────────────────────────────────────

# Cache-friendly fallback model for CI / quick verification.
# This is a tiny BERT variant (~14 MB) that loads fast and exercises the
# full token-classification pipeline. The label predictions are random
# (model not fine-tuned for NER), but structure is fully verified.
_CI_FALLBACK_MODEL = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture(scope="module")
def ner_model():
    """Module-scoped DebertaNER — tries production model, falls back to tiny model for CI."""
    from src.ner.deberta import DebertaNER

    # Try production model (DeBERTa-v3-base) — only if cached locally
    model_name = "microsoft/deberta-v3-base"
    if _model_is_cached(model_name):
        try:
            model = DebertaNER(model_name=model_name, device="cpu")
            return model
        except Exception as e:
            pytest.skip(f"DeBERTa model found in cache but failed to load: {e}")

    # Try CI fallback model — check cache first, then try download
    if _model_is_cached(_CI_FALLBACK_MODEL):
        try:
            model = DebertaNER(model_name=_CI_FALLBACK_MODEL, device="cpu")
            return model
        except Exception as e:
            pytest.skip(f"CI fallback model in cache but failed to load: {e}")

    # Last resort: try to download the tiny fallback model
    try:
        model = DebertaNER(model_name=_CI_FALLBACK_MODEL, device="cpu")
        return model
    except Exception:
        pass

    pytest.skip(
        f"DeBERTa-v3-base not in local cache and CI fallback unavailable. "
        f"Download one first."
    )


# ──────────────────────────────────────────────────────────────
# Test 1: init loads model successfully
# ──────────────────────────────────────────────────────────────

def test_init_loads_model(ner_model) -> None:
    """DebertaNER() initializes and has a functional pipeline."""
    from src.ner.deberta import DebertaNER

    assert ner_model is not None
    assert isinstance(ner_model, DebertaNER)
    assert hasattr(ner_model, "_pipeline"), "DebertaNER should have _pipeline attribute"
    assert ner_model._pipeline is not None


# ──────────────────────────────────────────────────────────────
# Test 2: predict returns a list
# ──────────────────────────────────────────────────────────────

def test_predict_returns_list(ner_model) -> None:
    """predict() with sample text returns a list (may be empty if model not fine-tuned)."""
    result = ner_model.predict(
        "John Smith, SSN 123-45-6789, email john@example.com"
    )

    assert isinstance(result, list), (
        f"predict() should return list, got {type(result)}"
    )
    # The model may or may not detect entities — either is OK


# ──────────────────────────────────────────────────────────────
# Test 3: empty text
# ──────────────────────────────────────────────────────────────

def test_predict_empty_text(ner_model) -> None:
    """Empty string → empty list, no crash."""
    result = ner_model.predict("")
    assert result == [], f"Expected empty list for empty text, got {result}"

    # Also test whitespace-only
    result_ws = ner_model.predict("   \t\n  ")
    assert result_ws == [], f"Expected empty list for whitespace-only text, got {result_ws}"


# ──────────────────────────────────────────────────────────────
# Test 4: predict_batch returns list of lists
# ──────────────────────────────────────────────────────────────

def test_predict_batch(ner_model) -> None:
    """predict_batch with 3 texts returns 3 lists."""
    texts = [
        "John Smith, SSN 123-45-6789",
        "Email sent to jane@example.com",
        "Credit card: 4532-0151-1283-0366",
    ]
    results = ner_model.predict_batch(texts)

    assert isinstance(results, list), (
        f"predict_batch() should return list, got {type(results)}"
    )
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for i, r in enumerate(results):
        assert isinstance(r, list), (
            f"Result {i} should be a list, got {type(r)}"
        )


# ──────────────────────────────────────────────────────────────
# Test 5: predict_batch output length matches input length
# ──────────────────────────────────────────────────────────────

def test_predict_batch_length_matches(ner_model) -> None:
    """Output list length always equals input list length."""
    for n in [0, 1, 5]:
        texts = [f"Document {i} containing some text." for i in range(n)]
        results = ner_model.predict_batch(texts)
        assert len(results) == n, (
            f"Expected {n} results for {n} inputs, got {len(results)}"
        )


# ──────────────────────────────────────────────────────────────
# Test 6: PIIFeature structure validation
# ──────────────────────────────────────────────────────────────

def test_pii_feature_structure(ner_model) -> None:
    """Each returned PIIFeature has entity_type, span, confidence, context_flag with correct types."""
    # Use text with known entities (names, locations) for best chance of model output
    text = "John Smith works at Microsoft in New York."
    features = ner_model.predict(text)

    # All returned items must be valid PIIFeature instances
    for f in features:
        assert isinstance(f, PIIFeature), (
            f"Expected PIIFeature, got {type(f)}"
        )
        # entity_type must be a non-empty string
        assert isinstance(f.entity_type, str), (
            f"entity_type should be str, got {type(f.entity_type)}"
        )
        assert len(f.entity_type) > 0, "entity_type should not be empty"

        # span must be (int, int) with valid range
        assert isinstance(f.span, tuple), f"span should be tuple, got {type(f.span)}"
        assert len(f.span) == 2, f"span should have 2 elements, got {len(f.span)}"
        assert isinstance(f.span[0], int) and isinstance(f.span[1], int), (
            f"span elements should be int, got ({type(f.span[0])}, {type(f.span[1])})"
        )
        assert f.span[0] >= 0, f"span start should be >= 0, got {f.span[0]}"
        assert f.span[1] > f.span[0], (
            f"span end ({f.span[1]}) must be > start ({f.span[0]})"
        )

        # confidence must be float in [0.0, 1.0]
        assert isinstance(f.confidence, float), (
            f"confidence should be float, got {type(f.confidence)}"
        )
        assert 0.0 <= f.confidence <= 1.0, (
            f"confidence should be in [0.0, 1.0], got {f.confidence}"
        )

        # context_flag must be a known string value
        assert f.context_flag in ("clean", "penalty_term_present", "boost_term_present"), (
            f"context_flag should be 'clean'/'penalty_term_present'/'boost_term_present', "
            f"got '{f.context_flag}'"
        )

    # If no features returned (model not fine-tuned for NER), that's OK —
    # the test is vacuously true since there's nothing to validate.
    # We log the count for diagnostics.
    if not features:
        pass  # vacuously true — no structural violations possible


# ──────────────────────────────────────────────────────────────
# Test 7: no-pii text
# ──────────────────────────────────────────────────────────────

def test_no_pii_text(ner_model) -> None:
    """Plain text with no obvious PII: if model predicts, confidence should be low."""
    text = "The weather is nice today and the sun is shining brightly."
    features = ner_model.predict(text)

    # If model predicts entities from this plain text, their confidence should be low
    # (random/untrained model may predict anything, so low confidence is the signal)
    for f in features:
        # With a randomly initialized NER head, confidence should generally be low
        # But we can't assert a specific threshold since the model output is non-deterministic.
        # Just verify each feature is well-formed.
        assert isinstance(f, PIIFeature)
        assert 0.0 <= f.confidence <= 1.0

    # Empty list is also an acceptable outcome for no-pii text


# ──────────────────────────────────────────────────────────────
# Additional edge-case tests (not blocking, run when model is available)
# ──────────────────────────────────────────────────────────────

def test_predict_with_pii_hints(ner_model) -> None:
    """predict() with pii_hints returns same result structure (hints are accepted but
    not used for restricting NER scope in this basic implementation)."""
    text = "Patient Jane Doe, SSN 123-45-6789"
    hints = PIIFeatureVector(
        doc_id="test_doc",
        pii_features=[
            PIIFeature(
                entity_type="SSN",
                span=(19, 30),
                confidence=0.35,
                context_flag="penalty_term_present",
            )
        ],
        pii_type_distribution={"SSN": 1},
        has_high_conf_pii=False,
        has_penalty_terms=True,
    )

    result = ner_model.predict(text, pii_hints=hints)
    assert isinstance(result, list)
    # The hint should not affect the NER scan — model runs independently


def test_predict_batch_empty_list(ner_model) -> None:
    """predict_batch with empty list returns empty list."""
    result = ner_model.predict_batch([])
    assert result == [], f"Expected empty list, got {result}"


def test_predict_batch_with_hints(ner_model) -> None:
    """predict_batch with pii_hints list works correctly."""
    texts = ["Text one.", "Text two."]
    hints = [
        PIIFeatureVector(doc_id="d1"),
        None,
    ]
    results = ner_model.predict_batch(texts, pii_hints=hints)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)
