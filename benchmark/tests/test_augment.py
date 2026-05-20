# benchmark/tests/test_augment.py
from benchmark.train.augment import (
    entity_substitution,
    tfidf_quality_filter,
    Augmenter,
)
from benchmark.train.config import TrainingConfig


SAMPLE_DOC = (
    "John Smith submitted the Q3 2023 financial report on October 15, 2023. "
    "The total revenue was $2,500,000 for Acme Corporation."
)


def test_entity_substitution_changes_entities():
    import random
    random.seed(42)
    doc = SAMPLE_DOC
    variants = [entity_substitution(doc) for _ in range(5)]
    for v in variants:
        assert "John Smith" not in v
        assert "Acme Corporation" not in v
        assert v != doc


def test_entity_substitution_preserves_structure():
    doc = "Employee: John Smith. Date: 2023-10-15. Amount: $1,000."
    result = entity_substitution(doc)
    assert "Employee:" in result
    assert "Date:" in result
    assert "Amount:" in result


def test_tfidf_quality_filter_accepts_similar():
    original = "The quarterly financial statement shows increased revenue across all sectors."
    good_variant = "The quarterly financial report indicates revenue growth in every sector."
    results = tfidf_quality_filter([good_variant], [original], min_sim=0.15, max_sim=0.95)
    assert len(results) == 1


def test_tfidf_quality_filter_rejects_dissimilar():
    original = "The quarterly financial statement shows increased revenue."
    bad_variant = "banana orange apple grape fruit smoothie recipe breakfast"
    results = tfidf_quality_filter([bad_variant], [original], min_sim=0.15, max_sim=0.95)
    assert len(results) == 0
