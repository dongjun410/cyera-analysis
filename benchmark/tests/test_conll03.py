import pytest
from cyera_bench.datasets.conll03 import Conll03Dataset


def test_entity_types():
    ds = Conll03Dataset()
    assert set(ds.entity_types) == {"PER", "ORG", "LOC", "MISC"}


def test_load_test_split():
    ds = Conll03Dataset()
    dataset = ds.load("test")
    assert len(dataset) > 0
    assert "tokens" in dataset.features
    assert "ner_tags" in dataset.features


def test_load_validation_split():
    ds = Conll03Dataset()
    dataset = ds.load("validation")
    assert len(dataset) > 0


def test_texts_method():
    ds = Conll03Dataset()
    texts = ds.texts("test")
    assert isinstance(texts, list)
    assert len(texts) > 0
    assert isinstance(texts[0], list)


def test_bio_tags_method():
    ds = Conll03Dataset()
    tags = ds.bio_tags("test")
    assert isinstance(tags, list)
    assert len(tags) > 0
