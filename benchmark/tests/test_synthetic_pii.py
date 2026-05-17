import pytest
from cyera_bench.datasets.synthetic_pii import SyntheticPiiDataset


@pytest.fixture
def ds():
    return SyntheticPiiDataset(size=100, seed=42)


def test_entity_types(ds):
    expected = {
        "CREDIT_CARD", "SSN", "EMAIL", "PHONE", "API_KEY",
        "PASSWORD", "BANK_ACCOUNT", "IP_ADDRESS", "URL",
        "DATE_OF_BIRTH", "DRIVERS_LICENSE", "PASSPORT",
    }
    assert set(ds.entity_types) == expected


def test_load_returns_dataset(ds):
    dataset = ds.load("test")
    assert len(dataset) > 0
    assert "tokens" in dataset.features
    assert "ner_tags" in dataset.features


def test_load_train_split_differs(ds):
    train = ds.load("train")
    test = ds.load("test")
    assert len(train) == 80
    assert len(test) == 20


def test_output_has_bio_format(ds):
    dataset = ds.load("test")
    for tag_seq in dataset["ner_tags"]:
        for tag in tag_seq:
            assert tag in (0, 1, 2)  # O, B-X, I-X
