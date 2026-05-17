import pytest
from cyera_bench.models.flan_t5 import FlanT5Model
from cyera_bench.types import Entity


def test_model_variants():
    variants = {
        "small": ("google/flan-t5-small", 77_000_000),
        "base": ("google/flan-t5-base", 250_000_000),
        "large": ("google/flan-t5-large", 780_000_000),
        "xl": ("google/flan-t5-xl", 3_000_000_000),
    }
    for variant, (expected_name, expected_params) in variants.items():
        m = FlanT5Model(variant=variant, device="cpu")
        assert m.name == expected_name
        assert m.param_count == expected_params


@pytest.mark.skip(reason="requires model download from HuggingFace (~500MB)")
def test_predict_returns_list_of_entity_lists():
    m = FlanT5Model(variant="base", device="cpu")
    texts = ["John works at Google in New York.", "Mary visited Paris."]
    results = m.predict(texts)
    assert len(results) == 2
    assert all(isinstance(e, Entity) for e in results[0])


def test_predict_empty_texts():
    m = FlanT5Model(variant="base", device="cpu")
    results = m.predict([])
    assert results == []


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        FlanT5Model(variant="nonexistent", device="cpu")
