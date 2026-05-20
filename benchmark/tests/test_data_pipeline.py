# benchmark/tests/test_data_pipeline.py
import pytest
from benchmark.train.data_pipeline import (
    TemplateEngine,
    TEMPLATES,
    build_phase1_dataset,
    compute_sampling_weights,
)
from benchmark.train.config import TrainingConfig


SAMPLE_TEXT = "The board of directors approved the annual budget for fiscal year 2024."
SAMPLE_L1_OPTIONS = ["Financial Reports", "HR Documents", "IT Security", "Legal Contracts"]
SAMPLE_LABEL = {"l1": "Financial Reports", "l2": "Annual Budget"}


def test_zero_shot_template():
    engine = TemplateEngine()
    result = engine.apply("zero_shot", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    assert "input" in result
    assert "target" in result
    assert "Financial Reports" in result["target"]
    assert SAMPLE_TEXT[:50] in result["input"]
    assert "Classify" in result["input"]


def test_few_shot_template():
    engine = TemplateEngine()
    example_pool = [
        ("Revenue by region Q2.docx", "Financial Reports"),
        ("Server firewall config.docx", "IT Security"),
    ]
    result = engine.apply(
        "few_shot", SAMPLE_TEXT, SAMPLE_L1_OPTIONS,
        SAMPLE_LABEL, example_pool=example_pool,
    )
    assert "input" in result
    assert "Financial Reports" in result.get("target", "")
    input_text = result["input"]
    assert "Revenue" in input_text or "firewall" in input_text.lower()


def test_cot_template():
    engine = TemplateEngine()
    result = engine.apply("cot", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    assert "Step 1" in result["input"]
    assert "Step 2" in result["input"]
    assert "Financial Reports" in result["target"]


def test_label_to_content_template():
    engine = TemplateEngine()
    result = engine.apply("label_to_content", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    assert "Financial Reports" in result["input"]
    assert len(result["target"]) > 0


def test_contrastive_template():
    engine = TemplateEngine()
    result = engine.apply("contrastive", SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
    input_text = result["input"]
    assert "Financial Reports" in input_text
    assert "wrong" in input_text.lower() or "incorrect" in input_text.lower() or "correct" in input_text.lower()


def test_all_templates_produce_valid_format():
    engine = TemplateEngine()
    for tpl_name in TEMPLATES:
        result = engine.apply(tpl_name, SAMPLE_TEXT, SAMPLE_L1_OPTIONS, SAMPLE_LABEL)
        assert isinstance(result, dict), f"{tpl_name}: result not a dict"
        assert "input" in result, f"{tpl_name}: missing 'input'"
        assert "target" in result, f"{tpl_name}: missing 'target'"
        assert len(result["input"]) > 0, f"{tpl_name}: empty input"
        assert len(result["target"]) > 0, f"{tpl_name}: empty target"


def test_compute_sampling_weights():
    dataset_sizes = {
        "20newsgroups": 7532,
        "ledgar": 10000,
        "ag_news": 120000,
        "dbpedia": 56000,
        "german_multifin": 2010,
    }
    weights = compute_sampling_weights(dataset_sizes)
    assert weights["german_multifin"] > weights["ag_news"]
    assert weights["ag_news"] < weights["20newsgroups"]
    assert all(w > 0 for w in weights.values())


def test_template_ratios_in_selection():
    """TemplateEngine.select_template with custom ratios respects the distribution."""
    ratios = {"zero_shot": 1.0, "few_shot": 0.0, "cot": 0.0, "label_to_content": 0.0, "contrastive": 0.0}
    engine = TemplateEngine(seed=42, template_ratios=ratios)
    for _ in range(20):
        assert engine.select_template() == "zero_shot"
