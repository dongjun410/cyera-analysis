# benchmark/tests/test_train_config.py
import pytest
import tempfile
from pathlib import Path
from benchmark.train.config import TrainingConfig


def test_default_config():
    cfg = TrainingConfig()
    assert cfg.model_name == "google/flan-t5-large"
    assert cfg.quantization == "8bit"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.phase1_epochs == 3
    assert cfg.phase2_epochs == 12
    assert cfg.phase2_lr < cfg.phase1_lr  # Phase 2 uses lower LR


def test_config_from_yaml():
    yaml_content = """
model_name: "google/flan-t5-base"
phase1_epochs: 5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    cfg = TrainingConfig.from_yaml(tmp_path)
    assert cfg.model_name == "google/flan-t5-base"
    assert cfg.phase1_epochs == 5
    assert cfg.lora_r == 16  # default preserved
    Path(tmp_path).unlink()


def test_config_validation():
    cfg = TrainingConfig(quantization="none")
    assert cfg.quantization is None

    cfg2 = TrainingConfig(phase1_lr=0.001, phase2_lr=0.0001)
    assert cfg2.phase1_lr > cfg2.phase2_lr
