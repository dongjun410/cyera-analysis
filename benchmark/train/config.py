# benchmark/train/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    # --- Model ---
    model_name: str = "google/flan-t5-large"
    quantization: str = "8bit"  # "8bit", "4bit", or "none"

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
    )

    # --- Phase 1: General Classification ---
    phase1_epochs: int = 3
    phase1_lr: float = 2e-4
    phase1_batch_size: int = 8
    phase1_grad_accum: int = 2
    phase1_max_length: int = 1024
    phase1_max_target_length: int = 128
    phase1_warmup_ratio: float = 0.1
    phase1_weight_decay: float = 0.01

    # --- Phase 2: DSPM Domain Adaptation ---
    phase2_epochs: int = 12
    phase2_lr: float = 5e-5
    phase2_batch_size: int = 4
    phase2_grad_accum: int = 2
    phase2_lora_dropout: float = 0.10
    phase2_max_length: int = 1024
    phase2_max_target_length: int = 128
    phase2_warmup_ratio: float = 0.1
    phase2_weight_decay: float = 0.01
    phase2_early_stopping_patience: int = 3

    # --- Data ---
    template_ratios: dict[str, float] = field(default_factory=lambda: {
        "zero_shot": 0.40,
        "few_shot": 0.20,
        "cot": 0.15,
        "label_to_content": 0.15,
        "contrastive": 0.10,
    })
    dbpedia_subsample: float = 0.10
    phase1_val_split: float = 0.05

    # --- Augmentation ---
    augment_back_translation_count: int = 2
    augment_entity_sub_count: int = 3
    augment_llm_synthesis_count: int = 5
    augment_quality_min_similarity: float = 0.15
    augment_quality_max_similarity: float = 0.95

    # --- Output ---
    output_dir: str = "benchmark/models/flan-t5-finetuned"
    seed: int = 42

    def __post_init__(self):
        if self.quantization is None:
            return
        if self.quantization.lower() == "none":
            self.quantization = None
        elif self.quantization not in ("8bit", "4bit"):
            raise ValueError(f"quantization must be '8bit', '4bit', or 'none', got '{self.quantization}'")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # Filter to dataclass fields only
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in field_names}
        return cls(**filtered)
