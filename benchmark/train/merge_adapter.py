"""Merge LoRA adapter weights into base model and export as standard HF model."""
from __future__ import annotations

import os
import shutil

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from benchmark.train.config import TrainingConfig


def merge_and_save(cfg: TrainingConfig) -> str:
    """Merge the Phase 2 LoRA adapter into the base model and save as HF format.
    Returns the output path."""
    output_path = os.path.join(cfg.output_dir, "merged")
    adapter_path = os.path.join(cfg.output_dir, "phase2_adapter")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. Run Phase 2 training first."
        )

    print(f"Loading base model: {cfg.model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    for subdir in ["phase1_checkpoints", "phase2_checkpoints"]:
        ckpt_path = os.path.join(cfg.output_dir, subdir)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
            print(f"Cleaned up: {ckpt_path}")

    print(f"Merged model ready at: {output_path}")
    return output_path
