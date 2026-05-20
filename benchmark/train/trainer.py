# benchmark/train/trainer.py
from __future__ import annotations

import os
from typing import Dict, List

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from benchmark.train.config import TrainingConfig
from benchmark.train.data_pipeline import (
    build_phase1_dataset,
    build_phase2_dataset,
    tokenize_dataset,
)


def _build_quantization_config(quantization: str | None) -> BitsAndBytesConfig | None:
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def _build_lora_config(cfg: TrainingConfig, dropout: float | None = None) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=dropout if dropout is not None else cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def setup_model_and_tokenizer(cfg: TrainingConfig):
    """Load base model with quantization, apply LoRA, return (peft_model, tokenizer)."""
    bnb_config = _build_quantization_config(cfg.quantization)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16 if cfg.quantization is None else None,
    )

    lora_config = _build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    return model, tokenizer


def train_phase1(cfg: TrainingConfig) -> PeftModel:
    """Phase 1: Train general document classification on open datasets."""
    print("=" * 60)
    print("PHASE 1: General Classification Training")
    print("=" * 60)

    model, tokenizer = setup_model_and_tokenizer(cfg)

    dataset = build_phase1_dataset(cfg)
    dataset = tokenize_dataset(
        dataset, tokenizer,
        max_length=cfg.phase1_max_length,
        max_target_length=cfg.phase1_max_target_length,
    )

    split = dataset.train_test_split(
        test_size=cfg.phase1_val_split, seed=cfg.seed,
    )
    train_ds = split["train"]
    val_ds = split["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
    )

    # Compute warmup steps from ratio (warmup_ratio deprecated in transformers 5.x)
    effective_batch = cfg.phase1_batch_size * cfg.phase1_grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    warmup_steps = int(steps_per_epoch * cfg.phase1_epochs * cfg.phase1_warmup_ratio)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "phase1_checkpoints"),
        per_device_train_batch_size=cfg.phase1_batch_size,
        per_device_eval_batch_size=cfg.phase1_batch_size,
        gradient_accumulation_steps=cfg.phase1_grad_accum,
        num_train_epochs=cfg.phase1_epochs,
        learning_rate=cfg.phase1_lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=cfg.phase1_weight_decay,
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.quantization is None,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_path = os.path.join(cfg.output_dir, "phase1_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Phase 1 adapter saved to {adapter_path}")

    return model


def train_phase2(
    cfg: TrainingConfig,
    dspm_texts: List[str],
    dspm_labels: List[Dict[str, str]],
    l1_options: List[str],
) -> PeftModel:
    """Phase 2: DSPM domain adaptation on augmented datasets."""
    print("=" * 60)
    print("PHASE 2: DSPM Domain Adaptation")
    print("=" * 60)

    bnb_config = _build_quantization_config(cfg.quantization)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16 if cfg.quantization is None else None,
    )

    phase1_path = os.path.join(cfg.output_dir, "phase1_adapter")
    if os.path.exists(phase1_path):
        model = PeftModel.from_pretrained(base_model, phase1_path, is_trainable=True)
        print(f"Loaded Phase 1 adapter from {phase1_path}")
    else:
        lora_config = _build_lora_config(cfg, dropout=cfg.phase2_lora_dropout)
        model = get_peft_model(base_model, lora_config)
        print("No Phase 1 adapter found, training from scratch with higher dropout")

    for module in model.modules():
        if hasattr(module, "dropout") and hasattr(module.dropout, "p"):
            if module.dropout.p != cfg.phase2_lora_dropout:
                module.dropout.p = cfg.phase2_lora_dropout

    dataset = build_phase2_dataset(dspm_texts, dspm_labels, l1_options, cfg)
    dataset = tokenize_dataset(
        dataset, tokenizer,
        max_length=cfg.phase2_max_length,
        max_target_length=cfg.phase2_max_target_length,
    )

    split = dataset.train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds = split["train"]
    val_ds = split["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True,
    )

    # Compute warmup steps from ratio
    effective_batch = cfg.phase2_batch_size * cfg.phase2_grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    warmup_steps = int(steps_per_epoch * cfg.phase2_epochs * cfg.phase2_warmup_ratio)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "phase2_checkpoints"),
        per_device_train_batch_size=cfg.phase2_batch_size,
        per_device_eval_batch_size=cfg.phase2_batch_size,
        gradient_accumulation_steps=cfg.phase2_grad_accum,
        num_train_epochs=cfg.phase2_epochs,
        learning_rate=cfg.phase2_lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=cfg.phase2_weight_decay,
        optim="adamw_torch",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.quantization is None,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.phase2_early_stopping_patience,
        )],
    )

    trainer.train()

    adapter_path = os.path.join(cfg.output_dir, "phase2_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Phase 2 adapter saved to {adapter_path}")

    return model
