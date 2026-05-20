"""CLI entry point for FLAN-T5 QLoRA fine-tuning.

Usage:
    python -m benchmark.scripts.run_finetune
    python -m benchmark.scripts.run_finetune --config path/to/config.yaml
    python -m benchmark.scripts.run_finetune --phase1-only
    python -m benchmark.scripts.run_finetune --phase2-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from benchmark.train.config import TrainingConfig
from benchmark.train.trainer import train_phase1, train_phase2
from benchmark.train.merge_adapter import merge_and_save
from benchmark.train.augment import Augmenter


def _load_dspm_data():
    """Load DSPM datasets from benchmark and prepare for Phase 2."""
    from benchmark.src.cyera_bench.datasets.dspm27 import Dspm27Dataset
    from benchmark.src.cyera_bench.datasets.ben25 import Ben25Dataset
    from benchmark.src.cyera_bench.datasets.cxh5types import Cxh5typesDataset

    datasets = {
        "dspm27": Dspm27Dataset(),
        "ben25": Ben25Dataset(),
        "cxh5types": Cxh5typesDataset(),
    }

    all_texts = []
    all_labels = []
    all_l1_set = set()

    for name, ds in datasets.items():
        texts, labels = ds.load()
        all_texts.extend(texts)
        all_labels.extend(labels)
        for l in labels:
            all_l1_set.add(l["l1"])

    return all_texts, all_labels, sorted(all_l1_set)


def main():
    parser = argparse.ArgumentParser(
        description="FLAN-T5 QLoRA Fine-Tuning for Document Classification"
    )
    parser.add_argument(
        "--config", type=str,
        default="benchmark/config/experiments/flan-t5-finetune.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--phase1-only", action="store_true",
        help="Run only Phase 1 (general classification)",
    )
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="Run only Phase 2 (DSPM adaptation). Requires Phase 1 adapter.",
    )
    parser.add_argument(
        "--skip-augmentation", action="store_true",
        help="Skip DSPM data augmentation (use raw data only)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip final adapter merge step",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = TrainingConfig.from_yaml(str(config_path))
    print(f"Loaded config from {config_path}")
    print(f"  Model: {cfg.model_name}")
    print(f"  Quantization: {cfg.quantization}")
    print(f"  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    print(f"  Output: {cfg.output_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    if not args.phase2_only:
        model = train_phase1(cfg)
        print("Phase 1 complete.")

    if not args.phase1_only:
        print("\nLoading DSPM data...")
        dspm_texts, dspm_labels, l1_options = _load_dspm_data()
        print(f"  Loaded {len(dspm_texts)} raw DSPM documents")

        if not args.skip_augmentation:
            print("Augmenting DSPM data...")
            augmenter = Augmenter(cfg)
            dspm_texts, dspm_labels = augmenter.augment(
                dspm_texts, dspm_labels,
            )
            print(f"  After augmentation: {len(dspm_texts)} documents")

        model = train_phase2(cfg, dspm_texts, dspm_labels, l1_options)
        print("Phase 2 complete.")

    if not args.skip_merge:
        output_path = merge_and_save(cfg)
        print(f"\nFine-tuned model ready: {output_path}")
        print(
            "To use in benchmark, set finetuned_path in your experiment config "
            "or pass to FlanT5ClassificationModel(finetuned_path=...)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
