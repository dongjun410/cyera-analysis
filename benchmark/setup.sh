#!/bin/bash
set -e
echo "=== Cyera FLAN-T5 Benchmark Setup ==="

if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "[WARN] No NVIDIA GPU detected. Will run on CPU."
fi

ENV_NAME="cyera-bench"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[OK] Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -y -n ${ENV_NAME} python=3.11
fi

echo "Installing PyTorch and dependencies..."
conda run -n ${ENV_NAME} pip install --upgrade pip

if command -v nvidia-smi &> /dev/null; then
    conda run -n ${ENV_NAME} pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    conda run -n ${ENV_NAME} pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing benchmark dependencies..."
conda run -n ${ENV_NAME} pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate: conda activate ${ENV_NAME}"
echo "Run:      python -m cyera_bench --config config/experiments/flan-t5-base-conll03.yaml"
