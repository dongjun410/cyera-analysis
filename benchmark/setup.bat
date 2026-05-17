@echo off
echo === Cyera FLAN-T5 Benchmark Setup ===

where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Conda not found. Please install Miniconda first.
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
) else (
    echo [WARN] No NVIDIA GPU detected. Will run on CPU.
)

set ENV_NAME=cyera-bench
conda env list | findstr /c:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] Conda environment '%ENV_NAME%' already exists.
) else (
    echo Creating conda environment '%ENV_NAME%' with Python 3.11...
    conda create -y -n %ENV_NAME% python=3.11
)

echo Installing PyTorch and dependencies...
conda run -n %ENV_NAME% pip install --upgrade pip
conda run -n %ENV_NAME% pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
conda run -n %ENV_NAME% pip install -e .

echo.
echo === Setup complete ===
echo Activate: conda activate %ENV_NAME%
echo Run:      python -m cyera_bench --config config/experiments/flan-t5-base-conll03.yaml
