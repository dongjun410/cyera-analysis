#!/usr/bin/env bash
# Full performance benchmark: all models × datasets × devices
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

PY="python"
SCRIPT="benchmark/test_performance.py"

echo "===== PHASE 1: Gemma GPU ====="
$PY $SCRIPT --model gemma-doc-label --dataset ben25 --device cuda
$PY $SCRIPT --model gemma-doc-label --dataset dspm27 --device cuda
$PY $SCRIPT --model gemma-doc-label --dataset cxh5types --device cuda

echo "===== PHASE 2: Gemma CPU ====="
echo "  (Restart gemma service with NUM_GPU=0 first, then press Enter)"
read -p "  Ready? "

$PY $SCRIPT --model gemma-doc-label --dataset ben25 --device cpu
$PY $SCRIPT --model gemma-doc-label --dataset dspm27 --device cpu
$PY $SCRIPT --model gemma-doc-label --dataset cxh5types --device cpu

echo "===== PHASE 3: Sklearn CPU ====="
$PY $SCRIPT --model doc-classifier-sklearn --dataset ben25 --device cpu
$PY $SCRIPT --model doc-classifier-sklearn --dataset dspm27 --device cpu
$PY $SCRIPT --model doc-classifier-sklearn --dataset cxh5types --device cpu

echo "===== PHASE 4: FlanT5 base GPU ====="
$PY $SCRIPT --model flan-t5-classification --variant base --dataset ben25 --device cuda
$PY $SCRIPT --model flan-t5-classification --variant base --dataset dspm27 --device cuda
$PY $SCRIPT --model flan-t5-classification --variant base --dataset cxh5types --device cuda

echo "===== PHASE 5: FlanT5 base CPU ====="
$PY $SCRIPT --model flan-t5-classification --variant base --dataset ben25 --device cpu
$PY $SCRIPT --model flan-t5-classification --variant base --dataset dspm27 --device cpu
$PY $SCRIPT --model flan-t5-classification --variant base --dataset cxh5types --device cpu

echo "===== PHASE 6: FlanT5 large GPU ====="
$PY $SCRIPT --model flan-t5-classification --variant large --dataset ben25 --device cuda
$PY $SCRIPT --model flan-t5-classification --variant large --dataset dspm27 --device cuda
$PY $SCRIPT --model flan-t5-classification --variant large --dataset cxh5types --device cuda

echo "===== PHASE 7: FlanT5 large CPU ====="
$PY $SCRIPT --model flan-t5-classification --variant large --dataset ben25 --device cpu
$PY $SCRIPT --model flan-t5-classification --variant large --dataset dspm27 --device cpu
$PY $SCRIPT --model flan-t5-classification --variant large --dataset cxh5types --device cpu

echo "===== ALL DONE ====="
