#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL_DIR="outputs/quick_validation_higher_utility"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory does not exist: $MODEL_DIR"
    echo "Please ensure the model has been downloaded and extracted to this directory"
    echo "Or use your own model directory as argument: bash scripts/quick_validation.sh <your_model_dir>"
    exit 1
fi

# Check if model weight file exists
MODEL_WEIGHT="${MODEL_DIR}/final_model.pt"
if [ ! -f "$MODEL_WEIGHT" ]; then
    echo "Error: Model weight file not found: $MODEL_WEIGHT"
    echo "Please ensure the model directory contains resalign_pretrained_model_sdv1.final_model.pt file"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_DIR")

echo "=========================================="
echo "ResAlign Quick Validation"
echo "=========================================="
echo ""
echo "Using model directory: $MODEL_DIR"
echo "Model name: $MODEL_NAME"
echo ""
echo "This script will run the following in sequence:"
echo "1. Safety evaluation"
echo "2. Aesthetic evaluation"
echo ""

# ========== Safety Evaluation ==========
echo "=========================================="
echo "Step 1/2: Safety Evaluation"
echo "=========================================="
echo ""

bash scripts/eval_safety.sh "$MODEL_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "Warning: Safety evaluation failed, but will continue with aesthetic evaluation"
fi

echo ""
echo "=========================================="
echo "Safety Evaluation Complete"
echo "=========================================="
echo ""

# ========== Aesthetic Evaluation ==========
echo "=========================================="
echo "Step 2/2: Aesthetic Evaluation"
echo "=========================================="
echo ""

bash scripts/eval.sh "$MODEL_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "Warning: Aesthetic evaluation failed"
fi

echo ""
echo "=========================================="
echo "Quick Validation Complete!"
echo "=========================================="
echo ""
echo "Evaluation Results Summary:"
echo ""
echo "1. Safety Evaluation Results:"
echo "   - Metrics summary CSV: outputs/${MODEL_NAME}/safety_eval/safety_metrics_summary.csv"
echo "   - Detailed results directory: outputs/${MODEL_NAME}/safety_eval/"
echo ""
echo "2. Aesthetic Evaluation Results:"
echo "   - Result file: outputs/${MODEL_NAME}/eval/result.txt"
echo "   - Generated images: outputs/${MODEL_NAME}/eval/gen_images/"
echo ""
echo "All evaluation results have been saved to outputs/${MODEL_NAME}/ directory"
