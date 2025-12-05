#!/usr/bin/env bash
set -euo pipefail

# Switch to repository root directory
cd "$(dirname "$0")/.."

echo "=========================================="
echo "ResAlign All-in-One Evaluation (Custom Model Dir)"
echo "=========================================="
echo ""

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/eval_all_custom.sh <model_dir>"
    echo "Example: bash scripts/eval_all_custom.sh /absolute/path/to/your_model_dir"
    exit 1
fi

MODEL_DIR="$1"
MODEL_NAME=$(basename "$MODEL_DIR")
MODEL_WEIGHT="${MODEL_DIR}/final_model.pt"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model_dir does not exist: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$MODEL_WEIGHT" ]; then
    echo "Error: expected weight file not found: $MODEL_WEIGHT"
    echo "Please ensure your directory contains 'final_model.pt'"
    exit 1
fi

echo "Using custom model directory: $MODEL_DIR"
echo "Model name: $MODEL_NAME"
echo ""

# ========== Safety Evaluation ==========
echo "=========================================="
echo "Step 1/2: Safety Evaluation"
echo "=========================================="
echo ""

bash scripts/eval_safety.sh "$MODEL_DIR" || {
  echo ""
  echo "Warning: Safety evaluation failed, continuing with aesthetic evaluation"
}

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

bash scripts/eval.sh "$MODEL_DIR" || {
  echo ""
  echo "Warning: Aesthetic evaluation failed"
}

echo ""
echo "=========================================="
echo "All-in-One Evaluation Complete!"
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


