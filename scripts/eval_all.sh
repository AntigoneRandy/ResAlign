#!/usr/bin/env bash
set -euo pipefail

# Switch to repository root directory
cd "$(dirname "$0")/.."

echo "=========================================="
echo "ResAlign All-in-One Evaluation Script"
echo "=========================================="
echo ""
echo "This script will run the following in sequence:"
echo "1. Safety evaluation"
echo "2. Aesthetic evaluation"
echo ""

# Find the latest training output directory under outputs
LATEST_OUTPUT=$(find outputs -maxdepth 1 -type d -name "*epochs-*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2-)

# If the above method fails, use fallback method
if [ -z "$LATEST_OUTPUT" ]; then
    LATEST_OUTPUT=$(ls -td outputs/*epochs-* 2>/dev/null | head -n 1)
fi

# If still fails, try simple find
if [ -z "$LATEST_OUTPUT" ]; then
    LATEST_OUTPUT=$(find outputs -maxdepth 1 -type d -name "*epochs-*" 2>/dev/null | sort -r | head -n 1)
fi

if [ -z "$LATEST_OUTPUT" ]; then
    echo "Error: Training output directory not found under outputs"
    echo "Please ensure training is completed"
    exit 1
fi

MODEL_DIR="$LATEST_OUTPUT"
MODEL_NAME=$(basename "$MODEL_DIR")

echo "Found latest training output directory: $MODEL_DIR"
echo "Model name: $MODEL_NAME"
echo ""

# ========== Safety Evaluation ==========
echo "=========================================="
echo "Step 1/2: Safety Evaluation"
echo "=========================================="
echo ""

bash scripts/eval_safety.sh

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

bash scripts/eval.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "Warning: Aesthetic evaluation failed"
fi

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

