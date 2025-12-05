#!/usr/bin/env bash
set -euo pipefail

# Switch to repository root directory
cd "$(dirname "$0")/.."

# If argument is provided, use the specified model directory; otherwise find the latest training output directory
if [ $# -ge 1 ]; then
    MODEL_DIR="$1"
else
    # Find the latest training output directory under outputs
    # Sort by modification time, take the latest
    LATEST_OUTPUT=$(find outputs -maxdepth 1 -type d -name "*epochs-*" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2-)
    
    # If the above method fails (some systems don't support -printf), use fallback method
    if [ -z "$LATEST_OUTPUT" ]; then
        LATEST_OUTPUT=$(ls -t outputs/*epochs-* 2>/dev/null | head -n 1)
    fi
    
    # If still fails, try simple find
    if [ -z "$LATEST_OUTPUT" ]; then
        LATEST_OUTPUT=$(find outputs -maxdepth 1 -type d -name "*epochs-*" 2>/dev/null | sort -r | head -n 1)
    fi
    
    if [ -z "$LATEST_OUTPUT" ]; then
        echo "Error: Training output directory not found under outputs"
        echo "Please ensure training is completed, or manually specify model directory as argument: bash scripts/eval.sh <model_dir>"
        exit 1
    fi
    
    MODEL_DIR="$LATEST_OUTPUT"
fi

echo "Using model directory: $MODEL_DIR"
echo "Starting evaluation..."

# Run evaluation script
python -m src.eval.eval_aesthetic \
  --model_dir "$MODEL_DIR" \
  --cuda_device 0 \
  --num_per_prompt 1 \
  --num_inference_steps 30 \
  --batch_size 32

echo "Evaluation complete! Results saved to: $MODEL_DIR/eval/"

