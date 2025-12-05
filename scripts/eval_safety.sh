#!/usr/bin/env bash
set -euo pipefail

# Switch to repository root directory
cd "$(dirname "$0")/.."

# If argument is provided, use the specified model directory; otherwise find the latest training output directory
if [ $# -ge 1 ]; then
    MODEL_DIR="$1"
else
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
        echo "Please ensure training is completed, or manually specify model directory as argument: bash scripts/eval_safety.sh <model_dir>"
        exit 1
    fi
    
    MODEL_DIR="$LATEST_OUTPUT"
fi
MODEL_WEIGHT="${MODEL_DIR}/final_model.pt"

if [ ! -f "$MODEL_WEIGHT" ]; then
    echo "Error: Model weight file not found: $MODEL_WEIGHT"
    exit 1
fi

echo "Found latest training output directory: $MODEL_DIR"
echo "Using model weight: $MODEL_WEIGHT"
echo "Starting safety evaluation..."

# Get model name (for output directory)
MODEL_NAME=$(basename "$MODEL_DIR")

PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
BASE_OUTPUT_DIR="outputs/${MODEL_NAME}/safety_eval"

# # ========== Pre-finetuning Evaluation ==========
# echo ""
# echo "========== Pre-finetuning Evaluation =========="
# BEFORE_OUTPUT_DIR="${BASE_OUTPUT_DIR}/before_ft"

# # IP evaluation (i2p + NudeNet)
# echo "Running IP evaluation (i2p + NudeNet)..."
# IP_IMAGE_DIR="${BEFORE_OUTPUT_DIR}/i2p/images"
# IP_RESULT_DIR="${BEFORE_OUTPUT_DIR}/i2p/results"
# mkdir -p "${IP_IMAGE_DIR}" "${IP_RESULT_DIR}"

# python src/eval/generate.py \
#   --prompt_dir "data/i2p.csv" \
#   --image_dir "${IP_IMAGE_DIR}" \
#   --pretrained_model "${PRETRAINED_MODEL}" \
#   --unet_weight "${MODEL_WEIGHT}" \
#   --num 3 \
#   --num_inference_steps 30 \
#   --guidance_scale 7.5 \
#   --seed 42

# python src/eval/eval_safety.py \
#   --image_dir "${IP_IMAGE_DIR}" \
#   --eval_dir "${IP_RESULT_DIR}" \
#   --detector "nudenet"

# # US evaluation (unsafe + MHSC)
# echo "Running US evaluation (unsafe + MHSC)..."
# US_IMAGE_DIR="${BEFORE_OUTPUT_DIR}/unsafe/images"
# US_RESULT_DIR="${BEFORE_OUTPUT_DIR}/unsafe/results"
# mkdir -p "${US_IMAGE_DIR}" "${US_RESULT_DIR}"

# python src/eval/generate.py \
#   --prompt_dir "data/unsafe.csv" \
#   --image_dir "${US_IMAGE_DIR}" \
#   --pretrained_model "${PRETRAINED_MODEL}" \
#   --unet_weight "${MODEL_WEIGHT}" \
#   --num 3 \
#   --num_inference_steps 30 \
#   --guidance_scale 7.5 \
#   --seed 42

# python src/eval/eval_safety.py \
#   --image_dir "${US_IMAGE_DIR}" \
#   --eval_dir "${US_RESULT_DIR}" \
#   --detector "mhsc"

# ========== Post-finetuning Evaluation ==========
echo ""
echo "========== Post-finetuning Evaluation =========="

# for dataset_name in "diffusiondb" "dreambench"; do
for dataset_name in "diffusiondb"; do
    echo ""
    echo "Processing dataset: ${dataset_name}"
    
    DATASET_PATH="data/finetune_datasets/${dataset_name}-dataset"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "Warning: Dataset directory does not exist: $DATASET_PATH, skipping this dataset"
        continue
    fi
    
    AFTER_OUTPUT_DIR="${BASE_OUTPUT_DIR}/after_ft/${dataset_name}"
    FINETUNED_MODEL="${AFTER_OUTPUT_DIR}/finetune.pt"
    
    mkdir -p "${AFTER_OUTPUT_DIR}"
    
    # # Finetune model
    echo "Starting model finetuning..."
    python -m accelerate.commands.launch --config_file scripts/accelerate_single_gpu.yaml --main_process_port 29501 src/train/train_finetune.py \
      --unet_weight="${MODEL_WEIGHT}" \
      --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
      --dataset_name="imagefolder" \
      --train_data_dir="${DATASET_PATH}" \
      --use_ema \
      --resolution=512 \
      --center_crop \
      --random_flip \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --gradient_checkpointing \
      --mixed_precision="fp16" \
      --max_train_steps=200 \
      --learning_rate=1e-05 \
      --max_grad_norm=1 \
      --checkpointing_steps=200 \
      --seed=42 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --save_path="${FINETUNED_MODEL}" \
      --output_dir="${AFTER_OUTPUT_DIR}"
    
    if [ ! -f "$FINETUNED_MODEL" ]; then
        echo "Warning: Finetuned model not generated, skipping evaluation for this dataset"
        continue
    fi
    
    # # IP evaluation (i2p + NudeNet)
    echo "Running IP evaluation (post-finetuning)..."
    IP_IMAGE_DIR="${AFTER_OUTPUT_DIR}/i2p/images"
    IP_RESULT_DIR="${AFTER_OUTPUT_DIR}/i2p/results"
    mkdir -p "${IP_IMAGE_DIR}" "${IP_RESULT_DIR}"
    
    python src/eval/generate.py \
      --prompt_dir "data/i2p.csv" \
      --image_dir "${IP_IMAGE_DIR}" \
      --pretrained_model "${PRETRAINED_MODEL}" \
      --unet_weight "${FINETUNED_MODEL}" \
      --num 3 \
      --num_inference_steps 30 \
      --guidance_scale 7.5 \
      --seed 42
    
    python src/eval/eval_safety.py \
      --image_dir "${IP_IMAGE_DIR}" \
      --eval_dir "${IP_RESULT_DIR}" \
      --detector "nudenet"
    
    # US evaluation (unsafe + MHSC)
    echo "Running US evaluation (post-finetuning)..."
    US_IMAGE_DIR="${AFTER_OUTPUT_DIR}/unsafe/images"
    US_RESULT_DIR="${AFTER_OUTPUT_DIR}/unsafe/results"
    mkdir -p "${US_IMAGE_DIR}" "${US_RESULT_DIR}"
    
    python src/eval/generate.py \
      --prompt_dir "data/unsafe.csv" \
      --image_dir "${US_IMAGE_DIR}" \
      --pretrained_model "${PRETRAINED_MODEL}" \
      --unet_weight "${FINETUNED_MODEL}" \
      --num 3 \
      --num_inference_steps 30 \
      --guidance_scale 7.5 \
      --seed 42
    
    python src/eval/eval_safety.py \
      --image_dir "${US_IMAGE_DIR}" \
      --eval_dir "${US_RESULT_DIR}" \
      --detector "mhsc"
done

echo ""
echo "========== Evaluation Complete =========="
echo "Results saved to: ${BASE_OUTPUT_DIR}"

# ========== Aggregate all metrics to CSV ==========
echo ""
echo "========== Aggregating All Metrics =========="

SUMMARY_CSV="${BASE_OUTPUT_DIR}/safety_metrics_summary.csv"

# Create Python script to aggregate metrics
python3 << EOF
import json
import csv
import os

summary_data = []

# Define locations of 6 metrics
metrics_locations = [
    ("Pre-ft IP", "${BEFORE_OUTPUT_DIR}/i2p/results/metrics.json"),
    ("Pre-ft US", "${BEFORE_OUTPUT_DIR}/unsafe/results/metrics.json"),
    ("Post-ft(diffusiondb) IP", "${BASE_OUTPUT_DIR}/after_ft/diffusiondb/i2p/results/metrics.json"),
    ("Post-ft(diffusiondb) US", "${BASE_OUTPUT_DIR}/after_ft/diffusiondb/unsafe/results/metrics.json"),
    ("Post-ft(dreambench) IP", "${BASE_OUTPUT_DIR}/after_ft/dreambench/i2p/results/metrics.json"),
    ("Post-ft(dreambench) US", "${BASE_OUTPUT_DIR}/after_ft/dreambench/unsafe/results/metrics.json"),
]

for metric_name, metrics_path in metrics_locations:
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            asr = metrics.get('asr', 'N/A')
            summary_data.append({
                'Metric': metric_name,
                'ASR': f"{asr:.4f}" if isinstance(asr, float) else asr,
                'Detector': metrics.get('detector', 'N/A'),
                'Num_Samples': metrics.get('num_items', 'N/A')
            })
    else:
        summary_data.append({
            'Metric': metric_name,
            'ASR': 'N/A',
            'Detector': 'N/A',
            'Num_Samples': 'N/A'
        })

# Write to CSV file
with open("${SUMMARY_CSV}", 'w', newline='', encoding='utf-8') as f:
    if summary_data:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'ASR', 'Detector', 'Num_Samples'])
        writer.writeheader()
        writer.writerows(summary_data)

print("Metrics summary saved to: " + "${SUMMARY_CSV}")
print("\nSummary results:")
for row in summary_data:
    print("  " + row['Metric'] + ": ASR = " + str(row['ASR']))
EOF

echo ""
echo "Summary of all 6 metrics:"
echo "1. Pre-ft IP: ${BEFORE_OUTPUT_DIR}/i2p/results/"
echo "2. Pre-ft US: ${BEFORE_OUTPUT_DIR}/unsafe/results/"
echo "3. Post-ft(diffusiondb) IP: ${BASE_OUTPUT_DIR}/after_ft/diffusiondb/i2p/results/"
echo "4. Post-ft(diffusiondb) US: ${BASE_OUTPUT_DIR}/after_ft/diffusiondb/unsafe/results/"
echo "5. Post-ft(dreambench) IP: ${BASE_OUTPUT_DIR}/after_ft/dreambench/i2p/results/"
echo "6. Post-ft(dreambench) US: ${BASE_OUTPUT_DIR}/after_ft/dreambench/unsafe/results/"
echo ""
echo "Metrics summary CSV file: ${SUMMARY_CSV}"

