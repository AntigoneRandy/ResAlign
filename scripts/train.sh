#!/usr/bin/env bash
set -euo pipefail

# Switch to repository root directory (script is located under scripts/)
cd "$(dirname "$0")/.."

python -m src.train.train \
  --inner_configs \
    configs/config-1.json \
    configs/config0.json \
    configs/config1.json \
    configs/config2.json \
    configs/config3.json \
    configs/config4.json \
    configs/config5.json \
    configs/config6.json \
    configs/config7.json \
    configs/config8.json \
    configs/config9.json \
    configs/config10.json \
    configs/config11.json \
    configs/config12.json \
    configs/config_full.json \
  --model_path CompVis/stable-diffusion-v1-4 \
  --cuda_device 2 \
  --erase_concept "nude people" \
  --train_method full \
  --num_epochs 160 \
  --learning_rate 2e-4 \
  --dynamic_weights \
  --final_outer_lambda 0.8 \
  --outer_lambda 0.8 \
  --final_retain_loss_weight 2 \
  --retain_loss_weight 1.6 \
  --meta_rate 0.3 \
  --retain_data_dir data/retain \
  --outer_data_dir data/train \
  --loss_mode ga \
  --inference_interval 10


