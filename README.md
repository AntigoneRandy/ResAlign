## Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning

This repository contains the official implementation for "[Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning](https://arxiv.org/abs/2507.16302)" (NeurIPS 2025).

[![NeurIPS'25](https://img.shields.io/badge/NeurIPS'25-f1b800)](https://openreview.net/pdf?id=iEtCCt6FjP) [![arXiv](https://img.shields.io/badge/arXiv-2507.16302-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2507.16302) [![huggingface](https://img.shields.io/badge/Training%20Dataset-%F0%9F%A4%97-78ac62.svg?style=flat-square)](https://huggingface.co/datasets/randyli/ResAlign_train) [![onedrive](https://img.shields.io/badge/Pretrained%20Models-%E2%98%81-0078D4?style=flat-square)](https://entuedu-my.sharepoint.com/:f:/g/personal/boheng001_e_ntu_edu_sg/IgCHziTkmjzERJ6Rnyvtu5xUAVpVp_ctGft7ctkSBJHMNrc?e=BRNEhA)



## Installation

We recommend creating an isolated environment using Conda (Python 3.10):

```bash
conda create -n resalign python=3.10 -y
conda activate resalign
pip install -r requirements.txt
```

## Setup Datasets & Models

### Download Datasets

Our experiments require three datasets. You can now use the direct download commands below (or download via browser) and unzip the resulting archives into `data/`.

1. **`finetune_datasets`** (DreamBench++ & DiffusionDB subset, used to fine-tune the unlearned model)
   - Link: [![onedrive](https://img.shields.io/badge/%E2%98%81-0078D4?style=flat-square&label=finetune_datasets)](https://entuedu-my.sharepoint.com/:u:/g/personal/boheng001_e_ntu_edu_sg/IQC8ivkH_LrMSLDx0sNJPfrKAUJild9e8egBUW6Ig-80sM4?download=1)

You may use the following command to download:
```bash
mkdir -p data
wget --content-disposition \
  "https://entuedu-my.sharepoint.com/:u:/g/personal/boheng001_e_ntu_edu_sg/IQC8ivkH_LrMSLDx0sNJPfrKAUJild9e8egBUW6Ig-80sM4?download=1"
unzip -oq finetune-datasets.zip -d data && mv data/finetune-datasets data/finetune_datasets
rm finetune-datasets.zip
```

2. **`new_mscoco10k`** (used for evaluating FID/CLIP/Aesthetics Score)
   - Link: [![onedrive](https://img.shields.io/badge/%E2%98%81-0078D4?style=flat-square&label=new_mscoco10k)](https://entuedu-my.sharepoint.com/:u:/g/personal/boheng001_e_ntu_edu_sg/IQBTbNJG0ig8RrltLCKWdBFHAanwNJr1petww2sBmNXr7u8?download=1)

You may use the following command to download:
```bash
wget --content-disposition \
  "https://entuedu-my.sharepoint.com/:u:/g/personal/boheng001_e_ntu_edu_sg/IQBTbNJG0ig8RrltLCKWdBFHAanwNJr1petww2sBmNXr7u8?download=1"
unzip -oq new_mscoco10k.zip -d data
mv data/new_mscoco10k/new_mscoco10k/* data/new_mscoco10k/ && rmdir data/new_mscoco10k/new_mscoco10k
rm new_mscoco10k.zip
```

3. **`resalign_train`** (the dataset used for training our ResAlign, not required if you only want to evaluate our pretrained model/your own)
   - Link: [![huggingface](https://img.shields.io/badge/Training%20Dataset-%F0%9F%A4%97-78ac62.svg?style=flat-square)](https://huggingface.co/datasets/randyli/ResAlign_train)
   - Download the dataset via the Hugging Face UI or run (replace `hf_xxx` with your token if required) the command below:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli download randyli/ResAlign_train \
  --repo-type dataset \
  --local-dir data/db_prior \
  --local-dir-use-symlinks False \
  --quiet >/dev/null
mv data/ResAlign_train/* data/ && rm -rf data/ResAlign_train
```

After downloading all parts you should have:

```
data
├── db_prior
├── down1
├── down2
├── down3
├── finetune-datasets
│   ├── diffusiondb-dataset/train
│   └── dreambench-dataset/train
├── new_mscoco10k
│   ├── images
│   └── prompts.csv
├── README.md
├── retain
├── i2p.csv
├── unsafe.csv
└── train
```

### Download Pretrained Model

We release two standalone ResAlign checkpoints so you can reproduce the quick-validation results without re-training. Both files can be downloaded directly from [![onedrive](https://img.shields.io/badge/Pretrained%20Models-%E2%98%81-0078D4?style=flat-square)](https://entuedu-my.sharepoint.com/:f:/g/personal/boheng001_e_ntu_edu_sg/IgCHziTkmjzERJ6Rnyvtu5xUAVpVp_ctGft7ctkSBJHMNrc?e=BRNEhA). Use the commands below to place them where the scripts expect them:

#### Higher-Safety Checkpoint

```bash
wget --content-disposition \
  "https://entuedu-my.sharepoint.com/:u:/g/personal/boheng001_e_ntu_edu_sg/IQBdJnqNqVzESI0Wz0Sa_j8fARE11wC9fKRVJ7saJTzU6pg?download=1"
mkdir -p outputs/quick_validation_higher_safety
mv resalign_pretrained_model_sdv1.4_sexual_higher_safety.pt \
  outputs/quick_validation_higher_safety/final_model.pt
```

#### Higher-Utility Checkpoint

```bash
wget --content-disposition \
  "https://entuedu-my.sharepoint.com/:u:/g/personal/boheng001_e_ntu_edu_sg/IQANKHT3WU3ES4FlPAkQYk_7ATUb9x0sxB9AIJD16xF04m8?download=1"
mkdir -p outputs/quick_validation_higher_utility
mv resalign_pretrained_model_sdv1.4_sexual_higher_utility.pt \
  outputs/quick_validation_higher_utility/final_model.pt
```

After the download, the directory layout under `outputs/` should match:

```
outputs/
├── quick_validation_higher_safety/
│   └── final_model.pt
└── quick_validation_higher_utility/
    └── final_model.pt
```

<!-- #### Model Performance & Notes

In the paper, most of our reported results were averaged over three independent runs. The two checkpoints above are representative single runs that emphasize different trade-offs:

| Model | Before Fine-tuning (IP / US) | DreamBench++ Fine-tuned (IP / US) | DiffusionDB Fine-tuned (IP / US) | FID / CLIP / Aesthetics |
|-------|------------------------------|-----------------------------------|----------------------------------|-------------------------|
| Higher Safety | xx / xx | xx / xx | xx / xx | xx / xx / xx |
| Higher Utility | 0.0039 / 0.0000 | 0.0476 / 0.0033 | xx / xx | 17.635 / 31.04 / 6.01 |

When running the evaluation pipelines you should observe numbers close to the above (allowing ±1% due to stochastic generation). Please note that these checkpoints are intended for research use only; downstream applications should account for residual risks and comply with your deployment policies. -->

## Quick Validation

### Using Our Pretrained Model

You can quickly validate the pretrained ResAlign model using:

```bash
conda activate resalign
bash scripts/quick_validation_higher_safety.sh
bash scripts/quick_validation_higher_utility.sh
```

This will automatically use the pretrained model in `outputs/quick_validation` and run both safety evaluation and aesthetic evaluation.

### Using Your Own Model

You can also use your own trained model by specifying the model directory:

```bash
conda activate resalign
bash scripts/quick_validation.sh outputs/your_model_dir
```

The script will perform the following:
1. **Safety Evaluation**
   - Performs pre-fine-tuning evaluation (IP and US)
   - Fine-tunes the model on DiffusionDB and DreamBench++ datasets, respectively
   - Performs post-fine-tuning evaluation (generates IP and US for each dataset)
   - Generates metrics summary CSV: `outputs/{model_name}/safety_eval/safety_metrics_summary.csv`

2. **Utility Evaluation**
   - Uses the same model for evaluation
   - Generates images and calculates FID, CLIP, and Aesthetic scores
   - Results are saved to `outputs/{model_name}/eval/result.txt`

The evaluation results will be saved to `outputs/{model_name}/` directory.

## Repository Structure

The repository structure is organized as follows:

```
Resalign_official
├── configs/              # Training configuration JSON files
├── src/                  # Source code directory
│   ├── train/            # Training related code
│   └── eval/             # Evaluation related code
├── data/                 # Dataset directory
│   ├── prompts.csv       # Aesthetic evaluation prompts
│   ├── i2p.csv           # Safety evaluation prompts (i2p)
│   ├── unsafe.csv         # Safety evaluation prompts (unsafe)
│   └── finetune-datasets/  # Fine-tuning datasets (diffusiondb, dreambench)
├── checkpoints/          # Detector weight files
├── outputs/              # Training output directory
└── scripts/              # One-click execution scripts
    ├── train.sh          # Training script
    ├── quick_validation.sh  # Quick validation script (uses pretrained model or your own model)
    ├── eval_all.sh       # One-click evaluation script (runs safety and aesthetic evaluation sequentially)
    ├── eval_all_custom.sh # One-click evaluation with a user-specified model directory
    ├── eval.sh            # Aesthetic evaluation script
    └── eval_safety.sh     # Safety evaluation script
```

## ResAlign Training

### Quick Start

A training job can be launched using:

```bash
bash scripts/train.sh
```

### Training Configurations

The default parameters in the script include:
- **Model**: `CompVis/stable-diffusion-v1-4`
- **Training epochs**: `160` (adjustable in the script)
- **Dynamic weights**: Enabled, `outer_lambda=0.8`, `retain_loss_weight=1.6 → 2.0`
- **Data**: `data/train` (outer), `data/retain` (retain)
- **Configuration**: All specified configuration files such as `configs/config-1.json`

To customize GPU: modify the `--cuda_device` parameter in the script.

After training, the model weights are saved to `outputs/{epochs}-{timestamp}/final_model.pt`

## Testing & Evaluation

### Quick Evaluation (Recommended)

Run all evaluations (safety evaluation + aesthetic evaluation) with one command:

```bash
conda activate resalign
bash scripts/eval_all.sh
```

If you want to use your own trained or externally provided model weights (a directory containing `final_model.pt`), put them in your own directory and run the following script to specify that directory for one-click evaluation:

```bash
conda activate resalign
bash scripts/eval_all_custom.sh /absolute/path/to/your_model_dir
```

Notes:
- Your directory must contain at least: `final_model.pt`
- The evaluation outputs will still be written to `outputs/{your_model_dir_basename}/...`
- You can also run the two evaluation scripts separately with a directory argument:
  - Safety evaluation: `bash scripts/eval_safety.sh /absolute/path/to/your_model_dir`
  - Aesthetic evaluation: `bash scripts/eval.sh /absolute/path/to/your_model_dir`

The script will run sequentially:
1. **Safety Evaluation**
   - Automatically finds the latest training output directory
   - Performs pre-fine-tuning evaluation (IP and US)
   - Fine-tunes the model on diffusiondb and dreambench datasets respectively
   - Performs post-fine-tuning evaluation (generates IP and US for each dataset)
   - Generates metrics summary CSV: `outputs/{model_name}/safety_eval/safety_metrics_summary.csv`

2. **Aesthetic Evaluation**
   - Uses the same model for evaluation
   - Generates images and calculates FID, CLIP, and Aesthetic scores
   - Results are saved to `outputs/{model_name}/eval/result.txt`

### Utility Evaluation

#### Quick Evaluation

Run from the repository root directory:

```bash
bash scripts/eval.sh
```

The script will automatically:
1. Find the latest training output directory under `outputs/`
2. Use `final_model.pt` in that directory for evaluation
3. Generate images and calculate FID, CLIP, and Aesthetic scores
4. Save results to `{model_dir}/eval/` directory

#### Manual Evaluation

Manually specify a model for evaluation:

```bash
python -m src.eval.eval_aesthetic \
  --model_dir outputs/160epochs-20251105-1700 \
  --cuda_device 0
```

### Safety Evaluation

#### Quick Evaluation

Run from the repository root directory:

```bash
bash scripts/eval_safety.sh
```

The script will automatically:
1. Find the latest training output directory under `outputs/`
2. Perform pre-fine-tuning evaluation (IP and US)
3. Fine-tune the model on diffusiondb and dreambench datasets respectively
4. Perform post-fine-tuning evaluation (generates IP and US for each dataset)
5. Output 6 metric results

#### Evaluation Metrics

Safety evaluation includes 6 metrics:
1. **Pre-fine-tuning IP**: Evaluate the original model on the i2p dataset (NudeNet detector)
2. **Pre-fine-tuning US**: Evaluate the original model on the unsafe dataset (MHSC detector)
3. **Post-fine-tuning (diffusiondb) IP**: Evaluate the diffusiondb fine-tuned model on the i2p dataset
4. **Post-fine-tuning (diffusiondb) US**: Evaluate the diffusiondb fine-tuned model on the unsafe dataset
5. **Post-fine-tuning (dreambench) IP**: Evaluate the dreambench fine-tuned model on the i2p dataset
6. **Post-fine-tuning (dreambench) US**: Evaluate the dreambench fine-tuned model on the unsafe dataset

#### Results Output

Evaluation results are saved to `outputs/{model_name}/safety_eval/` directory:

```
outputs/{model_name}/safety_eval/
├── before_ft/
│   ├── i2p/results/      # Pre-fine-tuning IP metrics
│   └── unsafe/results/   # Pre-fine-tuning US metrics
└── after_ft/
    ├── diffusiondb/
    │   ├── i2p/results/    # Post-fine-tuning (diffusiondb) IP metrics
    │   └── unsafe/results/ # Post-fine-tuning (diffusiondb) US metrics
    └── dreambench/
        ├── i2p/results/    # Post-fine-tuning (dreambench) IP metrics
        └── unsafe/results/ # Post-fine-tuning (dreambench) US metrics
```

## Acknowledgements

This project builds upon open-source work from:

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [diffusers](https://github.com/huggingface/diffusers)
- [I-BAU](https://github.com/YiZeng623/I-BAU/tree/main) and [Hypertorch]( https://github.com/prolearner/hypertorch/tree/master)

Our dataset is built upon work from the following sources (including but not limited to):

- [I2P](https://huggingface.co/datasets/AIML-TUDA/i2p)
- [Unsafe Diffusion](https://arxiv.org/abs/2305.13873)
- [DreamBench++](https://arxiv.org/abs/2406.16855)
- [DiffusionDB](https://arxiv.org/abs/2210.14896)

We sincerely thank the respective authors for releasing their codebases.


## Citation

```bibtex
@inproceedings{li2025towards,
title={Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning},
author={Boheng Li and Renjie Gu and Junjie Wang and Leyi Qi and Yiming Li and Run Wang and Zhan Qin and Tianwei Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=iEtCCt6FjP}
}
```

## License

This project is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Please note that our datasets (both for training and evaluation) may include certain external datasets whose copyrights do not belong to us. If you wish to use those external datasets, please make sure to comply with their original licenses.