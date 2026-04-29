# Doctor-GMPO: Medical AI RL Training with verl

A reinforcement learning training framework for medical consultation models, built on top of [verl](https://github.com/verl-project/verl). It fine-tunes LLMs using **GMPO** (Geometric Mean Policy Optimization) + **LoRA** for multi-turn patient-doctor dialogue.

Supported models: **Doctor-R1**, **Qwen3.5-9B** (converted to text-only backbone)

---

## Overview

This repository extends verl with the following customizations:

- **Multi-turn medical dialogue RL**: Simulates patient-doctor interactions using the KAMED dataset
- **GMPO algorithm**: Geometric mean policy optimization (`policy_loss.loss_mode=geo_mean`) as a drop-in replacement for standard GRPO
- **LoRA fine-tuning**: Reduces GPU memory footprint, enabling 9B-scale model training on 2 GPUs
- **Qwen3.5-9B support**: Includes a conversion script to strip the vision encoder from Qwen3.5-9B and produce a text-only `qwen3_next` checkpoint compatible with SGLang
- **SGLang rollout**: Uses SGLang as the inference engine for fast generation during rollout

## Repository Layout

```
verl_doctor/
├── verl/
│   ├── config/
│   │   ├── doctor_multiturn_grpo_w_interaction.yaml   # main training config
│   │   └── interaction_config/
│   │       └── doctor_interaction_config.yaml         # multi-turn interaction config
│   └── interactions/
│       └── doctor_interaction.py                      # patient-doctor interaction logic
├── traindata/                                         # train / validation data files
├── convert_kamed.py                                   # KAMED dataset conversion script
├── convert_qwen35_to_qwen3next.py                     # Qwen3.5-9B VLM → text-only conversion
├── run_gmpo_lora.sh                                   # Doctor-R1 training launcher
├── run_qwen35_gmpo_lora.sh                            # Qwen3.5-9B training launcher
├── run_gmpo_lora_sbatch.sh                            # SLURM batch job (Doctor-R1)
└── run_qwen35_gmpo_lora_sbatch.sh                     # SLURM batch job (Qwen3.5-9B)
```

## Setup

### Install dependencies

Using `uv` (recommended):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.10
source .venv/bin/activate

uv pip install -e ".[gpu]"
uv pip install sglang
```

Or using conda:

```bash
conda create -n doctor1 python=3.10
conda activate doctor1
pip install -e ".[gpu]"
pip install sglang
```

### Initialize submodules

```bash
git submodule update --init --recursive
```

## Data Preparation

### Convert the KAMED dataset

```bash
python convert_kamed.py
```

This converts raw KAMED session JSON files into the JSONL format expected by verl's `RLHFDataset`. Output files are written to `traindata/`:

| File | Description |
|------|-------------|
| `train.jsonl` | Full training set |
| `valid.jsonl` | Full validation set |
| `debug_train.jsonl` | Small debug split (Doctor-R1) |
| `debug_valid.jsonl` | Small debug validation (Doctor-R1) |
| `qwen35_train100.jsonl` | 100-sample debug training set (Qwen3.5-9B) |
| `qwen35_valid10.jsonl` | 10-sample debug validation set (Qwen3.5-9B) |

## Model Conversion (Qwen3.5-9B)

The Qwen3.5-9B checkpoint is a VLM. It must be converted to a text-only format before it can be used with SGLang and verl:

```bash
python convert_qwen35_to_qwen3next.py \
    --input /path/to/Qwen3.5-9B-VLM \
    --output /path/to/Qwen3.5-9B-text
```

The script performs three transforms:
1. Strips the vision encoder and MTP weights
2. Remaps weight keys: `model.language_model.X` → `model.X`
3. Merges split linear-attention projections (`in_proj_qkv + in_proj_z` → `in_proj_qkvz`, etc.)

## Training

### Doctor-R1 + GMPO LoRA

```bash
# Single-node, 2 GPUs
bash run_gmpo_lora.sh

# Override hyperparameters via env vars
LORA_RANK=64 LORA_ALPHA=128 TRAIN_BATCH_SIZE=4 bash run_gmpo_lora.sh

# Submit to a SLURM cluster
sbatch run_gmpo_lora_sbatch.sh
```

### Qwen3.5-9B + GMPO LoRA

```bash
# GPU count is inferred automatically from CUDA_VISIBLE_DEVICES
bash run_qwen35_gmpo_lora.sh

# Specify GPUs and model path explicitly
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH=/path/to/Qwen3.5-9B-text bash run_qwen35_gmpo_lora.sh

# Submit to a SLURM cluster
sbatch run_qwen35_gmpo_lora_sbatch.sh
```

### Key hyperparameters

| Parameter | Doctor-R1 default | Qwen3.5-9B default | Description |
|-----------|:-----------------:|:------------------:|-------------|
| `LORA_RANK` | 32 | 16 | LoRA rank |
| `LORA_ALPHA` | 64 | 32 | LoRA scaling factor |
| `TRAIN_BATCH_SIZE` | 2 | 2 | Training batch size |
| `OFFLOAD` | True | True | CPU offload (saves GPU memory) |
| `max_user_turns` | 1 | 1 | Max patient turns per episode |
| `max_assistant_turns` | 1 | 1 | Max doctor turns per episode |

Any verl config key can be appended directly to the launch command as `key=value` to override defaults.

## Multi-turn Interaction

Training uses verl's multi-turn rollout with a custom `DoctorInteraction` class that simulates patient feedback:

```yaml
# verl/config/interaction_config/doctor_interaction_config.yaml
interaction:
  - name: "kamed"
    class_name: "verl.interactions.doctor_interaction.DoctorInteraction"
```

Each episode proceeds as follows:
1. The patient sends an initial symptom description
2. The doctor model generates a follow-up question or diagnosis
3. `DoctorInteraction` scores the response and returns a reward signal
4. GMPO uses the reward to update the policy via LoRA gradients

## Acknowledgements

This project is built on [verl](https://github.com/verl-project/verl) by the ByteDance Seed Team. Training data comes from the [KAMED](https://github.com/RecipeMind/VRBot) medical dialogue dataset.

- verl paper: [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- Base model: [Doctor-R1](https://huggingface.co/Doctor-R1)
