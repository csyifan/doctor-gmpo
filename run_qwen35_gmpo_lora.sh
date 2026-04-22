#!/bin/bash
# Debug run: Qwen3.5-9B (converted to qwen3_next text backbone) + LoRA + GMPO
# Uses 100-sample doctor-r1 subset for quick debugging

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
unset ROCR_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export RAY_IGNORE_UNHANDLED_ERRORS=1
export NCCL_DEBUG=WARN
export API_PARALLEL=60

# Auto-detect network interface for Gloo (override with GLOO_SOCKET_IFNAME env var)
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$(ip -o -4 addr show | awk 'NR==1{print $2; exit}')}

# Auto-detect conda prefix for CUDA libs (override with CONDA_PREFIX env var)
_CONDA_PREFIX=${CONDA_PREFIX:-$(conda info --base)/envs/doctor1}
NVIDIA_PKGS=$_CONDA_PREFIX/lib/python3.10/site-packages/nvidia
if [ -d "$NVIDIA_PKGS" ]; then
    export CUDA_HOME=$NVIDIA_PKGS
    export LD_LIBRARY_PATH=$NVIDIA_PKGS/cuda_runtime/lib:$NVIDIA_PKGS/cuda_nvrtc/lib:$NVIDIA_PKGS/cublas/lib:$NVIDIA_PKGS/cudnn/lib:$NVIDIA_PKGS/nccl/lib:${LD_LIBRARY_PATH:-}
fi

set -x
ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

experiment_name='qwen35-gmpo-lora-debug'
export TRIAL_NAME=$experiment_name

CONFIG_PATH=$SCRIPT_DIR/verl/config
CONFIG_NAME='doctor_multiturn_grpo_w_interaction'
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
OFFLOAD=${OFFLOAD:-True}

# Data (100-sample subset)
train_files=./traindata/qwen35_train100.jsonl
val_files=./traindata/qwen35_valid10.jsonl

# Converted text-only model (override with MODEL_PATH env var)
model_path=${MODEL_PATH:-/nfs-stor/yifan.lu/ckpt/Qwen3.5-9B-text}

# Multi-turn
multi_turn_enable=True
interaction_config_path=verl/config/interaction_config/doctor_interaction_config.yaml
max_user_turns=1
max_assistant_turns=1

# LoRA
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

# Number of GPUs (derived from CUDA_VISIBLE_DEVICES)
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files=$train_files \
    data.val_files=$val_files \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.4 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.policy_loss.loss_mode=geo_mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.multi_turn.enable=$multi_turn_enable \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_user_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=$interaction_config_path \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    reward.reward_manager.name=naive \
    reward.num_workers=2 \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=verl_doctor_rl \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    "$@" \
    2>&1 | tee logs/qwen35_gmpo_lora_debug.log
