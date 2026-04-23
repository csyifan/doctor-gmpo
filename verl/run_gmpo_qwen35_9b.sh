#!/usr/bin/env bash
# GMPO training for Qwen3.5-9B on doctor consultation data
# Rollout backend: vllm (compatible with A40/SM86)
# Algorithm: GMPO (Geometric-Mean Policy Optimization)

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export LD_PRELOAD=/tmp/cuda_shim.so

set -x

ulimit -n 65535

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH=/workspace/data/Qwen3.5-9B
JUDGE_MODEL_PATH=/workspace/data/qwen-3.5-0.8B
TRAIN_FILES=/workspace/data/verl_traindata/qwen35_train100.jsonl
VAL_FILES=/workspace/data/verl_traindata/qwen35_valid10.jsonl
REWARD_FN_PATH=$(realpath "$(dirname "$0")/utils/reward_score/doctor_reward.py")

# ── Experiment ─────────────────────────────────────────────────────────────
experiment_name="Qwen3.5-9B-gmpo-doctor"
export TRIAL_NAME=$experiment_name

# ── GMPO-specific (Geometric-Mean PO) ──────────────────────────────────────
adv_estimator=grpo          # advantage estimator (group relative baseline)
loss_mode=geo_mean          # GMPO key: geometric mean IS ratio
clip_ratio=0.4              # recommended by GMPO paper

# ── Data ───────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE=16
MICRO_BATCH_SIZE=2
max_prompt_length=512
max_response_length=512

# ── Rollout ────────────────────────────────────────────────────────────────
rollout_n=4                          # responses per prompt
actor_rollout_name=vllm
actor_rollout_mode=async
actor_gpu_memory_utilization=0.5     # 9B模型权重18GB，0.5×44GB=22GB，含KV cache
actor_tensor_model_parallel_size=1

# ── Actor ──────────────────────────────────────────────────────────────────
actor_optim_lr=1e-6
use_kl_in_reward=False
actor_use_kl_loss=False
actor_kl_loss_coef=0.001
actor_kl_loss_type=low_var_kl
actor_entropy_coeff=0
OFFLOAD=True                         # offload optimizer to CPU (46 GB GPU)

# ── Judge 模型服务器 (qwen-3.5-0.8B) ──────────────────────────────────────
export JUDGE_HOST=localhost
export JUDGE_PORT=8001

# 如果 judge 服务器还没启动，则自动拉起
if ! curl -s "http://${JUDGE_HOST}:${JUDGE_PORT}/health" > /dev/null 2>&1; then
    echo "[INFO] 启动 judge 模型服务器: ${JUDGE_MODEL_PATH}"
    PYTHONPATH=/tmp/verl_src:$PYTHONPATH \
    /workspace/miniconda3/envs/doctor/bin/python3 -m vllm.entrypoints.openai.api_server \
        --model "${JUDGE_MODEL_PATH}" \
        --port "${JUDGE_PORT}" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.08 \
        --dtype bfloat16 \
        --trust-remote-code \
        --served-model-name judge \
        > /tmp/judge_server.log 2>&1 &
    JUDGE_PID=$!
    echo "[INFO] Judge 服务器 PID: ${JUDGE_PID}，等待启动..."
    for i in $(seq 1 60); do
        sleep 3
        if curl -s "http://${JUDGE_HOST}:${JUDGE_PORT}/health" > /dev/null 2>&1; then
            echo "[INFO] Judge 服务器已就绪"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "[ERROR] Judge 服务器启动超时，查看日志: /tmp/judge_server.log"
            exit 1
        fi
    done
else
    echo "[INFO] Judge 服务器已在运行"
fi

# ── Logging / checkpointing ────────────────────────────────────────────────
logger="['console']"
project_name=verl_doctor_gmpo
n_gpus_per_node=1
nnodes=1
save_freq=5
test_freq=2
total_epochs=10

PYTHONPATH=/tmp/verl_src:$PYTHONPATH \
/workspace/miniconda3/envs/doctor/bin/python3 -m verl.trainer.main_ppo \
    --config-path="$(realpath "$(dirname "$0")/config")" \
    --config-name="doctor_gmpo_vllm" \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=$actor_use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$actor_kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$actor_kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$actor_entropy_coeff \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=$actor_rollout_name \
    actor_rollout_ref.rollout.mode=$actor_rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=$actor_gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$actor_tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=$REWARD_FN_PATH \
    reward.custom_reward_function.name=compute_score \
    trainer.logger="$logger" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    "$@"
