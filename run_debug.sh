export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # disabled: conflicts with sglang torch_memory_saver
export RAY_IGNORE_UNHANDLED_ERRORS=1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO
export API_PARALLEL=60
export CUDA_HOME=/home/yifan.lu/.conda/envs/doctor1/lib/python3.10/site-packages/nvidia
NVIDIA_PKGS=/home/yifan.lu/.conda/envs/doctor1/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA_PKGS/cuda_runtime/lib:$NVIDIA_PKGS/cuda_nvrtc/lib:$NVIDIA_PKGS/cublas/lib:$NVIDIA_PKGS/cudnn/lib:$NVIDIA_PKGS/nccl/lib:${LD_LIBRARY_PATH:-}

set -x

ulimit -n 65535
experiment_name='debug-run'
export TRIAL_NAME=$experiment_name

CONFIG_PATH=/home/yifan.lu/verl_doctor/verl/config
CONFIG_NAME='doctor_multiturn_grpo_w_interaction'
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
OFFLOAD=${OFFLOAD:-True}

# Data Config
train_files=./traindata/debug_train.jsonl
val_files=./traindata/debug_valid.jsonl
model_path=/nfs-stor/yifan.lu/ckpt/Qwen3-0.6B

# Interaction Config
multi_turn_enable=True
interaction_config_path=verl/config/interaction_config/doctor_interaction_config.yaml
max_user_turns=1
max_assistant_turns=1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.multi_turn.enable=$multi_turn_enable \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_user_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=$interaction_config_path \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    reward.reward_manager.name=naive \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[console]' \
    trainer.project_name=verl_doctor_rl \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    "$@"
