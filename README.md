# Doctor-GMPO: Medical AI RL Training with verl

基于 [verl](https://github.com/verl-project/verl) 的医疗对话强化学习训练框架，使用 **GMPO**（Geometric Mean Policy Optimization）+ **LoRA** 对医疗问诊模型进行微调。

支持模型：**Doctor-R1**、**Qwen3.5-9B**（转换为 text-only backbone）

---

## 项目简介

本仓库在 verl 基础上进行了以下定制化开发：

- **医疗多轮对话 RL 训练**：基于 KAMED 数据集，模拟患者-医生多轮问诊交互
- **GMPO 算法**：使用几何平均策略优化（`policy_loss.loss_mode=geo_mean`），替代标准 GRPO 损失
- **LoRA 低秩微调**：显著降低显存占用，支持在 2 张 GPU 上训练 9B 级别模型
- **Qwen3.5-9B 支持**：包含将 Qwen3.5-9B VLM 检查点转换为纯文本 qwen3_next backbone 的工具脚本
- **SGLang 推理引擎**：rollout 阶段使用 SGLang 加速生成

## 目录结构

```
verl_doctor/
├── verl/
│   ├── config/
│   │   ├── doctor_multiturn_grpo_w_interaction.yaml   # 主训练配置
│   │   └── interaction_config/
│   │       └── doctor_interaction_config.yaml         # 多轮交互配置
│   └── interactions/
│       └── doctor_interaction.py                      # 患者-医生交互逻辑
├── traindata/                                         # 训练/验证数据
├── convert_kamed.py                                   # KAMED 数据集转换脚本
├── convert_qwen35_to_qwen3next.py                     # Qwen3.5-9B 模型转换脚本
├── run_gmpo_lora.sh                                   # Doctor-R1 训练启动脚本
├── run_qwen35_gmpo_lora.sh                            # Qwen3.5-9B 训练启动脚本
├── run_gmpo_lora_sbatch.sh                            # SLURM 批量提交（Doctor-R1）
└── run_qwen35_gmpo_lora_sbatch.sh                     # SLURM 批量提交（Qwen3.5-9B）
```

## 环境配置

### 依赖安装

```bash
# 推荐使用 uv 管理 Python 环境
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.10
source .venv/bin/activate

uv pip install -e ".[gpu]"
uv pip install sglang
```

或使用 conda：

```bash
conda create -n doctor1 python=3.10
conda activate doctor1
pip install -e ".[gpu]"
pip install sglang
```

### 子模块初始化

```bash
git submodule update --init --recursive
```

## 数据准备

### 使用 KAMED 数据集

```bash
# 将 KAMED 原始 JSON 会话文件转换为 verl 格式 JSONL
python convert_kamed.py
```

转换后的文件输出至 `traindata/`，包含：

| 文件 | 说明 |
|------|------|
| `train.jsonl` | 完整训练集 |
| `valid.jsonl` | 完整验证集 |
| `debug_train.jsonl` | 小样本调试集（Doctor-R1） |
| `debug_valid.jsonl` | 小样本验证集（Doctor-R1） |
| `qwen35_train100.jsonl` | Qwen3.5 100 样本调试训练集 |
| `qwen35_valid10.jsonl` | Qwen3.5 10 样本调试验证集 |

## 模型转换（Qwen3.5-9B）

Qwen3.5-9B 原始检查点为 VLM（视觉-语言模型），需转换为 text-only 格式才能与 SGLang + verl 兼容：

```bash
python convert_qwen35_to_qwen3next.py \
    --input /path/to/Qwen3.5-9B-VLM \
    --output /path/to/Qwen3.5-9B-text
```

转换内容：
- 剥离视觉编码器及 MTP 权重
- 重映射权重键名：`model.language_model.X` → `model.X`
- 合并分裂的 linear attention 投影矩阵

## 训练

### Doctor-R1 + GMPO LoRA

```bash
# 2 GPU 单机训练
bash run_gmpo_lora.sh

# 自定义参数
LORA_RANK=64 LORA_ALPHA=128 TRAIN_BATCH_SIZE=4 bash run_gmpo_lora.sh

# SLURM 集群提交
sbatch run_gmpo_lora_sbatch.sh
```

### Qwen3.5-9B + GMPO LoRA

```bash
# 自动检测 GPU 数量（CUDA_VISIBLE_DEVICES）
bash run_qwen35_gmpo_lora.sh

# 指定 GPU 和模型路径
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH=/path/to/Qwen3.5-9B-text bash run_qwen35_gmpo_lora.sh

# SLURM 集群提交
sbatch run_qwen35_gmpo_lora_sbatch.sh
```

### 关键训练超参数

| 参数 | Doctor-R1 默认值 | Qwen3.5-9B 默认值 | 说明 |
|------|-----------------|-------------------|------|
| `LORA_RANK` | 32 | 16 | LoRA 秩 |
| `LORA_ALPHA` | 64 | 32 | LoRA 缩放系数 |
| `TRAIN_BATCH_SIZE` | 2 | 2 | 训练 batch 大小 |
| `OFFLOAD` | True | True | CPU offload（省显存） |
| `max_user_turns` | 1 | 1 | 最大患者轮次 |
| `max_assistant_turns` | 1 | 1 | 最大医生轮次 |

所有脚本参数均支持通过环境变量或命令行末尾追加 `key=value` 覆盖。

## 多轮交互机制

训练采用 verl 的 multi-turn rollout，通过 `DoctorInteraction` 类模拟患者反馈：

```yaml
# verl/config/interaction_config/doctor_interaction_config.yaml
interaction:
  - name: "kamed"
    class_name: "verl.interactions.doctor_interaction.DoctorInteraction"
```

交互流程：
1. 患者发送初始症状描述
2. 医生模型生成问诊/诊断回复
3. `DoctorInteraction` 基于规则或模型评估奖励分数
4. RL 算法（GMPO）利用奖励信号更新策略

## 致谢

本项目基于 [verl](https://github.com/verl-project/verl)（ByteDance Seed Team）开发，训练数据使用 [KAMED](https://github.com/RecipeMind/VRBot) 医疗对话数据集。

- verl 论文：[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- Doctor-R1 模型基于 [Doctor-R1](https://huggingface.co/Doctor-R1) 初始化
