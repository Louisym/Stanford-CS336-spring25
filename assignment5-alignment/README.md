# CS336 Assignment 5: Model Alignment

This assignment implements modern alignment techniques for language models, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). The goal is to align pre-trained models with human preferences and improve their reasoning capabilities.

## 📋 Table of Contents
- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Computational Requirements](#computational-requirements)
- [Important Notes](#important-notes)

## Overview

Model alignment bridges the gap between pre-trained language models and human preferences. This assignment covers:

1. **Supervised Fine-Tuning (SFT)**: Train models on high-quality demonstrations
2. **Direct Preference Optimization (DPO)**: Learn from preference pairs without reward modeling
3. **Group Relative Policy Optimization (GRPO)**: Policy gradient methods for reasoning tasks
4. **Expert Iteration (EI)**: Iterative self-improvement through bootstrapping
5. **Reward Modeling**: Train models to predict human preferences
6. **Evaluation**: MATH dataset for mathematical reasoning

**Primary Focus**: MATH dataset (mathematical problem solving)

## Implementation Details

### Key Components

#### 1. Supervised Fine-Tuning (`tests/test_sft.py`, `scripts/sft_experiment.py`)
- **Objective**: Maximize likelihood of expert demonstrations
- **Loss**: Cross-entropy on completion tokens only
- **Use Case**: Initial alignment step, teaching format and style

#### 2. Direct Preference Optimization (`tests/test_dpo.py`)
- **Objective**: Learn from preference pairs (chosen vs. rejected)
- **Loss**: Bradley-Terry preference model
- **Advantages**: No reward model needed, stable training
- **Formula**: L_DPO = -log σ(β log(π/π_ref))

#### 3. Group Relative Policy Optimization (`tests/test_grpo.py`, `scripts/grpo_train_loop.py`)
- **Objective**: Maximize expected reward via policy gradients
- **Key Innovation**: Group normalization reduces variance
- **Algorithms**:
  - REINFORCE with baseline
  - GRPO-Clip for off-policy training
- **Use Case**: Complex reasoning tasks (MATH problems)

#### 4. Reward Function (`cs336_alignment/drgrpo_grader.py`)
- **Format Reward**: Checks if output follows required format
- **Answer Reward**: Verifies correctness of mathematical answer
- **Combined Reward**: Format (0/1) × Answer (0/1)

#### 5. Evaluation Metrics (`tests/test_metrics.py`)
- **Accuracy**: Percentage of correct answers
- **Format Compliance**: Percentage following output format
- **Average Reward**: Mean reward across validation set

**Note**: ⚠️ The **supplemental safety RLHF assignment** has **not been implemented**. Stay tuned for future updates!

## Project Structure

```
assignment5-alignment/
├── cs336_alignment/
│   ├── drgrpo_grader.py       # Reward computation for MATH
│   ├── prompts/               # Prompt templates
│   │   └── r1_zero.prompt     # Zero-shot reasoning prompt
│   └── utils/                 # Utility functions
├── scripts/
│   ├── grpo_train_loop.py     # Main GRPO training script ⭐
│   ├── sft_experiment.py      # SFT baseline experiments
│   ├── ei_experiment.py       # Expert Iteration experiments
│   ├── evaluate_safety.py     # Safety evaluation (supplement)
│   └── convert_dataset_format.py  # Data format converter
├── tests/
│   ├── adapters.py            # Adapter implementations ⭐
│   ├── test_sft.py            # SFT tests
│   ├── test_dpo.py            # DPO tests
│   ├── test_grpo.py           # GRPO tests
│   ├── test_metrics.py        # Evaluation metric tests
│   └── test_data.py           # Data processing tests
├── data/
│   └── math/
│       ├── train.jsonl        # 7,500 training examples
│       └── test.jsonl         # 5,000 test examples
├── GRPO_TRAINING_GUIDE.md     # Detailed GRPO usage guide ⭐
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## Setup

### 1. Install Dependencies

This project requires special handling for flash-attn:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all packages except flash-attn first
uv sync --no-install-package flash-attn

# Then install flash-attn separately
uv sync
```

### 2. Download MATH Dataset

Download the training and validation data:

```bash
# Option 1: Use provided download script
uv run python download_math_dataset.py

# Option 2: Manual download from HuggingFace
# The MATH dataset should be downloaded and placed in data/math/
```

Expected format (JSON Lines):
```json
{"question": "What is 2+2?", "answer": "4"}
```

### 3. Download Base Model

Download the base model (Qwen2.5-Math-1.5B) for alignment:

```bash
# On cluster, model is typically at:
# /data/a5-alignment/models/Qwen2.5-Math-1.5B

# Or download from HuggingFace
huggingface-cli download Qwen/Qwen2.5-Math-1.5B
```

## Usage

### Quick Start: GRPO Training

See **[GRPO_TRAINING_GUIDE.md](./GRPO_TRAINING_GUIDE.md)** for comprehensive instructions.

**Test Run** (10 steps, ~30 minutes on 2xH100):
```bash
uv run python scripts/grpo_train_loop.py \
    --n-grpo-steps 10 \
    --eval-steps 2 \
    --rollout-batch-size 64 \
    --train-batch-size 64 \
    --gradient-accumulation-steps 32 \
    --model-path ./Qwen2.5-Math-1.5B \
    --wandb-project cs336-grpo \
    --wandb-run-name test-run
```

**Full Training** (200 steps, default hyperparameters):
```bash
uv run python scripts/grpo_train_loop.py \
    --model-path ./Qwen2.5-Math-1.5B \
    --wandb-project cs336-grpo \
    --wandb-run-name grpo-baseline
```

### Supervised Fine-Tuning

Train with SFT on MATH demonstrations:

```bash
uv run python scripts/sft_experiment.py \
    --model-path ./Qwen2.5-Math-1.5B \
    --data-path ./data/math/train.jsonl \
    --num-epochs 3 \
    --learning-rate 2e-5 \
    --batch-size 32
```

### Expert Iteration

Iterative self-improvement via bootstrapping:

```bash
uv run python scripts/ei_experiment.py \
    --model-path ./Qwen2.5-Math-1.5B \
    --num-iterations 5 \
    --samples-per-question 8
```

### Custom DPO Training

For preference-based training (requires preference dataset):

```python
from cs336_alignment.dpo import dpo_loss

# Compute DPO loss
loss = dpo_loss(
    policy_chosen_logps=chosen_logps,
    policy_rejected_logps=rejected_logps,
    reference_chosen_logps=ref_chosen_logps,
    reference_rejected_logps=ref_rejected_logps,
    beta=0.1
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_sft.py          # SFT tests
uv run pytest tests/test_dpo.py          # DPO tests
uv run pytest tests/test_grpo.py         # GRPO tests
uv run pytest tests/test_metrics.py      # Evaluation metrics
uv run pytest tests/test_data.py         # Data processing

# Verbose output
uv run pytest -v -s
```

**Important**: Complete the adapter functions in `tests/adapters.py`:
- `run_tokenize_prompt_and_output` - Tokenization with response masking
- `run_compute_group_normalized_rewards` - Group-normalized advantages
- `run_compute_entropy` - Per-token entropy
- `run_get_response_log_probs` - Log probabilities from model
- `run_compute_naive_policy_gradient_loss` - Vanilla policy gradient
- `run_compute_grpo_clip_loss` - GRPO-Clip loss
- `run_compute_policy_gradient_loss` - Loss dispatcher
- `run_masked_mean` - Masked averaging
- `run_masked_normalize` - Masked normalization
- `run_grpo_microbatch_train_step` - Microbatch training step

## Computational Requirements

### Hardware Requirements

| Method | Minimum | Recommended |
|--------|---------|-------------|
| **SFT** | 1× GPU (16GB) | 1× A100 40GB |
| **DPO** | 1× GPU (24GB) | 1× A100 40GB |
| **GRPO** | 2× GPU (40GB each) | 2× H100 80GB |

**Why 2 GPUs for GRPO?**
- GPU 1: Model training
- GPU 2: vLLM inference server for rollouts

### Training Time Estimates

**Qwen2.5-Math-1.5B (1.5B parameters)**:

| Method | Steps/Epochs | Hardware | Time |
|--------|--------------|----------|------|
| **SFT** | 3 epochs | 1× A100 | 2-4 hours |
| **DPO** | 1 epoch | 1× A100 | 3-5 hours |
| **GRPO (test)** | 10 steps | 2× H100 | 30 minutes |
| **GRPO (full)** | 200 steps | 2× H100 | 8-12 hours |
| **Expert Iteration** | 5 iterations | 2× A100 | 1-2 days |

### Memory Usage

- **Model (FP16)**: ~3GB
- **Optimizer States (AdamW)**: ~6GB
- **Activations + Gradients**: ~8-12GB (batch-dependent)
- **vLLM KV Cache**: ~10-20GB
- **Total**: ~30-40GB per GPU (for GRPO)

**Tips for Memory Optimization**:
- Use gradient checkpointing
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (BF16)

## Important Notes

### ⚠️ Implementation Limitations

1. **Supplemental Assignment Not Implemented**: The safety RLHF assignment (see `cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`) has **not been completed**. This includes:
   - Safety-focused reward modeling
   - Constitutional AI techniques
   - Red-teaming evaluations
   - **Stay tuned for future updates!**

2. **Multi-GPU Required for GRPO**: GRPO training requires at least 2 GPUs:
   - One for PyTorch training
   - One for vLLM inference server
   - Cannot run on single GPU without modifications

3. **vLLM Dependency**: GRPO relies on vLLM for fast inference during rollouts. Ensure vLLM is properly installed and CUDA-compatible.

### 💡 Tips for Success

1. **Start with SFT**: Before trying GRPO, establish a strong SFT baseline to verify data quality and model capabilities.

2. **Hyperparameter Tuning**:
   - **Learning Rate**: Start with 1e-5 for GRPO, 2e-5 for SFT
   - **Batch Size**: Larger is better for GRPO (reduces variance)
   - **Group Size**: 8 works well for MATH (more = better signal, but slower)

3. **Monitor Training**:
   - Watch entropy: should gradually decrease but not collapse
   - Track reward trends: should increase steadily
   - Check format compliance: should be >90% early on

4. **Gradient Accumulation**: Use gradient accumulation to simulate larger batches without OOM errors:
   ```bash
   --train-batch-size 1024 \
   --gradient-accumulation-steps 256
   ```

5. **Checkpointing**: GRPO saves checkpoints every 10 steps by default. Adjust with `--save-interval`.

6. **Use Baseline**: REINFORCE with baseline significantly reduces variance compared to vanilla REINFORCE.

### 🔍 Expected Results

After training with GRPO on MATH dataset:

**Baseline (Pre-trained Qwen2.5-Math-1.5B)**:
- Accuracy: ~30-40%
- Format Compliance: ~85-95%

**After SFT (3 epochs)**:
- Accuracy: ~45-55%
- Format Compliance: ~95-98%

**After GRPO (200 steps)**:
- Accuracy: ~55-65%
- Format Compliance: ~98-99%
- Typical improvement: +10-20% absolute over SFT

**After Expert Iteration (5 iterations)**:
- Accuracy: ~60-70%
- Format Compliance: ~99%

### 📊 Algorithm Comparison

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **SFT** | Simple, stable, fast | Limited by data quality | Initial alignment, format teaching |
| **DPO** | No reward model needed, stable | Requires preference pairs | Preference alignment |
| **GRPO** | Highest performance, sparse rewards | Complex, high variance, 2 GPUs | Reasoning tasks, optimization |
| **Expert Iteration** | Self-improving, no human labels | Slow, requires good base model | Iterative refinement |

### 🐛 Troubleshooting

**vLLM Not Starting?**
- Check GPU availability for vLLM: `nvidia-smi`
- Ensure CUDA version compatibility
- Verify model path is correct

**High Variance in GRPO?**
- Increase `group_size` (more samples per question)
- Increase `train_batch_size` (more questions per batch)
- Use `reinforce_with_baseline` instead of `no_baseline`
- Enable `use_std_normalization`

**Low Reward/Accuracy?**
- Verify reward function is working: check logs for non-zero rewards
- Ensure data format is correct (use `convert_dataset_format.py`)
- Try SFT first to verify model can solve problems

**OOM Errors?**
- Reduce `rollout_batch_size`
- Reduce `train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing

### 🔗 Useful Resources

**Papers**:
- [InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF for instruction following
- [DPO](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [GRPO](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - Self-critique and improvement

**Tools**:
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference server
- [WandB](https://wandb.ai/) - Experiment tracking
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model library

## Assignment Handouts

For detailed assignment requirements and theoretical background, see:
- **Main Assignment**: [cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)
- **Supplemental (Not Implemented)**: [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

## Additional Documentation

- **GRPO Training Guide**: [GRPO_TRAINING_GUIDE.md](./GRPO_TRAINING_GUIDE.md) - Comprehensive GRPO usage instructions

## License

This code is provided for educational purposes as part of Stanford CS336.

---

# 中文版本 | Chinese Version

# CS336 作业 5: 模型对齐

本作业实现了现代语言模型对齐技术，包括监督微调（SFT）、直接偏好优化（DPO）和组相对策略优化（GRPO）。目标是使预训练模型与人类偏好对齐，并提高其推理能力。

## 📋 目录
- [概述](#概述-1)
- [实现细节](#实现细节-1)
- [项目结构](#项目结构-1)
- [环境配置](#环境配置-1)
- [使用指南](#使用指南-1)
- [测试](#测试-1)
- [计算资源需求](#计算资源需求-1)
- [重要说明](#重要说明-1)

## 概述

模型对齐弥合了预训练语言模型与人类偏好之间的差距。本作业涵盖：

1. **监督微调（SFT）**: 在高质量示范上训练模型
2. **直接偏好优化（DPO）**: 从偏好对中学习，无需奖励建模
3. **组相对策略优化（GRPO）**: 用于推理任务的策略梯度方法
4. **专家迭代（EI）**: 通过自举进行迭代自我改进
5. **奖励建模**: 训练模型预测人类偏好
6. **评估**: MATH 数据集用于数学推理

**主要关注**: MATH 数据集（数学问题求解）

## 实现细节

### 核心组件

#### 1. 监督微调 (`tests/test_sft.py`, `scripts/sft_experiment.py`)
- **目标**: 最大化专家示范的似然
- **损失**: 仅对完成 token 的交叉熵
- **使用场景**: 初始对齐步骤，教授格式和风格

#### 2. 直接偏好优化 (`tests/test_dpo.py`)
- **目标**: 从偏好对（选择 vs 拒绝）中学习
- **损失**: Bradley-Terry 偏好模型
- **优势**: 无需奖励模型，训练稳定
- **公式**: L_DPO = -log σ(β log(π/π_ref))

#### 3. 组相对策略优化 (`tests/test_grpo.py`, `scripts/grpo_train_loop.py`)
- **目标**: 通过策略梯度最大化期望奖励
- **关键创新**: 组归一化减少方差
- **算法**:
  - 带基线的 REINFORCE
  - 用于离策略训练的 GRPO-Clip
- **使用场景**: 复杂推理任务（MATH 问题）

#### 4. 奖励函数 (`cs336_alignment/drgrpo_grader.py`)
- **格式奖励**: 检查输出是否遵循所需格式
- **答案奖励**: 验证数学答案的正确性
- **组合奖励**: 格式 (0/1) × 答案 (0/1)

#### 5. 评估指标 (`tests/test_metrics.py`)
- **准确率**: 正确答案的百分比
- **格式合规性**: 遵循输出格式的百分比
- **平均奖励**: 验证集上的平均奖励

**注意**: ⚠️ **补充的安全 RLHF 作业**尚**未实现**。敬请期待后续更新！

## 项目结构

```
assignment5-alignment/
├── cs336_alignment/
│   ├── drgrpo_grader.py       # MATH 的奖励计算
│   ├── prompts/               # 提示模板
│   │   └── r1_zero.prompt     # 零样本推理提示
│   └── utils/                 # 工具函数
├── scripts/
│   ├── grpo_train_loop.py     # GRPO 主训练脚本 ⭐
│   ├── sft_experiment.py      # SFT 基线实验
│   ├── ei_experiment.py       # 专家迭代实验
│   ├── evaluate_safety.py     # 安全评估（补充）
│   └── convert_dataset_format.py  # 数据格式转换器
├── tests/
│   ├── adapters.py            # 适配器实现 ⭐
│   ├── test_sft.py            # SFT 测试
│   ├── test_dpo.py            # DPO 测试
│   ├── test_grpo.py           # GRPO 测试
│   ├── test_metrics.py        # 评估指标测试
│   └── test_data.py           # 数据处理测试
├── data/
│   └── math/
│       ├── train.jsonl        # 7,500 个训练样本
│       └── test.jsonl         # 5,000 个测试样本
├── GRPO_TRAINING_GUIDE.md     # 详细的 GRPO 使用指南 ⭐
├── pyproject.toml             # 依赖项
└── README.md                  # 本文件
```

## 环境配置

### 1. 安装依赖

本项目需要特殊处理 flash-attn：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 先安装除 flash-attn 之外的所有包
uv sync --no-install-package flash-attn

# 然后单独安装 flash-attn
uv sync
```

### 2. 下载 MATH 数据集

下载训练和验证数据：

```bash
# 选项 1: 使用提供的下载脚本
uv run python download_math_dataset.py

# 选项 2: 从 HuggingFace 手动下载
# MATH 数据集应下载并放置在 data/math/ 中
```

预期格式（JSON Lines）:
```json
{"question": "What is 2+2?", "answer": "4"}
```

### 3. 下载基础模型

下载用于对齐的基础模型（Qwen2.5-Math-1.5B）：

```bash
# 在集群上，模型通常位于：
# /data/a5-alignment/models/Qwen2.5-Math-1.5B

# 或从 HuggingFace 下载
huggingface-cli download Qwen/Qwen2.5-Math-1.5B
```

## 使用指南

### 快速开始: GRPO 训练

详细说明请参见 **[GRPO_TRAINING_GUIDE.md](./GRPO_TRAINING_GUIDE.md)**。

**测试运行**（10 步，2xH100 上约 30 分钟）:
```bash
uv run python scripts/grpo_train_loop.py \
    --n-grpo-steps 10 \
    --eval-steps 2 \
    --rollout-batch-size 64 \
    --train-batch-size 64 \
    --gradient-accumulation-steps 32 \
    --model-path ./Qwen2.5-Math-1.5B \
    --wandb-project cs336-grpo \
    --wandb-run-name test-run
```

**完整训练**（200 步，默认超参数）:
```bash
uv run python scripts/grpo_train_loop.py \
    --model-path ./Qwen2.5-Math-1.5B \
    --wandb-project cs336-grpo \
    --wandb-run-name grpo-baseline
```

### 监督微调

在 MATH 示范上使用 SFT 训练：

```bash
uv run python scripts/sft_experiment.py \
    --model-path ./Qwen2.5-Math-1.5B \
    --data-path ./data/math/train.jsonl \
    --num-epochs 3 \
    --learning-rate 2e-5 \
    --batch-size 32
```

### 专家迭代

通过自举进行迭代自我改进：

```bash
uv run python scripts/ei_experiment.py \
    --model-path ./Qwen2.5-Math-1.5B \
    --num-iterations 5 \
    --samples-per-question 8
```

### 自定义 DPO 训练

用于基于偏好的训练（需要偏好数据集）：

```python
from cs336_alignment.dpo import dpo_loss

# 计算 DPO 损失
loss = dpo_loss(
    policy_chosen_logps=chosen_logps,
    policy_rejected_logps=rejected_logps,
    reference_chosen_logps=ref_chosen_logps,
    reference_rejected_logps=ref_rejected_logps,
    beta=0.1
)
```

## 测试

运行综合测试套件：

```bash
# 运行所有测试
uv run pytest

# 运行特定测试模块
uv run pytest tests/test_sft.py          # SFT 测试
uv run pytest tests/test_dpo.py          # DPO 测试
uv run pytest tests/test_grpo.py         # GRPO 测试
uv run pytest tests/test_metrics.py      # 评估指标
uv run pytest tests/test_data.py         # 数据处理

# 详细输出
uv run pytest -v -s
```

**重要提示**: 在 `tests/adapters.py` 中完成适配器函数：
- `run_tokenize_prompt_and_output` - 带响应掩码的分词
- `run_compute_group_normalized_rewards` - 组归一化优势
- `run_compute_entropy` - 每个 token 的熵
- `run_get_response_log_probs` - 从模型获取对数概率
- `run_compute_naive_policy_gradient_loss` - 普通策略梯度
- `run_compute_grpo_clip_loss` - GRPO-Clip 损失
- `run_compute_policy_gradient_loss` - 损失分发器
- `run_masked_mean` - 掩码平均
- `run_masked_normalize` - 掩码归一化
- `run_grpo_microbatch_train_step` - 微批次训练步骤

## 计算资源需求

### 硬件要求

| 方法 | 最低配置 | 推荐配置 |
|--------|---------|-------------|
| **SFT** | 1× GPU (16GB) | 1× A100 40GB |
| **DPO** | 1× GPU (24GB) | 1× A100 40GB |
| **GRPO** | 2× GPU (每个 40GB) | 2× H100 80GB |

**为什么 GRPO 需要 2 个 GPU？**
- GPU 1: 模型训练
- GPU 2: vLLM 推理服务器用于 rollout

### 训练时间估计

**Qwen2.5-Math-1.5B（1.5B 参数）**:

| 方法 | 步骤/轮次 | 硬件 | 时间 |
|--------|--------------|----------|------|
| **SFT** | 3 轮 | 1× A100 | 2-4 小时 |
| **DPO** | 1 轮 | 1× A100 | 3-5 小时 |
| **GRPO（测试）** | 10 步 | 2× H100 | 30 分钟 |
| **GRPO（完整）** | 200 步 | 2× H100 | 8-12 小时 |
| **专家迭代** | 5 次迭代 | 2× A100 | 1-2 天 |

### 内存使用

- **模型（FP16）**: 约 3GB
- **优化器状态（AdamW）**: 约 6GB
- **激活 + 梯度**: 约 8-12GB（取决于批次）
- **vLLM KV 缓存**: 约 10-20GB
- **总计**: 每个 GPU 约 30-40GB（用于 GRPO）

**内存优化技巧**:
- 使用梯度检查点
- 减少批次大小
- 使用梯度累积
- 启用混合精度（BF16）

## 重要说明

### ⚠️ 实现限制

1. **补充作业未实现**: 安全 RLHF 作业（见 `cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`）**尚未完成**。这包括：
   - 以安全为重点的奖励建模
   - Constitutional AI 技术
   - 红队评估
   - **敬请期待后续更新！**

2. **GRPO 需要多 GPU**: GRPO 训练至少需要 2 个 GPU：
   - 一个用于 PyTorch 训练
   - 一个用于 vLLM 推理服务器
   - 无法在单 GPU 上运行，除非进行修改

3. **vLLM 依赖**: GRPO 依赖 vLLM 在 rollout 期间进行快速推理。确保 vLLM 已正确安装并兼容 CUDA。

### 💡 成功技巧

1. **从 SFT 开始**: 在尝试 GRPO 之前，建立强大的 SFT 基线以验证数据质量和模型能力。

2. **超参数调整**:
   - **学习率**: GRPO 从 1e-5 开始，SFT 从 2e-5 开始
   - **批次大小**: 越大越好（减少方差）
   - **组大小**: 8 对 MATH 效果良好（更多 = 更好的信号，但更慢）

3. **监控训练**:
   - 观察熵: 应逐渐减少但不崩溃
   - 跟踪奖励趋势: 应稳步增加
   - 检查格式合规性: 早期应 >90%

4. **梯度累积**: 使用梯度累积模拟更大批次而不出现 OOM 错误：
   ```bash
   --train-batch-size 1024 \
   --gradient-accumulation-steps 256
   ```

5. **检查点**: GRPO 默认每 10 步保存检查点。使用 `--save-interval` 调整。

6. **使用基线**: 带基线的 REINFORCE 显著减少方差（相比普通 REINFORCE）。

### 🔍 预期结果

在 MATH 数据集上使用 GRPO 训练后：

**基线（预训练 Qwen2.5-Math-1.5B）**:
- 准确率: 约 30-40%
- 格式合规性: 约 85-95%

**SFT 后（3 轮）**:
- 准确率: 约 45-55%
- 格式合规性: 约 95-98%

**GRPO 后（200 步）**:
- 准确率: 约 55-65%
- 格式合规性: 约 98-99%
- 典型改进: 比 SFT 绝对提高 +10-20%

**专家迭代后（5 次迭代）**:
- 准确率: 约 60-70%
- 格式合规性: 约 99%

### 📊 算法比较

| 方法 | 优点 | 缺点 | 使用场景 |
|--------|------|------|----------|
| **SFT** | 简单、稳定、快速 | 受数据质量限制 | 初始对齐、格式教学 |
| **DPO** | 无需奖励模型、稳定 | 需要偏好对 | 偏好对齐 |
| **GRPO** | 最高性能、稀疏奖励 | 复杂、高方差、2 个 GPU | 推理任务、优化 |
| **专家迭代** | 自我改进、无需人工标签 | 慢、需要良好的基础模型 | 迭代改进 |

### 🐛 故障排除

**vLLM 未启动？**
- 检查 vLLM 的 GPU 可用性: `nvidia-smi`
- 确保 CUDA 版本兼容性
- 验证模型路径正确

**GRPO 方差高？**
- 增加 `group_size`（每个问题更多样本）
- 增加 `train_batch_size`（每批次更多问题）
- 使用 `reinforce_with_baseline` 而非 `no_baseline`
- 启用 `use_std_normalization`

**奖励/准确率低？**
- 验证奖励函数是否工作: 检查日志中的非零奖励
- 确保数据格式正确（使用 `convert_dataset_format.py`）
- 先尝试 SFT 以验证模型能够解决问题

**OOM 错误？**
- 减少 `rollout_batch_size`
- 减少 `train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用梯度检查点

### 🔗 有用资源

**论文**:
- [InstructGPT](https://arxiv.org/abs/2203.02155) - 用于指令遵循的 RLHF
- [DPO](https://arxiv.org/abs/2305.18290) - 直接偏好优化
- [GRPO](https://arxiv.org/abs/2402.03300) - 组相对策略优化
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - 自我批评和改进

**工具**:
- [vLLM](https://github.com/vllm-project/vllm) - 快速推理服务器
- [WandB](https://wandb.ai/) - 实验跟踪
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - 模型库

## 作业说明

详细的作业要求和理论背景，请参阅：
- **主要作业**: [cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)
- **补充（未实现）**: [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

## 额外文档

- **GRPO 训练指南**: [GRPO_TRAINING_GUIDE.md](./GRPO_TRAINING_GUIDE.md) - 详细的 GRPO 使用说明

## 许可证

本代码仅供教育目的使用，是斯坦福 CS336 课程的一部分。
