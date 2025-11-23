# CS336 Assignment 3: Scaling Laws

This assignment explores empirical scaling laws in language models, investigating how model performance scales with compute, model size, and dataset size. The goal is to understand and predict language model behavior across different scales.

## 📋 Table of Contents
- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Computational Requirements](#computational-requirements)
- [Important Notes](#important-notes)

## Overview

Scaling laws describe how language model performance (measured by loss or perplexity) changes with:

1. **Model Size** (N): Number of non-embedding parameters
2. **Dataset Size** (D): Number of training tokens
3. **Compute Budget** (C): Total FLOPs used for training

This assignment involves:
- Training models of various sizes on different dataset sizes
- Measuring final loss and training dynamics
- Fitting power-law relationships to empirical data
- Predicting optimal model configurations for given compute budgets
- Understanding compute-optimal training (Chinchilla scaling laws)

## Implementation Details

### Key Components

#### 1. Transformer Language Model (`cs336_scaling/model.py`)

A standard Transformer decoder implementation with:
- **Token & Position Embeddings**: Learned embeddings for tokens and positions
- **Multi-Head Self-Attention**: Causal attention with dropout
- **Feed-Forward Networks**: GELU-activated FFNs
- **Layer Normalization**: Pre-norm architecture
- **Text Generation**: Autoregressive sampling with temperature/top-k

**Key Methods**:
- `forward()`: Standard forward pass returning logits
- `generate()`: Autoregressive text generation
- `from_pretrained()`: Load pre-trained models
- `get_num_params()`: Count non-embedding parameters

#### 2. Scaling Law Experiments

This assignment requires running systematic experiments to collect data points:

**Experiment Design**:
- Train models with different numbers of layers/dimensions
- Vary dataset size (e.g., 1M, 10M, 100M, 1B tokens)
- Measure loss at different training steps
- Collect compute metrics (FLOPs, training time)

**Analysis**:
- Fit power laws: L(N) = aN^(-α), L(D) = bD^(-β), L(C) = cC^(-γ)
- Determine compute-optimal allocation between N and D
- Predict performance for unseen configurations

## Project Structure

```
assignment3-scaling/
├── cs336_scaling/
│   ├── model.py               # Transformer LM implementation
│   └── __init__.py
├── data/                      # Training datasets (user-provided)
├── experiments/               # Experiment scripts and results (user-created)
├── cs336_spring2025_assignment3_scaling.pdf  # Assignment handout
├── pyproject.toml             # Dependencies
├── uv.lock                    # Lock file
└── README.md                  # This file
```

**Note**: This assignment is primarily experimental/analytical. You'll need to create your own:
- Training scripts
- Data processing pipelines
- Experiment tracking code
- Analysis notebooks

## Setup

### 1. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Prepare Datasets

You'll need datasets of various sizes. Options include:

**Option 1: Use Public Datasets**
```bash
mkdir -p data

# TinyStories (~1GB)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -O data/tinystories.txt

# OpenWebText (~40GB)
# Download from https://huggingface.co/datasets/Skylion007/openwebtext

# The Pile (825GB)
# Download from https://pile.eleuther.ai/
```

**Option 2: Create Synthetic Datasets**
```python
# Generate datasets of controlled sizes
import random
import string

def generate_synthetic_text(num_tokens, vocab_size=10000):
    """Generate random text for scaling experiments"""
    tokens = [str(random.randint(0, vocab_size-1)) for _ in range(num_tokens)]
    return ' '.join(tokens)

# Create 1M, 10M, 100M token datasets
for size in [1_000_000, 10_000_000, 100_000_000]:
    text = generate_synthetic_text(size)
    with open(f'data/synthetic_{size}.txt', 'w') as f:
        f.write(text)
```

## Usage

### 1. Define Model Configurations

Create models of different sizes by varying hyperparameters:

```python
from cs336_scaling.model import BasicsTransformerLM

# Small model (~10M params)
model_small = BasicsTransformerLM(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024
)

# Medium model (~50M params)
model_medium = BasicsTransformerLM(
    vocab_size=10000,
    context_length=512,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=2048
)

# Large model (~200M params)
model_large = BasicsTransformerLM(
    vocab_size=10000,
    context_length=1024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072
)
```

### 2. Run Scaling Experiments

Example training script for collecting scaling data:

```python
import torch
import numpy as np
from cs336_scaling.model import BasicsTransformerLM

def train_and_measure(model, dataset, num_steps, compute_budget):
    """
    Train model and measure loss at various checkpoints.

    Returns: dict with losses, compute used, final metrics
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []

    for step in range(num_steps):
        # Training step
        batch = get_batch(dataset)  # Your data loading logic
        logits = model(batch['input_ids'])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               batch['labels'].view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Record loss periodically
        if step % 100 == 0:
            losses.append(loss.item())

    return {
        'final_loss': losses[-1],
        'loss_curve': losses,
        'num_params': model.get_num_params(),
        'num_tokens': num_steps * batch_size * context_length
    }

# Run experiments
results = []
for model_config in [small_config, medium_config, large_config]:
    for dataset_size in [1e6, 1e7, 1e8]:
        model = BasicsTransformerLM(**model_config)
        result = train_and_measure(model, dataset_size, num_steps=10000)
        results.append(result)
```

### 3. Analyze Scaling Laws

Fit power laws to your experimental data:

```python
import numpy as np
from scipy.optimize import curve_fit

# Power law function: L = a * N^(-alpha)
def power_law(x, a, alpha):
    return a * np.power(x, -alpha)

# Collect data points
param_counts = [result['num_params'] for result in results]
final_losses = [result['final_loss'] for result in results]

# Fit power law
params, _ = curve_fit(power_law, param_counts, final_losses)
a, alpha = params

print(f"Scaling law: L(N) = {a:.4f} * N^(-{alpha:.4f})")

# Predict loss for unseen model size
new_model_size = 500_000_000  # 500M params
predicted_loss = power_law(new_model_size, a, alpha)
print(f"Predicted loss for 500M model: {predicted_loss:.4f}")
```

### 4. Compute-Optimal Training

Determine optimal model size for a given compute budget:

```python
def compute_optimal_allocation(compute_budget, alpha_N, alpha_D):
    """
    Given compute budget C and scaling exponents,
    find optimal N (params) and D (tokens).

    Chinchilla finding: N and D should scale proportionally with C.
    """
    # Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5
    N_opt = (compute_budget / constant_factor) ** 0.5
    D_opt = (compute_budget / constant_factor) ** 0.5

    return N_opt, D_opt
```

## Computational Requirements

### Hardware Requirements

| Experiment Scale | GPU | Training Time | Storage |
|-----------------|-----|---------------|---------|
| **Small-scale** (models up to 50M params) | RTX 3080/3090 | 1-5 hours per model | 10GB |
| **Medium-scale** (models up to 500M params) | A100 40GB | 5-20 hours per model | 50GB |
| **Large-scale** (models 1B+ params) | A100 80GB or multi-GPU | 1-3 days per model | 200GB+ |

### Recommended Experiment Grid

**Minimal Grid** (for quick iteration):
- Model sizes: 10M, 30M, 100M params
- Dataset sizes: 1M, 10M, 100M tokens
- Total experiments: 9 runs
- Estimated time: 5-10 hours on A100

**Comprehensive Grid** (for detailed analysis):
- Model sizes: 10M, 30M, 100M, 300M, 1B params
- Dataset sizes: 1M, 10M, 100M, 1B, 10B tokens
- Multiple seeds: 3 runs per configuration
- Total experiments: 75 runs
- Estimated time: 3-7 days on A100

### Compute Budget Estimation

For a single training run:
- **FLOPs**: ≈ 6 × N × D (forward + backward)
  - N = number of parameters
  - D = number of tokens
- **Example**: 100M param model on 1B tokens ≈ 6 × 10^8 × 10^9 = 6 × 10^17 FLOPs
- **A100 throughput**: ~300 TFLOPS → ~33 minutes

## Important Notes

### ⚠️ Implementation Limitations

1. **No Official API Access**: This assignment originally required access to a proprietary training API for running large-scale experiments. Since that API is unavailable:
   - **Alternative 1**: Use publicly available scaling law datasets/papers
   - **Alternative 2**: Run smaller-scale experiments with custom training scripts
   - **Alternative 3**: Use synthetic data for proof-of-concept analysis

2. **Verification Challenges**: Without official test cases, verify your implementation by:
   - Comparing trends with published scaling law papers (e.g., Chinchilla, Kaplan et al.)
   - Ensuring power-law fits make sense (exponents typically α ≈ 0.05-0.15)
   - Checking that larger models consistently achieve lower loss

3. **Resource Intensive**: Thorough scaling experiments require significant compute:
   - Consider starting with smaller models and datasets
   - Use learning rate sweeps to find optimal hyperparameters quickly
   - Leverage checkpointing to resume interrupted experiments

### 💡 Tips for Success

1. **Start Small**: Begin with a 3x3 grid (3 model sizes × 3 dataset sizes) to validate your setup.

2. **Fix Hyperparameters**: Keep learning rate, batch size, and other hyperparameters constant across experiments to isolate scaling effects.

3. **Use Log Scales**: Plot results on log-log axes to visualize power laws clearly.

4. **Track Everything**: Log all hyperparameters, random seeds, and environment details for reproducibility.

5. **Leverage Published Data**: Papers like Chinchilla and Kaplan et al. provide reference datasets you can use to validate your analysis methods.

6. **Automate Experiments**: Write scripts to automatically run the full experiment grid and collect results.

### 🔍 Expected Observations

Based on scaling law research, you should observe:

1. **Model Size Scaling**: Loss decreases as a power law with model size:
   - L(N) ∝ N^(-α) where α ≈ 0.05-0.15
   - Larger models consistently outperform smaller ones (given enough data)

2. **Data Size Scaling**: Loss decreases with dataset size:
   - L(D) ∝ D^(-β) where β ≈ 0.05-0.10
   - More data helps, but with diminishing returns

3. **Compute-Optimal Training** (Chinchilla Laws):
   - For a fixed compute budget C, optimal allocation scales both N and D with C
   - N_opt ∝ C^0.50, D_opt ∝ C^0.50
   - Training "small model for longer" is suboptimal compared to compute-optimal allocation

4. **Smooth Scaling**: Performance should scale smoothly without sudden jumps or drops.

### 📊 Analysis Deliverables

For this assignment, you typically need to produce:

1. **Scaling Curves**: Plots showing L(N), L(D), L(C) with fitted power laws
2. **Compute-Optimal Frontier**: Pareto frontier of best models at each compute budget
3. **Predictions**: Extrapolated performance for larger scales
4. **Report**: Analysis of findings, comparison with published literature

### 🔗 Reference Materials

**Key Papers**:
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla paper, 2022
- [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701) - Henighan et al., 2020

**Tools**:
- [WandB](https://wandb.ai/) - Experiment tracking
- [Matplotlib/Seaborn](https://matplotlib.org/) - Visualization
- [SciPy](https://scipy.org/) - Curve fitting

## Assignment Handout

For detailed assignment requirements and theoretical background, see:
- [cs336_spring2025_assignment3_scaling.pdf](./cs336_spring2025_assignment3_scaling.pdf)

## License

This code is provided for educational purposes as part of Stanford CS336.

---

# 中文版本 | Chinese Version

# CS336 作业 3: 缩放定律

本作业探讨语言模型中的实证缩放定律，研究模型性能如何随计算资源、模型大小和数据集大小变化。目标是理解和预测不同规模下的语言模型行为。

## 📋 目录
- [概述](#概述-1)
- [实现细节](#实现细节-1)
- [项目结构](#项目结构-1)
- [环境配置](#环境配置-1)
- [使用指南](#使用指南-1)
- [计算资源需求](#计算资源需求-1)
- [重要说明](#重要说明-1)

## 概述

缩放定律描述了语言模型性能（通过损失或困惑度衡量）如何随以下因素变化：

1. **模型大小** (N): 非嵌入参数数量
2. **数据集大小** (D): 训练 token 数量
3. **计算预算** (C): 训练使用的总浮点运算次数

本作业包括：
- 在不同数据集大小上训练各种规模的模型
- 测量最终损失和训练动态
- 将幂律关系拟合到实证数据
- 预测给定计算预算下的最优模型配置
- 理解计算最优训练（Chinchilla 缩放定律）

## 实现细节

### 核心组件

#### 1. Transformer 语言模型 (`cs336_scaling/model.py`)

标准的 Transformer 解码器实现，包含：
- **Token 和位置嵌入**: Token 和位置的可学习嵌入
- **多头自注意力**: 带 dropout 的因果注意力
- **前馈网络**: GELU 激活的 FFN
- **层归一化**: Pre-norm 架构
- **文本生成**: 支持温度/top-k 的自回归采样

**关键方法**:
- `forward()`: 标准前向传播，返回 logits
- `generate()`: 自回归文本生成
- `from_pretrained()`: 加载预训练模型
- `get_num_params()`: 统计非嵌入参数数量

#### 2. 缩放定律实验

本作业需要运行系统性实验以收集数据点：

**实验设计**:
- 训练不同层数/维度的模型
- 改变数据集大小（例如，1M、10M、100M、1B token）
- 在不同训练步骤测量损失
- 收集计算指标（FLOPs、训练时间）

**分析**:
- 拟合幂律: L(N) = aN^(-α), L(D) = bD^(-β), L(C) = cC^(-γ)
- 确定 N 和 D 之间的计算最优分配
- 预测未见配置的性能

## 项目结构

```
assignment3-scaling/
├── cs336_scaling/
│   ├── model.py               # Transformer 语言模型实现
│   └── __init__.py
├── data/                      # 训练数据集（用户提供）
├── experiments/               # 实验脚本和结果（用户创建）
├── cs336_spring2025_assignment3_scaling.pdf  # 作业说明
├── pyproject.toml             # 依赖项
├── uv.lock                    # 锁定文件
└── README.md                  # 本文件
```

**注意**: 本作业主要是实验/分析性的。你需要自行创建：
- 训练脚本
- 数据处理管道
- 实验跟踪代码
- 分析笔记本

## 环境配置

### 1. 安装依赖

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

### 2. 准备数据集

你需要各种大小的数据集。选项包括：

**选项 1: 使用公开数据集**
```bash
mkdir -p data

# TinyStories (~1GB)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -O data/tinystories.txt

# OpenWebText (~40GB)
# 从 https://huggingface.co/datasets/Skylion007/openwebtext 下载

# The Pile (825GB)
# 从 https://pile.eleuther.ai/ 下载
```

**选项 2: 创建合成数据集**
```python
# 生成受控大小的数据集
import random
import string

def generate_synthetic_text(num_tokens, vocab_size=10000):
    """为缩放实验生成随机文本"""
    tokens = [str(random.randint(0, vocab_size-1)) for _ in range(num_tokens)]
    return ' '.join(tokens)

# 创建 1M、10M、100M token 的数据集
for size in [1_000_000, 10_000_000, 100_000_000]:
    text = generate_synthetic_text(size)
    with open(f'data/synthetic_{size}.txt', 'w') as f:
        f.write(text)
```

## 使用指南

### 1. 定义模型配置

通过改变超参数创建不同大小的模型：

```python
from cs336_scaling.model import BasicsTransformerLM

# 小模型 (~10M 参数)
model_small = BasicsTransformerLM(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024
)

# 中型模型 (~50M 参数)
model_medium = BasicsTransformerLM(
    vocab_size=10000,
    context_length=512,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=2048
)

# 大模型 (~200M 参数)
model_large = BasicsTransformerLM(
    vocab_size=10000,
    context_length=1024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072
)
```

### 2. 运行缩放实验

收集缩放数据的示例训练脚本：

```python
import torch
import numpy as np
from cs336_scaling.model import BasicsTransformerLM

def train_and_measure(model, dataset, num_steps, compute_budget):
    """
    训练模型并在各个检查点测量损失。

    返回: 包含损失、计算使用量、最终指标的字典
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []

    for step in range(num_steps):
        # 训练步骤
        batch = get_batch(dataset)  # 你的数据加载逻辑
        logits = model(batch['input_ids'])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               batch['labels'].view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 定期记录损失
        if step % 100 == 0:
            losses.append(loss.item())

    return {
        'final_loss': losses[-1],
        'loss_curve': losses,
        'num_params': model.get_num_params(),
        'num_tokens': num_steps * batch_size * context_length
    }

# 运行实验
results = []
for model_config in [small_config, medium_config, large_config]:
    for dataset_size in [1e6, 1e7, 1e8]:
        model = BasicsTransformerLM(**model_config)
        result = train_and_measure(model, dataset_size, num_steps=10000)
        results.append(result)
```

### 3. 分析缩放定律

将幂律拟合到你的实验数据：

```python
import numpy as np
from scipy.optimize import curve_fit

# 幂律函数: L = a * N^(-alpha)
def power_law(x, a, alpha):
    return a * np.power(x, -alpha)

# 收集数据点
param_counts = [result['num_params'] for result in results]
final_losses = [result['final_loss'] for result in results]

# 拟合幂律
params, _ = curve_fit(power_law, param_counts, final_losses)
a, alpha = params

print(f"缩放定律: L(N) = {a:.4f} * N^(-{alpha:.4f})")

# 预测未见模型大小的损失
new_model_size = 500_000_000  # 500M 参数
predicted_loss = power_law(new_model_size, a, alpha)
print(f"500M 模型的预测损失: {predicted_loss:.4f}")
```

### 4. 计算最优训练

确定给定计算预算下的最优模型大小：

```python
def compute_optimal_allocation(compute_budget, alpha_N, alpha_D):
    """
    给定计算预算 C 和缩放指数，
    找到最优的 N（参数）和 D（token）。

    Chinchilla 发现: N 和 D 应与 C 成比例缩放。
    """
    # Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5
    N_opt = (compute_budget / constant_factor) ** 0.5
    D_opt = (compute_budget / constant_factor) ** 0.5

    return N_opt, D_opt
```

## 计算资源需求

### 硬件要求

| 实验规模 | GPU | 训练时间 | 存储 |
|-----------------|-----|---------------|---------:|
| **小规模** (最多 50M 参数的模型) | RTX 3080/3090 | 每个模型 1-5 小时 | 10GB |
| **中等规模** (最多 500M 参数的模型) | A100 40GB | 每个模型 5-20 小时 | 50GB |
| **大规模** (1B+ 参数的模型) | A100 80GB 或多 GPU | 每个模型 1-3 天 | 200GB+ |

### 推荐的实验网格

**最小网格**（用于快速迭代）:
- 模型大小: 10M、30M、100M 参数
- 数据集大小: 1M、10M、100M token
- 总实验数: 9 次运行
- 预计时间: 在 A100 上 5-10 小时

**综合网格**（用于详细分析）:
- 模型大小: 10M、30M、100M、300M、1B 参数
- 数据集大小: 1M、10M、100M、1B、10B token
- 多次运行: 每个配置 3 次
- 总实验数: 75 次运行
- 预计时间: 在 A100 上 3-7 天

### 计算预算估算

对于单次训练运行：
- **FLOPs**: ≈ 6 × N × D（前向 + 反向）
  - N = 参数数量
  - D = token 数量
- **示例**: 在 1B token 上训练 100M 参数模型 ≈ 6 × 10^8 × 10^9 = 6 × 10^17 FLOPs
- **A100 吞吐量**: 约 300 TFLOPS → 约 33 分钟

## 重要说明

### ⚠️ 实现限制

1. **无官方 API 访问**: 本作业最初需要访问专有训练 API 来运行大规模实验。由于该 API 不可用：
   - **替代方案 1**: 使用公开可用的缩放定律数据集/论文
   - **替代方案 2**: 使用自定义训练脚本运行较小规模的实验
   - **替代方案 3**: 使用合成数据进行概念验证分析

2. **验证挑战**: 没有官方测试用例，可通过以下方式验证你的实现：
   - 将趋势与已发表的缩放定律论文（例如 Chinchilla、Kaplan 等）进行比较
   - 确保幂律拟合合理（指数通常 α ≈ 0.05-0.15）
   - 检查较大模型是否始终获得较低损失

3. **资源密集**: 全面的缩放实验需要大量计算：
   - 考虑从较小的模型和数据集开始
   - 使用学习率扫描快速找到最优超参数
   - 利用检查点恢复中断的实验

### 💡 成功技巧

1. **从小处开始**: 从 3x3 网格（3 种模型大小 × 3 种数据集大小）开始验证你的设置。

2. **固定超参数**: 在实验中保持学习率、批次大小和其他超参数不变，以隔离缩放效应。

3. **使用对数刻度**: 在对数-对数坐标轴上绘制结果，以清晰地可视化幂律。

4. **跟踪一切**: 记录所有超参数、随机种子和环境详细信息以确保可重复性。

5. **利用已发表的数据**: Chinchilla 和 Kaplan 等论文提供了可用于验证分析方法的参考数据集。

6. **自动化实验**: 编写脚本自动运行完整的实验网格并收集结果。

### 🔍 预期观察

基于缩放定律研究，你应该观察到：

1. **模型大小缩放**: 损失随模型大小呈幂律下降：
   - L(N) ∝ N^(-α)，其中 α ≈ 0.05-0.15
   - 较大模型始终优于较小模型（给定足够的数据）

2. **数据大小缩放**: 损失随数据集大小下降：
   - L(D) ∝ D^(-β)，其中 β ≈ 0.05-0.10
   - 更多数据有帮助，但收益递减

3. **计算最优训练**（Chinchilla 定律）:
   - 对于固定的计算预算 C，最优分配使 N 和 D 都随 C 缩放
   - N_opt ∝ C^0.50, D_opt ∝ C^0.50
   - "长时间训练小模型"相比计算最优分配是次优的

4. **平滑缩放**: 性能应平滑缩放，没有突然的跳跃或下降。

### 📊 分析交付成果

对于本作业，你通常需要产出：

1. **缩放曲线**: 显示 L(N)、L(D)、L(C) 及拟合幂律的图表
2. **计算最优前沿**: 每个计算预算下最佳模型的 Pareto 前沿
3. **预测**: 对更大规模的外推性能
4. **报告**: 分析发现，与已发表文献的比较

### 🔗 参考资料

**关键论文**:
- [神经语言模型的缩放定律](https://arxiv.org/abs/2001.08361) - Kaplan 等，2020
- [训练计算最优的大型语言模型](https://arxiv.org/abs/2203.15556) - Chinchilla 论文，2022
- [自回归生成建模的缩放定律](https://arxiv.org/abs/2010.14701) - Henighan 等，2020

**工具**:
- [WandB](https://wandb.ai/) - 实验跟踪
- [Matplotlib/Seaborn](https://matplotlib.org/) - 可视化
- [SciPy](https://scipy.org/) - 曲线拟合

## 作业说明

详细的作业要求和理论背景，请参阅：
- [cs336_spring2025_assignment3_scaling.pdf](./cs336_spring2025_assignment3_scaling.pdf)

## 许可证

本代码仅供教育目的使用，是斯坦福 CS336 课程的一部分。
