# CS336 Assignment 1: Language Modeling Basics

This assignment implements the foundational components of a Transformer-based language model from scratch, including tokenization, model architecture, training infrastructure, and text generation.

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

This assignment covers the fundamental building blocks of language modeling:

1. **Tokenization**: Byte Pair Encoding (BPE) implementation
2. **Model Architecture**: Multi-layer Transformer decoder with:
   - Multi-head self-attention with RoPE (Rotary Position Embeddings)
   - Feed-forward networks with SwiGLU activation
   - Layer normalization (RMSNorm)
3. **Training Infrastructure**:
   - AdamW optimizer with weight decay
   - Learning rate scheduling (warmup + cosine decay)
   - Gradient clipping
   - Checkpointing
4. **Text Generation**: Auto-regressive sampling with temperature control

## Implementation Details

### Key Components

#### 1. Tokenizer (`cs336_basics/tokenizer.py`)
- **BPE Training**: Implements byte-pair encoding from scratch
- **Pre-tokenization**: Supports GPT-4 style regex-based pre-tokenization
- **Special Tokens**: Handles `<|endoftext|>` and other special tokens
- **Encoding/Decoding**: Efficient token sequence conversion

**Note**: ⚠️ CPU parallel tokenizer training is **not implemented**. Training runs on a single process.

#### 2. Transformer Model (`cs336_basics/model/`)
- **Attention**: Multi-head self-attention with causal masking
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Activation**: SwiGLU activation function
- **Normalization**: RMSNorm for improved training stability

#### 3. Training (`cs336_basics/train.py`)
- **Optimizer**: Custom AdamW implementation
- **LR Schedule**: Linear warmup + cosine annealing
- **Monitoring**: Integrated W&B logging and tqdm progress bars
- **Checkpointing**: Automatic model checkpointing with resume capability

#### 4. Generation (`cs336_basics/generate.py`)
- Temperature-controlled sampling
- Top-k and top-p (nucleus) sampling support
- Batch generation capabilities

## Project Structure

```
assignment1-basics/
├── cs336_basics/              # Main implementation module
│   ├── model/
│   │   ├── modules.py         # Attention, FFN, LayerNorm modules
│   │   └── transformer.py     # Full Transformer LM
│   ├── trainer/
│   │   ├── AdamW.py          # AdamW optimizer
│   │   ├── data_loading.py   # Data loading utilities
│   │   └── utils.py          # Training utilities (loss, lr schedule, clipping)
│   ├── tokenizer.py          # BPE tokenizer implementation
│   ├── train.py              # Training script
│   ├── generate.py           # Text generation utilities
│   └── check_pointing.py     # Checkpoint save/load
├── scripts/
│   ├── train_bpe.py          # Script to train BPE tokenizer
│   └── tokenize_test.py      # Tokenizer testing script
├── tests/
│   ├── adapters.py           # Test adapters (IMPORTANT: Connect your implementation here)
│   ├── test_tokenizer.py    # Tokenizer tests
│   ├── test_model.py         # Model architecture tests
│   ├── test_optimizer.py    # Optimizer tests
│   └── ...                   # Other test files
├── tokenizer/                # Saved tokenizer files (generated)
├── checkpoints/              # Model checkpoints (generated)
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Setup

### 1. Install Dependencies

This project uses `uv` for dependency management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (automatic with uv run)
uv sync
```

### 2. Download Training Data

Download the TinyStories and OpenWebText datasets:

```bash
mkdir -p data
cd data

# TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Usage

### Step 1: Train BPE Tokenizer

First, train a BPE tokenizer on your corpus:

```bash
uv run python scripts/train_bpe.py
```

This will:
- Train a BPE tokenizer with vocab size 10,000 on TinyStories
- Save vocabulary and merge rules to `tokenizer/` directory
- Takes approximately **5-15 minutes** on CPU (single-threaded)

**Configuration** (edit `scripts/train_bpe.py`):
- `vocab_size`: Target vocabulary size (default: 10,000)
- `special_tokens`: List of special tokens (default: `["<|endoftext|>"]`)
- `INPUT_PATH`: Path to training corpus

### Step 2: Tokenize Data

After training the tokenizer, convert your text data to token sequences:

```bash
# Create tokenized .dat files from text
uv run python scripts/tokenize_test.py
```

This creates memory-mapped `.dat` files for efficient training:
- `data/train.dat`: Tokenized training data
- `data/valid.dat`: Tokenized validation data

### Step 3: Train Language Model

Train the Transformer language model:

```bash
# Basic training (with W&B logging)
uv run python cs336_basics/train.py \
    --data_dir ./data \
    --wandb_project "cs336-basics" \
    --wandb_run_name "tinystories-baseline"

# Training without W&B
uv run python cs336_basics/train.py \
    --data_dir ./data \
    --no_wandb

# Custom configuration
uv run python cs336_basics/train.py \
    --data_dir ./data \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 64 \
    --train_steps 10000 \
    --max_lr 3e-4 \
    --no_wandb
```

**Key Training Arguments**:
- Model: `--d_model`, `--num_layers`, `--num_heads`, `--d_ff`
- Training: `--batch_size`, `--train_steps`, `--max_lr`, `--weight_decay`
- Monitoring: `--val_interval`, `--save_intervals`, `--log_intervals`
- Checkpointing: `--save_ckp_path`, `--resume_ckp`

**Resume from Checkpoint**:
```bash
uv run python cs336_basics/train.py \
    --data_dir ./data \
    --resume_ckp ./checkpoints/checkpoint_5000.pt
```

### Step 4: Generate Text

Generate text using a trained model:

```python
from cs336_basics.generate import generate_text
from cs336_basics.model.transformer import transformer_lm
import torch

# Load model
model = transformer_lm(vocab_size=10000, ...)
checkpoint = torch.load('checkpoints/checkpoint_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate
prompt = "Once upon a time"
generated = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8)
print(generated)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_model.py
uv run pytest tests/test_optimizer.py

# Run with verbose output
uv run pytest -v
```

**Important**: Before running tests, you must complete the adapter functions in `tests/adapters.py`. This file connects your implementation to the test suite.

### Test Coverage

- ✅ Tokenizer: BPE training, encoding, decoding, special tokens
- ✅ Model: Attention, FFN, layer norm, full transformer forward pass
- ✅ Optimizer: AdamW correctness, weight decay, learning rate
- ✅ Training Utils: Cross-entropy, gradient clipping, LR scheduling
- ✅ Serialization: Checkpoint save/load

## Computational Requirements

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | None (CPU works) | NVIDIA GPU with 8GB+ VRAM |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 10GB | 20GB+ |

### Training Time Estimates

**Tokenizer Training** (BPE on TinyStories, vocab_size=10,000):
- CPU (single-thread): ~10-15 minutes
- Note: Parallel training not implemented

**Model Training** (Default config: 4 layers, d_model=512, 6K steps):
- **CPU**: ~8-12 hours (not recommended)
- **Apple M1/M2 (MPS)**: ~2-3 hours
- **NVIDIA RTX 3090**: ~45-60 minutes
- **NVIDIA A100**: ~20-30 minutes

**Larger Model** (6 layers, d_model=768, 20K steps):
- **RTX 3090**: ~3-4 hours
- **A100**: ~1.5-2 hours

### Memory Usage

- **Small Model** (4 layers, d_model=512, batch_size=32): ~2-3GB GPU memory
- **Medium Model** (6 layers, d_model=768, batch_size=64): ~6-8GB GPU memory
- **Large Model** (12 layers, d_model=1024, batch_size=64): ~12-16GB GPU memory

**Tip**: Reduce `batch_size` if you encounter OOM errors.

## Important Notes

### ⚠️ Implementation Limitations

1. **No CPU Parallelization**: The BPE tokenizer training runs on a single CPU thread. Parallel training is not implemented.

2. **Memory-Mapped Data**: Training uses `np.memmap` for efficient data loading. Ensure `.dat` files are created before training.

3. **Device Compatibility**: Supports CUDA, MPS (Apple Silicon), and CPU. Auto-detection available with `--device auto`.

### 💡 Tips for Success

1. **Start Small**: Begin with the default configuration (4 layers, d_model=512) to verify everything works.

2. **Monitor Training**: Use W&B (`--wandb_project`) to track loss curves and learning rate schedules.

3. **Validate Frequently**: Set `--val_interval=100` to catch training issues early.

4. **Checkpoint Often**: Use `--save_intervals=1000` to avoid losing progress.

5. **Hyperparameter Tuning**:
   - Learning rate is critical: try `[1e-4, 3e-4, 1e-3]`
   - Warmup helps: use `--warm_up_it=500` for stable training
   - Gradient clipping prevents explosions: keep `--clip_grad_norm=1.0`

6. **Test First**: Run `uv run pytest` to ensure your implementation is correct before long training runs.

### 🔍 Debugging

**Tests Failing?**
- Check `tests/adapters.py` - all adapter functions must be implemented
- Ensure tokenizer is trained and saved to `tokenizer/` directory
- Verify data files exist in `data/` directory

**Training Issues?**
- Loss = NaN: Lower learning rate or increase gradient clipping
- Loss not decreasing: Check data loading, verify tokenization
- OOM errors: Reduce `--batch_size` or `--context_len`

**Slow Training?**
- Verify GPU is being used: check device output at start
- For Apple Silicon: ensure MPS backend is available (`torch.backends.mps.is_available()`)
- Reduce model size or batch size for faster iteration

## Assignment Handout

For detailed assignment requirements and theoretical background, see:
- [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE explanation
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU activation

## License

This code is provided for educational purposes as part of Stanford CS336.

---

# 中文版本 | Chinese Version

# CS336 作业1：语言建模基础

本作业从零实现基于Transformer的语言模型的基础组件，包括分词、模型架构、训练基础设施和文本生成。

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

本作业涵盖语言建模的基础构建块：

1. **分词**：字节对编码（BPE）实现
2. **模型架构**：多层Transformer解码器，包含：
   - 带RoPE（旋转位置嵌入）的多头自注意力
   - 带SwiGLU激活的前馈网络
   - 层归一化（RMSNorm）
3. **训练基础设施**：
   - 带权重衰减的AdamW优化器
   - 学习率调度（预热+余弦衰减）
   - 梯度裁剪
   - 检查点保存
4. **文本生成**：带温度控制的自回归采样

## 实现细节

### 核心组件

#### 1. 分词器 (`cs336_basics/tokenizer.py`)
- **BPE训练**：从零实现字节对编码
- **预分词**：支持GPT-4风格的正则表达式预分词
- **特殊token**：处理`<|endoftext|>`等特殊token
- **编码/解码**：高效的token序列转换

**注意**：⚠️ **未实现**CPU并行分词器训练。训练在单进程上运行。

#### 2. Transformer模型 (`cs336_basics/model/`)
- **注意力**：带因果掩码的多头自注意力
- **位置编码**：RoPE（旋转位置嵌入）
- **激活函数**：SwiGLU激活函数
- **归一化**：RMSNorm提升训练稳定性

#### 3. 训练 (`cs336_basics/train.py`)
- **优化器**：自定义AdamW实现
- **学习率调度**：线性预热+余弦退火
- **监控**：集成W&B日志和tqdm进度条
- **检查点**：自动模型检查点保存和恢复

#### 4. 生成 (`cs336_basics/generate.py`)
- 温度控制采样
- Top-k和top-p（nucleus）采样支持
- 批量生成能力

## 项目结构

```
assignment1-basics/
├── cs336_basics/              # 主要实现模块
│   ├── model/
│   │   ├── modules.py         # 注意力、FFN、LayerNorm模块
│   │   └── transformer.py     # 完整Transformer LM
│   ├── trainer/
│   │   ├── AdamW.py          # AdamW优化器
│   │   ├── data_loading.py   # 数据加载工具
│   │   └── utils.py          # 训练工具（损失、学习率调度、裁剪）
│   ├── tokenizer.py          # BPE分词器实现
│   ├── train.py              # 训练脚本
│   ├── generate.py           # 文本生成工具
│   └── check_pointing.py     # 检查点保存/加载
├── scripts/
│   ├── train_bpe.py          # BPE分词器训练脚本
│   └── tokenize_test.py      # 分词器测试脚本
├── tests/                     # 测试目录
├── tokenizer/                 # 保存的分词器文件（生成）
├── checkpoints/               # 模型检查点（生成）
└── README.md                  # 本文件
```

## 环境配置

### 1. 安装依赖

本项目使用 `uv` 进行依赖管理：

```bash
# 如果还没有安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖（使用uv run时自动安装）
uv sync
```

### 2. 下载训练数据

下载TinyStories和OpenWebText数据集：

```bash
mkdir -p data
cd data

# TinyStories数据集
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText样本
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 使用指南

### 步骤1：训练BPE分词器

首先在语料库上训练BPE分词器：

```bash
uv run python scripts/train_bpe.py
```

这将：
- 在TinyStories上训练词汇量为10,000的BPE分词器
- 保存词汇表和合并规则到`tokenizer/`目录
- 大约需要**5-15分钟**（CPU单线程）

### 步骤2：分词数据

训练分词器后，将文本数据转换为token序列：

```bash
uv run python scripts/tokenize_test.py
```

这会创建内存映射的`.dat`文件用于高效训练：
- `data/train.dat`：分词后的训练数据
- `data/valid.dat`：分词后的验证数据

### 步骤3：训练语言模型

训练Transformer语言模型：

```bash
# 基础训练（带W&B日志）
uv run python cs336_basics/train.py \
    --data_dir ./data \
    --wandb_project "cs336-basics" \
    --wandb_run_name "tinystories-baseline"

# 不使用W&B的训练
uv run python cs336_basics/train.py \
    --data_dir ./data \
    --no_wandb
```

**关键训练参数**：
- 模型：`--d_model`、`--num_layers`、`--num_heads`、`--d_ff`
- 训练：`--batch_size`、`--train_steps`、`--max_lr`、`--weight_decay`
- 监控：`--val_interval`、`--save_intervals`、`--log_intervals`

## 测试

运行完整测试套件：

```bash
# 运行所有测试
uv run pytest

# 运行特定测试模块
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_model.py
```

**重要**：运行测试前，必须完成`tests/adapters.py`中的适配器函数。

## 计算资源需求

### 硬件推荐

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 无（CPU可运行） | NVIDIA GPU 8GB+ VRAM |
| **内存** | 8GB | 16GB+ |
| **存储** | 10GB | 20GB+ |

### 训练时间估计

**分词器训练**（TinyStories上的BPE，词汇量=10,000）：
- CPU（单线程）：约10-15分钟
- 注意：未实现并行训练

**模型训练**（默认配置：4层，d_model=512，6K步）：
- **CPU**：约8-12小时（不推荐）
- **Apple M1/M2 (MPS)**：约2-3小时
- **NVIDIA RTX 3090**：约45-60分钟
- **NVIDIA A100**：约20-30分钟

### 内存使用

- **小模型**（4层，d_model=512，batch_size=32）：约2-3GB GPU内存
- **中模型**（6层，d_model=768，batch_size=64）：约6-8GB GPU内存
- **大模型**（12层，d_model=1024，batch_size=64）：约12-16GB GPU内存

**提示**：如遇OOM错误，降低`batch_size`。

## 重要说明

### ⚠️ 实现限制

1. **无CPU并行化**：BPE分词器训练在单个CPU线程上运行。未实现并行训练。

2. **内存映射数据**：训练使用`np.memmap`进行高效数据加载。训练前确保创建了`.dat`文件。

3. **设备兼容性**：支持CUDA、MPS（Apple Silicon）和CPU。使用`--device auto`自动检测。

### 💡 成功技巧

1. **从小开始**：先使用默认配置（4层，d_model=512）验证一切正常。

2. **监控训练**：使用W&B（`--wandb_project`）跟踪损失曲线和学习率调度。

3. **频繁验证**：设置`--val_interval=100`及早发现训练问题。

4. **经常保存检查点**：使用`--save_intervals=1000`避免丢失进度。

5. **超参数调优**：
   - 学习率至关重要：尝试`[1e-4, 3e-4, 1e-3]`
   - 预热有帮助：使用`--warm_up_it=500`稳定训练
   - 梯度裁剪防止爆炸：保持`--clip_grad_norm=1.0`

### 🔍 调试

**测试失败？**
- 检查`tests/adapters.py` - 所有适配器函数必须实现
- 确保分词器已训练并保存到`tokenizer/`目录
- 验证数据文件存在于`data/`目录

**训练问题？**
- Loss = NaN：降低学习率或增加梯度裁剪
- Loss不下降：检查数据加载，验证分词
- OOM错误：减少`--batch_size`或`--context_len`

## 作业说明

详细的作业要求和理论背景，请参阅：
- [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

## 许可证

本代码作为Stanford CS336的一部分，仅供教育目的使用。
