# CS336: Language Modeling from Scratch

This repository contains my implementations for all assignments from Stanford's CS336: Language Modeling from Scratch course (Spring 2025). The course covers the fundamentals of building, training, and deploying large language models.

## Course Overview

CS336 provides a comprehensive, hands-on approach to understanding language models by implementing them from the ground up. Topics include model architecture, systems optimization, scaling laws, data processing, and alignment techniques.

## Assignments

### [Assignment 1: Basics](./assignment1-basics)
**Core Topics:** Tokenization, Transformer Architecture, Language Model Training

Implementation of fundamental language modeling components:
- Byte Pair Encoding (BPE) tokenization
- Multi-head self-attention mechanisms
- Transformer decoder architecture
- Basic training loops and optimization
- Training on TinyStories and OpenWebText datasets

> **Note:** CPU parallel tokenizer training is not implemented in this assignment.

### [Assignment 2: Systems](./assignment2-systems)
**Core Topics:** Performance Optimization, Distributed Training, Systems Engineering

Deep dive into efficient LM training systems:
- Flash Attention and kernel-level optimizations
- Mixed precision training (fp16/bf16)
- Memory profiling and optimization
- Distributed data parallelism
- Model parallelism techniques

> **Note:** Distributed parallel training components are not implemented in this assignment.

### [Assignment 3: Scaling](./assignment3-scaling)
**Core Topics:** Scaling Laws, Compute-Optimal Training, Model Scaling

Exploration of scaling behaviors in language models:
- Empirical investigation of scaling laws
- Compute-optimal training strategies
- Model size vs. training data trade-offs
- Performance prediction across scales

> **Note:** Due to the lack of official API access, this assignment could not be fully tested. It is recommended to use custom synthetic datasets or publicly available datasets for verification.

### [Assignment 4: Data](./assignment4-data)
**Core Topics:** Data Curation, Quality Filtering, Dataset Construction

Building high-quality training datasets:
- Web scraping and data collection
- Content filtering and deduplication
- Quality classification models
- Data pipeline construction
- Training on curated datasets

> **Note:** This assignment was completed using smaller datasets. Large-scale dataset training was not performed.

### [Assignment 5: Alignment](./assignment5-alignment)
**Core Topics:** RLHF, Preference Learning, Model Safety

Aligning language models with human preferences:
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- Reward modeling
- Safety alignment techniques
- Instruction tuning

> **Note:** The supplemental assignment on safety RLHF has not been implemented yet. Stay tuned for future updates!

## Repository Structure

```
.
├── assignment1-basics/      # Fundamental LM implementation
├── assignment2-systems/     # Systems optimization
├── assignment3-scaling/     # Scaling law investigations
├── assignment4-data/        # Data processing pipeline
├── assignment5-alignment/   # Alignment techniques
└── README.md               # This file
```

Each assignment directory contains:
- Complete source code implementation
- Unit tests
- Assignment handout (PDF)
- Assignment-specific README with setup instructions

## Technical Stack

- **Language:** Python 3.11+
- **Deep Learning:** PyTorch
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Testing:** pytest
- **Key Libraries:**
  - `torch` - Deep learning framework
  - `flash-attn` - Optimized attention implementation
  - `einops` - Tensor operations
  - `wandb` - Experiment tracking

## Setup

### Prerequisites

1. Install `uv` for dependency management:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via pip
pip install uv

# or via Homebrew
brew install uv
```

2. Python 3.11 or higher

### Running Individual Assignments

Each assignment is self-contained. Navigate to the assignment directory and use `uv run`:

```bash
cd assignment1-basics
uv run pytest                    # Run tests
uv run python scripts/train.py   # Run training script
```

Dependencies are automatically installed based on each assignment's `pyproject.toml`.

## Key Learning Outcomes

Through these assignments, I gained practical experience in:

1. **Model Architecture:** Implementing Transformers from scratch, understanding attention mechanisms and layer normalization
2. **Training Infrastructure:** Building robust training loops, implementing gradient accumulation and mixed precision training
3. **Systems Optimization:** Profiling memory usage, optimizing compute kernels, implementing distributed training
4. **Scaling Principles:** Understanding how model performance scales with compute, data, and parameters
5. **Data Engineering:** Constructing high-quality datasets through filtering, deduplication, and quality assessment
6. **Alignment Techniques:** Implementing RLHF and preference optimization to align models with human values

## Course Information

- **Course:** CS336 - Language Modeling from Scratch
- **Institution:** Stanford University
- **Term:** Spring 2025
- **Course Website:** [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)

## Notes

- Large files (datasets, model checkpoints, trained models) have been excluded from this repository
- Each assignment includes comprehensive unit tests to verify implementations
- For detailed assignment requirements, refer to the PDF handouts in each assignment directory

## License

This code is provided for educational purposes. Please refer to individual assignment directories for specific licensing information.

---

# 中文版本 | Chinese Version

# CS336: 从零开始的语言模型

本仓库包含我完成的斯坦福大学 CS336《从零开始的语言模型》课程（2025年春季）的所有作业实现。该课程涵盖了构建、训练和部署大型语言模型的基础知识。

## 课程概述

CS336 提供了一种全面的实践方法来理解语言模型，通过从头实现来深入掌握。主题包括模型架构、系统优化、缩放定律、数据处理和对齐技术。

## 作业列表

### [作业 1: 基础](./assignment1-basics)
**核心主题：** 分词、Transformer架构、语言模型训练

实现语言建模的基础组件：
- 字节对编码（BPE）分词
- 多头自注意力机制
- Transformer解码器架构
- 基础训练循环和优化
- 在TinyStories和OpenWebText数据集上训练

> **注意：** 本作业未实现CPU并行分词器训练。

### [作业 2: 系统优化](./assignment2-systems)
**核心主题：** 性能优化、分布式训练、系统工程

深入研究高效的语言模型训练系统：
- Flash Attention和内核级优化
- 混合精度训练（fp16/bf16）
- 内存分析和优化
- 分布式数据并行
- 模型并行技术

> **注意：** 本作业未实现分布式并行训练组件。

### [作业 3: 缩放定律](./assignment3-scaling)
**核心主题：** 缩放定律、计算最优训练、模型扩展

探索语言模型中的缩放行为：
- 缩放定律的实证研究
- 计算最优训练策略
- 模型大小与训练数据的权衡
- 跨规模的性能预测

> **注意：** 由于缺乏官方API访问权限，本作业无法完全测试。建议使用自定义合成数据集或公开可用的数据集进行验证。

### [作业 4: 数据处理](./assignment4-data)
**核心主题：** 数据管理、质量过滤、数据集构建

构建高质量的训练数据集：
- 网页抓取和数据收集
- 内容过滤和去重
- 质量分类模型
- 数据pipeline构建
- 在精选数据集上训练

> **注意：** 本作业使用较小的数据集完成。未进行大规模数据集训练。

### [作业 5: 模型对齐](./assignment5-alignment)
**核心主题：** RLHF、偏好学习、模型安全

使语言模型与人类偏好对齐：
- 直接偏好优化（DPO）
- 组相对策略优化（GRPO）
- 奖励建模
- 安全对齐技术
- 指令调优

> **注意：** 安全RLHF补充作业尚未实现。敬请期待后续更新！

## 仓库结构

```
.
├── assignment1-basics/      # 基础语言模型实现
├── assignment2-systems/     # 系统优化
├── assignment3-scaling/     # 缩放定律研究
├── assignment4-data/        # 数据处理pipeline
├── assignment5-alignment/   # 对齐技术
└── README.md               # 本文件
```

每个作业目录包含：
- 完整的源代码实现
- 单元测试
- 作业说明（PDF）
- 作业专属的README及安装说明

## 技术栈

- **编程语言：** Python 3.11+
- **深度学习：** PyTorch
- **依赖管理：** [uv](https://github.com/astral-sh/uv)
- **测试：** pytest
- **关键库：**
  - `torch` - 深度学习框架
  - `flash-attn` - 优化的注意力实现
  - `einops` - 张量操作
  - `wandb` - 实验跟踪

## 环境配置

### 前置要求

1. 安装 `uv` 进行依赖管理：
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或通过 pip
pip install uv

# 或通过 Homebrew
brew install uv
```

2. Python 3.11 或更高版本

### 运行单个作业

每个作业都是独立的。进入作业目录并使用 `uv run`：

```bash
cd assignment1-basics
uv run pytest                    # 运行测试
uv run python scripts/train.py   # 运行训练脚本
```

依赖项会根据每个作业的 `pyproject.toml` 自动安装。

## 核心学习成果

通过这些作业，我获得了以下实践经验：

1. **模型架构：** 从零实现Transformers，理解注意力机制和层归一化
2. **训练基础设施：** 构建健壮的训练循环，实现梯度累积和混合精度训练
3. **系统优化：** 分析内存使用，优化计算内核，实现分布式训练
4. **缩放原理：** 理解模型性能如何随计算、数据和参数规模变化
5. **数据工程：** 通过过滤、去重和质量评估构建高质量数据集
6. **对齐技术：** 实现RLHF和偏好优化，使模型与人类价值观对齐

## 课程信息

- **课程：** CS336 - 从零开始的语言模型
- **院校：** 斯坦福大学
- **学期：** 2025年春季
- **课程网站：** [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)

## 说明

- 本仓库已排除大型文件（数据集、模型检查点、训练好的模型）
- 每个作业都包含全面的单元测试以验证实现
- 详细的作业要求请参考各作业目录中的PDF文档

## 许可证

本代码仅供教育目的使用。具体许可信息请参考各作业目录。
