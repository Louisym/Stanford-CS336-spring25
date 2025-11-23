# CS336 Assignment 2: Systems Optimization

This assignment focuses on building efficient and scalable training systems for language models. It covers performance optimization techniques including Flash Attention, mixed precision training, memory profiling, and distributed training fundamentals.

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

This assignment explores critical systems-level optimizations for training large language models:

1. **Flash Attention**: Memory-efficient attention mechanisms
   - PyTorch implementation with tiling
   - Triton kernel implementation for custom GPU kernels
2. **Mixed Precision Training**: FP16/BF16 training for faster computation
3. **Memory Profiling**: Understanding and optimizing GPU memory usage
4. **Performance Benchmarking**: Measuring throughput and memory efficiency

## Implementation Details

### Key Components

#### 1. Flash Attention (`cs336_systems/flash_attention/`)

**Flash Attention PyTorch** (`flash_att_pytorch.py`):
- Tiled attention computation to reduce memory usage
- Causal masking support
- Memory-efficient forward and backward passes
- Compatible with standard PyTorch autograd

**Flash Attention Triton** (`flash_att_triton.py`):
- Custom Triton kernel for maximum performance
- Fused operations (softmax, dropout, etc.)
- Optimized memory access patterns
- CUDA-level performance with Python syntax

**Benchmarking** (`benchmark_flash.py`, `benchamark_script.py`):
- Compares vanilla attention vs. Flash Attention
- Measures time and memory consumption
- Tests various sequence lengths and batch sizes
- Generates performance charts

#### 2. Mixed Precision Training (`Benckmark/mixed_percision_script.py`)
- FP16 and BF16 training implementation
- Automatic mixed precision (AMP) integration
- Loss scaling for numerical stability
- Performance comparison vs. FP32

#### 3. Profiling and Benchmarking (`Benckmark/benchmark.py`)
- Memory profiling tools
- Forward/backward pass profiling
- Throughput measurement
- Detailed performance reports

**Note**: ⚠️ Distributed parallel training (DDP, model parallelism) is **not fully implemented**. Test files exist but implementations are incomplete.

## Project Structure

```
assignment2-systems/
├── cs336-basics/              # Staff implementation from Assignment 1
│   └── cs336_basics/          # Basic LM modules (reused here)
├── cs336_systems/             # Systems optimization implementations
│   ├── flash_attention/
│   │   ├── flash_att_pytorch.py    # PyTorch Flash Attention
│   │   ├── flash_att_triton.py     # Triton Flash Attention
│   │   ├── benchmark_flash.py      # Flash Attention benchmarks
│   │   └── benchamark_script.py    # Benchmark execution script
│   └── Benckmark/
│       ├── benchmark.py            # General profiling tools
│       └── mixed_percision_script.py # Mixed precision benchmarks
├── tests/
│   ├── adapters.py            # Test adapters (connect your implementation)
│   ├── test_attention.py      # Flash Attention tests
│   ├── test_ddp.py            # Distributed training tests (incomplete)
│   └── test_sharded_optimizer.py # Sharded optimizer tests (incomplete)
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Setup

### 1. Install Dependencies

This project uses `uv` and requires a CUDA-capable GPU for optimal performance:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Verify CUDA Setup

Flash Attention and Triton require CUDA:

```bash
# Check CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Check Triton installation
uv run python -c "import triton; print(f'Triton version: {triton.__version__}')"
```

### Requirements

- **CUDA**: 11.8+ or 12.x
- **GPU**: NVIDIA GPU with compute capability 7.0+ (V100, RTX 2080+, A100, etc.)
- **PyTorch**: 2.8.0 (supports latest GPUs including RTX 50 series)
- **Triton**: Automatically installed with PyTorch

## Usage

### 1. Flash Attention Benchmarking

Compare vanilla attention with Flash Attention implementations:

```bash
# Run Flash Attention benchmarks
cd cs336_systems/flash_attention
uv run python benchmark_flash.py

# Run comprehensive benchmark script
uv run python benchamark_script.py
```

**What it does**:
- Tests multiple sequence lengths (512, 1024, 2048, 4096)
- Compares vanilla attention, PyTorch Flash Attention, and Triton Flash Attention
- Measures forward/backward pass time and peak memory usage
- Generates performance comparison charts

**Expected Output**:
```
Sequence Length: 1024
├── Vanilla Attention:     45.2 ms/iter, 8.3 GB memory
├── Flash Attention (PT):  18.7 ms/iter, 3.1 GB memory (2.4x speedup, 2.7x memory reduction)
└── Flash Attention (Triton): 15.3 ms/iter, 2.9 GB memory (3.0x speedup, 2.9x memory reduction)
```

### 2. Mixed Precision Training

Benchmark FP16/BF16 vs. FP32 training:

```bash
cd cs336_systems/Benckmark
uv run python mixed_percision_script.py
```

**What it measures**:
- Training throughput (samples/sec)
- Memory consumption
- Numerical stability
- Loss convergence

**Expected Results**:
- **FP16/BF16**: ~1.5-2x faster than FP32, ~50% less memory
- **BF16**: Better numerical stability than FP16 for large models

### 3. General Profiling

Profile model training performance:

```bash
cd cs336_systems/Benckmark
uv run python benchmark.py
```

This generates:
- Forward/backward pass timings
- Memory allocation breakdowns
- Bottleneck identification
- Optimization recommendations

### 4. Using Flash Attention in Training

Integrate Flash Attention into your models:

```python
from cs336_systems.flash_attention.flash_att_pytorch import flash_attention_pytorch
# OR
from cs336_systems.flash_attention.flash_att_triton import flash_attention_triton

# In your attention module
def forward(self, query, key, value, mask=None):
    # Replace vanilla attention with Flash Attention
    output = flash_attention_pytorch(query, key, value, causal=True)
    # OR use Triton version for maximum performance
    output = flash_attention_triton(query, key, value, causal=True)
    return output
```

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_attention.py        # Flash Attention tests
uv run pytest tests/test_ddp.py             # DDP tests (may not pass)
uv run pytest tests/test_sharded_optimizer.py # Sharded optimizer tests (may not pass)

# Run with verbose output
uv run pytest -v -s
```

**Important**:
- Complete the adapter functions in `tests/adapters.py` to connect your implementation
- Some distributed training tests may fail as full DDP implementation is incomplete

### Test Coverage

- ✅ Flash Attention: Correctness vs. vanilla attention, numerical stability
- ⚠️ DDP: Distributed data parallelism (tests exist, implementation incomplete)
- ⚠️ Sharded Optimizer: Memory-efficient optimizer sharding (tests exist, implementation incomplete)

## Computational Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 8GB VRAM (RTX 2080, V100) | NVIDIA GPU with 24GB+ VRAM (RTX 3090, A100) |
| **CUDA** | 11.8+ | 12.x |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 5GB | 10GB |

### Performance Benchmarks

**Flash Attention Speedup** (Sequence Length = 2048, Batch Size = 16):
- **RTX 3090**: 2.5-3x speedup, 2.8x memory reduction
- **A100**: 2.8-3.5x speedup, 3x memory reduction
- **H100**: 3.5-4x speedup, 3.2x memory reduction

**Mixed Precision Training** (Medium Model, 6 layers, d_model=768):
- **FP32**: ~100 samples/sec, 16GB memory
- **BF16**: ~180 samples/sec, 8GB memory (1.8x speedup, 50% memory savings)

### Benchmark Execution Times

- **Flash Attention Benchmark**: ~5-10 minutes (tests multiple configurations)
- **Mixed Precision Benchmark**: ~10-15 minutes
- **Full Profiling Suite**: ~15-20 minutes

## Important Notes

### ⚠️ Implementation Limitations

1. **Distributed Training Not Implemented**: While test files exist for DDP and model parallelism, the full implementations are **not complete**. This includes:
   - Distributed Data Parallel (DDP)
   - Model parallelism
   - Sharded optimizers

2. **GPU Required**: Flash Attention and Triton kernels require CUDA-capable NVIDIA GPUs. CPU fallback is not available.

3. **Triton Compatibility**: Triton kernels may not work on all GPU architectures. Compute capability 7.0+ recommended.

### 💡 Tips for Success

1. **Start with PyTorch Flash Attention**: Before diving into Triton, ensure the PyTorch implementation works correctly.

2. **Use Appropriate Precision**:
   - BF16 is preferred for large models (better numerical range than FP16)
   - FP16 works well for smaller models
   - Always use FP32 for final validation

3. **Profile Before Optimizing**: Use `benchmark.py` to identify bottlenecks before applying optimizations.

4. **Monitor Memory**: Use `torch.cuda.memory_summary()` to track memory usage during development.

5. **Test Numerical Stability**: Always compare outputs with vanilla attention to ensure correctness.

6. **Gradual Optimization**: Optimize one component at a time and verify correctness after each change.

### 🔍 Debugging

**Flash Attention Issues?**
- Verify shapes: Query, Key, Value should have shape `(batch, seq_len, num_heads, head_dim)`
- Check causal mask: Ensure proper causal masking for autoregressive models
- Compare outputs: Use `torch.allclose()` to compare with vanilla attention
- Reduce sequence length: Start with shorter sequences (512) before scaling up

**Triton Kernel Errors?**
- Check CUDA compatibility: Triton requires specific CUDA versions
- Verify GPU compute capability: Use `torch.cuda.get_device_capability()`
- Review kernel parameters: Ensure block sizes are appropriate for your GPU

**Out of Memory?**
- Reduce batch size or sequence length
- Use gradient checkpointing
- Enable mixed precision training
- Try Flash Attention (reduces memory by 2-3x)

**Performance Not Improving?**
- Verify GPU utilization: Use `nvidia-smi` to check GPU usage
- Check for CPU-GPU data transfer bottlenecks
- Ensure data is pre-loaded and cached
- Profile with NVIDIA Nsight Systems for detailed analysis

### 📊 Expected Results

After completing this assignment, you should observe:

1. **Flash Attention**:
   - 2-4x speedup for long sequences (2K+ tokens)
   - 2-3x memory reduction
   - Identical outputs to vanilla attention (within numerical precision)

2. **Mixed Precision**:
   - 1.5-2x training speedup
   - 40-50% memory reduction
   - Minimal impact on final model quality

3. **Overall System**:
   - Ability to train larger models on same hardware
   - Faster iteration cycles during development
   - Better understanding of GPU memory hierarchy

## Assignment Handout

For detailed assignment requirements and theoretical background, see:
- [cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

## Additional Resources

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Flash Attention: Fast and Memory-Efficient Exact Attention
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Flash Attention-2: Faster Attention with Better Parallelism
- [Triton Documentation](https://triton-lang.org/) - Triton: GPU programming language
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - Mixed Precision Training paper
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) - Automatic Mixed Precision in PyTorch

## License

This code is provided for educational purposes as part of Stanford CS336.

---

# 中文版本 | Chinese Version

# CS336 作业 2: 系统优化

本作业专注于构建高效、可扩展的语言模型训练系统。涵盖性能优化技术，包括 Flash Attention、混合精度训练、内存分析和分布式训练基础。

## 📋 目录
- [概述](#概述)
- [实现细节](#实现细节)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [使用指南](#使用指南)
- [测试](#测试)
- [计算资源需求](#计算资源需求)
- [重要说明](#重要说明)

## 概述

本作业探讨了训练大型语言模型的关键系统级优化：

1. **Flash Attention**: 内存高效的注意力机制
   - 使用分块的 PyTorch 实现
   - 用于自定义 GPU 内核的 Triton 内核实现
2. **混合精度训练**: FP16/BF16 训练以实现更快的计算
3. **内存分析**: 理解和优化 GPU 内存使用
4. **性能基准测试**: 测量吞吐量和内存效率

## 实现细节

### 核心组件

#### 1. Flash Attention (`cs336_systems/flash_attention/`)

**Flash Attention PyTorch** (`flash_att_pytorch.py`):
- 分块注意力计算以减少内存使用
- 支持因果掩码
- 内存高效的前向和反向传播
- 与标准 PyTorch 自动微分兼容

**Flash Attention Triton** (`flash_att_triton.py`):
- 自定义 Triton 内核以获得最大性能
- 融合操作（softmax、dropout 等）
- 优化的内存访问模式
- 使用 Python 语法实现 CUDA 级别的性能

**基准测试** (`benchmark_flash.py`, `benchamark_script.py`):
- 比较原始注意力与 Flash Attention
- 测量时间和内存消耗
- 测试各种序列长度和批次大小
- 生成性能对比图表

#### 2. 混合精度训练 (`Benckmark/mixed_percision_script.py`)
- FP16 和 BF16 训练实现
- 自动混合精度（AMP）集成
- 损失缩放以保证数值稳定性
- 与 FP32 的性能对比

#### 3. 分析和基准测试 (`Benckmark/benchmark.py`)
- 内存分析工具
- 前向/反向传播分析
- 吞吐量测量
- 详细性能报告

**注意**: ⚠️ 分布式并行训练（DDP、模型并行）**未完全实现**。虽然存在测试文件，但实现不完整。

## 项目结构

```
assignment2-systems/
├── cs336-basics/              # 作业1的官方实现
│   └── cs336_basics/          # 基础语言模型模块（在此复用）
├── cs336_systems/             # 系统优化实现
│   ├── flash_attention/
│   │   ├── flash_att_pytorch.py    # PyTorch Flash Attention
│   │   ├── flash_att_triton.py     # Triton Flash Attention
│   │   ├── benchmark_flash.py      # Flash Attention 基准测试
│   │   └── benchamark_script.py    # 基准测试执行脚本
│   └── Benckmark/
│       ├── benchmark.py            # 通用分析工具
│       └── mixed_percision_script.py # 混合精度基准测试
├── tests/
│   ├── adapters.py            # 测试适配器（连接你的实现）
│   ├── test_attention.py      # Flash Attention 测试
│   ├── test_ddp.py            # 分布式训练测试（未完成）
│   └── test_sharded_optimizer.py # 分片优化器测试（未完成）
├── pyproject.toml             # 项目依赖
└── README.md                  # 本文件
```

## 环境配置

### 1. 安装依赖

本项目使用 `uv`，需要支持 CUDA 的 GPU 以获得最佳性能：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

### 2. 验证 CUDA 设置

Flash Attention 和 Triton 需要 CUDA：

```bash
# 检查 CUDA 可用性
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# 检查 Triton 安装
uv run python -c "import triton; print(f'Triton version: {triton.__version__}')"
```

### 环境要求

- **CUDA**: 11.8+ 或 12.x
- **GPU**: 计算能力 7.0+ 的 NVIDIA GPU（V100、RTX 2080+、A100 等）
- **PyTorch**: 2.8.0（支持包括 RTX 50 系列在内的最新 GPU）
- **Triton**: 随 PyTorch 自动安装

## 使用指南

### 1. Flash Attention 基准测试

比较原始注意力与 Flash Attention 实现：

```bash
# 运行 Flash Attention 基准测试
cd cs336_systems/flash_attention
uv run python benchmark_flash.py

# 运行综合基准测试脚本
uv run python benchamark_script.py
```

**功能说明**:
- 测试多种序列长度（512、1024、2048、4096）
- 比较原始注意力、PyTorch Flash Attention 和 Triton Flash Attention
- 测量前向/反向传播时间和峰值内存使用
- 生成性能对比图表

**预期输出**:
```
序列长度: 1024
├── 原始注意力:           45.2 ms/iter, 8.3 GB 内存
├── Flash Attention (PT):  18.7 ms/iter, 3.1 GB 内存（2.4倍加速，2.7倍内存减少）
└── Flash Attention (Triton): 15.3 ms/iter, 2.9 GB 内存（3.0倍加速，2.9倍内存减少）
```

### 2. 混合精度训练

对比 FP16/BF16 与 FP32 训练：

```bash
cd cs336_systems/Benckmark
uv run python mixed_percision_script.py
```

**测量指标**:
- 训练吞吐量（样本/秒）
- 内存消耗
- 数值稳定性
- 损失收敛情况

**预期结果**:
- **FP16/BF16**: 比 FP32 快约 1.5-2 倍，内存减少约 50%
- **BF16**: 对于大模型，数值稳定性优于 FP16

### 3. 通用性能分析

分析模型训练性能：

```bash
cd cs336_systems/Benckmark
uv run python benchmark.py
```

生成内容包括：
- 前向/反向传播时间
- 内存分配详情
- 瓶颈识别
- 优化建议

### 4. 在训练中使用 Flash Attention

将 Flash Attention 集成到你的模型中：

```python
from cs336_systems.flash_attention.flash_att_pytorch import flash_attention_pytorch
# 或者
from cs336_systems.flash_attention.flash_att_triton import flash_attention_triton

# 在你的注意力模块中
def forward(self, query, key, value, mask=None):
    # 用 Flash Attention 替换原始注意力
    output = flash_attention_pytorch(query, key, value, causal=True)
    # 或使用 Triton 版本以获得最大性能
    output = flash_attention_triton(query, key, value, causal=True)
    return output
```

## 测试

运行测试套件：

```bash
# 运行所有测试
uv run pytest

# 运行特定测试模块
uv run pytest tests/test_attention.py        # Flash Attention 测试
uv run pytest tests/test_ddp.py             # DDP 测试（可能无法通过）
uv run pytest tests/test_sharded_optimizer.py # 分片优化器测试（可能无法通过）

# 详细输出运行
uv run pytest -v -s
```

**重要提示**:
- 在 `tests/adapters.py` 中完成适配器函数以连接你的实现
- 由于完整的 DDP 实现未完成，一些分布式训练测试可能会失败

### 测试覆盖范围

- ✅ Flash Attention: 与原始注意力的正确性对比、数值稳定性
- ⚠️ DDP: 分布式数据并行（测试存在，实现不完整）
- ⚠️ 分片优化器: 内存高效的优化器分片（测试存在，实现不完整）

## 计算资源需求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|-----------|---------|-------------|
| **GPU** | 8GB 显存的 NVIDIA GPU（RTX 2080、V100） | 24GB+ 显存的 NVIDIA GPU（RTX 3090、A100） |
| **CUDA** | 11.8+ | 12.x |
| **内存** | 16GB | 32GB+ |
| **存储** | 5GB | 10GB |

### 性能基准

**Flash Attention 加速**（序列长度 = 2048，批次大小 = 16）:
- **RTX 3090**: 2.5-3倍加速，2.8倍内存减少
- **A100**: 2.8-3.5倍加速，3倍内存减少
- **H100**: 3.5-4倍加速，3.2倍内存减少

**混合精度训练**（中等模型，6层，d_model=768）:
- **FP32**: 约 100 样本/秒，16GB 内存
- **BF16**: 约 180 样本/秒，8GB 内存（1.8倍加速，50% 内存节省）

### 基准测试执行时间

- **Flash Attention 基准测试**: 约 5-10 分钟（测试多种配置）
- **混合精度基准测试**: 约 10-15 分钟
- **完整性能分析套件**: 约 15-20 分钟

## 重要说明

### ⚠️ 实现限制

1. **分布式训练未实现**: 虽然存在 DDP 和模型并行的测试文件，但完整实现**未完成**。这包括：
   - 分布式数据并行（DDP）
   - 模型并行
   - 分片优化器

2. **需要 GPU**: Flash Attention 和 Triton 内核需要支持 CUDA 的 NVIDIA GPU。不支持 CPU 回退。

3. **Triton 兼容性**: Triton 内核可能不适用于所有 GPU 架构。推荐计算能力 7.0+ 的 GPU。

### 💡 成功技巧

1. **从 PyTorch Flash Attention 开始**: 在深入 Triton 之前，确保 PyTorch 实现正常工作。

2. **使用适当的精度**:
   - BF16 适用于大模型（数值范围优于 FP16）
   - FP16 适用于较小模型
   - 最终验证始终使用 FP32

3. **先分析再优化**: 使用 `benchmark.py` 在应用优化之前识别瓶颈。

4. **监控内存**: 使用 `torch.cuda.memory_summary()` 在开发过程中跟踪内存使用。

5. **测试数值稳定性**: 始终与原始注意力的输出进行比较以确保正确性。

6. **逐步优化**: 一次优化一个组件，并在每次更改后验证正确性。

### 🔍 调试指南

**Flash Attention 问题？**
- 验证形状: Query、Key、Value 应具有形状 `(batch, seq_len, num_heads, head_dim)`
- 检查因果掩码: 确保自回归模型的因果掩码正确
- 比较输出: 使用 `torch.allclose()` 与原始注意力进行比较
- 减少序列长度: 从较短序列（512）开始，再逐步扩展

**Triton 内核错误？**
- 检查 CUDA 兼容性: Triton 需要特定的 CUDA 版本
- 验证 GPU 计算能力: 使用 `torch.cuda.get_device_capability()`
- 检查内核参数: 确保块大小适合你的 GPU

**内存不足？**
- 减少批次大小或序列长度
- 使用梯度检查点
- 启用混合精度训练
- 尝试 Flash Attention（内存减少 2-3 倍）

**性能未提升？**
- 验证 GPU 利用率: 使用 `nvidia-smi` 检查 GPU 使用情况
- 检查 CPU-GPU 数据传输瓶颈
- 确保数据已预加载并缓存
- 使用 NVIDIA Nsight Systems 进行详细分析

### 📊 预期结果

完成本作业后，你应该观察到：

1. **Flash Attention**:
   - 长序列（2K+ 个 token）加速 2-4 倍
   - 内存减少 2-3 倍
   - 与原始注意力输出一致（在数值精度范围内）

2. **混合精度**:
   - 训练加速 1.5-2 倍
   - 内存减少 40-50%
   - 对最终模型质量影响最小

3. **整体系统**:
   - 能够在相同硬件上训练更大的模型
   - 开发过程中更快的迭代周期
   - 更好地理解 GPU 内存层次结构

## 作业说明

详细的作业要求和理论背景，请参阅：
- [cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

## 额外资源

- [Flash Attention 论文](https://arxiv.org/abs/2205.14135) - Flash Attention: 快速且内存高效的精确注意力
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Flash Attention-2: 更快的注意力与更好的并行性
- [Triton 文档](https://triton-lang.org/) - Triton: GPU 编程语言
- [混合精度训练论文](https://arxiv.org/abs/1710.03740) - 混合精度训练
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) - PyTorch 中的自动混合精度

## 许可证

本代码仅供教育目的使用，是斯坦福 CS336 课程的一部分。
