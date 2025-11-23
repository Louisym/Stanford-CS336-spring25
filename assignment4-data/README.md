# CS336 Assignment 4: Data Processing and Curation

This assignment focuses on building a high-quality training dataset for language models through web scraping, filtering, deduplication, and quality assessment. The goal is to construct a curated dataset that improves model performance.

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

Building high-quality training data is critical for language model performance. This assignment implements a complete data processing pipeline:

1. **Web Scraping**: Extract text from web archives (WARC/WET files)
2. **Language Identification**: Filter for desired languages (e.g., English)
3. **Quality Filtering**: Train and apply quality classifiers
4. **Deduplication**: Remove duplicate content
5. **PII Removal**: Detect and redact personally identifiable information
6. **Toxicity Filtering**: Remove toxic/harmful content
7. **Final Training**: Train language models on curated data

## Implementation Details

### Key Components

#### 1. Web Content Extraction (`tests/test_extract.py`)
- Parse WARC (Web ARChive) and WET (WARC Extracted Text) files
- Extract clean text from Common Crawl archives
- Handle encoding issues and malformed data

#### 2. Language Identification (`tests/test_langid.py`)
- Use fastText language detection models
- Filter for target languages (primarily English)
- Handle multilingual documents

#### 3. Quality Classification (`cs336_data/quality_classifier_train_script.py`, `quality_inference.py`)
- **Data Collection**: Scrape high-quality (Wikipedia) and low-quality text
- **Model Training**: Train fastText classifiers to distinguish quality levels
- **Inference**: Apply trained classifiers to filter datasets
- **Custom Scripts**:
  - `scrape_quality_data.py`: Collect training data for quality classifier
  - `sample_LQ.py`: Sample low-quality examples
  - `clean_HQ.py`: Clean high-quality Wikipedia data

#### 4. Deduplication (`tests/test_deduplication.py`)
- Remove exact duplicates (document-level)
- MinHash-based fuzzy deduplication
- Substring deduplication

#### 5. PII Detection (`tests/test_pii.py`)
- Detect email addresses, phone numbers, SSNs
- Redact or remove documents with PII
- Pattern-based and ML-based detection

#### 6. Toxicity Filtering (`tests/test_toxicity.py`)
- Use pre-trained toxicity classifiers
- Filter hateful, offensive, or harmful content
- Balance safety with over-filtering

**Note**: ⚠️ This implementation was trained on **smaller datasets**. Full-scale data processing on billions of tokens was not performed.

## Project Structure

```
assignment4-data/
├── cs336-basics/              # Training code from Assignment 1
├── cs336_data/                # Data processing implementations
│   ├── quality_classifier_train_script.py  # Train quality classifiers
│   ├── quality_inference.py               # Apply quality classifiers
│   ├── scrape_quality_data.py            # Scrape training data
│   ├── sample_LQ.py                      # Sample low-quality data
│   ├── clean_HQ.py                       # Clean Wikipedia data
│   ├── script_22.py                      # Custom processing script
│   └── script_23.py                      # Custom processing script
├── tests/
│   ├── test_extract.py        # Web content extraction tests
│   ├── test_langid.py         # Language identification tests
│   ├── test_quality.py        # Quality filtering tests
│   ├── test_deduplication.py  # Deduplication tests
│   ├── test_pii.py            # PII detection tests
│   └── test_toxicity.py       # Toxicity filtering tests
├── analyze_training_data.py   # Data analysis utilities
├── debug_classifier.py        # Classifier debugging tools
├── get_assets.sh              # Download required model assets
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## Setup

### 1. Install Dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Download Required Assets

Download pre-trained models for language identification and toxicity detection:

```bash
# Run the asset download script
bash get_assets.sh
```

This downloads:
- **fastText language ID model** (`lid.176.bin`): 126MB
- **Jigsaw toxicity classifiers** (`jigsaw_fasttext_bigrams_*.bin`): ~2GB

### 3. Download Sample Data

Get sample Common Crawl data:

```bash
mkdir -p data

# Download a single WARC file (~1GB compressed)
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/.../warc/CC-MAIN-...warc.gz

# Or download WET (extracted text, smaller)
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/.../wet/CC-MAIN-...warc.wet.gz
```

## Usage

### Step 1: Extract Web Content

Parse WARC/WET files to extract text:

```python
from cs336_data.extract import extract_text_from_wet

with open('example.warc.wet.gz', 'rb') as f:
    documents = extract_text_from_wet(f)

for doc in documents:
    print(f"URL: {doc['url']}")
    print(f"Text: {doc['text'][:200]}...")
```

### Step 2: Language Identification

Filter for English content:

```python
import fasttext

# Load language ID model
model = fasttext.load_model('lid.176.bin')

def is_english(text, threshold=0.5):
    predictions = model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    score = predictions[1][0]
    return lang == 'en' and score >= threshold

# Filter documents
english_docs = [doc for doc in documents if is_english(doc['text'])]
```

### Step 3: Train Quality Classifier

Collect training data and train a classifier:

```bash
# Step 3a: Scrape high-quality data (Wikipedia)
uv run python cs336_data/scrape_quality_data.py

# Step 3b: Sample low-quality data (from Common Crawl)
uv run python cs336_data/sample_LQ.py

# Step 3c: Clean high-quality data
uv run python cs336_data/clean_HQ.py

# Step 3d: Train quality classifier
uv run python cs336_data/quality_classifier_train_script.py
```

This produces a trained fastText model that classifies text quality.

### Step 4: Apply Quality Filtering

Use the trained classifier to filter your dataset:

```bash
uv run python cs336_data/quality_inference.py --input data/raw.txt --output data/filtered.txt
```

### Step 5: Deduplication

Remove duplicate documents:

```python
from cs336_data.dedup import deduplicate

# Exact deduplication
unique_docs = deduplicate(documents, method='exact')

# Fuzzy deduplication (MinHash)
unique_docs = deduplicate(documents, method='minhash', threshold=0.8)
```

### Step 6: PII Detection and Removal

Detect and remove PII:

```python
from cs336_data.pii import detect_pii, redact_pii

for doc in documents:
    if detect_pii(doc['text']):
        # Option 1: Skip document
        continue
        # Option 2: Redact PII
        doc['text'] = redact_pii(doc['text'])
```

### Step 7: Toxicity Filtering

Filter toxic content:

```python
import fasttext

# Load toxicity classifier
toxicity_model = fasttext.load_model('jigsaw_fasttext_bigrams_hatespeech_final.bin')

def is_toxic(text, threshold=0.5):
    pred = toxicity_model.predict(text, k=1)
    score = pred[1][0]
    return score >= threshold

# Filter documents
clean_docs = [doc for doc in documents if not is_toxic(doc['text'])]
```

### Step 8: Train Model on Curated Data

After filtering, train a language model:

```bash
# Tokenize curated data
uv run python scripts/tokenize_data.py --input data/curated.txt --output data/curated.dat

# Train model
uv run python cs336_basics/train.py --data_dir data/ --wandb_project "curated-lm"
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_extract.py
uv run pytest tests/test_langid.py
uv run pytest tests/test_quality.py
uv run pytest tests/test_deduplication.py
uv run pytest tests/test_pii.py
uv run pytest tests/test_toxicity.py

# Verbose output
uv run pytest -v -s
```

**Important**: Complete the adapter functions in `tests/adapters.py` to connect your implementation.

## Computational Requirements

### Hardware Requirements

| Task | Hardware | Time Estimate | Storage |
|------|----------|---------------|---------|
| **Quality Classifier Training** | CPU | 10-30 minutes | 5GB |
| **Processing 1GB WARC** | CPU | 5-15 minutes | 2GB |
| **Processing 100GB Common Crawl** | Multi-core CPU | 5-10 hours | 200GB |
| **Full Pipeline (1TB data)** | Multi-core server | 2-5 days | 2TB |

### Pipeline Performance

**Single WARC File Processing** (~1GB compressed):
- **Extraction**: ~2-5 minutes
- **Language ID**: ~5-10 minutes
- **Quality Filtering**: ~5-10 minutes
- **Deduplication**: ~10-20 minutes
- **Total**: ~30-45 minutes on 8-core CPU

**Large-Scale Processing** (100 WARC files, ~100GB):
- **With parallelization** (16 cores): ~8-12 hours
- **With distributed processing** (100+ cores): ~2-4 hours

### Storage Requirements

- **Raw Common Crawl** (compressed): ~1GB per WARC file
- **Extracted Text**: ~2-3x compression ratio
- **After Filtering**: ~10-30% of original (depending on quality threshold)
- **Final Training Data**: Typically 5-15% of raw data

## Important Notes

### ⚠️ Implementation Limitations

1. **Small-Scale Training**: This implementation was completed using **smaller datasets**. Full-scale processing of billions of tokens was not performed due to computational constraints.

2. **Asset Dependencies**: Requires downloading external models:
   - Language ID model (126MB)
   - Toxicity classifiers (~2GB)
   - Ensure `get_assets.sh` runs successfully

3. **Common Crawl Access**: Processing full Common Crawl requires:
   - Significant storage (PB-scale for full archive)
   - Distributed processing infrastructure
   - Consider using samples or specific segments for testing

### 💡 Tips for Success

1. **Start Small**: Test your pipeline on a single WARC file before scaling up.

2. **Quality vs. Quantity**: Aggressive filtering improves quality but reduces dataset size. Find the right balance.

3. **Parallelization**: Process multiple WARC files in parallel to speed up data collection.

4. **Monitoring**: Track statistics (documents processed, filtered out, reasons for filtering) to understand your pipeline.

5. **Iterative Development**:
   - First, implement basic extraction
   - Add filters one at a time
   - Validate each step before proceeding

6. **Sampling**: Train quality classifiers on representative samples rather than entire datasets.

### 🔍 Expected Results

After processing data through the full pipeline:

1. **Extraction**: ~50-70% of WARC files contain usable text
2. **Language Filtering**: ~30-50% is English (varies by Common Crawl segment)
3. **Quality Filtering**: ~10-30% passes quality thresholds
4. **Deduplication**: ~5-20% reduction in dataset size
5. **PII/Toxicity**: ~1-5% filtered out
6. **Final Yield**: ~5-15% of original data

**Quality Impact**: Models trained on curated data typically show:
- 10-30% lower perplexity compared to raw data
- Better performance on downstream tasks
- Reduced toxic generations

### 📊 Data Processing Best Practices

1. **Quality Classifier**:
   - Use Wikipedia as high-quality source
   - Sample diverse low-quality examples
   - Validate classifier on hold-out set
   - Consider multiple quality tiers (low/medium/high)

2. **Deduplication**:
   - Start with exact deduplication (fast, effective)
   - Use MinHash for fuzzy deduplication (slower, catches near-duplicates)
   - Consider document-level vs. paragraph-level deduplication

3. **PII Handling**:
   - Pattern-based detection is fast but may miss edge cases
   - ML-based detection is more accurate but slower
   - Decide between redaction vs. removal based on use case

4. **Toxicity**:
   - Set appropriate thresholds (too aggressive → loss of neutral content)
   - Consider context-aware filtering
   - Maintain diversity while removing harmful content

### 🔗 Useful Resources

- [Common Crawl](https://commoncrawl.org/) - Web crawl archive
- [fastText](https://fasttext.cc/) - Text classification and embeddings
- [WARC Format](https://iipc.github.io/warc-specifications/) - Web archive format
- [MinHash Tutorial](https://en.wikipedia.org/wiki/MinHash) - Fuzzy deduplication

## Assignment Handout

For detailed assignment requirements and theoretical background, see:
- [cs336_spring2025_assignment4_data.pdf](./cs336_spring2025_assignment4_data.pdf)

## License

This code is provided for educational purposes as part of Stanford CS336.

---

# 中文版本 | Chinese Version

# CS336 作业 4: 数据处理与管理

本作业专注于通过网页抓取、过滤、去重和质量评估构建高质量的语言模型训练数据集。目标是构建能够提升模型性能的精选数据集。

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

构建高质量的训练数据对语言模型性能至关重要。本作业实现了完整的数据处理流水线：

1. **网页抓取**: 从网页存档（WARC/WET 文件）提取文本
2. **语言识别**: 过滤所需语言（例如英语）
3. **质量过滤**: 训练和应用质量分类器
4. **去重**: 删除重复内容
5. **PII 移除**: 检测和删除个人身份信息
6. **毒性过滤**: 删除有毒/有害内容
7. **最终训练**: 在精选数据上训练语言模型

## 实现细节

### 核心组件

#### 1. 网页内容提取 (`tests/test_extract.py`)
- 解析 WARC（Web ARChive）和 WET（WARC Extracted Text）文件
- 从 Common Crawl 档案提取干净文本
- 处理编码问题和格式错误的数据

#### 2. 语言识别 (`tests/test_langid.py`)
- 使用 fastText 语言检测模型
- 过滤目标语言（主要是英语）
- 处理多语言文档

#### 3. 质量分类 (`cs336_data/quality_classifier_train_script.py`, `quality_inference.py`)
- **数据收集**: 抓取高质量（维基百科）和低质量文本
- **模型训练**: 训练 fastText 分类器以区分质量水平
- **推理**: 应用训练好的分类器来过滤数据集
- **自定义脚本**:
  - `scrape_quality_data.py`: 收集质量分类器的训练数据
  - `sample_LQ.py`: 采样低质量示例
  - `clean_HQ.py`: 清理高质量维基百科数据

#### 4. 去重 (`tests/test_deduplication.py`)
- 删除精确重复（文档级别）
- 基于 MinHash 的模糊去重
- 子字符串去重

#### 5. PII 检测 (`tests/test_pii.py`)
- 检测电子邮件地址、电话号码、社会安全号
- 删除或移除包含 PII 的文档
- 基于模式和基于机器学习的检测

#### 6. 毒性过滤 (`tests/test_toxicity.py`)
- 使用预训练的毒性分类器
- 过滤仇恨、冒犯或有害内容
- 在安全性和过度过滤之间取得平衡

**注意**: ⚠️ 本实现使用**较小的数据集**进行训练。未执行数十亿 token 的全规模数据处理。

## 项目结构

```
assignment4-data/
├── cs336-basics/              # 作业1的训练代码
├── cs336_data/                # 数据处理实现
│   ├── quality_classifier_train_script.py  # 训练质量分类器
│   ├── quality_inference.py               # 应用质量分类器
│   ├── scrape_quality_data.py            # 抓取训练数据
│   ├── sample_LQ.py                      # 采样低质量数据
│   ├── clean_HQ.py                       # 清理维基百科数据
│   ├── script_22.py                      # 自定义处理脚本
│   └── script_23.py                      # 自定义处理脚本
├── tests/
│   ├── test_extract.py        # 网页内容提取测试
│   ├── test_langid.py         # 语言识别测试
│   ├── test_quality.py        # 质量过滤测试
│   ├── test_deduplication.py  # 去重测试
│   ├── test_pii.py            # PII 检测测试
│   └── test_toxicity.py       # 毒性过滤测试
├── analyze_training_data.py   # 数据分析工具
├── debug_classifier.py        # 分类器调试工具
├── get_assets.sh              # 下载所需模型资源
├── pyproject.toml             # 依赖项
└── README.md                  # 本文件
```

## 环境配置

### 1. 安装依赖

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

### 2. 下载所需资源

下载语言识别和毒性检测的预训练模型：

```bash
# 运行资源下载脚本
bash get_assets.sh
```

这会下载：
- **fastText 语言识别模型** (`lid.176.bin`): 126MB
- **Jigsaw 毒性分类器** (`jigsaw_fasttext_bigrams_*.bin`): 约 2GB

### 3. 下载示例数据

获取示例 Common Crawl 数据：

```bash
mkdir -p data

# 下载单个 WARC 文件（压缩后约 1GB）
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/.../warc/CC-MAIN-...warc.gz

# 或下载 WET（提取的文本，更小）
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/.../wet/CC-MAIN-...warc.wet.gz
```

## 使用指南

### 步骤 1: 提取网页内容

解析 WARC/WET 文件以提取文本：

```python
from cs336_data.extract import extract_text_from_wet

with open('example.warc.wet.gz', 'rb') as f:
    documents = extract_text_from_wet(f)

for doc in documents:
    print(f"URL: {doc['url']}")
    print(f"Text: {doc['text'][:200]}...")
```

### 步骤 2: 语言识别

过滤英语内容：

```python
import fasttext

# 加载语言识别模型
model = fasttext.load_model('lid.176.bin')

def is_english(text, threshold=0.5):
    predictions = model.predict(text, k=1)
    lang = predictions[0][0].replace('__label__', '')
    score = predictions[1][0]
    return lang == 'en' and score >= threshold

# 过滤文档
english_docs = [doc for doc in documents if is_english(doc['text'])]
```

### 步骤 3: 训练质量分类器

收集训练数据并训练分类器：

```bash
# 步骤 3a: 抓取高质量数据（维基百科）
uv run python cs336_data/scrape_quality_data.py

# 步骤 3b: 采样低质量数据（来自 Common Crawl）
uv run python cs336_data/sample_LQ.py

# 步骤 3c: 清理高质量数据
uv run python cs336_data/clean_HQ.py

# 步骤 3d: 训练质量分类器
uv run python cs336_data/quality_classifier_train_script.py
```

这会生成一个训练好的 fastText 模型来分类文本质量。

### 步骤 4: 应用质量过滤

使用训练好的分类器过滤你的数据集：

```bash
uv run python cs336_data/quality_inference.py --input data/raw.txt --output data/filtered.txt
```

### 步骤 5: 去重

删除重复文档：

```python
from cs336_data.dedup import deduplicate

# 精确去重
unique_docs = deduplicate(documents, method='exact')

# 模糊去重（MinHash）
unique_docs = deduplicate(documents, method='minhash', threshold=0.8)
```

### 步骤 6: PII 检测和移除

检测和移除 PII：

```python
from cs336_data.pii import detect_pii, redact_pii

for doc in documents:
    if detect_pii(doc['text']):
        # 选项 1: 跳过文档
        continue
        # 选项 2: 删除 PII
        doc['text'] = redact_pii(doc['text'])
```

### 步骤 7: 毒性过滤

过滤有毒内容：

```python
import fasttext

# 加载毒性分类器
toxicity_model = fasttext.load_model('jigsaw_fasttext_bigrams_hatespeech_final.bin')

def is_toxic(text, threshold=0.5):
    pred = toxicity_model.predict(text, k=1)
    score = pred[1][0]
    return score >= threshold

# 过滤文档
clean_docs = [doc for doc in documents if not is_toxic(doc['text'])]
```

### 步骤 8: 在精选数据上训练模型

过滤后，训练语言模型：

```bash
# 对精选数据进行分词
uv run python scripts/tokenize_data.py --input data/curated.txt --output data/curated.dat

# 训练模型
uv run python cs336_basics/train.py --data_dir data/ --wandb_project "curated-lm"
```

## 测试

运行综合测试套件：

```bash
# 运行所有测试
uv run pytest

# 运行特定测试模块
uv run pytest tests/test_extract.py
uv run pytest tests/test_langid.py
uv run pytest tests/test_quality.py
uv run pytest tests/test_deduplication.py
uv run pytest tests/test_pii.py
uv run pytest tests/test_toxicity.py

# 详细输出
uv run pytest -v -s
```

**重要提示**: 在 `tests/adapters.py` 中完成适配器函数以连接你的实现。

## 计算资源需求

### 硬件要求

| 任务 | 硬件 | 时间估计 | 存储 |
|------|----------|---------------|---------:|
| **质量分类器训练** | CPU | 10-30 分钟 | 5GB |
| **处理 1GB WARC** | CPU | 5-15 分钟 | 2GB |
| **处理 100GB Common Crawl** | 多核 CPU | 5-10 小时 | 200GB |
| **完整流水线（1TB 数据）** | 多核服务器 | 2-5 天 | 2TB |

### 流水线性能

**单个 WARC 文件处理**（压缩后约 1GB）:
- **提取**: 约 2-5 分钟
- **语言识别**: 约 5-10 分钟
- **质量过滤**: 约 5-10 分钟
- **去重**: 约 10-20 分钟
- **总计**: 8 核 CPU 上约 30-45 分钟

**大规模处理**（100 个 WARC 文件，约 100GB）:
- **并行化**（16 核）: 约 8-12 小时
- **分布式处理**（100+ 核）: 约 2-4 小时

### 存储需求

- **原始 Common Crawl**（压缩）: 每个 WARC 文件约 1GB
- **提取的文本**: 约 2-3 倍压缩比
- **过滤后**: 原始数据的 10-30%（取决于质量阈值）
- **最终训练数据**: 通常为原始数据的 5-15%

## 重要说明

### ⚠️ 实现限制

1. **小规模训练**: 本实现使用**较小的数据集**完成。由于计算限制，未执行数十亿 token 的全规模处理。

2. **资源依赖**: 需要下载外部模型：
   - 语言识别模型（126MB）
   - 毒性分类器（约 2GB）
   - 确保 `get_assets.sh` 成功运行

3. **Common Crawl 访问**: 处理完整的 Common Crawl 需要：
   - 大量存储（完整存档为 PB 级）
   - 分布式处理基础设施
   - 考虑使用样本或特定片段进行测试

### 💡 成功技巧

1. **从小处开始**: 在扩大规模之前，先在单个 WARC 文件上测试你的流水线。

2. **质量与数量**: 积极的过滤可以提高质量但会减少数据集大小。找到正确的平衡点。

3. **并行化**: 并行处理多个 WARC 文件以加快数据收集速度。

4. **监控**: 跟踪统计信息（已处理的文档、已过滤的文档、过滤原因）以了解你的流水线。

5. **迭代开发**:
   - 首先，实现基本提取
   - 一次添加一个过滤器
   - 在继续之前验证每个步骤

6. **采样**: 在代表性样本而非整个数据集上训练质量分类器。

### 🔍 预期结果

在完整流水线处理数据后：

1. **提取**: 约 50-70% 的 WARC 文件包含可用文本
2. **语言过滤**: 约 30-50% 是英语（因 Common Crawl 片段而异）
3. **质量过滤**: 约 10-30% 通过质量阈值
4. **去重**: 数据集大小减少 5-20%
5. **PII/毒性**: 过滤掉 1-5%
6. **最终产出**: 原始数据的约 5-15%

**质量影响**: 在精选数据上训练的模型通常显示：
- 与原始数据相比，困惑度降低 10-30%
- 下游任务性能更好
- 减少有毒生成

### 📊 数据处理最佳实践

1. **质量分类器**:
   - 使用维基百科作为高质量来源
   - 采样多样化的低质量示例
   - 在保留集上验证分类器
   - 考虑多个质量层级（低/中/高）

2. **去重**:
   - 从精确去重开始（快速、有效）
   - 使用 MinHash 进行模糊去重（较慢，能捕获近似重复）
   - 考虑文档级别与段落级别去重

3. **PII 处理**:
   - 基于模式的检测快速但可能遗漏边缘情况
   - 基于机器学习的检测更准确但更慢
   - 根据使用场景决定删除还是替换

4. **毒性**:
   - 设置适当的阈值（过于激进 → 丢失中性内容）
   - 考虑上下文感知过滤
   - 在删除有害内容的同时保持多样性

### 🔗 有用资源

- [Common Crawl](https://commoncrawl.org/) - 网页爬取存档
- [fastText](https://fasttext.cc/) - 文本分类和嵌入
- [WARC 格式](https://iipc.github.io/warc-specifications/) - 网页存档格式
- [MinHash 教程](https://en.wikipedia.org/wiki/MinHash) - 模糊去重

## 作业说明

详细的作业要求和理论背景，请参阅：
- [cs336_spring2025_assignment4_data.pdf](./cs336_spring2025_assignment4_data.pdf)

## 许可证

本代码仅供教育目的使用，是斯坦福 CS336 课程的一部分。
