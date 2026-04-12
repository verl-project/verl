# DataObs

数据观察和分析管线，研究数据集特性与模型训练表现之间的关系。

## 运行

### 基础用法

```bash
cd /home/hrh/CoT-DataSynth/DataObs
python scripts/data_obs_pipeline.py \
  --data_path /data/open_datasets/GSM8K/train_messages.parquet \
  --n_splits 10 \
  --output_dir /data/hrh/COT
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | 必填 | 数据路径 (parquet/jsonl/json) |
| `--data_name` | 自动从 data_path 提取 | 输出子目录名 |
| `--n_splits` | 10 | 分割数量 |
| `--output_dir` | 必填 | 输出目录 |
| `--gpu_ids` | 0,1,2,3,4,5,6,7 | GPU 列表 |
| `--gpus_per_split` | 1 | 每个分割用多少 GPU |
| `--train_script` | scripts/sft.sh | 训练脚本 |
| `--model_id` | None | 模型路径 |
| `--cot_datasynth_dir` | /home/hrh/COT-DataSynth | CoT-DataSynth 目录 |
| `--run_training` | 否 | 运行训练 |
| `--run_analysis` | 否 | 运行分析 |
| `--skip_split` | 否 | 跳过数据分割 |
| `--skip_metrics` | 否 | 跳过指标计算 |
| `--seed` | 42 | 随机种子 |

### 输出目录结构

```
<output_dir>/<data_name>/
├── splits/              # 分割数据 (split_0.jsonl, ...)
├── metrics/            # 分割指标 (split_0_metrics.json, ...)
├── data_metrics_summary.csv  # 指标汇总
└── [training/]        # 训练结果 (如 run_training)
└── [results/]        # 分析结果 (如 run_analysis)
```

## 计算指标 (compute_metrics.py)

### 用法

```bash
# 计算所有指标
python scripts/compute_metrics.py --output_dir /data/hrh/COT/GSM8K

# 只计算特定指标
python scripts/compute_metrics.py --output_dir /data/hrh/COT/GSM8K --metrics diversity entropy

# 使用指定相似度计算 diversity
python scripts/compute_metrics.py --output_dir /data/hrh/COT/GSM8K --metrics diversity --similarity_type cosine

# 在 answer 上计算 diversity
python scripts/compute_metrics.py --output_dir /data/hrh/COT/GSM8K --metrics diversity --compute_on answer

# 计算需要模型的指标
python scripts/compute_metrics.py --output_dir /data/hrh/COT/GSM8K --metrics ppl ifd --model gpt2
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | 必填 | 输出目录 |
| `--data_name` | None | 数据子目录名 |
| `--split_ids` | None | 指定 split ID (如 --split_ids 0 1 2) |
| `--metrics` | 全部 | 要计算的指标 (如 --metrics diversity entropy) |
| `--model` | None | 模型名 (用于 ppl/ifd) |
| `--similarity_type` | jaccard | 相似度类型 |
| `--compute_on` | both | 计算对象 (prompt/answer/both) |

## 指标说明

### 1. 统计指标 (statistics)

| 指标 | 说明 |
|------|------|
| `num_samples` | 样本数量 |
| `avg_prompt_length` | 平均 prompt 长度 (字符) |
| `std_prompt_length` | prompt 长度标准差 |
| `min_prompt_length` / `max_prompt_length` | prompt 长度范围 |
| `avg_response_length` | 平均 response 长度 |
| `std_response_length` | response 长度标准差 |
| `min_response_length` / `max_response_length` | response 长度范围 |
| `num_unique_data_sources` | 不同数据源数量 |
| `num_unique_abilities` | 不同能力类型数量 |

### 2. 质量指标 (quality)

| 指标 | 说明 |
|------|------|
| `answer_coverage` | 有答案的样本比例 |
| `format_validity` | 格式有效样本比例 |
| `prompt_uniqueness` | prompt 唯一性 |

### 3. 多样性指标 (diversity)

基于相似度函数计算，数据多样性 = 1 - 平均相似度。

#### 支持的相似度函数

| 类型 | 说明 | 公式 |
|------|------|------|
| `jaccard` | 词级 Jaccard | $\frac{|A \cap B\|}{\|A \cup B|}$ |
| `levenshtein` | 编辑距离 | $1 - \frac{\text{edit\_dist}}{\max(\|A\|, \|B\|)}$ |
| `cosine` | TF-IDF 余弦 | $\cos(\vec{A}, \vec{B})$ |
| `jaro_winkler` | Jaro-Winkler | 基于前缀修正的相似度 |
| `ngram` | N-gram | 基于字符 n-gram 的相似度 |
| `bertouch` | BERT 语义 | 基于 sentence-transformers 的语义相似度 |
| `bleu` | BLEU | 词级 precision |
| `rouge` | ROUGE-L | 基于最长公共子序列 |

#### diversity 输出指标

| 指标 | 说明 |
|------|------|
| `diversity_score` | 多样性分数 = 1 - 平均相似度 |
| `avg_similarity` | 平均相似度 |
| `min_similarity` | 最小相似度 |
| `max_similarity` | 最大相似度 |
| `std_similarity` | 相似度标准差 |
| `similarity_type` | 使用的相似度类型 |

#### compute_on 选项

- `prompt`: 只在 prompt 上计算
- `answer`: 只在 answer 上计算
- `both`: 在 prompt + answer 上计算

### 4. 熵指标 (entropy)

| 指标 | 说明 |
|------|------|
| `avg_char_entropy` | 平均字符级熵 |
| `std_char_entropy` | 字符级熵标准差 |
| `min_char_entropy` / `max_char_entropy` | 字符级熵范围 |
| `avg_word_entropy` | 平均词级熵 |
| `std_word_entropy` | 词级熵标准差 |
| `min_word_entropy` / `max_word_entropy` | 词级熵范围 |

#### Shannon 熵公式

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

### 5. PPL 指标 (ppl)

使用语言模型计算文本的困惑度。

| 指标 | 说明 |
|------|------|
| `avg_ppl` | 平均困惑度 |
| `std_ppl` | 困惑度标准差 |
| `min_ppl` / `max_ppl` | 困惑度范围 |

- 高 PPL: 模型不确定 (可能是噪声文本)
- 低 PPL: 模型确定 (可能是通用/模板文本)

### 6. IFD 指标 (ifd)

Instruction Following Difficulty，衡量指令跟随难度。

$$IFD = \frac{\text{Loss}_{\text{with\_prompt}}}{\text{Loss}_{\text{without\_prompt}}}$$

| 指标 | 说明 |
|------|------|
| `avg_ifd` | 平均 IFD |
| `std_ifd` | IFD 标准差 |
| `min_ifd` / `max_ifd` | IFD 范围 |

- 高 IFD: 答案难以从 prompt 预测 (学习价值高)
- 低 IFD: 答案容易预测 (可能是通用答案)

## Lib 模块

### data_obs.py

- `DatasetMetrics`: 数据指标 dataclass
- `DataSplitter`: 数据分割器，将数据均匀分成 n 份，计算指标
- `GPUAllocator`: GPU 分配器
- `ResultCollector`: 训练结果收集器

### data_metrics.py

- `compute_data_statistics()`: 统计指标
- `compute_data_quality_metrics()`: 质量指标
- `compute_all_data_metrics()`: 全部基础指标

### advanced_metrics.py

- `compute_dataset_diversity()`: 多样性指标 (支持多种相似度)
- `compute_dataset_entropy()`: 熵指标
- `compute_ppl_metrics()`: PPL 指标
- `compute_ifd_metrics()`: IFD 指标
- `compute_advanced_data_metrics()`: 全部高级指标
- `SimilarityType`: 相似度类型枚举
- `DiversityConfig`: Diversity 配置
- `get_similarity_function()`: 获取相似度函数实例
- `compute_diversity_with_similarity()`: 统一 diversity 计算接口

### training_pipeline.py

- `TrainingPipeline`: 训练管线，管理多 GPU 训练任务

### analysis_pipeline.py

- `CorrelationAnalyzer`: 相关性分析
- `AnalysisVisualizer`: 结果可视化

## 数据格式

输入数据需包含字段：
- `prompt`: 问题
- `reward_model.ground_truth`: 答案
- `extra_info.answer`: 答案 (可选)
