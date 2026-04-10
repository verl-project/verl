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

## Lib 模块

### data_obs.py

- `DatasetMetrics`: 数据指标 dataclass
- `DataSplitter`: 数据分割器，将数据均匀分成 n 份，计算指标
- `GPUAllocator`: GPU 分配器
- `ResultCollector`: 训练结果收集器

### data_metrics.py

- `compute_all_data_metrics()`: 计算数据集统计和质量指标
  - 统计指标: token 数、prompt 长度、response 长度等
  - 质量指标: 重复率、空答案率、大写/小数点比例等

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

## 指标说明

### 统计指标

| 指标 | 说明 |
|------|------|
| `num_samples` | 样本数量 |
| `avg_prompt_length` | 平均 prompt 长度 (token) |
| `std_prompt_length` | prompt 长度标准差 |
| `min_prompt_length` | 最短 prompt |
| `max_prompt_length` | 最长 prompt |
| `avg_response_length` | 平均 response 长度 (token) |
| `std_response_length` | response 长度标准差 |
| `min_response_length` | 最短 response |
| `max_response_length` | 最长 response |

### 数据源/能力指标

| 指标 | 说明 |
|------|------|
| `num_unique_data_sources` | 不同数据源数量 |
| `num_unique_abilities` | 不同能力类型数量 |

### 质量指标

| 指标 | 说明 |
|------|------|
| `answer_coverage` | 有答案的样本比例 (extra_info.answer 非空) |
| `format_validity` | 格式有效样本比例 (有 prompt 和 reward_model) |
| `prompt_uniqueness` | prompt 唯一性 (不同 prompt 数 / 总数) |

### 高级指标

| 指标 | 说明 |
|------|------|
| `diversity_score` | 数据多样性分数 = 1 - 平均相似度 |
| `avg_similarity` | 平均 Jaccard 相似度 |
| `min_similarity` / `max_similarity` | 相似度最小/最大值 |
| `avg_char_entropy` / `std_char_entropy` | 字符级熵均值/标准差 |
| `avg_word_entropy` / `std_word_entropy` | 词级熵均值/标准差 |

#### 数学公式

**Jaccard 相似度**:
$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**多样性分数**:
$$\text{diversity} = 1 - \bar{S}$$
其中 $\bar{S}$ 是所有文本对的平均相似度。

**Shannon 熵**:
$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$
其中 $p(x_i)$ 是字符/词的频率概率。