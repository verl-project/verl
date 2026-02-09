# Cluster Analyse - RL Timeline 可视化工具

## 一、简介

Cluster Analyse 是一个强化学习性能数据快速分析的可视化工具，基于 VeRL 框架采集的 profiling 数据进行解析，生成强化学习各阶段的 Timeline 图表。

### 主要功能

- **数据解析**：支持解析 VeRL 框架采集的多格式 profiling 数据
- **并行处理**：利用多进程并行解析多个 Rank 的性能数据，提升处理效率
- **Timeline 可视化**：生成交互式 Timeline 甘特图，直观展示各 Rank 的事件分布
- **性能分析**：通过 Timeline 图表观察卡间负载不均衡、推理长尾等问题，帮助性能调优

### 软件依赖

- Python
- Pandas
- Plotly
- NumPy

## 二、快速使用

### 2.1 安装依赖

```bash
pip install pandas plotly numpy
```

### 2.2 采集 Profiling 数据

使用 VeRL 框架采集性能数据，详细参考：

[VeRL NPU Profiling 教程](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/ascend_profiling_zh.rst)

### 2.3 执行分析脚本

```bash
python cluster_analysis.py --input-path <profiling_data_path> --output-path <output_path>
```

## 三、命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-path` | `test` | Profiling 数据的原始路径 |
| `--profiler-type` | `mstx` | 性能数据种类（当前仅支持 mstx） |
| `--data-type` | `text` | 性能文件类型（当前仅支持 text） |
| `--output-path` | `test` | 输出路径 |
| `--vis-type` | `html` | 可视化类型（当前仅支持 html） |
| `--rank-list` | `all` | Rank ID 列表（当前仅支持 "all"） |

### 使用示例

```bash
# 基本使用
python cluster_analysis.py --input-path ./data --output-path ./output
```

## 四、输出说明

工具会在指定的输出路径下生成 HTML 文件（默认为 `rl_timeline.html`），包含：

- **交互式 Timeline 甘特图**：展示各 Rank 在不同时间段的事件分布
- **悬停信息**：鼠标悬停显示事件详细信息（名称、开始/结束时间、持续时间等）
- **排序功能**：支持按默认排序或按 Rank ID 排序
- **缩放与导航**：支持图表缩放和时间轴导航

### 图表交互功能

1. **Hover 模式切换**：
   - "Hover: Current Only" - 仅显示当前悬停的事件信息
   - "Hover: All Ranks" - 显示所有 Rank 在同一时间点的信息

2. **Y 轴排序切换**：
   - "Sort: Default" - 默认排序
   - "Sort: By Rank ID" - 按 Rank ID 排序

3. **导出图片**：点击右上角相机图标可导出 PNG 图片

## 五、注意事项

1. 当前版本仅支持 `mstx` 类型的 profiler 数据
2. RL 分析功能当前仅支持处理所有 Rank（`--rank-list` 参数暂不支持过滤）
3. 需至少采集 level0 及以上数据
4. 采用离散模式采集 `discrete=True`
5. 数据格式要求：
   - 输入路径下需包含 `*_ascend_pt` 目录
   - 每个 ascend_pt 目录下需包含 `profiler_info_*.json` 文件
   - trace_view.json 文件位于 `ASCEND_PROFILER_OUTPUT` 子目录中