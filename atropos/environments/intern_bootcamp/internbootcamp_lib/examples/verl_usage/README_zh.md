# VERL USAGE

本目录包含在VERL框架下加入bootcamp数据集进行训练的示例和数据处理脚本。以下是文件及其功能的简要概述：

---

## 目录结构

```
/InternBootcamp/examples/verl_usage
├── run_bootcamp.sh
├── bootcamp_reward_for_verl.py
└── verl_data_preprocess.py
```

---

### 1. `run_bootcamp.sh`

这是一个 Shell 脚本，用于设置和运行 VERL 训练实验。主要功能包括：

- 设置实验名称、数据路径等基本参数。
- 安装必要的依赖项并配置环境。
- 启动 VERL 训练器，并配置实验参数（如模型路径、批处理大小、学习率等）。

#### 使用方法
- 启动前，确保重要参数已配置正确，如`internbootcamp_path`,`model.path`,`trainer.default_local_dir`等
```bash
./run_bootcamp.sh
```

---

### 2. `bootcamp_reward_for_verl.py`

该Python脚本用于训练中动态计算 bootcamp 的奖励分数。需嵌入到verl框架中使用。

该脚本提供了一个函数 _default_compute_score，用于根据`data_source`、`model_output`和`groud_truth`(这里指验证model_output所需的各项参数组成的字典)计算分数。支持多种数据源，包括：


- **非训练场数据集**：通过 gsm8k, math, prime_math, prime_code 等模块计算分数。

- **训练场数据集**：动态导入外部训练场模块以计算分数。

- 如果数据源不受支持，则抛出 NotImplementedError。

### 3. `verl_data_preprocess.py`

该 Python 脚本用于将原始数据转换为 VERL 兼容的格式。主要功能包括：

- **递归处理输入目录或文件**：将 `.jsonl` 原始数据文件转换为 `.parquet` 格式，并保留目录结构。
- **数据划分**：根据文件路径判断数据属于 `train` 或 `test` 划分。
- **随机打乱与合并**：将多个 `.parquet` 文件合并为一个文件，并随机打乱数据。
- **元数据添加**：为每条数据添加 `split` 和其他必要信息。

#### 输出目录结构
```merged
examples/bootcamp_generator_outputs/<time_stamp>_for_verl_merged/
├── train/
│   └── bootcamps.parquet
└── test/
    └── bootcamps.parquet
```
```not merged
examples/bootcamp_generator_outputs/<time_stamp>_for_verl/
├── train/
│   ├── bootcamp1.parquet
│   ├── bootcamp2.parquet
│   └── ...
└── test/
    ├── bootcamp1.parquet
    ├── bootcamp2.parquet
    └── ...    
```

#### 示例命令
```bash
python examples/verl_usage/verl_preprocess.py --src examples/bootcamp_generator_outputs/2025-03-07-16:48:28
```

此命令将 `examples/bootcamp_generator_outputs/2025-03-07-16:48:28` 中的所有 `.jsonl` 文件转换为 VERL 兼容的 `.parquet` 文件，并输出到默认目录：
```
examples/bootcamp_generator_outputs/2025-03-07-16:48:28_for_verl_merged/
examples/bootcamp_generator_outputs/2025-03-07-16:48:28_for_verl/
```

---

### 4. 工作流程总结

1. **数据预处理**：
   - 使用 `verl_data_preprocess.py` 将原始 `.jsonl` 数据转换为 VERL 兼容的 `.parquet` 格式。
   - merged输出目录为 `<src>_for_verl_merged`，包含 `train/bootcamps.parquet` 和 `test/bootcamps.parquet` 文件。
   - not merged输出目录为 `<src>_for_verl`，包含多个 `.parquet` 文件，每个文件对应一个 `.jsonl` 文件。

2. **将bootcamp_reward_for_verl.py嵌入verl框架**
即如下代码段
```python
    elif data_source.startswith("bootcamp/"):
        try:
            import importlib
            import json
            bootcamp_name = data_source.split('/')[1]
            class_name = bootcamp_name[0].upper() + bootcamp_name[1:] + "bootcamp"
            module = importlib.import_module(f"internbootcamp")
            ground_truth = json.loads(ground_truth)
            return getattr(module, class_name).verify_score(solution_str, ground_truth, format_score=0)
        except Exception as e:
            print("Something woring with bootcamp reward because of ",e)
            return 0
```
将其中相关代码嵌入VeRL框架中即 `/verl/utils/reward_score/__init__.py`
```
/verl
└── utils
    └── reward_score
        └── __init__.py
```
3. **启动训练实验**：
   - 在 `run_bootcamp.sh` 中完成设置 `experiment_name`,`internbootcamp_path`,`train_files`,`test_files`,`actor_rollout_ref.model.path`,`trainer.default_local_dir`等重要实验参数。
   - 运行 `run_bootcamp.sh` 启动 VERL 训练。

---

通过以上工具和流程，您可以高效地准备数据并运行 VERL 实验。
