# InternBootcamp 评测指南

为了快速评估模型在不同训练场环境（bootcamp）中的性能，可以使用 `run_eval.py` 脚本。该脚本支持多种配置选项，能够灵活地适配不同的测试需求。

---

#### 示例运行命令

以下是一个完整的示例命令，展示了如何运行脚本进行评测。

```bash
cd InternBootcamp
python examples/unittests/run_eval.py \
    --url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model_name r1_32B \
    --test_dir /path/to/test_dir \
    --max_concurrent_requests 128 \
    --template r1 \
    --max_tokens 32768 \
    --temperature 0 \
    --timeout 6000 \
    --api_mode completion \
    --max_retries 16 \
    --max_retrying_delay 60
```

---

#### 参数说明

以下是脚本支持的主要参数及其含义：

| 参数名                  | 类型       | 示例值                                    | 描述                                                                 |
|-------------------------|------------|-------------------------------------------|----------------------------------------------------------------------|
| `--url`                 | str        | `http://127.0.0.1:8000/v1`                | OpenAI API 的基础 URL。                              |
| `--api_key`             | str        | `EMPTY`                                  | 访问模型服务所需的 API 密钥。默认为 `EMPTY`。                        |
| `--model_name`          | str        | `r1_32B`                                 | 使用的模型名称，例如 `r1_32B` 或其他自定义模型名称。                 |
| `--test_dir`            | str        | `/path/to/test_dir`                      | 包含测试数据的目录路径，目录中应包含多个 JSONL 文件。                |
| `--max_concurrent_requests` | int     | `128`                                    | 全局最大并发请求数量。                                               |
| `--template`            | str        | `r1`                                     | 预设的对话模板（如 `r1`, `qwen`, `internthinker`, `chatml`）。       |
| `--max_tokens`          | int        | `32768`                                  | 模型生成的最大 token 数量。                                          |
| `--temperature`         | float      | `0`                                      | 控制生成文本的随机性，值越低生成结果越确定性。                       |
| `--timeout`             | int        | `6000`                                   | 请求超时时间（毫秒）。                                               |
| `--api_mode`            | str        | `completion`                             | API 模式，可选值为 `completion` 或 `chat_completion`。               |
| `--sys_prompt`          | str        | `"You are an expert reasoner..."`        | 系统提示内容，仅在 `api_mode` 为 `chat_completion` 时生效。          |
| `--max_retries`         | int        | `16`                                     | 单个请求失败重试次数。                                              |
| `--max_retrying_delay`  | int        | `60`                                     | 最大重试延迟时间（秒）。                           |

##### 参数关系
- `--api_mode`为`chat_completion`时，`--sys_prompt`参数才有效。
- `--api_mode`为`completion`时，`--template`参数才有效。
- `--template` 可以选择预定义的 `TEMPLATE_MAP` 中的值：`r1`, `qwen`, `internthinker`, `chatml`。
- `--sys_prompt` 如果未提供，则使用模板中默认的系统提示（如有）。
---

#### 输出结果

评测结果将保存在 `examples/unittests/output/{model_name}_{test_dir}_{timestamp}` 目录下，具体包括以下内容：

1. **详细结果**：
   - 每个 JSONL 文件的评测结果会保存在 `output/details/` 子目录中，文件名为原始 JSONL 文件名。
   - 每条记录包含以下字段：
     - `id`: 样本 ID。
     - `prompt`: 输入提示。
     - `output`: 模型生成的输出。
     - `output_len`: 输出长度（token数）。
     - `ground_truth`: 真实答案。
     - `score`: 评分结果（由 `verify_score` 方法计算）。
     - `extracted_output`: 提取的输出内容（由 `extract_output` 方法提取）。

2. **元信息**：
   - 元信息保存在 `output/meta.jsonl` 中，包含每个沙箱的平均分和平均输出长度。

3. **汇总报告**：
   - 总结报告保存为 Excel 文件，路径为 `output/{model_name}_scores.xlsx`，包含以下内容：
     - 每个沙箱的平均分和平均输出长度。
     - 所有沙箱的总平均分和总平均输出长度。

4. **进度日志**：
   - 进度保存在 `output/progress.log` 中，实时显示每个数据集的处理进度及预计剩余时间。

5. **参数配置**：
   - 当前运行的完整参数配置保存在 `output/eval_args.json` 中，便于复现实验。

---

#### 注意事项

1. **并发设置**：
   - 根据机器性能与评测集情况调整 `--max_concurrent_requests` 参数，避免过多并发导致资源耗尽。

2. **URL存活检查**：
   - 在开始评测前，脚本会自动检测模型服务是否启动并注册了指定的 `model_name`。
   - 若服务未就绪，脚本会等待最多 60 分钟（默认配置），每隔 60 秒尝试连接一次。

3. **错误处理机制**：
   - 每个请求最多重试 `--max_retries` 次，采用指数退避策略（最多等待 `--max_retrying_delay` 秒）。
   - 如果所有重试均失败，程序将抛出异常并终止当前样本处理。

---

#### 示例输出目录结构

运行完成后，输出目录结构如下：

```
examples/unittests/output/
└── {model_name}_{test_dir}_{timestamp}/
    ├── details/
    │   ├── test_file1.jsonl
    │   ├── test_file2.jsonl
    │   └── ...
    ├── meta.jsonl
    ├── eval_args.json
    ├── progress.log
    └── {model_name}_scores.xlsx
```

---
