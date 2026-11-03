# Ascend CI / 本地底软环境对照

使用训练时的 Python、容器和环境变量，在训练启动前运行。采集器仅依赖 Python 标准库，
不会安装软件、改驱动、改模型、分配测试 tensor、调用 synchronize 或执行训练。
它会在一个有超时限制的独立进程里导入 torch / torch_npu / acl / vLLM 等，并查询设备。
导入和文件哈希仍会改变启动耗时及文件缓存；这不是对时序完全无扰动的证明。

## 采集

在实际运行的 verl 源码根目录执行：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# 与此 Qwen3-VL nightly launcher 相同的导出值：
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

python3 tests/special_npu/env_diagnostics/collect.py \
  --label local --repo "$PWD" \
  --asset /root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct \
  --asset /root/data/geo3k/train.parquet \
  --asset /root/data/geo3k/test.parquet \
  --output /root/results/ascend-env-local
```

使用独立下载包时，把命令中的脚本路径改成 `collect.py`。
输出目录必须不存在。`--asset` 会对所有文件做完整 SHA256，读取 8B 模型可能需要数分钟。
可重复传入实际 Hydra 配置文件、启动脚本、额外 checkpoint 或数据文件。
非标准 CANN 安装目录用 `--cann-root /path/to/cann`，可重复指定。
若导入时间较长，可以增加 `--import-timeout 300`。超时会记录错误并保留最后完成的导入结果。

## CI 接入与触发

`nightly_ci_ascend` 增加普通 `pull_request` 事件（目标 main，带源码／配置／诊断脚本路径过滤），不需要标签，不增加手动 dispatch 入口。

- PR 创建、更新或重新打开时，命中路径过滤即可运行 `nightlyCI_grpo_qwen3_vl_8b_Instruct_fsdp2_vllm_ascend`。使用正常 PR merge ref，在合并前运行完整 Qwen3-VL nightly：16 卡、train batch 32、mini batch 16、launcher 默认 15 步、原有 baseline 下载和检查。
- 该 job 无需等待 E2E 的 LLM job。原有 1-step `e2e_ascend` 文件保持不变，不额外接入这些诊断。
- 定时事件保持不变；其他 nightly jobs 的条件和内容完全不变，因此 PR 事件下它们跳过。
- 如果 PR CI 等待批准，按仓库既有流程请维护者批准，并在 PR 模板指定的 ci-request 渠道申请 CI。已有运行可使用 Re-run jobs，但须确认对应最新 PR 提交；重跑历史 Action 不会带上新增日志。

Artifact 名为 `ascend-diagnostics-<run id>-<attempt>`，保留 14 天：

- `environment/report.json`：环境、源码、模型和数据指纹。
- `weights/<host>-<pid>.jsonl`：各 worker 的阶段记录，包含 rank、模型实例标识、实际导入路径、动态库路径、参数和 buffer 元信息。
- `weight-summary.json`：阶段计数、非有限值和已记录 offload/reload 前后哈希不匹配；metadata 模式明确标记未校验数值，不输出通过结论。
- `training-console.log` 及原始 launcher 日志：完整训练输出，避免 Actions 控制台截断。
- 下载到本地的 baseline 和 checker：记录实际比较依据。

上传步骤使用 always()；训练失败也尝试上传，但宿主机丢失或强制终止仍可能无法上传。

### 权重诊断的边界

默认 `metadata` 模式不读取张量数值，不增加 tensor copy、归约、同步或 collective；仍有 Python 和文件 I/O 开销，不能声称完全不影响时序。未设置 `VERL_WEIGHT_DIAGNOSTICS_DIR` 时关闭。

记录 checkpoint 输入、set_model_state_dict 后、buffer 广播后、CPU offload 前后、设备 reload 前后、actor 导出前、vLLM 收到 bucket 和加载完成后的状态。每进程每阶段每模型默认最多 2 次；bucket 仅记录前 2 个，不能据此声称覆盖全部传输。通过 `VERL_WEIGHT_DIAGNOSTICS_LIMIT` 调整。

单独运行 `VERL_WEIGHT_DIAGNOSTICS_MODE=values` 才会同步设备、逐张量拷贝到 CPU，计算完整本地 shard SHA256 和非有限值数目。这是侵入性对照，可能掩盖竞态并增加 CPU 内存和耗时。CPU offload 前后同 rank、同模型、同名、同形状、同 dtype 的参数哈希应一致；checkpoint 输入是完整权重，FSDP 输出是 shard，vLLM 还会 TP 切分／融合，不能直接比较跨这些布局的哈希。该模式不自动证明 checkpoint 到 rollout 的端到端一致性。

初始化前的 meta tensor 不计算哈希。仅有元信息不能确认权重没有损坏；先用 metadata 复现，再决定是否运行 values 对照。

默认镜像 digest 为 `unknown`：容器内不能可靠推导宿主机实际拉取的 image digest。
如果能在宿主机或启动平台查到它，通过 `--image-digest sha256:...` 或环境变量
`REPRO_IMAGE_DIGEST` 传入。镜像 tag 相同不能作为镜像相同的证据。

## 采集内容

| 类别 | 内容 |
| --- | --- |
| 驱动 / 固件 / CANN | 安装及版本文件、latest 的实际目标、关键运行库内容哈希、额外 OPP 路径 |
| 硬件与容器 | npu-smi、拓扑、设备属性、CPU/NUMA、共享内存、meminfo、ulimit、cgroup 限制、内核模块 |
| Python 环境 | 解释器及真实路径、sys.path、全部包版本和安装位置、editable/direct_url 来源 |
| 实际导入 | torch/torch_npu/acl/Transformers/vLLM/vllm_ascend/Ray/ModelOpt/MindSpeed 等的来源和异常 |
| 内容一致性 | 相关包的 Python 源码、JSON/YAML 和原生 .so 哈希，Git HEAD、dirty diff 哈希和 untracked 列表 |
| 动态库 | 导入进程 `/proc/self/maps` 中实际映射库的绝对路径、realpath、SHA256；ldconfig 搜索表 |
| 运行配置 | Ascend/HCCL/torch/vLLM/Ray/线程/内存相关环境变量，PATH、PYTHONPATH、LD_LIBRARY_PATH 等 |
| 输入 | 模型权重、tokenizer/processor/config、自定义代码、数据或其他 --asset 的完整哈希 |

不枚举完整环境变量、不读取认证文件、不收集进程参数；已知敏感环境变量和 URL 凭据会脱敏。
报告仍包含机器/安装路径、CI runner 名称等定位信息。

边界：

- 动态库列表属于独立导入进程，**不是实际 Ray worker 首次 forward 后的库全集**。
  延迟加载的自定义 kernel 或 worker 内部改变的环境可能仍需第二轮定点采集。
- CANN 只哈希选定的关键库和版本文件，并非整个 OPP kernel 仓库。
  若怀疑 custom OPP 内容，可另传 `--asset /path/to/custom/opp` 做全量哈希。
- 文件扫描上限每棵目录 100000 个，达到上限会记录 `coverage.truncated=true`；
  内部目录软链接不递归展开，文件软链接会记录实际目标并哈希。
- 缺失工具、读取权限、导入异常和超时都会记录；没有采到不代表相同。
- 静态版本、资源占用或 CPU 数目不同本身不证明是精度根因。

## 对比

下载 CI artifact，把两边的 `report.json` 交给：

```bash
python3 compare.py /path/to/ci/report.json /root/results/ascend-env-local/report.json \
  --output /root/results/ascend-env-diff --full
```

先看 `diff.txt` 的 SOFTWARE / CONFIG / FILE CONTENT，再看 SYSTEM / RUNTIME。
`diff.json` 是结构化结果。`--full` 还输出原始报告全部字段的 `full-diff.json`，
包含安装路径、采集时间和内存瞬时值等噪声。

包/模型文件按相对路径和内容比对，避免仅因安装路径或 mtime 不同就误报。
实际加载库按 basename + 所有不同哈希对照，避免相同 SONAME 的多个版本相互覆盖。
完整原始路径仍保存在报告中。

建议优先检查：torch_npu 与 CANN/驱动的实际二进制、vLLM/vllm_ascend 的源码/脏改动、
ModelOpt/MindSpeed 的来源、任务队列/精度/编译相关环境变量、模型和 processor 内容。
然后查看 pinned host memory 相关资源限制及是否有残留进程/资源压力。

## 验证范围

`python3 -m unittest discover -s tests/utils -p 'test_*diagnostics_on_cpu.py'` 是 CPU 单元检查，验证脱敏、文件哈希变化、路径无关比较、
重复库名和缺失数据的处理。开发机器没有 Ascend；NPU/CANN 命令及底软一致性需要实机报告验证。
