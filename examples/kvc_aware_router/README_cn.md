# llm-router 推理快速开始

## 什么是 llm-router

llm-router 提供 Agentic RL 场景下基于 KV cache 状态 + 负载感知的智能路由。它作为 verl 的可插拔负载均衡器，用于多副本（data-parallel）vLLM 推理，替代默认的 `global_sticky_inflight` 路由器。

核心组件：
- `KVCAwareBalancer` — 路由框架，管理组件生命周期与路由决策
- `Collector` — Transport + Decoder 组合，采集用于决策的指标，如vLLM 的 KV 事件与 Prometheus 指标
- `Strategy` — 评分策略，基于 KV cache 命中率与负载计算综合得分
- `Store` — 单例存储，缓存采集到的指标与 KV block 状态

---

## 部署和运行指南

以下步骤从零开始部署 llm-router 推理环境。

## 1. 克隆项目

```bash
git clone https://github.com/verl-project/uni-agent.git
```

## 2. 创建 Docker 容器

```bash
# 可选环境变量（带默认值）：CONTAINER_NAME=hgq-swe  IMAGE_NAME=verlai/verl:vllm018.dev1  SHM_SIZE=10g
# DATA_DIR: 宿主数据盘路径（模型/数据集/wheels 都放这），按实际环境替换
CONTAINER_NAME=swe-xxx IMAGE_NAME=verlai/verl:vllm018.dev1 SHM_SIZE=10g \
docker run -d \
  --name ${CONTAINER_NAME:-hgq-swe} \
  --gpus all \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --shm-size=${SHM_SIZE:-10g} \
  -v <DATA_DIR>:<DATA_DIR> \
  -v /tmp:/tmp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  --entrypoint sleep \
  ${IMAGE_NAME:-verlai/verl:vllm018.dev1} \
  infinity

# 验证 GPU 可见
docker exec ${CONTAINER_NAME:-hgq-swe} nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

> `<DATA_DIR>` 是宿主数据盘路径（模型、数据集、swe-rex wheels 都放这），按实际环境替换（如 `/data1`）。该路径会挂载进容器，且需与 `agent_config_localdocker.yaml` 里 swe-rex wheels 的挂载源路径一致（见 §4.5）。

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CONTAINER_NAME` | `hgq-swe` | 容器名称 |
| `IMAGE_NAME` | `verlai/verl:vllm018.dev1` | 镜像名称 |
| `SHM_SIZE` | `10g` | 共享内存大小 |

## 3. 进入 Docker 容器并安装依赖

```bash
docker exec -it swe-xxx bash
cd /path/to/uni-agent

# 初始化 git submodule + 安装 verl 及依赖（清华源）
git submodule update --init --recursive
pip install --no-deps -e verl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install swe-rex loguru pydantic pydantic_settings boto3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir swebench -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 准备数据集

```bash
# 使用默认值 (modal) —— local docker 沙箱用 modal 即可
DEPLOYMENT=modal python examples/data_preprocess/swe_bench_verified.py --local-save-dir examples/kvc_aware_router

# 指定部署后端
DEPLOYMENT=vefaas python examples/data_preprocess/swe_bench_verified.py --local-save-dir examples/kvc_aware_router
```

DEPLOYMENT 决定写入 parquet 的沙箱镜像名：

| 值 | 生成的镜像名 | 适用场景 |
|------|------|------|
| `modal` | `swebench/sweb.eval.x86_64.*` | local docker / modal 部署 |
| `vefaas` | 阿里云 veFaaS 镜像 | 仅 veFaaS 部署 |
| `local` | 尚未实现 | — |

> **使用 local docker 沙箱时，`DEPLOYMENT=modal` 即可**（默认值），docker 会自动拉取 `swebench/sweb.eval.x86_64.*` 镜像。

输出文件：`examples/kvc_aware_router/swe_bench_verified_<deployment>.parquet`

## 4.5 预下载 swe-rex wheels + 拉取 SWE-bench 镜像（首次必做）

多并发推理前需解决两个问题，否则沙箱起不来导致大量样本 fail：

### 问题 1：500 沙箱并发 pip 打爆镜像源

每个沙箱启动都要 `pip install swe-rex`，500 个沙箱并发 pip 会把清华源打满（超时/卡死）。

**解决**：预下载 swe-rex + 全部依赖 wheels 到本地目录，沙箱挂载后 offline 安装（秒级，不走网络）：

```bash
# <WHEELS_DIR> 放在 §2 挂载的 <DATA_DIR> 下，使其容器内可见，如 /data1/swe_wheels
pip download swe-rex -d <WHEELS_DIR> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`agent_config_localdocker.yaml` 已配置把该目录挂载到沙箱 `/wheels`（只读）+ `--find-links /wheels`（清华源作 fallback，兼容 pydantic-core 等平台相关包）。注意 yaml 里的挂载源路径要和实际的 `<WHEELS_DIR>` 一致。

### 问题 2：从 parquet 找镜像列表 + 国内拉取 SWE-bench 镜像

**找镜像列表**：沙箱镜像名写在 parquet 的 `extra_info` 字段里（格式 `swebench/sweb.eval.x86_64.<instance>`）。用 pyarrow 解析：

```python
import json, pyarrow.parquet as pq
t = pq.read_table("examples/kvc_aware_router/swe_bench_verified_modal.parquet")
imgs = set()
for e in t.column("extra_info").to_pylist():
    e = json.loads(e) if isinstance(e, str) else e
    def find(o):
        if isinstance(o, dict):
            for v in o.values(): find(v)
        elif isinstance(o, str) and "sweb.eval" in o:
            imgs.add(o)
    find(e)
print(imgs)  # 完整镜像列表
```

**国内拉取**：Docker Hub 国内拉 `swebench/*` 镜像超时。用火山引擎 CR（国内快）拉源镜像再 tag 成 parquet 期望的格式：

```bash
# 火山引擎 CR 源：enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/<instance>:v2
# tag 成 parquet 期望的：swebench/<instance>:latest
docker pull enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/sweb.eval.x86_64.<instance>:v2
docker tag enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/sweb.eval.x86_64.<instance>:v2 swebench/sweb.eval.x86_64.<instance>:latest
```

对上面找到的每个镜像循环执行 pull+tag（已 tag 的可跳过）。可自行写循环脚本，或手动拉需要的子集。

## 5. 运行推理

`examples/kvc_aware_router/run_infer.sh` 是 `parallel_infer.py` 的薄包装。**默认参数（在 parallel_infer.py 中定义）= 2 卡单副本冒烟测试（1 样本）**。前 3 个是位置参数，其余通过 `$@` 透传给 `parallel_infer.py`。

```bash
# 冒烟测试（默认：2 卡、TP=1、1 样本）
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B

# 指定模型 + 数据集 + agent 配置
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B /path/to/dataset.parquet my_config.yaml

# 全量 8-GPU data-parallel（覆盖默认参数）
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B \
    --num-workers 8 --n-gpus-per-node 8 --tensor-parallel-size 2 \
    --max-num-seqs 64 --max-samples -1 --prompt-length 31744

# 带 MooncakeStoreConnector（跨副本 KV 共享，需先起 mooncake_master）
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B --enable-mooncake

# KVCAware router 始终启用（parallel_infer.py 内硬编码）。
# Ascend（vllm-ascend）后端 —— 选 MooncakeConnectorStoreV1 + mem 0.8：
bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B \
    --device ascend --enable-mooncake
```

位置参数：

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `MODEL_PATH` | 第 1 个 | **无（必传）** | 模型路径 |
| `DATA_PATH` | 第 2 个 | 同目录 `swe_bench_verified_modal.parquet` | 数据集路径 |
| `AGENT_CONFIG` | 第 3 个 | 同目录 `agent_config_localdocker.yaml` | Agent 配置路径 |

主要 CLI 参数（透传给 `parallel_infer.py`，详见 `--help`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-workers` | `1` | agent rollout worker 数（全量用 8） |
| `--n-gpus-per-node` | `2` | 每节点 GPU 数（全量用 8） |
| `--tensor-parallel-size` | `1` | tensor parallel（全量用 2） |
| `--max-num-seqs` | `256` | 每实例 vLLM 并发序列（24GB 卡用 64） |
| `--max-samples` | `-1` | 样本数（-1 = 全量） |
| `--prompt-length` | `4096` | prompt 长度（全量 1-token 退化修复用 31744） |
| `--response-length` | `8192` | response 长度 |
| `--max-model-len` | 空=引擎默认 | 模型上下文长度 |
| `--n` | `1` | 每 prompt rollout 数（全量用 4） |
| `--enable-mooncake` | 关 | 附加 mooncake connector（跨副本 KV 共享） |
| `--mooncake-config-path` | `mooncake_config.json` | mooncake 配置路径（配合 `--enable-mooncake`） |
| `--device` | `gpu` | 后端：`gpu`→MooncakeStoreConnector、mem 0.9；`ascend`→MooncakeConnectorStoreV1、mem 0.8 |

> `CUDA_VISIBLE_DEVICES` 在 run_infer.sh 调用前通过 shell 环境设（如 `CUDA_VISIBLE_DEVICES=6,7 bash run_infer.sh ...`）。concurrency 在 agent_config yaml 中配置。

> **PROMPT_LEN + RESPONSE_LEN 的 1-token 退化**：verl 的 max_tokens = min(response_length, prompt_length+response_length-prompt)，多轮 prompt 累积达到该和时 max_tokens 塌缩到 1。全量跑时把和控制在模型原生上下文以内（如 Qwen3-8B 40960 → 用 31744+8192=39936）。

run_infer.sh 会在启动前检查 MODEL_PATH / DATA_PATH / AGENT_CONFIG 是否存在，缺失则报错退出。

> **前置条件（全量多并发）**：必须先跑过 Step 4.5（预下载 wheels + 拉镜像），否则沙箱起不来。

> **日志重定向**：
> ```bash
> setsid nohup bash examples/kvc_aware_router/run_infer.sh /path/to/Qwen3-4B > logs/run.log 2>&1 &
> ```

## 6. 已知问题：transformers / numpy 异常

在 Docker / 内核 / 系统版本较旧的环境（如 Docker 20.10、内核 4.15、Ubuntu 18.04）下，同一镜像会触发以下两个运行时异常，导致推理起不来；在较新环境（如 Docker 26.x、内核 5.4、Ubuntu 20.04）上开箱即用，可跳过本节。根因是宿主运行时环境差异，非包/版本本身。

| 现象 | 根因 | 解决方案（容器内执行一次） |
|------|------|--------------------------|
| `import transformers` 报 `Backend should be defined in the BACKENDS_MAPPING. Offending backend: tf` | 旧环境下 transformers 5.3.0 backend 检查异常 | `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers==4.57.6"` |
| vllm worker 起不来，`RecursionError`（numpy `issubdtype`↔`__repr__` 递归，根源 `np.dtype(bfloat16)`） | 旧环境下 numpy 混入的 2.x overlay 触发 dtype repr 死循环 | `pip uninstall -y numpy && rm -rf /usr/local/lib/python3.12/dist-packages/numpy /usr/local/lib/python3.12/dist-packages/numpy-1.26.4.dist-info && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "numpy==1.26.4"` |

> `transformers==4.57.6` 会顺带降级 `huggingface_hub`→0.36.2；numpy 重装后 `pip check` 报 `opencv-python-headless requires numpy>=2` 可忽略。`rm` 类命令在共享机上执行前先审批。