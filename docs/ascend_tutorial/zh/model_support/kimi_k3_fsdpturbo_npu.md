# Kimi K3 FSDPTurbo NPU 使用指南

本文用于指导在 Ascend A3 上使用 verl + FSDPTurbo + vLLM-Ascend 运行 Kimi K3 多模态 MoE GRPO 训练。适配支持动态批量、训练权重同步和 rollout decode 图模式。

## 版本要求

| software | version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.10.0（CPU wheel，配合 torch_npu） |
| torch_npu | 2.10.0.post2 |
| CANN | 9.0.1 |
| vLLM | 0.26.0 配套源码及版本元数据 |
| Transformers | 5.10.4 |
| modelopt | 0.46.0 |
| FSDPTurbo | https://gitcode.com/wangdongleix/FSDPTurbo/tree/merge-backend-kimi-k3 |
| verl | https://github.com/wangdongleix/verl/tree/merge-backend-kimi-k3 |
| vLLM-Ascend | https://github.com/wangdongleix/vllm-ascend/tree/merge-backend-kimi-k3 |

请使用满足上述版本要求的 Ascend NPU 环境，并同时准备配套源码、Python 依赖及编译产物。源码版本与算子编译产物需要匹配。

其他运行依赖包括 Ray、Hydra/OmegaConf、TransferQueue、datasets、pyarrow、pandas、NumPy、Pillow、qwen_vl_utils、tensordict、torchdata、accelerate、peft，以及 reward 使用的 mathruler、pylatexenc。其他库的精确版本尚未在本文锁定。不要直接用 verl 的 CUDA extras 覆盖这套 NPU 环境。

## 模型和脚本

| model | 本地权重 | script |
| --- | --- | --- |
| Kimi K3 top16 Geo3K SFT | `/path/to/models/kimi-k3-geo3k-sft` | [run_kimi_k3_fsdpturbo.sh](../../../../../examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh) |


## 硬件和并行配置

以下为四机全层基线，共 64 张逻辑 NPU：

| nnodes | devices per node | FSDP | CP | EP | expert FSDP | rollout TP | rollout EP | rollout replicas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 64 | 1 | 16 | 4 | 16 | 16 | 4 |

global prompt batch 为 16，rollout n=4，生成 64 条训练样本，作为四机 bsz=1 的配置示例。开启动态批量后，实际 micro batch 还受序列长度及 token budget 影响，不能仅依据 `ppo_micro_batch_size_per_gpu` 判断实际批量。

## 数据和模型准备

使用已准备好的 Geo3K Parquet 数据：

```bash
export BACKEND_ROOT=/path/to/kimi-k3
export MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export TRAIN_FILE=/path/to/datasets/geo3k/train.parquet
export TEST_FILE=/path/to/datasets/geo3k/test.parquet
```

训练脚本会在 `$BACKEND_ROOT/models/fsdpturbo-<权重目录名>` 创建模型副本目录，链接原始权重，并复制 FSDPTurbo 的 modeling 代码。必须通过该准备流程，使实际加载的模型使用本次适配实现；只修改仓库内 modeling、仍直接加载旧模型目录，不能保证算子生效。

## 启动训练

先在所有节点配置相同依赖并加载 CANN 环境。执行前将 `/path/to/...` 和 `<...>` 占位符替换为实际配置。

### 启动 Ray 集群

在主节点执行：

```bash
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
ray start --head --port="$MASTER_PORT" \
  --node-ip-address="$MASTER_ADDR" \
  --resources='{"NPU":16}'
```

在其余三个 worker 节点分别执行：

```bash
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
export NODE_IP="<current-node-ip>"
ray start --address="$MASTER_ADDR:$MASTER_PORT" \
  --node-ip-address="$NODE_IP" \
  --resources='{"NPU":16}'
```

在主节点执行 `ray status --address="$MASTER_ADDR:$MASTER_PORT"`，确认四个节点、共 64 个 NPU 资源已就绪。

### 启动 FSDPTurbo

以下命令仅在主节点执行：

```bash
export BACKEND_ROOT=/path/to/kimi-k3
export MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export TRAIN_FILE=/path/to/datasets/geo3k/train.parquet
export TEST_FILE=/path/to/datasets/geo3k/test.parquet
export KIMI_RUN_MODE=multinode
export NNODES=4
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
export RAY_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
export NPUS_PER_NODE=16
export TURBO_EP_SIZE=16
export ROLLOUT_TP=16
export TRAIN_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export MAX_VISUAL_TOKENS=1024
export VERL_VLLM_MULTIMODAL_TOKEN_MARGIN=1024
export VERL_MAX_RESPONSE_LENGTH=2048
export VERL_VLLM_MAX_MODEL_LEN=4096
export VERL_VLLM_MAX_NUM_SEQS=16
export VERL_VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VERL_TOTAL_TRAINING_STEPS=3
export VERL_SAVE_FREQ=-1
export VERL_VLLM_ENFORCE_EAGER=False

bash "$BACKEND_ROOT/verl/examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh" \
  "++ray_kwargs.ray_init.address=$RAY_ADDRESS" \
  data.max_prompt_length=1024 \
  trainer.balance_batch=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=2147483648'
```

这是 prompt=1024、response=2048 的三步训练示例，不保存 checkpoint，也不主动开启 profiling。网络接口默认值与环境有关，迁移时应设置 `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 为实际互通网卡。
