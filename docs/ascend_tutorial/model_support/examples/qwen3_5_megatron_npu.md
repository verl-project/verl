# Qwen3.5 Megatron NPU 使用指南

Last updated: 06/08/2026.

本文用于指导在 Ascend NPU 上使用 verl + Megatron + vLLM 跑通 Qwen3.5-35B-A3B 和 Qwen3.5-122B-A10B GRPO 示例。

## 版本要求

| software | version |
| --- | --- |
| Docker image | `quay.io/ascend/verl:verl-9.0.0-a3-ubuntu22.04-py3.11-latest` |
| verl | commit `cdd9014f` |
| Python | 3.11 |
| CANN | 9.0.0 |
| Megatron-LM | 0.16.1 |
| MindSpeed | 0.16.0 |
| Megatron-Bridge | `de93536e` |

建议直接使用上表中的镜像：

```bash
docker pull quay.io/ascend/verl:verl-9.0.0-a3-ubuntu22.04-py3.11-latest
```

启动容器后，将 verl 固定到指定 commit：

```bash
git checkout cdd9014f
```

脚本注释中列出的额外依赖如下；镜像已包含的依赖不需要重复安装：

```bash
pip install viztracer flash-linear-attention nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon
```

## 模型和脚本

| model | HF model | script |
| --- | --- | --- |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` | `examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | `examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh` |

## 硬件和并行配置

示例脚本默认使用如下 NPU 配置，可以通过同名环境变量覆盖：

| model | nnodes | devices per node | TP | PP | CP | EP | ETP | GEN_TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B | 1 | 16 | 2 | 2 | 1 | 8 | 1 | 8 |
| Qwen3.5-122B-A10B | 4 | 16 | 2 | 4 | 1 | 16 | 1 | 16 |

## 数据和模型准备

脚本默认使用 Geo3K 数据集，并会下载到 `$HOME/data/geo3k`：

```bash
hf download tyzhu/geo3k --repo-type dataset --local-dir $HOME/data/geo3k
```

模型权重可以使用 Hugging Face 模型名，也可以提前下载到本地路径：

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/Qwen3.5-35B-A3B
hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/Qwen3.5-122B-A10B
```

## 启动训练

训练前需要先启动 Ray 集群。通用多节点说明可参考 [Multinode Training](../../../start/multinode.rst)，Ascend 多节点脚本示例可参考 [Ascend SGLang Best Practices](ascend_sglang_best_practices.rst)。

最小启动方式如下，单机任务只需要执行 head 节点命令；多机任务需要在其他节点执行 worker 节点命令。`MASTER_ADDR` 在所有节点上保持一致，`CURRENT_IP` 设置为当前节点 IP。

```bash
MASTER_ADDR=<head-node-ip>
CURRENT_IP=<current-node-ip>
NPUS_PER_NODE=16

# head node
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

# worker nodes, only needed for multi-node jobs
ray start --address="$MASTER_ADDR:6766" --node-ip-address=$CURRENT_IP --resources='{"NPU": '$NPUS_PER_NODE'}'

ray status
```

通过 `ray status` 确认 NPU 资源数量符合预期后，在主节点执行训练脚本。Qwen3.5-35B-A3B 默认需要 16 个 NPU 资源，Qwen3.5-122B-A10B 默认需要 64 个 NPU 资源。

### Qwen3.5-35B-A3B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B

bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

如果需要覆盖数据路径：

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
train_path=/path/to/train.parquet \
test_path=/path/to/test.parquet \
bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

### Qwen3.5-122B-A10B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B

bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

如果需要覆盖数据、保存路径或并行配置：

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
n_devices_per_node=16 \
nnodes=4 \
bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

## 注意事项

- 脚本会通过 `torch_npu` 自动识别 NPU 环境；如需手动指定，设置 `DEVICE=npu`。
- Qwen3.5 的 Gated Delta Net 当前不使用 packed sequence，因此脚本中保持 `use_remove_padding=False` 和 `use_dynamic_bsz=False`。
- NPU 分支会设置 `vanilla_mbridge=False`、`use_flash_attn=True`、`moe_token_dispatcher_type=alltoall` 等 Ascend 适配参数。
