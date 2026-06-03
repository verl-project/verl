# Qwen3.5-122B-A10B NPU 使用指南

Last updated: 06/03/2026.

本文用于指导在 Ascend NPU 上使用 verl + Megatron + vLLM 跑通 Qwen3.5-122B-A10B GRPO 示例。

## 版本要求

| software | version |
| --- | --- |
| Docker image | `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` |
| verl | commit `cdd9014f` |
| Python | 3.11 |
| CANN | 8.5.2 |
| Megatron-LM | 0.16.1 |
| MindSpeed | 0.16.0 |
| Megatron-Bridge | `de93536e` |
| Model | `Qwen/Qwen3.5-122B-A10B` |

建议直接使用上表中的镜像，并将 verl 固定到指定 commit：

```bash
git checkout cdd9014f
```

如果镜像中缺少依赖，可以在容器内补充安装：

```bash
pip install viztracer flash-linear-attention nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon
```

## 硬件配置

推荐使用 4 机 NPU。示例脚本默认配置为：

```bash
TP=2
PP=2
CP=1
EP=8
ETP=1
GEN_TP=8
n_devices_per_node=16
nnodes=4
```

## 数据和模型准备

脚本默认使用 Geo3K 数据集，并会下载到 `$HOME/data/geo3k`：

```bash
hf download tyzhu/geo3k --repo-type dataset --local-dir $HOME/data/geo3k
```

模型权重可以使用 Hugging Face 模型名，也可以提前下载到本地路径：

```bash
hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/Qwen3.5-122B-A10B
```

## 启动训练

在 Ray 集群已启动后，在主节点执行：

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B

bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

如果需要覆盖数据、保存路径或并行配置，可以通过环境变量传入：

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
