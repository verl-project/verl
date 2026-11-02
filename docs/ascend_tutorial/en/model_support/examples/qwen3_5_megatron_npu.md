# Qwen3.5 Megatron NPU Usage Guide

Last updated: 09/29/2026.

This document guides you through running Qwen3.5-35B-A3B and Qwen3.5-122B-A10B GRPO examples on Ascend NPU using verl + Megatron + vLLM.

## Version Requirements

Use the dependency combination from the [vLLM NPU Installation Script](../../../../../scripts/install_vllm_mcore_npu.sh):

| Component | Version |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.23.0 / 0.23.0 |
| transformers | 5.10.4 |
| Megatron-LM / MindSpeed / MegatronAdaptor | `core_r0.18.0` |
| Megatron-Bridge | `v0.5.0` |
| MindSpeed-Bridge | repository default branch (no fixed tag/commit) |

The installation script also installs TransformerEngineNPU and MindSpeed-Ops. You can install according to the [Ascend Installation Guide](../../get_start/install_guidance.rst), or use an A3 vLLM image consistent with this combination.

## Models and Scripts

| model             | HF model | script |
|-------------------| --- | --- |
| Qwen3.5-35B-A3B   | `Qwen/Qwen3.5-35B-A3B` | `examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | `examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh` |
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | `examples/grpo_trainer/run_qwen3_5_397b_megatron.sh` |

## Hardware and Parallel Configuration

The example scripts use the following default NPU configuration, which can be overridden through environment variables with the same name:

| model | nnodes | devices per node | TP | PP | CP | EP | ETP | GEN_DP | GEN_TP | GEN_EP |
| --- |--------| --- | --- | --- | --- |----| --- |---|----|----|
| Qwen3.5-35B-A3B | 1 | 16 | 2 | 2 | 1 | 8  | 1 | 1 | 8 | 1 |
| Qwen3.5-122B-A10B | 4 | 16 | 2 | 4 | 1 | 16 | 1 | 1 | 16 | 1 |
| Qwen3.5-397B-A17B | 16 | 16 | 2 | 4 | 1 | 64 | 1 | 16 | 16 | 256 |

## Data and Model Preparation

The scripts use the Geo3K dataset by default, which will be downloaded to `$HOME/data/geo3k`:

```bash
hf download tyzhu/geo3k --repo-type dataset --local-dir $HOME/data/geo3k
```

You can use the Hugging Face model name for the model weights, or download them to a local path in advance:

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/Qwen3.5-35B-A3B
hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/Qwen3.5-122B-A10B
hf download Qwen/Qwen3.5-397B-A17B --local-dir /path/to/Qwen3.5-397B-A17B
```

## Start Training

Before training, you need to start the Ray cluster. For general multi-node instructions, see [Multinode Training](../../../../start/multinode.rst). For Ascend multi-node script examples, see [Ascend SGLang Best Practices](ascend_sglang_best_practices.rst).

The minimal startup method is as follows. For single-node tasks, only the head node command needs to be executed; for multi-node tasks, the worker node command needs to be executed on other nodes. `MASTER_ADDR` must be consistent across all nodes, and `CURRENT_IP` should be set to the current node IP.

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

After confirming using `ray status` that the NPU resource count meets expectations, execute the training script on the master node. Qwen3.5-35B-A3B requires 16 NPU resources by default, and Qwen3.5-122B-A10B requires 64 NPU resources by default.

### Qwen3.5-35B-A3B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B

bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

If you need to override the data path:

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

If you need to override data, save path, or parallel configuration:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-122B-A10B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
NDEVICES_PER_NODE=16 \
nnodes=4 \
bash examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh
```

### Qwen3.5-397B-A17B

```bash
export DEVICE=npu
export HF_MODEL_PATH=/path/to/Qwen3.5-397B-A17B

bash examples/grpo_trainer/run_qwen3_5_397b_megatron.sh
```

If you need to override data, save path, or parallel configuration:

```bash
DEVICE=npu \
HF_MODEL_PATH=/path/to/Qwen3.5-397B-A17B \
train_files=/path/to/train.parquet \
test_files=/path/to/test.parquet \
save_path=/path/to/checkpoints \
NDEVICES_PER_NODE=16 \
nnodes=16 \
bash examples/grpo_trainer/run_qwen3_5_397b_megatron.sh
```

## Precautions

- The script automatically detects the NPU environment using `torch_npu`; to specify manually, set `DEVICE=npu`.
- Qwen3.5's Gated Delta Net currently does not use packed sequences, so the scripts keep `use_remove_padding=False` and `use_dynamic_bsz=False`.
- The NPU branch sets `use_flash_attn=True`, `moe_token_dispatcher_type=alltoall`, and other Ascend adaptation parameters.
