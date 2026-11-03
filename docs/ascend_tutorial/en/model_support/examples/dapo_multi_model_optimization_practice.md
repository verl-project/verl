# DAPO Model Optimization Practice

## DAPO Introduction

Last updated: 07/03/2026.

The DAPO paper can be found at: [DAPO](https://arxiv.org/pdf/2503.14476), which includes the following key technologies.

* **Clip-Higher**: By clipping the upper bound of the importance sampling ratio, it promotes system diversity and avoids entropy collapse.
* **Dynamic Sampling**: Improves training efficiency and stability. DAPO proposes a strategy for performing dynamic sampling and filters out prompt groups with accuracy equal to 1 or 0, thereby keeping the number of prompts with valid gradients consistent across batches.
* **Token-level Policy Gradient Loss**: Crucial in long chain-of-thought reinforcement learning (long-CoT RL) scenarios.
* **Overlong Reward Shaping**: Reduces reward noise and stabilizes training.

In verl, you can make the following settings to run the DAPO algorithm.

- **Reward model management strategy is DAPO**
  In the DAPO algorithm, this must be configured as dapo.

```bash
reward_model.reward_manager.name=dapo
```

- **Clip-Higher**
  `clip_ratio_low` and `clip_ratio_high` specify $\varepsilon_{\text {low }}$ and $\varepsilon_{\text {high }}$ in the DAPO objective function.

```bash
clip_ratio_low=0.2  # Lower clip ratio bound, default value is 0.2
clip_ratio_high=0.28 # Upper clip ratio bound, default value is 0.28
```

- **Dynamic Sampling Configuration**
  Setting `filter_groups.enable` to `True` filters out groups with identical outputs for the `metric`. For example, for the `acc` metric, it filters out groups where all outputs have accuracy of 1 or 0.
  The trainer uses `gen_batch_size` for repeated sampling until enough qualifying groups are generated, or the upper limit specified by `max_num_gen_batches` is reached.

```bash
data.gen_batch_size=${gen_prompt_bsz}
algorithm.filter_groups.enable=${enable_filter_groups} # Dynamic sampling switch
algorithm.filter_groups.metric=${filter_groups_metric} # Use accuracy as the filter criterion
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} # Maximum number of generation batches, the maximum number of times to repeatedly generate data
```

- **Token-level Loss**
  Setting `loss_agg_mode` to `token-mean` means computing the average of the (policy gradient) loss across all tokens in all sequences within a batch.

```bash
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
# Note: "token-mean" is the default behavior.
```

- **Reward Model Overlong Response Penalty Configuration**
  Setting `overlong_buffer.enable` to `True` penalizes outputs whose length is too long but still within the hard context limit. Specifically, when the output length exceeds `max_response_length - overlong_buffer.len` and exceeds by 0 to `overlong_buffer.len` tokens, the penalty value increases linearly from `0` to `overlong_buffer.penalty_factor`.

```bash
reward_model.overlong_buffer.enable=${enable_overlong_buffer} # Enable overlong buffer penalty, enable the penalty mechanism for overlong outputs
reward_model.overlong_buffer.len=${overlong_buffer_len}  # Buffer length, defines the buffer tokens, maximum penalty strength
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}   # Penalty factor, maximum penalty strength
```

For the code involving these parameters, refer to: [Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)

## Hardware Requirements


Currently, Atlas 800T A3 and Atlas 900 A3 SuperPoD are supported. Completing this best practice requires 1 Atlas 900 A3 SuperPoD. For key software versions, refer to: [Ascend Quick Start Guide](../../get_start/quick_start.rst)


## Installing the Base Environment

| software      | version                                                    |
| ------------- | ---------------------------------------------------------- |
| Python        | 3.11                                                       |
| CANN          | ==9.0.0.B160 (CANN900B160)                                 |
| torch         | ==2.9.0                                                    |
| torch_npu     | ==2.9.0                                                    |
| triton_ascend | ==3.2.1                                                    |
| verl          | main                                                       |
| vllm          | v0.18.0                                                    |
| vllm-ascend   | v0.18.0                                                    |
| transformers  | 5.3.0                                                      |

In this practice, we specify a verl commit id to avoid introducing other issues. Note: The main branch may cause patch issues due to iterative refactoring. For a stable version, switch to `release/v0.8.0`.

```bash
cd verl
git checkout main
# Specify the corresponding recipe version
git submodule update --init --recursive recipe
cd recipe
git checkout main
```

## Model Training

### Dataset Preparation

The Geometry3k dataset is a geometry-specific dataset jointly developed by UCLA and Zhejiang University, primarily designed for Visual Question Answering (VQA) tasks. The dataset contains 3002 samples in total, constructed using both image and text modalities. The text modality covers various geometry problem descriptions, while images present the geometric figures from the problems as visual charts, including basic geometric shapes such as triangles, circles, and quadrilaterals, as well as positional, nesting, and intersection relationships between different figures. You can download the original dataset from Hugging Face: [Geometry3k ](https://huggingface.co/datasets/hiyouga/geometry3k)

```shell
# Download raw data and preprocess
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

### Weight Download

Download the model weights from Hugging Face: [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)

### jemalloc Installation

To ensure Ray processes can properly reclaim memory, install and enable the jemalloc library for memory management.

#### Ubuntu Operating System

Install jemalloc from the OS package repository (Note: requires Ubuntu version >= 20.04):

```shell
sudo apt install libjemalloc2
```

Before starting the task, execute the following command to import jemalloc through environment variables. First verify the file exists using **find /usr -name libjemalloc.so.2**:

```shell
# arm64 architecture
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64 architecture
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler Operating System

Execute the following command to install jemalloc from the OS package repository

```shell
yum install jemalloc
```

If the above methods do not install correctly, you can build and install from source. Download the latest stable version from the jemalloc official website: https://github.com/jemalloc/jemalloc/releases/

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

### Global Environment Variables

- To ensure Ray processes can properly reclaim memory, install and enable the jemalloc library for memory management. This helps manage memory better and avoids OOM during long-running tasks.

```bash
# Set the jemalloc environment variable according to the actual installation path, for example /usr/local/lib/libjemalloc.so.2 (verify the file exists using find /usr -name libjemalloc.so.2)
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

- Some models are optimized by vllm-ascend. However, in some cases, the optimized model may not be suitable. Setting this value to 0 disables the optimized model.

```bash
export USE_OPTIMIZED_MODEL=0
```

- Enable vLLM V1

```bash
export VLLM_USE_V1=1
```

- Fallback configuration for Ascend multi-card communication. Extends the connection timeout to avoid training startup failures due to slow connections in cluster environments.

```bash
export HCCL_CONNECT_TIMEOUT=5400
```

- Controls whether vLLM enables NZ optimization on Ascend chips

```bash
export VLLM_ASCEND_ENABLE_NZ=0
```

### Training
```bash
# Model Weights Paths
MODEL_PATH=hf_weights/Qwen3-VL-30B-A3B-Instruct
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=$RAY_DATA_HOME/datasets/geo3k/train.parquet
TEST_FILE=$RAY_DATA_HOME/datasets/geo3k/test.parquet

# Save frequency, -1 means no saving by default. Modify this parameter if evaluation is needed.
trainer.save_freq=-1
```

- For single-node Qwen3-VL-30B tasks, use the following script to launch training.

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, for example rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export CPU_AFFINITY_CONF=2
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_VERSION="0.18.0"

# Modify to the corresponding master node IP
MASTER_ADDR="IP FOR MASTER NODE"
# Number of NPUs per node
NPUS_PER_NODE=16
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

bash recipe/dapo/run_dapo_qwen3_vl_30b_fsdp2_npu.sh
```
- For multi-node Qwen3-VL-30B tasks, we recommend using the following script to launch large-scale multi-node training. Modify `NNODES` and `NPUS_PER_NODE` as needed, and update the parameters `trainer.nnodes` and `trainer.n_gpus_per_node` in the configuration script accordingly.

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, for example rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export CPU_AFFINITY_CONF=2
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_VERSION="0.18.0"

# Modify to the current test case path
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

# Number of compute nodes in the cluster
NNODES=2
# Number of NPUs per node
NPUS_PER_NODE=8
# Modify to the corresponding master node IP
MASTER_ADDR="IP FOR MASTER NODE"
# Modify to the current node's communication network interface
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# Get current IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # Master node startup
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # Check if device_count equals NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # Worker node tries to register with master node until success
  while true; do
      # Try to connect to the ray cluster
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # Check if connection is successful
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
```
DEFAULT_SH: Modify to the configuration sh file path used for training. In this case, set it to the [Qwen3_VL_30B](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen3_vl_30b_fsdp2_npu.sh) path.

NNODES and NPUS_PER_NODE: Modify to the number of nodes and NPUs per node. In this case, 2 and 8 respectively.

MASTER_ADDR: Modify to the corresponding master node IP. All nodes should have the same MASTER_ADDR.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Modify to the corresponding communication network interface. The communication network interface can be obtained with the following command:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Optimization Reference

- **Enable Dynamic Batch Size**
  Dynamically adjusts the batch size based on the maximum total tokens per GPU (ppo_max_token_len_per_gpu)

```bash
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
```

- **Maximum Total Tokens per GPU**
  When `use_dynamic_bsz=True`, the maximum number of tokens a single GPU can process in a micro-batch

```bash
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
```

- **Micro-batch Size per GPU**
  When `use_dynamic_bsz=True`, the framework uses this value as the initial batch size, then adjusts up/down based on `ppo_max_token_len_per_gpu`

```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

- **Enable FSDP2 Framework**
  "Shard model parameters, gradients, and optimizer states across different GPUs" to avoid OOM from loading the full model on a single card.

```bash
# Enable FSDP2 framework
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2
critic.strategy=fsdp2

# FSDP2 only: reshards after forward pass to reduce memory usage.
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
# FSDP2 only: whether to reshard after model forward pass to save memory.
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
```

- **Enable Expert Parallel Configuration**
  Specifies how many GPUs are used to compute different expert networks in parallel.

```bash
# Expert parallel configuration for MoE architecture Actor model
actor_rollout_ref.rollout.expert_parallel_size=8
```

