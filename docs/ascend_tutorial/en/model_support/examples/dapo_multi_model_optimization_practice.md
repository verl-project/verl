# DAPO Model Optimization Practice

## DAPO Introduction

Last updated: 07/03/2026.

For the DAPO paper, refer to [DAPO](https://arxiv.org/pdf/2503.14476), which includes the following key technologies.

* **Clip-Higher**: Promotes system diversity by clipping the upper bound of the importance sampling ratio and prevents entropy collapse.
* **Dynamic Sampling**: Improves training efficiency and stability. DAPO proposes a strategy for performing dynamic sampling and filters out prompt groups with an accuracy of 1 or 0, thereby maintaining a consistent number of prompts with valid gradients across batches.
* **Token-level Policy Gradient Loss**: Is crucial in long chain-of-thought reinforcement learning (long-CoT RL) scenarios.
* **Overlong Reward Shaping**: Reduces reward noise and stabilizes training.

In verl, you can use the following configuration to run the DAPO algorithm.

- **The management strategy for the reward model is DAPO**
  In the DAPO algorithm, you must configure it as dapo.

```bash
reward_model.reward_manager.name=dapo
```

- **Clip-Higher**
  `clip_ratio_low` and `clip_ratio_high` specify $\varepsilon_{\text {low }}$ and $\varepsilon_{\text {high }}$ in the DAPO objective function.

```bash
clip_ratio_low=0.2  # Lower bound of the clip ratio, default value is 0.2
clip_ratio_high=0.28 # Upper bound of the clip ratio, default value is 0.28
```

- **Related configurations for dynamic sampling**
  Setting `filter_groups.enable` to `True` filters out groups with identical output `metric` values. For example, for the `acc` metric, it filters out groups where the output accuracy is all 1 or 0.
  The trainer uses `gen_batch_size` to repeatedly sample until it generates enough qualified groups or reaches the limit specified by `max_num_gen_batches`.

```bash
data.gen_batch_size=${gen_prompt_bsz}
algorithm.filter_groups.enable=${enable_filter_groups} # Dynamic sampling switch
algorithm.filter_groups.metric=${filter_groups_metric} # Use accuracy as the filter criterion
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} # Maximum number of generation batches, the maximum number of times to repeatedly generate data
```

- **Token-level Loss**
  Setting `loss_agg_mode` to `token-mean` means calculating the average of the (policy gradient) loss across all tokens in all sequences within a batch.

```bash
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
# Note: "token-mean" is the default behavior.
```

- **Reward model penalty configuration for overly long responses**
  Setting `overlong_buffer.enable` to `True` penalizes outputs that are too long but still do not exceed the hard context limit. Specifically, when the output length exceeds `max_response_length - overlong_buffer.len` and the excess is between `0` and `overlong_buffer.len` tokens, the penalty value increases linearly from `0` to `overlong_buffer.penalty_factor`.

```bash
reward_model.overlong_buffer.enable=${enable_overlong_buffer} # Enable the overlong buffer penalty; enables the penalty mechanism for overlong outputs
reward_model.overlong_buffer.len=${overlong_buffer_len}  # Buffer length; defines the buffer tokens, maximum penalty strength
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}   # Penalty factor, maximum penalty strength
```

For the code related to the relevant parameters, refer to [Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md).

## Hardware Requirements


Currently, Atlas 800T A3 and Atlas 900 A3 SuperPoD are supported. To complete this best practice, you need one Atlas 900 A3 SuperPoD. For key software versions, refer to [Ascend Quick Start Guide](../../get_start/quick_start.rst).


## Installing the Basic Environment

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

In this practice, we specify the verl commit ID to avoid introducing other issues. Note: The main branch may cause patch issues due to iterative refactoring. If you need a stable version, switch to `release/v0.8.0`.

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

The Geometry3k dataset is a geometry-specific dataset jointly developed by the University of California, Los Angeles (UCLA) and Zhejiang University. It is primarily designed for research and model training in visual question answering (VQA) tasks. The dataset contains a total of 3,002 samples and is constructed using two modalities: image and text. The text modality covers various geometric problem descriptions, while the image modality presents geometric figure information through visual diagrams. These include basic geometric shapes such as triangles, circles, and quadrilaterals, as well as positional, nesting, and intersection relationships between different figures. You can download the original dataset from the Hugging Face Hub: [Geometry3k ](https://huggingface.co/datasets/hiyouga/geometry3k)

```shell
# Download raw data and preprocess
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

### Download Weights

Download the corresponding model weights from the Hugging Face repository: [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)

### jemalloc installation

To ensure that Ray processes can properly reclaim memory, install and enable the jemalloc library for memory management.

#### Ubuntu operating system

Install jemalloc through the operating system source (Note: requires Ubuntu version >= 20.04):

```shell
sudo apt install libjemalloc2
```

Before starting the task, run the following command to import jemalloc using environment variables. First, verify that the file exists using **find /usr -name libjemalloc.so.2**:

```shell
# arm64 architecture
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64 architecture
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler operating system

Run the following command to install jemalloc from the operating system source.

```shell
yum install jemalloc
```

If the preceding method does not install successfully, you can install it by compiling the source code. Go to the official jemalloc website to download the latest stable version at https://github.com/jemalloc/jemalloc/releases/.

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

### Importing Global Variables

- To ensure that the Ray process can properly reclaim memory, install and enable the jemalloc library for memory management. This helps better manage memory and avoid out of memory (OOM) errors during long-running processes.

```bash
# Set the jemalloc environment variable based on the actual installation path. For example, the installation path is /usr/local/lib/libjemalloc.so.2 (you can run find /usr -name libjemalloc.so.2 to check whether the file exists)
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
```

- Certain models are optimized using vllm ascend. However, in some cases, the optimized model may not be applicable. In this case, set this value to 0 to disable the optimized model.

```bash
export USE_OPTIMIZED_MODEL=0
```

- Enable vLLM V1

```bash
export VLLM_USE_V1=1
```

- A fallback configuration for Ascend multi-device communication that extends the connection timeout, preventing training startup failures due to slow connections in a cluster environment.

```bash
export HCCL_CONNECT_TIMEOUT=5400
```

- Controls whether to enable NZ optimization for vLLM on Ascend chips.

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

# Save frequency. -1 means no saving by default. Modify this parameter if you need to evaluate.
trainer.save_freq=-1
```

- For the single-machine task Qwen3-VL-30B, you can use the following script to start training.

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export CPU_AFFINITY_CONF=2
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_VERSION="0.18.0"

# Change to the corresponding master node IP
MASTER_ADDR="IP FOR MASTER NODE"
# Number of NPUs per node
NPUS_PER_NODE=16
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

bash recipe/dapo/run_dapo_qwen3_vl_30b_fsdp2_npu.sh
```
- For multi-node Qwen3-VL-30B tasks, we recommend using the following script to launch large-scale multi-node training. Modify `NNODES` and `NPUS_PER_NODE` as needed, and update the parameters `trainer.nnodes` and `trainer.n_gpus_per_node` in the configuration script to match.

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export VLLM_ASCEND_ENABLE_NZ=0
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export CPU_AFFINITY_CONF=2
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_VERSION="0.18.0"

# Change to the path of the test case you need to run
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

# Number of compute nodes in the cluster
NNODES=2
# Number of NPUs per node
NPUS_PER_NODE=8
# Change to the IP address of the master node
MASTER_ADDR="IP FOR MASTER NODE"
# Change to the network interface of the current node
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# Get the current IP address
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # Start the master node
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # Check whether device_count equals NNODES
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
  # The worker node attempts to register with the master node until successful
  while true; do
      # Attempt to connect to the Ray cluster
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # Check whether the connection is successful
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
DEFAULT_SH: Change to the path of the configuration shell script used for training. In this case, change it to the [Qwen3_VL_30B](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen3_vl_30b_fsdp2_npu.sh) path.

NNODES and NPUS_PER_NODE: Modify them to use the number of nodes and the number of NPUs per node. In this case, the values are 2 and 8, respectively.

MASTER_ADDR: Change it to the IP address of the corresponding master node. That is, the MASTER_ADDR of all nodes should be the same.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Change them to the corresponding communication NIC. You can obtain the communication NIC using the following command:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Optimization Reference

- **Enable dynamic batch size**
  Dynamically adjusts the batch size based on the maximum total number of tokens per GPU (ppo_max_token_len_per_gpu).

```bash
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
```

- **Maximum total number of tokens that a single GPU can process**
  When `use_dynamic_bsz=True`, the maximum number of tokens that a single GPU can process in a micro-batch

```bash
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
```

- **Microbatch size per GPU**
  When `use_dynamic_bsz=True`, the framework uses this value as the initial batch size and adjusts it up or down based on `ppo_max_token_len_per_gpu`.

```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

- **Enable the FSDP2 framework**
  "Sharding model parameters, gradients, and optimizer states across different GPUs" prevents device memory overflow caused by loading the full model on a single device.

```bash
# Enable the FSDP2 framework
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2
critic.strategy=fsdp2

# Only for FSDP2: reshard after forward pass to reduce memory usage.
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
# Only for FSDP2: whether to reshard after the model forward pass to save memory.
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
```

- **Enable expert parallelism configuration**
  Specify the number of GPUs used to compute different expert networks in parallel.

```bash
# Expert parallelism configuration for the Actor model in the MoE architecture
actor_rollout_ref.rollout.expert_parallel_size=8
```


