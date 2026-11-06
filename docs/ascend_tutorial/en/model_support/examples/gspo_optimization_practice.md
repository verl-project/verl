# NPU Qwen3-32B GSPO Optimization Practice

Last updated: 07/03/2026.

To ensure a good experience, switch verl to the main branch with commit id 9d05508f5e3bd8ecb70cf94ab10dc087b57a716d. Note: The main branch may cause patch issues due to iterative refactoring. For a stable version, switch to `release/v0.8.0`.

The training script for this practice: [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh) (`examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh`).

## Algorithm Adaptation

GSPO elevates the optimization granularity from the **token level** to the **sequence level**, avoiding the **sharp variance increase** that GRPO encounters and which leads to training instability. This increases training stability and also improves the convergence speed to some extent.

To successfully invoke the GSPO algorithm in the verl repository, the following configuration is required

```bash
# Core algorithm configuration  
algorithm.adv_estimator=grpo \                    # Use GRPO advantage estimator  
algorithm.use_kl_in_reward=False \                # Do not add KL penalty to reward  
# GSPO policy loss mode  
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \ # Enable GSPO policy loss
# Minimal clipping range (GSPO feature)  
actor_rollout_ref.actor.clip_ratio_low=0.0003 \   # Lower clip bound, paper recommended value  
actor_rollout_ref.actor.clip_ratio_high=0.0004 \  # Upper clip bound, paper recommended value  
# KL configuration (GSPO does not use KL loss)  
actor_rollout_ref.actor.use_kl_loss=False \       # Disable KL loss  
actor_rollout_ref.actor.kl_loss_coef=0.0 \        # Set KL loss coefficient to 0  
# Sequence-level loss aggregation mode (GSPO core)  
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \ # Sequence-level mean, recommended by GSPO paper  
# Batch configuration  
actor_rollout_ref.rollout.n=16 \                  # Generate 16 responses per prompt (group sampling)
```

The entry function is typically `verl.trainer.main_ppo`. For a complete runnable example, see [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh).

## Base Environment

Currently, Atlas 800T A3 and Atlas 900 A3 SuperPoD are supported. Completing this best practice requires 4 Atlas 800T A3 machines.

### Install Base Environment

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


```bash
cd verl
git checkout main
# Specify the corresponding recipe version
git submodule update --init --recursive recipe
```

### Obtain Weights

Download the model weights from Hugging Face: [Qwen/Qwen3-32B · Hugging Face](https://huggingface.co/Qwen/Qwen3-32B)

### Dataset Preparation

```bash
# Download the math-17k dataset
git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

# Download the AIME_2024 test dataset
git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
```

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

Before starting the task, execute the following command to import jemalloc through environment variables:

```shell
# Set the environment variable according to the actual installation path, for example /usr/local/lib/libjemalloc.so.2 (verify the file exists using find /usr -name libjemalloc.so.2)
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

### Multi-Node Task Startup

For the multi-node task in this practice, use the following script to launch

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2

# Modify to the current test case path
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=4
NPUS_PER_NODE=16
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

DEFAULT_SH: Modify to the configuration sh file path used for training. In this case, set it to [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh) (that is, `examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh`).

NNODES and NPUS_PER_NODE: Modify to the number of nodes and NPUs per node. In this case, 4 and 16 respectively.

MASTER_ADDR: Modify to the corresponding master node IP. All nodes should have the same MASTER_ADDR.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Modify to the corresponding communication network interface. The communication network interface can be obtained with the following command:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Performance Tuning

Optimization covers four aspects: training, inference, scheduling, and others.

### Training

#### Dynamic bsz

```bash
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
```

**This optimization mainly adjusts the two parameters above. Note that setting them too large will cause OOM.**

**Mainly adjust** `actor_ppo_max_token_len`. Increasing it reduces training time. Adjusting `infer_ppo_max_token_len` has no obvious benefit and can be left unchanged.

**The roles of these two parameters are described below:**

**These two parameters control the maximum number of tokens processed per GPU in dynamic batch size mode.**

- **`actor_ppo_max_token_len`**: Maximum number of tokens per GPU during PPO updates (forward + backward pass) for the Actor model
- **`infer_ppo_max_token_len`**: Maximum number of tokens per GPU when computing log probabilities during inference (Reference policy and Rollout)

### Inference

#### ACLgraph+FULL_DECODE_ONLY

Optimization for inference operator dispatch, providing an average performance gain of about `15%~20%`.

First, let's look at enabling only **ACLgraph**, as follows:

```bash
# Enable ACLgraph+FULL_DECODE_ONLY (Note: when set to False, TASK_QUEUE_ENABLE must be set to 1, otherwise an error will occur)
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY'
```

After `FULL_DECODE_ONLY` is successfully enabled, the following output appears:

![FULL_DECODE_ONLY result](https://github.com/wucong25/verl-data/blob/main/ascend_acl_graph.png)

**`cudagraph_capture_sizes` Parameter Setting Guide**

The values set in cudagraph_capture_sizes correspond to batch sizes. Here, the batch size is not the batch size corresponding to the DP domain in the configuration, but rather the batch size relative to vLLM, measured in **tokens**.

The default generated algorithm is as follows, for reference:

![cudagraph_capture_sizes](https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png)

##### Inference Backend Switching

Usage: `export VLLM_ATTENTION_BACKEND=XFORMERS`

![VLLM_ATTENTION_BACKEND](https://github.com/wucong25/verl-data/blob/main/ascend_vllm_attn_backend.png)

Note: Some backends are not supported in older vllm-ascend versions.

##### Enable vLLM v1

Usage: `export VLLM_USE_V1=1`

Can be kept on permanently; it generally provides positive gains.

### Scheduling

#### AIV

To enable: set `export HCCL_OP_EXPANSION_MODE="AIV"`

The HCCL_OP_EXPANSION_MODE environment variable configures where communication algorithm orchestration is expanded. Supported values are:

- AI_CPU: The communication algorithm orchestration is expanded on the AI CPU compute unit on the device side.
- AIV: The communication algorithm orchestration is expanded on the Vector Core compute unit on the device side.
- HOST: The communication algorithm orchestration is expanded on the host-side CPU. The device side automatically selects the appropriate scheduler based on the hardware model.
- HOST_TS: The communication algorithm orchestration is expanded on the host-side CPU. The host dispatches tasks to the device's Task Scheduler, which then schedules and executes the tasks.

The following describes two expansion mechanisms:

##### HOST Expansion

<img src="https://github.com/wucong25/verl-data/blob/main/en/figures/ascend_task_queue1.png" alt="image-20260113194257095" style="zoom:50%;" />

- The software stack runs on the host CPU, expanding communication algorithms into individual tasks.
- Each task calls the runtime interface and is dispatched to the device's rtsqueue.
- STARS fetches tasks sequentially from the rtsqueue.
- Calls SDMA and RDMA engines respectively based on task type.
  **Single operator bottleneck**: hostbound, each task submission takes 2~5us. A communication operator has hundreds of tasks. In single-operator scenarios, tasks are not cached on the device; each is executed as it is dispatched.

##### AICPU Mechanism Expansion

<img src="https://github.com/wucong25/verl-data/blob/main/en/figures/ascend_task_queue3.png" alt="image-20260113194333218" style="zoom:50%;" />

- The host side does not dispatch individual tasks. Instead, it treats communication operators as kernels and places them on the communication operator kernel queue.
- STARS schedules kernels from the kernel queue stream and places them on the AiCPU for execution.
- AICPU calls the function (kernel), uses a thread to execute the kernel function, expands communication tasks within the function, places tasks on the rtsqueue, and STARS invokes them.
- Reduces host-aicpu interactions from hundreds to one.
- Task submission occurs on AICPU, with partial merging of submissions.

#### TASK_QUEUE_ENABLE

**Usage:** `export TASK_QUEUE_ENABLE=2`

TASK_QUEUE_ENABLE: dispatch optimization. Set to 1 for graph mode (that is, when graph mode is enabled, set this to 1), and 2 for non-graph mode.

Schematic diagram:

![ascend task queue](https://github.com/wucong25/verl-data/blob/main/en/figures/ascend_task_queue2.png)

##### CPU Affinity Optimization

**Usage:** `export CPU_AFFINITY_CONF=1`

For detailed setting principles, see: https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/ENV/docs/en/environment_variable_reference/CPU_AFFINITY_CONF.md

### Others

The following summarizes tuning configurations for several global environment variables. Since these parameters often provide positive gains in both training and inference phases, and there are currently insufficient fine-grained ablation experiments to strictly distinguish their respective contributions to training or inference, they are grouped here for continued monitoring and further analysis.

#### Enable jemalloc

Usage (note: jemalloc library must be installed first): `export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2`

#### Multi-Stream Reuse

Optimization in terms of memory.

To enable: `export MULTI_STREAM_MEMORY_REUSE=1`

Principle introduction: https://www.hiascend.com/document/detail/en/Pytorch/latest/devguide/fwfeatures/docs/en/framework_feature_guide_pytorch/multistream_memory_reuse.md

#### VLLM_ASCEND_ENABLE_FLASHCOMM

Usage: `export VLLM_ASCEND_ENABLE_FLASHCOMM=1`

Enables FLASHCOMM high-speed communication optimization technology specific to Ascend NPU.

Reference: https://vllm-ascend.readthedocs.io/en/latest/user_guide/release_notes.html

#### VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

Usage: `export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1`

Enables dense computation optimization for large model inference on Ascend NPU.

Reference: https://vllm-ascend.readthedocs.io/en/latest/user_guide/release_notes.html

#### VLLM_ASCEND_ENABLE_PREFETCH_MLP

Usage: `export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`

Enables the weight prefetch mechanism for MLP layers.

<img src="https://github.com/wucong25/verl-data/blob/main/en/figures/ascend_prefetch.png" alt="image-20251124173132677" style="zoom:50%;" />

### verl Framework Parameter Settings

The following are memory-related settings (note: the optimizations here will more or less lead to some degradation in throughput):

```bash
# Gradient Checkpointing
# Purpose: Saves memory by recomputing activations, trading computation for memory. It does not save intermediate activations during forward pass and recomputes them during backward pass, which can significantly reduce memory usage and allow larger batch sizes.
actor_rollout_ref.model.enable_gradient_checkpointing=True \

# Parameter Offload
# Purpose: Offloads model parameters to CPU memory and loads them back to GPU during training.
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \

# Optimizer Offload
# Purpose: Offloads optimizer states (for example, Adam momentum) to CPU. Optimizer states typically occupy a large amount of memory (for Adam, 8 extra bytes per parameter); offloading saves memory.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \

# Free Cache Engine
# Purpose: Releases the inference engine's KV cache and weights during the training phase. This is the core optimization of 3D-HybridEngine, allowing inference and training to alternate on the same GPU, significantly reducing memory requirements.
actor_rollout_ref.rollout.free_cache_engine=True \

#  Entropy Computation Optimization
# entropy_checkpointing: Enables recomputation for entropy computation during training, reducing peak memory
# entropy_from_logits_with_chunking: Processes logits tensors in chunks (for example, 2048 tokens per group) to avoid loading the entire [bsz*seq_len, vocab] tensor at once
actor_rollout_ref.actor.entropy_checkpointing=True \
actor_rollout_ref.ref.entropy_checkpointing=True \
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \

# Inference Engine Memory Configuration
# gpu_memory_utilization: Controls the proportion of GPU memory used by vLLM (0.90 = 90%)
# enforce_eager=False: Enables CUDA graphs to accelerate inference, but uses additional memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
actor_rollout_ref.rollout.enforce_eager=False \
```

## NPU Tuning Reference Articles

Environment variables: [Environment Variable List - TorchNPU - Ascend Community](https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/ENV/docs/en/environment_variable_reference/env_variable_list.md)

Community performance tuning tutorial: [Model Performance Tuning - TorchNPU - Ascend Community](https://www.hiascend.com/document/detail/en/ModelZoo/traditional_model_train/PyTorch/docs/en/performance_tuning/performance_overview.md)
