# Precision Alignment Guide

When performing reinforcement learning (RL) training in the VeRL framework, **precision alignment** is a key step to ensure that the training process is reproducible and debuggable.

This document summarizes the methods for precision alignment between NPU and GPU on VeRL for reference.

Last updated: 05/09/2026.

## 1. Environment and Weight Alignment

### 1.1 Dependency Version Alignment

The versions of VeRL and transformers must be strictly aligned; otherwise, the precision results will be directly affected.

For other key dependencies (torch, megatron, vllm), if strict alignment is not possible, priority should be given to keeping them consistent or close.

### 1.2 Model Weight Alignment

Check whether the model weights and the config.json file are completely identical.


## 2. Input Data Alignment

Add the following configuration to the verl training startup script:

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. Configuration Alignment

When performing precision alignment between NPU and GPU, you need to check whether the configurations are fully aligned. This includes:
1. Directly compare the configurations written in the scripts.
2. Save logs during the run, collect the configurations printed in the logs for comparison. You can compare whether the default parameter configurations are consistent, and you must ensure that key parameters are aligned.


## 4. Fixing Determinism

### 4.1 Fixing Random Seeds

Install `msprobe` in the environment:

```bash
pip install mindstudio-probe
```

Add the determinism function at the beginning of the worker file:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 Fixing Communication Environment Variables

In multi-card communication scenarios:

- Under HCCL communication (default scenario):
  
   -  export CLOSE_MATMUL_K_SHIFT=1
   -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
   -  export HCCL_DETERMINISTIC="true"
   -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- Under LCCL communication (enabled through export HCCL_OP_EXPANSION_MODE="AIV"):

   -  export CLOSE_MATMUL_K_SHIFT=1
   -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
   -  export LCCL_DETERMINISTIC=1
   -  export ATB_LLM_LCOC_ENABLE=0
   -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

In single-card no-communication scenarios:

   -  export CLOSE_MATMUL_K_SHIFT=1
   -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
   -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. Verifying Training Precision

### 5.1 Training Instrumentation (Stubbing)

**Instrumentation (stubbing)** means preserving the input and output data of the current stage to facilitate comparative analysis of the results. When troubleshooting precision issues, instrumentation is needed to help locate the problem. A common instrumentation approach is to directly dump the data from the rollout phase.

**Step 1: Generate baseline data on the GPU environment**

Run the GPU script once with the following configuration enabled:

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

This saves the inference result of each step as a jsonl file.

**Step 2: Reproduce and verify on the NPU environment**

Enable the following parameters on the NPU, reuse the sequence generated in the previous step, and run end-to-end:

```bash
skip.rollout.enable=True \
skip.rollout.dump_dir=/path/to/rollout_dump \
```

**Step 3: Compare metrics**

With the same stubbed inference input, consistent training configuration, and fixed randomness, compare whether there are differences in the rewards/pg_loss/grad_norm values between NPU and GPU.


## 6. Verifying Inference Precision

### 6.1 resharding

Before inference officially starts, vllm performs a **dummy run** that infers one token to estimate the memory usage during inference and then allocates memory. You can specify the load_format parameter during vLLM's LLM initialization to determine whether the dummy run uses randomly initialized weights (dummy) or real weights (safetensors). In VeRL, this parameter is specified through **actor_rollout_ref.rollout.load_format**.

When garbled inference output occurs, if the engine initialization uses **load_format=dummy**, then the sharding is highly likely to have a problem. Even if the output becomes normal after switching to safetensors, the sharding still has issues and the forward pass needs to be compared.


### 6.2  Inference Result Alignment

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

Save the inference result of each step as a jsonl file. You can directly open the jsonl file to quickly confirm whether the whole-network inference result is garbled, which is used to scope inference precision issues.


Before dumping inference data, if reproducing the inference precision issue requires a lot of resources, you can first try to reduce the cost of reproducing the inference precision issue by reducing the reproduction scale and the amount of data that needs to be dumped and compared. In multi-batch and long-sequence scenarios, you can try to reproduce by sending single-batch requests and reducing the sequence length.


## 7. Dump Comparison

[Precision Debugger](../../../en/dev_guide/precision_analysis/precision_debugger.md). After locating the stage where the problem occurs, you can use the msprobe tool to dump data for detailed localization.

During inference or training, the model may produce outputs that deviate from expectations, generate anomalies, or even produce numerical instability issues such as NaN/Inf. To locate the root cause, you need to finely monitor the model execution path, collect intermediate features, weights, activation values, and the inputs and outputs of each key layer, and record context information such as prompts, tensor dtype, and hardware configuration. By capturing these core tensors and metadata, you can systematically trace the source of precision degradation or numerical errors.
