# Precision Alignment Guide

When you perform reinforcement learning (RL) training in the VeRL framework, **precision alignment** is critical for ensuring a reproducible and debuggable training process.

This document summarizes the methods for performing precision alignment for NPUs and GPUs in VeRL for reference.

Last updated: 05/09/2026.

## 1. Environment and Weight Alignment

### 1.1 Dependency Version Alignment

You must strictly align the versions of VeRL and transformers; otherwise, the precision results are directly affected.

If you cannot achieve strong alignment for other key dependencies (torch, megatron, vllm), prioritize keeping them consistent or similar.

### 1.2 Model Weight Alignment

Check whether the model weights and the config.json file are completely consistent.


## 2. Input Data Alignment

Add the following configuration to the verl training startup script:

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. Configure Alignment

When performing precision alignment between an NPU and a GPU, check whether the configurations are completely aligned. This includes:
1. Directly compare the configurations written by the script.
2. Save logs during the running process; collect the configurations from the console log output for comparison; verify the consistency of the default parameter configurations; and ensure that the key parameters are aligned.


## 4. Fixing Determinism

### 4.1 Fixing Random Seeds

Install `msprobe` in the environment:

```bash
pip install mindstudio-probe
```

Add a deterministic function at the beginning of the worker file:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 Fixed Communication Environment Variables

In multiple-device communication scenarios:

- In HCCL communication (default scenario):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- Under LCCL communication (enabled by exporting HCCL_OP_EXPANSION_MODE="AIV"):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

On a single device without communication:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. Verify training precision

### 5.1 Training Instrumentation

**Stubbing** refers to retaining the input and output data of the current stage to facilitate comparative analysis of the results. When troubleshooting precision issues, you need to perform stubbing to help locate the problem. A common stubbing method is to directly dump the data from the rollout stage.

**Step 1: Generate baseline data in a GPU environment**

First, run a GPU script and enable the following configuration:

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```
You can save the inference results of each step as a JSONL file.

**Step 2: Reproduce and verify in the NPU environment**

Enable the following parameters on the NPU, reuse the sequence generated in the previous step, and run end-to-end:

```bash
skip.rollout.enable=True \
skip.rollout.dump_dir=/path/to/rollout_dump \
```

**Step 3: Compare metrics**

Use the same stubbed inference results as input, keep the training configuration consistent, and fix the randomness. Then, compare the rewards/pg_loss/grad_norm values between the NPU and the GPU to check for differences.


## 6. Verifying Inference Precision

### 6.1 resharding

Before inference officially starts, vLLM performs a **dummy run**. It infers one token to evaluate the device memory usage during inference and then allocates the device memory. In vLLM, you can set the load_format parameter during LLM initialization to determine whether the dummy run uses randomly initialized weights (dummy) or real weights (safetensors). In VeRL, you specify this parameter using **actor_rollout_ref.rollout.load_format**.

When garbled output occurs during inference, if the engine initialization method is **load_format=dummy**, the sharding is highly likely to have issues. Even if the output becomes normal after switching to safetensors, the sharding still has issues, and you need to compare the forward pass.


### 6.2  Inference Result Alignment

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

Save the inference results of each step as a jsonl file. You can directly open the jsonl file to quickly check whether the full-network inference results are garbled. This helps you isolate inference precision issues.


Before dumping inference data, if reproducing the inference precision issue consumes excessive resources, try reducing the reproduction cost, scale, and data to dump and compare. In multi-batch and long-sequence scenarios, try to reproduce the issue by sending a single-batch request and reducing the sequence length.


## 7. Dump comparison

[Precision Debugger](../../../en/dev_guide/precision_analysis/precision_debugger.md): After you locate the stage where the problem occurs, use the msprobe tool to dump data for detailed analysis.

During inference or training, the model may produce outputs that deviate from expectations, generate anomalies, or even encounter numerical instability issues such as NaN/Inf. To locate the root cause, you need to perform fine-grained monitoring of the model execution path. This includes collecting intermediate features, weights, activation values, and the inputs and outputs of each key layer. You also need to record context information such as prompts, tensor dtypes, and hardware configurations. By capturing these core tensors and metadata, you can systematically trace the source of precision degradation or numerical errors.




