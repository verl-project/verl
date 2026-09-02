# NPU Advanced Features Guide

> This document introduces the advanced features and optimization capabilities of the Ascend NPU in the verl ecosystem for developer reference.
>
Last updated: 05/13/2026.

---

## Contents

- [NPU Advanced Features Guide](#npu-advanced-features-guide)
  - [Contents](#Contents)
  - [1. Advanced Features of the Inference Backend](#1-advanced-features-of-the-inference-backend)
    - [1.1 vLLM Inference Backend](#11-vllm-inference-backend)
    - [1.2 SGLang Inference Backend](#12-sglang-inference-backend)
      - [Advanced Parameter Configuration](#advanced-parameter-configuration)
  - [2. Training Backend Advanced Features](#2-training-backend-advanced-features)
    - [2.1 FSDP Training Backend](#21-fsdp-training-backend)
    - [2.2 Megatron Training Backend](#22-megatron-training-backend)
      - [MindSpeed Monkey Patch Framework Principle](#mindspeed-monkey-patch-framework-principle)
      - [Megatron Advanced Parameter Configuration](#megatron-advanced-parameter-configuration)
        - [Memory and Compute Optimization](#memory-and-compute-optimization)
        - [Fused Operator Acceleration](#fused-operator-acceleration)
        - [Pipeline Parallelism Optimization](#pipeline-parallelism-optimization)
        - [Weight Management](#weight-management)
  - [3. Performance Optimization Features](#3-performance-optimization-features)
    - [3.1 Memory Optimization](#31-memory-optimization)
    - [3.2 Compute Acceleration](#32-compute-acceleration)
    - [3.3 Parallel Strategy](#33-parallel-strategy)
  - [4. Mixture of Experts (MoE) Features](#4-mixture-of-experts-moe-features)
    - [vLLM/SGLang Inference MoE Support](#vllmsglang-inference-moe-support)
    - [Megatron Training MoE Support](#megatron-training-moe-support)
  - [5. Limitations and Precautions](#5-limitations-and-precautions)
  - [Appendix: Parameter Quick Reference](#appendix-parameter-quick-reference)
    - [Inference Backend Parameters Quick Reference](#inference-backend-parameters-quick-reference)
    - [Training Backend Parameter Quick Reference](#training-backend-parameter-quick-reference)

---

## 1. Advanced Features of the Inference Backend

Currently, verl supports two mainstream inference backends, vLLM and SGLang, both of which run on Ascend NPUs. The following lists the advanced feature parameters supported by each backend.

### 1.1 vLLM inference backend

Ascend supports the vLLM inference backend through the **vllm-ascend plugin**. This plugin follows the [RFC](https://github.com/vllm-project/vllm/issues/11162) and provides a pluggable interface to decouple the Ascend NPU from vLLM.

---

### 1.2 SGLang inference backend

Ascend NPU supports related features through continuous development and maintenance in the SGLang community, involving the following core components:

| Component | Description |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of optimized inference kernels for Ascend NPU, including attention mechanisms, normalization, activation functions, LoRA adapters, and so on |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | The Ascend implementation of DeepEP, providing highly optimized expert parallelism (EP) communication kernels for MoE models |

#### Advanced Parameter Configuration

| SGLang parameter | Corresponding verl general parameter | Description |
|:---|:---|:---|
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **Attention backend selection** — Set this to `ascend` on the NPU to call the Ascend optimized kernel. |
| `quantization` | `actor_rollout_ref.rollout.quantization` | **Quantization support** — Supports model quantization loading and inference. |


> For more SGLang NPU feature parameters, refer to the [sglang community NPU feature support documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

---

## 2. Training Backend Advanced Features

### 2.1 FSDP Training Backend

Ascend provides FSDP-related support capabilities through `torch_npu`.

### 2.2 Megatron Training Backend

Megatron is a model parallelism training framework introduced by NVIDIA. To run Megatron on an NPU, you must also install **MindSpeed** to provide underlying support. MindSpeed uses the **Monkey Patch** technology to seamlessly replace key Megatron components, achieving NPU adaptation.

#### MindSpeed Monkey Patch Framework Principle

**Trigger entry:**
```python
from mindspeed.megatron_adaptor import repatch
```

**Call chain:**
```
repatch
├── Executes the megatron_adaptor.py module import
├── Imports the features_manager module
├── Executes mindspeed/features_manager/__init__.py
├── The @AutoExecuteFunction decorator is triggered
├── patch_features() is automatically executed
└── Performs the apply_features_pre_patches and apply_features_patches operations
```

**Core components:**

| Component | Responsibility |
|:---|:---|
| `Patch` class | Implements dynamic replacement of functions and classes, and supports stacking multiple layers of decorators |
| `parse_path()` | Dynamically imports and creates modules |
| `MindSpeedPatchesManager` | A global singleton that manages all patch registrations |
| `MindSpeedFeature` | The base feature class. Each feature integrates the patch system through inheritance |

#### Megatron Advanced Parameter Configuration

##### Memory and Compute Optimization

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | **Pipeline output deallocation** — Releases output data after the tensor is sent to the next PP stage, reducing the device memory peak. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | **Recomputation granularity control** — Optional values are `full`, `selective`, and `none`. `full` recomputes the entire Transformer layer, and `selective` recomputes only the core attention part. The default value is `none`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | **Recomputation method** — Requires `recompute_granularity=full`. Optional values are `uniform` and `block`. The default value is `None`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | **Number of recomputation layers** — Requires `recompute_granularity=full`. A larger value reduces device memory usage but increases computation cost. The value must be divisible by the number of model layers in the current process. |

##### Fused Operator Acceleration

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | **Flash Attention** — Whether to use Flash Attention to accelerate attention computation, default `true` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | **Fused Rotary Position Embedding** — Uses fused operators to accelerate RoPE computation, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | **Fused SwiGLU** — Uses fused operators to accelerate the SwiGLU activation function, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | **Persistent LayerNorm** — Uses a persistent strategy to optimize LayerNorm, default `False` |

##### Pipeline Parallelism Optimization

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | **Loss layer pipeline split** — Treats the loss layer as a standard Transformer layer for pipeline splitting. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | **Embedding layer pipeline split** — Treats the input embedding layer as a standard Transformer layer for pipeline splitting. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | **Number of layers in the first stage** — Specifies the number of layers in the first pipeline stage. The default value is `none`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | **Number of layers in the last stage** — Specifies the number of layers in the last pipeline stage. The default value is `none`. |

##### Weight Management

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | **MBridge weight conversion** — Enable mbridge for weight format conversion |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | **Distributed checkpoint** — Save and load weights in distributed format. The default value is `False` |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | **Distributed weight path** — The load path for the distributed checkpoint. The default value is `null` |

---

## 3. Performance Optimization Features

### 3.1 Memory Optimization

| Feature | Inference/Training | Description |
|:---|:---|:---|
| KV Cache dynamic release (`free_cache_engine`) | Inference (vLLM) | Automatically offloads the KV Cache after the generation phase; enabled by default |
| Memory saving mode (`enable_memory_saver`) | Inference (SGLang) | Supports dynamic device memory release and restoration; verl defaults to `True` |
| Parameter CPU offloading (`param_offload`) | Training (FSDP/Megatron) | Offloads model weights to the CPU |
| Optimizer CPU offloading (`optimizer_offload`) | Training (FSDP/Megatron) | Offloads optimizer states to the CPU |
| Chunked entropy computation (`entropy_from_logits_with_chunking`) | Training (FSDP) | Computes entropy in chunks to reduce peak device memory usage |
| Entropy computation chunk size (`entropy_from_logits_chunk_size`) | Training (FSDP) | Chunk size for entropy computation |
| Entropy computation recomputation (`entropy_checkpointing`) | Training (FSDP) | Enables recomputation for entropy computation |
| Pipeline output deallocation (`deallocate_pipeline_outputs`) | Training (Megatron) | Releases transferred tensors in pipeline parallelism scenarios |
| Activation recomputation (`recompute_granularity`) | Training (Megatron) | Supports three granularity levels: full, selective, and none |

### 3.2 Compute Acceleration

| Feature | Inference/Training | Description |
|:---|:---|:---|
| Chunked prefill (`enable_chunked_prefill`) | Inference (vLLM) | Splits large prefill into chunks and processes them with the decode batch |
| Prefix caching (`enable_prefix_caching`) | Inference (vLLM) | Automatically caches shared prefixes to reduce redundant computation |
| Flash Attention | Training (Megatron) | Uses Flash Attention to accelerate attention computation; enabled by default |
| Fused rotary position embedding (`use_fused_rotary_pos_emb`) | Training (Megatron) | Uses fused operators to accelerate RoPE |
| Fused SwiGLU (`use_fused_swiglu`) | Training (Megatron) | Uses fused operators to accelerate the SwiGLU activation function |
| Persistent LayerNorm (`persist_layer_norm`) | Training (Megatron) | Optimizes the LayerNorm execution strategy |
| Group GEMM (`moe_grouped_gemm`) | Training (Megatron) | Group GEMM optimization for MoE scenarios |

### 3.3 Parallel Strategy

| Parallelism Type | vLLM | SGLang | FSDP | Megatron | Description |
|:---|:---|:---|:---|:---|:---|
| Data Parallelism (DP) | ✅ | ✅ | ✅ | ✅ | Data dimension parallelism |
| Tensor Parallelism (TP) | ✅ | ✅ | — | ✅ | Intra-layer tensor splitting |
| Pipeline Parallelism (PP) | — | — | — | ✅ | Inter-layer pipeline splitting |
| Expert Parallelism (EP) | ✅ | ✅ | — | ✅ | MoE expert dimension parallelism |
| Sequence Parallelism (SP/Ulysses) | ✅ | ✅ | ✅ | ✅ | Sequence dimension splitting, supports long sequences |
| Context Parallelism (CP) | ✅ | — | — | ✅ | Context parallel processing |

---

## 4. Mixture of Experts (MoE) features

### vLLM/SGLang Inference MoE Support

- **Expert Parallelism (EP)** — assigns different experts to different NPU devices using the `ep_size` parameter
- SGLang provides highly optimized EP communication kernels through [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)

### Megatron Training MoE Support

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallelism (EP) size. The default value is `1`. |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | TP extends the EP size. The default value is `null`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | **Group GEMM** — Uses Group GEMM to optimize expert computation in MoE scenarios. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | **Routing data type** — The data type for routing and weighted average of expert outputs. You can select `fp32` or `fp64`. The default value is `fp32`, which improves stability in multi-expert scenarios. |

---

## 5. Limitations and Precautions

1. **mbridge and VPP are mutually exclusive**
   - `actor_rollout_ref.actor.megatron.use_mbridge` and `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` (VPP) **cannot be enabled at the same time**
   - Because verl enables mbridge by default, you must manually set `use_mbridge` to `False` when using VPP.

2. **Differences between FSDP1 and FSDP2**
   - `forward_prefetch` and `use_orig_params` apply only to FSDP1.
   - FSDP2 is the default recommended version. For API support, refer to the [Ascend PyTorch Version Description](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md).

3. **Recomputation parameter dependencies**
   - `recompute_method` takes effect only when `recompute_granularity='full'` is set.
   - `recompute_num_layers` takes effect only when `recompute_granularity='full'` is set.
   - When `recompute_method='uniform'`, `recompute_num_layers` specifies the Transformer layers per recomputation unit and must be divisible by the model layers of the current process.

4. **SGLang NPU-specific configurations**
   - Set `attention_backend` to `ascend` to invoke the Ascend optimized kernels.
   - verl enables `enable_memory_saver` by default, so no additional configuration is required.

---

## Appendix: Parameter Quick Reference

### Inference Backend Parameters Quick Reference

| Parameter category | vLLM parameter | SGLang parameter | verl general parameter |
|:---|:---|:---|:---|
| Model path | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| Device memory control | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| Graph mode | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| Quantization | `quantization` | `quantization` | `actor_rollout_ref.rollout.quantization` |
| Maximum sequence length | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| Maximum concurrency | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| Tokenizer | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| Remote code | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP parallelism | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP parallelism | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP parallelism | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### Training backend parameter quick reference

| Parameter Category | FSDP Parameter | Megatron Parameter |
|:---|:---|:---|
| Parameter offload | `fsdp_config.param_offload` | `megatron.param_offload` |
| Optimizer offload | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| Sequence parallelism | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| Recomputation granularity | — | `override_transformer_config.recompute_granularity` |
| Distributed checkpoint | — | `use_dist_checkpointing` |
