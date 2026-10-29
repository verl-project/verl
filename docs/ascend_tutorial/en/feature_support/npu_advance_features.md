# NPU Advanced Feature Guide

> This document introduces the advanced features and optimization capabilities of Ascend NPU in the verl ecosystem for developers' reference.
>
Last updated: 05/13/2026.

---

## Table of Contents

- [NPU Advanced Feature Guide](#npu-advanced-feature-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Inference Backend Advanced Features](#1-inference-backend-advanced-features)
    - [1.1 vLLM Inference Backend](#11-vllm-inference-backend)
    - [1.2 SGLang Inference Backend](#12-sglang-inference-backend)
      - [Advanced Parameter Configuration](#advanced-parameter-configuration)
  - [2. Training Backend Advanced Features](#2-training-backend-advanced-features)
    - [2.1 FSDP Training Backend](#21-fsdp-training-backend)
    - [2.2 Megatron Training Backend](#22-megatron-training-backend)
      - [MindSpeed Monkey Patch Framework Principle](#mindspeed-monkey-patch-framework-principle)
      - [Megatron Advanced Parameter Configuration](#megatron-advanced-parameter-configuration)
        - [Memory and Computation Optimization](#memory-and-computation-optimization)
        - [Fused Operator Acceleration](#fused-operator-acceleration)
        - [Pipeline Parallelism Optimization](#pipeline-parallelism-optimization)
        - [Weight Management](#weight-management)
  - [3. Performance Optimization Features](#3-performance-optimization-features)
    - [3.1 Memory Optimization](#31-memory-optimization)
    - [3.2 Computation Acceleration](#32-computation-acceleration)
    - [3.3 Parallelism Strategies](#33-parallelism-strategies)
  - [4. Mixture of Experts (MoE) Features](#4-mixture-of-experts-moe-features)
    - [vLLM/SGLang Inference MoE Support](#vllmsglang-inference-moe-support)
    - [Megatron Training MoE Support](#megatron-training-moe-support)
  - [5. Limitations and Precautions](#5-limitations-and-precautions)
  - [Appendix: Parameter Quick Reference](#appendix-parameter-quick-reference)
    - [Inference Backend Parameter Quick Reference](#inference-backend-parameter-quick-reference)
    - [Training Backend Parameter Quick Reference](#training-backend-parameter-quick-reference)

---

## 1. Inference Backend Advanced Features

verl currently supports two mainstream inference backends: vLLM and SGLang, both of which can run on Ascend NPU. The following lists the advanced feature parameters supported by each backend.

### 1.1 vLLM Inference Backend

Ascend supports the vLLM inference backend through the **vllm-ascend plugin**. This plugin follows [RFC](https://github.com/vllm-project/vllm/issues/11162), providing a pluggable interface that decouples Ascend NPU from vLLM.

---

### 1.2 SGLang Inference Backend

Ascend supports related functionality through continuous contribution and maintenance to the SGLang community, involving the following core components:

| Component | Description |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of optimized inference kernels for Ascend NPU, including attention, normalization, activation functions, LoRA adapters, and so on |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | Ascend implementation of DeepEP, providing highly optimized expert parallel (EP) communication kernels for MoE models |

#### Advanced Parameter Configuration

| SGLang Parameter | verl Corresponding Common Parameter | Description |
|:---|:---|:---|
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **Attention backend selection** — Should be set to `ascend` on NPU to invoke Ascend optimized kernels |
| `quantization` | `actor_rollout_ref.rollout.quantization` | **Quantization support** — Supports model quantization loading and inference |


> For more SGLang NPU feature parameters, refer to the [sglang community NPU feature support documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features)

---

## 2. Training Backend Advanced Features

### 2.1 FSDP Training Backend

Ascend provides FSDP-related support capabilities through `torch_npu`.

### 2.2 Megatron Training Backend

Megatron is a model parallel training framework launched by NVIDIA. Running on NPU requires additionally installing **MindSpeed** to provide underlying support. MindSpeed uses the **Monkey Patch** technique to transparently replace key Megatron components, achieving NPU adaptation.

#### MindSpeed Monkey Patch Framework Principle

**Trigger entry:**
```python
from mindspeed.megatron_adaptor import repatch
```

**Call chain:**
```
repatch
├── Execute megatron_adaptor.py module import
├── Import features_manager module
├── Execute mindspeed/features_manager/__init__.py
├── @AutoExecuteFunction decorator triggers
├── patch_features() auto-executes
└── Perform apply_features_pre_patches and apply_features_patches operations
```

**Core components:**

| Component | Responsibility |
|:---|:---|
| `Patch` class | Implements dynamic replacement of functions/classes, supports stacking multiple layers of decorators |
| `parse_path()` | Dynamic module import and creation |
| `MindSpeedPatchesManager` | Global singleton managing all patch registrations |
| `MindSpeedFeature` | Feature base class, each feature integrates into the patch system through inheritance |

#### Megatron Advanced Parameter Configuration

##### Memory and Computation Optimization

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | **Pipeline output release** — Release output data after tensors are sent to the next PP stage, reducing peak memory, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | **Recomputation granularity control** — Options: `full` / `selective` / `none`. `full` recomputes the entire Transformer layer, `selective` only recomputes the core attention part, default `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | **Recomputation method** — Requires `recompute_granularity=full`, options: `uniform` / `block`, default `None` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | **Recomputation layers** — Requires `recompute_granularity=full`, larger value means smaller memory footprint and higher computation cost, must be divisible by the number of model layers in the current process |

##### Fused Operator Acceleration

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | **Flash Attention** — Whether to use Flash Attention to accelerate attention computation, default `true` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | **Fused rotary position embedding** — Uses fused operators to accelerate RoPE computation, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | **Fused SwiGLU** — Uses fused operators to accelerate the SwiGLU activation function, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | **Persistent LayerNorm** — Uses a persistent strategy to optimize LayerNorm, default `False` |

##### Pipeline Parallelism Optimization

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | **Loss layer pipeline partitioning** — Treats the loss layer as a standard Transformer layer in partitioning, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | **Embedding layer pipeline partitioning** — Treats the input embedding layer as a standard Transformer layer in partitioning, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | **First stage layers** — Specifies the number of layers in the first pipeline stage, default `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | **Last stage layers** — Specifies the number of layers in the last pipeline stage, default `none` |

##### Weight Management

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | **Megatron-Bridge weight conversion** — Currently needs to be kept `True` |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | **Distributed checkpoint** — Uses distributed format to save/load weights, default `False` |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | **Distributed weights path** — Distributed checkpoint loading path, default `null` |

---

## 3. Performance Optimization Features

### 3.1 Memory Optimization

| Feature | Inference/Training | Description |
|:---|:---|:---|
| KV Cache dynamic release (`free_cache_engine`) | Inference (vLLM) | Automatically unloads KV Cache after the generation phase, enabled by default |
| Memory saver mode (`enable_memory_saver`) | Inference (SGLang) | Supports dynamic memory release/resume, verl default `True` |
| Parameter CPU offload (`param_offload`) | Training (FSDP/Megatron) | Offloads model weights to CPU |
| Optimizer CPU offload (`optimizer_offload`) | Training (FSDP/Megatron) | Offloads optimizer state to CPU |
| Chunked entropy computation (`entropy_from_logits_with_chunking`) | Training (FSDP) | Computes entropy in chunks to reduce peak memory |
| Entropy computation chunk size (`entropy_from_logits_chunk_size`) | Training (FSDP) | Entropy computation chunk size |
| Entropy computation recomputation (`entropy_checkpointing`) | Training (FSDP) | Enables recomputation for entropy computation |
| Pipeline output release (`deallocate_pipeline_outputs`) | Training (Megatron) | Releases transmitted tensors in PP scenarios |
| Activation recomputation (`recompute_granularity`) | Training (Megatron) | Supports three-level granularity control: full/selective/none |

### 3.2 Computation Acceleration

| Feature | Inference/Training | Description |
|:---|:---|:---|
| Chunked prefill (`enable_chunked_prefill`) | Inference (vLLM) | Splits large prefill into chunks and batches with decode |
| Prefix caching (`enable_prefix_caching`) | Inference (vLLM) | Automatically caches shared prefixes, reducing redundant computation |
| Flash Attention | Training (Megatron) | Uses Flash Attention to accelerate attention computation, enabled by default |
| Fused rotary position embedding (`use_fused_rotary_pos_emb`) | Training (Megatron) | Fused operators to accelerate RoPE |
| Fused SwiGLU (`use_fused_swiglu`) | Training (Megatron) | Fused operators to accelerate the SwiGLU activation function |
| Persistent LayerNorm (`persist_layer_norm`) | Training (Megatron) | Optimizes LayerNorm execution strategy |
| Group GEMM (`moe_grouped_gemm`) | Training (Megatron) | Group GEMM optimization for MoE scenarios |

### 3.3 Parallelism Strategies

| Parallelism Type | vLLM | SGLang | FSDP | Megatron | Description |
|:---|:---|:---|:---|:---|:---|
| Data parallelism (DP) | ✅ | ✅ | ✅ | ✅ | Data dimension parallelism |
| Tensor parallelism (TP) | ✅ | ✅ | — | ✅ | Intra-layer tensor sharding |
| Pipeline parallelism (PP) | — | — | — | ✅ | Inter-layer pipeline sharding |
| Expert parallelism (EP) | ✅ | ✅ | — | ✅ | MoE expert dimension parallelism |
| Sequence parallelism (SP/Ulysses) | ✅ | ✅ | ✅ | ✅ | Sequence dimension sharding, supports long sequences |
| Context parallelism (CP) | ✅ | — | — | ✅ | Context parallel processing |

---

## 4. Mixture of Experts (MoE) Features

### vLLM/SGLang Inference MoE Support

- **Expert parallelism (EP)** — Configured through the `ep_size` parameter, assigning different experts to different NPU devices
- SGLang provides highly optimized EP communication kernels through [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)

### Megatron Training MoE Support

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallelism (EP) size, default `1` |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | TP-extended EP size, default `null` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | **Group GEMM** — Uses Group GEMM to optimize expert computation in MoE scenarios, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | **Routing data type** — Data type for routing and the weighted average of expert outputs, options: `fp32`/`fp64`, default `fp32`, improves stability in multi-expert scenarios |

---

## 5. Limitations and Precautions

1. **FSDP1 vs FSDP2 differences**
   - `forward_prefetch` and `use_orig_params` only apply to FSDP1
   - FSDP2 is the default recommended version. For API support level, refer to the [Ascend PyTorch version notes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md)

2. **Recomputation parameter dependencies**
   - `recompute_method` requires `recompute_granularity='full'` to take effect
   - `recompute_num_layers` requires `recompute_granularity='full'` to take effect
   - When `recompute_method='uniform'`, `recompute_num_layers` indicates the number of Transformer layers per recomputation unit, and must be divisible by the number of model layers in the current process

4. **SGLang NPU-specific configuration**
   - `attention_backend` must be set to `ascend` to invoke Ascend optimized kernels
   - `enable_memory_saver` is enabled by default in verl, no additional configuration required

---

## Appendix: Parameter Quick Reference

### Inference Backend Parameter Quick Reference

| Parameter Category | vLLM Parameter | SGLang Parameter | verl Common Parameter |
|:---|:---|:---|:---|
| Model path | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| Memory control | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| Graph mode | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| Quantization | `quantization` | `quantization` | `actor_rollout_ref.rollout.quantization` |
| Maximum sequence length | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| Maximum concurrency | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| Tokenizer | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| Remote code | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP parallelism | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP parallelism | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP parallelism | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### Training Backend Parameter Quick Reference

| Parameter Category | FSDP Parameter | Megatron Parameter |
|:---|:---|:---|
| Parameter offload | `fsdp_config.param_offload` | `megatron.param_offload` |
| Optimizer offload | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| Sequence parallelism | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| Recomputation granularity | — | `override_transformer_config.recompute_granularity` |
| Distributed Checkpoint | — | `use_dist_checkpointing` |
