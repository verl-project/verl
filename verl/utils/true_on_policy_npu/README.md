# Train-Inference Consistency Patch System for VERL (NPU)

This module ensures consistency between training (Megatron/MindSpeed) and inference (vLLM) backends when running on Ascend NPU.

## Usage

### 1. Enable Runtime Patches

```bash
export TRAIN_INFER_CONSIST=1
```

### 2. Apply Source Code Fixes (Required)

Before running, these files **must** be directly modified in their respective repositories:

**vllm/vllm/model_executor/layers/linear.py:**
```python
# Lines 1407-1426: Change all-reduce to reduce-scatter + all-gather
if self.reduce_results and self.tp_size > 1:
    # Split all-reduce into reduce-scatter + all-gather to match training
    pad_size = (self.tp_size - (output_parallel.shape[0] % self.tp_size)) % self.tp_size
    if pad_size > 0:
        output_parallel = torch.nn.functional.pad(output_parallel, (0, 0, 0, pad_size))
    scattered = tensor_model_parallel_reduce_scatter(output_parallel, dim=0)
    output = tensor_model_parallel_all_gather(scattered, dim=0)
    if pad_size > 0:
        output = output[:-pad_size]
```

**vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:**
```python
# Import order fix (lines 17-30)
from typing import Optional

import torch
import torch.nn.functional as F  # Must come before torch_npu
import torch_npu
from torch.nn.functional import pad
...
```

**vllm-ascend/vllm_ascend/ops/activation.py:**
```python
# Lines 43-44: SwiGLU using SiLU
else:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]
```

**vllm-ascend/vllm_ascend/ascend_forward_context.py:**
```python
# Lines 284-285: Force ALLTOALL when TRAIN_INFER_CONSIST=1
if os.environ.get("TRAIN_INFER_CONSIST", "0") == "1":
    moe_comm_type = MoECommType.ALLTOALL
```

**vllm-ascend/vllm_ascend/attention/attention_v1.py:**
```python
# Lines 590-660: Batch invariant FlashAttention handling
if os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1":
    # Handle tokens with FlashAttention
    ...
```

**vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:**
```python
# Import order fix (lines 23-36)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.comm_utils import (
    async_all_to_all, gather_from_sequence_parallel_region)

# TokenDispatcherWithAll2AllV.token_dispatch() additions:
# Line 443: num_permuted calculation for weights processing
num_permuted = permutated_local_input_tokens.shape[0]

# Lines 460-469: topk_weights scatter logic
flat_weights = topk_weights.view(-1)
permuted_weights = torch.zeros(num_permuted, 1, dtype=flat_weights.dtype, device=flat_weights.device)
permuted_weights.scatter_(
    0,
    reversed_local_input_permutation_mapping.unsqueeze(-1).long(),
    flat_weights.unsqueeze(-1))

# Lines 475-483: AllToAll for weights + token permute
_, global_weights, weights_handle = async_all_to_all(
    permuted_weights, output_splits, input_splits, self.ep_group)
weights_handle.wait()
if self.num_local_experts > 1 and global_input_tokens_local_experts_indices is not None:
    global_weights, _ = torch_npu.npu_moe_token_permute(
        global_weights, global_input_tokens_local_experts_indices)

# Line 503: Return with topk_scales
topk_scales=global_weights,

# Line 645: _combine_postprocess with topk_weights
probs=torch.ones_like(context_metadata["topk_weights"]),
```

**MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py:**
```python
# Import order fix (lines 4-16)
import math
import os
from typing import Optional

import torch
from torch import Tensor
import torch_npu
from flash_attn import flash_attn_varlen_func
...
```

### 3. Run VERL

```bash
python -m verl.trainer.main_ppo ...
```

## Tested Configuration

### Model and Dataset

- Model: `Qwen3-4B`
- Train dataset: `dapo-math-17k.parquet`
- Validation dataset: `dapo-math-17k.parquet`

### Runtime Topology (from test script)

- Device: NPU (`trainer.device=npu`)
- Nodes / NPUs: `NNODES=1`, `NPUS_PER_NODE=16`
- Training backend: Megatron (`pipeline_model_parallel_size=4`, `tensor_model_parallel_size=4`, `context_parallel_size=1`)
- Inference backend: vLLM (`tensor_model_parallel_size=4`, `VLLM_USE_V1=1`)
- Key flags: `TRAIN_INFER_CONSIST=1`, `VLLM_BATCH_INVARIANT=1`

### Dependency Revisions

- `vllm`: release `v0.13.0` (commit `72506c9`)
- `vllm-ascend`: release `v0.13.0rc3` (commit `1dee1509`)
- `Megatron-LM`: release `core_v0.12.1` (commit `a845aa7`)
- `MindSpeed`: release `v2.3.0_core_r0.12.1` (commit `674226a1`)

## Complete File Modification Checklist

### 详细修改清单
vLLM-Ascend modifications are maintained in local notes (not included in this MR).

### vllm (Inference Backend)
- [x] `vllm/model_executor/layers/linear.py` - RowParallelLinear RS+AG (lines 1407-1426)

### vllm-ascend (NPU Inference) - 核心修改
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py` - **5处修改** (lines 443, 460-469, 475-483, 503, 645)
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py` - Import顺序 (line 20-21)
- [x] `vllm-ascend/vllm_ascend/ops/activation.py` - SwiGLU SiLU-based (lines 43-44)
- [x] `vllm-ascend/vllm_ascend/ascend_forward_context.py` - MoE ALLTOALL强制 (lines 284-285)
- [x] `vllm-ascend/vllm_ascend/attention/attention_v1.py` - FlashAttention batch invariant (lines 590-660)

### vllm-ascend (NPU Inference) - 其他修改
- [x] `vllm-ascend/vllm_ascend/worker/worker.py` - init_batch_invariance调用 (line 530-532)
- [x] `vllm-ascend/vllm_ascend/distributed/parallel_state.py` - 额外进程组
- [x] `vllm-ascend/vllm_ascend/distributed/utils.py` - 通信工具函数
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py` - 通信模式
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py` - 通信方法选择
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/comm_utils.py` - AllToAll实现
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py` - Prepare/Finalize逻辑
- [x] `vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py` - tid2eid支持
- [x] `vllm-ascend/vllm_ascend/ops/linear_op.py` - RS+AG模式
- [x] `vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py` - TP通信组
- [x] `vllm-ascend/vllm_ascend/batch_invariant.py` - 批不变性初始化

### MindSpeed (Training Backend)
- [x] `MindSpeed/mindspeed/core/transformer/flash_attention/flash_attention/adaptor.py` - Import order

### Megatron-LM (Training Backend)
- [x] No source modifications - all handled via runtime environment variables

## Architecture

```
verl/utils/true_on_policy_npu/
├── __init__.py          # Entry point and patch orchestration
├── vllm_patch.py        # RowParallelLinear RS+AG patch
├── vllm_ascend_patch.py # Batch invariant + SwiGLU patches
├── megatron_patch.py    # Communication determinism env vars
├── mindspeed_patch.py   # Documentation placeholder
└── README.md            # This file
```

## Runtime Patches (Auto-applied)

| Patch | File | Description |
|-------|------|-------------|
| RowParallelLinear | vllm_patch.py | RS+AG instead of all-reduce |
| Batch Invariant | vllm_ascend_patch.py | Deterministic operators |
| AscendSiluAndMul | vllm_ascend_patch.py | SiLU-based SwiGLU |
| Attention | vllm_ascend_patch.py | Batch invariant handling |
| TokenDispatcher | vllm_ascend_patch.py | topk_weights AllToAll handling |
| Communication | megatron_patch.py | HCCL/NCCL determinism |

## Important Notes

1. **Source code fixes cannot be runtime patched** - Import order and some conditional logic must be in source
2. **Runtime patches are idempotent** - Can be safely applied multiple times
3. **Patches are backend-specific** - Only apply when training=megatron-like + inference=vllm + device=npu
4. **Reference implementation** - MindSpeed's `layernorm_column_parallel_linear.py` shows correct RS+AG pattern

## Debugging

```bash
export VERL_LOGGING_LEVEL=DEBUG
export TRAIN_INFER_CONSIST=1
```

## Why Separate Source and Runtime Fixes?

**Source fixes required for:**
- Import statement ordering (executes at module load time)
- Conditional logic based on module-level state
- Complex function modifications that depend on internal imports

**Runtime patches for:**
- Method body modifications (can be monkey-patched)
- Environment variable setup
- Logging and instrumentation
