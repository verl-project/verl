# Copyright 2024-2025 BAAI and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU sequence packing utilities, attention monkey patches, and memory layout helpers."""

import logging
import os
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import logprobs_from_logits

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default sequence packing constants
SEQUENCE_ALIGNMENT_MULTIPLE = 128


def unwrap_metadata(val):
    """Recursively unwraps metadata values (lists, single-element tensors) to standard Python types."""
    if isinstance(val, list):
        if len(val) > 0:
            return unwrap_metadata(val[0])
        return None
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        elif val.numel() > 1:
            return val.flatten()[0].item()
    return val


def monkey_patch_varlen_attention_tpu():
    """Patches TorchTitan's VarlenAttention forward method to use native scaled_dot_product_attention on TPU."""
    try:
        from torchtitan.models.common.attention import VarlenAttention

        def tpu_varlen_forward(self, xq, xk, xv, *, attention_masks, scale=None, **kwargs):
            if hasattr(attention_masks, "cu_seq_q") or hasattr(attention_masks, "cu_seqlens_q") or hasattr(attention_masks, "cu_seqlens"):
                cu_seqs = getattr(
                    attention_masks,
                    "cu_seq_q",
                    getattr(attention_masks, "cu_seqlens_q", getattr(attention_masks, "cu_seqlens", None)),
                )
                total_tokens = xq.shape[1]
                positions = torch.arange(total_tokens, device=xq.device)
                seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
                same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
                causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)

                mask = same_seq_mask & causal_mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif isinstance(attention_masks, torch.Tensor):
                if attention_masks.dim() == 4:
                    mask = attention_masks.to(torch.bool)
                else:
                    seq_len = xq.shape[1]
                    positions = torch.arange(seq_len, device=xq.device)
                    causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

                    padding_mask = attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
                    mask = causal_mask & padding_mask
            else:
                seq_len = xq.shape[1]
                positions = torch.arange(seq_len, device=xq.device)
                mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

            q = xq.transpose(1, 2)
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)

            if q.shape[1] != k.shape[1]:
                num_repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            return attn_out.transpose(1, 2)

        VarlenAttention.forward = tpu_varlen_forward
        logger.info("Successfully patched VarlenAttention.forward for TPU execution.")
    except Exception as e:
        logger.warning(f"Failed to patch VarlenAttention: {e}")


def compute_global_batch_num_tokens(data: TensorDict, dp_group, tp_size: int) -> Any:
    """Computes global batch token count for loss normalization on TPU.

    On TPU, performing CPU-side all-reduce for loss normalization avoids graph desynchronization
    and JIT compilation lockups across ranks.
    """
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


def synchronize_tpu_loss(loss: torch.Tensor):
    """Materializes forward graph loss without blocking to split XLA forward and backward compilation passes."""
    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(loss, wait=False)
    except ImportError:
        pass


def compute_tpu_max_seq_len(input_ids: torch.Tensor) -> int:
    """Rounds up sequence length to a multiple of 128 to ensure memory stride alignment on TPU."""
    offsets = input_ids.offsets()
    offsets_tpu = torch.empty(offsets.shape, dtype=offsets.dtype, device=offsets.device)
    offsets_tpu.copy_(offsets)
    max_seq_len = int(max(offsets_tpu.cpu().diff()))
    max_seq_len = (
        (max_seq_len + SEQUENCE_ALIGNMENT_MULTIPLE - 1) // SEQUENCE_ALIGNMENT_MULTIPLE
    ) * SEQUENCE_ALIGNMENT_MULTIPLE
    return max_seq_len


def safe_to_padded_tensor(nt: Any, padding: Any = 0, output_size: Any = None) -> torch.Tensor:
    """Safely converts a NestedTensor to a padded dense tensor on TPU.

    PyTorch TPU currently lacks native C++ kernel support for `aten::_jagged_to_padded_dense_forward`.
    Falling back to `unbind()` + tensor slice assignment avoids operator runtime errors on TPU.
    """
    if not getattr(nt, "is_nested", False):
        return nt
    try:
        return torch.nested.to_padded_tensor(nt, padding=padding, output_size=output_size)
    except Exception:
        tensors = nt.unbind()
        if not tensors:
            return torch.empty(output_size if output_size is not None else (0,), device=nt.device, dtype=nt.dtype)
        if output_size is None:
            batch_size = len(tensors)
            max_len = max(t.shape[0] for t in tensors)
            trailing_dims = tensors[0].shape[1:]
            output_size = (batch_size, max_len, *trailing_dims)
        out = torch.full(output_size, padding, device=tensors[0].device, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            slices = (i,) + tuple(slice(0, s) for s in t.shape)
            out[slices] = t
        return out
