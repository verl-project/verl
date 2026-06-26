# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Memory-efficient log-probability gather utilities for the FSDP teacher worker."""

from __future__ import annotations

import torch


def chunked_gather_logprobs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Gather ``log_softmax(logits)[..., topk_ids]`` without materializing the full
    ``log_softmax`` tensor.

    Logits are processed in fp32 in contiguous chunks of ``chunk_size`` leading-dim
    rows, so peak memory is ``chunk_size * V * 4`` bytes per side instead of ``N * V * 4``.

    Args:
        logits: ``(..., V)``. May be on any device / in any floating dtype.
        topk_ids: ``(..., K)``. ``int64``. Leading dims must match ``logits``.
        chunk_size: Number of leading-dim rows to process per inner iteration.
            Must be ``> 0``. Smaller values reduce peak memory.

    Returns:
        Tensor of shape ``(..., K)`` with the same dtype and device as ``logits``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if logits.shape[:-1] != topk_ids.shape[:-1]:
        raise ValueError(
            f"Leading shapes of logits {tuple(logits.shape)} and topk_ids "
            f"{tuple(topk_ids.shape)} must match (only the trailing dim may differ)."
        )

    leading_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    k = topk_ids.shape[-1]

    flat_logits = logits.reshape(-1, vocab_size)
    # topk_ids may arrive on CPU; gather requires it on logits.device.
    flat_topk = topk_ids.reshape(-1, k).to(flat_logits.device)
    n = flat_logits.size(0)
    out = torch.empty((n, k), dtype=logits.dtype, device=logits.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_fp32 = flat_logits[start:end].float()
        log_z = torch.logsumexp(chunk_fp32, dim=-1, keepdim=True)
        gathered = torch.gather(chunk_fp32, dim=-1, index=flat_topk[start:end])
        out[start:end] = (gathered - log_z).to(logits.dtype)

    return out.reshape(*leading_shape, k)
