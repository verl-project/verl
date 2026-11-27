#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import typing

import torch
import torch.distributed as dist


def configure_liger_flsce(
    *,
    max_tokens: int,
    hidden_size: int,
    local_vocab_size: int,
    process_group: dist.ProcessGroup,
    device: torch.device,
) -> bool:
    from .liger import configure_liger_flsce as configure

    return configure(
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        local_vocab_size=local_vocab_size,
        process_group=process_group,
        device=device,
    )


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    *,
    impl_backend: str = "triton",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fused linear projection, target log-probabilities, and token-level Shannon entropy.

    Fuses ``logits = (hidden @ weight.T) / temperature`` with online softmax, log-probability
    extraction, and entropy computation so the full ``[num_tokens, vocab_size]`` logits tensor
    does not need to be materialized in device memory during forward or backward passes.
    Supports both single-rank execution and vocabulary-parallel tensor parallelism (TP).

    Args:
        hidden (torch.Tensor): Input hidden states of shape ``(..., hidden_size)`` (typically
            ``(num_tokens, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``).
        weight (torch.Tensor): Language-model head weight matrix of shape
            ``(local_vocab_size, hidden_size)``, where ``local_vocab_size = vocab_size // tp_size``
            when ``dist_process_group`` is provided.
        labels (torch.Tensor): Target global vocabulary indices. Supports either single-label
            targets with ``labels.numel() == num_tokens`` (e.g., ``(num_tokens,)`` or
            ``(batch_size, seq_len)``) or multi-label queries of shape ``(..., topk)`` matching
            ``hidden.shape[:-1] + (topk,)`` (supported by ``"triton"`` and ``"tilelang"`` with
            ``reduction="none"`` and ``torch.int64`` dtype).
        temperature (float, optional): Positive scaling factor dividing the raw logits before
            softmax. Defaults to ``1.0`` (``None`` is treated as ``1.0``).
        reduction (str, optional): Reduction applied to ``logprobs``. ``"triton"`` supports
            ``"none"``, ``"sum"``, and ``"mean"``; ``"tilelang"`` and ``"liger"`` support
            ``"none"`` only. Note that ``logprobs`` are negative cross-entropy values (``<= 0``),
            and ``entropy`` is always returned unreduced per token. Defaults to ``"none"``.
        dist_process_group (dist.ProcessGroup, optional): Tensor-parallel process group across
            which ``weight`` is sharded along the vocabulary dimension. Note that in TP backward,
            ``hidden`` gradients are rank-local partial sums and must be all-reduced across
            ``dist_process_group`` by the caller when needed (except where handled internally by
            ``"liger"``). Defaults to ``None`` (single rank).
        impl_backend (str, optional): Kernel implementation backend to dispatch to:
            - ``"triton"``: Triton fused kernel on CUDA GPUs and Ascend NPUs.
            - ``"tilelang"``: TileLang fused kernel on CUDA GPUs and Ascend NPUs.
            - ``"liger"``: Liger TP-FLSCE kernel (requires ``dist_process_group`` and single-label).
            Defaults to ``"triton"``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A ``(logprobs, entropy)`` pair in ``torch.float32``:
            - ``logprobs``: Selected target log-probabilities (i.e., ``-cross_entropy``). Has shape
              ``(..., topk)`` for multi-label queries, token-wise shape when ``reduction="none"``
              with single-label targets, or a scalar ``0D`` tensor when ``reduction`` is ``"sum"``
              or ``"mean"``.
            - ``entropy``: Token-level Shannon entropy ``-sum(p * log(p))`` for each hidden row.
    """
    if not isinstance(impl_backend, str):
        raise TypeError(f"impl_backend must be a string, got {type(impl_backend)}")
    impl_backend = impl_backend.lower()
    if temperature is None:
        temperature = 1.0
    if reduction is None:
        reduction = "none"

    num_tokens = hidden.numel() // hidden.shape[-1]
    if impl_backend == "triton":
        if labels.numel() != num_tokens and (labels.ndim != hidden.ndim or labels.shape[:-1] != hidden.shape[:-1]):
            raise ValueError("labels must contain one or more labels for every hidden row")
    elif impl_backend == "liger" and labels.numel() != num_tokens:
        raise ValueError(f"The {impl_backend} linear cross entropy backend only supports one label per token")

    if impl_backend == "triton":
        from .triton import linear_cross_entropy as triton_lce

        return triton_lce(hidden, weight, labels, float(temperature), str(reduction), dist_process_group)

    if impl_backend == "liger":
        from .liger import linear_cross_entropy as liger_lce

        return liger_lce(hidden, weight, labels, float(temperature), str(reduction), dist_process_group)

    if impl_backend == "tilelang":
        if hidden.device.type not in ("cuda", "npu"):
            raise RuntimeError("The TileLang linear cross entropy backend requires a CUDA or Ascend NPU device")
        if reduction != "none":
            raise ValueError("The TileLang linear cross entropy backend only supports reduction='none'")

        from .tilelang import linear_cross_entropy as tilelang_lce

        return tilelang_lce(hidden, weight, labels, float(temperature), dist_process_group)

    raise ValueError(
        f"Unsupported linear cross entropy backend {impl_backend!r}; choose 'triton', 'liger', or 'tilelang'"
    )


__all__ = ["configure_liger_flsce", "linear_cross_entropy"]
