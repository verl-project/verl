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

import hashlib

import torch
import torch.distributed as dist

_LIGER_FUNCTION = None


def _require_liger_runtime():
    global _LIGER_FUNCTION

    if _LIGER_FUNCTION is not None:
        return _LIGER_FUNCTION

    try:
        from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyTPFunction
    except ImportError as exc:
        raise RuntimeError("The Liger TP-FLSCE backend requires `liger-kernel==0.8.4`") from exc

    _LIGER_FUNCTION = LigerFusedLinearScaledCrossEntropyTPFunction
    return _LIGER_FUNCTION


def configure_liger_flsce(
    *,
    max_tokens: int,
    hidden_size: int,
    local_vocab_size: int,
    process_group: dist.ProcessGroup,
    device: torch.device,
) -> bool:
    """Configure Liger collectively on WORLD, including non-output PP stages."""
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (max_tokens, hidden_size, local_vocab_size)
    ):
        raise ValueError("Liger TP-FLSCE workspace dimensions must be positive integers")
    if process_group is None:
        raise ValueError("A tensor-parallel process group is required to configure Liger TP-FLSCE")
    try:
        from liger_kernel.ops.configure import FusedLinearCrossEntropyConfig, configure
    except ImportError as exc:
        raise RuntimeError(
            "Megatron's Liger backend requires `liger-kernel==0.8.4` with the public configure API."
        ) from exc

    partitions = [None] * dist.get_world_size()
    dist.all_gather_object(partitions, tuple(dist.get_process_group_ranks(process_group)))
    partition_id = hashlib.sha256(repr(partitions).encode("ascii")).hexdigest()
    group_name = f"verl_tp_{partition_id}"
    return configure(
        process_groups={group_name: process_group},
        bootstrap_group=dist.group.WORLD,
        device=device,
        flsce=FusedLinearCrossEntropyConfig(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            local_vocab_size=local_vocab_size,
            group=group_name,
        ),
    )


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: str,
    dist_process_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(reduction, str):
        raise TypeError(f"reduction must be a string, got {type(reduction)}")
    if reduction.lower() != "none":
        raise NotImplementedError("The Liger TP-FLSCE backend currently supports reduction='none' only")
    if dist_process_group is None:
        raise ValueError("A tensor-parallel process group is required for the Liger TP-FLSCE backend")
    if hidden.ndim not in (2, 3):
        raise ValueError(f"hidden must be 2D or 3D, got shape {tuple(hidden.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")

    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1).to(torch.int64)
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError(f"hidden has {hidden.shape[0]} tokens, but labels has {labels.shape[0]} elements")
    if hidden.shape[0] == 0:
        raise ValueError("The Liger TP-FLSCE backend requires at least one token")

    nll, entropy = _require_liger_runtime().apply(
        hidden,
        weight,
        labels,
        dist_process_group,
        temperature=temperature,
        ignore_index=-100,
        return_entropy=True,
    )
    return -nll, entropy
