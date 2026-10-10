#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import typing

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
    """Configure through Liger collectively on WORLD, including non-output PP stages."""
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

    # Names must agree across WORLD but must not be reused for a different
    # partition, even when a later engine uses the same TP size.
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


def _linear_cross_entropy_liger(
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

    tp_function = _require_liger_runtime()

    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1).to(torch.int64)
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError(f"hidden has {hidden.shape[0]} tokens, but labels has {labels.shape[0]} elements")
    if hidden.shape[0] == 0:
        raise ValueError("The Liger TP-FLSCE backend requires at least one token")

    nll, entropy = tp_function.apply(
        hidden,
        weight,
        labels,
        dist_process_group,
        temperature=temperature,
        ignore_index=-100,
        return_entropy=True,
    )
    return -nll, entropy


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: typing.Optional[float] = 1.0,
        reduction: typing.Optional[str] = "none",
        dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    ) -> list[torch.Tensor]:
        """_summary_

        Args:
            ctx (_type_): _description_
            hidden (torch.Tensor): (batch_size, num_tokens, hidden_size) -> (batch_size * num_tokens, hidden_size)
            weight (torch.Tensor): (vocab_size, hidden_size)
            labels (torch.Tensor): (batch_size, num_tokens) -> (batch_size * num_tokens, )
            temperature (typing.Optional[float], optional): _description_. Defaults to 1.0.
            reduction (typing.Optional[str], optional): _description_. Defaults to "none".
            dist_process_group (typing.Optional[dist.ProcessGroup], optional): _description_. Defaults to None.

        Returns:
            typing.List[torch.Tensor]: _description_
        """

        assert isinstance(temperature, float), f"temperature must be a float, but got {type(temperature)}"
        assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            from . import kernels

            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

            original_hidden_shape = hidden.shape
            if len(hidden.shape) != 2:
                hidden = hidden.view(-1, hidden.shape[-1])  # (batch_size * num_tokens, hidden_size)
            if len(labels.shape) != 1:
                labels = labels.view(-1)

            logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(
                hidden, weight, labels, REDUCTION, temperature, dist_process_group
            )

            ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
            ctx.original_hidden_shape = original_hidden_shape
            ctx.REDUCTION = REDUCTION
            ctx.dist_process_group = dist_process_group
            ctx.should_return_fp32_grad = False
            ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> list[torch.Tensor]:
        from . import kernels

        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION
            dist_process_group = ctx.dist_process_group
            should_return_fp32_grad = ctx.should_return_fp32_grad
            temperature = ctx.temperature

            d_hidden, d_weight = kernels.efficient_entropy_backward(
                dlogprobs,
                dentropy,
                hidden,
                weight,
                labels,
                _maximum,
                _accumulate,
                _entropy_b,
                REDUCTION,
                should_return_fp32_grad,
                temperature,
                dist_process_group,
            )
            d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return (d_hidden, d_weight, None, None, None, None)


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
    if not isinstance(impl_backend, str):
        raise TypeError(f"impl_backend must be a string, got {type(impl_backend)}")
    impl_backend = impl_backend.lower()
    if impl_backend == "triton":
        return LinearCrossEntropy.apply(
            hidden,
            weight,
            labels,
            temperature,
            reduction,
            dist_process_group,
        )
    if impl_backend == "liger":
        return _linear_cross_entropy_liger(
            hidden,
            weight,
            labels,
            temperature,
            reduction,
            dist_process_group,
        )
    raise ValueError(f"Unsupported linear cross entropy backend {impl_backend!r}; choose 'triton' or 'liger'")
