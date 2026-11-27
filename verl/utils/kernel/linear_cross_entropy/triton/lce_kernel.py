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

from verl.plugin.platform import get_platform


def _get_backend_kernels(device_type: str):
    if device_type == "cuda":
        from . import kernels

        return kernels
    if device_type == "npu":
        from . import lce_asc

        return lce_asc
    raise ValueError(f"Unsupported device type: {device_type}")


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
        """Return selected log probabilities and one entropy per hidden row."""
        assert isinstance(temperature, float), f"temperature must be a float, but got {type(temperature)}"
        assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
        with get_platform().nvtx_range("LinearCrossEntropy-forward"):
            backend_kernels = _get_backend_kernels(hidden.device.type)

            reduction_enum = backend_kernels.get_entropy_reduction_enum_number(reduction.lower())

            original_hidden_shape = hidden.shape
            multi_label = labels.ndim == hidden.ndim and labels.shape[:-1] == hidden.shape[:-1]
            if multi_label and labels.shape[-1] == 1 and reduction.lower() != "none":
                multi_label = False
            if multi_label:
                if not labels.shape[-1] or reduction.lower() != "none":
                    raise ValueError("multi-label queries require a positive query count and reduction='none'")
                if labels.dtype != torch.int64:
                    raise ValueError("multi-label queries require int64 labels")
                hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
                weight = weight.contiguous()
                labels = labels.reshape(hidden.shape[0], labels.shape[-1]).contiguous()
            else:
                hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
                weight = weight.contiguous()
                labels = labels.reshape(-1).contiguous()

            logprobs, entropy, maximum, accumulate, entropy_b = backend_kernels.efficient_entropy_forward(
                hidden,
                weight,
                labels,
                reduction_enum,
                temperature,
                dist_process_group,
            )

            ctx.save_for_backward(hidden, weight, labels, maximum, accumulate, entropy_b)
            ctx.original_hidden_shape = original_hidden_shape
            ctx.reduction_enum = reduction_enum
            ctx.dist_process_group = dist_process_group
            ctx.should_return_fp32_grad = False
            ctx.temperature = temperature
        if multi_label:
            return logprobs.reshape(*original_hidden_shape[:-1], labels.shape[-1]), entropy.reshape(
                original_hidden_shape[:-1]
            )
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> list[torch.Tensor]:
        with get_platform().nvtx_range("LinearCrossEntropy-backward"):
            hidden, weight, labels, maximum, accumulate, entropy_b = ctx.saved_tensors
            backend_kernels = _get_backend_kernels(hidden.device.type)

            d_hidden, d_weight = backend_kernels.efficient_entropy_backward(
                dlogprobs.reshape_as(labels).contiguous() if labels.ndim == 2 else dlogprobs.contiguous(),
                dentropy.reshape(-1).contiguous(),
                hidden,
                weight,
                labels,
                maximum,
                accumulate,
                entropy_b,
                ctx.reduction_enum,
                ctx.should_return_fp32_grad,
                ctx.temperature,
                ctx.dist_process_group,
            )
            d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return (d_hidden, d_weight, None, None, None, None)


linear_cross_entropy = LinearCrossEntropy.apply
