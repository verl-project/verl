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

import typing

import torch
import torch.distributed as dist


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
):
    """Fused linear + cross-entropy (+ token entropy).

    Always returns ``(logprobs, entropy)``. On Ascend NPU, dispatches to CANN
    vocab-parallel fused CE when available (``VERL_NPU_LCE_BACKEND=auto|cann``).
    Hybrid backward: CANN CE when ``dentropy==0``; chunked PyTorch CE+entropy
    (Triton-aligned) when ``dentropy!=0``.
    """
    if temperature is None:
        temperature = 1.0
    if reduction is None:
        reduction = "none"

    if hidden.device.type == "npu":
        from verl.utils.kernel.npu.cann_linear_ce import CannLinearCrossEntropy, should_use_cann_linear_ce

        if should_use_cann_linear_ce(hidden.device):
            hidden_2d = hidden.view(-1, hidden.shape[-1]) if len(hidden.shape) != 2 else hidden
            labels_1d = labels.view(-1) if len(labels.shape) != 1 else labels
            return CannLinearCrossEntropy.apply(
                hidden_2d, weight, labels_1d, float(temperature), str(reduction), dist_process_group
            )
        raise RuntimeError(
            "Fused linear_cross_entropy on NPU requires CANN torch_npu APIs "
            "(VERL_NPU_LCE_BACKEND=auto|cann). Disable use_fused_kernels, or install "
            "a torch_npu build that provides the fused linear-CE ops."
        )

    return LinearCrossEntropy.apply(hidden, weight, labels, float(temperature), str(reduction), dist_process_group)
