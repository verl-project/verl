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
    """Dispatch fused linear cross entropy to the selected implementation."""
    if not isinstance(impl_backend, str):
        raise TypeError(f"impl_backend must be a string, got {type(impl_backend)}")
    impl_backend = impl_backend.lower()
    if temperature is None:
        temperature = 1.0
    if reduction is None:
        reduction = "none"

    num_tokens = hidden.numel() // hidden.shape[-1]
    if impl_backend == "triton" and hidden.device.type != "npu":
        if labels.numel() != num_tokens and (labels.ndim != hidden.ndim or labels.shape[:-1] != hidden.shape[:-1]):
            raise ValueError("labels must contain one or more labels for every hidden row")
    elif impl_backend in ("triton", "liger") and labels.numel() != num_tokens:
        raise ValueError(f"The {impl_backend} linear cross entropy backend only supports one label per token")

    if impl_backend == "triton":
        if hidden.device.type == "npu":
            from verl.utils.kernel.npu.cann_linear_ce import CannLinearCrossEntropy, should_use_cann_linear_ce

            if should_use_cann_linear_ce(hidden.device):
                hidden_2d = hidden.view(-1, hidden.shape[-1]) if hidden.ndim != 2 else hidden
                labels_1d = labels.view(-1) if labels.ndim != 1 else labels
                return CannLinearCrossEntropy.apply(
                    hidden_2d,
                    weight,
                    labels_1d,
                    float(temperature),
                    str(reduction),
                    dist_process_group,
                )
            raise RuntimeError(
                "Fused linear_cross_entropy on NPU requires CANN torch_npu APIs "
                "(VERL_NPU_LCE_BACKEND=auto|cann). Disable use_fused_kernels, or install "
                "a torch_npu build that provides the fused linear-CE ops."
            )

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
