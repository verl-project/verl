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

from verl.plugin.platform import get_platform

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


_FULL_VOCAB_BACKENDS = ("cut_cross_entropy", "ml_cross_entropy")


def validate_backend_for_vocab_parallel(configured_backend):
    """Reject full-vocabulary backends for vocab-sharded engines such as Megatron TP.

    apple/ml-cross-entropy needs every vocabulary row on each device, so a
    tensor-parallel output head (which shards the vocabulary) cannot use it.
    Callers pass the configured ``fused_kernel_options.impl_backend`` value.
    """
    if isinstance(configured_backend, str) and configured_backend.lower() in _FULL_VOCAB_BACKENDS:
        raise ValueError(
            f"fused_kernel_options.impl_backend={configured_backend!r} is not supported by the "
            "Megatron engine: apple/ml-cross-entropy requires the full vocabulary on each device, "
            "but Megatron shards the output head across tensor-parallel ranks. Use 'triton' "
            "(optionally with use_liger) or disable use_fused_kernels."
        )


_CUT_CROSS_ENTROPY = None
_CUT_CROSS_ENTROPY_INSTALL_COMMAND = (
    'pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"'
)
# Budget for the scratch used by the entropy estimator. The per-chunk matmul
# upcasts both operands to at least fp32 so its logits match the fp32
# accumulation used by the Triton kernel; keeping the chunk small bounds the
# extra memory below the unfused logits path.
_CUT_CROSS_ENTROPY_ENTROPY_CHUNK_BYTES = 16 * 1024 * 1024


def _require_cut_cross_entropy():
    global _CUT_CROSS_ENTROPY

    if _CUT_CROSS_ENTROPY is not None:
        return _CUT_CROSS_ENTROPY

    try:
        from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy
    except ImportError as exc:
        raise RuntimeError(
            "The cut_cross_entropy fused linear cross entropy backend requires the optional "
            "`cut-cross-entropy` package (apple/ml-cross-entropy). Install it with:\n    "
            f"{_CUT_CROSS_ENTROPY_INSTALL_COMMAND}\n"
            "It also requires PyTorch >= 2.4, Triton >= 3.0, and an Ampere or newer CUDA GPU."
        ) from exc

    _CUT_CROSS_ENTROPY = cce_linear_cross_entropy
    return _CUT_CROSS_ENTROPY


def _check_cut_cross_entropy_device(device: torch.device):
    if device.type != "cuda":
        raise RuntimeError(
            "The cut_cross_entropy fused linear cross entropy backend requires a CUDA device "
            f"(got {device.type!r}). Disable use_fused_kernels or select another impl_backend."
        )


def _entropy_accumulate_dtype(dtype: torch.dtype) -> torch.dtype:
    # The Triton kernel accumulates logits in fp32; fp16/bf16 operands must be
    # upcast for the entropy estimator to agree with it. Wider inputs stay put.
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


def _cut_cross_entropy_entropy_chunk(num_tokens: int, hidden_size: int, itemsize: int) -> int:
    # The upcast weight chunk [chunk, hidden] and the logits [tokens, chunk] are
    # alive at the same time, so size the chunk to keep both within budget.
    row_bytes = (max(int(num_tokens), 1) + max(int(hidden_size), 1)) * itemsize
    return max(1, _CUT_CROSS_ENTROPY_ENTROPY_CHUNK_BYTES // row_bytes)


class _ExpectedLogit(torch.autograd.Function):
    """Compute ``sum_j softmax(z)_j * z_j`` without materializing the logits.

    ``lse`` comes from Cut Cross Entropy, so both paths share the same
    normalization. Vocabulary is processed in chunks in both directions, which
    bounds the extra memory to one chunk of fp32 logits.
    """

    @staticmethod
    def forward(ctx, hidden, weight, lse, valid):
        compute_dtype = _entropy_accumulate_dtype(hidden.dtype)
        itemsize = torch.empty((), dtype=compute_dtype).element_size()
        chunk = _cut_cross_entropy_entropy_chunk(hidden.shape[0], weight.shape[1], itemsize)
        safe_lse = torch.where(valid, lse, torch.full_like(lse, float("inf"))).to(compute_dtype)
        hidden_work = hidden.to(compute_dtype)
        expected = torch.zeros((hidden.shape[0],), dtype=compute_dtype, device=hidden.device)

        for start in range(0, weight.shape[0], chunk):
            stop = min(start + chunk, weight.shape[0])
            logits = hidden_work @ weight[start:stop].to(compute_dtype).t()
            probs = torch.exp(logits - safe_lse[:, None])
            expected += (probs * logits).sum(dim=-1)

        ctx.save_for_backward(hidden, weight, safe_lse, valid, expected)
        ctx.chunk = chunk
        ctx.compute_dtype = compute_dtype
        return expected

    @staticmethod
    def backward(ctx, grad_expected):
        hidden, weight, safe_lse, valid, expected = ctx.saved_tensors
        chunk = ctx.chunk
        compute_dtype = ctx.compute_dtype
        grad_expected = torch.where(valid, grad_expected, torch.zeros_like(grad_expected)).to(compute_dtype)
        hidden_work = hidden.to(compute_dtype)
        # Accumulate the hidden gradient in fp32 even when hidden is bf16: it is
        # summed over every vocabulary chunk.
        grad_hidden = torch.zeros_like(hidden_work)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, weight.shape[0], chunk):
            stop = min(start + chunk, weight.shape[0])
            weight_chunk = weight[start:stop].to(compute_dtype)
            logits = hidden_work @ weight_chunk.t()
            probs = torch.exp(logits - safe_lse[:, None])
            # d/dz_j sum_i p_i z_i = p_j * (1 + z_j - sum_i p_i z_i)
            grad_logits = grad_expected[:, None] * probs * (1.0 + logits - expected[:, None])
            grad_hidden += grad_logits @ weight_chunk
            grad_weight[start:stop] = (grad_logits.t() @ hidden_work).to(weight.dtype)

        return grad_hidden.to(hidden.dtype), grad_weight, None, None


def _linear_cross_entropy_cut_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: str,
    dist_process_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reject unsupported devices before importing the optional package so that
    # NPU/CPU users get a "CUDA-only" message instead of install instructions.
    _check_cut_cross_entropy_device(hidden.device)
    cce_linear_cross_entropy = _require_cut_cross_entropy()

    if not isinstance(reduction, str):
        raise TypeError(f"reduction must be a string, got {type(reduction)}")
    if reduction.lower() != "none":
        raise NotImplementedError("The cut_cross_entropy backend currently supports reduction='none' only")
    if dist_process_group is not None:
        raise RuntimeError(
            "The cut_cross_entropy backend does not support tensor-parallel process groups because "
            "apple/ml-cross-entropy needs the full vocabulary on every device. Use the triton backend "
            "for vocab-parallel output heads."
        )
    if hidden.ndim not in (2, 3):
        raise ValueError(f"hidden must be 2D or 3D, got shape {tuple(hidden.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")

    hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    num_tokens = hidden.shape[0]
    if labels.numel() != num_tokens:
        raise ValueError(
            "The cut_cross_entropy linear cross entropy backend only supports one label per token; "
            "multi-label/top-k queries are not supported"
        )
    labels = labels.reshape(-1).to(torch.int64).contiguous()

    scaled_hidden = hidden / float(temperature) if temperature not in (None, 1.0) else hidden
    nll, lse = cce_linear_cross_entropy(
        scaled_hidden,
        weight,
        labels,
        bias=None,
        reduction="none",
        return_lse=True,
        ignore_index=-100,
    )

    valid = labels != -100
    expected_logit = _ExpectedLogit.apply(scaled_hidden, weight, lse, valid)
    log_probs = torch.where(valid, -nll, torch.zeros_like(nll))
    entropy = torch.where(valid, lse - expected_logit, torch.zeros_like(lse))
    return log_probs, entropy


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
        """Return selected log probabilities and one entropy per hidden row.

        A trailing query axis on labels is preserved for unreduced output.
        Single-label reductions retain the original flattened-token interface.
        """
        assert isinstance(temperature, float), f"temperature must be a float, but got {type(temperature)}"
        assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
        with get_platform().nvtx_range("LinearCrossEntropy-forward"):
            from . import kernels

            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

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
                hidden = hidden.view(-1, hidden.shape[-1])
                labels = labels.reshape(-1).contiguous()

            logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(
                hidden, weight, labels, REDUCTION, temperature, dist_process_group
            )

            ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
            ctx.original_hidden_shape = original_hidden_shape
            ctx.REDUCTION = REDUCTION
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
        from . import kernels

        with get_platform().nvtx_range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION
            dist_process_group = ctx.dist_process_group
            should_return_fp32_grad = ctx.should_return_fp32_grad
            temperature = ctx.temperature

            d_hidden, d_weight = kernels.efficient_entropy_backward(
                dlogprobs.reshape_as(labels).contiguous() if labels.ndim == 2 else dlogprobs.contiguous(),
                dentropy.reshape(-1).contiguous(),
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
    """Fused linear + cross-entropy (+ token entropy).

    Always returns ``(logprobs, entropy)``. On Ascend NPU, dispatches to CANN
    vocab-parallel fused CE when available (``VERL_NPU_LCE_BACKEND=auto|cann``).
    Hybrid backward: CANN CE when ``dentropy==0``; chunked PyTorch CE+entropy
    (Triton-aligned) when ``dentropy!=0``.

    ``impl_backend="cut_cross_entropy"`` (alias ``"ml_cross_entropy"``) uses
    apple/ml-cross-entropy for the log-probability path when the optional
    ``cut-cross-entropy`` package is installed. It requires a CUDA device with
    bf16 tensor cores, the full vocabulary on every device, and a single label
    per token.
    """
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
    elif impl_backend in ("triton", "liger", "cut_cross_entropy", "ml_cross_entropy") and labels.numel() != num_tokens:
        raise ValueError(f"The {impl_backend} linear cross entropy backend only supports one label per token")
    if impl_backend == "triton":
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
    if impl_backend == "liger":
        return _linear_cross_entropy_liger(
            hidden,
            weight,
            labels,
            temperature,
            reduction,
            dist_process_group,
        )
    if impl_backend == "tilelang":
        if hidden.device.type != "cuda":
            raise RuntimeError("The TileLang linear cross entropy backend is CUDA-only")
        if reduction != "none":
            raise ValueError("The TileLang linear cross entropy backend only supports reduction='none'")

        from verl.utils.kernel.tilelang.linear_cross_entropy import linear_cross_entropy as tilelang_lce

        return tilelang_lce(
            hidden,
            weight,
            labels,
            float(temperature),
            dist_process_group,
        )
    if impl_backend in ("cut_cross_entropy", "ml_cross_entropy"):
        return _linear_cross_entropy_cut_cross_entropy(
            hidden,
            weight,
            labels,
            temperature,
            reduction,
            dist_process_group,
        )
    raise ValueError(
        f"Unsupported linear cross entropy backend {impl_backend!r}; choose 'triton', 'liger', "
        "'tilelang', or 'cut_cross_entropy'"
    )
