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
"""
Implementations of the linear cross entropy with token entropy kernel.
"""
import os
import typing
from dataclasses import dataclass
import itertools

import torch
import torch.distributed as dist

from verl.utils.device import get_device_capability, get_device_name, is_cuda_available


def _is_on_accelerator(t: torch.Tensor) -> bool:
    """Device-agnostic replacement for ``t.is_cuda``.

    ``Tensor.is_cuda`` is True only for real NVIDIA/ROCm CUDA tensors; on other
    backends (MLU, NPU, …) it returns False even though the tensor lives on the
    accelerator. These checks exist only to guard against accidentally passing
    CPU tensors to the kernels, so test ``type != 'cpu'`` instead.
    """
    return t.device.type != "cpu"


try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
    SUPPORT_CUDA_TMA = is_cuda_available and get_device_capability()[0] >= 9 and hasattr(tl, "make_tensor_descriptor")

    from triton.backends.mlu import driver
    _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device()) 
    TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster") 
except ImportError:
    HAVE_TRITON = False
    SUPPORT_CUDA_TMA = False

from verl.utils.device import get_torch_device


if not HAVE_TRITON:
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    @contextmanager
    def null_decorator(*args, **kwargs):
        if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
            return args[0]
        else:

            def inner(func):
                return func

            return inner

    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    tl = MagicMock()

elif SUPPORT_CUDA_TMA:
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: typing.Optional[int]):
        return torch.empty(size, device=get_device_name(), dtype=torch.int8)

    # https://github.com/triton-lang/triton/commit/43625fc968b693ab51884ca95adbcf3e43483fd0
    # Triton 3.5.0 stores allocators in ContextVar; values do not propagate to new
    # threads by default. Some execution paths in verl use thread pools (e.g.,
    # concurrent.futures), so we set a ContextVar *default* to avoid falling
    # back to NullAllocator in worker threads.
    try:
        import contextvars

        import triton.runtime._allocation as _triton_allocation

        if isinstance(getattr(_triton_allocation, "_allocator", None), contextvars.ContextVar):
            _triton_allocation._allocator = contextvars.ContextVar(
                _triton_allocation._allocator.name,
                default=alloc_fn,
            )
    except (ImportError, AttributeError):
        pass

    triton.set_allocator(alloc_fn)


@dataclass
class EntropyReductionEnum:
    """
    Enum for the reduction method of cross entropy.
    """

    _None = 0
    _Sum = 1
    _Mean = 2


def get_entropy_reduction_enum_number(reduction: str) -> int:
    """
    Get the enum number for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if reduction == "none":
        _enum = EntropyReductionEnum._None
    elif reduction == "sum":
        _enum = EntropyReductionEnum._Sum
    elif reduction == "mean":
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return _enum


def get_entropy_reduction_enum(ce_reduction: int) -> EntropyReductionEnum:
    """
    Get the enum for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if ce_reduction == 0:
        _enum = EntropyReductionEnum._None
    elif ce_reduction == 1:
        _enum = EntropyReductionEnum._Sum
    elif ce_reduction == 2:
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid ce_reduction: {ce_reduction}")
    return _enum


@dataclass
class BackwardEnum:
    """
    Enum for the backward method.
    """

    _Total_Fuse_MN = (
        0  # Fuse d_logits & d_hidden & d_weight, no intermediate storage, requires fp32 for d_hidden & d_weight
    )
    _Total_Separate = 1  # Store d_logits, no special requirements for d_hidden & d_weight
    _Split_Dlogits_N = 2  # split d_logits along its N dimension, aka. vocab_size
    _Split_Dlogits_M = 3  # split d_logits along its M dimension, aka. num_tokens


@dataclass
class Config:
    """Configuration for efficient entropy kernel operations.

    Args:
        _backward (BackwardEnum): Backward computation method. Defaults to BackwardEnum._Split_Dlogits_N.
        _use_triton (bool): Whether to use Triton kernels for computation. Defaults to True.
    """

    _backward: BackwardEnum = BackwardEnum._Split_Dlogits_N
    _use_triton: bool = True


_config = Config()


def set_backward_method(backward_method: BackwardEnum):
    """
    Set the backward method.
    """
    global _config
    _config._backward = backward_method


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 1024, "BLOCK_SIZE_K": 256},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_kernel_general_mainloop(
    rank,
    hidden_ptr,
    weight_ptr,
    labels_ptr,
    num_tokens,
    hidden_size,
    vocab_size,
    vocab_per_split,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    global_logprobs_scalar_ptr,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_splits
    core_num_tiles = tl.cdiv(total_tiles, num_jobs)
    tile_start = core_num_tiles * pid
    cnt = tl.maximum(tl.minimum(core_num_tiles, total_tiles - tile_start), 0)

    if USE_TMA:
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    for i in tl.range(cnt):
        tile_idx = tile_start + i
        pid_m = tile_idx % num_pid_m
        pid_n = tile_idx // num_pid_m

        if pid_m == 0 and pid_n == 0:
            tl.store(global_logprobs_scalar_ptr, 0.0)

        # create pointers for the first blocks of hidden
        start_offs_am = pid_m * BLOCK_SIZE_M
        offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        if not USE_TMA:
            hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

        labels = tl.load(labels_ptr + offs_am, mask=offs_am < num_tokens)

        _max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
        _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        _entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for n in range(0, num_pid_n):
            start_offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N
            offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)

            logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            if not USE_TMA:
                weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                hidden_ptrs_ = hidden_ptrs + k * BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs_ = weight_ptrs + k *BLOCK_SIZE_K * stride_weight_k
                if USE_TMA:
                    start_offs_k = k * BLOCK_SIZE_K
                    _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                    _weight = weight_desc.load([start_offs_bn, start_offs_k])
                else:
                    _hidden = tl.load(
                        hidden_ptrs_,
                        mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                        other=0.0,
                    )

                    _weight = tl.load(
                        weight_ptrs_,
                        mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                        & (offs_bn[:, None] < (min((pid_n + 1) * vocab_per_split, vocab_size))),
                        other=0.0,
                    )

                logits = tl.dot(_hidden, _weight.trans(), logits)

            # scale logits by temperature
            logits *= rcp_temperature

            # update global maximum
            _max_old = _max
            m_pid_n = tl.max(logits, axis=1)
            _max = tl.maximum(_max_old, m_pid_n)

            exp_logits = tl.exp(logits - _max[:, None])
            coeff = tl.exp(_max_old - _max)
            _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

            _entropy_b = _entropy_b * coeff + tl.sum(logits * exp_logits, axis=1)

            local_label = labels - (start_offs_bn + rank * vocab_size)
            label_col = tl.where((local_label >= 0) & (local_label < BLOCK_SIZE_N), local_label, 0)
            label_logit = tl.gather(logits, label_col[:, None], axis=1).reshape((BLOCK_SIZE_M,))
            in_band = (local_label >= 0) & (local_label < BLOCK_SIZE_N)
            _logprobs += tl.where(in_band, label_logit, 0.0)

        offs_max_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_max_n = pid_n
        maximum_ptrs = max_ptr + offs_max_n * stride_max_n + offs_max_m * stride_max_m
        tl.store(maximum_ptrs, _max, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))

        accu_ptrs = accu_ptr + offs_max_n * stride_accu_n + offs_max_m * stride_accu_m
        tl.store(accu_ptrs, _accu, mask=(offs_max_m < num_tokens) & (offs_max_n[None] < num_splits))
        entropy_b_ptrs = entropy_b_ptr + offs_max_n * stride_entropy_b_n + offs_max_m * stride_entropy_b_m
        tl.store(entropy_b_ptrs, _entropy_b, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))

        vocab_left_idx = pid_n * vocab_per_split + rank * vocab_size
        vocab_right_idx = min((pid_n + 1) * vocab_per_split, vocab_size) + rank * vocab_size
        mask = (labels >= vocab_left_idx) & (labels < vocab_right_idx)
        mask &= offs_am < num_tokens
        global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
        # tl.atomic_add(global_logprobs_ptrs, _logprobs, mask=mask)
        tl.store(global_logprobs_ptrs, _logprobs, mask=mask)


def epilogue_block_sizes(num_splits):
    if num_splits <= 128:
        return 128, 128
    if num_splits <= 256:
        return 64, 256
    return 32, 512


@triton.jit
def efficient_entropy_triton_kernel_epilogue(
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    num_tokens,
    num_splits,
    global_max_ptr,
    stride_global_max: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    global_entropy_ptr,
    stride_global_entropy: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    global_logprobs_scalar_ptr,
    reduction: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    core_num_tokens = tl.cdiv(num_tokens, num_jobs)
    idx_token = core_num_tokens * pid_m
    limit = tl.minimum(idx_token + core_num_tokens, num_tokens)
    offs_m = idx_token + tl.arange(0, BLOCK_SIZE_M)
    cnt = tl.cdiv(core_num_tokens, BLOCK_SIZE_M)
    for i in tl.range(cnt):
        offs_m_cur = i * BLOCK_SIZE_M + offs_m
        mask_m = offs_m_cur < limit

        offs_n = tl.arange(0, BLOCK_SIZE_N)
        mask_n = mask_m[:, None] & (offs_n[None, :] < num_splits)
        max_ptrs = max_ptr + offs_m_cur[:, None] * stride_max_m + offs_n[None, :] * stride_max_n
        accu_ptrs = accu_ptr + offs_m_cur[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n
        entropy_b_ptrs = entropy_b_ptr + offs_m_cur[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n

        _max = tl.load(max_ptrs, mask=mask_n)
        global_max = tl.max(_max, axis=1)
        _scale = tl.exp(_max - global_max[:, None])
        _accu = tl.load(accu_ptrs, mask=mask_n)
        _entropy_b = tl.load(entropy_b_ptrs, mask=mask_n)
        global_accu = tl.sum(_scale * _accu, axis=1)
        global_entropy_b = tl.sum(_scale * _entropy_b, axis=1)

        lse = tl.log(global_accu) + global_max
        global_logprobs_ptrs = global_logprobs_ptr + offs_m_cur * stride_global_logprobs
        global_logprobs = tl.load(global_logprobs_ptrs, mask=mask_m) - lse
        tl.store(global_max_ptr + offs_m_cur * stride_global_max, global_max, mask=mask_m)
        tl.store(global_entropy_b_ptr + offs_m_cur * stride_global_entropy_b, tl.fdiv(global_entropy_b, global_accu), mask=mask_m)
        tl.store(global_accu_ptr + offs_m_cur * stride_global_accu, global_accu, mask=mask_m)
        tl.store(global_entropy_ptr + offs_m_cur * stride_global_entropy, lse - tl.fdiv(global_entropy_b, global_accu), mask=mask_m)

        # update logprobs
        if reduction == 0:
            tl.store(global_logprobs_ptrs, global_logprobs, mask=mask_m)
        elif reduction == 1:
            global_logprobs_scalar = tl.sum(global_logprobs, axis=0)
            tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)
        elif reduction == 2:
            global_logprobs_scalar = 1.0 / num_tokens * tl.sum(global_logprobs, axis=0)
            tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)


def epilogue_tp_block_sizes(num_splits):
    if num_splits <= 128:
        return 64, 128
    if num_splits <= 256:
        return 64, 256
    return 32, 512


@triton.jit
def efficient_entropy_triton_kernel_epilogue_tp(
    num_tokens,
    num_splits,
    reduced_max_ptr,
    stride_reduced_max_m: tl.int64,
    stride_reduced_max_n: tl.int64,
    original_max_ptr,
    stride_original_max_m: tl.int64,
    stride_original_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_max_ptr,
    stride_global_max: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    core_num_tokens = tl.cdiv(num_tokens, num_jobs)
    idx_token = core_num_tokens * pid_m
    limit = tl.minimum(idx_token + core_num_tokens, num_tokens)
    offs_m = idx_token + tl.arange(0, BLOCK_SIZE_M)
    cnt = tl.cdiv(core_num_tokens, BLOCK_SIZE_M)

    for i in tl.range(cnt):
        offs_m_cur = i * BLOCK_SIZE_M + offs_m
        mask_m = offs_m_cur < limit

        global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_n = mask_m[:, None] & (offs_n[None, :] < num_splits)

            _reduced_max = tl.load(
                reduced_max_ptr + offs_m_cur[:, None] * stride_reduced_max_m + offs_n[None, :] * stride_reduced_max_n,
                mask=mask_n,
            )
            _original_max = tl.load(
                original_max_ptr + offs_m_cur[:, None] * stride_original_max_m + offs_n[None, :] * stride_original_max_n,
                mask=mask_n,
            )
            _accu = tl.load(
                accu_ptr + offs_m_cur[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
                mask=mask_n,
            )
            _entropy_b = tl.load(
                entropy_b_ptr + offs_m_cur[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
                mask=mask_n,
            )

            _max_old = global_max
            _local_max = tl.max(_reduced_max, axis=1)
            global_max = tl.maximum(global_max, _local_max)

            _coeff = tl.exp(_max_old - global_max)
            _scale = tl.exp(_original_max - global_max[:, None])
            _scaled_accu = _scale * _accu
            _scaled_entropy_b = _scale * _entropy_b
            global_accu = _coeff * global_accu + tl.sum(_scaled_accu, axis=1)
            global_entropy_b = _coeff * global_entropy_b + tl.sum(_scaled_entropy_b, axis=1)

        # store
        tl.store(global_max_ptr + offs_m_cur * stride_global_max, global_max, mask=mask_m)
        tl.store(global_accu_ptr + offs_m_cur * stride_global_accu, global_accu, mask=mask_m)
        tl.store(global_entropy_b_ptr + offs_m_cur * stride_global_entropy_b, global_entropy_b, mask=mask_m)


def epilogue_tp_update_config():
    """MLU-friendly autotune configs (shared by the optimized mHC kernels).

    The MLU triton backend silently resets num_warps>4 -> 4 and num_warps==2 -> 1,
    so the effective warp set is {1, 4}; we only emit those. Larger BLOCK_SIZE_C
    amortizes the small H_pre / H_post / H_res reloads across the c-tiles of an
    m row and improves memory coalescing.
    """
    warps = [1]
    stages =[2]
    
    configs = []
    for w, s in itertools.product(warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M":1024}, num_warps=w, num_stages=s)
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1":
        configs = configs[:1]
    return configs


@triton.autotune(configs=epilogue_tp_update_config(), key=[])
@triton.jit
def efficient_entropy_triton_epilogue_tp_update(
    num_tokens,
    logprobs_ptr,
    stride_logprobs: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accumulate_ptr,
    stride_accumulate: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    entropy_ptr,
    stride_entropy: tl.int64,
    logprobs_scalar_ptr,
    reduction: int,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    core_num_tokens = tl.cdiv(num_tokens, num_jobs)
    idx_token = core_num_tokens * pid_m
    limit = tl.minimum(idx_token + core_num_tokens, num_tokens)
    offs_m = idx_token + tl.arange(0, BLOCK_SIZE_M)
    cnt = tl.cdiv(core_num_tokens, BLOCK_SIZE_M)
    for i in tl.range(cnt):
        offs_m_cur = i * BLOCK_SIZE_M + offs_m
        mask_m = offs_m_cur < limit
        maximum = tl.load(maximum_ptr + offs_m_cur * stride_maximum, mask=mask_m)
        accumulate = tl.load(accumulate_ptr + offs_m_cur * stride_accumulate, mask=mask_m, other=1)

        entropy_b = tl.load(entropy_b_ptr + offs_m_cur * stride_entropy_b, mask=mask_m)
        entropy_b = tl.fdiv(entropy_b, accumulate)
        tl.store(entropy_b_ptr + offs_m_cur * stride_entropy_b, entropy_b, mask=mask_m)

        tmp = tl.log(accumulate) + maximum
        entropy = tmp - entropy_b
        tl.store(entropy_ptr + offs_m_cur * stride_entropy, entropy, mask=mask_m)

        logprobs = tl.load(logprobs_ptr + offs_m_cur * stride_logprobs, mask=mask_m, other=0)
        logprobs = logprobs - tmp

        if reduction == 0:
            tl.store(logprobs_ptr + offs_m_cur * stride_logprobs, logprobs, mask=mask_m)
        elif reduction == 1:
            logprobs_scalar = tl.sum(logprobs, axis=0)
            tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
        elif reduction == 2:
            logprobs_scalar = 1 / num_tokens.to(tl.float32) * tl.sum(logprobs, axis=0)
            tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)


_dedicated_stream, _dedicated_events = None, None


def efficient_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: typing.Optional[int] = 2,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
) -> list[torch.Tensor]:
    """
    forward host function
    """
    assert _is_on_accelerator(hidden) and _is_on_accelerator(weight) and _is_on_accelerator(labels)
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if dist_process_group is not None and not hasattr(efficient_entropy_forward, "_initialized"):
        global _dedicated_stream, _dedicated_events
        _dedicated_stream = get_torch_device().Stream(hidden.device)
        _dedicated_events = [get_torch_device().Event() for _ in range(2)]
        efficient_entropy_forward._initialized = True

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    if REDUCTION == EntropyReductionEnum._None:
        if dist_process_group is None:
            logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        else:
            logprobs = torch.zeros((num_tokens,), device=hidden.device, dtype=torch.float32)
    elif REDUCTION in (EntropyReductionEnum._Sum, EntropyReductionEnum._Mean):
        logprobs = torch.empty((), device=hidden.device, dtype=torch.float32)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    entropy = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    assert logprobs.is_contiguous() and entropy.is_contiguous()

    maximum = torch.empty_like(entropy)
    accumulate_and_entropy_b = torch.empty((num_tokens * 2,), device=hidden.device, dtype=torch.float32)
    accumulate_and_entropy_b_view = accumulate_and_entropy_b.view(2, num_tokens)
    accumulate = accumulate_and_entropy_b_view[0, :]
    entropy_b = accumulate_and_entropy_b_view[1, :]
    assert maximum.is_contiguous() and accumulate.is_contiguous() and entropy_b.is_contiguous()

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _entropy_b = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)

    if REDUCTION == EntropyReductionEnum._None:
        _logprobs = logprobs
    else:
        _logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)

    assert _accu.is_contiguous() and _entropy_b.is_contiguous() and _max.is_contiguous()
    assert _is_on_accelerator(_accu) and _is_on_accelerator(_entropy_b) and _is_on_accelerator(_max)

    if _config._use_triton:
        def mainloop_grid(meta):
            return (TOTAL_CORE_NUM,)

        efficient_entropy_kernel_general_mainloop[mainloop_grid](
            _rank,
            hidden,
            weight,
            labels,
            num_tokens,
            hidden_size,
            vocab_size,
            vocab_per_split,
            hidden.stride(0),
            hidden.stride(1),
            weight.stride(0),
            weight.stride(1),
            _max,
            _max.stride(0),
            _max.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            _logprobs,
            _logprobs.stride(0),
            logprobs,
            1.0 / temperature,
            USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
        )
    else:
        raise AssertionError("Triton is required for efficient entropy kernel")

    epilogue_BM, epilogue_BN = epilogue_block_sizes(num_splits)

    def epilogue_grid(meta):
        return (min(TOTAL_CORE_NUM, triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"])),)

    if dist_process_group is None:
        efficient_entropy_triton_kernel_epilogue[epilogue_grid](
            _max,
            _max.stride(0),
            _max.stride(1),
            num_tokens,
            num_splits,
            maximum,
            maximum.stride(0),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            accumulate,
            accumulate.stride(0),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            entropy_b,
            entropy_b.stride(0),
            entropy,
            entropy.stride(0),
            _logprobs,
            _logprobs.stride(0),
            logprobs,
            REDUCTION,
            epilogue_BM,
            epilogue_BN,
            num_warps=1,
            num_stages=2,
        )
    else:
        epilogue_tp_BM, epilogue_tp_BN = epilogue_tp_block_sizes(num_splits)
        _max_backup = _max.clone()
        dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=dist_process_group)

        get_torch_device().current_stream().record_event(_dedicated_events[0])
        with get_torch_device().stream(_dedicated_stream):
            _dedicated_stream.wait_event(_dedicated_events[0])
            dist.all_reduce(_logprobs, op=dist.ReduceOp.SUM, group=dist_process_group)
            _dedicated_stream.record_event(_dedicated_events[1])

        efficient_entropy_triton_kernel_epilogue_tp[epilogue_grid](
            num_tokens,
            num_splits,
            _max,
            _max.stride(0),
            _max.stride(1),
            _max_backup,
            _max_backup.stride(0),
            _max_backup.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            entropy_b,
            entropy_b.stride(0),
            epilogue_tp_BM,
            epilogue_tp_BN,
            num_warps=1,
            num_stages=1,
        )
        get_torch_device().current_stream().wait_event(_dedicated_events[1])

        dist.all_reduce(accumulate_and_entropy_b, op=dist.ReduceOp.SUM, group=dist_process_group)

        # update logprobs & entropy
        efficient_entropy_triton_epilogue_tp_update[epilogue_grid](
            num_tokens,
            _logprobs,
            _logprobs.stride(0),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            entropy_b,
            entropy_b.stride(0),
            entropy,
            entropy.stride(0),
            logprobs,
            REDUCTION,
        )

    return (logprobs, entropy, maximum, accumulate, entropy_b)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 16},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_mainloop_MN(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    backward mainloop, where d_logits & d_hidden & d_weight are fused.

    Each program owns TWO adjacent vocab tiles (effective vocab width
    ``2*BLOCK_SIZE_N``) and shares one ``hidden`` GEMM stream and one set of
    per-token reduction vectors across both, so ``hidden`` (the dominant
    traffic, streamed in both the forward logits K-loop and the backward
    d_hidden/d_weight K-loop) is re-streamed once per vocab-pair rather than
    once per vocab tile. See the autotune comment for the rationale.
    """
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, 2 * BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_tiles = num_pid_m * num_pid_n
    core_num_tiles = tl.cdiv(total_tiles, num_jobs)
    tile_start = core_num_tiles * pid
    cnt = tl.maximum(tl.minimum(core_num_tiles, total_tiles - tile_start), 0)

    if USE_TMA:
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    for i in tl.range(cnt):
        tile_idx = tile_start + i
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
        pid_n = (tile_idx % num_pid_in_group) // group_size_m

        start_offs_am = pid_m * BLOCK_SIZE_M
        offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
        offs_n0 = (2 * pid_n) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_n1 = (2 * pid_n + 1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
        accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)  # epsilon to avoid division by zero
        accu_rcp = tl.fdiv(1.0, accu)

        d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
        if reduction == 0:  # none
            d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
        elif reduction == 1:  # sum
            d_logprobs = tl.load(d_logprobs_ptr)
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        else:  # mean
            d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        d_logprobs = -1 * d_logprobs

        entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
        labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

        if not USE_TMA:
            hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            if USE_TMA:
                start_offs_k = k * BLOCK_SIZE_K
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight0 = weight_desc.load([offs_n0[0], start_offs_k])
                _weight1 = weight_desc.load([offs_n1[0], start_offs_k])
            else:
                _hidden = tl.load(
                    hidden_ptrs + k * BLOCK_SIZE_K * stride_hidden_k,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight0 = tl.load(
                    weight_ptrs0 + k * BLOCK_SIZE_K * stride_weight_k,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
                    other=0.0,
                )
                _weight1 = tl.load(
                    weight_ptrs1 + k * BLOCK_SIZE_K * stride_weight_k,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
                    other=0.0,
                )
            logits0 = tl.dot(_hidden, _weight0.T, logits0)
            logits1 = tl.dot(_hidden, _weight1.T, logits1)

        logits0 *= rcp_temperature
        logits1 *= rcp_temperature

        exp_logits0 = tl.exp(logits0 - maximum[:, None])
        exp_logits1 = tl.exp(logits1 - maximum[:, None])

        mask0 = (offs_n0 + rank * vocab_size)[None, :] == labels[:, None]
        mask1 = (offs_n1 + rank * vocab_size)[None, :] == labels[:, None]
        d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
        d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
        d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
        d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

        d_logits0 *= rcp_temperature
        d_logits1 *= rcp_temperature

        if not USE_TMA:
            d_hidden_ptrs = d_hidden_ptr + offs_am[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
            d_weight_ptrs0 = d_weight_ptr + offs_n0[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k
            d_weight_ptrs1 = d_weight_ptr + offs_n1[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k

        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            start_offs_k = k * BLOCK_SIZE_K
            if USE_TMA:
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            else:
                _hidden = tl.load(
                    hidden_ptrs + k * BLOCK_SIZE_K * stride_hidden_k,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
            _d_weight0 = tl.dot(d_logits0.trans(), _hidden.to(tl.float32))
            tl.atomic_add(
                d_weight_ptrs0 + k * BLOCK_SIZE_K * stride_d_weight_k,
                _d_weight0,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
            )
            _d_weight1 = tl.dot(d_logits1.trans(), _hidden.to(tl.float32))
            tl.atomic_add(
                d_weight_ptrs1 + k * BLOCK_SIZE_K * stride_d_weight_k,
                _d_weight1,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
            )

            if USE_TMA:
                _weight0 = weight_desc.load([offs_n0[0], start_offs_k])
                _weight1 = weight_desc.load([offs_n1[0], start_offs_k])
            else:
                _weight0 = tl.load(
                    weight_ptrs0 + k * BLOCK_SIZE_K * stride_weight_k,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
                    other=0.0,
                )
                _weight1 = tl.load(
                    weight_ptrs1 + k * BLOCK_SIZE_K * stride_weight_k,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
                    other=0.0,
                )
            _d_hidden0 = tl.dot(d_logits0, _weight0.to(tl.float32))
            _d_hidden1 = tl.dot(d_logits1, _weight1.to(tl.float32))
            tl.atomic_add(
                d_hidden_ptrs + k * BLOCK_SIZE_K * stride_d_hidden_k,
                _d_hidden0 + _d_hidden1,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
            )
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_d_hidden(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(hidden_size, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    total_tiles = num_pid_m * num_pid_k
    core_num_tiles = tl.cdiv(total_tiles, num_jobs)
    tile_start = core_num_tiles * pid
    cnt = tl.maximum(tl.minimum(core_num_tiles, total_tiles - tile_start), 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for i in tl.range(cnt):
        tile_idx = tile_start + i
        # L2-cache swizzling decode (same grouping as the GEMM mainloop)
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
        pid_k = (tile_idx % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        result_offs_k = pid_k * BLOCK_SIZE_K + offs_k

        maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens, other=0.0)
        accu = tl.load(accu_ptr + offs_m * stride_accu, mask=offs_m < num_tokens, other=1e-6)
        accu_rcp = tl.fdiv(1.0, accu)
        d_entropy = tl.load(d_entropy_ptr + offs_m * stride_d_entropy, mask=offs_m < num_tokens, other=0.0)
        if reduction == 0:
            d_logprobs = tl.load(d_logprobs_ptr + offs_m * stride_d_logprobs, mask=offs_m < num_tokens, other=0.0)
        elif reduction == 1:
            d_logprobs = tl.load(d_logprobs_ptr)
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        else:
            d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        d_logprobs = -1 * d_logprobs

        entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens, other=0.0)
        labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=0)

        d_hidden = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for n in range(0, tl.cdiv(vocab_size, 2 * BLOCK_SIZE_N)):
            offs_n0 = (2 * n) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_n1 = (2 * n + 1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

            # iterate over hidden_size to get logits -- one hidden stream feeds both tiles
            logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_m[:, None] < num_tokens),
                    other=0.0,
                )
                _weight0 = tl.load(
                    weight_ptrs0,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
                    other=0.0,
                )
                _weight1 = tl.load(
                    weight_ptrs1,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
                    other=0.0,
                )

                logits0 = tl.dot(_hidden, _weight0.trans(), logits0)
                logits1 = tl.dot(_hidden, _weight1.trans(), logits1)

                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs0 += BLOCK_SIZE_K * stride_weight_k
                weight_ptrs1 += BLOCK_SIZE_K * stride_weight_k

            # scale logits by temperature
            logits0 *= rcp_temperature
            logits1 *= rcp_temperature

            exp_logits0 = tl.exp(logits0 - maximum[:, None])
            exp_logits1 = tl.exp(logits1 - maximum[:, None])

            mask0 = (offs_n0 + rank * vocab_size)[None, :] == labels[:, None]
            mask1 = (offs_n1 + rank * vocab_size)[None, :] == labels[:, None]
            d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
            d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
            d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
            d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

            # scale d_logits
            d_logits0 *= rcp_temperature
            d_logits1 *= rcp_temperature

            # calculate d_hidden -- both tiles contribute to the same (M,K) accumulator
            weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + result_offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + result_offs_k[None, :] * stride_weight_k)
            _weight0 = tl.load(
                weight_ptrs0, mask=(result_offs_k[None, :] < hidden_size) & (offs_n0[:, None] < vocab_size), other=0.0
            )
            _weight1 = tl.load(
                weight_ptrs1, mask=(result_offs_k[None, :] < hidden_size) & (offs_n1[:, None] < vocab_size), other=0.0
            )
            d_hidden = tl.dot(d_logits0.to(weight_ptr.dtype.element_ty), _weight0, d_hidden)
            d_hidden = tl.dot(d_logits1.to(weight_ptr.dtype.element_ty), _weight1, d_hidden)

        # write back
        tl.store(
            d_hidden_ptr + offs_m[:, None] * stride_d_hidden_m + result_offs_k[None, :] * stride_d_hidden_k,
            d_hidden,
            mask=(offs_m[:, None] < num_tokens) & (result_offs_k[None, :] < hidden_size),
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_d_hidden_mouter(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(vocab_size, 2 * BLOCK_SIZE_N)

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Per-token vectors: loaded ONCE and shared across all vocab tiles.
    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_m * stride_accu, mask=offs_m < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(d_entropy_ptr + offs_m * stride_d_entropy, mask=offs_m < num_tokens, other=0.0)
    if reduction == 0:
        d_logprobs = tl.load(d_logprobs_ptr + offs_m * stride_d_logprobs, mask=offs_m < num_tokens, other=0.0)
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=0)

    for pid_n in range(0, num_pid_n):
        offs_n0 = (2 * pid_n) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_n1 = (2 * pid_n + 1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
        weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        # forward logits GEMM: hidden streamed ONCE per pair, two weight tiles
        logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_m[:, None] < num_tokens),
                other=0.0,
            )
            _weight0 = tl.load(
                weight_ptrs0,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
                other=0.0,
            )
            _weight1 = tl.load(
                weight_ptrs1,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
                other=0.0,
            )
            logits0 = tl.dot(_hidden, _weight0.trans(), logits0)
            logits1 = tl.dot(_hidden, _weight1.trans(), logits1)
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs0 += BLOCK_SIZE_K * stride_weight_k
            weight_ptrs1 += BLOCK_SIZE_K * stride_weight_k

        logits0 *= rcp_temperature
        logits1 *= rcp_temperature

        exp_logits0 = tl.exp(logits0 - maximum[:, None])
        exp_logits1 = tl.exp(logits1 - maximum[:, None])

        mask0 = (offs_n0 + rank * vocab_size)[None, :] == labels[:, None]
        mask1 = (offs_n1 + rank * vocab_size)[None, :] == labels[:, None]
        d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
        d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
        d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
        d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

        d_logits0 *= rcp_temperature
        d_logits1 *= rcp_temperature
        d_logits0_b = d_logits0.to(weight_ptr.dtype.element_ty)
        d_logits1_b = d_logits1.to(weight_ptr.dtype.element_ty)

        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            result_offs_k = k * BLOCK_SIZE_K + offs_k
            weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + result_offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + result_offs_k[None, :] * stride_weight_k)
            _weight0 = tl.load(
                weight_ptrs0,
                mask=(result_offs_k[None, :] < hidden_size) & (offs_n0[:, None] < vocab_size),
                other=0.0,
            )
            _weight1 = tl.load(
                weight_ptrs1,
                mask=(result_offs_k[None, :] < hidden_size) & (offs_n1[:, None] < vocab_size),
                other=0.0,
            )
            _d_hidden = tl.dot(d_logits0_b, _weight0) + tl.dot(d_logits1_b, _weight1)
            d_hidden_ptrs = d_hidden_ptr + offs_m[:, None] * stride_d_hidden_m + result_offs_k[None, :] * stride_d_hidden_k
            tl.atomic_add(
                d_hidden_ptrs,
                _d_hidden,
                mask=(offs_m[:, None] < num_tokens) & (result_offs_k[None, :] < hidden_size),
            )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 16},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_d_weight(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """backward d_weight.

    Each program owns TWO adjacent vocab tiles (effective vocab width
    ``2*BLOCK_SIZE_N``) and shares one ``hidden`` GEMM stream across them, which
    amortises the per-tile hidden re-streaming — the kernel's dominant cost.
    Two logits accumulators and two weight streams live in NRAM concurrently, so
    BLOCK_SIZE_N is the per-tile width (64) and the (vocab/2BN x hidden/BK) tile
    space is self-distributed across TOTAL_CORE_NUM cores (one program per core,
    striding over tiles) — same convention as
    ``efficient_entropy_backward_kernel_general_d_logits``. Launching one program
    per tile instead would exceed the MLU per-dimension grid limit (65535): the
    backend scales grid_0 by num_warps when checking the limit, so even ~64k
    tiles become ~256k tasks at num_warps=4.
    """
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    num_pid_n = tl.cdiv(vocab_size, 2 * BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(hidden_size, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    total_tiles = num_pid_n * num_pid_k
    core_num_tiles = tl.cdiv(total_tiles, num_jobs)
    tile_start = core_num_tiles * pid
    cnt = tl.maximum(tl.minimum(core_num_tiles, total_tiles - tile_start), 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for i in tl.range(cnt):
        tile_idx = tile_start + i
        # L2-cache swizzling decode (same grouping as the GEMM mainloop)
        group_id = tile_idx // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_M
        group_size_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_M)
        pid_n = first_pid_n + ((tile_idx % num_pid_in_group) % group_size_n)
        pid_k = (tile_idx % num_pid_in_group) // group_size_n

        offs_n0 = (2 * pid_n) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_n1 = (2 * pid_n + 1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        result_offs_k = pid_k * BLOCK_SIZE_K + offs_k

        d_weight0 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        d_weight1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        for m in range(0, tl.cdiv(num_tokens, BLOCK_SIZE_M)):
            offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

            maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens, other=0.0)
            accu = tl.load(accu_ptr + offs_m * stride_accu, mask=offs_m < num_tokens, other=1e-6)
            accu_rcp = tl.fdiv(1.0, accu)
            d_entropy = tl.load(d_entropy_ptr + offs_m * stride_d_entropy, mask=offs_m < num_tokens, other=0.0)
            if reduction == 0:
                d_logprobs = tl.load(d_logprobs_ptr + offs_m * stride_d_logprobs, mask=offs_m < num_tokens, other=0.0)
            elif reduction == 1:
                d_logprobs = tl.load(d_logprobs_ptr)
                d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
            else:
                d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
                d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
            d_logprobs = -1 * d_logprobs

            entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens, other=0.0)
            labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=0)

            hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

            logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_m[:, None] < num_tokens),
                    other=0.0,
                )
                _weight0 = tl.load(
                    weight_ptrs0,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
                    other=0.0,
                )
                _weight1 = tl.load(
                    weight_ptrs1,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
                    other=0.0,
                )

                logits0 = tl.dot(_hidden, _weight0.trans(), logits0)
                logits1 = tl.dot(_hidden, _weight1.trans(), logits1)

                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs0 += BLOCK_SIZE_K * stride_weight_k
                weight_ptrs1 += BLOCK_SIZE_K * stride_weight_k

            logits0 *= rcp_temperature
            logits1 *= rcp_temperature

            exp_logits0 = tl.exp(logits0 - maximum[:, None])
            exp_logits1 = tl.exp(logits1 - maximum[:, None])

            mask0 = (offs_n0 + rank * vocab_size)[None, :] == labels[:, None]
            mask1 = (offs_n1 + rank * vocab_size)[None, :] == labels[:, None]
            d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
            d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
            d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
            d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

            d_logits0 *= rcp_temperature
            d_logits1 *= rcp_temperature

            hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + result_offs_k[None, :] * stride_hidden_k)
            _hidden = tl.load(
                hidden_ptrs, mask=(result_offs_k[None, :] < hidden_size) & (offs_m[:, None] < num_tokens), other=0.0
            )
            d_weight0 = tl.dot(d_logits0.to(d_weight_ptr.dtype.element_ty).trans(), _hidden, d_weight0)
            d_weight1 = tl.dot(d_logits1.to(d_weight_ptr.dtype.element_ty).trans(), _hidden, d_weight1)

        # write back
        tl.store(
            d_weight_ptr + offs_n0[:, None] * stride_d_weight_n + result_offs_k[None, :] * stride_d_weight_k,
            d_weight0,
            mask=(offs_n0[:, None] < vocab_size) & (result_offs_k[None, :] < hidden_size),
        )
        tl.store(
            d_weight_ptr + offs_n1[:, None] * stride_d_weight_n + result_offs_k[None, :] * stride_d_weight_k,
            d_weight1,
            mask=(offs_n1[:, None] < vocab_size) & (result_offs_k[None, :] < hidden_size),
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_d_weight_mouter(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """backward d_weight — M-outer variant (stream hidden once per token block).

    Each program owns ONE token block and iterates over ALL adjacent vocab-tile
    pairs, so ``hidden[BM, full_hidden]`` and the per-token reduction vectors are
    streamed ONCE per program (shared across every vocab pair) instead of once
    per vocab tile. This is the inverse of ``efficient_entropy_backward_kernel_d_weight``'s
    nest (which iterates M inside vocab-pair x K). It wins big when there are
    enough M-blocks to fill the cores and hidden is large enough that the
    re-stream savings dominate; the host dispatch falls back to the pair kernel
    otherwise. d_weight is reduced across M-blocks via ``tl.atomic_add``.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(vocab_size, 2 * BLOCK_SIZE_N)

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Per-token vectors: loaded ONCE and shared across all vocab tiles.
    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_m * stride_accu, mask=offs_m < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(d_entropy_ptr + offs_m * stride_d_entropy, mask=offs_m < num_tokens, other=0.0)
    if reduction == 0:
        d_logprobs = tl.load(d_logprobs_ptr + offs_m * stride_d_logprobs, mask=offs_m < num_tokens, other=0.0)
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs

    entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=0)

    for pid_n in range(0, num_pid_n):
        offs_n0 = (2 * pid_n) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_n1 = (2 * pid_n + 1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs0 = weight_ptr + (offs_n0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
        weight_ptrs1 = weight_ptr + (offs_n1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        # forward logits GEMM: hidden streamed ONCE, two weight tiles
        logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_m[:, None] < num_tokens),
                other=0.0,
            )
            _weight0 = tl.load(
                weight_ptrs0,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n0[:, None] < vocab_size),
                other=0.0,
            )
            _weight1 = tl.load(
                weight_ptrs1,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_n1[:, None] < vocab_size),
                other=0.0,
            )
            logits0 = tl.dot(_hidden, _weight0.trans(), logits0)
            logits1 = tl.dot(_hidden, _weight1.trans(), logits1)
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs0 += BLOCK_SIZE_K * stride_weight_k
            weight_ptrs1 += BLOCK_SIZE_K * stride_weight_k

        logits0 *= rcp_temperature
        logits1 *= rcp_temperature

        exp_logits0 = tl.exp(logits0 - maximum[:, None])
        exp_logits1 = tl.exp(logits1 - maximum[:, None])

        mask0 = (offs_n0 + rank * vocab_size)[None, :] == labels[:, None]
        mask1 = (offs_n1 + rank * vocab_size)[None, :] == labels[:, None]
        d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
        d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
        d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
        d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

        d_logits0 *= rcp_temperature
        d_logits1 *= rcp_temperature
        d_logits0_b = d_logits0.to(d_weight_ptr.dtype.element_ty)
        d_logits1_b = d_logits1.to(d_weight_ptr.dtype.element_ty)

        # d_weight[n,k] = sum_m d_logits[m,n] * hidden[m,k]: reduce across the
        # M-blocks this and other programs own, so accumulate with atomic_add.
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            result_offs_k = k * BLOCK_SIZE_K + offs_k
            hidden_ptrs = hidden_ptr + (offs_m[:, None] * stride_hidden_m + result_offs_k[None, :] * stride_hidden_k)
            _hidden = tl.load(
                hidden_ptrs,
                mask=(result_offs_k[None, :] < hidden_size) & (offs_m[:, None] < num_tokens),
                other=0.0,
            )
            _d_weight0 = tl.dot(d_logits0_b.trans(), _hidden)
            _d_weight1 = tl.dot(d_logits1_b.trans(), _hidden)
            d_weight_ptrs0 = d_weight_ptr + offs_n0[:, None] * stride_d_weight_n + result_offs_k[None, :] * stride_d_weight_k
            d_weight_ptrs1 = d_weight_ptr + offs_n1[:, None] * stride_d_weight_n + result_offs_k[None, :] * stride_d_weight_k
            tl.atomic_add(
                d_weight_ptrs0,
                _d_weight0,
                mask=(offs_n0[:, None] < vocab_size) & (result_offs_k[None, :] < hidden_size),
            )
            tl.atomic_add(
                d_weight_ptrs1,
                _d_weight1,
                mask=(offs_n1[:, None] < vocab_size) & (result_offs_k[None, :] < hidden_size),
            )


# NOTE: split tile from d_logits' perspective
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 16},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # N tiles are decoded in (2*BLOCK_SIZE_N)-wide pairs; BLOCK_SIZE_N below is
    # the per-tile width and the program writes two adjacent BN-wide stores.
    num_pid_n = tl.cdiv(vocab_size, 2 * BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_tiles = num_pid_m * num_pid_n
    core_num_tiles = tl.cdiv(total_tiles, num_jobs)
    tile_start = core_num_tiles * pid
    cnt = tl.maximum(tl.minimum(core_num_tiles, total_tiles - tile_start), 0)

    if USE_TMA:
        # TMA descriptors describe the whole tensors and are tile-independent,
        # so create them once outside the per-tile loop.
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    for i in tl.range(cnt):
        tile_idx = tile_start + i
        # L2-cache swizzling decode (same grouping as the GEMM mainloop)
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
        pid_n = (tile_idx % num_pid_in_group) // group_size_m

        start_offs_am = pid_m * BLOCK_SIZE_M
        offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
        # two adjacent N tiles, base at the pair start
        start_offs_bn = pid_n * (2 * BLOCK_SIZE_N)
        offs_bn0 = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
        offs_bn1 = start_offs_bn + BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
        accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)  # epsilon to avoid division by zero
        accu_rcp = tl.fdiv(1.0, accu)

        d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
        if reduction == 0:  # none
            d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
        elif reduction == 1:  # sum
            d_logprobs = tl.load(d_logprobs_ptr)
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        else:  # mean
            d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        d_logprobs = -1 * d_logprobs

        entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
        labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

        logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        if not USE_TMA:
            hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            weight_ptrs0 = weight_ptr + (offs_bn0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_bn1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            start_offs_k = k * BLOCK_SIZE_K
            if USE_TMA:
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight0 = weight_desc.load([start_offs_bn, start_offs_k])
                _weight1 = weight_desc.load([start_offs_bn + BLOCK_SIZE_N, start_offs_k])
            else:
                hidden_ptrs_ = BLOCK_SIZE_K * stride_hidden_k * k + hidden_ptrs
                weight_ptrs0_ = BLOCK_SIZE_K * stride_weight_k * k + weight_ptrs0
                weight_ptrs1_ = BLOCK_SIZE_K * stride_weight_k * k + weight_ptrs1
                _hidden = tl.load(
                    hidden_ptrs_,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight0 = tl.load(
                    weight_ptrs0_,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn0[:, None] < vocab_size),
                    other=0.0,
                )
                _weight1 = tl.load(
                    weight_ptrs1_,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn1[:, None] < vocab_size),
                    other=0.0,
                )
            logits0 = tl.dot(_hidden, _weight0.T, logits0)
            logits1 = tl.dot(_hidden, _weight1.T, logits1)

        # scale logits by temperature
        logits0 *= rcp_temperature
        logits1 *= rcp_temperature

        exp_logits0 = tl.exp(logits0 - maximum[:, None])
        exp_logits1 = tl.exp(logits1 - maximum[:, None])

        mask0 = (offs_bn0 + rank * vocab_size)[None, :] == labels[:, None]
        mask1 = (offs_bn1 + rank * vocab_size)[None, :] == labels[:, None]
        d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
        d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
        d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
        d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

        # scale d_logits by temperature
        d_logits0 *= rcp_temperature
        d_logits1 *= rcp_temperature

        # store d_logits (two adjacent N tiles)
        tl.store(
            d_logits_ptr + offs_am[:, None] * stride_d_logits_m + offs_bn0[None, :] * stride_d_logits_n,
            d_logits0,  # will be implicitly converted to d_logits_ptrs.dtype.element_ty
            mask=(offs_am[:, None] < num_tokens) & (offs_bn0[None, :] < vocab_size),
        )
        tl.store(
            d_logits_ptr + offs_am[:, None] * stride_d_logits_m + offs_bn1[None, :] * stride_d_logits_n,
            d_logits1,
            mask=(offs_am[:, None] < num_tokens) & (offs_bn1[None, :] < vocab_size),
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 16},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # N tiles are decoded in (2*BLOCK_SIZE_N)-wide pairs; BN below is the
    # per-tile width and the program writes two adjacent BN-wide stores.
    num_pid_n = tl.cdiv(vocab_per_split, 2 * BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_tiles = num_pid_m * num_pid_n
    core_num_tiles = tl.cdiv(total_tiles, num_jobs)
    tile_start = core_num_tiles * pid
    cnt = tl.maximum(tl.minimum(core_num_tiles, total_tiles - tile_start), 0)

    if USE_TMA:
        # TMA descriptors describe the whole tensors and are tile-independent,
        # so create them once outside the per-tile loop.
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    for i in tl.range(cnt):
        tile_idx = tile_start + i
        # L2-cache swizzling decode (same grouping as the GEMM mainloop)
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
        pid_n = (tile_idx % num_pid_in_group) // group_size_m

        start_offs_am = pid_m * BLOCK_SIZE_M
        offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
        # two adjacent N tiles, base at the pair start
        start_offs_bn = split_idx * vocab_per_split + pid_n * (2 * BLOCK_SIZE_N)
        offs_bn0 = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
        offs_bn1 = start_offs_bn + BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
        accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)
        accu_rcp = tl.fdiv(1.0, accu)
        d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
        if reduction == 0:
            d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
        elif reduction == 1:
            d_logprobs = tl.load(d_logprobs_ptr)
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        else:
            d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
            d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
        d_logprobs = -1 * d_logprobs
        entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
        labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

        logits0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        logits1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        if not USE_TMA:
            hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
            weight_ptrs0 = weight_ptr + (offs_bn0[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            weight_ptrs1 = weight_ptr + (offs_bn1[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
            vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)

        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            if USE_TMA:
                start_offs_k = k * BLOCK_SIZE_K
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight0 = weight_desc.load([start_offs_bn, start_offs_k])
                _weight1 = weight_desc.load([start_offs_bn + BLOCK_SIZE_N, start_offs_k])
            else:
                hidden_ptrs_ = k * BLOCK_SIZE_K * stride_hidden_k + hidden_ptrs
                weight_ptrs0_ = k * BLOCK_SIZE_K * stride_weight_k + weight_ptrs0
                weight_ptrs1_ = k * BLOCK_SIZE_K * stride_weight_k + weight_ptrs1
                _hidden = tl.load(
                    hidden_ptrs_,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight0 = tl.load(
                    weight_ptrs0_,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn0[:, None] < vocab_right_bound),
                    other=0.0,
                )
                _weight1 = tl.load(
                    weight_ptrs1_,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn1[:, None] < vocab_right_bound),
                    other=0.0,
                )
            logits0 = tl.dot(_hidden, _weight0.T, logits0)
            logits1 = tl.dot(_hidden, _weight1.T, logits1)

        logits0 *= rcp_temperature
        logits1 *= rcp_temperature
        exp_logits0 = tl.exp(logits0 - maximum[:, None])
        exp_logits1 = tl.exp(logits1 - maximum[:, None])

        mask0 = (offs_bn0 + rank * vocab_size)[None, :] == labels[:, None]
        mask1 = (offs_bn1 + rank * vocab_size)[None, :] == labels[:, None]
        d_logits0 = d_logprobs[:, None] * (exp_logits0 * accu_rcp[:, None] - mask0)
        d_logits0 += d_entropy[:, None] * (-exp_logits0 * accu_rcp[:, None]) * (logits0 - entropy_b[:, None])
        d_logits1 = d_logprobs[:, None] * (exp_logits1 * accu_rcp[:, None] - mask1)
        d_logits1 += d_entropy[:, None] * (-exp_logits1 * accu_rcp[:, None]) * (logits1 - entropy_b[:, None])

        d_logits0 *= rcp_temperature
        d_logits1 *= rcp_temperature

        # filter d_logits with mask and store the two adjacent N tiles
        result_offs_n0 = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
        result_offs_n1 = pid_n * (2 * BLOCK_SIZE_N) + BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask0 = (offs_am[:, None] < num_tokens) & (result_offs_n0[None, :] < vocab_per_split)
        mask1 = (offs_am[:, None] < num_tokens) & (result_offs_n1[None, :] < vocab_per_split)

        tl.store(
            d_logits_ptr + offs_am[:, None] * stride_d_logits_m + result_offs_n0[None, :] * stride_d_logits_n,
            d_logits0, mask0,
        )
        tl.store(
            d_logits_ptr + offs_am[:, None] * stride_d_logits_m + result_offs_n1[None, :] * stride_d_logits_n,
            d_logits1, mask1,
        )


def efficient_entropy_backward(
    dlogprobs: torch.Tensor,
    dentropy: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    acc: torch.Tensor,
    entropy_b: torch.Tensor,
    reduction: typing.Optional[int] = 2,
    should_return_fp32_grad: bool = False,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
) -> list[torch.Tensor]:
    """
    backward host function
    """
    assert _is_on_accelerator(hidden) and _is_on_accelerator(weight) and _is_on_accelerator(labels)
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    if REDUCTION == EntropyReductionEnum._None:
        assert dlogprobs.shape == (num_tokens,)
    else:
        assert dlogprobs.dim() == 0

    assert dlogprobs.is_contiguous() and dentropy.is_contiguous()
    assert _is_on_accelerator(dlogprobs) and _is_on_accelerator(dentropy)
    assert dlogprobs.device == hidden.device and dlogprobs.device == dentropy.device
    assert dentropy.shape == (num_tokens,)

    d_hidden, d_weight = None, None
    if _config._backward == BackwardEnum._Total_Fuse_MN or should_return_fp32_grad:
        d_hidden = torch.zeros_like(hidden, dtype=torch.float32, device=hidden.device)
        d_weight = torch.zeros_like(weight, dtype=torch.float32, device=weight.device)
    else:
        d_hidden = torch.empty_like(hidden, dtype=hidden.dtype, device=hidden.device)
        d_weight = torch.empty_like(weight, dtype=hidden.dtype, device=weight.device)
    assert d_hidden.is_contiguous() and d_weight.is_contiguous()

    assert maximum.is_contiguous() and acc.is_contiguous()
    assert maximum.device == hidden.device and acc.device == hidden.device
    assert maximum.shape == labels.shape == acc.shape
    assert _is_on_accelerator(maximum) and _is_on_accelerator(acc)

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    assert entropy_b.is_contiguous() and _is_on_accelerator(entropy_b)
    assert entropy_b.shape == (num_tokens,)

    if _config._backward == BackwardEnum._Total_Fuse_MN:
        def mainloop_grid(meta):
            return (TOTAL_CORE_NUM,)

        efficient_entropy_backward_kernel_general_mainloop_MN[mainloop_grid](
            num_tokens,
            hidden_size,
            vocab_size,
            _rank,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs,
            dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
            REDUCTION,
            entropy_b,
            entropy_b.stride(0),
            d_hidden,
            d_hidden.stride(0),
            d_hidden.stride(1),
            d_weight,
            d_weight.stride(0),
            d_weight.stride(1),
            1.0 / temperature,
            USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
        )

    elif _config._backward == BackwardEnum._Total_Separate:
        _d_logits = torch.empty((num_tokens, vocab_size), device=hidden.device, dtype=hidden.dtype).contiguous()
        assert _d_logits.is_contiguous()

        if _config._use_triton:
            d_logits_grid = (TOTAL_CORE_NUM,)

            efficient_entropy_backward_kernel_general_d_logits[d_logits_grid](
                num_tokens,
                hidden_size,
                vocab_size,
                _rank,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                labels,
                labels.stride(0),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs,
                dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
                REDUCTION,
                entropy_b,
                entropy_b.stride(0),
                _d_logits,
                _d_logits.stride(0),
                _d_logits.stride(1),
                1.0 / temperature,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
            )

            torch.matmul(_d_logits, weight, out=d_hidden)
            torch.matmul(_d_logits.T, hidden, out=d_weight)
        else:
            raise AssertionError("Triton is required for efficient entropy kernel")

    elif _config._backward == BackwardEnum._Split_Dlogits_N:
        vocab_per_split = 9504
        num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

        _d_logits = torch.empty((num_tokens, vocab_per_split), device=hidden.device, dtype=hidden.dtype).contiguous()
        assert _d_logits.is_contiguous()

        d_logits_grid = (TOTAL_CORE_NUM,)

        for split_idx in range(num_splits):
            efficient_entropy_backward_kernel_general_d_logits_split_N[d_logits_grid](
                split_idx,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                _rank,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                labels,
                labels.stride(0),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs,
                dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
                REDUCTION,
                entropy_b,
                entropy_b.stride(0),
                _d_logits,
                _d_logits.stride(0),
                _d_logits.stride(1),
                1.0 / temperature,
                USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
            )

            if split_idx == (num_splits - 1):
                vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size) - split_idx * vocab_per_split
                _d_logits = _d_logits[:, :vocab_right_bound].contiguous()

            if split_idx == 0:
                torch.matmul(
                    _d_logits, weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :], out=d_hidden
                )
            else:
                d_hidden += torch.matmul(
                    _d_logits, weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :]
                )
            torch.matmul(
                _d_logits.T, hidden, out=d_weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :]
            )

    elif _config._backward == BackwardEnum._Split_Dlogits_M:
        raise NotImplementedError("BackwardEnum._Split_Dlogits_M is not implemented yet")

    return d_hidden, d_weight


# M-outer d_weight kernel launch grid: one program per token block (M-tile).
_D_WEIGHT_MOUTER_BM = 256


def _d_weight_use_mouter(num_tokens: int, hidden_size: int) -> bool:
    num_pid_m = (num_tokens + _D_WEIGHT_MOUTER_BM - 1) // _D_WEIGHT_MOUTER_BM
    return num_pid_m >= 6 and hidden_size >= 1536


def launch_efficient_entropy_backward_kernel_d_weight(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    acc: torch.Tensor,
    dentropy: torch.Tensor,
    dlogprobs: torch.Tensor,
    reduction: int,
    entropy_b: torch.Tensor,
    d_weight: torch.Tensor,
    rcp_temperature: float,
):
    if _d_weight_use_mouter(num_tokens, hidden_size):
        grid = ((num_tokens + _D_WEIGHT_MOUTER_BM - 1) // _D_WEIGHT_MOUTER_BM,)
        d_weight.zero_()
        efficient_entropy_backward_kernel_d_weight_mouter[grid](
            num_tokens, hidden_size, vocab_size, rank,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum, maximum.stride(0),
            acc, acc.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, dlogprobs.stride(0) if dlogprobs.dim() > 0 else 0,
            reduction,
            entropy_b, entropy_b.stride(0),
            d_weight, d_weight.stride(0), d_weight.stride(1),
            rcp_temperature,
        )
    else:
        grid = (TOTAL_CORE_NUM,)
        efficient_entropy_backward_kernel_d_weight[grid](
            num_tokens, hidden_size, vocab_size, rank,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum, maximum.stride(0),
            acc, acc.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, dlogprobs.stride(0) if dlogprobs.dim() > 0 else 0,
            reduction,
            entropy_b, entropy_b.stride(0),
            d_weight, d_weight.stride(0), d_weight.stride(1),
            rcp_temperature,
        )


_D_HIDDEN_MOUTER_BM = 64


def _d_hidden_use_mouter(num_tokens: int, hidden_size: int) -> bool:
    num_pid_m = (num_tokens + _D_HIDDEN_MOUTER_BM - 1) // _D_HIDDEN_MOUTER_BM
    return num_pid_m >= 2 and hidden_size >= 128


def launch_efficient_entropy_backward_kernel_d_hidden(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    acc: torch.Tensor,
    dentropy: torch.Tensor,
    dlogprobs: torch.Tensor,
    reduction: int,
    entropy_b: torch.Tensor,
    d_hidden: torch.Tensor,
    rcp_temperature: float,
):
    if _d_hidden_use_mouter(num_tokens, hidden_size):
        grid = ((num_tokens + _D_HIDDEN_MOUTER_BM - 1) // _D_HIDDEN_MOUTER_BM,)
        #     # The M-outer kernel reduces d_hidden across vocab pairs with ``tl.atomic_add``
        #     # (vs the pair kernel's disjoint writes). Each launch must therefore start
        #     # from a zeroed d_hidden so the atomics accumulate exactly one full gradient
        #     # rather than piling onto whatever the caller left in the buffer. zero_() is a
        #     # single device memset — negligible vs the 100-1100 ms kernel, and it makes
        #     # repeated launches (warmup / do_bench) correctness-neutral, matching the pair
        #     # kernel's overwrite-on-every-launch contract.
        d_hidden.zero_()
        efficient_entropy_backward_kernel_d_hidden_mouter[grid](
            num_tokens, hidden_size, vocab_size, rank,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum, maximum.stride(0),
            acc, acc.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, dlogprobs.stride(0) if dlogprobs.dim() > 0 else 0,
            reduction,
            entropy_b, entropy_b.stride(0),
            d_hidden, d_hidden.stride(0), d_hidden.stride(1),
                rcp_temperature,
            )
    else:
        grid = (TOTAL_CORE_NUM,)
        efficient_entropy_backward_kernel_d_hidden[grid](
            num_tokens, hidden_size, vocab_size, rank,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum, maximum.stride(0),
            acc, acc.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, dlogprobs.stride(0) if dlogprobs.dim() > 0 else 0,
            reduction,
            entropy_b, entropy_b.stride(0),
            d_hidden, d_hidden.stride(0), d_hidden.stride(1),
            rcp_temperature,
        )
