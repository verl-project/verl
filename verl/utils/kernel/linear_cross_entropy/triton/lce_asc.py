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
"""
Ascend NPU Triton implementation of fused linear cross entropy with token entropy.
"""

import typing
from dataclasses import dataclass

import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl
    from triton.runtime import driver as _triton_driver

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    _triton_driver = None

if not HAVE_TRITON:
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    @contextmanager
    def null_decorator(*args, **kwargs):
        if len(kwargs) == 0 and args and callable(args[0]):
            return args[0]

        def inner(func):
            return func

        return inner

    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    tl = MagicMock()


@dataclass
class EntropyReductionEnum:
    """Enum for the reduction method of cross entropy."""

    _None = 0
    _Sum = 1
    _Mean = 2


def get_entropy_reduction_enum_number(reduction: str) -> int:
    """Get the enum number for the reduction method of cross entropy."""
    if reduction == "none":
        return EntropyReductionEnum._None
    if reduction == "sum":
        return EntropyReductionEnum._Sum
    if reduction == "mean":
        return EntropyReductionEnum._Mean
    raise ValueError(f"Invalid reduction: {reduction}")


def get_entropy_reduction_enum(ce_reduction: int) -> int:
    """Validate and return the enum value for the reduction method of cross entropy."""
    if ce_reduction == 0:
        return EntropyReductionEnum._None
    if ce_reduction == 1:
        return EntropyReductionEnum._Sum
    if ce_reduction == 2:
        return EntropyReductionEnum._Mean
    raise ValueError(f"Invalid ce_reduction: {ce_reduction}")


@dataclass
class BackwardEnum:
    """Enum for the backward method."""

    _Total_Fuse_MN = 0  # Fuse d_logits & d_hidden & d_weight via atomic_add
    _Total_Separate = 1  # Materialize full d_logits, then matmul for d_hidden & d_weight
    _Split_Dlogits_N = 2  # Split d_logits along vocab_size (N) dimension
    _Split_Dlogits_M = 3  # Split d_logits along num_tokens (M) dimension


@dataclass
class Config:
    """Configuration for Ascend Triton linear cross entropy kernels."""

    _backward: int = BackwardEnum._Split_Dlogits_N
    _use_triton: bool = True


_config = Config()


def set_backward_method(backward_method: int):
    """Set the backward method."""
    global _config
    _config._backward = backward_method


def _ensure_ascend_triton_driver() -> None:
    """Select the Ascend NPU Triton driver when transfer_to_npu shims torch.cuda."""
    if _triton_driver is None or _triton_driver._active is not None:
        return
    try:
        from triton.backends import backends

        ascend_backend = backends.get("ascend")
        if ascend_backend is not None and ascend_backend.driver.is_active():
            _triton_driver.set_active(ascend_backend.driver())
    except Exception:
        pass


def _get_npu_core_counts(device: torch.device) -> tuple[int, int]:
    """Return (num_cube_cores, num_vector_cores) for the target Ascend NPU."""
    _ensure_ascend_triton_driver()
    dev_idx = device.index if device.index is not None else torch.npu.current_device()
    props = torch.npu.get_device_properties(dev_idx)
    return props.cube_core_num, props.vector_core_num


@triton.jit
def efficient_entropy_target_logits_kernel(
    hidden_ptr,
    weight_ptr,
    labels_ptr,
    target_logits_ptr,
    num_tokens,
    hidden_size,
    vocab_size,
    topk,
    vocab_start: tl.int64,
    stride_hidden_m: tl.int64,
    stride_hidden_k,
    stride_weight_n: tl.int64,
    stride_weight_k,
    stride_labels_m: tl.int64,
    stride_labels_k,
    stride_target_m: tl.int64,
    stride_target_k,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute target logits for (token, topk) queries on Ascend vector cores."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    total_jobs = num_tokens * topk
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)

    for job in range(pid, total_jobs, num_programs):
        row = job // topk
        col = job % topk
        label = tl.load(labels_ptr + row.to(tl.int64) * stride_labels_m + col * stride_labels_k).to(tl.int64)
        local_label = label - vocab_start

        if (local_label >= 0) & (local_label < vocab_size):
            acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
            h_row_ptr = hidden_ptr + row.to(tl.int64) * stride_hidden_m
            w_row_ptr = weight_ptr + local_label * stride_weight_n
            for k_start in range(0, hidden_size, BLOCK_SIZE_K):
                offs_k = k_start + offs_k_base
                k_mask = offs_k < hidden_size
                h_val = tl.load(h_row_ptr + offs_k * stride_hidden_k, mask=k_mask, other=0.0).to(tl.float32)
                w_val = tl.load(w_row_ptr + offs_k * stride_weight_k, mask=k_mask, other=0.0).to(tl.float32)
                acc += h_val * w_val
            logit = tl.sum(acc, axis=0) * rcp_temperature
            tl.store(target_logits_ptr + row.to(tl.int64) * stride_target_m + col * stride_target_k, logit)
        else:
            tl.store(target_logits_ptr + row.to(tl.int64) * stride_target_m + col * stride_target_k, 0.0)


@triton.jit
def efficient_entropy_kernel_general_mainloop(
    hidden_ptr,
    weight_ptr,
    num_tokens,
    hidden_size,
    vocab_size,
    stride_hidden_m: tl.int64,
    stride_hidden_k,
    stride_weight_n: tl.int64,
    stride_weight_k,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """Forward mainloop computing tile-local online softmax and weighted-logit statistics."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_tiles, num_programs):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_am < num_tokens
        offs_m_i64 = offs_am.to(tl.int64)
        hidden_ptrs_base = hidden_ptr + (offs_m_i64[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_bn < vocab_size
        weight_ptrs_base = weight_ptr + (
            offs_bn[:, None].to(tl.int64) * stride_weight_n + offs_k[None, :] * stride_weight_k
        )

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            k_rem = hidden_size - k * BLOCK_SIZE_K
            k_mask = offs_k[None, :] < k_rem
            k_offset = k * BLOCK_SIZE_K
            _hidden = tl.load(
                hidden_ptrs_base + k_offset * stride_hidden_k,
                mask=mask_m[:, None] & k_mask,
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs_base + k_offset * stride_weight_k,
                mask=mask_n[:, None] & k_mask,
                other=0.0,
            )
            logits = tl.dot(_hidden, tl.trans(_weight), logits)

        logits = logits * rcp_temperature
        logits_for_lse = tl.where(mask_n[None, :], logits, -float("inf"))
        _max = tl.max(logits_for_lse, axis=1)
        exp_logits = tl.exp(logits_for_lse - _max[:, None])
        _accu = tl.sum(exp_logits, axis=1)
        _entropy_b = tl.sum(logits * exp_logits, axis=1)

        out_mask = mask_m & (pid_n < num_pid_n)
        tl.store(max_ptr + offs_m_i64 * stride_max_m + pid_n * stride_max_n, _max, mask=out_mask)
        tl.store(accu_ptr + offs_m_i64 * stride_accu_m + pid_n * stride_accu_n, _accu, mask=out_mask)
        tl.store(
            entropy_b_ptr + offs_m_i64 * stride_entropy_b_m + pid_n * stride_entropy_b_n,
            _entropy_b,
            mask=out_mask,
        )


@triton.jit
def efficient_entropy_local_reduce_kernel(
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n,
    num_tokens,
    num_splits,
    local_max_ptr,
    stride_local_max,
    local_accu_ptr,
    stride_local_accu,
    local_entropy_b_ptr,
    stride_local_entropy_b,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Merge vocabulary-split partial statistics on one rank."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(num_tokens, BLOCK_SIZE_M)

    for tile_m in range(pid, num_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < num_tokens
        offs_m_i64 = offs_m.to(tl.int64)

        rank_max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
        rank_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        rank_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for split_blk in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
            offs_n = split_blk * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_mn = mask_m[:, None] & (offs_n[None, :] < num_splits)

            _max = tl.load(
                max_ptr + offs_m_i64[:, None] * stride_max_m + offs_n[None, :] * stride_max_n,
                mask=mask_mn,
                other=-float("inf"),
            )
            _accu = tl.load(
                accu_ptr + offs_m_i64[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
                mask=mask_mn,
                other=0.0,
            )
            _entropy_b = tl.load(
                entropy_b_ptr + offs_m_i64[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
                mask=mask_mn,
                other=0.0,
            )

            _max_old = rank_max
            _local_max = tl.max(_max, axis=1)
            rank_max = tl.maximum(rank_max, _local_max)
            safe_max = tl.where(mask_m, rank_max, 0.0)

            _scale = tl.exp(_max - safe_max[:, None])
            _coeff = tl.exp(tl.where(mask_m, _max_old - safe_max, 0.0))
            rank_accu = _coeff * rank_accu + tl.sum(_scale * _accu, axis=1)
            rank_entropy_b = _coeff * rank_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

        tl.store(local_max_ptr + offs_m_i64 * stride_local_max, rank_max, mask=mask_m)
        tl.store(local_accu_ptr + offs_m_i64 * stride_local_accu, rank_accu, mask=mask_m)
        tl.store(local_entropy_b_ptr + offs_m_i64 * stride_local_entropy_b, rank_entropy_b, mask=mask_m)


def linear_cross_entropy_fwd_local_reduce(num_splits: int, BLOCK_M: int = 16, BLOCK_S: int = 64):
    """Return a callable merging [M, num_splits] partial softmax/entropy statistics."""

    def _runner(
        partial_max: torch.Tensor,
        partial_sum: torch.Tensor,
        partial_weighted_sum: torch.Tensor,
        local_max: torch.Tensor,
        local_sum: torch.Tensor,
        local_weighted_sum: torch.Tensor,
    ):
        num_tokens = partial_max.shape[0]
        _, num_vec_cores = _get_npu_core_counts(partial_max.device)
        num_tiles = triton.cdiv(num_tokens, BLOCK_M)
        grid = (min(num_tiles, num_vec_cores),)
        efficient_entropy_local_reduce_kernel[grid](
            partial_max,
            partial_max.stride(0),
            partial_max.stride(1),
            partial_sum,
            partial_sum.stride(0),
            partial_sum.stride(1),
            partial_weighted_sum,
            partial_weighted_sum.stride(0),
            partial_weighted_sum.stride(1),
            num_tokens,
            num_splits,
            local_max,
            local_max.stride(0),
            local_sum,
            local_sum.stride(0),
            local_weighted_sum,
            local_weighted_sum.stride(0),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_S,
        )

    return _runner


@triton.jit
def efficient_entropy_triton_kernel_epilogue(
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n,
    num_tokens,
    num_splits,
    global_max_ptr,
    stride_global_max,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n,
    global_accu_ptr,
    stride_global_accu,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n,
    global_entropy_b_ptr,
    stride_global_entropy_b,
    global_entropy_ptr,
    stride_global_entropy,
    target_logits_ptr,
    stride_target_m: tl.int64,
    stride_target_k,
    logprobs_ptr,
    stride_logprobs_m: tl.int64,
    stride_logprobs_k,
    topk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Single-rank forward epilogue reducing splits and writing logprobs & entropy."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(num_tokens, BLOCK_SIZE_M)

    for tile_m in range(pid, num_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < num_tokens
        offs_m_i64 = offs_m.to(tl.int64)

        global_max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
        global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for split_blk in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
            offs_n = split_blk * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_mn = mask_m[:, None] & (offs_n[None, :] < num_splits)

            _max = tl.load(
                max_ptr + offs_m_i64[:, None] * stride_max_m + offs_n[None, :] * stride_max_n,
                mask=mask_mn,
                other=-float("inf"),
            )
            _accu = tl.load(
                accu_ptr + offs_m_i64[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
                mask=mask_mn,
                other=0.0,
            )
            _entropy_b = tl.load(
                entropy_b_ptr + offs_m_i64[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
                mask=mask_mn,
                other=0.0,
            )

            _max_old = global_max
            _local_max = tl.max(_max, axis=1)
            global_max = tl.maximum(global_max, _local_max)
            safe_max = tl.where(mask_m, global_max, 0.0)

            _scale = tl.exp(_max - safe_max[:, None])
            _coeff = tl.exp(tl.where(mask_m, _max_old - safe_max, 0.0))
            global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
            global_entropy_b = _coeff * global_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

        safe_accu = tl.where(mask_m, global_accu, 1.0)
        lse = global_max + tl.log(safe_accu)
        mean_logits = global_entropy_b / safe_accu
        entropy = lse - mean_logits

        tl.store(global_max_ptr + offs_m_i64 * stride_global_max, global_max, mask=mask_m)
        tl.store(global_accu_ptr + offs_m_i64 * stride_global_accu, global_accu, mask=mask_m)
        tl.store(global_entropy_b_ptr + offs_m_i64 * stride_global_entropy_b, mean_logits, mask=mask_m)
        tl.store(global_entropy_ptr + offs_m_i64 * stride_global_entropy, entropy, mask=mask_m)

        for q in range(0, topk):
            t_logit = tl.load(
                target_logits_ptr + offs_m_i64 * stride_target_m + q * stride_target_k,
                mask=mask_m,
                other=0.0,
            )
            tl.store(
                logprobs_ptr + offs_m_i64 * stride_logprobs_m + q * stride_logprobs_k,
                t_logit - lse,
                mask=mask_m,
            )


@triton.jit
def efficient_entropy_tp_prepare_reduce_kernel(
    local_max_ptr,
    global_max_ptr,
    local_accu_ptr,
    local_entropy_b_ptr,
    target_logits_ptr,
    stride_target_m: tl.int64,
    stride_target_k,
    reduced_ptr,
    stride_reduced_m: tl.int64,
    stride_reduced_k,
    num_tokens,
    topk,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Rescale rank-local statistics to global_max and pack with target_logits for TP all_reduce."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(num_tokens, BLOCK_SIZE_M)

    for tile_m in range(pid, num_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < num_tokens
        offs_m_i64 = offs_m.to(tl.int64)

        l_max = tl.load(local_max_ptr + offs_m_i64, mask=mask_m, other=0.0)
        g_max = tl.load(global_max_ptr + offs_m_i64, mask=mask_m, other=0.0)
        l_accu = tl.load(local_accu_ptr + offs_m_i64, mask=mask_m, other=0.0)
        l_ent_b = tl.load(local_entropy_b_ptr + offs_m_i64, mask=mask_m, other=0.0)

        scale = tl.exp(l_max - g_max)
        row_out = reduced_ptr + offs_m_i64 * stride_reduced_m
        tl.store(row_out + 0 * stride_reduced_k, l_accu * scale, mask=mask_m)
        tl.store(row_out + 1 * stride_reduced_k, l_ent_b * scale, mask=mask_m)

        for q in range(0, topk):
            t_logit = tl.load(
                target_logits_ptr + offs_m_i64 * stride_target_m + q * stride_target_k,
                mask=mask_m,
                other=0.0,
            )
            tl.store(row_out + (q + 2) * stride_reduced_k, t_logit, mask=mask_m)


@triton.jit
def efficient_entropy_tp_finalize_kernel(
    global_max_ptr,
    reduced_ptr,
    stride_reduced_m: tl.int64,
    stride_reduced_k,
    accumulate_ptr,
    entropy_b_ptr,
    entropy_ptr,
    logprobs_ptr,
    stride_logprobs_m: tl.int64,
    stride_logprobs_k,
    num_tokens,
    topk,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Unpack TP-reduced statistics and produce global logprobs, entropy, accumulate, entropy_b."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(num_tokens, BLOCK_SIZE_M)

    for tile_m in range(pid, num_tiles, num_programs):
        offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < num_tokens
        offs_m_i64 = offs_m.to(tl.int64)

        g_max = tl.load(global_max_ptr + offs_m_i64, mask=mask_m, other=0.0)
        row_in = reduced_ptr + offs_m_i64 * stride_reduced_m
        g_accu = tl.load(row_in + 0 * stride_reduced_k, mask=mask_m, other=1.0)
        g_ent_b_sum = tl.load(row_in + 1 * stride_reduced_k, mask=mask_m, other=0.0)

        safe_accu = tl.where(mask_m, g_accu, 1.0)
        lse = g_max + tl.log(safe_accu)
        mean_logits = g_ent_b_sum / safe_accu
        entropy = lse - mean_logits

        tl.store(accumulate_ptr + offs_m_i64, g_accu, mask=mask_m)
        tl.store(entropy_b_ptr + offs_m_i64, mean_logits, mask=mask_m)
        tl.store(entropy_ptr + offs_m_i64, entropy, mask=mask_m)

        for q in range(0, topk):
            t_logit = tl.load(row_in + (q + 2) * stride_reduced_k, mask=mask_m, other=0.0)
            tl.store(
                logprobs_ptr + offs_m_i64 * stride_logprobs_m + q * stride_logprobs_k,
                t_logit - lse,
                mask=mask_m,
            )


def efficient_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: typing.Optional[int] = 2,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward host function for Ascend NPU Triton linear cross entropy."""
    assert hidden.device.type == "npu" and weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() in (1, 2)
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]
    if not _config._use_triton:
        raise AssertionError("Triton is required for efficient entropy kernel")

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    num_tokens, hidden_size = hidden.shape
    topk = labels.shape[1] if labels.ndim == 2 else 1
    vocab_size = weight.shape[0]
    vocab_start = _rank * vocab_size

    REDUCTION = get_entropy_reduction_enum(reduction)
    if REDUCTION not in (EntropyReductionEnum._None, EntropyReductionEnum._Sum, EntropyReductionEnum._Mean):
        raise ValueError(f"Invalid reduction: {reduction}")

    num_cube_cores, num_vec_cores = _get_npu_core_counts(hidden.device)
    rcp_temperature = 1.0 / float(temperature)

    labels_2d = labels.view(num_tokens, topk)
    target_logits = torch.empty((num_tokens, topk), device=hidden.device, dtype=torch.float32)

    target_block_k = min(1024, max(64, triton.next_power_of_2(min(hidden_size, 1024))))
    total_target_jobs = num_tokens * topk
    efficient_entropy_target_logits_kernel[(min(total_target_jobs, num_vec_cores),)](
        hidden,
        weight,
        labels_2d,
        target_logits,
        num_tokens,
        hidden_size,
        vocab_size,
        topk,
        vocab_start,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        labels_2d.stride(0),
        labels_2d.stride(1),
        target_logits.stride(0),
        target_logits.stride(1),
        rcp_temperature,
        BLOCK_SIZE_K=target_block_k,
    )

    block_m = 64
    block_n = 256
    block_k = 128 if hidden_size >= 128 else 64
    num_splits = triton.cdiv(vocab_size, block_n)

    _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _entropy_b = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)

    total_mainloop_tiles = triton.cdiv(num_tokens, block_m) * num_splits

    efficient_entropy_kernel_general_mainloop[(min(total_mainloop_tiles, num_cube_cores),)](
        hidden,
        weight,
        num_tokens,
        hidden_size,
        vocab_size,
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
        rcp_temperature,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )

    entropy = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    maximum = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    accumulate = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    entropy_b = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    per_token_logprobs = torch.empty((num_tokens, topk), device=hidden.device, dtype=torch.float32)

    epi_block_m = 32
    epi_block_n = 64
    epi_grid = (min(triton.cdiv(num_tokens, epi_block_m), num_vec_cores),)

    if dist_process_group is None:
        efficient_entropy_triton_kernel_epilogue[epi_grid](
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
            target_logits,
            target_logits.stride(0),
            target_logits.stride(1),
            per_token_logprobs,
            per_token_logprobs.stride(0),
            per_token_logprobs.stride(1),
            topk,
            BLOCK_SIZE_M=epi_block_m,
            BLOCK_SIZE_N=epi_block_n,
        )
    else:
        local_max = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        local_accu = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        local_entropy_b = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)

        efficient_entropy_local_reduce_kernel[epi_grid](
            _max,
            _max.stride(0),
            _max.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            num_tokens,
            num_splits,
            local_max,
            local_max.stride(0),
            local_accu,
            local_accu.stride(0),
            local_entropy_b,
            local_entropy_b.stride(0),
            BLOCK_SIZE_M=epi_block_m,
            BLOCK_SIZE_N=epi_block_n,
        )

        maximum.copy_(local_max)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=dist_process_group)

        reduced = torch.empty((num_tokens, topk + 2), device=hidden.device, dtype=torch.float32)
        prep_block_m = 64
        prep_grid = (min(triton.cdiv(num_tokens, prep_block_m), num_vec_cores),)
        efficient_entropy_tp_prepare_reduce_kernel[prep_grid](
            local_max,
            maximum,
            local_accu,
            local_entropy_b,
            target_logits,
            target_logits.stride(0),
            target_logits.stride(1),
            reduced,
            reduced.stride(0),
            reduced.stride(1),
            num_tokens,
            topk,
            BLOCK_SIZE_M=prep_block_m,
        )

        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=dist_process_group)

        efficient_entropy_tp_finalize_kernel[prep_grid](
            maximum,
            reduced,
            reduced.stride(0),
            reduced.stride(1),
            accumulate,
            entropy_b,
            entropy,
            per_token_logprobs,
            per_token_logprobs.stride(0),
            per_token_logprobs.stride(1),
            num_tokens,
            topk,
            BLOCK_SIZE_M=prep_block_m,
        )

    if REDUCTION == EntropyReductionEnum._None:
        logprobs = per_token_logprobs if labels.ndim == 2 else per_token_logprobs.view(labels.shape)
    elif REDUCTION == EntropyReductionEnum._Sum:
        logprobs = per_token_logprobs.sum()
    else:
        logprobs = per_token_logprobs.sum() / float(num_tokens)

    return logprobs, entropy, maximum, accumulate, entropy_b


@triton.jit
def _compute_tile_d_logits(
    logits,
    maximum,
    accu_rcp,
    entropy_b,
    d_entropy,
    labels_ptr,
    stride_labels_m: tl.int64,
    stride_labels_k: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs_m: tl.int64,
    stride_d_logprobs_k: tl.int64,
    d_logprobs_sum_ptr,
    offs_am,
    mask_m,
    offs_bn,
    mask_n,
    vocab_start: tl.int64,
    vocab_size,
    topk: int,
    rcp_temperature: tl.float32,
):
    """Compute [BLOCK_SIZE_M, BLOCK_SIZE_N] d_logits in fp32 for one (M, N) tile."""
    logits = logits * rcp_temperature
    probs = tl.where(mask_n[None, :], tl.exp(logits - maximum[:, None]) * accu_rcp[:, None], 0.0)
    offs_m_i64 = offs_am.to(tl.int64)

    g_sum = tl.load(d_logprobs_sum_ptr + offs_m_i64, mask=mask_m, other=0.0).to(tl.float32)
    d_logits = (-g_sum[:, None] - d_entropy[:, None] * (logits - entropy_b[:, None])) * probs

    vocab_end = vocab_start + vocab_size.to(tl.int64)
    for q in range(0, topk):
        lbl_q = tl.load(
            labels_ptr + offs_m_i64 * stride_labels_m + q * stride_labels_k,
            mask=mask_m,
            other=-1,
        ).to(tl.int64)
        in_shard = mask_m & (lbl_q >= vocab_start) & (lbl_q < vocab_end)
        local_lbl = tl.where(in_shard, (lbl_q - vocab_start).to(tl.int32), -1)
        gq = tl.load(
            d_logprobs_ptr + offs_m_i64 * stride_d_logprobs_m + q * stride_d_logprobs_k,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        match_q = offs_bn[None, :] == local_lbl[:, None]
        d_logits += tl.where(match_q, gq[:, None], 0.0)

    d_logits = tl.where(mask_m[:, None] & mask_n[None, :], d_logits * rcp_temperature, 0.0)
    return d_logits


@triton.jit
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx,
    num_tokens,
    hidden_size,
    vocab_size,
    vocab_per_split,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k,
    maximum_ptr,
    stride_maximum,
    accu_ptr,
    stride_accu,
    d_entropy_ptr,
    stride_d_entropy,
    d_logprobs_sum_ptr,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Recompute one vocabulary chunk of logits and produce its dense fp32 d_logits term."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    split_vocab_offset = split_idx * vocab_per_split
    vocab_right_bound = min(split_vocab_offset + vocab_per_split, vocab_size)
    valid_split_vocab = vocab_right_bound - split_vocab_offset

    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(valid_split_vocab, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_tiles, num_programs):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_am < num_tokens
        offs_m_i64 = offs_am.to(tl.int64)

        result_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_bn = split_vocab_offset + result_offs_n
        mask_n = offs_bn < vocab_right_bound

        maximum = tl.load(maximum_ptr + offs_m_i64 * stride_maximum, mask=mask_m, other=0.0).to(tl.float32)
        accu = tl.load(accu_ptr + offs_m_i64 * stride_accu, mask=mask_m, other=1.0).to(tl.float32)
        accu_rcp = 1.0 / tl.where(mask_m, accu, 1.0)
        d_entropy = tl.load(d_entropy_ptr + offs_m_i64 * stride_d_entropy, mask=mask_m, other=0.0).to(tl.float32)
        entropy_b = tl.load(entropy_b_ptr + offs_m_i64 * stride_entropy_b, mask=mask_m, other=0.0).to(tl.float32)
        g_sum = tl.load(d_logprobs_sum_ptr + offs_m_i64, mask=mask_m, other=0.0).to(tl.float32)

        hidden_ptrs_base = hidden_ptr + (offs_m_i64[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs_base = weight_ptr + (
            offs_bn[:, None].to(tl.int64) * stride_weight_n + offs_k[None, :] * stride_weight_k
        )

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            k_rem = hidden_size - k * BLOCK_SIZE_K
            k_mask = offs_k[None, :] < k_rem
            k_offset = k * BLOCK_SIZE_K
            _hidden = tl.load(
                hidden_ptrs_base + k_offset * stride_hidden_k,
                mask=mask_m[:, None] & k_mask,
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs_base + k_offset * stride_weight_k,
                mask=mask_n[:, None] & k_mask,
                other=0.0,
            )
            logits = tl.dot(_hidden, tl.trans(_weight), logits)

        logits = logits * rcp_temperature
        probs = tl.where(mask_n[None, :], tl.exp(logits - maximum[:, None]) * accu_rcp[:, None], 0.0)
        d_logits = (-g_sum[:, None] - d_entropy[:, None] * (logits - entropy_b[:, None])) * probs
        d_logits = tl.where(mask_m[:, None] & mask_n[None, :], d_logits * rcp_temperature, 0.0)

        store_mask = mask_m[:, None] & (result_offs_n[None, :] < valid_split_vocab)
        tl.store(
            d_logits_ptr + offs_m_i64[:, None] * stride_d_logits_m + result_offs_n[None, :] * stride_d_logits_n,
            d_logits,
            mask=store_mask,
        )


@triton.jit(do_not_specialize=["topk"])
def efficient_entropy_backward_kernel_general_mainloop_MN(
    num_tokens,
    hidden_size,
    vocab_size,
    vocab_start: tl.int64,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k,
    labels_ptr,
    stride_labels_m: tl.int64,
    stride_labels_k: tl.int64,
    maximum_ptr,
    stride_maximum,
    accu_ptr,
    stride_accu,
    d_entropy_ptr,
    stride_d_entropy,
    d_logprobs_ptr,
    stride_d_logprobs_m: tl.int64,
    stride_d_logprobs_k: tl.int64,
    d_logprobs_sum_ptr,
    entropy_b_ptr,
    stride_entropy_b,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k,
    topk: int,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused backward mainloop accumulating d_hidden and d_weight via atomic_add."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(pid, total_tiles, num_programs):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_am < num_tokens
        offs_m_i64 = offs_am.to(tl.int64)

        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_bn < vocab_size
        offs_n_i64 = offs_bn.to(tl.int64)

        maximum = tl.load(maximum_ptr + offs_m_i64 * stride_maximum, mask=mask_m, other=0.0).to(tl.float32)
        accu = tl.load(accu_ptr + offs_m_i64 * stride_accu, mask=mask_m, other=1.0).to(tl.float32)
        accu_rcp = 1.0 / tl.where(mask_m, accu, 1.0)
        d_entropy = tl.load(d_entropy_ptr + offs_m_i64 * stride_d_entropy, mask=mask_m, other=0.0).to(tl.float32)
        entropy_b = tl.load(entropy_b_ptr + offs_m_i64 * stride_entropy_b, mask=mask_m, other=0.0).to(tl.float32)

        hidden_ptrs_base = hidden_ptr + (offs_m_i64[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs_base = weight_ptr + (offs_n_i64[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            k_rem = hidden_size - k * BLOCK_SIZE_K
            k_mask = offs_k[None, :] < k_rem
            k_offset = k * BLOCK_SIZE_K
            _hidden = tl.load(
                hidden_ptrs_base + k_offset * stride_hidden_k,
                mask=mask_m[:, None] & k_mask,
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs_base + k_offset * stride_weight_k,
                mask=mask_n[:, None] & k_mask,
                other=0.0,
            )
            logits = tl.dot(_hidden, tl.trans(_weight), logits)

        d_logits = _compute_tile_d_logits(
            logits,
            maximum,
            accu_rcp,
            entropy_b,
            d_entropy,
            labels_ptr,
            stride_labels_m,
            stride_labels_k,
            d_logprobs_ptr,
            stride_d_logprobs_m,
            stride_d_logprobs_k,
            d_logprobs_sum_ptr,
            offs_am,
            mask_m,
            offs_bn,
            mask_n,
            vocab_start,
            vocab_size,
            topk,
            rcp_temperature,
        )

        d_logits_cast = d_logits.to(hidden_ptr.dtype.element_ty)
        d_logits_t_cast = tl.trans(d_logits_cast)

        d_hidden_ptrs_base = d_hidden_ptr + (
            offs_m_i64[:, None] * stride_d_hidden_m + offs_k[None, :] * stride_d_hidden_k
        )
        d_weight_ptrs_base = d_weight_ptr + (
            offs_n_i64[:, None] * stride_d_weight_n + offs_k[None, :] * stride_d_weight_k
        )

        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            k_rem = hidden_size - k * BLOCK_SIZE_K
            k_mask = offs_k[None, :] < k_rem
            k_offset = k * BLOCK_SIZE_K

            _hidden = tl.load(
                hidden_ptrs_base + k_offset * stride_hidden_k,
                mask=mask_m[:, None] & k_mask,
                other=0.0,
            )
            _d_weight = tl.dot(d_logits_t_cast, _hidden)
            tl.atomic_add(
                d_weight_ptrs_base + k_offset * stride_d_weight_k,
                _d_weight,
                mask=mask_n[:, None] & k_mask,
            )

            _weight = tl.load(
                weight_ptrs_base + k_offset * stride_weight_k,
                mask=mask_n[:, None] & k_mask,
                other=0.0,
            )
            _d_hidden = tl.dot(d_logits_cast, _weight)
            tl.atomic_add(
                d_hidden_ptrs_base + k_offset * stride_d_hidden_k,
                _d_hidden,
                mask=mask_m[:, None] & k_mask,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward host function for Ascend NPU Triton linear cross entropy."""
    assert hidden.device.type == "npu" and weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() in (1, 2)
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]
    if not _config._use_triton:
        raise AssertionError("Triton is required for efficient entropy kernel")

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    num_tokens, hidden_size = hidden.shape
    topk = labels.shape[1] if labels.ndim == 2 else 1
    vocab_size = weight.shape[0]
    vocab_start = _rank * vocab_size

    REDUCTION = get_entropy_reduction_enum(reduction)
    if REDUCTION == EntropyReductionEnum._None:
        assert dlogprobs.shape == labels.shape
    else:
        assert dlogprobs.dim() == 0

    assert dlogprobs.is_contiguous() and dentropy.is_contiguous()
    assert dlogprobs.device == hidden.device and dentropy.device == hidden.device
    assert dentropy.shape == (num_tokens,)
    assert maximum.is_contiguous() and acc.is_contiguous() and entropy_b.is_contiguous()
    assert maximum.device == hidden.device and acc.device == hidden.device and entropy_b.device == hidden.device
    assert maximum.shape == (num_tokens,) == acc.shape == entropy_b.shape

    num_cube_cores, _ = _get_npu_core_counts(hidden.device)
    rcp_temperature = 1.0 / float(temperature)

    labels_2d = labels.view(num_tokens, topk)
    if REDUCTION == EntropyReductionEnum._None:
        dlogprobs_2d = dlogprobs.view(num_tokens, topk)
        if topk == 1:
            dlogprobs_sum = dlogprobs_2d.view(num_tokens).to(torch.float32)
        else:
            dlogprobs_sum = dlogprobs_2d.to(torch.float32).sum(dim=-1)
    elif REDUCTION == EntropyReductionEnum._Sum:
        dlogprobs_sum = dlogprobs.to(torch.float32).expand(num_tokens).contiguous()
        dlogprobs_2d = dlogprobs_sum.view(num_tokens, 1)
    else:
        dlogprobs_sum = (dlogprobs.to(torch.float32) * (1.0 / float(num_tokens))).expand(num_tokens).contiguous()
        dlogprobs_2d = dlogprobs_sum.view(num_tokens, 1)

    stride_dlp_m = dlogprobs_2d.stride(0)
    stride_dlp_k = dlogprobs_2d.stride(1)

    block_m = 64
    block_n = 64
    block_k = 128 if hidden_size >= 128 else 64
    group_size_m = 8

    if _config._backward == BackwardEnum._Total_Fuse_MN:
        d_hidden_fp32 = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=hidden.device)
        d_weight_fp32 = torch.zeros((vocab_size, hidden_size), dtype=torch.float32, device=weight.device)

        total_tiles = triton.cdiv(num_tokens, block_m) * triton.cdiv(vocab_size, block_n)
        efficient_entropy_backward_kernel_general_mainloop_MN[(min(total_tiles, num_cube_cores),)](
            num_tokens,
            hidden_size,
            vocab_size,
            vocab_start,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels_2d,
            labels_2d.stride(0),
            labels_2d.stride(1),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            dentropy,
            dentropy.stride(0),
            dlogprobs_2d,
            stride_dlp_m,
            stride_dlp_k,
            dlogprobs_sum,
            entropy_b,
            entropy_b.stride(0),
            d_hidden_fp32,
            d_hidden_fp32.stride(0),
            d_hidden_fp32.stride(1),
            d_weight_fp32,
            d_weight_fp32.stride(0),
            d_weight_fp32.stride(1),
            topk,
            rcp_temperature,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=group_size_m,
        )
        if should_return_fp32_grad:
            return d_hidden_fp32, d_weight_fp32
        return d_hidden_fp32.to(hidden.dtype), d_weight_fp32.to(weight.dtype)

    if _config._backward in (BackwardEnum._Total_Separate, BackwardEnum._Split_Dlogits_N):
        split_block_n = 128
        if _config._backward == BackwardEnum._Total_Separate:
            vocab_per_split = vocab_size
            num_splits = 1
        else:
            max_vocab_chunk = 8192
            aligned_vocab = triton.cdiv(vocab_size, split_block_n) * split_block_n
            vocab_per_split = min(max_vocab_chunk, aligned_vocab)
            num_splits = triton.cdiv(vocab_size, vocab_per_split)

        _d_logits_fp32 = torch.empty((num_tokens, vocab_per_split), device=hidden.device, dtype=torch.float32)
        _d_logits = (
            torch.empty((num_tokens, vocab_per_split), device=hidden.device, dtype=hidden.dtype)
            if hidden.dtype != torch.float32
            else _d_logits_fp32
        )
        scaled_dlogprobs = dlogprobs_2d.to(torch.float32) * rcp_temperature

        accumulate_fp32 = should_return_fp32_grad or (
            num_splits > 1 and hidden.dtype in (torch.float16, torch.bfloat16)
        )

        if accumulate_fp32:
            d_hidden_accum = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=hidden.device)
            _d_hidden_chunk = (
                torch.empty((num_tokens, hidden_size), dtype=hidden.dtype, device=hidden.device)
                if hidden.dtype != torch.float32
                else None
            )
        else:
            d_hidden = torch.empty(hidden.shape, device=hidden.device, dtype=hidden.dtype)

        d_weight_compute = torch.empty(weight.shape, device=weight.device, dtype=hidden.dtype)

        for split_idx in range(num_splits):
            split_offset = split_idx * vocab_per_split
            valid_vocab = min(vocab_per_split, vocab_size - split_offset)
            total_tiles = triton.cdiv(num_tokens, block_m) * triton.cdiv(valid_vocab, split_block_n)

            efficient_entropy_backward_kernel_general_d_logits_split_N[(min(total_tiles, num_cube_cores),)](
                split_idx,
                num_tokens,
                hidden_size,
                vocab_size,
                vocab_per_split,
                hidden,
                hidden.stride(0),
                hidden.stride(1),
                weight,
                weight.stride(0),
                weight.stride(1),
                maximum,
                maximum.stride(0),
                acc,
                acc.stride(0),
                dentropy,
                dentropy.stride(0),
                dlogprobs_sum,
                entropy_b,
                entropy_b.stride(0),
                _d_logits_fp32,
                _d_logits_fp32.stride(0),
                _d_logits_fp32.stride(1),
                rcp_temperature,
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=split_block_n,
                BLOCK_SIZE_K=block_k,
                GROUP_SIZE_M=group_size_m,
            )

            _d_logits_fp32_valid = _d_logits_fp32 if valid_vocab == vocab_per_split else _d_logits_fp32[:, :valid_vocab]
            local_labels = labels_2d - (vocab_start + split_offset)
            in_split = (local_labels >= 0) & (local_labels < valid_vocab)
            _d_logits_fp32_valid.scatter_add_(
                1,
                local_labels.clamp(0, valid_vocab - 1),
                torch.where(in_split, scaled_dlogprobs, 0.0),
            )

            if hidden.dtype != torch.float32:
                if valid_vocab == vocab_per_split:
                    _d_logits.copy_(_d_logits_fp32)
                    _d_logits_valid = _d_logits
                else:
                    _d_logits_valid = _d_logits_fp32_valid.to(hidden.dtype).contiguous()
            else:
                _d_logits_valid = (
                    _d_logits_fp32 if valid_vocab == vocab_per_split else _d_logits_fp32_valid.contiguous()
                )

            weight_slice = slice(split_offset, split_offset + valid_vocab)

            if accumulate_fp32:
                if _d_hidden_chunk is not None:
                    torch.matmul(_d_logits_valid, weight[weight_slice], out=_d_hidden_chunk)
                    d_hidden_accum.add_(_d_hidden_chunk)
                else:
                    d_hidden_accum.add_(torch.matmul(_d_logits_valid, weight[weight_slice]))
            elif split_idx == 0:
                torch.matmul(_d_logits_valid, weight[weight_slice], out=d_hidden)
            else:
                d_hidden += torch.matmul(_d_logits_valid, weight[weight_slice])

            torch.matmul(_d_logits_valid.T, hidden, out=d_weight_compute[weight_slice])

        if accumulate_fp32:
            d_hidden = d_hidden_accum if should_return_fp32_grad else d_hidden_accum.to(hidden.dtype)
        d_weight = d_weight_compute.to(torch.float32) if should_return_fp32_grad else d_weight_compute.to(weight.dtype)
        return d_hidden, d_weight

    if _config._backward == BackwardEnum._Split_Dlogits_M:
        raise NotImplementedError("BackwardEnum._Split_Dlogits_M is not implemented yet")

    raise ValueError(f"Unsupported backward method: {_config._backward}")
