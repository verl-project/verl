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
"""CANN vocab-parallel fused linear cross-entropy for Ascend NPU.

Uses (newer ``npu_*`` names preferred; older aliases also accepted):

  - ``torch_npu.npu_fused_linear_online_max_sum`` / ``fused_linear_online_max_sum``
  - ``torch_npu.npu_fused_cross_entropy_loss_with_max_sum`` / ``fused_cross_entropy_loss_with_max_sum``
  - ``..._backward`` / ``..._grad`` / ``fused_linear_cross_entropy_loss_with_max_sum_grad``

Policy (verl) — hybrid mid-term path:
  - Forward always returns ``(logprobs, entropy)``, matching the CUDA fused API.
  - ``dentropy == 0``: CANN CE-only backward.
  - ``dentropy != 0``: chunked PyTorch CE+entropy backward (Triton ``d_logits``).
  - Chunk size: ``VERL_NPU_LCE_CHUNK_SIZE`` (default 2048).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_LOGGED_CANN_DISPATCH = False

# (role, candidate attribute names on torch_npu). Prefer renamed npu_* APIs first.
_CANN_API_CANDIDATES: dict[str, tuple[str, ...]] = {
    "online_max_sum": (
        "npu_fused_linear_online_max_sum",
        "fused_linear_online_max_sum",
    ),
    "ce_with_max_sum": (
        "npu_fused_cross_entropy_loss_with_max_sum",
        "fused_cross_entropy_loss_with_max_sum",
    ),
    "backward": (
        "npu_fused_linear_cross_entropy_loss_with_max_sum_backward",
        "npu_fused_linear_cross_entropy_loss_with_max_sum_grad",
        "fused_linear_cross_entropy_loss_with_max_sum_grad",
        "fused_linear_cross_entropy_loss_with_max_sum_backward",
    ),
}


def _resolve_cann_api(torch_npu_mod, role: str) -> Optional[Callable]:
    for name in _CANN_API_CANDIDATES[role]:
        fn = getattr(torch_npu_mod, name, None)
        if callable(fn):
            return fn
    return None


def _resolve_cann_apis():
    """Return ``(online_max_sum, ce_with_max_sum, backward)`` or ``None`` if incomplete."""
    try:
        import torch_npu
    except ImportError:
        return None
    online = _resolve_cann_api(torch_npu, "online_max_sum")
    ce = _resolve_cann_api(torch_npu, "ce_with_max_sum")
    bwd = _resolve_cann_api(torch_npu, "backward")
    if online is None or ce is None or bwd is None:
        return None
    return online, ce, bwd


def is_cann_linear_ce_available() -> bool:
    """Return True if torch_npu exposes the full CANN fused linear-CE API set."""
    return _resolve_cann_apis() is not None


def _missing_cann_api_hint() -> str:
    try:
        import torch_npu
    except ImportError:
        return "torch_npu is not installed"
    missing = []
    for role, names in _CANN_API_CANDIDATES.items():
        if _resolve_cann_api(torch_npu, role) is None:
            missing.append(f"{role} (tried: {', '.join(names)})")
    return "; ".join(missing) if missing else "unknown"


def should_use_cann_linear_ce(device: torch.device) -> bool:
    """Whether ``linear_cross_entropy`` should dispatch to CANN on this device.

    ``VERL_NPU_LCE_BACKEND``:
      - ``auto`` (default): use CANN when APIs exist
      - ``cann``: require CANN APIs (raise if missing)
      - ``triton``: do not use CANN (caller falls back / errors on NPU)
    """
    if device.type != "npu":
        return False
    backend = os.environ.get("VERL_NPU_LCE_BACKEND", "auto").lower()
    if backend == "triton":
        return False
    if backend == "cann":
        if not is_cann_linear_ce_available():
            raise RuntimeError(
                "VERL_NPU_LCE_BACKEND=cann but required torch_npu fused linear-CE APIs are missing: "
                + _missing_cann_api_hint()
            )
        return True
    return is_cann_linear_ce_available()


def _resolve_vocab_range(
    weight: torch.Tensor,
    dist_process_group: Optional[dist.ProcessGroup],
) -> tuple[int, int]:
    """Return ``[vocab_start, vocab_end)`` — end is exclusive (Megatron / Ascend style).

    Ascend docs/examples use ``vocab_end = start + weight.size(0)`` (e.g. start=0,
    end=64 for a 64-row shard), not ``start + size - 1``.
    """
    vocab_local = weight.shape[0]
    if dist_process_group is None:
        return 0, vocab_local
    tp_rank = dist.get_rank(dist_process_group)
    vocab_start = tp_rank * vocab_local
    return vocab_start, vocab_start + vocab_local


def _sync_online_stats_for_tp(
    logits_max: torch.Tensor,
    sum_exp_logits: torch.Tensor,
    predicted_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    dist_process_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_max_local = logits_max.clone()
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=dist_process_group)

    max_delta = logits_max_local - logits_max
    sum_exp_logits.mul_(torch.exp(max_delta))
    # Half-open range [start, end), matching vocab_end from _resolve_vocab_range.
    owned = (labels >= vocab_start) & (labels < vocab_end)
    predicted_logits = torch.where(owned, predicted_logits + max_delta, predicted_logits)

    dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=dist_process_group)
    dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=dist_process_group)
    return logits_max, sum_exp_logits, predicted_logits


def _entropy_from_local_softmax(
    softmax: torch.Tensor,
    dist_process_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    log_p = torch.log(softmax.clamp_min(1e-12))
    entropy = -(softmax * log_p).sum(dim=-1)
    if dist_process_group is not None:
        dist.all_reduce(entropy, op=dist.ReduceOp.SUM, group=dist_process_group)
    return entropy


def _lce_chunk_size() -> int:
    """Vocab chunk size for hybrid CE+entropy PyTorch backward."""
    raw = os.environ.get("VERL_NPU_LCE_CHUNK_SIZE", "2048")
    try:
        size = int(raw)
    except ValueError as e:
        raise ValueError(f"VERL_NPU_LCE_CHUNK_SIZE must be an int, got {raw!r}") from e
    if size < 1:
        raise ValueError(f"VERL_NPU_LCE_CHUNK_SIZE must be >= 1, got {size}")
    return size


def _expand_dlogprobs_for_tokens(
    dlogprobs: torch.Tensor,
    reduction: str,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Map upstream ``dlogprobs`` to per-token grads, then negate like Triton."""
    if reduction == "none":
        return (-dlogprobs.reshape(-1)).to(dtype=torch.float32)
    scale = 1.0 if reduction == "sum" else (1.0 / max(num_tokens, 1))
    return torch.full(
        (num_tokens,),
        float(-dlogprobs.reshape(()).item() * scale),
        device=device,
        dtype=torch.float32,
    )


def _chunked_ce_entropy_backward(
    dlogprobs: torch.Tensor,
    dentropy: Optional[torch.Tensor],
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    accumulate: torch.Tensor,
    temperature: float,
    reduction: str,
    vocab_start: int,
    dist_process_group: Optional[dist.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked CE+entropy backward aligned with Triton ``efficient_entropy_backward``.

    Expects ``hidden`` already scaled by ``1/temperature`` (same as CANN forward).
    Returns grads w.r.t. the original (unscaled) hidden and weight.
    """
    if chunk_size is None:
        chunk_size = _lce_chunk_size()

    num_tokens = hidden.shape[0]
    vocab_local = weight.shape[0]
    device = hidden.device

    d_lp = _expand_dlogprobs_for_tokens(dlogprobs, reduction, num_tokens, device)
    if dentropy is None:
        d_ent = torch.zeros(num_tokens, device=device, dtype=torch.float32)
    else:
        d_ent = dentropy.reshape(-1).to(dtype=torch.float32)

    maximum = maximum.reshape(-1).to(dtype=torch.float32)
    accumulate = accumulate.reshape(-1).to(dtype=torch.float32).clamp_min(1e-12)
    accu_rcp = torch.reciprocal(accumulate)
    labels_i64 = labels.reshape(-1).to(dtype=torch.int64)

    hidden_f = hidden.to(dtype=torch.float32)
    weight_f = weight.to(dtype=torch.float32)

    # Pass 1: entropy_b = E_p[logits]
    entropy_b_num = torch.zeros(num_tokens, device=device, dtype=torch.float32)
    for start in range(0, vocab_local, chunk_size):
        end = min(start + chunk_size, vocab_local)
        logits = torch.matmul(hidden_f, weight_f[start:end].T)
        exp_logits = torch.exp(logits - maximum[:, None])
        entropy_b_num = entropy_b_num + (exp_logits * logits).sum(dim=-1)

    if dist_process_group is not None:
        dist.all_reduce(entropy_b_num, op=dist.ReduceOp.SUM, group=dist_process_group)
    entropy_b = entropy_b_num * accu_rcp

    # Pass 2: d_logits -> d_hidden / d_weight
    d_hidden = torch.zeros_like(hidden_f)
    d_weight = torch.zeros_like(weight_f)
    for start in range(0, vocab_local, chunk_size):
        end = min(start + chunk_size, vocab_local)
        w_chunk = weight_f[start:end]
        logits = torch.matmul(hidden_f, w_chunk.T)
        exp_logits = torch.exp(logits - maximum[:, None])
        softmax = exp_logits * accu_rcp[:, None]

        global_ids = torch.arange(
            vocab_start + start,
            vocab_start + end,
            device=device,
            dtype=labels_i64.dtype,
        )
        mask_f = (global_ids[None, :] == labels_i64[:, None]).to(dtype=torch.float32)

        d_logits = d_lp[:, None] * (softmax - mask_f)
        d_logits = d_logits + d_ent[:, None] * (-softmax) * (logits - entropy_b[:, None])

        d_hidden = d_hidden + torch.matmul(d_logits, w_chunk)
        d_weight[start:end] = torch.matmul(d_logits.T, hidden_f)

    if temperature != 1.0:
        d_hidden = d_hidden * (1.0 / temperature)

    return d_hidden, d_weight


class CannLinearCrossEntropy(torch.autograd.Function):
    """Fused linear + CE via CANN; chunked CE+H backward when dentropy != 0."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
        reduction: str = "none",
        dist_process_group: Optional[dist.ProcessGroup] = None,
    ):
        apis = _resolve_cann_apis()
        if apis is None:
            raise RuntimeError("CANN fused linear-CE APIs are unavailable: " + _missing_cann_api_hint())
        online_max_sum, ce_with_max_sum, _backward_fn = apis

        global _LOGGED_CANN_DISPATCH
        if not _LOGGED_CANN_DISPATCH:
            logger.info(
                "linear_cross_entropy: using CANN fused vocab-parallel CE on NPU (online=%s, ce=%s)",
                getattr(online_max_sum, "__name__", online_max_sum),
                getattr(ce_with_max_sum, "__name__", ce_with_max_sum),
            )
            _LOGGED_CANN_DISPATCH = True

        assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
        assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]
        assert temperature > 0.0
        reduction = reduction.lower()
        assert reduction in ("none", "sum", "mean")

        # Ascend fused CE is more reliable in the memory-saving path
        # (return_logits=False + backward via logits_max/sum_exp_logits).
        # Materializing vocab logits (return_logits=True) has been observed to
        # corrupt a small fraction of per-token losses on some torch_npu builds.
        # Opt in with VERL_NPU_LCE_RETURN_LOGITS=1 when metric entropy is needed.
        return_logits = os.environ.get("VERL_NPU_LCE_RETURN_LOGITS", "0").lower() in (
            "1",
            "true",
            "yes",
        )

        if hidden.dtype not in (torch.float16, torch.bfloat16):
            hidden_in = hidden.to(torch.bfloat16)
            weight_in = weight.to(torch.bfloat16)
        else:
            hidden_in = hidden
            weight_in = weight
        hidden_in = hidden_in.contiguous()
        weight_in = weight_in.contiguous()

        if temperature != 1.0:
            hidden_in = hidden_in * (1.0 / temperature)

        vocab_start, vocab_end = _resolve_vocab_range(weight_in, dist_process_group)
        # Ascend ops commonly expect int32 targets.
        labels_i = labels.to(torch.int32).contiguous()

        (
            logits_max,
            sum_exp_logits,
            predicted_logits,
            target_mask,
            masked_target,
            vocab_parallel_logits,
        ) = online_max_sum(
            hidden_in,
            weight_in,
            labels_i,
            vocab_start,
            vocab_end,
            return_logits,
        )

        if dist_process_group is not None:
            logits_max, sum_exp_logits, predicted_logits = _sync_online_stats_for_tp(
                logits_max,
                sum_exp_logits,
                predicted_logits,
                labels_i,
                vocab_start,
                vocab_end,
                dist_process_group,
            )

        ce_kwargs = {"label_smoothing": 0.0}
        if (
            return_logits
            and vocab_parallel_logits is not None
            and isinstance(vocab_parallel_logits, torch.Tensor)
            and vocab_parallel_logits.numel() > 0
        ):
            ce_kwargs["vocab_parallel_logits"] = vocab_parallel_logits

        loss, softmax = ce_with_max_sum(
            logits_max,
            sum_exp_logits,
            predicted_logits,
            **ce_kwargs,
        )

        logprobs = -loss
        if reduction == "sum":
            logprobs = logprobs.sum()
        elif reduction == "mean":
            logprobs = logprobs.mean()
        elif reduction == "none":
            # Ensure a flat [num_tokens] tensor for Megatron THD postprocess.
            logprobs = logprobs.reshape(-1).contiguous()

        if softmax is None or (isinstance(softmax, torch.Tensor) and softmax.numel() == 0):
            entropy = torch.zeros(hidden_in.shape[0], device=hidden_in.device, dtype=torch.float32)
        else:
            entropy = _entropy_from_local_softmax(softmax, dist_process_group)
        if reduction == "none":
            entropy = entropy.reshape(-1).contiguous()

        # Always keep labels + online stats for the hybrid CE+H path.
        # Prefer memory-saving CANN backward (logits_max/sum_exp). Fall back to
        # softmax when return_logits=True and softmax is present.
        use_softmax_bwd = (
            return_logits and softmax is not None and isinstance(softmax, torch.Tensor) and softmax.numel() > 0
        )
        if use_softmax_bwd:
            ctx.save_for_backward(
                hidden_in,
                weight_in,
                labels_i,
                target_mask,
                masked_target,
                logits_max,
                sum_exp_logits,
                softmax,
            )
            ctx.bwd_mode = "softmax"
        else:
            ctx.save_for_backward(
                hidden_in,
                weight_in,
                labels_i,
                target_mask,
                masked_target,
                logits_max,
                sum_exp_logits,
            )
            ctx.bwd_mode = "max_sum"
        ctx.temperature = float(temperature)
        ctx.reduction = reduction
        ctx.vocab_start = int(vocab_start)
        ctx.dist_process_group = dist_process_group
        ctx.original_hidden_dtype = hidden.dtype
        ctx.original_weight_dtype = weight.dtype
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor):
        temperature = ctx.temperature
        reduction = ctx.reduction
        bwd_mode = getattr(ctx, "bwd_mode", "max_sum")
        vocab_start = getattr(ctx, "vocab_start", 0)
        dist_process_group = getattr(ctx, "dist_process_group", None)

        if bwd_mode == "softmax":
            (
                hidden_in,
                weight_in,
                labels_i,
                target_mask,
                masked_target,
                logits_max,
                sum_exp_logits,
                softmax,
            ) = ctx.saved_tensors
        else:
            (
                hidden_in,
                weight_in,
                labels_i,
                target_mask,
                masked_target,
                logits_max,
                sum_exp_logits,
            ) = ctx.saved_tensors
            softmax = None

        if dentropy is not None and torch.any(dentropy != 0):
            d_hidden, d_weight = _chunked_ce_entropy_backward(
                dlogprobs,
                dentropy,
                hidden_in,
                weight_in,
                labels_i,
                logits_max,
                sum_exp_logits,
                temperature,
                reduction,
                vocab_start,
                dist_process_group,
            )
            return (
                d_hidden.to(ctx.original_hidden_dtype),
                d_weight.to(ctx.original_weight_dtype),
                None,
                None,
                None,
                None,
            )

        apis = _resolve_cann_apis()
        if apis is None:
            raise RuntimeError("CANN fused linear-CE APIs are unavailable: " + _missing_cann_api_hint())
        _online, _ce, backward_fn = apis

        if reduction == "none":
            d_loss = -dlogprobs
        else:
            num_tokens = hidden_in.shape[0]
            scale = 1.0 if reduction == "sum" else (1.0 / max(num_tokens, 1))
            d_loss = torch.full(
                (num_tokens,),
                float(-dlogprobs.item() * scale),
                device=hidden_in.device,
                dtype=torch.float32,
            )

        d_hidden, d_weight = backward_fn(
            d_loss,
            hidden_in,
            weight_in,
            target_mask,
            masked_target,
            0.0,
            logits_max,
            sum_exp_logits,
            softmax,
        )

        if temperature != 1.0:
            d_hidden = d_hidden * (1.0 / temperature)

        return (
            d_hidden.to(ctx.original_hidden_dtype),
            d_weight.to(ctx.original_weight_dtype),
            None,
            None,
            None,
            None,
        )
