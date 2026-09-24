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
"""Score centering for training-inference mismatch (https://arxiv.org/abs/2609.20807).

The sampler's top-k head is exact; its tail is modeled as the trainer's tail rescaled to the
sampler's tail mass, so the centering term only needs the k head log-probs of the trainer.
"""

import math
from collections.abc import Callable
from typing import Optional

import torch
from tensordict import TensorDict

from verl.trainer.ppo.rollout_corr_helper import _parse_rollout_is_threshold
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss

TOPK_LOG_PROB_CHUNK_SIZE = 4096


def dummy_rollout_topk(
    num_rows: int, k: int, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a uniform top-k head for rows the loss masks out (e.g. prompt or padding positions).

    Args:
        num_rows: Number of rows to fill.
        k: Head size.
        device: Device for the returned tensors, default None (CPU).

    Returns:
        Tuple containing:
            ids: Token ids 0..k-1 repeated per row, shape (num_rows, k), dtype int32.
            log_probs: Uniform log-probs -log(k), shape (num_rows, k), dtype float32.
    """
    ids = torch.arange(k, dtype=torch.int32, device=device).expand(num_rows, k).clone()
    log_probs = torch.full((num_rows, k), -math.log(k), dtype=torch.float32, device=device)
    return ids, log_probs


def pad_rollout_topk(
    response_topk_ids: torch.Tensor | list,
    response_topk_log_probs: torch.Tensor | list,
    *,
    k: int,
    prompt_width: int,
    response_width: int,
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay the sampler's response-token heads over the padded full sequence.

    Prompts are left-padded to ``prompt_width`` and responses are right-padded to
    ``response_width``, so response token r sits at sequence column ``prompt_width + r``
    and is predicted by the logits row ``prompt_width - 1 + r``; its sampler head goes there.
    Rows outside the response (prompt and padding) get the uniform dummy head.

    Args:
        response_topk_ids: Sampler top-k token ids for the response tokens, shape
            (response_length, k), array-like of ints.
        response_topk_log_probs: Sampler top-k log-probs matching ``response_topk_ids``,
            shape (response_length, k), array-like of floats.
        k: Head size.
        prompt_width: Padded prompt length.
        response_width: Padded response length.
        response_length: Number of real (non-padding) response tokens.

    Returns:
        Tuple containing:
            ids: Full-sequence top-k ids, shape (1, prompt_width + response_width, k), dtype int32.
            log_probs: Full-sequence top-k log-probs matching ``ids``, same shape, dtype float32.
    """
    ids = torch.as_tensor(response_topk_ids, dtype=torch.int32)[:response_length]
    log_probs = torch.as_tensor(response_topk_log_probs, dtype=torch.float32)[:response_length]
    if ids.shape != (response_length, k) or log_probs.shape != (response_length, k):
        raise ValueError(
            f"score centering needs one sampler top-{k} head per response token, "
            f"got {tuple(ids.shape)} for {response_length} tokens"
        )
    full_ids, full_log_probs = dummy_rollout_topk(prompt_width + response_width, k)
    start = prompt_width - 1
    full_ids[start : start + response_length] = ids
    full_log_probs[start : start + response_length] = log_probs
    return full_ids.unsqueeze(0), full_log_probs.unsqueeze(0)


class _TopKLogProbsFromLogits(torch.autograd.Function):
    """log_softmax(logits).gather(ids) with a bounded fp32 workspace in forward and backward.

    Only the original logits, the ids and two scalars per row are saved; the softmax is
    recomputed per chunk in backward.
    """

    @staticmethod
    def forward(ctx, logits, token_ids, chunk_size):
        n = logits.shape[0]
        chunk_size = chunk_size if chunk_size > 0 else max(n, 1)
        output = torch.empty(token_ids.shape, device=logits.device, dtype=torch.float32)
        maxima = torch.empty((n, 1), device=logits.device, dtype=torch.float32)
        denominators = torch.empty_like(maxima)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            maximum = work.max(dim=-1, keepdim=True).values
            targets = work.gather(-1, token_ids[start:end])
            work.sub_(maximum).exp_()
            denominator = work.sum(dim=-1, keepdim=True)
            output[start:end] = (targets - maximum) - denominator.log()
            maxima[start:end] = maximum
            denominators[start:end] = denominator
            del work
        ctx.save_for_backward(logits, token_ids, maxima, denominators)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, token_ids, maxima, denominators = ctx.saved_tensors
        n = logits.shape[0]
        grad_input = torch.empty_like(logits)
        for start in range(0, n, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            work.sub_(maxima[start:end]).exp_().div_(denominators[start:end])
            grad = grad_output[start:end].float()
            work.mul_(-grad.sum(dim=-1, keepdim=True))
            work.scatter_add_(-1, token_ids[start:end], grad)
            grad_input[start:end] = work
            del work
        return grad_input, None, None


def topk_log_probs_from_logits(
    logits: torch.Tensor, token_ids: torch.Tensor, chunk_size: int = TOPK_LOG_PROB_CHUNK_SIZE
) -> torch.Tensor:
    """Compute full-vocab-normalized log-probs of ``logits`` at ``token_ids``, chunked for memory.

    Equivalent to ``torch.log_softmax(logits, dim=-1).gather(-1, token_ids)`` but never
    materializes the full (N, V) softmax at once: the normalizer is recomputed per chunk in
    forward and backward, bounding the extra fp32 workspace to (chunk_size, V).

    Args:
        logits: Trainer logits, shape (N, V), any float dtype.
        token_ids: Token ids to gather, shape (N, k), any integer dtype.
        chunk_size: Number of rows processed per chunk. Default: 4096.

    Returns:
        Log-probs of ``token_ids`` under the full-vocab softmax of ``logits``, shape (N, k),
        dtype float32, with gradient to ``logits``.
    """
    return _TopKLogProbsFromLogits.apply(logits, token_ids.long(), chunk_size)


def score_centering_weight_fn(
    rollout_is: Optional[str], rollout_is_threshold: str | float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the token-level IS rule as a function of a ratio, shared by the head and sampled token.

    Args:
        rollout_is: Importance-sampling mode score centering composes with: None (no IS weight)
            or "token" (TIS or IcePop, depending on ``rollout_is_threshold``).
        rollout_is_threshold: Threshold specification, see ``_parse_rollout_is_threshold``: a
            single float or float-like string upper-clamps (TIS); a "lower_upper" string zeros
            ratios outside the band (IcePop).

    Returns:
        A function mapping a ratio tensor to its weight tensor of the same shape.
    """
    if rollout_is is None:
        return torch.ones_like
    if rollout_is != "token":
        raise ValueError(f"score centering composes with token-level IS only, got rollout_is={rollout_is!r}")
    upper, lower = _parse_rollout_is_threshold(rollout_is_threshold)
    if lower is None:
        return lambda ratio: ratio.clamp(max=upper)
    return lambda ratio: torch.where((ratio >= lower) & (ratio <= upper), ratio, torch.zeros_like(ratio))


def score_centering_correction(
    train_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-token centering term ``sum_H sg[q w - alpha p] log p`` and the head masses.

    The sampler's tail (outside the top-k head H) is modeled as the trainer's tail rescaled by
    ``alpha = rho * weight_fn(1 / rho)`` with ``rho = (1 - q_mass) / (1 - p_mass)``, so the
    correction only needs the k head log-probs of both distributions. When the head covers the
    whole vocabulary, ``rho`` is ill-conditioned and the correction's value is not meaningful;
    only its gradient is (it still matches the exact full-vocabulary centering term).

    Args:
        train_head_log_probs: Trainer's full-vocab-normalized log-probs at the head ids,
            shape (N, k), any float dtype, with gradient to the trainer logits.
        sampler_head_log_probs: Sampler's full-vocab-normalized log-probs at the same head ids,
            shape (N, k), any float dtype.
        weight_fn: IS weight rule applied to the ratio p/q on the head, as returned by
            ``score_centering_weight_fn``.
        eps: Numerical floor for the tail masses in ``rho``. Default: 1e-6.

    Returns:
        Tuple containing:
            correction: Centering term to subtract from the policy-gradient loss, shape (N,),
                dtype float32, with gradient to the trainer logits.
            sampler_head_mass: Sampler's head probability mass, shape (N,), dtype float32.
            train_head_mass: Trainer's head probability mass, shape (N,), dtype float32.
    """
    train_head_log_probs = train_head_log_probs.float()
    with torch.no_grad():
        p = train_head_log_probs.exp()
        q = sampler_head_log_probs.float().exp()
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
        alpha = rho * weight_fn(1.0 / rho)
        head_weights = weight_fn((train_head_log_probs - sampler_head_log_probs.float()).exp())
        residual = q * head_weights - alpha.unsqueeze(-1) * p
    correction = (residual * train_head_log_probs).sum(-1)
    return correction, q_mass, p_mass


def score_centering_logits_processor(
    student_logits: torch.Tensor, data: TensorDict, config: ActorConfig, data_format: str = "thd"
) -> dict[str, torch.Tensor]:
    """Per-token centering term from the trainer logits, in the engine's logits-processor slot.

    Args:
        student_logits: Trainer logits, already temperature-scaled, shape (1, nnz/sp, V).
        data: Micro input batch, holding the nested ``rollout_topk_ids``/``rollout_topk_log_probs``
            sampler heads of shape [B, j, k].
        config: Actor configuration, used for ``policy_loss.rollout_correction``.
        data_format: "thd" or "bshd". Only "thd" is supported.

    Returns:
        Dict with, each of shape (1, nnz/sp):
            sc_correction: Centering term, with gradient to ``student_logits``.
            sc_sampler_head_mass: Sampler's head probability mass, detached.
            sc_train_head_mass: Trainer's head probability mass, detached.
    """
    if data_format != "thd":
        raise NotImplementedError("score centering supports the thd (remove padding) format only.")
    topk_ids = data["rollout_topk_ids"].values().unsqueeze(0)
    topk_log_probs = data["rollout_topk_log_probs"].values().unsqueeze(0)
    if get_ulysses_sequence_parallel_world_size() > 1:
        topk_ids = slice_input_tensor(topk_ids, dim=1)
        topk_log_probs = slice_input_tensor(topk_log_probs, dim=1)
    assert topk_ids.shape[:2] == topk_log_probs.shape[:2] == student_logits.shape[:2], (
        topk_ids.shape,
        topk_log_probs.shape,
        student_logits.shape,
    )
    rollout_correction = config.policy_loss.rollout_correction
    weight_fn = score_centering_weight_fn(
        rollout_correction.get("rollout_is", None), rollout_correction.get("rollout_is_threshold", 2.0)
    )
    train_head_log_probs = topk_log_probs_from_logits(student_logits.squeeze(0), topk_ids.squeeze(0))
    correction, sampler_head_mass, train_head_mass = score_centering_correction(
        train_head_log_probs, topk_log_probs.squeeze(0), weight_fn
    )
    return {
        "sc_correction": correction.unsqueeze(0),
        "sc_sampler_head_mass": sampler_head_mass.unsqueeze(0),
        "sc_train_head_mass": train_head_mass.unsqueeze(0),
    }


def score_centering_ppo_loss(
    config: ActorConfig,
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Actor loss function used both for the logits processor and the final policy loss.
    - student_logits is not None, compute the centering term in the logits processor.
    - student_logits is None, compute the final policy loss.

    Args:
        config: Actor configuration.
        model_output: Model output, including log_probs and sc_correction.
        data: Micro input batch, contains the nested rollout top-k head.
        dp_group: Data parallel group for ``ppo_loss``.
        student_logits: (1, nnz/sp, V).
        data_format: "thd" or "bshd". Only "thd" is supported.

    Returns:
        student_logits is not None: dict from ``score_centering_logits_processor``.
        student_logits is None: the (loss, metrics) tuple from ``ppo_loss``.
    """
    if student_logits is not None:
        return score_centering_logits_processor(student_logits, data, config, data_format)
    return ppo_loss(config, model_output, data, dp_group)
