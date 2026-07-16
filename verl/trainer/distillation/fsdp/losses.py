# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


from typing import Optional

import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
    input_temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.
        input_temperature: optional [B, T, 1] (broadcastable) factor multiplied onto
            ``logits`` before the softmax. Used to undo the engine's sampling-temperature
            division so the distillation operates on RAW logits (see ``compute_forward_kl_topk``).

    Returns:
        [B, T, K] tensor with the same dtype as `logits`.
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    flat_temp = input_temperature.reshape(-1, 1) if input_temperature is not None else None  # [N, 1]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)

    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        if flat_temp is not None:
            chunk_logits_fp32 = chunk_logits_fp32 * flat_temp[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def kl_divergence(
    log_q: torch.Tensor, log_p: torch.Tensor, clip_tau: Optional[float] = None
) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities.

    When ``clip_tau`` is set, each per-vocab contribution ``ℓ_{n,v} = p*(log_p - log_q)``
    is clamped to at most ``clip_tau`` *before* the vocab sum (OPSD pointwise KL
    clipping, arXiv:2601.18734 §4.3.3): a few stylistic tokens otherwise dominate the
    per-position divergence and the gradient. This is distinct from
    ``DistillationLossConfig.loss_max_clamp``, which clamps the post-sum scalar.
    """
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    if clip_tau is not None:
        kld = kld.clamp_max(clip_tau)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    student_logits_temperature: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5
        student_logits_temperature: optional (1, seqlen/sp_size, 1) factor that the engine already
            divided ``student_logits`` by (the rollout/sampling temperature, applied for the policy
            log-probs in ``transformer_impl.py``). We multiply it back so the distillation target is
            built from RAW logits and only ``loss_temperature`` shapes it, matching the reference OPSD
            (siyan-zhao/OPSD), where the distillation temperature is applied once to the raw student
            logits. Without this undo the student is divided by temperature twice (rollout T, then
            ``loss_temperature``), softening it relative to the teacher. None == no undo (T_engine=1).

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    loss_temperature = getattr(loss_config, "loss_temperature", None)
    if use_chunked_topk:
        if loss_temperature is not None:
            raise NotImplementedError("loss_temperature is not supported together with use_chunked_topk.")
        # log_softmax is monotonic and scale-invariant under a positive per-token factor, so
        # topk(student_logits) == topk(softmax(student_logits)) regardless of the temperature undo.
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
            input_temperature=student_logits_temperature,
        )
    else:
        # Undo the engine's sampling-temperature division so the distillation sees RAW logits
        # (see ``student_logits_temperature`` in the docstring). The later loss_temperature path
        # gathers from ``student_logits``, so reassigning here makes both paths consistent.
        if student_logits_temperature is not None:
            student_logits = student_logits * student_logits_temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    # Diagnostic masses are reported at T=1 (full-vocab gathered log-probs), before any
    # temperature renormalization below, so student_mass/teacher_mass stay interpretable.
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    if loss_temperature is not None:
        # Reference OPSD parity (siyan-zhao/OPSD generalized_jsd_loss top-k path): divide BOTH logits by the
        # distillation temperature T and renormalize over the teacher's top-k support before the KL. For the
        # teacher we only hold top-k log-probs (vLLM, T=1), but softmax over the top-k set is invariant to the
        # unknown full-vocab normalizer, so log_softmax(teacher_topk_log_probs / T) exactly recovers the
        # temperature-scaled, top-k-renormalized teacher distribution. As topk -> vocab this -> full-vocab.
        student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_ids)
        student_topk_log_probs = F.log_softmax(student_topk_logits / loss_temperature, dim=-1)
        teacher_topk_log_probs = F.log_softmax(teacher_topk_log_probs / loss_temperature, dim=-1)

    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(
        log_q=student_topk_log_probs,
        log_p=teacher_topk_log_probs,
        clip_tau=getattr(loss_config, "clip_tau", None),  # OPSD pointwise KL clip (per-vocab)
    )

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
