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

"""Chunked selected-token log-probabilities for score centering on the VeOmni engine.

Streams the LM-head projection in ``[chunk_size, V]`` fp32 blocks during forward and
recomputes it per chunk in backward, so long-context / large-vocabulary MoE models
evaluate both the sampled-token log-prob and the ``k`` sampler-head log-probs without
materializing a full ``[T, V]`` logits tensor.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Optional

import torch

from verl.utils.ulysses import slice_input_tensor


class _ChunkedSelectedLogProbs(torch.autograd.Function):
    """Chunked linear projection + full-vocabulary log-softmax gather for sampled and top-k tokens."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        topk_ids: torch.Tensor,
        temperature: float,
        chunk_size: int,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)

        compute_dtype = hidden_states.dtype
        if torch.is_autocast_enabled(hidden_states.device.type):
            compute_dtype = torch.get_autocast_dtype(hidden_states.device.type)
        acc_dtype = torch.float64 if compute_dtype == torch.float64 else torch.float32

        orig_label_shape = labels.shape
        orig_hidden_shape = hidden_states.shape
        k = topk_ids.shape[-1]
        h_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        l_1d = labels.reshape(-1)
        ids_2d = topk_ids.reshape(-1, k)
        num_tokens = l_1d.shape[0]

        log_probs = torch.empty(num_tokens, device=h_2d.device, dtype=acc_dtype)
        entropy = torch.empty(num_tokens, device=h_2d.device, dtype=acc_dtype)
        head_log_probs = torch.empty((num_tokens, k), device=h_2d.device, dtype=acc_dtype)

        with torch.autocast(h_2d.device.type, enabled=False):
            w = weight.to(compute_dtype)
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                h_chunk = h_2d[start:end].to(compute_dtype)
                l_chunk = l_1d[start:end]
                ids_chunk = ids_2d[start:end]

                logits = (h_chunk @ w.t()).to(acc_dtype)
                if temperature != 1.0:
                    logits.div_(temperature)
                logp = logits.log_softmax(dim=-1)

                mask = l_chunk != ignore_index
                safe_labels = l_chunk.clamp(min=0).unsqueeze(-1)
                sampled_lp = logp.gather(-1, safe_labels).squeeze(-1)
                log_probs[start:end] = torch.where(mask, sampled_lp, torch.zeros_like(sampled_lp))

                chunk_entropy = -(logp.exp() * logp).sum(dim=-1)
                entropy[start:end] = torch.where(mask, chunk_entropy, torch.zeros_like(chunk_entropy))
                head_log_probs[start:end] = logp.gather(-1, ids_chunk)

        ctx.save_for_backward(h_2d, weight, l_1d, ids_2d)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.compute_dtype = compute_dtype
        ctx.acc_dtype = acc_dtype
        ctx.orig_hidden_shape = orig_hidden_shape

        return log_probs.view(orig_label_shape), entropy.view(orig_label_shape), head_log_probs

    @staticmethod
    def backward(
        ctx,
        dlog_probs: Optional[torch.Tensor],
        dentropy: Optional[torch.Tensor],
        dhead_log_probs: Optional[torch.Tensor],
    ):
        if dlog_probs is None and dentropy is None and dhead_log_probs is None:
            return None, None, None, None, None, None, None

        h_2d, weight, l_1d, ids_2d = ctx.saved_tensors
        num_tokens, k = ids_2d.shape
        compute_dtype = ctx.compute_dtype
        acc_dtype = ctx.acc_dtype

        dlp_1d = dlog_probs.reshape(-1).to(acc_dtype) if dlog_probs is not None else None
        dent_1d = dentropy.reshape(-1).to(acc_dtype) if dentropy is not None else None
        dhead_2d = dhead_log_probs.reshape(-1, k).to(acc_dtype) if dhead_log_probs is not None else None

        dhidden = torch.empty_like(h_2d) if ctx.needs_input_grad[0] else None
        dweight = torch.zeros_like(weight, dtype=acc_dtype) if ctx.needs_input_grad[1] else None

        with torch.autocast(h_2d.device.type, enabled=False):
            w = weight.to(compute_dtype)
            for start in range(0, num_tokens, ctx.chunk_size):
                end = min(start + ctx.chunk_size, num_tokens)
                h_chunk = h_2d[start:end].to(compute_dtype)
                l_chunk = l_1d[start:end]
                ids_chunk = ids_2d[start:end]

                logits = (h_chunk @ w.t()).to(acc_dtype)
                if ctx.temperature != 1.0:
                    logits.div_(ctx.temperature)
                logp = logits.log_softmax(dim=-1)
                probs = logp.exp()
                dlogits = torch.zeros_like(probs)
                mask = (l_chunk != ctx.ignore_index).to(acc_dtype)

                if dlp_1d is not None:
                    safe_labels = l_chunk.clamp(min=0).unsqueeze(-1)
                    dlp_masked = (dlp_1d[start:end] * mask).unsqueeze(-1)
                    dlogits.sub_(probs * dlp_masked)
                    dlogits.scatter_add_(-1, safe_labels, dlp_masked)

                if dent_1d is not None:
                    dent_masked = (dent_1d[start:end] * mask).unsqueeze(-1)
                    chunk_entropy = -(probs * logp).sum(dim=-1, keepdim=True)
                    dlogits.sub_(dent_masked * probs * (logp + chunk_entropy))

                if dhead_2d is not None:
                    dhead = dhead_2d[start:end]
                    dlogits.sub_(probs * dhead.sum(dim=-1, keepdim=True))
                    dlogits.scatter_add_(-1, ids_chunk, dhead)

                if ctx.temperature != 1.0:
                    dlogits.div_(ctx.temperature)

                dlogits_cast = dlogits.to(compute_dtype)
                if dhidden is not None:
                    dhidden[start:end] = (dlogits_cast @ w).to(h_2d.dtype)
                if dweight is not None:
                    dweight.add_((dlogits_cast.t() @ h_chunk).to(acc_dtype))

        return (
            dhidden.view(ctx.orig_hidden_shape) if dhidden is not None else None,
            dweight.to(weight.dtype) if dweight is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_selected_linear_log_probs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(log_probs, entropy, head_log_probs)`` from ``hidden_states`` and ``weight`` in chunks."""
    if chunk_size <= 0 or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(temperature, int | float) or not (0.0 < temperature < float("inf")):
        raise ValueError("temperature must be a finite positive scalar")
    return _ChunkedSelectedLogProbs.apply(
        hidden_states,
        weight,
        labels.long(),
        topk_ids.long(),
        float(temperature),
        int(chunk_size),
        int(ignore_index),
    )


def pack_veomni_topk_ids(
    rollout_topk_ids: torch.Tensor,
    input_ids: torch.Tensor,
    pad_size: int,
    *,
    use_ulysses_sp: bool = False,
) -> torch.Tensor:
    """Pack nested ``rollout_topk_ids`` into the local ``(nnz/sp, k)`` layout seen by the LM head."""
    if not rollout_topk_ids.is_nested or not torch.equal(rollout_topk_ids.offsets(), input_ids.offsets()):
        raise ValueError("rollout_topk_ids must share the nested jagged offsets of input_ids")
    ids = rollout_topk_ids.values().long()
    if ids.ndim != 2 or ids.shape[-1] == 0:
        raise ValueError("Packed rollout_topk_ids must have shape [tokens, k] with k > 0")
    if use_ulysses_sp:
        return slice_input_tensor(ids.unsqueeze(0), dim=1, padding=True).squeeze(0)
    if pad_size > 0:
        ids = torch.cat([ids, ids.new_zeros((pad_size, ids.shape[-1]))], dim=0)
    return ids.contiguous()


@dataclass
class VeOmniScoreCenteringRequest:
    """Per-forward state passing top-k head IDs into VeOmni's fused LM-head hook."""

    topk_ids: Optional[torch.Tensor] = None
    head_log_probs: Optional[torch.Tensor] = None
    chunk_size: int = 1024


_active_request: ContextVar[Optional[VeOmniScoreCenteringRequest]] = ContextVar(
    "verl_veomni_score_centering_request", default=None
)


@contextmanager
def veomni_score_centering_context():
    """Scope a fresh ``VeOmniScoreCenteringRequest`` to one engine forward step."""
    request = VeOmniScoreCenteringRequest()
    token = _active_request.set(request)
    try:
        yield request
    finally:
        _active_request.reset(token)


def current_veomni_score_centering_request() -> Optional[VeOmniScoreCenteringRequest]:
    """Return the active ``VeOmniScoreCenteringRequest`` if inside ``veomni_score_centering_context``."""
    return _active_request.get()


def make_veomni_score_centering_adapter(original):
    """Wrap VeOmni's ``chunk_logprobs_function`` to emit ``head_log_probs`` when requested."""

    @wraps(original)
    def adapted(
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int = 1024,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        request = _active_request.get()
        if request is None or request.topk_ids is None:
            return original(
                hidden_states,
                weights,
                labels,
                chunk_size=chunk_size,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                temperature=temperature,
                **kwargs,
            )
        if request.head_log_probs is not None:
            raise RuntimeError("score centering expected a single fused LM-head call per engine forward")
        if shift_labels is None:
            raise ValueError("VeOmni score centering requires globally shifted labels (shift_labels)")
        if shift_labels.numel() != request.topk_ids.shape[0]:
            raise ValueError(
                f"VeOmni score centering topk_ids length ({request.topk_ids.shape[0]}) does not match "
                f"shift_labels ({shift_labels.numel()})"
            )
        log_probs, entropy, head_log_probs = chunked_selected_linear_log_probs(
            hidden_states,
            weights,
            shift_labels,
            request.topk_ids,
            temperature=temperature,
            chunk_size=request.chunk_size,
            ignore_index=ignore_index,
        )
        request.head_log_probs = head_log_probs.contiguous()
        return log_probs, entropy

    adapted._verl_sc_adapter = True
    return adapted


def install_veomni_score_centering_adapter() -> None:
    """Install the score-centering wrapper onto ``veomni.ops.kernels.cross_entropy`` once."""
    from veomni.ops.kernels import cross_entropy

    original = cross_entropy.chunk_logprobs_function
    if not getattr(original, "_verl_sc_adapter", False):
        cross_entropy.chunk_logprobs_function = make_veomni_score_centering_adapter(original)
