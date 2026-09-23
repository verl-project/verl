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

from typing import Optional

import torch

from verl.models.transformers.lm_head import fp32_mm

try:
    from liger_kernel.ops import (
        LigerFusedLinearScaledCrossEntropyFunction as _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY,
    )
except ModuleNotFoundError as exc:
    if exc.name != "liger_kernel":
        raise
    _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY = None

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False


def _compute_logits(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    temperature: float,
    lm_head_dtype: Optional[str],
) -> tuple[torch.FloatTensor, torch.dtype]:
    if lm_head_dtype == "float32":
        return fp32_mm(hidden_states, vocab_weights.t()) / temperature, torch.float32
    logits = (hidden_states @ vocab_weights.t()) / temperature
    return logits.float(), logits.dtype


def _fused_linear_for_ppo_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
    lm_head_dtype: Optional[str] = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    logits, orig_dtype = _compute_logits(hidden_states, vocab_weights, temperature, lm_head_dtype)

    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)

    if _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        per_token_entropy_loss = cross_entropy_loss(logits, input_ids)[0]
        token_log_probs = -per_token_entropy_loss
    else:
        # Fallback to original PyTorch implementation
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    assert token_log_probs.dtype == torch.float32
    return token_log_probs, entropy.to(orig_dtype)


def _fused_linear_for_ppo_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
    lm_head_dtype: Optional[str] = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    logits, orig_dtype = _compute_logits(hidden_states, vocab_weights, temperature, lm_head_dtype)

    probs = logits.softmax(dim=-1)

    dlogits = 0

    # Gradient from log_probs
    if dlog_probs is not None:
        one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)

    # Gradient from entropy
    if dentropy is not None:
        log_probs = logits.log_softmax(dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    if lm_head_dtype == "float32":
        dlogits = dlogits / temperature
        dhidden_states = fp32_mm(dlogits, vocab_weights)
        dvocab_weights = fp32_mm(dlogits.t(), hidden_states)
    else:
        dlogits = dlogits.to(orig_dtype) / temperature
        dhidden_states = dlogits @ vocab_weights
        dvocab_weights = dlogits.t() @ hidden_states

    return dhidden_states, dvocab_weights


class FusedLinearForPPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
        lm_head_dtype: Optional[str] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        ctx.set_materialize_grads(False)

        # Cast to a 2D tensor of the shape [T, D] for ease of working
        orig_ndim = hidden_states.ndim
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"
        output_shape = input_ids.shape

        orig_batch_size = -1
        if orig_ndim == 3:
            assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
            orig_batch_size = hidden_states.shape[0]
            hidden_states_requires_grad = hidden_states.requires_grad
            hidden_states = hidden_states.flatten(0, 1)
            hidden_states.requires_grad_(hidden_states_requires_grad)
            input_ids = input_ids.flatten(0, 1)

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        # Logits are upcasted to fp32 before computing log_probs, which are also fp32
        log_probs = torch.zeros(
            output_shape, device=hidden_states.device, dtype=torch.float32, requires_grad=output_requires_grad
        )
        entropy_dtype = torch.float32 if lm_head_dtype == "float32" else hidden_states.dtype
        entropy = torch.zeros(
            output_shape,
            device=hidden_states.device,
            dtype=entropy_dtype,
            requires_grad=output_requires_grad,
        )
        flat_log_probs = log_probs.flatten()
        flat_entropy = entropy.flatten()

        # Perform forward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)

            chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
                lm_head_dtype=lm_head_dtype,
            )
            flat_log_probs[chunk_start:chunk_end] = chunk_log_probs
            flat_entropy[chunk_start:chunk_end] = chunk_entropy

        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        ctx.orig_batch_size = orig_batch_size
        ctx.orig_ndim = orig_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.lm_head_dtype = lm_head_dtype

        return log_probs, entropy

    @staticmethod
    def backward(ctx, dlog_probs: Optional[torch.FloatTensor], dentropy: Optional[torch.FloatTensor]):
        assert dlog_probs is not None or dentropy is not None

        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        orig_batch_size = ctx.orig_batch_size
        orig_ndim = ctx.orig_ndim
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        lm_head_dtype = ctx.lm_head_dtype

        # Here orig_ndim refers to the orig_ndim of hidden_states
        if orig_ndim == 3:
            if dlog_probs is not None:
                dlog_probs = dlog_probs.flatten()
            if dentropy is not None:
                dentropy = dentropy.flatten()

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            accumulation_dtype = torch.float32 if lm_head_dtype == "float32" else vocab_weights.dtype
            dvocab_weights = torch.zeros_like(vocab_weights, dtype=accumulation_dtype)

        # Perform backward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_dlog_probs = None
            if dlog_probs is not None:
                chunk_dlog_probs = dlog_probs[chunk_start:chunk_end]
            chunk_dentropy = None
            if dentropy is not None:
                chunk_dentropy = dentropy[chunk_start:chunk_end]

            h, v = _fused_linear_for_ppo_bwd(
                dlog_probs=chunk_dlog_probs,
                dentropy=chunk_dentropy,
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
                lm_head_dtype=lm_head_dtype,
            )

            if hidden_states.requires_grad:
                dhidden_states[chunk_start:chunk_end] += h.to(hidden_states.dtype)
            if vocab_weights.requires_grad:
                dvocab_weights += v

        if dvocab_weights is not None and dvocab_weights.dtype != vocab_weights.dtype:
            dvocab_weights = dvocab_weights.to(vocab_weights.dtype)

        # Cast the output back to the original input dimension
        if orig_ndim == 3 and hidden_states.requires_grad:
            hidden_size = hidden_states.shape[-1]
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        return (
            dhidden_states,  # hidden_states
            dvocab_weights,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
            None,  # lm_head_dtype
        )


class FusedLinearForPPO(torch.nn.Module):
    def __init__(self, chunk_size: int = 512, impl_backend: str = "torch", lm_head_dtype: Optional[str] = None):
        super().__init__()

        if impl_backend not in ("torch", "liger"):
            raise ValueError(f"Unsupported FusedLinearForPPO backend: {impl_backend}. Choose 'torch' or 'liger'.")
        self.chunk_size = chunk_size
        self.impl_backend = impl_backend
        if lm_head_dtype not in (None, "float32"):
            raise ValueError(f"Unsupported lm_head_dtype: {lm_head_dtype!r}")
        self.lm_head_dtype = lm_head_dtype

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        input_ids = input_ids.to(torch.int64)
        if (
            self.impl_backend == "torch"
            or self.lm_head_dtype == "float32"
            or _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY is None
        ):
            return FusedLinearForPPOFunction.apply(
                hidden_states,
                vocab_weights,
                input_ids,
                temperature,
                self.chunk_size,
                self.lm_head_dtype,
            )

        if hidden_states.ndim not in (2, 3):
            raise ValueError(f"hidden_states must be 2D or 3D, got shape {tuple(hidden_states.shape)}")
        if input_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match hidden_states shape "
                f"{tuple(hidden_states.shape[:-1])}"
            )

        output_shape = input_ids.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        input_ids = input_ids.reshape(-1)

        nll, entropy = _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            -100,
            1,
            True,
        )
        log_probs = -nll

        return log_probs.reshape(output_shape), entropy.reshape(output_shape)
