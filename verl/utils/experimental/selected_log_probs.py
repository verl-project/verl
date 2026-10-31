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

"""Selected policy log probabilities with chunked forward/backward recomputation."""

import math

import torch


def prediction_candidates(ids, *, pad_size=0, sp_rank=0, sp_size=1):
    if ids.ndim != 2 or pad_size < 0 or sp_size < 1 or not 0 <= sp_rank < sp_size:
        raise ValueError("candidate rows and sequence partition must be valid")
    shifted = ids.roll(-1, 0)
    if pad_size:
        shifted = torch.cat([shifted, ids.new_zeros((pad_size, ids.shape[-1]))])
    if len(shifted) % sp_size:
        raise ValueError("padded candidate rows must divide the sequence parallel size")
    return shifted.chunk(sp_size, dim=0)[sp_rank].contiguous()


class _SelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weights, ids, temperature, chunk_size):
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_ids = ids.reshape(flat_hidden.shape[0], ids.shape[-1])
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        output = torch.empty(flat_ids.shape, dtype=dtype, device=hidden.device)
        for start in range(0, len(flat_hidden), chunk_size):
            stop = start + chunk_size
            logits = (flat_hidden[start:stop] @ weights.T).to(dtype) / temperature
            output[start:stop] = logits.log_softmax(-1).gather(-1, flat_ids[start:stop])
        ctx.save_for_backward(hidden, weights, ids)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.autocast_enabled = torch.is_autocast_enabled(hidden.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(hidden.device.type)
        return output.reshape(ids.shape)

    @staticmethod
    def backward(ctx, grad):
        hidden = ctx.saved_tensors[0]
        with torch.autocast(hidden.device.type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            return _SelectedLogProbs._backward(ctx, grad)

    @staticmethod
    def _backward(ctx, grad):
        hidden, weights, ids = ctx.saved_tensors
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_ids = ids.reshape(len(flat_hidden), ids.shape[-1])
        flat_grad = grad.reshape(flat_ids.shape)
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        hidden_grad = torch.empty_like(flat_hidden) if ctx.needs_input_grad[0] else None
        weights_grad = torch.zeros_like(weights) if ctx.needs_input_grad[1] else None
        for start in range(0, len(flat_hidden), ctx.chunk_size):
            stop = start + ctx.chunk_size
            states = flat_hidden[start:stop]
            upstream = flat_grad[start:stop].to(dtype)
            logits = (states @ weights.T).to(dtype) / ctx.temperature
            logits_grad = -logits.softmax(-1) * upstream.sum(-1, keepdim=True)
            logits_grad.scatter_add_(-1, flat_ids[start:stop], upstream)
            logits_grad = (logits_grad / ctx.temperature).to(hidden.dtype)
            if hidden_grad is not None:
                hidden_grad[start:stop] = logits_grad @ weights
            if weights_grad is not None:
                weights_grad.add_(logits_grad.T @ states)
        return (hidden_grad.reshape(hidden.shape) if hidden_grad is not None else None, weights_grad, None, None, None)


def selected_log_probs(hidden, weights, ids, *, temperature=1.0, chunk_size=128):
    if not math.isfinite(temperature) or temperature <= 0 or chunk_size <= 0:
        raise ValueError("temperature and chunk_size must be positive and finite")
    if hidden.ndim < 2 or weights.ndim != 2 or hidden.shape[-1] != weights.shape[-1]:
        raise ValueError("hidden states and vocabulary weights must have matching hidden dimensions")
    if ids.dtype != torch.long or ids.shape[:-1] != hidden.shape[:-1]:
        raise ValueError("candidate ids must be int64 and align with every hidden-state row")
    if hidden.dtype != weights.dtype or hidden.device != weights.device or hidden.device != ids.device:
        raise ValueError("hidden states, weights and candidates must share their device and model dtype")
    return _SelectedLogProbs.apply(hidden, weights, ids, float(temperature), int(chunk_size))
