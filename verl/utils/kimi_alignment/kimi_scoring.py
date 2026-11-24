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
"""Candidate exact Ascend V2 scoring with an analytical first-order VJP.

Enabled through KIMI_ALIGNMENT_V2_SCORING. Scores use this caller's
logits; no rollout scores or hidden activations are read.
"""

import torch


class _V2Logprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs

        shape = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1).long()
        ignored = labels == -100
        safe_labels = labels.masked_fill(ignored, 0)
        output = compute_token_logprobs(flat, safe_labels[:, None]).squeeze(-1)
        output = output.masked_fill(ignored, 0)
        ctx.save_for_backward(logits, safe_labels, ignored)
        return output.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, ignored = ctx.saved_tensors
        flat = logits.reshape(-1, logits.shape[-1])
        cotangent = grad_output.reshape(-1).float().masked_fill(ignored, 0)
        grad = torch.empty_like(flat)
        # Bound the FP32 temporary rather than duplicating the full vocabulary
        # activation for every token in a training microbatch.
        for start in range(0, flat.shape[0], 128):
            end = min(start + 128, flat.shape[0])
            value = -flat[start:end].float().softmax(-1)
            value.scatter_add_(1, labels[start:end, None], torch.ones_like(value[:, :1]))
            value.mul_(cotangent[start:end, None])
            grad[start:end].copy_(value.to(flat.dtype))
        return grad.reshape(logits.shape), None


def kimi_v2_logprobs(logits, labels, inplace_backward=True, batch_invariant=False, temperature=None):
    if batch_invariant or temperature is not None:
        raise ValueError("Ascend V2 scoring expects already-scaled logits; batch_invariant must be False")
    if logits.device.type != "npu":
        raise ValueError("This candidate is specific to Ascend V2 sampling")
    return _V2Logprobs.apply(logits, labels)


def masked_entropy(logits, chunk_size=128):
    shape = logits.shape[:-1]
    flat = logits.reshape(-1, logits.shape[-1])
    result = []
    for chunk in flat.split(chunk_size, dim=0):
        values = chunk.float()
        # Subtract a common offset before both reductions, and normalize the
        # exponentials explicitly. This avoids cancellation and accumulated
        # probability-mass error from the NPU's fused log_softmax output.
        shifted = values - values.amax(-1, keepdim=True).detach()
        weights = shifted.exp()
        total = weights.sum(-1, keepdim=True)
        probabilities = weights / total
        safe_shifted = shifted.masked_fill(torch.isneginf(values), 0)
        result.append(total.log().squeeze(-1) - (probabilities * safe_shifted).sum(-1))
    if not result:
        return logits.sum(dim=-1).float()
    return torch.cat(result).reshape(shape)
