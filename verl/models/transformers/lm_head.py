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

from types import MethodType
from typing import Optional

import torch
from torch import nn


def fp32_mm(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Matrix multiply with FP32 accumulation and output."""
    with torch.autocast(device_type=left.device.type, enabled=False):
        if left.device.type == "cuda" and left.dtype in (torch.float16, torch.bfloat16) and left.dtype == right.dtype:
            return torch.mm(left, right, out=out, out_dtype=torch.float32)
        return torch.mm(left.float(), right.float(), out=out)


class _Fp32LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        output_shape = (*hidden_states.shape[:-1], weight.shape[0])
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = torch.empty(output_shape, device=hidden_states.device, dtype=torch.float32)
        fp32_mm(flat_hidden, weight.t(), out=output.view(-1, weight.shape[0]))
        if bias is not None:
            output.add_(bias.float())

        ctx.save_for_backward(hidden_states, weight)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, weight = ctx.saved_tensors
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        grad_hidden = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_hidden = fp32_mm(flat_grad_output, weight).to(hidden_states.dtype).view_as(hidden_states)
        if ctx.needs_input_grad[1]:
            grad_weight = fp32_mm(flat_grad_output.t(), flat_hidden).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = flat_grad_output.sum(dim=0).to(ctx.bias_dtype)

        return grad_hidden, grad_weight, grad_bias


def fp32_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a linear projection whose accumulation and output are FP32."""
    return _Fp32LinearFunction.apply(hidden_states, weight, bias)


def _fp32_lm_head_forward(
    self: nn.Linear,
    hidden_states: torch.Tensor,
    *,
    token_ids: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    fused_backend: str = "torch",
):
    if token_ids is None:
        return fp32_linear(hidden_states, self.weight, self.bias)
    if self.bias is not None:
        raise NotImplementedError("Fused FP32 lm_head does not support bias.")
    if fused_backend == "triton":
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        return linear_cross_entropy(hidden_states, self.weight, token_ids, temperature, "none")

    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    return FusedLinearForPPO(impl_backend=fused_backend, lm_head_dtype="float32")(
        hidden_states,
        self.weight,
        token_ids,
        temperature,
    )


def install_fp32_lm_head(model: nn.Module) -> nn.Linear:
    """Patch the output layer in place without replacing parameters or state-dict keys."""
    output_layer = model.get_output_embeddings()
    if not isinstance(output_layer, nn.Linear):
        raise TypeError(
            "lm_head_dtype='float32' currently requires get_output_embeddings() to return torch.nn.Linear; "
            f"got {type(output_layer).__name__}."
        )
    output_layer.forward = MethodType(_fp32_lm_head_forward, output_layer)
    return output_layer
