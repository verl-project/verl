# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compatibility for the released VeOmni chunk-gradient implementation."""

import importlib
import importlib.metadata
from functools import wraps


class _BackwardContextWithInputGradFlags:
    """Delegate context metadata while restoring flags on storage-sharing aliases."""

    def __init__(self, context):
        self._context = context
        hidden, weight, labels = context.saved_tensors
        self.saved_tensors = (
            hidden.detach().requires_grad_(context.needs_input_grad[0]),
            weight.detach().requires_grad_(context.needs_input_grad[1]),
            labels,
        )

    def __getattr__(self, name):
        return getattr(self._context, name)


def _needs_input_grad_backward_adapter(original):
    """Repair the 0.1.11 backward's requires_grad checks without changing math.

    Saved forward reshapes need not retain input requires_grad (observed even
    for contiguous nonleaf inputs on PyTorch 2.13). The authoritative flags are
    ctx.needs_input_grad. Detached aliases share
    storage and do not mutate the saved tensors or copy activations. This
    release kernel implements first-order gradients only.
    """

    @wraps(original)
    def adapted(ctx, *grads):
        return original(_BackwardContextWithInputGradFlags(ctx), *grads)

    adapted._verl_input_grad_flags_compat = True
    return adapted


def install_chunk_logprobs_compat():
    """Repair first-order gradients in the VeOmni 0.1.11 chunked log-probability op."""
    if importlib.metadata.version("veomni") != "0.1.11":
        return False
    module = importlib.import_module("veomni.ops.kernels.cross_entropy.chunk_logprobs")
    kernel = module._ChunkedLinearLogProbs
    if not getattr(kernel.backward, "_verl_input_grad_flags_compat", False):
        kernel.backward = staticmethod(_needs_input_grad_backward_adapter(kernel.backward))
    return True
