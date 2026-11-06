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

from types import MethodType

from torch.distributed.tensor import DTensor

try:
    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method
except ImportError:
    from torch.distributed._composable.fsdp import FSDPModule, register_fsdp_forward_method


def _fused_forward(self, hidden_states, labels, temperature, backend, gather_weights=False, cast_hidden_states=False):
    weight = self.weight
    # Preserve the VLM adapters' non-FSDP DTensor handling. Under FSDP2,
    # the registered pre-forward hook has already materialized this weight.
    if gather_weights and isinstance(weight, DTensor):
        weight = weight.full_tensor().to(hidden_states.device)
    if cast_hidden_states:
        hidden_states = hidden_states.to(weight.dtype)
    if backend == "triton":
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        return linear_cross_entropy(hidden_states, weight, labels, temperature, "none")

    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    return FusedLinearForPPO(impl_backend=backend).forward(
        hidden_states=hidden_states, vocab_weights=weight, input_ids=labels, temperature=temperature
    )


def fused_lm_head_forward(
    lm_head, hidden_states, labels, temperature, backend, *, gather_weights=False, cast_hidden_states=False
):
    """Run fused projection inside the head's FSDP2 forward/backward lifecycle.

    Reading a separately sharded head's weight from its parent bypasses
    unsharding, mixed precision, offload and backward bookkeeping. Register a
    separate entry point so ordinary ``lm_head(hidden_states)`` still returns
    logits and the head keeps its own FSDP communication group.
    """
    args = (hidden_states, labels, temperature, backend, gather_weights, cast_hidden_states)
    if isinstance(lm_head, FSDPModule):
        if not hasattr(lm_head, "_verl_fused_forward"):
            lm_head._verl_fused_forward = MethodType(_fused_forward, lm_head)
            register_fsdp_forward_method(lm_head, "_verl_fused_forward")
        return lm_head._verl_fused_forward(*args)
    return _fused_forward(lm_head, *args)
