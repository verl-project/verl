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

"""Actual BF16 Triton output kernel agrees with an eager vocabulary reference."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.models.mcore.model_forward_fused import FusedOutputProcessorContext, fused_output_processor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA fused output kernels")


@pytest.fixture(scope="module")
def tp_group():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("nccl")
    assert dist.get_world_size() == 1, "This numerical smoke uses one vocabulary partition"
    yield dist.group.WORLD
    if owned_group:
        dist.destroy_process_group()


@pytest.mark.parametrize("tied", [False, True])
def test_actual_fused_output_values_and_gradients(tp_group, tied):
    torch.manual_seed(83)
    hidden = torch.randn(4, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.05)
    labels = torch.randint(128, (2, 4), device="cuda")
    result = fused_output_processor(
        hidden_states=hidden,
        output_layer=SimpleNamespace(weight=weight, tp_group=tp_group),
        output_weight=weight if tied else None,
        labels=labels,
        context=FusedOutputProcessorContext(0.7),
        config=SimpleNamespace(sequence_parallel=False, config_logger_dir=""),
    )
    logits = F.linear(hidden.transpose(0, 1).float(), weight.float()) / 0.7
    logp = logits.log_softmax(-1)
    expected_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(result["log_probs"].float(), expected_logp, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(result["entropy"].float(), expected_entropy, atol=2e-2, rtol=2e-2)
    actual_grads = torch.autograd.grad(
        result["log_probs"].mean() + 0.01 * result["entropy"].mean(), (hidden, weight), retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected_logp.mean() + 0.01 * expected_entropy.mean(), (hidden, weight))
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=5e-4, rtol=5e-2)
