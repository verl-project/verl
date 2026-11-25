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
"""CPU unit tests for TiledMLP gradient accumulation (issue #8839)."""

import pytest
import torch
import torch.nn as nn

from verl.models.transformers.tiled_mlp import TiledMLP, _mlp_forward_fn


class _SimpleSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize(
    ("token_counts", "shards", "expect_fewer_chunks"),
    [
        ((300, 9), 4, True),
        ((300, 10), 4, False),
        ((5000, 3969), 64, True),
        ((5000, 3970), 64, False),
        ((16, 1), 4, True),
        ((16, 3), 4, True),
        ((32, 9, 17), 4, True),
    ],
)
def test_tiled_mlp_gradient_accumulation_with_fewer_chunks(
    token_counts: tuple[int, ...],
    shards: int,
    expect_fewer_chunks: bool,
):
    """TiledMLP must preserve accumulated parameter gradients when torch.chunk returns fewer than `shards` chunks."""
    torch.manual_seed(0)
    hidden_size = 8
    mlp = _SimpleSwiGLUMLP(hidden_size=hidden_size, intermediate_size=16).double()

    if expect_fewer_chunks:
        assert any(len(torch.chunk(torch.empty(n, hidden_size), chunks=shards, dim=0)) < shards for n in token_counts)
    else:
        assert all(len(torch.chunk(torch.empty(n, hidden_size), chunks=shards, dim=0)) == shards for n in token_counts)

    ref_inputs = [torch.randn(1, n, hidden_size, dtype=torch.float64, requires_grad=True) for n in token_counts]
    tiled_inputs = [x.detach().clone().requires_grad_(True) for x in ref_inputs]

    ref_outputs = []
    for x in ref_inputs:
        out = mlp(x)
        ref_outputs.append(out.detach().clone())
        out.sum().backward()
    ref_param_grads = [p.grad.clone() for p in mlp.parameters()]
    ref_input_grads = [x.grad.clone() for x in ref_inputs]

    for p in mlp.parameters():
        p.grad = None

    tiled_outputs = []
    for x in tiled_inputs:
        out = TiledMLP.apply(_mlp_forward_fn, mlp, x, shards, list(mlp.parameters()))
        tiled_outputs.append(out.detach().clone())
        out.sum().backward()

    for out_tiled, out_ref in zip(tiled_outputs, ref_outputs, strict=True):
        torch.testing.assert_close(out_tiled, out_ref, rtol=1e-12, atol=1e-12)

    for x_tiled, grad_ref in zip(tiled_inputs, ref_input_grads, strict=True):
        assert x_tiled.grad is not None
        torch.testing.assert_close(x_tiled.grad, grad_ref, rtol=1e-12, atol=1e-12)

    for param, grad_ref in zip(mlp.parameters(), ref_param_grads, strict=True):
        assert param.grad is not None
        rel_err = ((param.grad - grad_ref).abs().max() / grad_ref.abs().max()).item()
        assert rel_err < 1e-12, f"max relative grad error {rel_err:.2e} >= 1e-12"
        torch.testing.assert_close(param.grad, grad_ref, rtol=1e-12, atol=1e-12)
