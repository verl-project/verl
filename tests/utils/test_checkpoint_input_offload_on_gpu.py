# Copyright contributors to the PEAR project
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

import functools

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.activation_offload import CheckpointInputOffload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the copies run on CUDA streams")

WIDTH, TOKENS, LAYERS = 1024, 2048, 6  # each layer's input 8 MiB in fp32, above the offload's minimum


class Block(torch.nn.Module):
    """A residual block checkpointed the way transformers' gradient checkpointing sets a layer up."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(WIDTH, WIDTH)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)

    def forward(self, x):
        return x + self._gradient_checkpointing_func(self.inner, x)

    def inner(self, x):
        return torch.tanh(self.linear(x))


class Run:
    """One forward and backward of the blocks, with or without the offload: the gradients and the inputs packed."""

    @staticmethod
    def gradients(offload: CheckpointInputOffload | None) -> tuple[list, list]:
        torch.manual_seed(0)
        model = torch.nn.Sequential(*[Block() for _ in range(LAYERS)]).cuda()
        packed = []
        if offload is not None:
            offload.install(model)
            pack = offload.pack
            offload.pack = lambda tensor: packed.append(tensor.shape) or pack(tensor)
        x = torch.randn(TOKENS, WIDTH, device="cuda", requires_grad=True)
        model(x).square().sum().backward()
        return [p.grad for p in model.parameters()] + [x.grad], packed


def test_the_offloaded_inputs_give_the_same_gradients():
    expected, _ = Run.gradients(None)
    offload = CheckpointInputOffload()
    gradients, packed = Run.gradients(offload)

    assert all(torch.equal(a, b) for a, b in zip(expected, gradients, strict=True))
    assert packed.count(torch.Size([TOKENS, WIDTH])) == LAYERS  # every block's input went through the hooks
    assert offload.pending == []  # and came back exactly once
