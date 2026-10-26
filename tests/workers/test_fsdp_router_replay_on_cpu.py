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

"""FSDP engine R3 router replay: ``routed_experts`` reaches the model packed like ``input_ids``."""

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

LAYERS, TOPK = 3, 2


class _ReplayModel(torch.nn.Module):
    def forward(self, input_ids, routed_experts=None):
        return input_ids


class _PlainModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        return input_ids


def _engine(module, use_ulysses_sp=False):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = module
    engine.use_ulysses_sp = use_ulysses_sp
    return engine


def _micro_batch(with_routing=True):
    offsets = torch.tensor([0, 3, 5])
    fields = {"input_ids": torch.nested.nested_tensor_from_jagged(torch.arange(5), offsets)}
    if with_routing:
        routed = torch.arange(1, 5 * LAYERS * TOPK + 1, dtype=torch.int16).view(5, LAYERS, TOPK)
        fields["routed_experts"] = torch.nested.nested_tensor_from_jagged(routed, offsets)
    return TensorDict(fields, batch_size=2)


@pytest.mark.parametrize("pad_size", [0, 4], ids=["no-pad", "static-pad"])
def test_packs_like_input_ids(pad_size):
    micro_batch = _micro_batch()
    packed = _engine(_ReplayModel())._pack_routed_experts(micro_batch, {"pad_size": pad_size}, True)

    assert packed.dtype == torch.int64
    assert packed.shape == (1, 5 + pad_size, LAYERS, TOPK)
    torch.testing.assert_close(packed[0, :5], micro_batch["routed_experts"].values().long())
    assert torch.count_nonzero(packed[0, 5:]) == 0


@pytest.mark.parametrize(
    "module,with_routing,use_remove_padding,use_ulysses_sp",
    [
        (_PlainModel(), True, True, False),
        (_ReplayModel(), False, True, False),
        (_ReplayModel(), True, False, False),
        (_ReplayModel(), True, True, True),
    ],
    ids=["model-ignores-routing", "batch-without-routing", "padded-inputs", "ulysses-sp"],
)
def test_rejects_unsupported(module, with_routing, use_remove_padding, use_ulysses_sp):
    with pytest.raises(NotImplementedError):
        _engine(module, use_ulysses_sp)._pack_routed_experts(
            _micro_batch(with_routing), {"pad_size": 0}, use_remove_padding
        )
