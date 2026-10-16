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
"""`position_ids` is optional, and the Ulysses input-slicing wrapper has to treat it that way.

`patch_vlm_for_ulysses_input_slicing` gates on `inputs_embeds is not None` and then slices
`position_ids` unconditionally. `slice_input_tensor` reads `x.size(dim)` on its first line, so a
caller that omits `position_ids` — which transformers allows, the model then deriving positions
from the sequence length it is handed — hit `AttributeError: 'NoneType' object has no attribute
'size'` instead of running.

The wrapper is installed for qwen2_vl, qwen3_vl, glm4v, kimi_vl and qwen3_5, so this is on the
shared sequence-parallel path rather than one model's.
"""

import torch

from verl.models.transformers import monkey_patch


class _Recorder(torch.nn.Module):
    """Stands in for the text model: records the kwargs the wrapper forwards."""

    def forward(self, **kwargs):
        return {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in kwargs.items()}


def _patched_recorder(monkeypatch, sp_size=2, sp_rank=0):
    """A _Recorder with the wrapper installed, and slicing done without a process group."""
    monkeypatch.setattr(monkey_patch, "get_ulysses_sequence_parallel_world_size", lambda *a, **k: sp_size)

    def slice_without_dist(x, dim, padding=True, group=None):
        parts = x.size(dim) // sp_size
        index = [slice(None)] * x.dim()
        index[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
        return x[tuple(index)].contiguous()

    monkeypatch.setattr(monkey_patch, "slice_input_tensor", slice_without_dist)

    cls = type("Recorder", (_Recorder,), {})
    monkey_patch.patch_vlm_for_ulysses_input_slicing(cls)
    return cls()


def test_slices_both_when_position_ids_are_given(monkeypatch):
    model = _patched_recorder(monkeypatch)

    seen = model(inputs_embeds=torch.randn(1, 64, 8), position_ids=torch.arange(64).view(1, 64))

    assert seen["inputs_embeds"] == (1, 32, 8)
    assert seen["position_ids"] == (1, 32)


def test_slices_embeds_when_position_ids_are_omitted(monkeypatch):
    """The regression: this raised AttributeError from slice_input_tensor's `x.size(dim)`."""
    model = _patched_recorder(monkeypatch)

    seen = model(inputs_embeds=torch.randn(1, 64, 8))

    assert seen["inputs_embeds"] == (1, 32, 8)
    assert "position_ids" not in seen or seen["position_ids"] is None


def test_explicit_none_position_ids_stays_none(monkeypatch):
    """Passing it explicitly as None is the same as omitting it, and must not be sliced."""
    model = _patched_recorder(monkeypatch)

    seen = model(inputs_embeds=torch.randn(1, 64, 8), position_ids=None)

    assert seen["inputs_embeds"] == (1, 32, 8)
    assert seen["position_ids"] is None


def test_each_rank_gets_its_own_shard(monkeypatch):
    """Slicing still follows the rank, so the guard did not flatten it to rank 0."""
    embeds = torch.arange(64, dtype=torch.float32).view(1, 64, 1)

    first = _patched_recorder(monkeypatch, sp_rank=0)
    rank0 = first(inputs_embeds=embeds, position_ids=torch.arange(64).view(1, 64))
    assert rank0["position_ids"] == (1, 32)

    second = _patched_recorder(monkeypatch, sp_rank=1)
    rank1 = second(inputs_embeds=embeds, position_ids=torch.arange(64).view(1, 64))
    assert rank1["inputs_embeds"] == (1, 32, 1)


def test_no_slicing_below_sp_size_two(monkeypatch):
    model = _patched_recorder(monkeypatch, sp_size=1)

    seen = model(inputs_embeds=torch.randn(1, 64, 8))

    assert seen["inputs_embeds"] == (1, 64, 8)
