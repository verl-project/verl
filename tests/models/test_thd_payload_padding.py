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

"""Top-k alignment rows must remain absent after either CP partition."""

import pytest
import torch

from verl.models.mcore import util


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("rank", [0, 1])
def test_logprob_padding_remains_negative_infinity(monkeypatch, layout, rank):
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda: rank)
    payload = torch.nested.nested_tensor([torch.full((3, 2), -2.0)], layout=torch.jagged)
    packed, _, _ = util.preprocess_thd_engine(payload, cp_layout=layout, pad_value=float("-inf"))
    expected_live_rows = 1 if (layout == "zigzag" and rank == 0) or (layout == "contiguous" and rank == 1) else 2
    assert torch.isfinite(packed).all(dim=-1).sum().item() == expected_live_rows
    assert torch.isneginf(packed).all(dim=-1).sum().item() == 2 - expected_live_rows
    assert torch.all(packed[torch.isfinite(packed)] == -2.0)
