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
"""VL processors must not call get_rope_index on text-only batches."""

from types import SimpleNamespace

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils.model import compute_position_id_with_mask


class _ExplodingVLProcessor:
    def get_rope_index(self, *args, **kwargs):
        raise TypeError("NoneType is not an iterator")


def test_text_only_batch_skips_vl_get_rope_index():
    worker = object.__new__(AgentLoopWorker)
    worker.processor = _ExplodingVLProcessor()

    attention_mask = torch.ones(1, 6, dtype=torch.long)
    input_ids = torch.arange(6, dtype=torch.long).unsqueeze(0)

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={},
    )

    expected = compute_position_id_with_mask(attention_mask)
    torch.testing.assert_close(position_ids, expected)


def test_multimodal_grid_still_uses_get_rope_index():
    worker = object.__new__(AgentLoopWorker)
    worker.processor = SimpleNamespace(get_rope_index=lambda **kwargs: (torch.zeros(3, 1, 4, dtype=torch.long), None))

    attention_mask = torch.ones(1, 4, dtype=torch.long)
    input_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={"image_grid_thw": torch.tensor([[1, 2, 2]])},
    )

    assert position_ids.shape[0] == 1
    assert position_ids.shape[-1] == 4
