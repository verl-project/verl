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
"""Position IDs from M-RoPE processors must stay batchable on CPU."""

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

    expected = compute_position_id_with_mask(attention_mask).unsqueeze(1).expand(-1, 4, -1)
    torch.testing.assert_close(position_ids, expected)


def test_text_only_batch_skips_audio_rope_kwargs_hook():
    class AudioProcessor:
        def get_rope_index_kwargs(self, multi_modal_inputs):
            raise AssertionError("text-only batches must not call the audio hook")

        def get_rope_index(self, *args, **kwargs):
            raise AssertionError("text-only batches must not call get_rope_index")

    worker = object.__new__(AgentLoopWorker)
    worker.processor = AudioProcessor()
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    input_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={},
    )

    expected = compute_position_id_with_mask(attention_mask).unsqueeze(1).expand(-1, 4, -1)
    torch.testing.assert_close(position_ids, expected)


def test_text_only_and_vision_position_ids_can_be_concatenated():
    worker = object.__new__(AgentLoopWorker)
    worker.processor = SimpleNamespace(
        get_rope_index=lambda **kwargs: (torch.zeros(3, 1, kwargs["input_ids"].shape[-1], dtype=torch.long), None)
    )

    attention_mask = torch.ones(1, 4, dtype=torch.long)
    input_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    text_position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={},
    )
    vision_position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={"image_grid_thw": torch.tensor([[1, 2, 2]])},
    )

    assert torch.cat([text_position_ids, vision_position_ids], dim=0).shape == (2, 4, 4)


def test_non_visual_multimodal_inputs_do_not_fallback():
    calls = []

    def get_rope_index(*, input_ids, **kwargs):
        calls.append(kwargs)
        return torch.zeros(3, 1, input_ids.shape[-1], dtype=torch.long), None

    worker = object.__new__(AgentLoopWorker)
    worker.processor = SimpleNamespace(get_rope_index=get_rope_index)
    input_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    for multi_modal_inputs in (
        {"input_features": torch.zeros(1, 2, 3)},
        {"feature_attention_mask": torch.ones(1, 3, dtype=torch.long)},
    ):
        AgentLoopWorker._compute_position_ids(
            worker,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
        )

    assert len(calls) == 2


def test_audio_rope_kwargs_hook_still_uses_get_rope_index():
    calls = []

    class AudioProcessor:
        def get_rope_index_kwargs(self, multi_modal_inputs):
            calls.append("hook")
            return {"audio_seqlens": multi_modal_inputs["feature_attention_mask"].sum(-1)}

        def get_rope_index(self, *, input_ids, audio_seqlens, **kwargs):
            calls.append("rope")
            torch.testing.assert_close(audio_seqlens, torch.tensor([2]))
            return torch.zeros(3, 1, input_ids.shape[-1], dtype=torch.long), None

    worker = object.__new__(AgentLoopWorker)
    worker.processor = AudioProcessor()
    input_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={
            "input_features": torch.zeros(1, 2, 3),
            "feature_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
    )

    assert calls == ["hook", "rope"]
    assert position_ids.shape == (1, 4, 3)


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
