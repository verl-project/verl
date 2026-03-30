# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch

from verl.utils.model import build_qwen_vl_position_ids, is_qwen_vl_processor
from verl.workers.rollout.schemas import AsyncRolloutRequest


class _DummyTokenizer:
    def __init__(self):
        self._token_to_id = {
            "<|vision_start|>": 10,
            "<|image_pad|>": 11,
            "<|video_pad|>": 12,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id[token]


class _Qwen2VLImageProcessor:
    def __init__(self):
        self.merge_size = 2


class _DummyQwen2VLProcessor:
    def __init__(self):
        self.image_processor = _Qwen2VLImageProcessor()
        self.tokenizer = _DummyTokenizer()

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
    ):
        del attention_mask, image_grid_thw, second_per_grid_ts
        seq_len = input_ids.shape[-1]
        vision_position_ids = torch.zeros((3, 1, seq_len), dtype=torch.long)
        if video_grid_thw is not None:
            video_grid = video_grid_thw[0]
            frame_count = int(video_grid[0].item())
            patch_tokens_per_frame = int((video_grid[1].item() // self.image_processor.merge_size) * (video_grid[2].item() // self.image_processor.merge_size))
            video_token_positions = (input_ids[0] == self.tokenizer.convert_tokens_to_ids("<|video_pad|>")).nonzero(as_tuple=False).flatten()
            for frame_idx in range(frame_count):
                start = frame_idx * patch_tokens_per_frame
                end = start + patch_tokens_per_frame
                vision_position_ids[0, 0, video_token_positions[start:end]] = frame_idx
        return vision_position_ids, None


def _make_video_sample(frame_count: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    patch_tokens_per_frame = 4
    text_prefix = torch.tensor([101, 102, 10], dtype=torch.long)
    video_tokens = torch.full((frame_count * patch_tokens_per_frame,), 12, dtype=torch.long)
    text_suffix = torch.tensor([103], dtype=torch.long)
    input_ids = torch.cat((text_prefix, video_tokens, text_suffix), dim=0)
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {"video_grid_thw": torch.tensor([[frame_count, 4, 4]], dtype=torch.long)}
    return input_ids, attention_mask, multi_modal_inputs


def test_qwen_vl_rollout_position_ids_use_text_plus_vision_axes():
    processor = _DummyQwen2VLProcessor()
    input_ids, attention_mask, multi_modal_inputs = _make_video_sample(frame_count=2)

    position_ids = AsyncRolloutRequest._get_position_ids(
        processor,
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        multi_modal_inputs=multi_modal_inputs,
    )

    assert position_ids.shape == (4, input_ids.numel())
    assert torch.equal(position_ids[0], torch.arange(input_ids.numel(), dtype=torch.long))
    assert torch.all(position_ids[1:] >= 0)


def test_qwen_vl_mixed_video_lengths_collate_to_padded_four_axis_position_ids():
    processor = _DummyQwen2VLProcessor()

    position_ids_list = []
    expected_lengths = []
    for frame_count in (2, 3):
        input_ids, attention_mask, multi_modal_inputs = _make_video_sample(frame_count=frame_count)
        position_ids = build_qwen_vl_position_ids(
            processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
        )
        position_ids_list.append(position_ids)
        expected_lengths.append(input_ids.numel())

    max_seq_len = max(expected_lengths)
    padded_position_ids = torch.stack(
        [torch.nn.functional.pad(position_ids, (0, max_seq_len - position_ids.shape[-1]), value=0) for position_ids in position_ids_list],
        dim=0,
    ).transpose(0, 1)

    assert is_qwen_vl_processor(processor)
    assert padded_position_ids.shape == (4, len(position_ids_list), max_seq_len)
    assert torch.equal(padded_position_ids[0, 0, : expected_lengths[0]], torch.arange(expected_lengths[0]))
    assert torch.equal(padded_position_ids[0, 1, : expected_lengths[1]], torch.arange(expected_lengths[1]))
