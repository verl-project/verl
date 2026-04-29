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
import pytest
import torch

from verl.experimental.agent_loop.vllm_mm_processor import build_vllm_mm_processor_data


class DummyQwen3VLVideoProcessor:
    temporal_patch_size = 2
    fps = 2
    min_frames = 1
    max_frames = 128


class DummyQwen3VLProcessor:
    video_processor = DummyQwen3VLVideoProcessor()


class DummyQwen2VLProcessor:
    pass


def test_build_qwen3_video_processor_output_adds_timestamps():
    model_inputs = {
        "pixel_values_videos": torch.ones(8, 3),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
    }
    videos = [
        (
            torch.zeros(4, 3, 16, 16),
            {
                "fps": 30.0,
                "frames_indices": [0, 1, 2, 3],
                "total_num_frames": 4,
                "do_sample_frames": False,
            },
        )
    ]

    mm_data = build_vllm_mm_processor_data(
        processor=DummyQwen3VLProcessor(),
        model_inputs=model_inputs,
        videos=videos,
    )

    assert mm_data is not None
    assert set(mm_data["video"]) == {"pixel_values_videos", "video_grid_thw", "timestamps"}
    assert mm_data["video"]["pixel_values_videos"] is model_inputs["pixel_values_videos"]
    assert mm_data["video"]["video_grid_thw"] is model_inputs["video_grid_thw"]
    assert mm_data["video"]["timestamps"][0] == pytest.approx([1 / 60, 5 / 60])


def test_build_qwen2_video_processor_output_keeps_second_per_grid_ts():
    model_inputs = {
        "pixel_values_videos": torch.ones(8, 3),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
        "second_per_grid_ts": torch.tensor([0.5]),
    }
    videos = [
        (
            torch.zeros(4, 3, 16, 16),
            {
                "fps": 30.0,
                "frames_indices": [0, 1, 2, 3],
                "total_num_frames": 4,
                "do_sample_frames": False,
            },
        )
    ]

    mm_data = build_vllm_mm_processor_data(
        processor=DummyQwen2VLProcessor(),
        model_inputs=model_inputs,
        videos=videos,
    )

    assert mm_data is not None
    assert set(mm_data["video"]) == {"pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}
    assert mm_data["video"]["second_per_grid_ts"] is model_inputs["second_per_grid_ts"]


def test_build_image_processor_output():
    model_inputs = {
        "pixel_values": torch.ones(4, 3),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    mm_data = build_vllm_mm_processor_data(
        processor=DummyQwen2VLProcessor(),
        model_inputs=model_inputs,
        images=[object()],
    )

    assert mm_data is not None
    assert set(mm_data) == {"image"}
    assert mm_data["image"]["pixel_values"] is model_inputs["pixel_values"]
    assert mm_data["image"]["image_grid_thw"] is model_inputs["image_grid_thw"]
