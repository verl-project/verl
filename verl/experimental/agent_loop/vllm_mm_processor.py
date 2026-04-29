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
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch


def _class_name(obj: Any) -> str:
    return obj.__class__.__name__ if obj is not None else ""


def _is_qwen3_vl_processor(processor: Any) -> bool:
    return any(
        "Qwen3VL" in _class_name(obj)
        for obj in (
            processor,
            getattr(processor, "image_processor", None),
            getattr(processor, "video_processor", None),
        )
    )


def _as_mapping(model_inputs: Any) -> Mapping[str, Any]:
    if model_inputs is None:
        return {}
    return dict(model_inputs)


def _extract_video_metadata(videos: Any) -> Optional[list[dict[str, Any]]]:
    if videos is None:
        return None

    metadata = []
    for item in videos:
        if not isinstance(item, tuple) or len(item) != 2:
            return None
        metadata.append(dict(item[1]))
    return metadata


def calculate_qwen3_timestamps(
    *,
    processor: Any,
    metadata: Mapping[str, Any],
    do_sample_frames: Optional[bool] = None,
) -> list[float]:
    """Match vLLM Qwen3-VL timestamp calculation for pre-sampled videos."""
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        raise ValueError("Qwen3-VL timestamp calculation requires processor.video_processor")

    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size"))
    indices = list(metadata["frames_indices"])
    video_fps = float(metadata["fps"])

    if do_sample_frames is None:
        do_sample_frames = bool(metadata.get("do_sample_frames", False))

    if do_sample_frames:
        total_num_frames = int(metadata["total_num_frames"])
        sampled_fps = getattr(video_processor, "fps", None)
        if sampled_fps is None:
            raise ValueError("sampled fps is required when do_sample_frames=True")

        num_frames = int(total_num_frames / video_fps * float(sampled_fps))
        min_frames = getattr(video_processor, "min_frames", None)
        max_frames = getattr(video_processor, "max_frames", None)
        if min_frames is not None:
            num_frames = max(num_frames, int(min_frames))
        if max_frames is not None:
            num_frames = min(num_frames, int(max_frames))
        num_frames = min(num_frames, total_num_frames)
        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int).tolist()

    if len(indices) % temporal_patch_size != 0:
        pad_count = temporal_patch_size - len(indices) % temporal_patch_size
        indices.extend([indices[-1]] * pad_count)

    seconds = [idx / video_fps for idx in indices]
    return [
        (seconds[idx] + seconds[idx + temporal_patch_size - 1]) / 2
        for idx in range(0, len(seconds), temporal_patch_size)
    ]


def _normalize_timestamps(timestamps: Any, *, num_videos: int) -> Any:
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.tolist()
    if num_videos == 1 and timestamps and isinstance(timestamps[0], (int, float)):
        return [timestamps]
    return timestamps


def _validate_timestamps(timestamps: list[list[float]], video_grid_thw: torch.Tensor) -> None:
    for idx, per_video in enumerate(timestamps):
        grid_t = int(video_grid_thw[idx, 0])
        if len(per_video) != grid_t:
            raise ValueError(
                "Qwen3-VL timestamps length must match video_grid_thw[:, 0]: "
                f"video {idx} has {len(per_video)} timestamps but grid_t={grid_t}"
            )


def build_vllm_mm_processor_data(
    *,
    processor: Any,
    model_inputs: Any,
    images: Any = None,
    videos: Any = None,
) -> dict[str, Any] | None:
    """Build vLLM multi_modal_data from HF processor outputs.

    The returned dict is meant for vLLM branches that accept processor-output
    dictionaries as multimodal payloads. Raw media stays owned by the caller for
    training-side post-processing.
    """
    inputs = _as_mapping(model_inputs)
    multi_modal_data: dict[str, Any] = {}

    if images is not None and "pixel_values" in inputs and "image_grid_thw" in inputs:
        multi_modal_data["image"] = {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    if videos is not None and "pixel_values_videos" in inputs and "video_grid_thw" in inputs:
        video_data = {
            "pixel_values_videos": inputs["pixel_values_videos"],
            "video_grid_thw": inputs["video_grid_thw"],
        }
        if "second_per_grid_ts" in inputs:
            video_data["second_per_grid_ts"] = inputs["second_per_grid_ts"]

        timestamps = inputs.get("timestamps")
        video_metadata = _extract_video_metadata(videos)
        if timestamps is None and _is_qwen3_vl_processor(processor):
            if video_metadata is None:
                raise ValueError("Qwen3-VL processor-output video requires video metadata for timestamps")
            timestamps = [
                calculate_qwen3_timestamps(
                    processor=processor,
                    metadata=metadata,
                    do_sample_frames=metadata.get("do_sample_frames", False),
                )
                for metadata in video_metadata
            ]

        if timestamps is not None:
            timestamps = _normalize_timestamps(timestamps, num_videos=int(video_data["video_grid_thw"].shape[0]))
            _validate_timestamps(timestamps, video_data["video_grid_thw"])
            video_data["timestamps"] = timestamps

        multi_modal_data["video"] = video_data

    return multi_modal_data or None
