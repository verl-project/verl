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

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch

_QWEN3_TIMESTAMP_MODEL_TYPES = {
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}

# HF Qwen3 VL processors (Qwen3VLProcessor / Qwen3VLVideoProcessor / Qwen3_5*) do NOT expose
# `model_type` or `.config` attributes, so the class name is the only reliable identifier in
# practice. The `_QWEN3_TIMESTAMP_MODEL_TYPES` set is kept as a defense for processors that do
# expose a config (e.g. wrappers, dummies in tests).
_QWEN3_TIMESTAMP_CLASS_MARKERS = (
    "Qwen3VL",
    "Qwen3VLMoe",
    "Qwen3_5",
)


def _is_qwen3_timestamp_processor(processor: Any) -> bool:
    for obj in (
        processor,
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
        getattr(processor, "config", None),
    ):
        if obj is None:
            continue
        cls_name = obj.__class__.__name__
        if any(marker in cls_name for marker in _QWEN3_TIMESTAMP_CLASS_MARKERS):
            return True
        config = getattr(obj, "config", obj)
        if str(getattr(config, "model_type", "") or "").lower() in _QWEN3_TIMESTAMP_MODEL_TYPES:
            return True
    return False


def _extract_video_metadata(videos: Sequence[Any]) -> list[dict[str, Any]]:
    metadata = []
    for idx, item in enumerate(videos):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"video item {idx} must be a (tensor, metadata) tuple, got {type(item).__name__}")
        metadata.append(dict(item[1]))
    return metadata


def calculate_qwen3_timestamps(
    *,
    processor: Any,
    metadata: Mapping[str, Any],
    do_sample_frames: Optional[bool] = None,
) -> list[float]:
    """Match vLLM Qwen3/Qwen3.5 VL timestamp calculation."""
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        raise ValueError("Qwen3-style VL timestamp calculation requires processor.video_processor")

    temporal_patch_size = int(video_processor.temporal_patch_size)
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

    # Qwen3-VL groups frames into temporal patches of size `temporal_patch_size` and emits one
    # timestamp per patch (the patch midpoint, computed below). Pad with the final frame so the
    # frame count is a multiple of the patch size; otherwise the patch loop drops a tail patch.
    if len(indices) % temporal_patch_size != 0:
        pad_count = temporal_patch_size - len(indices) % temporal_patch_size
        indices.extend([indices[-1]] * pad_count)

    seconds = [idx / video_fps for idx in indices]
    return [
        (seconds[idx] + seconds[idx + temporal_patch_size - 1]) / 2
        for idx in range(0, len(seconds), temporal_patch_size)
    ]


def _normalize_timestamps(timestamps: Any, *, num_videos: int) -> list[list[float]]:
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.tolist()
    if num_videos == 1 and timestamps and isinstance(timestamps[0], (int, float)):
        return [timestamps]
    return timestamps


def _validate_timestamps(timestamps: list[list[float]], video_grid_thw: torch.Tensor) -> None:
    if len(timestamps) != int(video_grid_thw.shape[0]):
        raise ValueError(
            "Qwen3-VL timestamps count must match video_grid_thw batch size: "
            f"got {len(timestamps)} timestamps entries for {int(video_grid_thw.shape[0])} videos"
        )
    for idx, per_video in enumerate(timestamps):
        grid_t = int(video_grid_thw[idx, 0])
        if len(per_video) != grid_t:
            raise ValueError(
                "Qwen3-VL timestamps length must match video_grid_thw[:, 0]: "
                f"video {idx} has {len(per_video)} timestamps but grid_t={grid_t}"
            )


def _get_tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and hasattr(processor, "encode"):
        tokenizer = processor
    if tokenizer is None:
        raise ValueError("A tokenizer is required to build direct multimodal placeholders")
    return tokenizer


def _token_id(processor: Any, tokenizer: Any, attr_name: str, token: str) -> int:
    value = getattr(processor, attr_name, None)
    if value is not None:
        return int(value)
    value = getattr(tokenizer, attr_name, None)
    if value is not None:
        return int(value)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        value = tokenizer.convert_tokens_to_ids(token)
        if value is not None:
            return int(value)
    raise ValueError(f"Unable to resolve token id for {token!r}")


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _merge_length(processor: Any) -> int:
    for candidate in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
        processor,
    ):
        if candidate is None:
            continue
        merge_size = getattr(candidate, "merge_size", None)
        if merge_size is None:
            merge_size = getattr(candidate, "spatial_merge_size", None)
        if merge_size is not None:
            return int(merge_size) ** 2
    raise ValueError("Unable to resolve Qwen VL merge_size from processor")


def _expected_token_counts(grid_thw: torch.Tensor, *, merge_length: int) -> list[int]:
    return (grid_thw.prod(-1) // merge_length).tolist()


def _find_token_runs(prompt_ids: Sequence[int], token_id: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < len(prompt_ids):
        if prompt_ids[idx] != token_id:
            idx += 1
            continue
        start = idx
        while idx < len(prompt_ids) and prompt_ids[idx] == token_id:
            idx += 1
        runs.append((start, idx - start))
    return runs


def _find_subsequence(prompt_ids: Sequence[int], target: Sequence[int], *, start: int) -> int:
    if not target:
        raise ValueError("Cannot locate an empty multimodal placeholder")

    target_list = list(target)
    target_len = len(target_list)
    first = target_list[0]
    max_start = len(prompt_ids) - target_len
    for idx in range(start, max_start + 1):
        if prompt_ids[idx] == first and list(prompt_ids[idx : idx + target_len]) == target_list:
            return idx
    raise ValueError("Unable to locate a preprocessed multimodal placeholder in prompt ids")


def _build_token_run_placeholders(
    *,
    prompt_ids: Sequence[int],
    token_id: int,
    expected_lengths: Sequence[int],
    placeholder_cls: Any,
    modality: str,
) -> list[Any]:
    runs = _find_token_runs(prompt_ids, token_id)
    if len(runs) != len(expected_lengths):
        raise ValueError(f"Expected {len(expected_lengths)} {modality} placeholders, but found {len(runs)} token runs")

    placeholders = []
    for item_idx, ((offset, length), expected_length) in enumerate(zip(runs, expected_lengths, strict=True)):
        if int(length) != int(expected_length):
            raise ValueError(f"{modality} placeholder {item_idx} has length {length}, but expected {expected_length}")
        placeholders.append(placeholder_cls(offset=offset, length=length))
    return placeholders


def _qwen3_video_replacement(
    *,
    tokenizer: Any,
    timestamps: Sequence[float],
    tokens_per_frame: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    video_token_id: int,
) -> list[int]:
    token_ids: list[int] = []
    for timestamp in timestamps:
        # Keep this aligned with vLLM's Qwen3VLMultiModalProcessor.get_video_repl.
        token_ids.extend(_encode(tokenizer, f"<{timestamp:.1f} seconds>"))
        token_ids.append(vision_start_token_id)
        token_ids.extend([video_token_id] * tokens_per_frame)
        token_ids.append(vision_end_token_id)
    return token_ids


def _build_qwen3_video_placeholders(
    *,
    prompt_ids: Sequence[int],
    processor: Any,
    video_grid_thw: torch.Tensor,
    timestamps: Sequence[Sequence[float]],
    placeholder_cls: Any,
    merge_length: int,
) -> list[Any]:
    tokenizer = _get_tokenizer(processor)
    vision_start_token_id = _token_id(processor, tokenizer, "vision_start_token_id", "<|vision_start|>")
    vision_end_token_id = _token_id(processor, tokenizer, "vision_end_token_id", "<|vision_end|>")
    video_token_id = _token_id(processor, tokenizer, "video_token_id", "<|video_pad|>")

    placeholders = []
    start = 0
    for item_idx, per_video_timestamps in enumerate(timestamps):
        grid = video_grid_thw[item_idx]
        tokens_per_frame = int(grid[1:].prod()) // merge_length
        replacement = _qwen3_video_replacement(
            tokenizer=tokenizer,
            timestamps=per_video_timestamps,
            tokens_per_frame=tokens_per_frame,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            video_token_id=video_token_id,
        )
        offset = _find_subsequence(prompt_ids, replacement, start=start)
        # is_embed marks positions consumed by video patch embeddings vs. surrounding text-only
        # tokens (vision_start/end and "<seconds>" timestamp text). vLLM only routes embed-mask
        # positions to the vision encoder; getting this wrong corrupts the prompt.
        is_embed = torch.tensor([token_id == video_token_id for token_id in replacement], dtype=torch.bool)
        placeholders.append(
            placeholder_cls(
                offset=offset,
                length=len(replacement),
                is_embed=is_embed,
            )
        )
        start = offset + len(replacement)
    return placeholders


def _tensor_hash(hasher: Any, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous()
    hasher.update(str(tuple(tensor.shape)).encode())
    hasher.update(str(tensor.dtype).encode())
    byte_tensor = tensor.reshape(1).view(torch.uint8) if tensor.ndim == 0 else tensor.view(torch.uint8)
    hasher.update(byte_tensor.numpy().tobytes())


def _hash_value(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _tensor_hash(hasher, value)
    elif isinstance(value, Mapping):
        for key in sorted(value):
            hasher.update(str(key).encode())
            _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(str(len(value)).encode())
        for item in value:
            _hash_value(hasher, item)
    else:
        hasher.update(repr(value).encode())


def _hash_mm_item(item: Any) -> str:
    # Must use SHA-256 to match vLLM's mm prefix-cache hash function. Switching to a different
    # algorithm silently breaks cache hits without raising — observed only as a perf regression.
    hasher = hashlib.sha256()
    _hash_value(hasher, item.get_data())
    return hasher.hexdigest()


def _build_hashes(mm_kwargs: Any) -> dict[str, list[str]]:
    return {modality: [_hash_mm_item(item) for item in items] for modality, items in mm_kwargs.items()}


def _build_vllm_mm_kwargs(
    inputs: Mapping[str, Any],
    *,
    has_images: bool,
    has_videos: bool,
    merge_length: int,
) -> Any:
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems

    config_by_key = {}
    if has_images:
        image_grid_thw = inputs["image_grid_thw"]
        image_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = image_grid_sizes // merge_length
        if "pixel_values" in inputs:
            config_by_key["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes)
        if "image_embeds" in inputs:
            config_by_key["image_embeds"] = MultiModalFieldConfig.flat_from_sizes("image", image_embed_grid_sizes)
        config_by_key["image_grid_thw"] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)

    if has_videos:
        video_grid_thw = inputs["video_grid_thw"]
        video_grid_sizes = video_grid_thw.prod(-1)
        video_embed_grid_sizes = video_grid_sizes // merge_length
        if "pixel_values_videos" in inputs:
            config_by_key["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes)
        if "video_embeds" in inputs:
            config_by_key["video_embeds"] = MultiModalFieldConfig.flat_from_sizes("video", video_embed_grid_sizes)
        config_by_key["video_grid_thw"] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
        if "second_per_grid_ts" in inputs:
            config_by_key["second_per_grid_ts"] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)
        if "timestamps" in inputs:
            config_by_key["timestamps"] = MultiModalFieldConfig.batched("video", keep_on_cpu=True)

    return MultiModalKwargsItems.from_hf_inputs(inputs, config_by_key)


def build_vllm_preprocessed_multimodal_input(
    *,
    prompt_ids: Sequence[int],
    processor: Any,
    model_inputs: Any,
    images: Any = None,
    videos: Any = None,
) -> Any | None:
    """Construct a vLLM ``MultiModalInput`` payload from HF processor outputs.

    vLLM normally re-runs the HF processor inside the engine. This bypass lets verl reuse
    the processor outputs already computed during chat-template application, avoiding the
    redundant pass per turn in multi-turn rollouts. Returns ``None`` when there is no
    multimodal payload, so callers can fall back to the legacy raw-media path.
    """
    from vllm.inputs import mm_input
    from vllm.multimodal.inputs import PlaceholderRange

    inputs: dict[str, Any] = dict(model_inputs) if model_inputs else {}
    has_images = (
        images is not None and "image_grid_thw" in inputs and ("pixel_values" in inputs or "image_embeds" in inputs)
    )
    has_videos = (
        videos is not None
        and "video_grid_thw" in inputs
        and ("pixel_values_videos" in inputs or "video_embeds" in inputs)
    )
    if not has_images and not has_videos:
        return None

    merge_length = _merge_length(processor)
    tokenizer = _get_tokenizer(processor)
    is_qwen3_ts = _is_qwen3_timestamp_processor(processor)

    if has_videos:
        timestamps = inputs.get("timestamps")
        if timestamps is None and is_qwen3_ts:
            video_metadata = _extract_video_metadata(videos)
            timestamps = [
                calculate_qwen3_timestamps(
                    processor=processor,
                    metadata=metadata,
                    do_sample_frames=metadata.get("do_sample_frames", False),
                )
                for metadata in video_metadata
            ]

        if timestamps is not None:
            timestamps = _normalize_timestamps(timestamps, num_videos=int(inputs["video_grid_thw"].shape[0]))
            _validate_timestamps(timestamps, inputs["video_grid_thw"])
            inputs["timestamps"] = timestamps

    mm_kwargs = _build_vllm_mm_kwargs(
        inputs,
        has_images=has_images,
        has_videos=has_videos,
        merge_length=merge_length,
    )

    mm_placeholders: dict[str, list[Any]] = {}
    if has_images:
        image_token_id = _token_id(processor, tokenizer, "image_token_id", "<|image_pad|>")
        image_lengths = _expected_token_counts(inputs["image_grid_thw"], merge_length=merge_length)
        mm_placeholders["image"] = _build_token_run_placeholders(
            prompt_ids=prompt_ids,
            token_id=image_token_id,
            expected_lengths=image_lengths,
            placeholder_cls=PlaceholderRange,
            modality="image",
        )

    if has_videos:
        if is_qwen3_ts and "timestamps" in inputs:
            mm_placeholders["video"] = _build_qwen3_video_placeholders(
                prompt_ids=prompt_ids,
                processor=processor,
                video_grid_thw=inputs["video_grid_thw"],
                timestamps=inputs["timestamps"],
                placeholder_cls=PlaceholderRange,
                merge_length=merge_length,
            )
        else:
            video_token_id = _token_id(processor, tokenizer, "video_token_id", "<|video_pad|>")
            video_lengths = _expected_token_counts(inputs["video_grid_thw"], merge_length=merge_length)
            mm_placeholders["video"] = _build_token_run_placeholders(
                prompt_ids=prompt_ids,
                token_id=video_token_id,
                expected_lengths=video_lengths,
                placeholder_cls=PlaceholderRange,
                modality="video",
            )

    return mm_input(
        prompt_token_ids=list(prompt_ids),
        mm_kwargs=mm_kwargs,
        mm_hashes=_build_hashes(mm_kwargs),
        mm_placeholders=mm_placeholders,
    )


def refresh_vllm_preprocessed_multimodal_prompt_ids(
    preprocessed_multimodal_input: Any | None,
    *,
    prompt_ids: Sequence[int],
) -> Any | None:
    """Update ``prompt_token_ids`` in place between turns. mm_kwargs/hashes/placeholders are
    keyed by position+modality and remain valid as long as the prompt structure is preserved."""
    if preprocessed_multimodal_input is None:
        return None
    preprocessed_multimodal_input["prompt_token_ids"] = list(prompt_ids)
    return preprocessed_multimodal_input
