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
"""Which pixels a Megatron-Bridge Qwen-VL rank reads.

Only the first pipeline stage runs the ViT. With Qwen3-VL ``vision_dp_when_cp`` each CP rank of that
stage runs it on a share of the micro-batch's frames and the embeddings are all-gathered, yet the
model still takes the pixels of every frame and slices its own share. :func:`plan_cp_vision_rows`
computes that share so the rank fetches only those rows, and inside :func:`cp_local_vision_inputs`
the model's split accepts them as already local.
"""

import contextlib
import contextvars

import torch

_CP_LOCAL_VISION = contextvars.ContextVar("verl_cp_local_vision", default=False)
_PATCHED = False


def _cp_frame_counts(num_frames: int, cp_size: int) -> list[int]:
    per_rank, remain = divmod(num_frames, cp_size)
    return [per_rank + (1 if rank < remain else 0) for rank in range(cp_size)]


def plan_cp_vision_rows(grids: list[dict], cp_rank: int, cp_size: int) -> list[list[list]] | None:
    """Pixel rows of each sample of a micro-batch that ``cp_rank`` encodes.

    The frames of the micro-batch, in sample order, are split by count over the CP ranks, the first
    ``num_frames % cp_size`` ranks taking one more. ``grids`` holds each sample's ``multi_modal_inputs``
    (only ``image_grid_thw`` / ``video_grid_thw`` are read). Returns ``[key, start, end]`` triples per
    sample, or None when the micro-batch mixes images and videos: the model then orders them by their
    tokens in the full sequence, so the rank needs all of them.
    """
    has_images = any(g is not None and g.get("image_grid_thw") is not None for g in grids)
    has_videos = any(g is not None and g.get("video_grid_thw") is not None for g in grids)
    if has_images and has_videos:
        return None
    plan: list[list[list]] = [[] for _ in grids]
    if not has_images and not has_videos:
        return plan
    key, grid_key = ("pixel_values", "image_grid_thw") if has_images else ("pixel_values_videos", "video_grid_thw")

    frame_rows = []  # (sample, pixel rows of the frame)
    for sample, g in enumerate(grids):
        grid = None if g is None else g.get(grid_key)
        if grid is not None:
            for t, h, w in grid.reshape(-1, 3).tolist():
                frame_rows.extend([(sample, h * w)] * t)

    counts = _cp_frame_counts(len(frame_rows), cp_size)
    first = sum(counts[:cp_rank])
    last = first + counts[cp_rank]
    offsets = [0] * len(grids)
    for frame, (sample, rows) in enumerate(frame_rows):
        if first <= frame < last:
            if plan[sample]:
                plan[sample][0][2] += rows
            else:
                plan[sample].append([key, offsets[sample], offsets[sample] + rows])
        offsets[sample] += rows
    return plan


@contextlib.contextmanager
def cp_local_vision_inputs(enabled: bool = True):
    """Within this context the pixels given to a patched Qwen3-VL are this CP rank's share only."""
    token = _CP_LOCAL_VISION.set(enabled)
    try:
        yield
    finally:
        _CP_LOCAL_VISION.reset(token)


def _local_cp_split(cp_size: int, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
    """``qwen3vl_cp_split`` for rank-local pixels: only the grid is split into frames and counted per rank."""
    if pixel_values is None:
        return None, None, None, None
    frames = image_grid_thw.repeat_interleave(image_grid_thw[:, 0].to(torch.long), dim=0)
    frames[:, 0] = 1
    cp_img_num = _cp_frame_counts(frames.shape[0], cp_size)
    return pixel_values, frames, cp_img_num, [0] * cp_size


def _local_vision_cp_data(vision_data, vision_grid_thw, square_merge_size, cp_img_num, images_padded, cp_rank, cp_size):
    """``get_vision_cp_data`` for rank-local pixels: ``vision_data`` already is this rank's share."""
    assert cp_size == len(cp_img_num)
    assert not any(images_padded), "rank-local Qwen3-VL pixels are never padded"
    seqlens = torch.repeat_interleave(vision_grid_thw[:, 1] * vision_grid_thw[:, 2], vision_grid_thw[:, 0])
    bounds = [0]
    for count in cp_img_num:
        bounds.append(bounds[-1] + count)
    start, end = bounds[cp_rank], bounds[cp_rank + 1]
    expected_rows = int(seqlens[start:end].sum())
    if vision_data.shape[0] != expected_rows:
        raise RuntimeError(
            f"CP rank {cp_rank} got {vision_data.shape[0]} pixel rows, "
            f"but its {end - start} frames have {expected_rows}"
        )
    seqlens_list = [seqlens[bounds[i] : bounds[i + 1]] // square_merge_size for i in range(cp_size)]
    return vision_data, vision_grid_thw[start:end], seqlens_list


def _patch_qwen3_vl_model() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen3_vl_model
    except ImportError:
        return False
    if not hasattr(qwen3_vl_model, "qwen3vl_cp_split") or not hasattr(qwen3_vl_model, "get_vision_cp_data"):
        return False

    cp_split, vision_cp_data = qwen3_vl_model.qwen3vl_cp_split, qwen3_vl_model.get_vision_cp_data

    def qwen3vl_cp_split(*args, **kwargs):
        return (_local_cp_split if _CP_LOCAL_VISION.get() else cp_split)(*args, **kwargs)

    def get_vision_cp_data(*args, **kwargs):
        return (_local_vision_cp_data if _CP_LOCAL_VISION.get() else vision_cp_data)(*args, **kwargs)

    qwen3_vl_model.qwen3vl_cp_split = qwen3vl_cp_split
    qwen3_vl_model.get_vision_cp_data = get_vision_cp_data
    _PATCHED = True
    return True


def is_qwen_vl_model(model) -> bool:
    """Whether ``model`` (unwrapped) is a Megatron-Bridge Qwen-VL, whose ViT runs on the first PP stage only."""
    model_classes = []
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

        model_classes.append(Qwen3VLModel)
    except ImportError:
        pass
    try:
        from megatron.bridge.models.qwen_vl.modeling_qwen25_vl import Qwen25VLModel

        model_classes.append(Qwen25VLModel)
    except ImportError:
        pass
    return bool(model_classes) and isinstance(model, tuple(model_classes))


def enable_cp_local_vision(model) -> bool:
    """Whether ``model`` (unwrapped) encodes a CP share of the images, patching it to accept only that share."""
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    except ImportError:
        return False
    if not isinstance(model, Qwen3VLModel) or not getattr(model.config, "vision_dp_when_cp", False):
        return False
    return _patch_qwen3_vl_model()
