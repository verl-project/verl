# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect
from functools import cache
from io import BytesIO
from typing import Any, Callable, Optional

import torch
from PIL import Image


@cache
def _signature_params(func: Callable[..., Any]) -> frozenset[str]:
    """Cached parameter set for ``func``.

    ``qwen-vl-utils`` has shipped multiple incompatible signatures
    (``image_patch_size`` and ``return_video_metadata`` were added in v0.0.13,
    and some internal forks still lag behind). Callers use this to forward only
    kwargs the installed function actually accepts.
    """
    try:
        return frozenset(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _select_supported(func: Callable[..., Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of ``candidate`` whose keys are in ``func``'s signature."""
    supported = _signature_params(func)
    return {k: v for k, v in candidate.items() if k in supported}


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    from qwen_vl_utils import fetch_image

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    return fetch_image(image, **_select_supported(fetch_image, {"image_patch_size": image_patch_size}))


def safe_process_vision_info(
    messages: list[dict],
    *,
    image_patch_size: int = 14,
    return_video_metadata: bool = False,
) -> tuple[Optional[list[Image.Image]], Optional[list[Any]]]:
    """Backward-compatible wrapper around ``qwen_vl_utils.process_vision_info``.

    Forwards only the kwargs the installed function signature accepts so verl
    keeps working across upstream releases and any internal qwen-vl-utils fork
    that hasn't been resynced. When ``return_video_metadata`` is not supported,
    videos come back as bare tensors; ``build_multimodal_processor_inputs``
    handles both shapes and still pins ``do_sample_frames=False`` as a guardrail.
    """
    from qwen_vl_utils import process_vision_info

    candidate: dict[str, Any] = {"image_patch_size": image_patch_size}
    if return_video_metadata:
        candidate["return_video_metadata"] = True
    return process_vision_info(messages, **_select_supported(process_vision_info, candidate))


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> torch.Tensor:
    """Converts a video dict into a ``[n_frames, 3, H, W]`` tensor.

    Forwards the v0.0.13+ kwargs (``image_patch_size``,
    ``return_video_sample_fps``, ``return_video_metadata``) only when the
    installed ``fetch_video`` signature accepts them, so verl stays compatible
    with older qwen-vl-utils variants that don't expose them.
    """
    from qwen_vl_utils import fetch_video

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    candidate = {
        "image_patch_size": image_patch_size,
        "return_video_sample_fps": return_video_sample_fps,
        "return_video_metadata": return_video_metadata,
    }
    return fetch_video(video, **_select_supported(fetch_video, candidate))


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}
