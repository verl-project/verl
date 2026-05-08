from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask

from .types import Trajectory

# Behavior mirrors legacy `_compute_multi_modal_inputs` / `_compute_position_ids`
# in `verl/experimental/agent_loop/agent_loop.py`.


def _split_videos_and_metadata(videos: list[Any] | None) -> tuple[list[Any] | None, list[Any] | None]:
    if not videos:
        return videos, None

    first_video = videos[0]
    if isinstance(first_video, tuple) and len(first_video) == 2:
        split_videos, video_metadata = zip(*videos, strict=False)
        return list(split_videos), list(video_metadata)

    return list(videos), None


def _to_plain_tensor_dict(processor_output) -> dict[str, torch.Tensor]:
    if hasattr(processor_output, "convert_to_tensors"):
        processor_output = processor_output.convert_to_tensors("pt")
    return dict(processor_output)


def compute_multi_modal_inputs(
    processor,
    input_ids: torch.Tensor,
    multi_modal_data: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    """Return processor-produced multimodal tensors for a single sample."""
    if processor is None or not multi_modal_data:
        return {}

    images = multi_modal_data.get("images")
    videos, video_metadata = _split_videos_and_metadata(multi_modal_data.get("videos"))
    current_text = processor.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
    multi_modal_inputs = _to_plain_tensor_dict(
        processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_sample_frames=False,
        )
    )
    multi_modal_inputs.pop("input_ids", None)
    multi_modal_inputs.pop("attention_mask", None)

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
        multi_modal_inputs["images_seqlens"] = images_seqlens
    return multi_modal_inputs


def compute_position_ids(
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return text-only or multimodal-aware position ids for a single sample."""
    if processor is None:
        return compute_position_id_with_mask(attention_mask)

    multi_modal_kwargs = {
        "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
        "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
    }
    if multi_modal_inputs.get("mm_token_type_ids") is not None:
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[0][input_ids[0] == processor.image_token_id] = 1
        mm_token_type_ids[0][input_ids[0] == processor.video_token_id] = 2
        multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

    vision_position_ids, _ = processor.get_rope_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **multi_modal_kwargs,
    )
    vision_position_ids = vision_position_ids.transpose(0, 1)

    valid_mask = attention_mask[0].bool()
    text_position_ids = torch.ones((1, input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)
    text_position_ids = text_position_ids.unsqueeze(0)
    return torch.cat((text_position_ids, vision_position_ids), dim=1)


def apply_multi_modal_postprocess(
    batch: TensorDict,
    trajectories: Sequence[Trajectory],
    processor,
) -> TensorDict:
    """Apply multimodal helper logic per sample and attach non-tensor outputs."""
    if len(batch) != len(trajectories):
        raise ValueError(f"batch/trajectory size mismatch: {len(batch)} != {len(trajectories)}")

    result = batch.copy()
    position_ids = []
    multi_modal_data = []
    multi_modal_inputs = []

    for sample_index, trajectory in enumerate(trajectories):
        sample_input_ids = result["input_ids"][sample_index : sample_index + 1]
        sample_attention_mask = result["attention_mask"][sample_index : sample_index + 1]
        sample_multi_modal_data = trajectory.multi_modal_data
        sample_multi_modal_inputs = compute_multi_modal_inputs(
            processor,
            sample_input_ids,
            sample_multi_modal_data,
        )
        position_ids.append(
            compute_position_ids(
                processor,
                sample_input_ids,
                sample_attention_mask,
                sample_multi_modal_inputs,
            )
        )
        multi_modal_data.append(sample_multi_modal_data)
        multi_modal_inputs.append(sample_multi_modal_inputs)

    result["position_ids"] = torch.cat(position_ids, dim=0)
    tu.assign_non_tensor(
        result,
        multi_modal_data=multi_modal_data,
        multi_modal_inputs=multi_modal_inputs,
    )
    return result
