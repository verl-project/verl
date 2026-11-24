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
import re

import torch


class ProcessorAdapter:
    def __init__(self, processor):
        self._processor = processor

    def decode_for_processor(self, tokenizer, token_ids: list[int]) -> str:
        """Decode model inputs for another multimodal processor pass."""
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def __getattr__(self, name):
        try:
            processor = object.__getattribute__(self, "_processor")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(processor, name)


class KimiK3Adapter(ProcessorAdapter):
    # Kimi maps this token to ``media_placeholder_token_id`` (163605 in the
    # released config). It has no meaning in a generated response because no
    # image feature accompanies it. Exposing the token lets the generic
    # rollout logits mask resolve and ban it via ``get_processor_token_id``.
    image_token = "<|media_pad|>"

    _MEDIA_PROMPT_PATTERN = re.compile(
        r"<\|media_begin\|>image \d+x\d+<\|media_content\|><\|media_pad\|><\|media_end\|>"
    )
    _VLLM_MEDIA_PROMPT = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"

    def decode_for_processor(self, tokenizer, token_ids: list[int]) -> str:
        """Restore raw image placeholders without dropping Kimi media tokens."""
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        return self._MEDIA_PROMPT_PATTERN.sub("<|kimi_image_placeholder|>", text)

    @staticmethod
    def _validate_media(video_data, audio_data):
        if video_data is not None or audio_data is not None:
            raise ValueError("KimiK3Processor only supports image inputs")

    def __call__(self, *, text, images=None, videos=None, audio=None, **kwargs):
        self._validate_media(videos, audio)
        if isinstance(text, list | tuple):
            if len(text) != 1:
                raise ValueError("KimiK3Processor only supports one text input at a time")
            text = text[0]

        if images:
            medias = [
                image if isinstance(image, dict) and image.get("type") == "image" else {"type": "image", "image": image}
                for image in images
            ]
            # Native checkpoints expose either scalar or batched text APIs.
            text_annotation = inspect.signature(self._processor.__call__).parameters["text"].annotation
            native_text = text if text_annotation in (str, "str") else [text]
            return self._processor(medias=medias, text=native_text, **kwargs)

        return self._processor.tokenizer(text=text, **kwargs)

    @classmethod
    def prepare_vllm_prompt_ids(cls, prompt_ids: list[int], tokenizer, image_data) -> list[int]:
        if not image_data:
            return prompt_ids

        prompt = tokenizer.decode(prompt_ids)
        vllm_prompt = cls._MEDIA_PROMPT_PATTERN.sub(cls._VLLM_MEDIA_PROMPT, prompt)
        if vllm_prompt == prompt:
            return prompt_ids
        from verl.utils.tokenizer import normalize_token_ids

        return normalize_token_ids(tokenizer.encode(vllm_prompt))

    @classmethod
    def build_vllm_multimodal_data(cls, image_data=None, video_data=None, audio_data=None) -> dict:
        cls._validate_media(video_data, audio_data)
        if image_data is None:
            return {}
        return {
            "vision_chunk": [
                image if isinstance(image, dict) and image.get("type") == "image" else {"type": "image", "image": image}
                for image in image_data
            ]
        }



class KimiK31Adapter(KimiK3Adapter):
    """Use the standard image contract of the vLLM 0.30 Kimi backend."""

    _MEDIA_PROMPT_PATTERN = re.compile(
        r"<\|media_begin\|>image \d+x\d+<\|media_content\|>(?:<\|media_pad\|>)+<\|media_end\|>"
    )
    _VLLM_MEDIA_PROMPT = "<|kimi_image_placeholder|>"

    @classmethod
    def build_vllm_multimodal_data(cls, image_data=None, video_data=None, audio_data=None) -> dict:
        cls._validate_media(video_data, audio_data)
        if image_data is None:
            return {}
        return {
            "image": [
                image["image"] if isinstance(image, dict) and image.get("type") == "image" else image
                for image in image_data
            ]
        }


def kimi_k3_logits_to_input_indices(
    input_ids: torch.Tensor,
    multi_modal_inputs: dict,
    config,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Map raw Kimi K3 token positions to positions after image expansion.

    Kimi K3 replaces each ``media_placeholder_token_id`` with all projected
    image features before running the language model.  The returned LM logits
    therefore have the expanded sequence length, whereas verl's labels and
    response masks retain the original one-placeholder sequence.  Selecting
    the first ``raw_sequence_length`` logits shifts every token after an image.

    The index for the placeholder itself is the *last* feature position.  Its
    causal logit is consequently the one that predicts the first text token
    after the image, matching the original-token label shift used by verl.
    """
    if getattr(config, "model_type", None) not in {"kimi_k3", "kimi_k31"}:
        return None
    if input_ids.ndim != 2:
        raise ValueError(f"Kimi K3 logit alignment expects 2-D input_ids, got {tuple(input_ids.shape)}")

    image_token_id = getattr(config, "media_placeholder_token_id", None)
    if image_token_id is None:
        raise ValueError("Kimi K3 config is missing media_placeholder_token_id")
    if padding_mask is not None:
        if padding_mask.shape != input_ids.shape:
            raise ValueError(
                "Kimi K3 padding mask must match input_ids: "
                f"mask={tuple(padding_mask.shape)}, input_ids={tuple(input_ids.shape)}"
            )
        padding_mask = padding_mask.to(device=input_ids.device, dtype=torch.bool)

    image_mask = input_ids == image_token_id
    if padding_mask is not None:
        image_mask &= padding_mask
    image_count = int(image_mask.sum().item())
    if image_count == 0:
        return None

    grid_thws = multi_modal_inputs.get("grid_thws")
    if not isinstance(grid_thws, torch.Tensor):
        raise ValueError("Kimi K3 image inputs require a grid_thws tensor for logit alignment")
    grid_thws = grid_thws.reshape(-1, 3).to(device=input_ids.device, dtype=torch.long)
    if grid_thws.shape[0] != image_count:
        raise ValueError(
            f"Kimi K3 image placeholder/grid count mismatch: placeholders={image_count}, grids={grid_thws.shape[0]}"
        )

    vision_config = getattr(config, "vision_config", None)
    merge_kernel_size = getattr(vision_config, "merge_kernel_size", None)
    merge_type = getattr(vision_config, "merge_type", None)
    if merge_type != "sd2_tpool" or not merge_kernel_size or len(merge_kernel_size) != 2:
        raise ValueError(
            "Unsupported Kimi K3 vision merge configuration for logit alignment: "
            f"merge_type={merge_type!r}, merge_kernel_size={merge_kernel_size!r}"
        )
    kernel_height, kernel_width = (int(value) for value in merge_kernel_size)
    heights, widths = grid_thws[:, 1], grid_thws[:, 2]
    if bool(((heights % kernel_height) != 0).any()) or bool(((widths % kernel_width) != 0).any()):
        raise ValueError("Kimi K3 grid dimensions must be divisible by merge_kernel_size")

    # tpool_patch_merger averages the temporal dimension and spatially merges
    # kernel_height x kernel_width patches, so t is deliberately absent here.
    feature_lengths = (heights // kernel_height) * (widths // kernel_width)
    occupations = torch.ones_like(input_ids, dtype=torch.long)
    occupations[image_mask] = feature_lengths
    if padding_mask is not None:
        occupations[~padding_mask] = 0
    indices = occupations.cumsum(dim=-1) - 1

    # Megatron receives a right-padded dense view of verl's jagged input and
    # passes the same explicit mask to KimiK3Model.build_multimodal_layout.
    # That model always left-aligns the expanded valid tokens, so its mapping is
    # complete here.  The legacy HF/FSDP path below retains the padding-side
    # compatibility logic used by KimiK3ForConditionalGeneration.
    if padding_mask is not None:
        return indices

    # Mirror KimiK3ForConditionalGeneration._merge_input_ids_with_image_features.
    # It left-aligns right-padded batches and right-aligns left/no-padding batches.
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("Kimi K3 config is missing pad_token_id")
    left_padding = not bool((input_ids[:, -1] == pad_token_id).sum().item())
    if left_padding:
        max_embed_dim = occupations.sum(dim=-1).max()
        image_padding = max_embed_dim - 1 - indices[:, -1]
        indices = indices + image_padding[:, None]
    return indices


_PROCESSOR_ADAPTERS = {
    "KimiK3Processor": KimiK3Adapter,
}
_VLLM_ADAPTERS = {
    "kimi_k3": KimiK3Adapter,
    "kimi_k31": KimiK31Adapter,
}


def adapt_processor(processor):
    adapter_cls = _PROCESSOR_ADAPTERS.get(processor.__class__.__name__)
    return processor if adapter_cls is None else adapter_cls(processor)


def get_vllm_adapter(model_type):
    return _VLLM_ADAPTERS.get(model_type)


_KIMI_K3_RESPONSE_CLOSE = "<|close|> response <|sep|>"


def add_kimi_k3_stop_conditions(
    sampling_params: dict,
    model_type: str | None,
    end_of_msg_token_id: int | None,
) -> None:
    """Stop Kimi-K3 after its first complete assistant response."""
    if model_type not in {"kimi_k3", "kimi_k31"} or sampling_params.get("ignore_eos", False):
        return
    if end_of_msg_token_id is None:
        raise ValueError("Kimi K3 config must define eos_token_id")

    stop_token_ids = list(sampling_params.get("stop_token_ids") or [])
    if end_of_msg_token_id not in stop_token_ids:
        stop_token_ids.append(end_of_msg_token_id)
    sampling_params["stop_token_ids"] = stop_token_ids

    stop = sampling_params.get("stop")
    stop_strings = [stop] if isinstance(stop, str) else list(stop or [])
    if _KIMI_K3_RESPONSE_CLOSE not in stop_strings:
        stop_strings.append(_KIMI_K3_RESPONSE_CLOSE)
    sampling_params["stop"] = stop_strings
