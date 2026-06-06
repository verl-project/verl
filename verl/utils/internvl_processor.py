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

"""Minimal processor wrapper for InternVL2_5 models.

InternVL does not ship a standard HuggingFace multimodal processor —
:class:`AutoProcessor` falls back to the tokenizer. This wrapper provides
the interface that verl's agent loop expects, including prompt expansion
from ``<image>`` to ``<IMG_CONTEXT>`` tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.image_processing_utils import ImageProcessingMixin

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class _FakeImageProcessor(ImageProcessingMixin):
    """Minimal image_processor stub — actual pixel encoding is done by vLLM."""

    patch_size: int = 14  # InternVL vision encoder uses 14x14 patches
    image_mean: list[float] = [0.485, 0.456, 0.406]
    image_std: list[float] = [0.229, 0.224, 0.225]


class InternVLProcessor(ProcessorMixin):
    """Minimal processor for InternVL2_5 that wraps the tokenizer.

    Provides the interface verl's :class:`AgentLoopBase` expects:

    * :meth:`apply_chat_template` – delegates to tokenizer
    * :meth:`__call__` – tokenizes text + preprocesses images into ``pixel_values``
      and ``image_flags``
    * :attr:`image_processor` – stub with ``patch_size``
    * :attr:`image_token_id` – ``<img>`` token id
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer: PreTrainedTokenizer, config):
        image_processor = _FakeImageProcessor()
        super().__init__(image_processor, tokenizer)
        self._config = config
        self.image_processor = image_processor
        # InternVL config
        self._image_size = getattr(config, "force_image_size", 448)
        self._min_dynamic_patch = getattr(config, "min_dynamic_patch", 1)
        self._max_dynamic_patch = getattr(config, "max_dynamic_patch", 12)
        self._use_thumbnail = getattr(config, "use_thumbnail", True)
        # Number of vision tokens per tile after pixel_shuffle
        patch_size = config.vision_config.patch_size
        downsample_ratio = config.downsample_ratio
        self._num_image_token = int((self._image_size // patch_size) ** 2 * (downsample_ratio**2))

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<img>")

    @property
    def image_token(self) -> str:
        return "<img>"

    @property
    def video_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<video>")

    @property
    def video_token(self) -> str:
        return "<video>"

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(
        self,
        text: list[str] | None = None,
        images: list[Image.Image] | None = None,
        videos=None,
        video_metadata=None,
        return_tensors: str = "pt",
        do_sample_frames: bool = False,
        replace_image: bool = False,
        **kwargs,
    ):
        """Tokenize text + preprocess images.

        When ``replace_image=False`` (default): keeps ``<image>`` as-is in
        text.  vLLM handles ``<image>`` → ``<IMG_CONTEXT>`` expansion
        internally via its HF processor.

        When ``replace_image=True``: replaces ``<image>`` with
        ``<IMG_CONTEXT>`` tokens before tokenisation (used for the model
        training forward pass).
        """
        from transformers import BatchFeature
        from vllm.model_executor.models.internvl import image_to_pixel_values_internvl

        # Phase 1: preprocess images to know num_patches per image.
        all_pixel_values = []
        patches_per_image = []

        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            for img in images:
                if isinstance(img, Image.Image):
                    pv = image_to_pixel_values_internvl(
                        img,
                        input_size=self._image_size,
                        min_num=self._min_dynamic_patch,
                        max_num=self._max_dynamic_patch,
                        use_thumbnail=self._use_thumbnail,
                    )
                    all_pixel_values.append(pv)
                    patches_per_image.append(pv.shape[0])
                else:
                    raise TypeError(f"Unsupported image type: {type(img)}")

        # Phase 2: tokenize text, optionally expanding <image> placeholders.
        if text is not None:
            if replace_image:
                modified_text = []
                image_idx = 0
                for t in text:
                    while "<image>" in t and image_idx < len(patches_per_image):
                        num_patches = patches_per_image[image_idx]
                        feature_size = num_patches * self._num_image_token
                        replacement = f"<img>{'<IMG_CONTEXT>' * feature_size}</img>"
                        t = t.replace("<image>", replacement, 1)
                        image_idx += 1
                    modified_text.append(t)
                tokenized = self.tokenizer(modified_text, return_tensors=return_tensors, **kwargs)
            else:
                tokenized = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        else:
            tokenized = {}

        result: dict = {}
        result.update(tokenized)

        if all_pixel_values:
            result["pixel_values"] = torch.cat(all_pixel_values, dim=0)

        # image_flags: one flag per tile group in pixel_values.
        if "pixel_values" in result:
            num_tile_groups = result["pixel_values"].shape[0]
            result["image_flags"] = torch.ones(num_tile_groups, dtype=torch.int32).unsqueeze(-1)
        elif text is not None:
            result["image_flags"] = torch.zeros(len(text), dtype=torch.int32).unsqueeze(-1)

        return BatchFeature(result, tensor_type=return_tensors)

    def expand_prompt_ids(self, prompt_ids: list[int], num_tiles: int) -> list[int]:
        """Replace ``<image>`` with ``<img><IMG_CONTEXT>*N</img>``.

        Works at the text level to avoid BPE context-merging issues: when
        ``<image>`` appears in a chat template (e.g. ``<image>\\n``), the
        trailing ``>`` may merge with the following newline into a single
        token (e.g. ``>\\n``=397), making token-level pattern matching
        fragile.  Decode→replace→re-tokenize is deterministic because
        ``<img>``, ``<IMG_CONTEXT>``, and ``</img>`` are all special tokens
        that are immune to BPE merges.
        """
        feature_size = num_tiles * self._num_image_token
        expansion_text = f"<img>{'<IMG_CONTEXT>' * feature_size}</img>"

        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        text = text.replace("<image>", expansion_text, 1)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_rope_index(self, input_ids, attention_mask, **multi_modal_kwargs):
        """Compute position IDs for InternVL (1D RoPE, no 3D mRoPE).

        InternVL uses standard 1D RoPE on its Qwen2 LLM backbone.  We return
        ``(3, 1, seq_len)`` shaped position IDs where all three dimensions are
        identical.  The caller (:meth:`_compute_position_ids`) transposes to
        ``(1, 3, seq_len)`` and concatenates with text positions to get a
        final ``(1, 4, seq_len)`` tensor.
        """
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
        # Expand to (3, batch, seq_len) — caller expects this shape.
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return position_ids, None
