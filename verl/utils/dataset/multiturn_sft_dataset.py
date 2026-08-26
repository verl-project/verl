# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
import os
import re
from functools import wraps
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer import build_multimodal_processor_inputs, get_processor_token_id
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def once(func):
    """Decorator to ensure a function runs only once. Subsequent calls do nothing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "called"):
            wrapper.called = True
            return func(*args, **kwargs)

    return wrapper


@once
def print_assembled_message(tokenizer, message_list, input_ids, loss_mask, attn_mask, tools):
    """
    Print the message after applying the chat template
    """

    tokenized = tokenizer.apply_chat_template(message_list, add_generation_prompt=False, tokenize=False, tools=tools)
    sep = "\n\n"
    str = f"tokenized entire message:\n{tokenized}"
    str += sep
    decoded_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    str += f"tokenized seperately    :\n{tokenizer.decode(decoded_ids)}"

    logger.debug(str)


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int, optional): Limit the number of samples. Defaults to -1 (use all).
        hf_model_type (str, optional): Root Hugging Face model_type used to select the CT builder.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        hf_model_type: Optional[str] = None,
    ):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        self.messages_key = config.get("messages_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.continuous_token_model_family = config.get("continuous_token_model_family", "auto")
        self.mm_processor_kwargs = config.get("mm_processor_kwargs", {})
        self.hf_model_type = hf_model_type
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        self._continuous_token_builders: dict[Optional[bool], ContinuousTokenBuilder] = {}

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # default loader loads some list as np.ndarray, which fails the tokenizer
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def __len__(self):
        return len(self.messages)

    def _get_continuous_token_builder(self, enable_thinking: Optional[bool]) -> ContinuousTokenBuilder:
        """Return a CT builder whose template kwargs match one sample."""
        if enable_thinking not in self._continuous_token_builders:
            apply_chat_template_kwargs = dict(self.apply_chat_template_kwargs)
            if enable_thinking is not None:
                apply_chat_template_kwargs["enable_thinking"] = enable_thinking
            self._continuous_token_builders[enable_thinking] = create_continuous_token_builder(
                self.tokenizer,
                model_family=self.continuous_token_model_family,
                hf_model_type=self.hf_model_type,
                chat_template_kwargs=apply_chat_template_kwargs,
                mm_processor_kwargs=self.mm_processor_kwargs,
                processor=self.processor,
            )
        return self._continuous_token_builders[enable_thinking]

    @staticmethod
    def _collect_media(messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
        images: list[Any] = []
        videos: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"image", "image_url"} and block.get("image") is not None:
                    images.append(block["image"])
                elif block.get("type") == "video" and block.get("video") is not None:
                    videos.append(block["video"])
        return images, videos

    def _build_continuous_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble an SFT trajectory with the same append-only contract as RL TITO."""
        if not messages:
            raise ValueError("MultiTurnSFTDataset requires at least one message")

        builder = self._get_continuous_token_builder(enable_thinking)
        first_assistant_index = next(
            (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
            len(messages),
        )
        if first_assistant_index == 0:
            raise ValueError("Continuous Token SFT requires a non-assistant prompt before the first assistant message")

        runtime_messages = list(messages[:first_assistant_index])
        initial_images, initial_videos = self._collect_media(runtime_messages)
        runtime_token_ids = builder.build_initial_tokens(
            runtime_messages,
            tools=tools,
            images=initial_images or None,
            videos=initial_videos or None,
        )
        loss_mask = [0] * len(runtime_token_ids)

        index = first_assistant_index
        while index < len(messages):
            assistant_message = messages[index]
            if assistant_message.get("role") != "assistant":
                raise ValueError(
                    "Continuous Token SFT expected an assistant message after a generation prompt, "
                    f"got role={assistant_message.get('role')!r} at index {index}"
                )

            assistant_token_ids = builder.tokenize_assistant_message(assistant_message, tools=tools)
            merge_result = builder.merge_assistant_tokens(runtime_token_ids, assistant_token_ids)
            loss_mask, _ = builder.align_response_metadata(merge_result, loss_mask)
            runtime_token_ids = merge_result.token_ids
            runtime_messages.append(assistant_message)
            index += 1

            non_assistant_end = index
            while non_assistant_end < len(messages) and messages[non_assistant_end].get("role") != "assistant":
                non_assistant_end += 1
            if non_assistant_end == index:
                if index < len(messages):
                    raise ValueError("Continuous Token SFT does not support consecutive assistant messages")
                continue

            updated_messages = [*runtime_messages, *messages[index:non_assistant_end]]
            merge_result = builder.merge_non_assistant_tokens(
                runtime_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
            )
            loss_mask, _ = builder.align_response_metadata(merge_result, loss_mask)
            runtime_token_ids = merge_result.token_ids
            runtime_messages = updated_messages
            index = non_assistant_end

        input_ids = torch.tensor(runtime_token_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        if input_ids.shape != loss_mask_tensor.shape:
            raise AssertionError(
                f"Continuous Token input/loss shape mismatch: {input_ids.shape} != {loss_mask_tensor.shape}"
            )
        return input_ids, loss_mask_tensor, attention_mask

    def _build_multi_modal_inputs(
        self,
        input_ids: torch.Tensor,
        messages: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Rebuild only multimodal tensors from the final TITO token stream."""
        if self.processor is None:
            return {}

        images, videos = self._collect_media(messages)
        image_token_id = get_processor_token_id(self.processor, "image")
        video_token_id = get_processor_token_id(self.processor, "video")
        collapse_ids = {token_id for token_id in (image_token_id, video_token_id) if token_id is not None}
        collapsed_ids: list[int] = []
        previous_token_id = None
        for token_id in input_ids.tolist():
            if token_id in collapse_ids and token_id == previous_token_id:
                continue
            collapsed_ids.append(token_id)
            previous_token_id = token_id

        current_text = self.tokenizer.decode(collapsed_ids, skip_special_tokens=True)
        processor_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images or None,
            videos=videos or None,
            mm_processor_kwargs=self.mm_processor_kwargs or None,
        )
        processor_inputs.pop("input_ids", None)
        processor_inputs.pop("attention_mask", None)
        processor_inputs.pop("mm_token_type_ids", None)
        if hasattr(processor_inputs, "convert_to_tensors"):
            processor_inputs = processor_inputs.convert_to_tensors("pt")
        multi_modal_inputs = dict(processor_inputs)
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
            )
        return multi_modal_inputs

    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", "video": video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = convert_nested_value_to_list_recursive(example[self.messages_key])
        images = example[self.image_key] if self.image_key in example else []
        videos = example[self.video_key] if self.video_key in example else []

        image_offset, video_offset = 0, 0
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                continue

            if self.image_key not in example and self.video_key not in example:
                if self.processor is not None:
                    message["content"] = [{"type": "text", "text": content}]
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    video = process_video(videos[video_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        # 1. Build the initial prompt once, then append assistant and
        # non-assistant tokens with the same CT merge contract used by RL TITO.
        input_ids, loss_mask, attention_mask = self._build_continuous_tokens(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        multi_modal_inputs = self._build_multi_modal_inputs(input_ids, messages)

        print_assembled_message(self.tokenizer, messages, input_ids, loss_mask, attention_mask, tools)

        # 2. handle position_ids for Qwen-VL series models
        image_processor = getattr(self.processor, "image_processor", None)
        image_processor_name = type(image_processor).__name__ if image_processor is not None else ""
        if "Qwen" in image_processor_name and "VLImageProcessor" in image_processor_name:
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )  # (3, seq_len)
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # 3. handle padding
        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length and self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
