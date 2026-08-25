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
from verl.utils.tokenizer.chat_template import apply_chat_template, extract_system_prompt_and_generation

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
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
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
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

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

        # system prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # generation prompt: <|im_start|>assistant\n
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )
        self._following_user_context_ids: dict[Optional[bool], torch.Tensor] = {}
        self._following_user_assistant_header_ids: dict[tuple[Optional[bool], bool], torch.Tensor] = {}

    def __len__(self):
        return len(self.messages)

    def _process_message_group(
        self,
        index: int,
        messages: list[dict[str, Any]],
        full_message: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process one independently renderable message group.

        A group is normally one message. Neighbor-dependent messages receive
        either adjacent messages in their group or a temporary following user
        context so their output remains identical to full-conversation rendering.

        * an initial ``system`` plus its following ``user`` message;
        * one or more consecutive ``tool`` messages.

        Args:
            index: turn index in the conversation
            messages: One message or a neighbor-dependent message group
            full_message: Complete conversation used to detect required right context
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        role = messages[0]["role"]
        has_following_user = self._has_following_user_query(full_message, index + len(messages))
        use_following_user_context = len(messages) == 1 and role in {"assistant", "system"} and has_following_user

        inputs = self._render_message_group(
            messages=messages,
            tools=tools,
            enable_thinking=enable_thinking,
            use_following_user_context=use_following_user_context,
        )

        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # remove system prompt if exists
        if index != 0 and role != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if role == "assistant":
            if len(messages) != 1:
                raise ValueError("An assistant message cannot share an SFT rendering group")
            loss_mask = torch.ones_like(attention_mask)
            if use_following_user_context:
                assistant_prefix = self._assistant_header_ids_with_following_user(
                    tools=tools,
                    enable_thinking=enable_thinking,
                    remove_system_prompt=index != 0,
                )
                assistant_prefix_length = len(assistant_prefix)
                if len(input_ids) < assistant_prefix_length or not torch.equal(
                    input_ids[:assistant_prefix_length], assistant_prefix
                ):
                    raise AssertionError("Rendered assistant message does not start with the inferred header")
            else:
                assistant_prefix_length = len(self.generation_prompt)
            loss_mask[:assistant_prefix_length] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def _text_message(self, role: str, text: str) -> dict[str, Any]:
        content: str | list[dict[str, str]] = text
        if self.processor is not None:
            content = [{"type": "text", "text": text}]
        return {"role": role, "content": content}

    def _empty_user_message(self) -> dict[str, Any]:
        return self._text_message("user", "")

    @staticmethod
    def _is_user_query(message: dict[str, Any]) -> bool:
        """Whether a user message is a real query rather than a tool-response envelope."""
        if message["role"] != "user":
            return False

        content = message.get("content", "")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
                    parts.append(str(item.get("text", "")))
                else:
                    # Preserve a non-text item in the boundary check. Qwen's
                    # render_content emits a vision marker for the same item.
                    parts.append("\ufffc")
            text = "".join(parts).strip()
        else:
            return True

        return not (text.startswith("<tool_response>") and text.endswith("</tool_response>"))

    @classmethod
    def _has_following_user_query(cls, full_message: list[dict[str, Any]], start: int) -> bool:
        return any(cls._is_user_query(message) for message in full_message[start:])

    @staticmethod
    def _common_prefix_length(input_ids: torch.Tensor, prefix_ids: list[int] | torch.Tensor) -> int:
        length = 0
        max_length = min(len(input_ids), len(prefix_ids))
        while length < max_length and input_ids[length] == prefix_ids[length]:
            length += 1
        return length

    def _assistant_header_ids_with_following_user(
        self,
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
        remove_system_prompt: bool,
    ) -> torch.Tensor:
        """Infer the assistant header without comparing it against real message content."""
        cache_key = (enable_thinking, remove_system_prompt)
        if tools is None and cache_key in self._following_user_assistant_header_ids:
            return self._following_user_assistant_header_ids[cache_key]

        # Empty and non-empty bodies diverge exactly at the content boundary. Their
        # shared prefix therefore cannot consume real content that happens to start
        # like a generation prompt, such as a literal ``<think>`` token.
        probe_ids = []
        for text in ("", "x"):
            probe_inputs = self._render_message_group(
                messages=[self._text_message("assistant", text)],
                tools=tools,
                enable_thinking=enable_thinking,
                use_following_user_context=True,
            )
            ids = probe_inputs["input_ids"][0]
            if remove_system_prompt:
                ids = ids[len(self.system_prompt) :]
            probe_ids.append(ids)

        header_length = self._common_prefix_length(probe_ids[0], probe_ids[1])
        if header_length == min(len(ids) for ids in probe_ids):
            raise AssertionError("Assistant header probes did not produce distinct message bodies")

        header_ids = probe_ids[0][:header_length]
        if tools is None:
            self._following_user_assistant_header_ids[cache_key] = header_ids
        return header_ids

    def _render_message_group(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
        use_following_user_context: bool,
    ) -> dict[str, torch.Tensor]:
        """Render a group and remove a temporary user used only as right context."""
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        render_messages = [*messages, self._empty_user_message()] if use_following_user_context else messages
        inputs = apply_chat_template(
            processor,
            messages=render_messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        if not use_following_user_context:
            return inputs

        # Render the temporary user without tools, then remove any tokenizer-level
        # default system prompt. The remaining tokens are the exact trailing user
        # span appended above, even when ``tools`` changes the leading system block.
        if enable_thinking not in self._following_user_context_ids:
            context_inputs = dict(
                apply_chat_template(
                    processor,
                    messages=[self._empty_user_message()],
                    tools=None,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **apply_chat_template_kwargs,
                )
            )
            self._following_user_context_ids[enable_thinking] = context_inputs["input_ids"][0][
                len(self.system_prompt) :
            ]
        context_ids = self._following_user_context_ids[enable_thinking]
        context_length = len(context_ids)
        rendered_ids = inputs["input_ids"][0]
        if (
            context_length == 0
            or len(rendered_ids) < context_length
            or not torch.equal(rendered_ids[-context_length:], context_ids)
        ):
            raise AssertionError("Temporary following-user context is not a removable token suffix")

        sequence_length = len(rendered_ids)
        for key in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"):
            value = inputs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == sequence_length:
                inputs[key] = value[..., :-context_length]
        return inputs

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Backward-compatible wrapper for callers overriding the old hook."""
        return self._process_message_group(
            index=index,
            messages=[message],
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _message_group_end(messages: list[dict[str, Any]], start: int) -> int:
        """Return the exclusive end of the next independently renderable group."""
        role = messages[start]["role"]

        # Qwen3.5/3.6 reject a system-only conversation. System and the first
        # user turn are both non-trainable, so rendering them together preserves
        # the exact full-conversation prefix without changing the loss mask.
        if start == 0 and role == "system" and start + 1 < len(messages) and messages[start + 1]["role"] == "user":
            return start + 2

        # Neighbor-aware templates wrap a consecutive tool run in one user
        # envelope. Rendering the complete run preserves all tool-response
        # blocks without duplicating the envelope around every result.
        if role == "tool":
            end = start + 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            return end

        return start + 1

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

        # 1. Tokenize independently renderable message groups. Most groups are
        # one message; initial system+user and consecutive tool messages must
        # retain their template-neighbor context.
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        i = 0
        while i < len(messages):
            end = self._message_group_end(messages, i)
            message_group = messages[i:end]
            if len(message_group) == 1:
                _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                    index=i,
                    message=message_group[0],
                    full_message=messages,
                    tools=tools if i == 0 else None,
                    enable_thinking=enable_thinking,
                )
            else:
                _input_ids, _loss_mask, _attention_mask, _inputs = self._process_message_group(
                    index=i,
                    messages=message_group,
                    full_message=messages,
                    tools=tools if i == 0 else None,
                    enable_thinking=enable_thinking,
                )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)
            i = end

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        print_assembled_message(self.tokenizer, messages, input_ids, loss_mask, attention_mask, tools)
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes
        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if k == "mm_token_type_ids":
                keys_to_remove.append(k)
                continue
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                # Check if all tensors in the list have the same shape
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]

        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        # 2. handle position_ids for Qwen-VL series models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
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

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """Check concatenated input_ids of message-group rendering equals
        apply_chat_template to the whole conversation.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to message groups and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)
