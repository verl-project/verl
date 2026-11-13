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

    The conversation is assembled by rendering each turn separately and
    concatenating. That contract requires a CONCATENABLE chat template: one
    whose per-turn renders compose to the whole-conversation render, which in
    practice means the role markers must be emitted by each turn's own render
    (not by inter-turn transition logic that only fires on
    ``add_generation_prompt`` or loop conditions). The DeepSeek-V4.1
    checkpoint ships no template at all (and the one the current upstream repo
    carries is transition-based, not concatenable) - inject a custom template
    via ``model.custom_chat_template``; ``sanity_check`` enforces the
    concat-equivalence on the first item and raises under the default
    ``ignore_input_ids_mismatch=False``.

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

        # Extract messages list from dataframe. Pre-tokenized datasets carry their ids
        # instead of a conversation, so the column is optional in that case.
        if self.messages_key in self.dataframe.columns:
            self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
        elif "input_ids" in self.dataframe.columns:
            self.messages = [[] for _ in range(len(self.dataframe))]
        else:
            raise KeyError(
                f"dataset has neither the messages column {self.messages_key!r} nor pre-tokenized "
                f"input_ids; columns are {list(self.dataframe.columns)}"
            )

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

    def __len__(self):
        return len(self.messages)

    def _select_render_processor(self, messages: list[dict[str, Any]]):
        """Pick the renderer for chat-template application over these messages.

        Text-only messages render via the tokenizer, which honors the injected
        custom chat template. The multimodal processor IGNORES chat_template
        and renders assistant turns with its own internal format ([BOS,
        content, EOS] - no role marker; asserted for DeepseekV41Processor on
        the DSV4.1 alignment work, not pinned by an in-repo test), so routing
        text-only turns through it silently drops the marker from the
        training stream; it is only required when a turn carries image/video
        parts that need processor-side expansion. Note this routing also
        means multi-part TEXT content is joined by the template (no
        separator), not by the processor's "\\n\\n" join.
        """
        if self.processor is None:
            return self.tokenizer
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") in ("image", "video") for part in content
            ):
                return self.processor
        return self.tokenizer

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process a single message and return its tokenized representation.

        Args:
            index: turn index in the conversation
            message: A single message dictionary
            images: List of images to be used
            videos: List of videos to be used
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        processor = self._select_render_processor([message])
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = apply_chat_template(
            processor,
            messages=[message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # DeepSeek V4.1 returns one vision-token-type row for every token after
        # image expansion.  Keep it aligned with this turn while removing the
        # per-turn system prefix below; the rows are concatenated after all
        # turns have been tokenized.
        vision_token_types = inputs.get("vision_token_types")
        if vision_token_types is not None:
            vision_token_types = vision_token_types[0]

        # head-match: does this turn's render actually start with the system
        # prompt tokens? Decides the strip below.
        _head_matches = False
        if len(self.system_prompt) > 0:
            _head = input_ids[: len(self.system_prompt)]
            _head = _head.tolist() if hasattr(_head, "tolist") else list(_head)
            _sp = self.system_prompt
            _sp = _sp.tolist() if hasattr(_sp, "tolist") else list(_sp)
            _head_matches = _head == _sp

        # remove system prompt if exists - guarded by the head-match check and
        # only for non-first turns: the first turn's render carries the
        # conversation's opening system block, which stays in the stream (the
        # model conditions on it; the whole-conversation render includes it).
        # Templates whose non-first turns carry a different prefix (e.g. the
        # DSV4.1 marker template: the assistant turn starts with
        # <｜Assistant｜>, not BOS) must not be stripped - stripping would
        # delete the marker instead of the block.
        if index != 0 and message["role"] != "system" and _head_matches:
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]
            if vision_token_types is not None:
                vision_token_types = vision_token_types[len(self.system_prompt) :]

        if vision_token_types is not None:
            if vision_token_types.shape != input_ids.shape:
                raise ValueError(
                    "vision_token_types must stay aligned with input_ids after per-turn system-prefix removal: "
                    f"{vision_token_types.shape=} {input_ids.shape=}"
                )
            inputs["vision_token_types"] = vision_token_types

        # Build the loss mask AFTER the (possibly skipped) strip, zeroing the
        # scaffolding by VALUE: the sequence's leading system-prompt block
        # (when the strip was skipped) and the generation-prompt tokens that
        # actually follow it. Verifying each part's presence instead of
        # assuming lengths keeps this correct when the renderer's shape does
        # not match the tokenizer-derived lengths (e.g. an image-bearing turn
        # rendered by the processor has no generation-prompt header after the
        # block) - zeroing by assumption would mask content tokens there.
        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            _ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            _sp = self.system_prompt
            _sp = _sp.tolist() if hasattr(_sp, "tolist") else list(_sp)
            _gp = self.generation_prompt
            _gp = _gp.tolist() if hasattr(_gp, "tolist") else list(_gp)
            _zero_width = 0
            if _sp and _ids[: len(_sp)] == _sp:
                _zero_width = len(_sp)
            if _gp and _ids[_zero_width : _zero_width + len(_gp)] == _gp:
                _zero_width += len(_gp)
            loss_mask[:_zero_width] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

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

        # Pre-tokenized rows: the ids come from an engine that already decided the
        # exact sequence (a rollout, or the teacher side of a distillation datum).
        # Re-tokenizing such a row through a chat template cannot be trusted to
        # reproduce the same ids, so when the caller supplies them they are used
        # verbatim and the message path is skipped entirely.
        if row_dict.get("input_ids") is not None:
            return self._sample_from_token_ids(row_dict)

        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        # 1. tokenize each message
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        input_id_lengths = []
        vision_token_type_rows = []
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                index=i,
                message=message,
                full_message=messages,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            input_id_lengths.append(_input_ids.shape[0])
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            vision_token_type_rows.append(_inputs.pop("vision_token_types", None))
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        # Unlike image patches and image grids, vision_token_types has one
        # value per input token.  It therefore must be concatenated along the
        # sequence dimension across turns instead of being filtered as a
        # fixed-shape image tensor.
        if any(row is not None for row in vision_token_type_rows):
            text_type = -1
            aligned_rows = [
                row if row is not None else torch.full((length,), text_type, dtype=torch.long)
                for row, length in zip(vision_token_type_rows, input_id_lengths, strict=False)
            ]
            if any(row.shape != (length,) for row, length in zip(aligned_rows, input_id_lengths, strict=False)):
                raise ValueError("vision_token_types rows must match their tokenized message lengths")
            vision_token_types = torch.cat(aligned_rows, dim=0).unsqueeze(0)
        else:
            vision_token_types = None

        # Optional teacher distribution, precomputed for distillation (on-policy
        # self-distillation reads the teacher under a privileged context, which is a
        # separate forward and therefore a separate pass). Row i holds the teacher's
        # top-k for the token following position i, the same alignment target_tokens
        # uses; a length mismatch is a data bug that would train on shifted targets,
        # so it fails here rather than silently misaligning.
        teacher_ids = row_dict.get("teacher_ids")
        teacher_logprobs = row_dict.get("teacher_logprobs")
        if (teacher_ids is None) != (teacher_logprobs is None):
            raise ValueError(
                "teacher_ids and teacher_logprobs must be provided together; "
                f"got {type(teacher_ids).__name__} and {type(teacher_logprobs).__name__}"
            )
        teacher_tensors = {}
        if teacher_ids is not None:
            teacher_ids_t = torch.tensor(teacher_ids, dtype=torch.int64)
            teacher_logprobs_t = torch.tensor(teacher_logprobs, dtype=torch.float32)
            if teacher_ids_t.shape != teacher_logprobs_t.shape:
                raise ValueError(
                    f"teacher_ids {tuple(teacher_ids_t.shape)} and teacher_logprobs "
                    f"{tuple(teacher_logprobs_t.shape)} must have the same shape"
                )
            if teacher_ids_t.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    f"teacher rows ({teacher_ids_t.shape[0]}) must match the tokenized length "
                    f"({input_ids.shape[0]}); the teacher distribution is position-aligned by construction"
                )
            teacher_tensors = {"teacher_ids": teacher_ids_t, "teacher_logprobs": teacher_logprobs_t}

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

        if vision_token_types is not None and all(
            key in multi_modal_inputs for key in ("pixel_values", "image_grid_hws")
        ):
            multi_modal_inputs["vision_token_types"] = vision_token_types

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
        if (
            sequence_length > self.max_length
            and "vision_token_types" in multi_modal_inputs
            and self.truncation != "error"
        ):
            raise ValueError("Multimodal sequences require truncation='error'")
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
                if "vision_token_types" in multi_modal_inputs:
                    multi_modal_inputs["vision_token_types"] = F.pad(
                        multi_modal_inputs["vision_token_types"],
                        (0, self.max_length - sequence_length),
                        value=-1,
                    )
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                    if "vision_token_types" in multi_modal_inputs:
                        multi_modal_inputs["vision_token_types"] = multi_modal_inputs["vision_token_types"][
                            ..., -self.max_length :
                        ]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                    if "vision_token_types" in multi_modal_inputs:
                        multi_modal_inputs["vision_token_types"] = multi_modal_inputs["vision_token_types"][
                            ..., : self.max_length
                        ]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                **teacher_tensors,
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
                if "vision_token_types" in multi_modal_inputs:
                    multi_modal_inputs["vision_token_types"] = multi_modal_inputs["vision_token_types"][
                        ..., : self.max_length
                    ]

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                **teacher_tensors,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def _sample_from_token_ids(self, row_dict: dict) -> dict:
        """Build a sample from exact token ids, then apply the configured truncation.

        External teacher lookups key the returned tokens, so teacher export and
        training must use the same max_length and truncation settings.
        """
        input_ids = torch.tensor(row_dict["input_ids"], dtype=torch.int64)
        length = input_ids.shape[0]
        raw_mask = row_dict.get("loss_mask")
        if raw_mask is None:
            # no mask given: supervise everything after the first token
            loss_mask = torch.ones(length, dtype=torch.int64)
        else:
            loss_mask = torch.tensor(raw_mask, dtype=torch.int64)
            if loss_mask.shape[0] != length:
                raise ValueError(
                    f"loss_mask has {loss_mask.shape[0]} entries for {length} tokens; "
                    "the mask is token-level and must cover every token"
                )

        teacher_ids = row_dict.get("teacher_ids")
        teacher_logprobs = row_dict.get("teacher_logprobs")
        teacher_tensors = {}
        if (teacher_ids is None) != (teacher_logprobs is None):
            raise ValueError("teacher_ids and teacher_logprobs must be provided together")
        if teacher_ids is not None:
            ids_t = torch.tensor(teacher_ids, dtype=torch.int64)
            lp_t = torch.tensor(teacher_logprobs, dtype=torch.float32)
            if ids_t.shape != lp_t.shape or ids_t.shape[0] != length:
                raise ValueError(f"teacher rows {tuple(ids_t.shape)} must align with the {length} tokens")
            # Padded mode can ship these as tensors: a [max_length, topk] row stacks
            # into a batch-aligned tensor that the engine slices per micro-batch along
            # with the logits. No-padding cannot — the rows are ragged and verl's
            # non-tensor channel does not carry them reliably — so it keeps lists and
            # the loss builds its tensors from them.
            # Deliberately NOT attached to the sample: the teacher payload is per-token
            # and ragged, and every route through the batch (tensor key, nested key,
            # non-tensor) either fails the batch-dimension contract or is flattened on
            # the way in. The trainer hands it to the loss function directly, per datum.
            teacher_tensors = {}

        if length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"pre-tokenized row has {length} tokens > max_length={self.max_length}")
            if self.truncation == "left":
                token_slice = slice(-self.max_length, None)
            elif self.truncation == "right":
                token_slice = slice(None, self.max_length)
            else:
                raise ValueError(f"Unknown truncation mode {self.truncation}")
            input_ids = input_ids[token_slice]
            loss_mask = loss_mask[token_slice]
            length = input_ids.shape[0]

        # Token-level masks cannot supervise the first retained token: there is
        # no preceding model position for it. This also keeps packed CE's flat
        # mask shift from supervising the previous sample's phantom last entry.
        if length:
            loss_mask[0] = 0

        # passthrough ids: the caller keys precomputed data (teacher distributions) by
        # this column, and the collator carries scalars as non-tensor batch entries
        passthrough = {}
        if row_dict.get("sample") is not None:
            passthrough["sample"] = int(row_dict["sample"])

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return {
                "input_ids": input_ids,
                "position_ids": torch.arange(length, dtype=torch.int64),
                "loss_mask": loss_mask,
                **teacher_tensors,
                **passthrough,
            }

        pad_len = self.max_length - length
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded = torch.full((self.max_length,), pad_id, dtype=torch.int64)
        padded[:length] = input_ids
        attention_mask = torch.zeros(self.max_length, dtype=torch.int64)
        attention_mask[:length] = 1
        mask_padded = torch.zeros(self.max_length, dtype=torch.int64)
        mask_padded[:length] = loss_mask
        res = {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(self.max_length, dtype=torch.int64),
            "loss_mask": mask_padded,
            **passthrough,
        }
        if teacher_tensors and isinstance(teacher_tensors["teacher_ids"], torch.Tensor):
            pad_k = teacher_tensors["teacher_ids"].shape[1]
            t_ids = torch.zeros((self.max_length, pad_k), dtype=torch.int64)
            t_lp = torch.full((self.max_length, pad_k), float("-inf"), dtype=torch.float32)
            t_ids[:length] = teacher_tensors["teacher_ids"]
            t_lp[:length] = teacher_tensors["teacher_logprobs"]
            res["teacher_ids"] = t_ids
            res["teacher_logprobs"] = t_lp
        del pad_len
        return res

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """Check concatenated input_ids of apply_chat_template to each turn equals
        apply_chat_template to whole messages.
        """
        renderers = [self._select_render_processor([m]) for m in messages]
        if any(r is not renderers[0] for r in renderers):
            # Mixed renderers within one conversation (an image/video turn goes
            # through the processor, text-only turns through the tokenizer):
            # there is no single whole-conversation render to compare against,
            # so the per-turn-vs-whole check is meaningless here.
            logger.warning_once(
                "MultiTurnSFTDataset assembles this conversation from mixed renderers "
                "(image/video turns via the processor, text-only turns via the tokenizer), "
                "so the per-turn-vs-whole input_ids check is skipped for it."
            )
            return
        processor = renderers[0]
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
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "Per-turn assembly requires a CONCATENABLE chat template: one whose role markers are "
            "emitted by each turn's own render, not by inter-turn transition logic that only fires "
            "on add_generation_prompt or loop conditions (e.g. the DeepSeek-V4.1 "
            "chat_template.jinja on the current upstream revision is transition-based and cannot "
            "be used here - inject a custom template via model.custom_chat_template).\n"
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
