# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Individual Contributors
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

"""
Off-policy dataset for GKD training using ground truth answers.

This dataset loads prompt-answer pairs (e.g., from GSM8K) and prepares them
for off-policy distillation by:
1. Tokenizing the prompt (question)
2. Tokenizing the answer (ground truth solution)
3. Concatenating them to form the full sequence
4. Preparing the data in the same format as on-policy rollout output

No generation is needed - the answers are used directly as responses.
"""

import copy
import logging
import os
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class GKDOffPolicyDataset(Dataset):
    """
    Off-policy dataset for GKD training using ground truth answers.

    This dataset is designed for off-policy distillation where we use
    pre-existing (prompt, answer) pairs instead of generating responses.

    Args:
        data_files: Path(s) to Parquet file(s) containing prompt and answer
        tokenizer: Tokenizer for text to token IDs
        config: Configuration with keys like:
            - prompt_key: Key for prompts in the dataset (default: "question")
            - answer_key: Key for answers in the dataset (default: "answer")
            - max_prompt_length: Maximum length for prompts
            - max_response_length: Maximum length for responses (answers)
            - truncation: How to handle overflow ("error", "left", "right")
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.config = config

        # Configuration
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "question")
        self.answer_key = config.get("answer_key", "answer")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 512)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy_to_local

        for i, parquet_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(
                src=parquet_file, cache_dir=self.cache_dir, use_shm=False
            )

    def _read_files(self):
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"Off-policy dataset len: {len(self.dataframe)}")

        # Sample if needed
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"Selected {self.max_samples} random samples out of {total}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Returns a dictionary containing:
        - input_ids: Full sequence (prompt + answer)
        - attention_mask: Attention mask for the full sequence
        - position_ids: Position IDs
        - responses: The answer tokens only
        - raw_prompt_ids: The prompt tokens only

        This format matches the output of on-policy rollout, allowing seamless
        integration with the existing training pipeline.
        """
        row_dict: dict = dict(self.dataframe[item])

        # Get prompt and answer
        prompt_data = row_dict[self.prompt_key]
        # Support nested key like "extra_info.answer"
        answer_text = row_dict
        for key_part in self.answer_key.split('.'):
            answer_text = answer_text[key_part]

        # Handle prompt: could be a string or a list of messages
        if isinstance(prompt_data, str):
            # Simple string prompt
            prompt_text = prompt_data
        elif isinstance(prompt_data, list):
            # Chat format: apply chat template
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config"
                )
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_data, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt_data)}")

        # Tokenize prompt and answer separately
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)

        # Handle prompt length
        if len(prompt_tokens) > self.max_prompt_length:
            if self.truncation == "left":
                prompt_tokens = prompt_tokens[-self.max_prompt_length:]
            elif self.truncation == "right":
                prompt_tokens = prompt_tokens[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(prompt_tokens)} exceeds max_prompt_length {self.max_prompt_length}"
                )

        # Handle answer length
        if len(answer_tokens) > self.max_response_length:
            if self.truncation == "left":
                answer_tokens = answer_tokens[-self.max_response_length:]
            elif self.truncation == "right":
                answer_tokens = answer_tokens[:self.max_response_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Answer length {len(answer_tokens)} exceeds max_response_length {self.max_response_length}"
                )

        # Concatenate prompt + answer to form full sequence
        full_sequence = prompt_tokens + answer_tokens

        # Create tensors
        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Pad prompt to max_prompt_length (left padding for prompt)
        prompt_padding_length = self.max_prompt_length - len(prompt_tokens)
        if prompt_padding_length > 0:
            prompt_pad = torch.full(
                (prompt_padding_length,),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            input_ids = torch.cat([prompt_pad, input_ids])
            attention_mask = torch.cat([
                torch.zeros(prompt_padding_length, dtype=torch.bool),
                attention_mask
            ])

        # Pad answer to max_response_length (right padding for answer)
        answer_padding_length = self.max_response_length - len(answer_tokens)
        if answer_padding_length > 0:
            answer_pad = torch.full(
                (answer_padding_length,),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, answer_pad])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(answer_padding_length, dtype=torch.bool)
            ])

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        # Create response tensor (answer tokens, padded)
        responses = torch.tensor(answer_tokens, dtype=torch.long)
        if answer_padding_length > 0:
            responses = torch.cat([responses, answer_pad])

        # Build output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "raw_prompt_ids": prompt_tokens,
        }

        # Pass through other metadata
        if "data_source" in row_dict:
            output["data_source"] = row_dict["data_source"]
        if "index" in row_dict:
            output["index"] = row_dict["index"]

        return output
