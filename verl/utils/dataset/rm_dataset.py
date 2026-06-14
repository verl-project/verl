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

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from verl.utils import hf_tokenizer


def download_files_distributed(download_fn):
    import torch.distributed

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            # download files
            download_fn()

        torch.distributed.barrier()
    else:
        # download anyway
        download_fn()


class RMDataset(Dataset):
    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        max_length=1024,
        add_eos=True,
        cache_dir="~/.cache/verl/rm",
        max_samples: int = -1,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.cache_dir = os.path.expanduser(cache_dir)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        self.add_eos = add_eos
        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        def _download_files():
            from verl.utils.fs import copy, is_non_local

            os.makedirs(self.cache_dir, exist_ok=True)
            assert os.path.exists(self.cache_dir)
            for i, parquet_file in enumerate(self.parquet_files):
                if is_non_local(parquet_file):
                    dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                    if not os.path.exists(dst):
                        copy(src=parquet_file, dst=dst)
                    self.parquet_files[i] = dst

        download_files_distributed(_download_files)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
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

        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def _pad_to_length(self, input_ids, attention_mask):
        curr_length = input_ids.shape[-1]

        if curr_length < self.max_length:
            input_ids = torch.cat(
                (input_ids, torch.zeros(size=(self.max_length - curr_length,), dtype=input_ids.dtype)), dim=-1
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(size=(self.max_length - curr_length,), dtype=attention_mask.dtype)), dim=-1
            )
        elif curr_length > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return input_ids, attention_mask

    def __getitem__(self, item):
        prompt = self.prompts[item]
        chosen_response = self.chosen_responses[item]
        rejected_response = self.rejected_responses[item]

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        chosen_response_ids = self.tokenizer(chosen_response, return_tensors="pt")["input_ids"][0]
        rejected_response_ids = self.tokenizer(rejected_response, return_tensors="pt")["input_ids"][0]

        if self.add_eos:
            chosen_response_ids = torch.cat((chosen_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)
            rejected_response_ids = torch.cat(
                (rejected_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1
            )

        chosen_input_ids = torch.cat((prompt_ids, chosen_response_ids), dim=-1)
        chosen_attention_mask = torch.ones_like(chosen_input_ids)

        rejected_input_ids = torch.cat((prompt_ids, rejected_response_ids), dim=-1)
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        chosen_input_ids, chosen_attention_mask = self._pad_to_length(chosen_input_ids, chosen_attention_mask)
        rejected_input_ids, rejected_attention_mask = self._pad_to_length(rejected_input_ids, rejected_attention_mask)

        input_ids = torch.stack((chosen_input_ids, rejected_input_ids), dim=0)
        attention_mask = torch.stack((chosen_attention_mask, rejected_attention_mask), dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
