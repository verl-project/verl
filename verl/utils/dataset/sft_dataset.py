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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import numpy as np
import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, max_samples: int = -1):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.ground_truth_key = config.get("ground_truth_key", None)
        # Normalize Hydra/OmegaConf types early (ListConfig/DictConfig -> plain Python)
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(self.ground_truth_key):
                self.ground_truth_key = OmegaConf.to_container(self.ground_truth_key, resolve=True)
        except Exception:
            pass

        # Auto-enable return_metadata if ground_truth_key is set
        self.return_metadata = config.get("return_metadata", self.ground_truth_key is not None)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.max_samples = max_samples
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

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

        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        def series_to_item(x):
            # unwrap singleton Series/ndarray
            while isinstance(x, (pd.Series, np.ndarray)) and len(x) == 1:
                x = x[0]

            # pyarrow Scalar -> python
            try:
                import pyarrow as pa
                if isinstance(x, pa.Scalar):
                    x = x.as_py()
            except Exception:
                pass

            return x

        def normalize_key(k):
            """
            Turn weird config/list-like keys into a plain Python str key.
            Handles:
            - OmegaConf ListConfig(['ground_truth'])
            - ['ground_truth']
            - np.ndarray(['ground_truth'])
            - np.str_('ground_truth')
            - other non-str objects
            """
            # OmegaConf ListConfig behaves like list
            try:
                from omegaconf import ListConfig
                if isinstance(k, ListConfig):
                    k = list(k)
            except Exception:
                pass

            # unwrap singleton list/tuple
            if isinstance(k, (list, tuple)) and len(k) == 1:
                k = k[0]

            # unwrap singleton numpy array
            if isinstance(k, np.ndarray):
                if k.size == 1:
                    k = k.item()
                else:
                    # if someone passed a multi-element array as key, make it a python list
                    k = k.tolist()

            # numpy scalar -> python
            if isinstance(k, (np.generic,)):
                k = k.item()

            # finally make sure it's a str
            if not isinstance(k, str):
                k = str(k)

            return k

        def extract_ground_truth(row, ground_truth_key):
            """
            row: pandas Series (self.dataframe.iloc[item])
            ground_truth_key:
            - list-like path: ['reward_model','ground_truth'] (maybe with weird element types)
            - string: 'reward_model.ground_truth' or 'reward_model' etc.
            """
            row = series_to_item(row)

            if ground_truth_key is None:
                return None

            # Case 1: list-like path
            if isinstance(ground_truth_key, (list, tuple)):
                cur = row
                for raw_k in ground_truth_key:
                    cur = series_to_item(cur)
                    k = normalize_key(raw_k)

                    # If current object is a pandas Series, cur[k] expects a column name (string)
                    # If current object is a dict, cur[k] is dict access.
                    cur = cur[k]
                return series_to_item(cur)

            # Case 2: string key
            k = normalize_key(ground_truth_key)

            # Support dotted flattened keys if present as a column
            if k in self.dataframe.columns:
                return series_to_item(row[k])

            # If dotted path but stored as nested dict in a cell, navigate it
            if "." in k and k.split(".", 1)[0] in row.index:
                parts = k.split(".")
                cur = row[parts[0]]
                for p in parts[1:]:
                    cur = series_to_item(cur)
                    cur = cur[normalize_key(p)]
                return series_to_item(cur)

            # Fallback: direct access
            return series_to_item(row[k])

        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        # Add metadata if requested
        if self.return_metadata:
            result["prompt"] = prompt
            result["response"] = response
            from omegaconf import OmegaConf
            if self.ground_truth_key is not None:
                
                try:
                    row = self.dataframe.iloc[item]
                    result["ground_truth"] = extract_ground_truth(row, self.ground_truth_key)
                except Exception as e:
                    # Keep debug concise; avoid log explosion
                    if item < 5:
                        row = self.dataframe.iloc[item]
                        rm = row.get("reward_model", None) if hasattr(row, "get") else (row["reward_model"] if "reward_model" in row else None)
                        print(f"\n[GT DEBUG] item={item} error={repr(e)}", flush=True)
                        print("[GT DEBUG] ground_truth_key:", self.ground_truth_key, flush=True)
                        try:
                            print("[GT DEBUG] ground_truth_key element types:",
                                [type(k) for k in self.ground_truth_key] if isinstance(self.ground_truth_key, (list, tuple)) else type(self.ground_truth_key),
                                flush=True)
                        except Exception:
                            pass
                        print("[GT DEBUG] columns:", list(self.dataframe.columns), flush=True)
                        print("[GT DEBUG] type(reward_model):", type(rm), flush=True)
                        print("[GT DEBUG] reward_model:", rm, flush=True)
                    result["ground_truth"] = None

        return result
            