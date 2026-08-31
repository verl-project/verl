# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Adapter for the canonical Slime math JSONL files.

Slime stores the prompt and answer as ``prompt`` and ``label``.  VERL's DAPO
reward manager expects the same answer under ``reward_model.ground_truth`` and
uses ``data_source`` to select its math verifier.  Keeping this conversion in a
dataset adapter lets the matched BF16/W4A4 runs consume the exact Slime input
files without producing another, potentially drifting, dataset copy.
"""

import json

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids


class SlimeMathDataset(RLHFDataset):
    """Expose canonical Slime math samples using VERL's reward schema."""

    def _read_files_and_tokenize(self):
        """Read the small canonical JSONL files without HF cache locking.

        ``datasets.load_dataset("json")`` builds a shared Arrow cache and can
        wait indefinitely on a stale lock when a Ray task is cancelled during
        startup.  These inputs are only about 10 MiB, so constructing Arrow
        tables directly from the JSONL records is both cheap and deterministic.
        """
        dataframes = []
        for data_file in self.data_files:
            if not data_file.endswith((".json", ".jsonl")):
                raise ValueError(f"SlimeMathDataset requires JSON/JSONL input, got {data_file}")
            rows = []
            with open(data_file, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"{data_file}:{line_number} is not a JSON object")
                    rows.append(row)
            dataframes.append(datasets.Dataset.from_list(rows))

        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if 0 < self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        """Filter text prompts without Arrow cache, workers, or progress IPC.

        The canonical Slime inputs are small enough to tokenize serially.  A
        plain loop avoids ``datasets.Dataset.filter`` hanging in a Ray driver
        after it has emitted multiprocessing/progress events.
        """
        if not self.filter_overlong_prompts:
            return dataframe
        if self.processor is not None:
            raise ValueError("SlimeMathDataset only supports text prompts")

        dataframe = self.dataframe if dataframe is None else dataframe
        apply_kwargs = dict(**self.apply_chat_template_kwargs)
        if self.tool_schemas is not None:
            apply_kwargs["tools"] = self.tool_schemas

        # Keep explicit tokenization to avoid Transformers version defaults.
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)

        keep_indices = []
        for index in range(len(dataframe)):
            doc = dataframe[index]
            try:
                tokenized_prompt = self.tokenizer.apply_chat_template(
                    doc[self.prompt_key], add_generation_prompt=True, tokenize=True, **apply_kwargs
                )
                prompt_length = len(normalize_token_ids(tokenized_prompt))
            except Exception as exc:
                raise RuntimeError(f"failed to tokenize Slime prompt at index {index}") from exc
            if prompt_length <= self.max_prompt_length:
                keep_indices.append(index)

        filtered = dataframe.select(keep_indices)
        print(f"filter dataset len: {len(filtered)}")
        return filtered

    def __getitem__(self, item):
        row = super().__getitem__(item)
        if "label" not in row:
            raise KeyError("canonical Slime math sample is missing 'label'")

        row["data_source"] = "math_dapo"
        row["ability"] = "math"
        row["reward_model"] = {
            "style": "rule",
            "ground_truth": str(row.pop("label")),
        }
        row["extra_info"]["index"] = item
        row["index"] = item
        return row
