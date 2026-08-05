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
Preprocess the lmms-lab/multimodal-open-r1-8k-verified dataset to parquet format.

Images are kept as raw bytes (no decode, no resize).
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir", default="~/data/openr1mm", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "lmms-lab/multimodal-open-r1-8k-verified"
    dataset = datasets.load_dataset(data_source)

    instruction = (
        "You FIRST think about the reasoning process as an internal monologue "
        "and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE enclosed within <answer> </answer> tags."
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            solution = example.pop("solution")
            img = example.pop("image")

            prompt_content = f"<image>\n{problem}\n\n{instruction}"

            # Keep image as raw bytes dict to avoid lossy re-encoding.
            # The Qwen VL processor handles resize at runtime.
            if isinstance(img, dict) and "bytes" in img:
                image_data = img
            elif isinstance(img, bytes):
                image_data = {"bytes": img}
            else:
                image_data = img

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_content}],
                "images": [image_data],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": problem,
                    "answer": solution,
                },
            }
            return data

        return process_fn

    full_dataset = dataset["train"]
    full_dataset = full_dataset.cast_column("image", datasets.Image(decode=False))
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset["train"].map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = split_dataset["test"].map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    columns = ["data_source", "prompt", "images", "ability", "reward_model", "extra_info"]
    train_dataset = train_dataset.select_columns(columns)
    test_dataset = test_dataset.select_columns(columns)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
