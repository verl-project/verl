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
Preprocess the open-r1/codeforces dataset to parquet format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--language", default="python",
                        choices=["python", "cpp", "all"],
                        help="Filter by language (default: python).")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "codeforces"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "verifiable-prompts")
    else:
        dataset = datasets.load_dataset("open-r1/codeforces", "verifiable-prompts")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Filter by language
    if args.language != "all":
        train_dataset = train_dataset.filter(
            lambda x: x.get("language") == args.language
        )
        test_dataset = test_dataset.filter(
            lambda x: x.get("language") == args.language
        )

    def make_map_fn(split):
        def process_fn(example, idx):
            official_tests = example["official_tests"]            
            ground_truth = {
                "inputs": [t["input"] for t in official_tests],
                "outputs": [t["output"] for t in official_tests],
            }

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user", 
                        "content": example["prompt"]
                    }
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(ground_truth),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "language": example["language"],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Saved {len(train_dataset)} train examples to {local_save_dir}/train.parquet")
    print(f"Saved {len(test_dataset)} test examples to {local_save_dir}/test.parquet")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
