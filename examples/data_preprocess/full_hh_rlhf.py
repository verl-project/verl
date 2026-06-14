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
- Preprocess data and split the training set into 75% for training RM and 25% for validting RM.
- All the training data is used to train SFT and RL.
- Both chosen and rejected is used to train SFT
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from verl.utils.fs import copy, makedirs


def generate_sft_dataset(target_hdfs_path_dir, local_dir="~/data/full_hh_rlh/sft", local_dataset_path=None):
    if local_dataset_path is not None:
        dataset = load_dataset(local_dataset_path)
    else:
        dataset = load_dataset("Dahoas/full-hh-rlhf")
    output = {"prompt": [], "response": []}
    for data in tqdm(dataset["train"]):
        # add chosen
        output["prompt"].append(data["prompt"])
        output["response"].append(data["chosen"])

        # add rejection
        output["prompt"].append(data["prompt"])
        output["response"].append(data["rejected"])

    df = pd.DataFrame(output)

    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, "train.parquet")

    df.to_parquet(path=local_path)

    if target_hdfs_path_dir is not None:
        hdfs_dir = target_hdfs_path_dir + "/" + "train.parquet"
        makedirs(hdfs_dir)

        copy(local_path, hdfs_dir)


def generate_rm_dataset(target_hdfs_path_dir, local_dir="~/data/full_hh_rlh/rm", local_dataset_path=None):
    if local_dataset_path is not None:
        train_dataset = load_dataset(local_dataset_path, split="train[:75%]")
        test_dataset = load_dataset(local_dataset_path, split="train[-25%:]")
    else:
        train_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train[:75%]")
        test_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train[-25%:]")

    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    for dataset, name in zip([train_dataset, test_dataset], ["train", "test"], strict=True):
        output = {"prompt": [], "chosen": [], "rejected": []}
        for data in tqdm(dataset):
            # add chosen
            output["prompt"].append(data["prompt"])
            output["chosen"].append(data["chosen"])
            output["rejected"].append(data["rejected"])

        df = pd.DataFrame(output)

        local_path = os.path.join(local_dir, name + ".parquet")

        df.to_parquet(path=local_path)

        if target_hdfs_path_dir is not None:
            hdfs_dir = target_hdfs_path_dir + "/" + name + ".parquet"
            makedirs(hdfs_dir)

            copy(local_path, hdfs_dir)


def generate_rl_dataset(target_hdfs_path_dir, local_dir="~/data/full_hh_rlhf/rl", local_dataset_path=None):
    if local_dataset_path is not None:
        dataset = load_dataset(local_dataset_path)
    else:
        dataset = load_dataset("Dahoas/full-hh-rlhf")
    train_dataset = dataset["train"]

    data_source = "Dahoas/full-hh-rlhf"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            response = example.pop("response")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                    "ground_truth": response,  # should not be used
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    local_dir = os.path.expanduser(local_dir)
    local_path = os.path.join(local_dir, "train.parquet")
    train_dataset.to_parquet(local_path)

    if target_hdfs_path_dir is not None:
        hdfs_dir = target_hdfs_path_dir + "/" + "train.parquet"
        makedirs(hdfs_dir)

        copy(local_path, hdfs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["sft", "rm", "rl"], required=True)
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", type=str, required=False, default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/full_hh_rlhf",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    if args.split == "sft":
        generate_sft_dataset(args.hdfs_dir, os.path.join(local_save_dir, args.split), args.local_dataset_path)
    elif args.split == "rm":
        generate_rm_dataset(args.hdfs_dir, os.path.join(local_save_dir, args.split), args.local_dataset_path)
    elif args.split == "rl":
        generate_rl_dataset(args.hdfs_dir, os.path.join(local_save_dir, args.split), args.local_dataset_path)
    else:
        raise NotImplementedError
