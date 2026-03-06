# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""
Preprocess the llamafactory/pokemon-gpt4o-captions dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/pokemon-gpt4o-captions",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "llamafactory/pokemon-gpt4o-captions"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    def map_fn(row: dict):
        messages = []
        conversation = row.pop("conversations")
        for conv in conversation:
            if conv["from"] == "gpt":
                role = "assistant"
            elif conv["from"] == "human":
                role = "user"
            else:
                raise ValueError(f"Unknown role: {conv['from']}")
            messages.append(
                {
                    "role": role,
                    "content": conv["value"],
                }
            )

        row["messages"] = messages
        return row

    dataset = dataset["train"].map(map_fn, num_proc=16)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
