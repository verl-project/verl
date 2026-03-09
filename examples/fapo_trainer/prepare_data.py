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
Preprocess the dataset to parquet format
"""

import argparse
import os
from functools import partial

from datasets import concatenate_datasets, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    question, prompt, ground_truth = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, "question": question},
    }
    return data


def build_aime2024_dataset():
    def process_aime2024(example):
        question, ground_truth = example["Problem"], str(example["Answer"])
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(example_map_fn, process_fn=process_aime2024, data_source="aime24", ability="Math", split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_aime2025_dataset():
    def process_aime2025(example):
        question, ground_truth = example["problem"], str(example["solution"])
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "yentinglin/aime_2025"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(example_map_fn, process_fn=process_aime2025, data_source="aime25", ability="Math", split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_diamond_dataset():
    import random

    GPQA_QUERY_TEMPLATE = (
        "{Question}\n"
        "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        "Please reason step by step, and put your final answer (only the choice letter) within \\boxed{{}}."
    )

    def process_gpqa_diamond(example):
        choices = [
            example["Incorrect Answer 1"].strip(),
            example["Incorrect Answer 2"].strip(),
            example["Incorrect Answer 3"].strip(),
        ]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"].strip())
        question = example["Question"]
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            Question=question,
        )
        gold_choice = "ABCD"[gold_index]
        return question, query_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_gpqa_diamond, data_source="gpqa-diamond", ability="General", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_dapo_train_dataset():
    def process_dapo(example):
        question, ground_truth = example["prompt"], example["solution"]
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "open-r1/DAPO-Math-17k-Processed"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, "all", split="train")
    map_fn = partial(example_map_fn, process_fn=process_dapo, data_source="math-dapo", ability="Math", split="train")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/genrm")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")

    args = parser.parse_args()

    train_dataset = build_dapo_train_dataset()
    train_dataset = concatenate_datasets([train_dataset for _ in range(20)])

    test_datasets = []
    # AIME 2024
    aime24_dataset = build_aime2024_dataset()
    test_datasets.extend([aime24_dataset for _ in range(32)])
    # AIME 2025
    aime25_dataset = build_aime2025_dataset()
    test_datasets.extend([aime25_dataset for _ in range(32)])
    # GPQA Diamond
    gpqa_dataset = build_gpqa_diamond_dataset()
    test_datasets.extend([gpqa_dataset for _ in range(4)])
    test_dataset = concatenate_datasets(test_datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "fapo-train-boxed.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "fapo-test-full-boxed.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
