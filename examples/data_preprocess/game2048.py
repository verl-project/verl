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
Generate the 2048 GRPO dataset as a parquet file.

Unlike GSM8K (which has real question-answer pairs), the 2048 task uses a
single fixed prompt repeated many times. The model must generate a Python
strategy function that can play 2048 autonomously; the reward comes entirely
from executing that function against the game engine.

Usage:
    python examples/data_preprocess/game2048.py \
        --local_save_dir ~/data/game2048 \
        --train_size 1000 \
        --test_size 100
"""

import argparse
import os

import datasets

DATA_SOURCE = "game2048"

PROMPT = """
Create a new short 2048 strategy using only native Python code.
You are given a 4x4 list of lists of integers representing the current board state (0 means empty).
Output one action for "W" (up), "A" (left), "S" (down), "D" (right) as the optimal next move.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "W" # Example
```
All helper functions must be defined inside `strategy`. Only output the short function `strategy`.
""".strip()

def make_row(idx: int, split: str) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "ability": "game",
        # ground_truth is not used for scoring (reward comes from code execution),
        # but the verl reward manager expects this field to exist.
        "reward_model": {"style": "rule", "ground_truth": "win"},
        "extra_info": {
            "split": split,
            "index": idx,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/game2048")
    parser.add_argument("--train_size", type=int, default=1000,
                        help="Number of prompt copies in the training set.")
    parser.add_argument("--test_size", type=int, default=100,
                        help="Number of prompt copies in the test set.")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_rows = [make_row(i, "train") for i in range(args.train_size)]
    test_rows = [make_row(i, "test") for i in range(args.test_size)]

    train_dataset = datasets.Dataset.from_list(train_rows)
    test_dataset = datasets.Dataset.from_list(test_rows)

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Train: {len(train_dataset)} rows → {train_path}")
    print(f"Test:  {len(test_dataset)} rows  → {test_path}")
    print(f"\nSample row:\n{train_dataset[0]}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
