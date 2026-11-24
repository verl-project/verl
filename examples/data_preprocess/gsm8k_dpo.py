# Copyright 2026 Amazon.com Inc and/or its affiliates
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
Build a small preference dataset from GSM8K for DPO smoke tests.

Each row has a plain-string ``prompt``, ``chosen`` and ``rejected``. The chosen response is the
reference solution; the rejected response is the same solution with a wrong final answer. It is
meant for CI and quick end-to-end checks, not as a training recipe.
"""

import argparse
import os
import re

import datasets

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_FINAL_ANSWER = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*$")


def make_wrong_answer(solution: str) -> str:
    """Replace the final '#### N' with '#### N+1' (keeping it an integer when N is one)."""
    match = _FINAL_ANSWER.search(solution)
    if match is None:
        raise ValueError("solution has no final '#### <number>' line")
    value = float(match.group(1).replace(",", ""))
    wrong = value + 1
    wrong_str = str(int(wrong)) if wrong.is_integer() else str(wrong)
    return solution[: match.start(1)] + wrong_str


def to_preference(example):
    answer = example["answer"]
    return {
        "prompt": f"{example['question']} {INSTRUCTION}",
        "chosen": answer,
        "rejected": make_wrong_answer(answer),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Local copy of openai/gsm8k, if any.")
    parser.add_argument("--local_save_dir", default="~/data/gsm8k_dpo", help="Where to write train/test parquet.")
    args = parser.parse_args()

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main")

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for split in ("train", "test"):
        columns = dataset[split].column_names
        dataset[split].map(to_preference, remove_columns=columns).to_parquet(os.path.join(save_dir, f"{split}.parquet"))
