# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess GSM8K dataset for AgentBay code_interpreter tool use.

The model solves math problems by writing Python code, executing it in AgentBay
sandbox, and then providing the final answer in #### format.

Usage:
    python gsm8k_agentbay.py --local_save_dir ~/data/gsm8k_agentbay
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

SYSTEM_PROMPT = (
    "You are a math expert. Solve the given math problem step by step.\n"
    "You have access to a `code_interpreter` tool that can execute Python code "
    "in a secure sandbox. Use it to perform calculations.\n\n"
    "Workflow:\n"
    "1. Reason about the problem\n"
    "2. Write Python code to compute the answer using `code_interpreter`\n"
    "3. Read the execution result\n"
    "4. Put your final numerical answer after ####\n\n"
    "Example final answer format: #### 42"
)


def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/gsm8k_agentbay")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "openai/gsm8k"

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)

            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Saved {len(train_dataset)} train / {len(test_dataset)} test examples to {local_save_dir}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
