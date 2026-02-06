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
Task description:
Given a random word and a random char, count the number of occurrence of char in the word.

Create CoT dataset that split the word into separate char. Then list the char and count the occurrence.

The word set comes from shakespeare
"""

import os.path
import random

prompt_template = "How many {} are there in word {}?"


def generate_random_char():
    return chr(97 + random.randint(0, 25))


def create_prompt_response(min_length=3, max_length=5):
    # randomly generate a length
    word_length = random.randint(min_length, max_length)
    # randomly generate a target count number. This makes the target number
    target_count_number = random.randint(1, word_length)

    char_lst = []
    # generate the word
    # step 1: generate the target word
    target_char = generate_random_char()

    for _ in range(target_count_number):
        char_lst.append(target_char)

    # step 2: generate other words
    for _ in range(word_length - target_count_number):
        while True:
            char = generate_random_char()
            if char != target_char:
                char_lst.append(char)
                break

    # step 3: random permute char_lst
    random.shuffle(char_lst)

    word = "-".join(char_lst)

    prompt = prompt_template.format(target_char, word)
    final_answer = []

    # cot
    number = 0
    for i, char in enumerate(char_lst):
        cot = f"{char}"
        if char != target_char:
            cot += " != "
        else:
            cot += " = "
            number += 1
        cot += f"{target_char}."

        final_answer.append(cot)

    conclusion = f"\\boxed{{{number}}} {target_char} in {word}."

    final_answer.append(conclusion)

    final_answer = "\n".join(final_answer)

    return prompt, final_answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_number", type=int, default=10000)
    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--data_path", type=str, default="~/data/char_count")

    args = vars(parser.parse_args())

    total_number = args["total_number"]
    min_length = args["min_length"]
    max_length = args["max_length"]
    data_path = args["data_path"]
    data_path = os.path.expanduser(data_path)

    full_output = []
    for _ in range(total_number):
        output = create_prompt_response(min_length=min_length, max_length=max_length)
        full_output.append(output)

    # random reorder
    random.shuffle(full_output)

    # split for train and test
    train_split_len = int(0.9 * len(full_output))
    train_outputs = full_output[:train_split_len]
    test_output = full_output[train_split_len:]

    sft_train_dataset = {"messages": []}

    for o in train_outputs:
        messages = [
            {"role": "user", "content": o[0]},
            {"role": "assistant", "content": o[1]},
        ]

        sft_train_dataset["messages"].append(messages)

    sft_test_dataset = {"messages": []}

    for o in test_output:
        messages = [
            {"role": "user", "content": o[0]},
            {"role": "assistant", "content": o[1]},
        ]
        sft_test_dataset["messages"].append(messages)

    import pandas as pd

    sft_train_dataset = pd.DataFrame(data=sft_train_dataset)
    sft_test_dataset = pd.DataFrame(data=sft_test_dataset)

    folder = os.path.join(data_path, "sft")

    os.makedirs(folder, exist_ok=True)

    sft_train_dataset.to_parquet(os.path.join(folder, "train.parquet"))
    sft_test_dataset.to_parquet(os.path.join(folder, "test.parquet"))

    # build RL dataset
    rl_train_dataset = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}

    rl_test_dataset = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}

    from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

    for o in train_outputs:
        prompt = o[0]
        response = o[1]
        prompt_with_template = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rl_train_dataset["prompt"].append(prompt_with_template)
        rl_train_dataset["data_source"].append("char_count")
        rl_train_dataset["ability"].append("other")
        rl_train_dataset["reward_model"].append(
            {"style": "rule", "ground_truth": remove_boxed(last_boxed_only_string(response))}
        )
        rl_train_dataset["extra_info"].append({"response": response})

    for o in test_output:
        prompt = o[0]
        response = o[1]
        prompt_with_template = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rl_test_dataset["prompt"].append(prompt_with_template)
        rl_test_dataset["data_source"].append("char_count")
        rl_test_dataset["ability"].append("other")
        rl_test_dataset["reward_model"].append(
            {"style": "rule", "ground_truth": remove_boxed(last_boxed_only_string(response))}
        )
        rl_test_dataset["extra_info"].append({"response": response})

    rl_train_dataset = pd.DataFrame(data=rl_train_dataset)
    rl_test_dataset = pd.DataFrame(data=rl_test_dataset)

    folder = os.path.join(data_path, "rl")

    os.makedirs(folder, exist_ok=True)

    rl_train_dataset.to_parquet(os.path.join(folder, "train.parquet"))
    rl_test_dataset.to_parquet(os.path.join(folder, "test.parquet"))
