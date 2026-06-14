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
"""Task and environment definition for digit completion."""

import numpy as np


class DigitCompletion:
    """
    The implementation of a simple digit completion task.
    The prompt is a sequence of numbers with fixed difference. The task is to complete the next N numbers.
    If the max number is reached, the next number should be modulo with max number.

    For example,
    - prompt = [1, 2, 3]
    - N = 5
    - max_number = 6

    the response should be [4, 5, 6, 7%6, 8%6] = [4, 5, 6, 0, 1]

    Note that the tokenizer is char-level to increase the difficulty.
    """

    def __init__(self, max_number: int, max_diff: int, max_num_in_response: int, seed=0):
        """

        Args:
            max_number: the maximum number allowed in the arithmetic sequence
            max_diff: the maximum diff. The actual common diff will be sampled from [0, max_diff]
            max_num_in_response: the maximum number in the response
        """
        super().__init__()
        self.max_number = max_number
        self.max_diff = max_diff
        self.max_num_in_response = max_num_in_response
        assert self.max_num_in_response < 10
        assert self.max_number > 0
        assert self.max_diff > 0
        self.max_number_length = len(str(max_number))
        # {num1},{num2}:{max_num_in_response},{max_number}
        self._prompt_length = self.max_number_length * 2 + 4 + self.max_number_length  # no negative is allowed

        self.np_rng = np.random.default_rng(seed=seed)

    def __str__(self):
        return (
            f"Prompt length: {self.prompt_length}. Response length: {self.response_length}, "
            f"Max number: {self.max_number}. Max diff: {self.max_diff}, "
            f"Max number in response: {self.max_num_in_response}"
        )

    def get_state(self):
        return {"rng": self.np_rng}

    def set_state(self, state):
        assert "rng" in state, "rng must be inside state"
        self.np_rng = state["rng"]

    @property
    def prompt_length(self):
        return self._prompt_length

    @property
    def response_length(self):
        # number length + comma length + [EOS]
        # The actual number times 1.5 to allow 'U'
        return (self.max_num_in_response * self.max_number_length + (self.max_num_in_response - 1) + 1) * 2

    def add(self, a, b):
        return (a + b) % self.max_number

    def get_all_prompts(self):
        all_prompts = []
        for first_num in range(self.max_number + 1):
            for diff in range(0, self.max_diff + 1):
                second_num = self.add(first_num, diff)
                for num_to_complete in range(self.max_num_in_response + 1):
                    prompt = str(first_num) + "," + str(second_num) + f":{self.max_number},{num_to_complete}"
                    all_prompts.append(prompt)
        return all_prompts

    def sample_str_prompts(self):
        # step 1: sample initial numbers
        first_num = self.np_rng.integers(self.max_number + 1)
        diff = self.np_rng.integers(self.max_diff + 1)
        second_num = self.add(first_num, diff)
        num_to_complete = self.np_rng.integers(self.max_num_in_response + 1)
        prompt = str(first_num) + "," + str(second_num) + f":{self.max_number},{num_to_complete}"
        return prompt

    def sample_batch_str_prompts(self, batch_size):
        str_prompts = []
        for _ in range(batch_size):
            str_prompts.append(self.sample_str_prompts())
        return str_prompts


def compute_attention_mask(prompts, pad_token_id):
    mask = np.ones_like(prompts)
    mask[prompts == pad_token_id] = 0
    return mask


def compute_position_id_with_mask(mask):
    return np.clip(np.cumsum(mask, axis=-1) - 1, a_min=0, a_max=None)


def generate_ground_truth_response(prompt: str):
    """Generate ground truth response given a prompt."""
    num, info = prompt.split(":")
    num1, num2 = num.split(",")
    max_number, num_to_gen = info.split(",")
    num1 = int(num1)
    num2 = int(num2)
    max_number = int(max_number)
    num_to_gen = int(num_to_gen)
    diff = (num2 - num1) % max_number
    results = []
    last_num = num2
    for _ in range(num_to_gen):
        curr = (last_num + diff) % max_number
        results.append(str(curr))
        last_num = curr
    response = ",".join(results)
    return response


def compute_reward(prompt: str, response: str, sequence_reward=1.0):
    """We compute dense reward here so that we can directly train RL without SFT"""
    response_length = len(response)
    ground_truth_response = generate_ground_truth_response(prompt)
    per_token_reward = sequence_reward / (len(ground_truth_response) + 1)  # including [EOS]

    # pad
    reward = np.zeros(response_length, dtype=np.float32)  # this assumes that each char is a token
    # assign reward until mismatches
    ground_truth_idx = 0
    for i in range(response_length):
        if ground_truth_idx == len(ground_truth_response):
            break

        ground_truth_response_token = ground_truth_response[ground_truth_idx]
        response_token = response[i]
        if ground_truth_response_token == response_token:
            reward[i] = per_token_reward
            ground_truth_idx += 1
        else:
            # no matches
            break

    return reward, {"ground_truth_response": ground_truth_response}


if __name__ == "__main__":
    task = DigitCompletion(max_number=20, max_diff=3, max_num_in_response=5)
    print(task.sample_str_prompts())

    prompt = "7,8:20,0"
    response = ""
    print(compute_reward(prompt, response))

    prompt = "7,8:20,0"
    response = "E000"
    print(compute_reward(prompt, response))

    prompt = "9,10:20,2"
    response = "11,12,13"
    print(compute_reward(prompt, response))
