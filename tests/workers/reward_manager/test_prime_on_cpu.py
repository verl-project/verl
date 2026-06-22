# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""CPU test that PrimeRewardManager honors a custom reward_fn_key."""

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.prime import PrimeRewardManager


class _DummyTokenizer:
    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["dummy"] * len(token_ids)


# Must be module-level: prime scores via ProcessPoolExecutor, which pickles it.
def _compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0


def test_prime_reward_manager_uses_custom_reward_fn_key():
    batch_size, prompt_len, response_len = 1, 2, 2
    # Note the absence of a "data_source" column: the task key is "ability".
    data = DataProto.from_single_dict(
        {
            "prompts": torch.zeros((batch_size, prompt_len), dtype=torch.long),
            "responses": torch.zeros((batch_size, response_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, prompt_len + response_len), dtype=torch.long),
            "ability": np.array(["math"] * batch_size, dtype=object),
            "reward_model": np.array([{"ground_truth": "1"}] * batch_size, dtype=object),
        }
    )
    manager = PrimeRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=1,
        compute_score=_compute_score,
        reward_fn_key="ability",
    )

    reward_tensor = manager(data)

    assert reward_tensor.shape == (batch_size, response_len)
    assert reward_tensor.sum().item() == 1.0
