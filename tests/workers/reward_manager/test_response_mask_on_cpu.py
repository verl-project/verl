# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager


class ToyTokenizer:
    def __init__(self):
        self.vocab = {1: "prompt", 10: "assistant", 11: "final", 99: "TOOL_LEAK", 0: "<pad>"}

    def decode(self, token_ids, skip_special_tokens=True):
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        pieces = [self.vocab[int(token_id)] for token_id in ids]
        if skip_special_tokens:
            pieces = [piece for piece in pieces if piece != "<pad>"]
        return " ".join(pieces)


def test_naive_reward_manager_uses_response_mask_for_tool_observations():
    captured = {}

    def compute_score(data_source, solution_str, ground_truth, extra_info):
        captured["solution_str"] = solution_str
        captured["full_response_str"] = extra_info["full_response_str"]
        return 1.0 if "TOOL_LEAK" not in solution_str else -1.0

    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[10, 11, 99, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0, 0]]),
        },
        non_tensors={
            "data_source": np.array(["tool_task"], dtype=object),
            "reward_model": np.array([{"ground_truth": "final"}], dtype=object),
        },
    )
    manager = NaiveRewardManager(tokenizer=ToyTokenizer(), num_examine=0, compute_score=compute_score)

    reward = manager(data)

    assert captured["solution_str"] == "assistant final"
    assert captured["full_response_str"] == "assistant final TOOL_LEAK"
    torch.testing.assert_close(reward, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
