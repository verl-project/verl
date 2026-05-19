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
from verl.workers.reward_manager.batch import BatchRewardManager
from verl.workers.reward_manager.naive import NaiveRewardManager
from verl.workers.reward_manager.response_utils import select_response_ids_for_reward


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


def test_select_response_ids_for_reward_returns_empty_for_all_zero_response_mask():
    response_ids = torch.tensor([99, 98, 0])
    response_attention_mask = torch.tensor([1, 1, 0])
    response_mask = torch.tensor([0, 0, 0])

    valid_response_ids, reward_index = select_response_ids_for_reward(
        response_ids=response_ids,
        response_attention_mask=response_attention_mask,
        response_mask=response_mask,
    )

    assert valid_response_ids.tolist() == []
    assert reward_index == 0


def test_batch_reward_manager_reuses_filtered_response_for_scoring_and_logging(capsys):
    captured = {}

    def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
        captured["solution_strs"] = solution_strs
        captured["full_response_str"] = extra_infos[0]["full_response_str"]
        return [1.0 if "TOOL_LEAK" not in solution_strs[0] else -1.0]

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
    manager = BatchRewardManager(tokenizer=ToyTokenizer(), num_examine=1, compute_score=compute_score)

    reward = manager(data)
    printed = capsys.readouterr().out

    assert captured["solution_strs"] == ["assistant final"]
    assert captured["full_response_str"] == "assistant final TOOL_LEAK"
    assert "[response] assistant final\n" in printed
    assert "TOOL_LEAK" not in printed
    torch.testing.assert_close(reward, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
