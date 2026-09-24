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

from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.reward_variance_filter import (
    apply_reward_variance_filter,
    get_reward_variance_filter_mask,
)


def test_selects_smallest_stable_variance_mass_prefix():
    rewards = torch.tensor([-3.0, -2.0, 3.0, 2.0, -1.0, 1.0])
    uids = np.array(["high", "medium", "high", "medium", "low", "low"], dtype=object)

    mask, metrics = get_reward_variance_filter_mask(rewards, uids, top_p=0.8, selection_eps=0.0)

    assert mask.tolist() == [True, True, True, True, False, False]
    assert metrics["reward_variance_filtering/num_kept_groups"] == 2.0
    assert metrics["reward_variance_filtering/selected_variance_ratio"] == 26.0 / 28.0


def test_ties_use_first_seen_group_order():
    rewards = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    uids = np.array(["first", "second", "first", "second"], dtype=object)

    mask, _ = get_reward_variance_filter_mask(rewards, uids, top_p=0.5, selection_eps=0.0)

    assert mask.tolist() == [True, False, True, False]


def test_all_zero_batch_is_dropped_unless_explicitly_included():
    rewards = torch.tensor([1.0, 1.0, 2.0, 2.0])
    uids = np.array(["a", "a", "b", "b"], dtype=object)

    excluded, _ = get_reward_variance_filter_mask(rewards, uids)
    included, _ = get_reward_variance_filter_mask(rewards, uids, include_zero=True)

    assert not excluded.any()
    assert included.all()


def test_top_k_keeps_fixed_number_of_highest_variance_groups():
    rewards = torch.tensor([-3.0, -2.0, 3.0, 2.0, -1.0, 1.0])
    uids = np.array(["high", "medium", "high", "medium", "low", "low"], dtype=object)

    mask, metrics = get_reward_variance_filter_mask(rewards, uids, strategy="top_k", top_k=2)

    assert mask.tolist() == [True, True, True, True, False, False]
    assert metrics["reward_variance_filtering/num_kept_groups"] == 2.0


def test_top_k_excludes_zero_variance_groups_unless_included():
    rewards = torch.tensor([-1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    uids = np.array(["signal", "flat_a", "signal", "flat_a", "flat_b", "flat_b"], dtype=object)

    excluded, _ = get_reward_variance_filter_mask(rewards, uids, strategy="top_k", top_k=2)
    included, _ = get_reward_variance_filter_mask(rewards, uids, strategy="top_k", top_k=2, include_zero=True)

    assert excluded.tolist() == [True, False, True, False, False, False]
    assert included.tolist() == [True, True, True, True, False, False]


def test_selection_epsilon_can_drop_a_near_zero_signal_batch():
    rewards = torch.tensor([0.0, 0.1])
    uids = np.array(["tiny", "tiny"], dtype=object)

    mask, _ = get_reward_variance_filter_mask(rewards, uids, top_p=0.9, selection_eps=0.01)

    assert not mask.any()


def test_dataproto_filter_masks_complete_groups_without_mutating_rewards():
    data = DataProto.from_single_dict(
        {
            "token_level_scores": torch.tensor([[-3.0, 0.0], [3.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
            "response_mask": torch.ones(4, 2),
            "uid": np.array(["high", "high", "low", "low"], dtype=object),
        }
    )
    original_scores = data.batch["token_level_scores"].clone()
    config = SimpleNamespace(
        get=lambda name, default=None: {
            "top_p": 0.7,
            "top_k": 1,
            "strategy": "top_p",
            "include_zero": False,
            "variance_ddof": 1,
            "selection_eps": 0.0,
        }.get(name, default)
    )

    filtered, _ = apply_reward_variance_filter(data, config)

    assert filtered.batch["response_mask"].tolist() == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    torch.testing.assert_close(filtered.batch["token_level_scores"], original_scores)
