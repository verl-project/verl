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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _trainer(train_batch_size: int = 2, rollout_n: int = 4) -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=train_batch_size),
        algorithm=SimpleNamespace(filter_groups=SimpleNamespace(metric="acc")),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=rollout_n)),
    )
    return trainer


def _batch(group_rewards: list[list[float]], prefix: str) -> DataProto:
    rollout_n = len(group_rewards[0])
    flat_rewards = np.asarray(group_rewards, dtype=np.float32).reshape(-1)
    uids = np.repeat([f"{prefix}-{index}" for index in range(len(group_rewards))], rollout_n)
    return DataProto.from_dict(
        tensors={
            "rm_scores": torch.as_tensor(flat_rewards).unsqueeze(-1),
            "token_level_scores": torch.as_tensor(flat_rewards).unsqueeze(-1),
            "token_level_rewards": torch.as_tensor(flat_rewards).unsqueeze(-1),
        },
        non_tensors={"uid": uids, "acc": flat_rewards},
    )


def test_dynamic_sampling_filters_zero_variance_and_selects_one_update_batch():
    trainer = _trainer()
    first = _batch([[0, 0, 0, 0], [0, 1, 0, 1]], "first")
    accumulated, generated, kept = trainer._filter_dynamic_sampling_groups(first, None)
    assert (generated, kept, len(accumulated)) == (2, 1, 4)

    second = _batch([[1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0]], "second")
    accumulated, generated, kept = trainer._filter_dynamic_sampling_groups(second, accumulated)
    assert (generated, kept, len(accumulated)) == (3, 2, 12)

    selected, surplus = trainer._finalize_dynamic_sampling_batch(accumulated)
    assert len(selected) == 2 * 4
    assert surplus == 1
    assert list(dict.fromkeys(selected.non_tensor_batch["uid"])) == ["first-1", "second-1"]


def test_dynamic_sampling_rejects_incomplete_prompt_group():
    trainer = _trainer(rollout_n=4)
    bad = DataProto.from_dict(
        tensors={
            "rm_scores": torch.zeros(3, 1),
            "token_level_scores": torch.zeros(3, 1),
            "token_level_rewards": torch.zeros(3, 1),
        },
        non_tensors={"uid": np.asarray(["same"] * 3), "acc": np.asarray([0.0, 1.0, 0.0])},
    )
    with pytest.raises(ValueError, match="complete prompt groups"):
        trainer._filter_dynamic_sampling_groups(bad, None)
