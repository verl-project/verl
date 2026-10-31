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
from typing import NamedTuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_base as trainer_base
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu

GDPO_KEYS = ["acc", "fmt"]


class _Row(NamedTuple):
    uid: int
    session: int
    index: int
    prompt_len: int
    response_len: int
    rewards: tuple[float, float]


# One output per session: two prompts (uids), rollout.n = 2 sessions each.
SINGLE_OUTPUT_ROWS = [
    _Row(0, 0, 0, 3, 3, (1.0, 0.5)),
    _Row(0, 1, 0, 2, 2, (0.0, 1.0)),
    _Row(1, 0, 0, 4, 4, (0.0, 1.0)),
    _Row(1, 1, 0, 1, 1, (1.0, 0.0)),
]
# Session (0, 0) also emits an intermediate output, which the agent loop gives the final output's rewards.
MULTI_OUTPUT_ROWS = [
    _Row(0, 0, 0, 2, 5, (1.0, 0.5)),
    _Row(0, 0, 1, 3, 3, (1.0, 0.5)),
    *SINGLE_OUTPUT_ROWS[1:],
]


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _FakeTQ:
    """In-memory stand-in for the TransferQueue key-value API used by the trainer."""

    def __init__(self, fields: dict):
        self.store = dict(fields)

    def kv_batch_get(self, keys, partition_id, select_fields):
        return tu.get_tensordict({k: self.store[k] for k in select_fields if k in self.store})

    def kv_batch_put(self, keys, partition_id, fields):
        for k in fields.keys():
            self.store[k] = fields[k]
        return SimpleNamespace(keys=keys, partition_id=partition_id, extra_info={})


class _Batch(SimpleNamespace):
    def __len__(self):
        return len(self.keys)


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _make_trainer(colocated_reward: bool):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "gdpo",
                "gdpo_reward_keys": GDPO_KEYS,
                "gamma": 1.0,
                "lam": 1.0,
                "use_kl_in_reward": False,
                "norm_adv_by_std_in_grpo": True,
                "rollout_correction": None,
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=None if colocated_reward else [MagicMock()]
    )
    return trainer


def _key(row: _Row) -> str:
    return f"{row.uid}_{row.session}_{row.index}"


def _rm_scores(rows):
    scores = []
    for row in rows:
        score = torch.zeros(row.response_len)
        score[-1] = sum(row.rewards)
        scores.append(score)
    return scores


def _fake_tq(rows, colocated_reward: bool):
    fields = {
        "uid": torch.tensor([row.uid for row in rows]),
        "response_mask": _nested([torch.ones(row.response_len, dtype=torch.long) for row in rows]),
        "rm_scores": _nested(_rm_scores(rows)),
    }
    if colocated_reward:
        fields.update({key: np.array([row.rewards[k] for row in rows]) for k, key in enumerate(GDPO_KEYS)})
    else:
        fields["extra_fields"] = [
            {"reward_extra_info": {key: row.rewards[k] for k, key in enumerate(GDPO_KEYS)}} for row in rows
        ]
    return _FakeTQ(fields)


def _v0_advantages(rows):
    """GDPO advantages from the v0 trainer's padded [prompt | response] layout, which holds one
    output per session: the session's final one. Every output of a session gets that advantage."""
    final = {}
    for row in rows:
        if (row.uid, row.session) not in final or final[(row.uid, row.session)].index < row.index:
            final[(row.uid, row.session)] = row
    finals = list(final.values())
    max_prompt = max(row.prompt_len for row in finals)
    max_response = max(row.response_len for row in finals)
    prompts = torch.zeros(len(finals), max_prompt, dtype=torch.long)
    attention_mask = torch.zeros(len(finals), max_prompt + max_response, dtype=torch.long)
    response_mask = torch.zeros(len(finals), max_response, dtype=torch.long)
    token_level_rewards = torch.zeros(len(finals), max_response)
    for i, (row, score) in enumerate(zip(finals, _rm_scores(finals), strict=True)):
        attention_mask[i, max_prompt - row.prompt_len : max_prompt + row.response_len] = 1
        response_mask[i, : row.response_len] = 1
        token_level_rewards[i, : row.response_len] = score
    data = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
        },
        non_tensors={
            "uid": np.array([row.uid for row in finals], dtype=object),
            **{key: np.array([row.rewards[k] for row in finals]) for k, key in enumerate(GDPO_KEYS)},
        },
    )
    config = OmegaConf.create({"gdpo_reward_keys": GDPO_KEYS})
    data = compute_advantage(data, adv_estimator="gdpo", num_repeat=2, config=config)
    per_session = {(row.uid, row.session): data.batch["advantages"][i, 0] for i, row in enumerate(finals)}
    return [per_session[(row.uid, row.session)].expand(row.response_len) for row in rows]


@pytest.mark.parametrize("rows", [SINGLE_OUTPUT_ROWS, MULTI_OUTPUT_ROWS], ids=["single_output", "multi_output"])
@pytest.mark.parametrize("colocated_reward", [False, True])
def test_v1_gdpo_matches_v0(monkeypatch, rows, colocated_reward):
    fake_tq = _fake_tq(rows, colocated_reward)
    monkeypatch.setattr(trainer_base, "tq", fake_tq)
    trainer = _make_trainer(colocated_reward)
    batch = _Batch(keys=[_key(row) for row in rows], partition_id="train", extra_info={})

    trainer._compute_advantage(batch, metrics={})

    advantages = fake_tq.store["advantages"].unbind()
    expected = _v0_advantages(rows)
    # GDPO gives each sample a non-trivial advantage here, so a zero or mismatched result would fail.
    assert all(row.abs().sum() > 0 for row in expected)
    for actual_row, expected_row in zip(advantages, expected, strict=True):
        torch.testing.assert_close(actual_row, expected_row)
