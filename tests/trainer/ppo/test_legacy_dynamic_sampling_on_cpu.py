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
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

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


@pytest.mark.parametrize("filter_groups", [False, True])
@pytest.mark.parametrize("balance_batch", [False, True])
def test_fit_preserves_reward_response_alignment(monkeypatch, filter_groups, balance_batch):
    """Exercise real fit/filter/balance/GRPO code, stopping before worker training."""
    from verl.trainer.ppo import ray_trainer

    class ReachedAdvantages(Exception):
        pass

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": False,
                "total_epochs": 1,
                "balance_batch": balance_batch,
            },
            "data": {"train_batch_size": 2},
            "algorithm": {
                "adv_estimator": "grpo",
                "use_kl_in_reward": False,
                "gamma": 1.0,
                "lam": 1.0,
                "filter_groups": {"enable": filter_groups, "metric": "acc", "max_num_gen_batches": 1},
            },
            "actor_rollout_ref": {
                "rollout": {"n": 4, "temperature": 1.0},
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                },
            },
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
        }
    )
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._load_checkpoint = Mock()
    trainer._start_profiling = Mock()
    trainer.checkpoint_manager = Mock()
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer._get_dp_size = lambda *_: 2
    trainer.use_rm = trainer.use_reference_policy = trainer.use_critic = False
    trainer.total_training_steps = 1
    trainer.train_dataloader = [{"input_ids": torch.ones(2, 1, dtype=torch.long)}]
    trainer._get_gen_batch = lambda batch: batch
    lengths = torch.tensor([2, 8, 4, 7, 3, 6, 1, 5])
    response_mask = torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1)
    # Different sequence lengths make stale EOS positions observable in addition
    # to the scalar reward/accuracy permutation.
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    rm_scores = torch.zeros(8, 8)
    rm_scores[torch.arange(8), lengths - 1] = rewards
    generated = DataProto.from_dict(
        tensors={
            "responses": torch.arange(8).unsqueeze(1).expand(-1, 8),
            "attention_mask": torch.cat([torch.ones(8, 1), response_mask], dim=1),
            "response_mask": response_mask,
            "rm_scores": rm_scores,
        },
        non_tensors={"acc": rewards.numpy(), "multi_modal_inputs": np.array([{} for _ in range(8)])},
        meta_info={"reward_extra_keys": ["acc"], "timing": {}},
    )
    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=lambda _: generated)
    trainer._compute_old_log_prob = lambda batch: (
        DataProto.from_dict(
            tensors={"old_log_probs": torch.zeros(len(batch), 8), "entropys": torch.zeros(len(batch), 8)}
        ),
        0.0,
    )
    monkeypatch.setattr("verl.utils.tracking.Tracking", lambda **_: Mock())
    monkeypatch.setattr(ray_trainer.SkipManager, "init", lambda _: None)
    monkeypatch.setattr(ray_trainer.SkipManager, "set_step", lambda _: None)
    compute_advantage = ray_trainer.compute_advantage

    def check_advantages(batch, **kwargs):
        ids = batch.batch["responses"][:, 0]
        if balance_batch:
            assert not torch.equal(ids, torch.arange(8)), "fixture must exercise a real permutation"
        torch.testing.assert_close(batch.batch["token_level_scores"], rm_scores[ids])
        torch.testing.assert_close(batch.batch["token_level_rewards"], rm_scores[ids])
        np.testing.assert_array_equal(batch.non_tensor_batch["acc"], rewards[ids].numpy())
        batch = compute_advantage(batch, **kwargs)
        # Each prompt has two successes and two failures. Advantage signs must
        # follow each response's own reward, regardless of its DP position.
        expected_positive = (rewards[ids] > 0).unsqueeze(1).expand(-1, 8)
        mask = batch.batch["response_mask"].bool()
        assert torch.equal((batch.batch["advantages"] > 0)[mask], expected_positive[mask])
        raise ReachedAdvantages

    monkeypatch.setattr(ray_trainer, "compute_advantage", check_advantages)
    with pytest.raises(ReachedAdvantages):
        trainer.fit()


@pytest.mark.parametrize("observed_steps", [None, 2, [1, 2], 1, [1, 1]])
def test_dynamic_sampling_checks_observed_actor_steps(monkeypatch, observed_steps):
    from verl.trainer.ppo import ray_trainer
    from verl.utils import tensordict_utils as tu

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": True}},
            "actor_rollout_ref": {
                "rollout": {"n": 4, "temperature": 1.0, "multi_turn": {"enable": False}},
                "actor": {
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "data_loader_seed": 42,
                    "shuffle": False,
                },
            },
        }
    )
    metrics = {"mfu": 0.0}
    if observed_steps is not None:
        metrics["optimizer_steps"] = observed_steps
    trainer.actor_rollout_wg = SimpleNamespace(
        update_actor=lambda _: tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics})
    )
    monkeypatch.setattr(ray_trainer, "left_right_2_no_padding", lambda data: data)
    batch = DataProto.from_dict(tensors={"responses": torch.zeros(8, 1)})
    if observed_steps is None or not np.all(np.asarray(observed_steps) == 1):
        with pytest.raises(RuntimeError, match="observed optimizer step"):
            trainer._update_actor(batch)
    else:
        output = trainer._update_actor(batch)
        assert output.meta_info["metrics"]["actor/optimizer_steps"] == observed_steps
