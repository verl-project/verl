# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Run through the scheduler: importing the trainer also imports Ray."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _trainer(*, filtering=True):
    trainer = object.__new__(RayPPOTrainer)
    dataset = list(range(4))
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(17))
    trainer.train_dataloader = StatefulDataLoader(dataset, batch_size=1, num_workers=0, sampler=sampler)
    trainer.config = OmegaConf.create({"algorithm": {"filter_groups": {"enable": filtering}}, "trainer": {}})
    trainer.global_steps = 4  # Divisible by len(loader), despite a refill.
    trainer._data_epoch = 0
    trainer._data_batches_consumed = 0
    return trainer


def _values(iterator):
    return [int(row.item()) for row in iterator]


@pytest.mark.parametrize("legacy", [False, True])
def test_refill_cross_epoch_resume_preserves_prompt_sequence_and_rng(tmp_path, legacy):
    original = _trainer()
    first_epoch = _values(iter(original.train_dataloader))
    iterator = iter(original.train_dataloader)
    first_refill = int(next(iterator).item())
    original._data_epoch = 1
    original._data_batches_consumed = 1  # Five batches, but only four updates.
    if legacy:
        torch.save(original.train_dataloader.state_dict(), tmp_path / "data.pt")
    else:
        original._save_dataloader_checkpoint(str(tmp_path))
    expected = _values(iterator) + _values(iter(original.train_dataloader))
    restored = _trainer()
    if legacy:
        restored.config.trainer.dataloader_resume_epoch = 1
    restored._restore_dataloader_checkpoint(str(tmp_path))
    assert (restored._data_epoch, restored._data_batches_consumed) == (1, 1)
    actual = _values(iter(restored.train_dataloader)) + _values(iter(restored.train_dataloader))
    assert actual == expected
    assert sorted(first_epoch) == [0, 1, 2, 3]
    assert sorted([first_refill] + actual[:3]) == [0, 1, 2, 3]


@pytest.mark.parametrize("finished", [False, True])
def test_real_epoch_boundary_restores_sampler_before_advancing(tmp_path, finished):
    original = _trainer()
    iterator = iter(original.train_dataloader)
    for _ in range(4):
        next(iterator)
    if finished:
        with pytest.raises(StopIteration):
            next(iterator)
    original._data_epoch, original._data_batches_consumed = 0, 4
    original._save_dataloader_checkpoint(str(tmp_path))
    expected = _values(iter(original.train_dataloader))
    restored = _trainer()
    restored._restore_dataloader_checkpoint(str(tmp_path))
    assert restored._data_epoch == int(finished)
    if not finished:
        # Saved immediately after the last batch: finish the restored epoch
        # before creating the next one. Do not replay that last batch.
        assert _values(iter(restored.train_dataloader)) == []
    assert _values(iter(restored.train_dataloader)) == expected


def test_legacy_dynamic_checkpoint_requires_explicit_epoch(tmp_path):
    trainer = _trainer()
    next(iter(trainer.train_dataloader))
    torch.save(trainer.train_dataloader.state_dict(), tmp_path / "data.pt")
    with pytest.raises(ValueError, match="dataloader_resume_epoch"):
        _trainer()._restore_dataloader_checkpoint(str(tmp_path))


def test_legacy_non_dynamic_checkpoint_uses_saved_cursor_at_boundary(tmp_path):
    original = _trainer(filtering=False)
    iterator = iter(original.train_dataloader)
    for _ in range(4):
        next(iterator)
    torch.save(original.train_dataloader.state_dict(), tmp_path / "data.pt")
    expected = _values(iter(original.train_dataloader))
    restored = _trainer(filtering=False)
    restored._restore_dataloader_checkpoint(str(tmp_path))
    assert restored._data_epoch == 0
    assert _values(iter(restored.train_dataloader)) == []
    assert _values(iter(restored.train_dataloader)) == expected


@pytest.mark.parametrize("use_rm,separate", [(False, False), (True, False), (True, True)])
def test_colocated_reward_lifecycle_across_refill(use_rm, separate):
    trainer = _trainer()
    trainer.use_rm = use_rm
    trainer.config.reward = {"reward_model": {"enable_resource_pool": separate}}
    events = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep"), wake_up_replicas=lambda: events.append("wake")
    )
    for _ in range(2):
        trainer._wake_rollout_for_refill()
        events.append("generate")
        trainer._sleep_rollout_before_reward()
        events.append("reward")
    trainer._sleep_rollout_before_training()
    events.append("train")
    if use_rm and not separate:
        assert events == ["generate", "sleep", "reward", "wake", "generate", "sleep", "reward", "train"]
    else:
        assert events == ["generate", "reward", "generate", "reward", "sleep", "train"]
    assert not trainer._rollout_asleep_for_reward


def test_fit_counts_refill_batches_independently_of_updates(monkeypatch):
    from verl.trainer.ppo import ray_trainer

    class ReachedAdvantages(Exception):
        pass

    trainer = _trainer()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": False,
                "total_epochs": 1,
                "balance_batch": False,
            },
            "data": {"train_batch_size": 1},
            "algorithm": {
                "adv_estimator": "grpo",
                "use_kl_in_reward": False,
                "gamma": 1.0,
                "lam": 1.0,
                "filter_groups": {"enable": True, "metric": "acc", "max_num_gen_batches": 2},
            },
            "actor_rollout_ref": {
                "rollout": {"n": 2, "temperature": 1.0},
                "actor": {
                    "ppo_mini_batch_size": 1,
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
    trainer.use_rm = trainer.use_reference_policy = trainer.use_critic = False
    trainer.total_training_steps = 1
    trainer.train_dataloader = [{"input_ids": torch.ones(1, 1, dtype=torch.long)} for _ in range(2)]
    trainer._get_gen_batch = lambda batch: batch
    outputs = []
    for scores in ([0.0, 0.0], [0.0, 1.0]):
        outputs.append(
            DataProto.from_dict(
                tensors={
                    "responses": torch.ones(2, 1, dtype=torch.long),
                    "response_mask": torch.ones(2, 1),
                    "attention_mask": torch.ones(2, 2),
                    "rm_scores": torch.tensor(scores).unsqueeze(1),
                },
                non_tensors={"acc": np.asarray(scores), "multi_modal_inputs": np.array([{}, {}])},
                meta_info={"reward_extra_keys": ["acc"], "timing": {}},
            )
        )
    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=lambda _: outputs.pop(0))
    trainer._compute_old_log_prob = lambda batch: (
        DataProto.from_dict(tensors={"old_log_probs": torch.zeros(2, 1), "entropys": torch.zeros(2, 1)}),
        0.0,
    )
    monkeypatch.setattr("verl.utils.tracking.Tracking", lambda **_: Mock())
    monkeypatch.setattr(ray_trainer.SkipManager, "init", lambda _: None)
    monkeypatch.setattr(ray_trainer.SkipManager, "set_step", lambda _: None)

    def check_progress(batch, **kwargs):
        assert trainer.global_steps == 1
        assert (trainer._data_epoch, trainer._data_batches_consumed) == (0, 2)
        raise ReachedAdvantages

    monkeypatch.setattr(ray_trainer, "compute_advantage", check_progress)
    with pytest.raises(ReachedAdvantages):
        trainer.fit()
