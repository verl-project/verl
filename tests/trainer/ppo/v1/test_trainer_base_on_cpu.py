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

from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


def test_codapo_builds_focused_batch_with_fresh_independent_uids():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create({"algorithm": {"codapo_top_k": 2}})
    prompt_batch = tu.get_tensordict(
        {
            "uid": np.array(["q0", "q1", "q2", "q3"], dtype=object),
            "row": torch.arange(4),
        }
    )
    rollout_data = tu.get_tensordict(
        {
            "uid": np.repeat(np.array(["q0", "q1", "q2", "q3"], dtype=object), 2),
            "codapo_values": torch.tensor([0.1, 0.1, 0.9, 0.9, 0.8, 0.8, 0.2, 0.2]),
        }
    )
    rollout_batch = KVBatchMeta(partition_id="train", keys=[f"key-{row}" for row in range(8)], tags=[{}] * 8)
    with patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", return_value=rollout_data):
        focused_batch = trainer._build_codapo_resampled_batch(prompt_batch, rollout_batch, {})

    focused_uids = [tu.unwrap_non_tensor_data(uid) for uid in focused_batch["uid"]]
    assert focused_batch["row"].tolist() == [1, 2, 1, 2]
    assert len(set(focused_uids)) == 4
    assert set(focused_uids).isdisjoint({"q0", "q1", "q2", "q3"})


def test_codapo_step_uses_pending_focused_batch_before_fetching_new_prompts():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 2}})
    trainer.parameter_sync_step = 1
    trainer._codapo_enabled = True
    trainer._codapo_resampled_batch = None
    trainer.global_steps = 8  # Phase is state-driven, not tied to step parity.
    original_batch = tu.get_tensordict({"uid": np.array(["q0", "q1"], dtype=object)})
    focused_batch = tu.get_tensordict({"uid": np.array(["f0", "f1"], dtype=object)})
    rollout_batch = KVBatchMeta(partition_id="train", keys=["k0", "k1"], tags=[{}, {}])

    with (
        patch.object(trainer, "_next_train_batch", return_value=original_batch) as fetch,
        patch.object(trainer, "_submit_batch_to_rollout") as submit,
        patch.object(trainer, "_step_once", return_value=rollout_batch),
        patch.object(trainer, "_build_codapo_resampled_batch", return_value=focused_batch),
    ):
        trainer.step({}, {})
        trainer.global_steps += 1
        trainer.step({}, {})

    fetch.assert_called_once_with()
    assert [call.args[0] for call in submit.call_args_list] == [original_batch, focused_batch]
    assert trainer._codapo_resampled_batch is None
