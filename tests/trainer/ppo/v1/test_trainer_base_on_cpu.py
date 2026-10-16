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
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


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
                    "trainer_mode": trainer_mode,
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
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.streaming_dataloader.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert len(dataloader.call_args_list) == 1
    assert dataloader.call_args.kwargs["dataset"] == [{}]
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_streaming_dataloader_keeps_train_iterator_out_of_trainer():
    trainer = _trainer_with_filter_groups({"enable": False}, trainer_mode="separate_async")
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
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
    ):
        trainer._init_dataloader()

    assert not hasattr(trainer, "train_dataloader")
    assert len(dataloader.call_args_list) == 1
    assert dataloader.call_args.kwargs["dataset"] == [{}]


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        assert trainer._resolve_filter_groups_metric() == "acc"

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 dataloader; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


@pytest.mark.parametrize("score_centering", [True, False])
def test_update_actor_passes_score_centering_flag(score_centering):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "policy_loss": {"rollout_correction": {"score_centering": score_centering}},
                },
                "rollout": {"n": 4, "temperature": 1.0},
            }
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.0}}
    batch = SimpleNamespace(extra_info={})

    trainer._update_actor(batch, {})

    assert batch.extra_info["score_centering"] is score_centering
