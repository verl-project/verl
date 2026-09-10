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

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _epoch_exhaustion_trainer(**trainer_overrides) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "total_epochs": 1,
                "save_freq": 50,
                **trainer_overrides,
            }
        }
    )
    trainer.global_steps = 1
    return trainer


def _fit_stub_trainer(
    *,
    global_steps: int,
    steps_per_epoch: int,
    total_epochs: int = 1,
    total_training_steps: int = 1000,
    save_freq: int = 50,
) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.parameter_sync_step = 1
    trainer.steps_per_epoch = steps_per_epoch
    trainer.total_training_steps = total_training_steps
    trainer.global_steps = global_steps
    trainer.logger = MagicMock()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": False,
                "total_epochs": total_epochs,
                "save_freq": save_freq,
                "test_freq": 0,
            },
            "global_profiler": {"steps": None},
            "data": {"train_batch_size": 64},
        }
    )
    return trainer


def _run_fit(trainer: _StubTrainer, *, step_side_effect=None):
    batch = MagicMock(keys=[], partition_id="train")

    with (
        patch("verl.trainer.ppo.v1.trainer_base.SkipManager") as skip_manager,
        patch("verl.trainer.ppo.v1.trainer_base.Tracking"),
        patch("verl.trainer.ppo.v1.trainer_base.ValidationGenerationsLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.DapoFilteredRewardTableLogger"),
        patch("verl.trainer.ppo.v1.trainer_base.tqdm", return_value=MagicMock()),
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_clear"),
        patch.object(trainer, "_reissue_inflight_prompts"),
        patch.object(trainer, "on_train_begin"),
        patch.object(trainer, "on_step_begin"),
        patch.object(trainer, "_start_profiling"),
        patch.object(trainer, "_stop_profiling"),
        patch.object(trainer, "_consume_sync_metrics", return_value={}),
        patch.object(trainer, "_compute_metrics"),
        patch.object(trainer, "step", side_effect=step_side_effect or (lambda metrics, timing_raw: batch)),
        patch.object(trainer, "on_train_end") as on_train_end,
        patch.object(trainer, "_shutdown_dump_executor") as shutdown_dump_executor,
        patch.object(trainer, "_save_checkpoint") as save_checkpoint,
    ):
        trainer.fit(agent_loop_manager=MagicMock())
        skip_manager.init.assert_called_once()
        skip_manager.set_step.assert_called()

    return save_checkpoint, on_train_end, shutdown_dump_executor


def test_should_force_save_when_epochs_exhaust_with_completed_steps():
    trainer = _epoch_exhaustion_trainer(save_freq=50)
    trainer.global_steps = 238

    assert trainer._should_force_save_epoch_exhaustion_checkpoint(
        completed_training_steps=1,
        current_epoch=1,
    )


def test_should_not_force_save_on_zero_iteration_resume():
    trainer = _epoch_exhaustion_trainer(save_freq=50)
    trainer.global_steps = 238

    assert not trainer._should_force_save_epoch_exhaustion_checkpoint(
        completed_training_steps=0,
        current_epoch=1,
    )


def test_should_not_force_save_on_save_boundary():
    trainer = _epoch_exhaustion_trainer(save_freq=50)
    trainer.global_steps = 101

    assert not trainer._should_force_save_epoch_exhaustion_checkpoint(
        completed_training_steps=1,
        current_epoch=1,
    )


def test_fit_zero_iteration_resume_does_not_save_or_crash():
    trainer = _fit_stub_trainer(global_steps=237, steps_per_epoch=100, save_freq=50)

    save_checkpoint, on_train_end, shutdown_dump_executor = _run_fit(trainer)

    save_checkpoint.assert_not_called()
    on_train_end.assert_called_once()
    shutdown_dump_executor.assert_called_once()
    assert trainer.global_steps == 238


def test_fit_saves_once_when_epochs_exhaust_off_boundary():
    trainer = _fit_stub_trainer(
        global_steps=0,
        steps_per_epoch=2,
        total_training_steps=100,
        save_freq=3,
    )

    save_checkpoint, on_train_end, shutdown_dump_executor = _run_fit(trainer)

    save_checkpoint.assert_called_once()
    on_train_end.assert_called_once()
    shutdown_dump_executor.assert_called_once()
    assert trainer.global_steps == 3


def test_fit_does_not_duplicate_save_on_save_boundary():
    trainer = _fit_stub_trainer(
        global_steps=0,
        steps_per_epoch=2,
        total_training_steps=100,
        save_freq=2,
    )

    save_checkpoint, on_train_end, shutdown_dump_executor = _run_fit(trainer)

    assert save_checkpoint.call_count == 1
    on_train_end.assert_called_once()
    shutdown_dump_executor.assert_called_once()
    assert trainer.global_steps == 3


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
