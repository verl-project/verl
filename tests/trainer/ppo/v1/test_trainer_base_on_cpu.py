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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_base as trainer_base_module
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, apply_greedy_sampling_params


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _validation_trainer(monkeypatch, partial_kv, *, fail_restore=False):
    """Exercise the real validation loop with CPU queue data and a guarded replica."""
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {},
            "actor_rollout_ref": {"rollout": {"partial_rollout_kv_offload": partial_kv, "val_kwargs": {"n": 1}}},
            "trainer": {},
        }
    )
    trainer.global_steps = 7
    trainer.val_dataloader = [{"raw_prompt": ["question"]} for _ in range(2)]
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    events = []
    state = {"admitting": True, "restored": True}

    def generate(batch):
        assert state["admitting"], "generation admission remained closed after validation"
        events.append("generate")

    def sleep():
        state.update(admitting=not partial_kv, restored=False)
        events.append("sleep")

    def reward(batch):
        assert not state["restored"]
        events.append("reward")
        return batch

    def update(global_steps=None):
        if partial_kv:
            assert global_steps == trainer.global_steps, "KV restoration requires the committed policy version"
        events.append(("update", global_steps))
        if fail_restore:
            raise RuntimeError("weight restore failed")
        state["restored"] = True

    def resume():
        assert state["restored"], "generation resumed before successful restoration"
        state["admitting"] = True
        events.append("resume")

    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=generate)
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=MagicMock(side_effect=sleep),
        update_weights=MagicMock(side_effect=update),
        resume_generation_replicas=MagicMock(side_effect=resume),
    )
    trainer._compute_reward_colocate = reward
    trainer.replay_buffer = SimpleNamespace(
        sample=lambda **kwargs: (SimpleNamespace(keys=["uid_0_0"], partition_id="val"), None)
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: "text")
    trainer._maybe_log_val_generations = MagicMock()
    trainer._val_metrics_update = MagicMock(return_value={"score": 1.0})

    def batch_get(*, select_fields, **kwargs):
        if "prompts" in select_fields:
            return {
                key: SimpleNamespace(to_padded_tensor=lambda **kwargs: torch.tensor([[1]])) for key in select_fields
            }
        return {"uid": np.array(["uid"]), "rm_scores": torch.tensor([[1.0]]), "num_turns": np.array([1])}

    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", MagicMock())
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", batch_get)
    monkeypatch.setattr(trainer_base_module.tq, "kv_clear", MagicMock())
    return trainer, events, state


@pytest.mark.parametrize("partial_kv", [False, True])
def test_validation_restores_version_and_admission_between_batches(monkeypatch, partial_kv):
    trainer, events, state = _validation_trainer(monkeypatch, partial_kv)
    assert trainer._validate() == {"score": 1.0}
    expected = ["generate", "sleep", "reward", ("update", trainer.global_steps)]
    if partial_kv:
        expected.append("resume")
    assert events == expected * 2
    assert state["restored"] and state["admitting"]
    # Training immediately after initial validation must also be able to submit.
    trainer.agent_loop_manager.generate_sequences(None)
    assert events[-1] == "generate"


def test_failed_validation_restore_keeps_partial_kv_admission_closed(monkeypatch):
    trainer, _, state = _validation_trainer(monkeypatch, True, fail_restore=True)
    with pytest.raises(RuntimeError, match="weight restore failed"):
        trainer._validate()
    trainer.checkpoint_manager.resume_generation_replicas.assert_not_called()
    assert not state["admitting"]


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
        patch("verl.utils.dataset.dataloader.logger.warning") as warning,
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
