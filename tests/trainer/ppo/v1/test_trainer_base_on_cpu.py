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
from transfer_queue import KVBatchMeta

import verl.trainer.ppo.v1.trainer_base as trainer_module
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


def test_fit_loads_actor_then_clears_trained_batch_before_checkpoint(monkeypatch):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "fit-order",
                "logger": [],
                "val_before_train": False,
                "total_epochs": 1,
                "save_freq": 1,
                "test_freq": -1,
                "rollout_data_dir": None,
            },
            "global_profiler": {"steps": None},
        }
    )
    trainer.global_steps = 0
    trainer.steps_per_epoch = 1
    trainer.total_training_steps = 1
    trainer._resume_checkpoint_dir = "/resume/global_step_0"
    events = []

    class _RemoteLoad:
        def remote(self, folder, global_steps):
            events.append(("load", folder, global_steps))
            return "load-ref"

    trainer.streaming_dataloader = SimpleNamespace(load_checkpoint=_RemoteLoad())
    trainer.on_train_begin = lambda: events.append(("train_begin",))
    trainer.on_step_begin = lambda: events.append(("step_begin",))
    trainer._start_profiling = lambda: None
    trainer._stop_profiling = lambda: None
    batch = KVBatchMeta(partition_id="train", keys=["prompt_0_0"], tags=[{}])
    trainer.step = lambda _metrics, _timing: events.append(("step",)) or batch
    trainer._fetch_metric_data = lambda value: events.append(("fetch_metrics", value.keys)) or {}
    trainer._save_checkpoint = lambda: events.append(("save",))
    trainer.on_step_end = lambda: events.append(("step_end",))
    trainer._consume_sync_metrics = lambda: {}
    trainer._compute_metrics = lambda *_args, **_kwargs: None
    trainer._shutdown_dump_executor = lambda: events.append(("shutdown",))

    logger = SimpleNamespace(log=lambda **_kwargs: None)
    monkeypatch.setattr(trainer_module, "Tracking", lambda **_kwargs: logger)
    monkeypatch.setattr(trainer_module, "ValidationGenerationsLogger", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        trainer_module, "DapoFilteredRewardTableLogger", lambda **_kwargs: SimpleNamespace(log=lambda *_a: None)
    )
    monkeypatch.setattr(trainer_module.ray, "get", lambda value: events.append(("ray_get", value)) or value)
    monkeypatch.setattr(trainer_module.SkipManager, "init", lambda _config: None)
    monkeypatch.setattr(trainer_module.SkipManager, "set_step", lambda _step: None)
    monkeypatch.setattr(
        trainer_module.tq,
        "kv_clear",
        lambda *, keys, partition_id: events.append(("clear", keys, partition_id)),
    )
    monkeypatch.setattr(trainer_module, "pprint", lambda *_args, **_kwargs: None)

    trainer.fit(streaming_dataloader=trainer.streaming_dataloader)

    assert events.index(("load", "/resume/global_step_0", 1)) < events.index(("step",))
    assert events.index(("fetch_metrics", ["prompt_0_0"])) < events.index(("clear", ["prompt_0_0"], "train"))
    assert events.index(("clear", ["prompt_0_0"], "train")) < events.index(("save",))
