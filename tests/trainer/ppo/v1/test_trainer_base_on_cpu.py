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

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
        }
    )
    return trainer


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "v1": {
                    "sync": {},
                    "sampler": {
                        "custom_sampler": {"path": "custom.py", "name": "CustomSampler"},
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            }
        }
    )

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs


def test_builtin_filter_groups_warns_when_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups(
        {"enable": True, "metric": "acc", "max_num_gen_batches": 10},
    )

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        metric = trainer._resolve_filter_groups_metric()

    assert metric == "acc"
    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "refill continues until enough groups are sampleable.",
        10,
    )


def test_builtin_filter_groups_does_not_warn_without_generation_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        metric = trainer._resolve_filter_groups_metric()

    assert metric == "acc"
    warning.assert_not_called()
