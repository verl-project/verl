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

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_main import (
    _set_fully_async_total_training_steps_before_worker_init,
)


def _make_config(actor_total_training_steps=-1):
    return OmegaConf.create(
        {
            "rollout": {
                "total_rollout_steps": 3200,
            },
            "async_training": {
                "require_batches": 2,
                "trigger_parameter_sync_step": 4,
            },
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 100,
                    "optim": {
                        "total_training_steps": actor_total_training_steps,
                    },
                },
            },
        }
    )


def test_sets_actor_total_training_steps_before_workers_build_scheduler():
    config = _make_config()

    total_train_steps = _set_fully_async_total_training_steps_before_worker_init(config)

    assert total_train_steps == 4
    assert config.actor_rollout_ref.actor.optim.total_training_steps == 4


def test_preserves_explicit_actor_total_training_steps():
    config = _make_config(actor_total_training_steps=17)

    total_train_steps = _set_fully_async_total_training_steps_before_worker_init(config)

    assert total_train_steps == 4
    assert config.actor_rollout_ref.actor.optim.total_training_steps == 17
