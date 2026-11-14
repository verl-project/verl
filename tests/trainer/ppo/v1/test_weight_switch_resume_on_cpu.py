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

"""A naive weight update resumes rollout replicas exactly once per switch.

A second resume can admit requests between the two calls; with vLLM MoE and data
parallelism a new DP wave then sends ranks down different branches and the switch hangs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class _Replica:
    def __init__(self):
        self.resumes = 0

    async def resume_generation(self):
        self.resumes += 1


def _naive_manager(replica):
    async def _weights_updated():
        return None

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "naive"
    manager.replicas = [replica]
    manager.actor_wg = SimpleNamespace(update_weights=lambda **kwargs: [_weights_updated()])
    return manager


def _trainer(cls, manager):
    trainer = cls.__new__(cls)
    trainer.checkpoint_manager = manager
    trainer.global_steps = 1
    trainer.timing_raw = {}
    return trainer


def test_naive_update_weights_leaves_resuming_to_the_caller():
    replica = _Replica()
    _naive_manager(replica).update_weights(global_steps=1)
    assert replica.resumes == 0


def test_sync_step_end_resumes_the_replicas_it_aborted_once():
    replica = _Replica()
    _trainer(PPOTrainerSync, _naive_manager(replica)).on_step_end()
    assert replica.resumes == 1


def test_colocate_async_step_end_resumes_once():
    replica = _Replica()
    _trainer(PPOTrainerColocateAsync, _naive_manager(replica)).on_step_end()
    assert replica.resumes == 1


def test_hybrid_switch_to_rollout_resumes_once():
    replica = _Replica()
    trainer = _trainer(PPOTrainerSeparateAsync, _naive_manager(replica))
    trainer._enable_hybrid_replicas = True
    trainer.add_replicas_to_balancer = MagicMock()

    trainer.switch_to_rollout(already_registered=True)

    assert replica.resumes == 1
    trainer.add_replicas_to_balancer.assert_not_called()
