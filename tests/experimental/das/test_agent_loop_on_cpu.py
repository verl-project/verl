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
"""Manager-side iteration driver and the TQ manager wiring (no Ray cluster)."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tests.experimental.das.fakes import FakeActor, fake_ray_get
from verl.experimental.das import agent_loop as agent_loop_module
from verl.experimental.das.agent_loop import DASIterationDriver, das_config_from
from verl.experimental.das.service import SuffixDataService
from verl.workers.config.das import DASConfig


@pytest.fixture
def store(monkeypatch):
    store = SuffixDataService(DASConfig())
    monkeypatch.setattr(agent_loop_module, "get_or_create_suffix_service", lambda cfg: FakeActor(store))
    monkeypatch.setattr(agent_loop_module, "ray", SimpleNamespace(get=fake_ray_get))
    return store


def make_config(enable=True):
    rollout = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")
    rollout.name = "vllm"
    rollout.das.enable = enable
    return OmegaConf.create({"actor_rollout_ref": {"rollout": rollout, "model": {"path": "x"}}})


def test_das_config_from_whole_config():
    cfg = das_config_from(make_config())
    assert isinstance(cfg, DASConfig) and cfg.enable is True


def test_iteration_driver_advances_once_per_step_and_skips_validate(store):
    driver = DASIterationDriver(DASConfig())
    driver.begin_iteration(3)
    assert store.begin_iteration(0)["iteration"] == 3
    driver.begin_iteration(3)  # re-issued prompts of the same step
    driver.begin_iteration(2)  # stale
    driver.begin_iteration(9, validate=True)  # eval batch
    driver.begin_iteration(None)
    assert store.begin_iteration(0)["iteration"] == 3
    driver.begin_iteration(4)
    assert store.begin_iteration(0)["iteration"] == 4


def test_tq_manager_wires_worker_class_and_step_boundary(store, monkeypatch):
    pytest.importorskip("transfer_queue")
    from tensordict import TensorDict

    from verl.experimental.das.agent_loop_tq import DASAgentLoopManagerTQ, DASAgentLoopWorkerTQ
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
    from verl.utils import tensordict_utils as tu

    dispatched = []
    monkeypatch.setattr(AgentLoopManagerTQ, "generate_sequences", lambda self, prompts: dispatched.append(prompts))

    manager = DASAgentLoopManagerTQ(make_config(), llm_client=None)
    assert manager.agent_loop_workers_class.__ray_actor_class__ is DASAgentLoopWorkerTQ

    batch = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor_data(batch, "global_steps", 5)
    manager.generate_sequences(batch)
    assert store.begin_iteration(0)["iteration"] == 5
    assert len(dispatched) == 1

    val = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor_data(val, "global_steps", 6)
    tu.assign_non_tensor_data(val, "validate", True)
    manager.generate_sequences(val)
    assert store.begin_iteration(0)["iteration"] == 5  # validation never advances the store
    assert len(dispatched) == 2
