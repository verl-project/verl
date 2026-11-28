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
"""CPU unit tests for the checkpoint-engine first-init timeout guardrail (#6967).

Most of these exercise ``wait_for_group_init`` -- the controller-side bound that
``CheckpointEngineManager.build_process_group()`` wraps around its
``ray.get`` of the workers' ``init_process_group`` futures. ``ray.get`` is
mocked so they run on CPU with no Ray cluster: a hung init must become a
fast, clear ``CheckpointEngineInitError``; a normal init must pass through; a
real init error must propagate unchanged; the env var must set the bound; and
an invalid timeout must be rejected before any wait starts.

The ``build_process_group`` test runs a local Ray cluster with stand-in worker
groups, to check the bound covers the real init wait on both the actor and the
rollout side.
"""

import time
from unittest.mock import patch

import pytest
import ray

from verl.checkpoint_engine._group_init import (
    CheckpointEngineInitError,
    wait_for_group_init,
)
from verl.checkpoint_engine.base import CheckpointEngineManager


def test_returns_results_when_init_completes():
    with patch("verl.checkpoint_engine._group_init.ray.get", return_value=["ok0", "ok1"]) as mock_get:
        assert wait_for_group_init(["ref0", "ref1"], timeout_s=5.0) == ["ok0", "ok1"]
        mock_get.assert_called_once()


def test_raises_clear_error_on_timeout():
    with patch(
        "verl.checkpoint_engine._group_init.ray.get",
        side_effect=ray.exceptions.GetTimeoutError("timed out"),
    ):
        with pytest.raises(CheckpointEngineInitError, match="#6967"):
            wait_for_group_init(["ref"], timeout_s=0.1)


def test_reraises_non_timeout_errors():
    with patch(
        "verl.checkpoint_engine._group_init.ray.get",
        side_effect=RuntimeError("nccl boom"),
    ):
        with pytest.raises(RuntimeError, match="nccl boom"):
            wait_for_group_init(["ref"], timeout_s=5.0)


def test_env_var_sets_default_timeout(monkeypatch):
    monkeypatch.setenv("VERL_CKPT_ENGINE_INIT_TIMEOUT_S", "12.5")
    captured = {}

    def fake_get(refs, timeout=None):
        captured["timeout"] = timeout
        return []

    with patch("verl.checkpoint_engine._group_init.ray.get", side_effect=fake_get):
        wait_for_group_init(["ref"])
    assert captured["timeout"] == 12.5


# ray.get treats timeout=-1 as "wait forever" and cannot convert inf/nan, so these
# must be rejected before any wait starts.
_UNREACHABLE_GET = AssertionError("ray.get must not be called with an invalid timeout")


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", "ten"])
def test_rejects_invalid_env_timeout(monkeypatch, value):
    monkeypatch.setenv("VERL_CKPT_ENGINE_INIT_TIMEOUT_S", value)
    with patch("verl.checkpoint_engine._group_init.ray.get", side_effect=_UNREACHABLE_GET):
        with pytest.raises(ValueError, match="VERL_CKPT_ENGINE_INIT_TIMEOUT_S"):
            wait_for_group_init(["ref"])


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_timeout_argument(timeout_s):
    with patch("verl.checkpoint_engine._group_init.ray.get", side_effect=_UNREACHABLE_GET):
        with pytest.raises(ValueError):
            wait_for_group_init(["ref"], timeout_s=timeout_s)


@pytest.fixture
def ray_local():
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


@ray.remote
def _stalled_init():
    # Far longer than the test's timeout; an unbounded wait would return after it.
    time.sleep(10)


class _FakeBackend:
    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        return {}, {}


class _FakeWorkerGroup:
    """Stands in for a RayWorkerGroup: ``execute_checkpoint_engine`` returns one ObjectRef per rank."""

    def __init__(self, world_size, stall_init):
        self.world_size = world_size
        self.stall_init = stall_init

    def execute_checkpoint_engine(self, method, **kwargs):
        if method[0] == "init_process_group" and self.stall_init:
            return [_stalled_init.remote() for _ in range(self.world_size)]
        return [ray.put({}) for _ in range(self.world_size)]


@pytest.mark.parametrize("stalled_side", ["actor", "rollout"])
def test_build_process_group_fails_fast_when_init_stalls(ray_local, monkeypatch, stalled_side):
    monkeypatch.setenv("VERL_CKPT_ENGINE_INIT_TIMEOUT_S", "0.5")
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend_cls = _FakeBackend
    manager.actor_wg = _FakeWorkerGroup(world_size=2, stall_init=stalled_side == "actor")
    rollout = _FakeWorkerGroup(world_size=2, stall_init=stalled_side == "rollout")

    with pytest.raises(CheckpointEngineInitError):
        manager.build_process_group(rollout)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
