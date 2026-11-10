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

"""CPU coverage for TransferQueue rollout caching and replay."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils.skip import RolloutTqSkip, SkipManager, SkipManagerConfig
from verl.utils.skip.base_skip import SKIP_REGISTRY
from verl.utils.skip.config import RolloutTqSkipConfig


@pytest.fixture(autouse=True)
def reset_manager():
    SkipManager.config = None
    SkipManager.step = -1
    SkipManager.skip_instances = {}
    yield
    SkipManager.config = None
    SkipManager.step = -1
    SkipManager.skip_instances = {}


@pytest.fixture
def config(tmp_path):
    return OmegaConf.create(
        {
            "skip": {"rollout_tq": {"enable": True, "dump_dir": str(tmp_path), "steps": [1, 2], "action": "cache"}},
            "trainer": {
                "project_name": "test",
                "experiment_name": "cache",
                "v1": {"trainer_mode": "separate_async", "separate_async": {"parameter_sync_step": 2}},
            },
            "data": {"train_batch_size": 2, "gen_batch_size": None, "max_prompt_length": 4, "max_response_length": 8},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )


@pytest.fixture
def cache(config):
    SkipManager.init(config)
    return SkipManager.skip_instances["rollout_tq"]


@pytest.fixture
def tq(monkeypatch):
    module = SimpleNamespace(kv_batch_get=Mock(), kv_batch_put=Mock())
    monkeypatch.setitem(sys.modules, "transfer_queue", module)
    return module


def save_batch(cache, tq, step, uid):
    tq.kv_batch_get.return_value = TensorDict({"value": torch.tensor([[1], [2]])}, batch_size=[2])
    batch = SimpleNamespace(keys=[f"{uid}_0_0", f"{uid}_1_0"], tags=[{"global_steps": 0}] * 2, partition_id="train")
    cache.prepare_data(step, batch, global_steps=step)
    return batch


def test_only_tq_role_registered(config):
    assert set(SKIP_REGISTRY) == {"rollout_tq"}
    assert set(SkipManagerConfig.__dataclass_fields__) >= {"rollout_tq"}
    assert "rollout" not in SkipManagerConfig.__dataclass_fields__
    assert "async_rollout" not in SkipManagerConfig.__dataclass_fields__
    SkipManager.init(config)
    assert set(SkipManager.skip_instances) == {"rollout_tq"}
    assert SkipManager.skip_instances["rollout_tq"].gbs == 2


@pytest.mark.parametrize("kwargs", [{"action": "random"}, {"steps": ["bad"]}, {"enable": 1}])
def test_invalid_config(kwargs):
    with pytest.raises(AssertionError):
        RolloutTqSkipConfig(**kwargs)


def test_cache_path_and_complete_step(cache, tq):
    assert isinstance(cache, RolloutTqSkip)
    assert cache._get_project_dump_dir().name == "GBS2_N2_in4_out8"
    assert not cache.has_v1_cache(1)
    save_batch(cache, tq, 1, "old-a")
    assert not cache.has_v1_cache(1)
    save_batch(cache, tq, 1, "old-b")
    assert cache.has_v1_cache(1)
    assert not cache.should_save(1)
    assert not cache.should_save(2, "val")
    assert cache.should_save(2, "train")
    assert not cache.has_v1_cache(2)


@pytest.mark.parametrize("sync_steps", [1, 2])
def test_cached_trajectories_remapped_and_injected(cache, tq, sync_steps):
    cache.parameter_sync_step = sync_steps
    for i in range(sync_steps):
        save_batch(cache, tq, 1, f"old-{i}")
    assert cache.maybe_load_and_inject(1, [f"new-{i}" for i in range(sync_steps)])
    trajectory_call, prompt_call = tq.kv_batch_put.call_args_list
    args = trajectory_call.kwargs
    assert args["keys"] == [f"new-{i}_{j}_0" for i in range(sync_steps) for j in range(2)]
    assert args["fields"]["value"].flatten().tolist() == [1, 2] * sync_steps
    assert all(tag["min_global_steps"] == tag["max_global_steps"] == 1 for tag in args["tags"])
    assert prompt_call.kwargs["keys"] == [f"new-{i}" for i in range(sync_steps)]
    assert all(tag["status"] == "finished" for tag in prompt_call.kwargs["tags"])


def test_repeat_prefers_exact_then_previous_then_future(cache, tq):
    from verl.utils.skip.base_skip import SkipAction

    cache.action = SkipAction.REPEAT
    cache.parameter_sync_step = 1
    assert cache._resolve_load_step_v1(2) == -1
    save_batch(cache, tq, 1, "a")
    save_batch(cache, tq, 4, "b")
    assert cache._resolve_load_step_v1(4) == 4
    assert cache._resolve_load_step_v1(3) == 1
    assert cache._resolve_load_step_v1(0) == 1


@pytest.mark.parametrize("enabled, step, hit", [(False, 1, False), (True, 3, False), (True, 1, False), (True, 1, True)])
def test_submit_keeps_dataloader_aligned(cache, enabled, step, hit):
    cache.enable = enabled
    cache.maybe_load_and_inject = Mock(return_value=hit)
    SkipManager.set_step(step)

    class Trainer:
        _next_train_batch = Mock(return_value={"uid": ["fresh"]})
        _submit_batch_to_rollout = Mock()

        @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
        def submit(self):
            self._submit_batch_to_rollout(self._next_train_batch())

    trainer = Trainer()
    trainer.submit()
    trainer._next_train_batch.assert_called_once()
    eligible = enabled and step in cache.steps
    assert cache.maybe_load_and_inject.call_count == int(eligible)
    assert trainer._submit_batch_to_rollout.call_count == int(not (eligible and hit))


def test_sample_saves_each_inner_batch_but_not_validation(cache, tq):
    SkipManager.set_step(1)
    tq.kv_batch_get.return_value = TensorDict({"value": torch.tensor([[1], [2]])}, batch_size=[2])

    class Buffer:
        @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
        def sample(self, partition):
            return SimpleNamespace(keys=["a_0_0", "a_1_0"], tags=[{}] * 2, partition_id=partition), {"metric": 1}

    buffer = Buffer()
    buffer.sample("val")
    tq.kv_batch_get.assert_not_called()
    for _ in range(2):
        _, metrics = buffer.sample("train")
        assert metrics == {"metric": 1}
    assert cache.has_v1_cache(1)
    assert tq.kv_batch_get.call_count == 2
    buffer.sample("train")
    assert tq.kv_batch_get.call_count == 2


def test_decorator_before_initialization():
    class Trainer:
        @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
        def submit(self):
            return "normal"

    assert Trainer().submit() == "normal"
