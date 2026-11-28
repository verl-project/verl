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


@pytest.mark.parametrize("kwargs", [{"action": "random"}, {"steps": ["bad"]}, {"enable": 1}, {"load_mode": "unknown"}])
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


@pytest.fixture
def config_cache(config):
    config.skip.rollout_tq.load_mode = "config"
    config.trainer.v1.separate_async.parameter_sync_step = 1
    SkipManager.init(config)
    SkipManager.set_step(1)
    return SkipManager.skip_instances["rollout_tq"]


def save_keys(cache, tq, keys):
    from verl.utils.tensordict_utils import assign_non_tensor_stack

    data = TensorDict({"value": torch.arange(len(keys))}, batch_size=[len(keys)])
    if keys:
        assign_non_tensor_stack(data, "uid", [key.rsplit("_", 2)[0] for key in keys])
        assign_non_tensor_stack(data, "session_id", [int(key.rsplit("_", 2)[1]) for key in keys])
    tq.kv_batch_get.return_value = data
    tags = [{"source_row": i} for i in range(len(keys))]
    cache.prepare_data(1, SimpleNamespace(keys=keys, tags=tags, partition_id="train"), global_steps=1)


@pytest.mark.parametrize(
    "keys, requested_n",
    [
        ([], None),  # No prompt groups.
        (["a_0_0", "a_1_0"], None),  # Too few prompt groups.
        (["a_0_0", "a_1_0", "b_0_0", "b_1_0", "c_0_0", "c_1_0"], None),  # Too many.
        (["a_0_0", "a_0_1", "b_0_0", "b_1_0"], None),  # Right number of rows, missing session.
        (["a_0_0", "a_2_0", "b_0_0", "b_1_0"], None),  # Session IDs disagree with rollout.n.
        (["a_0_0", "a_1_0", "a_1_0", "b_0_0", "b_1_0"], None),  # Duplicate trajectory key.
        pytest.param(["a_0_0", "a_0_00", "a_1_0", "b_0_0", "b_1_0"], None, id="duplicate-numeric-index"),
        (["a_0_0", "a_1_0", "b_0_0", "b_1_0"], [1, 2]),  # A prompt overrides rollout.n.
        (["a_0_0", "a_1_0", "b_0_0", "b_1_0"], 1),  # An override broadcast to every prompt.
    ],
)
def test_config_mismatch_submits_the_original_batch(config_cache, tq, keys, requested_n, caplog):
    save_keys(config_cache, tq, keys)
    batch = {"uid": ["new-a", "new-b"]}
    if requested_n is not None:
        batch["__rollout_n__"] = requested_n

    class Trainer:
        _next_train_batch = Mock(return_value=batch)
        _submit_batch_to_rollout = Mock()

        @SkipManager.annotate_tq(role="rollout_tq", phase="submit")
        def submit(self):
            raise AssertionError("enabled SkipManager should handle submission")

    trainer = Trainer()
    trainer.submit()
    trainer._next_train_batch.assert_called_once_with()
    trainer._submit_batch_to_rollout.assert_called_once_with(batch)
    tq.kv_batch_put.assert_not_called()
    assert "fresh rollout" in caplog.text


def test_config_replays_all_outputs_of_the_configured_sessions(config_cache, tq):
    keys = ["old-b_1_1", "old-a_0_0", "old-b_1_0", "old-a_1_0", "old-b_0_0"]
    save_keys(config_cache, tq, keys)
    assert config_cache.maybe_load_and_inject(1, ["new-b", "new-a"])
    trajectories, prompts = (call.kwargs for call in tq.kv_batch_put.call_args_list)
    assert prompts["keys"] == ["new-b", "new-a"]
    assert all(tag["status"] == "finished" for tag in prompts["tags"])
    expected = [key.replace("old-", "new-") for key in keys]
    rows = trajectories["fields"]["value"].tolist()
    assert sorted(rows) == list(range(len(keys)))
    assert trajectories["keys"] == [expected[i] for i in rows]
    assert list(trajectories["fields"]["uid"]) == [key.rsplit("_", 2)[0] for key in trajectories["keys"]]
    assert list(trajectories["fields"]["session_id"]) == [int(key.rsplit("_", 2)[1]) for key in trajectories["keys"]]
    assert [tag["source_row"] for tag in trajectories["tags"]] == rows


@pytest.mark.parametrize("bad_field", ["uid", "session_id", "keys", "tags"])
def test_config_validates_every_inner_batch_before_writing(config_cache, tq, bad_field):
    from verl.utils.tensordict_utils import assign_non_tensor_stack

    config_cache.parameter_sync_step = 2
    save_keys(config_cache, tq, ["a_0_0", "a_1_0"])
    save_keys(config_cache, tq, ["b_0_0", "b_1_0"])
    path = config_cache._get_v1_inner_dir(1, 1) / config_cache.tq_batch_name
    payload = torch.load(path, weights_only=False)
    if bad_field in ("keys", "tags"):
        payload[bad_field].pop()
    else:
        values = ["a", "b"] if bad_field == "uid" else [1, 1]
        assign_non_tensor_stack(payload["tensordict"], bad_field, values)
    torch.save(payload, path)
    assert not config_cache.maybe_load_and_inject(1, ["new-a", "new-b"])
    tq.kv_batch_put.assert_not_called()


def test_config_repeat_checks_each_request_against_the_selected_dump(config_cache, tq):
    from verl.utils.skip.base_skip import SkipAction

    config_cache.action = SkipAction.REPEAT
    save_batch(config_cache, tq, 1, "old")
    assert not config_cache.maybe_load_and_inject(2, ["a", "b"])
    tq.kv_batch_put.assert_not_called()
    assert config_cache.maybe_load_and_inject(2, ["new"])
    trajectories = tq.kv_batch_put.call_args_list[0].kwargs
    assert trajectories["keys"] == ["new_0_0", "new_1_0"]
    assert list(trajectories["fields"]["uid"]) == ["new", "new"]
    assert all(tag["global_steps"] == 2 for tag in trajectories["tags"])


def test_config_propagates_queue_write_errors(config_cache, tq):
    save_keys(config_cache, tq, ["a_0_0", "a_1_0"])
    tq.kv_batch_put.side_effect = [None, RuntimeError("completion write failed")]
    with pytest.raises(RuntimeError, match="completion write failed"):
        config_cache.maybe_load_and_inject(1, ["new"])


def test_config_replays_broadcast_row_metadata(config_cache, tq):
    from tensordict.tensorclass import NonTensorData

    config_cache.n = 1
    tq.kv_batch_get.return_value = TensorDict(
        {"uid": NonTensorData("old"), "session_id": NonTensorData(0), "value": torch.arange(2)}, batch_size=[2]
    )
    batch = SimpleNamespace(keys=["old_0_0", "old_0_1"], tags=[{}, {}], partition_id="train")
    config_cache.prepare_data(1, batch, global_steps=1)
    assert config_cache.maybe_load_and_inject(1, ["new"])
    trajectories = tq.kv_batch_put.call_args_list[0].kwargs
    assert trajectories["keys"] == ["new_0_0", "new_0_1"]
    assert list(trajectories["fields"]["uid"]) == ["new", "new"]


def save_inner(config_cache, tq, keys, **fields):
    tq.kv_batch_get.return_value = TensorDict(fields, batch_size=[len(keys)])
    batch = SimpleNamespace(keys=keys, tags=[{} for _ in keys], partition_id="train")
    config_cache.prepare_data(1, batch, global_steps=1)


def test_config_replays_broadcast_uid_from_each_inner(config_cache, tq):
    from tensordict.tensorclass import NonTensorData

    config_cache.parameter_sync_step = 2
    for uid in ("a", "b"):
        save_inner(config_cache, tq, [f"{uid}_0_0", f"{uid}_1_0"], uid=NonTensorData(uid), session_id=torch.arange(2))
    assert config_cache.maybe_load_and_inject(1, ["new-a", "new-b"])
    *writes, prompts = (call.kwargs for call in tq.kv_batch_put.call_args_list)
    assert [key for w in writes for key in w["keys"]] == ["new-a_0_0", "new-a_1_0", "new-b_0_0", "new-b_1_0"]
    assert [uid for w in writes for uid in w["fields"].get("uid").tolist()] == ["new-a", "new-a", "new-b", "new-b"]
    assert [s for w in writes for s in w["fields"]["session_id"].tolist()] == [0, 1, 0, 1]
    assert prompts["keys"] == ["new-a", "new-b"]


def test_config_rejects_wrong_broadcast_session_in_second_inner(config_cache, tq, caplog):
    from tensordict.tensorclass import NonTensorData

    config_cache.parameter_sync_step = 2
    config_cache.n = 1
    save_inner(config_cache, tq, ["a_0_0"], session_id=NonTensorData(0))
    save_inner(config_cache, tq, ["b_0_0"], session_id=NonTensorData(1))
    assert not config_cache.maybe_load_and_inject(1, ["new-a", "new-b"])
    tq.kv_batch_put.assert_not_called()
    assert "session_id" in caplog.text


def test_config_preserves_session_column_across_inners(config_cache, tq):
    from tensordict.tensorclass import NonTensorData

    config_cache.parameter_sync_step = 2
    save_inner(config_cache, tq, ["a_0_0", "a_0_1"], uid=NonTensorData("a"), session_id=NonTensorData(0))
    save_inner(config_cache, tq, ["a_1_0"], session_id=NonTensorData(1))
    assert config_cache.maybe_load_and_inject(1, ["new"])
    *writes, prompts = (call.kwargs for call in tq.kv_batch_put.call_args_list)
    assert [key for w in writes for key in w["keys"]] == ["new_0_0", "new_0_1", "new_1_0"]
    assert [s for w in writes for s in w["fields"].get("session_id").tolist()] == [0, 0, 1]
    assert prompts["keys"] == ["new"]


def test_config_recollects_when_inner_trajectory_fields_differ(config_cache, tq, caplog):
    config_cache.parameter_sync_step = 2
    save_inner(config_cache, tq, ["a_0_0", "a_1_0"], responses=torch.arange(2))
    save_inner(config_cache, tq, ["b_0_0", "b_1_0"], rewards=torch.arange(2))
    assert not config_cache.maybe_load_and_inject(1, ["new-a", "new-b"])
    tq.kv_batch_put.assert_not_called()
    assert "different trajectory fields" in caplog.text


@pytest.mark.parametrize("lengths", [((2, 2), (3, 3)), ((2, 2), (2, 3)), ((2, 3), (2, 2))])
def test_config_replays_different_response_lengths_across_inners(config_cache, tq, lengths):
    from tensordict.tensorclass import NonTensorData

    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    config_cache.parameter_sync_step = 2
    for uid, sizes in zip(("a", "b"), lengths, strict=True):
        data = list_of_dict_to_tensordict([{"responses": torch.arange(size)} for size in sizes])
        data["source"] = NonTensorData(uid)
        tq.kv_batch_get.return_value = data
        batch = SimpleNamespace(keys=[f"{uid}_0_0", f"{uid}_1_0"], tags=[{"source": uid}] * 2, partition_id="train")
        config_cache.prepare_data(1, batch, global_steps=1)
    assert config_cache.maybe_load_and_inject(1, ["new-a", "new-b"])
    *writes, prompts = (call.kwargs for call in tq.kv_batch_put.call_args_list)
    rows = [row.tolist() for w in writes for row in w["fields"]["responses"].unbind()]
    assert rows == [list(range(size)) for sizes in lengths for size in sizes]
    assert [s for w in writes for s in w["fields"]["source"]] == ["a", "a", "b", "b"]
    assert [tag["source"] for w in writes for tag in w["tags"]] == ["a", "a", "b", "b"]
    assert prompts["keys"] == ["new-a", "new-b"]


def test_config_does_not_publish_finished_prompts_after_an_inner_write_fails(config_cache, tq):
    config_cache.parameter_sync_step = 2
    for uid in ("a", "b"):
        save_inner(config_cache, tq, [f"{uid}_0_0", f"{uid}_1_0"], responses=torch.arange(2))
    tq.kv_batch_put.side_effect = [None, RuntimeError("second trajectory write failed")]
    with pytest.raises(RuntimeError, match="second trajectory write failed"):
        config_cache.maybe_load_and_inject(1, ["new-a", "new-b"])
    assert [call.kwargs["keys"] for call in tq.kv_batch_put.call_args_list] == [
        ["new-a_0_0", "new-a_1_0"],
        ["new-b_0_0", "new-b_1_0"],
    ]
