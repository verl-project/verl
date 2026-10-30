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

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.streaming_dataloader import StreamingDataloader, StreamingDataloaderAsync
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.config import RolloutTqSkipConfig
from verl.utils.skip.rollout_skip import RolloutTqSkip
from verl.utils.skip.skip_manager import SkipManager


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"skip-test-{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
def reset_skip_manager():
    previous = (SkipManager.config, SkipManager.step, SkipManager.skip_instances)
    yield
    SkipManager.config, SkipManager.step, SkipManager.skip_instances = previous


def _config(tmp_path, *, trainer_mode="sync", steps=(1,), parameter_sync_step=1, action="cache"):
    return OmegaConf.create(
        {
            "skip": {
                "rollout_tq": {
                    "enable": True,
                    "dump_dir": str(tmp_path),
                    "steps": list(steps),
                    "action": action,
                }
            },
            "trainer": {
                "experiment_name": "skip-test",
                "project_name": "verl",
                "v1": {
                    "trainer_mode": trainer_mode,
                    trainer_mode: {"parameter_sync_step": parameter_sync_step},
                },
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
            "data": {
                "train_batch_size": 1,
                "gen_batch_size": 1,
                "max_prompt_length": 8,
                "max_response_length": 16,
            },
        }
    )


def _skip(tmp_path, **kwargs):
    config = _config(tmp_path, **kwargs)
    local_config = omega_conf_to_dataclass(config.skip.rollout_tq, RolloutTqSkipConfig)
    return RolloutTqSkip(local_config, config)


def _write_inner(
    skip,
    step,
    inner_idx,
    *,
    uid="cached",
    values=(1.0, 2.0),
    min_global_steps=None,
    max_global_steps=None,
):
    directory = skip._get_v1_inner_dir(step, inner_idx)
    directory.mkdir(parents=True, exist_ok=True)
    keys = [f"{uid}_0_0", f"{uid}_1_0"]
    min_global_steps = step - 1 if min_global_steps is None else min_global_steps
    max_global_steps = step - 1 if max_global_steps is None else max_global_steps
    tags = [
        {
            "global_steps": step,
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
        },
        {
            "global_steps": step,
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
        },
    ]
    torch.save(
        {
            "tensordict": tu.get_tensordict({"cached_value": torch.tensor(values)}),
            "keys": keys,
            "tags": tags,
            "global_steps": step,
        },
        directory / skip.tq_batch_name,
    )
    (directory / skip.meta_name).write_text(json.dumps({"global_steps": step}))


def _clear_partition(partition_id):
    keys = list((tq.kv_list(partition_id=partition_id) or {}).get(partition_id, {}))
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def test_async_steps_must_be_a_contiguous_prefix(tmp_path):
    skip = _skip(tmp_path, trainer_mode="separate_async", steps=(1, 3))

    with pytest.raises(ValueError, match="contiguous prefix"):
        skip.configure_async_run(start_step=1)


def test_async_prefix_is_capture_or_replay_but_never_mixed(tmp_path):
    skip = _skip(tmp_path, trainer_mode="colocate_async", steps=(1, 2))
    assert skip.configure_async_run(start_step=1) == "capture"

    _write_inner(skip, 1, 0)
    with pytest.raises(ValueError, match="fully cached or entirely uncached"):
        skip.configure_async_run(start_step=1)

    _write_inner(skip, 2, 0)
    assert skip.configure_async_run(start_step=1) == "replay"
    assert skip.is_async_replay_step(1)
    assert skip.is_async_replay_step(2)
    assert not skip.is_async_replay_step(3)


def test_async_prefix_rejects_partial_inner_cache(tmp_path):
    skip = _skip(tmp_path, trainer_mode="separate_async", steps=(1,), parameter_sync_step=2)
    _write_inner(skip, 1, 0)

    with pytest.raises(ValueError, match="partial caches"):
        skip.configure_async_run(start_step=1)


def test_cached_sample_loads_one_inner_batch_per_call(tq_init, partition_id, tmp_path):
    skip = _skip(tmp_path, trainer_mode="separate_async", steps=(1,), parameter_sync_step=2)
    _write_inner(skip, 1, 0, values=(1.0, 2.0))
    _write_inner(skip, 1, 1, uid="second", values=(3.0, 4.0))
    assert skip.configure_async_run(start_step=1) == "replay"

    first = skip.load_v1_sample(1, ["fresh-a"], global_steps=1, partition_id=partition_id)
    second = skip.load_v1_sample(1, ["fresh-b"], global_steps=1, partition_id=partition_id)

    assert first.keys == ["fresh-a_0_0", "fresh-a_1_0"]
    assert second.keys == ["fresh-b_0_0", "fresh-b_1_0"]
    assert tq.kv_batch_get(keys=first.keys, partition_id=partition_id)["cached_value"].tolist() == [1.0, 2.0]
    assert tq.kv_batch_get(keys=second.keys, partition_id=partition_id)["cached_value"].tolist() == [3.0, 4.0]
    with pytest.raises(RuntimeError, match="more than 2 sample calls"):
        skip.load_v1_sample(1, ["fresh-c"], global_steps=1, partition_id=partition_id)
    tq.kv_clear(keys=[*first.keys, *second.keys], partition_id=partition_id)


@pytest.mark.parametrize(
    ("action", "target_step", "expected_min", "expected_max"),
    [
        ("cache", 2, 0, 1),
        ("repeat", 5, 3, 4),
    ],
)
def test_replay_preserves_relative_staleness_and_span(
    tq_init,
    partition_id,
    tmp_path,
    action,
    target_step,
    expected_min,
    expected_max,
):
    skip = _skip(tmp_path, steps=(target_step,), action=action)
    _write_inner(skip, 2, 0, min_global_steps=0, max_global_steps=1)

    batch = skip.load_dump_data(
        step=target_step,
        new_prompt_uids=["fresh"],
        n=2,
        global_steps=target_step,
        partition_id=partition_id,
        mark_prompts_finished=False,
    )

    assert {tag["global_steps"] for tag in batch.tags} == {target_step}
    assert {tag["min_global_steps"] for tag in batch.tags} == {expected_min}
    assert {tag["max_global_steps"] for tag in batch.tags} == {expected_max}
    assert {(tag["max_global_steps"] - tag["min_global_steps"] + 1) for tag in batch.tags} == {2}
    assert {(target_step - 1 - tag["max_global_steps"]) for tag in batch.tags} == {0}
    assert {(target_step - 1 - tag["min_global_steps"]) for tag in batch.tags} == {1}
    tq.kv_clear(keys=batch.keys, partition_id=partition_id)


def test_actor_sample_returns_exact_cached_batch_without_original_sampling(tq_init, tmp_path):
    partition_id = "train"
    skip = _skip(tmp_path, trainer_mode="sync", steps=(1,))
    _write_inner(skip, 1, 0)
    SkipManager.skip_instances = {"rollout_tq": skip}

    actor = object.__new__(StreamingDataloader)
    actor.trainer_mode = "sync"
    actor.global_steps = 0
    requested = []
    actor._next_train_batch = lambda count: requested.append(count) or tu.get_tensordict({"uid": ["fresh"]})

    async def fail_request(*_args, **_kwargs):
        raise AssertionError("cache hit must not run the original sample request")

    actor._submit_request = fail_request
    batch, metrics, count = asyncio.run(actor.sample(1, 1, partition_id))

    assert requested == [1]
    assert batch.keys == ["fresh_0_0", "fresh_1_0"]
    assert metrics == {}
    assert count == 2
    assert tq.kv_batch_get(keys=batch.keys, partition_id=partition_id)["cached_value"].tolist() == [1.0, 2.0]
    tq.kv_clear(keys=batch.keys, partition_id=partition_id)


def test_actor_sample_cache_miss_runs_and_saves_original_result(tq_init, tmp_path):
    partition_id = "train"
    skip = _skip(tmp_path, trainer_mode="sync", steps=(1,))
    SkipManager.skip_instances = {"rollout_tq": skip}
    keys = ["generated_0_0", "generated_1_0"]
    tags = [{"global_steps": 1}, {"global_steps": 1}]
    tq.kv_batch_put(
        keys=keys,
        fields=tu.get_tensordict({"cached_value": torch.tensor([5.0, 6.0])}),
        tags=tags,
        partition_id=partition_id,
    )

    actor = object.__new__(StreamingDataloader)
    actor.trainer_mode = "sync"

    async def sample_request(*_args, **_kwargs):
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags), {"metric": 1}, 2

    actor._submit_request = sample_request
    result = asyncio.run(actor.sample(1, 1, partition_id))

    assert result[1:] == ({"metric": 1}, 2)
    assert skip._check_valid_v1_step(1)
    tq.kv_clear(keys=keys, partition_id=partition_id)


def test_cached_train_step_does_not_intercept_validation(tmp_path):
    skip = _skip(tmp_path, trainer_mode="sync", steps=(1,))
    _write_inner(skip, 1, 0)
    SkipManager.skip_instances = {"rollout_tq": skip}
    actor = object.__new__(StreamingDataloader)
    actor.trainer_mode = "sync"
    calls = []

    async def sample_request(*args, **kwargs):
        calls.append((args, kwargs))
        return KVBatchMeta(partition_id="val", keys=[], tags=[]), {}, 0

    actor._submit_request = sample_request
    prompts = tu.get_tensordict({"uid": ["validation"]})
    asyncio.run(actor.sample(1, 1, "val", prompts))

    assert len(calls) == 1


def test_async_replay_gate_stops_refill_only_inside_prefix(tmp_path):
    skip = SimpleNamespace(
        is_enabled=lambda: True,
        is_async_replay_step=lambda step: step <= 2,
    )
    SkipManager.skip_instances = {"rollout_tq": skip}
    actor = object.__new__(StreamingDataloaderAsync)
    actor.global_steps = 1
    actor.default_rollout_n = 2
    actor.watermark_sessions = 4
    actor.max_outstanding_groups = None
    actor.unfinished_session_count = lambda *_args: 0
    requested = []
    actor._refill_exact = lambda count: requested.append(count) or count

    assert actor._refill_to_watermark() == 0
    actor.global_steps = 3
    assert actor._refill_to_watermark() == 2
    assert requested == [2]


def test_async_load_configures_replay_before_starting_coordinator(tq_init, tmp_path):
    _clear_partition("train")
    skip = _skip(tmp_path, trainer_mode="separate_async", steps=(1,))
    _write_inner(skip, 1, 0)
    SkipManager.skip_instances = {"rollout_tq": skip}

    actor = object.__new__(StreamingDataloaderAsync)
    actor.train_dataloader_it = None
    actor._train_state_ready = False
    actor._validation_active = True
    actor._reissue_inflight_prompts = lambda: pytest.fail("cached prefix must not reissue rollout")
    actor._ensure_coordinator = lambda: None
    actor._wakeup = SimpleNamespace(set=lambda: None)

    assert actor.load_checkpoint(None, global_steps=1) == {"streaming_dataloader/reissued_groups": 0.0}
    assert skip.is_async_replay_step(1)
    assert actor._train_state_ready is True
    assert actor._validation_active is False


def test_async_replay_rejects_checkpointed_train_groups(tq_init, tmp_path):
    _clear_partition("train")
    skip = _skip(tmp_path, trainer_mode="colocate_async", steps=(1,))
    _write_inner(skip, 1, 0)
    SkipManager.skip_instances = {"rollout_tq": skip}
    tq.kv_put(key="outstanding", partition_id="train", tag={"is_prompt": True, "status": "pending"})

    actor = object.__new__(StreamingDataloaderAsync)
    actor.train_dataloader_it = None
    with pytest.raises(ValueError, match="empty train TransferQueue"):
        actor.load_checkpoint(None, global_steps=1)
    _clear_partition("train")
