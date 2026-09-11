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
"""Sample-level dispatch: arithmetic plus a TransferQueue mixed-group case."""

import uuid
from unittest.mock import MagicMock

import pytest
import torch
import transfer_queue as tq

from verl.trainer.ppo.v1.replay_buffer import (
    ReplayBufferAsync,
    SampleLevelReplayBuffer,
    compute_sample_level_dispatch,
    count_inflight_samples,
)


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    return f"test-{uuid.uuid4().hex}"


def _uid() -> str:
    return uuid.uuid4().hex


def _traj(uid: str, session_id: int) -> str:
    return f"{uid}_{session_id}_0"


def test_count_inflight_pending_and_partial_running():
    a, b, c = "aaa", "bbb", "ccc"
    pending = {a}
    running = {b, c}
    trajs = {_traj(b, 0), _traj(c, 0), _traj(c, 1)}
    # a: 2 pending, b: 1 left, c: done -> 3
    assert count_inflight_samples(pending, running, trajs, group_size=2) == 3


def test_dispatch_stops_when_batch_is_ready():
    dispatch, credit = compute_sample_level_dispatch(
        group_size=8,
        max_inflight_samples=64,
        inflight_samples=16,
        sample_credit=32,
        sampleable_groups=8,
        target_groups=8,
    )
    assert dispatch == 0
    assert credit == 32


def test_dispatch_seeds_empty_pipeline_without_spending_credit():
    dispatch, credit = compute_sample_level_dispatch(
        group_size=8,
        max_inflight_samples=64,
        inflight_samples=0,
        sample_credit=5,
        sampleable_groups=0,
        target_groups=8,
    )
    assert dispatch == 8
    assert credit == 5


def test_dispatch_converts_mixed_group_credit():
    # 8 completions from mixed groups == 1 prompt; cap has room for 2.
    dispatch, credit = compute_sample_level_dispatch(
        group_size=8,
        max_inflight_samples=64,
        inflight_samples=48,
        sample_credit=20,
        sampleable_groups=1,
        target_groups=8,
    )
    assert dispatch == 2
    assert credit == 4


def test_dispatch_respects_inflight_cap():
    dispatch, credit = compute_sample_level_dispatch(
        group_size=8,
        max_inflight_samples=64,
        inflight_samples=60,
        sample_credit=80,
        sampleable_groups=0,
        target_groups=8,
    )
    assert dispatch == 0
    assert credit == 80


def test_dispatch_rejects_bad_group_size():
    with pytest.raises(ValueError, match="group_size"):
        compute_sample_level_dispatch(
            group_size=0,
            max_inflight_samples=8,
            inflight_samples=0,
            sample_credit=0,
            sampleable_groups=0,
            target_groups=1,
        )


def test_prompt_level_would_wait_sample_level_does_not(tq_init, partition_id):
    """1+1 completions from two unfinished groups still dispatch one prompt."""
    group_a, group_b = _uid(), _uid()
    for uid, session in ((group_a, 0), (group_b, 0)):
        tq.kv_put(
            key=_traj(uid, session),
            partition_id=partition_id,
            fields={"input_ids": torch.tensor([1])},
            tag={"is_prompt": False, "seq_len": 1, "global_steps": 0},
        )
        tq.kv_put(
            key=uid,
            partition_id=partition_id,
            tag={"is_prompt": True, "status": "running", "global_steps": 0},
        )

    calls: list[int] = []

    def refill_fn(n: int) -> int:
        calls.append(n)
        return n

    rb = SampleLevelReplayBuffer(
        trainer_mode="colocate_async",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        refill_fn=refill_fn,
        group_size=2,
        max_inflight_samples=6,
    )
    try:
        rb._sync_metadata_from_transfer_queue()
        dispatched = rb._maybe_dispatch(partition_id, sampleable_keys=set(), target_count=2)
        assert dispatched == 1
        assert calls == [1]
        assert rb._sample_credit == 0
    finally:
        keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition_id)


def test_build_replay_buffer_wires_sample_level_from_rollout_n():
    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    class _T(PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    trainer = _T.__new__(_T)
    trainer.trainer_mode = "colocate_async"
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": {"enable": False}},
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {"rollout": {"n": 8}},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    "colocate_async": {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 8,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                        "dispatch_mode": "sample_level",
                    },
                }
            },
        }
    )
    trainer._add_prompts_to_generate = lambda n: n
    rb = trainer._build_replay_buffer()
    assert type(rb) is SampleLevelReplayBuffer
    assert rb.group_size == 8
    assert rb.max_inflight_samples == 64
    assert rb.trainer_owns_dispatch is False

    trainer.config.trainer.v1.sampler.dispatch_mode = "batch"
    default_rb = trainer._build_replay_buffer()
    assert type(default_rb) is ReplayBufferAsync
    assert getattr(default_rb, "trainer_owns_dispatch", True) is True


def test_trainer_skips_batch_dump_when_sampler_owns_dispatch():
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    class _T(PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    trainer = _T.__new__(_T)
    trainer.replay_buffer = SampleLevelReplayBuffer.__new__(SampleLevelReplayBuffer)
    trainer.replay_buffer.trainer_owns_dispatch = False
    trainer._add_batch_to_generate = MagicMock()

    assert trainer._should_add_batch_to_generate() is False
    assert trainer.prepare_step() == {}
    trainer._add_batch_to_generate.assert_not_called()
