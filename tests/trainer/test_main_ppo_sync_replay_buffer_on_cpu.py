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

import threading
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from verl.trainer import main_ppo_sync
from verl.trainer.main_ppo_sync import AgentLoopManagerTQ, ReplayBuffer, _get_transfer_queue_cleanup_keys
from verl.utils import tensordict_utils as tu


class FakeKVBatchMeta:
    def __init__(self, partition_id: str, keys: list[str], tags: list[dict]):
        self.partition_id = partition_id
        self.keys = keys
        self.tags = tags
        self.extra_info = {}


@pytest.fixture(autouse=True)
def mock_kv_batch_meta(monkeypatch):
    monkeypatch.setattr(main_ppo_sync, "KVBatchMeta", FakeKVBatchMeta)


@pytest.fixture
def replay_buffer():
    buffer = ReplayBuffer.__new__(ReplayBuffer)
    buffer.partitions = defaultdict(dict)
    buffer.poll_interval = 0
    buffer.lock = threading.Lock()
    return buffer


def test_replay_buffer_returns_complete_rollout_batch(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "train",
        {
            "prompt_a": {"global_steps": 1, "status": "finished", "expected_rollouts": 2},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
            "prompt_a_0_1": {"global_steps": 1, "status": "success"},
            "prompt_a_1_0": {"global_steps": 1, "status": "success"},
        },
    )

    batch = replay_buffer.sample(partition_id="train", global_steps=1)

    assert batch.partition_id == "train"
    assert batch.keys == ["prompt_a_0_0", "prompt_a_0_1", "prompt_a_1_0"]
    assert batch.extra_info["rollout_root_keys"] == ["prompt_a"]


def test_replay_buffer_cleanup_keys_include_success_rows_and_root_markers(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "val",
        {
            "prompt_a": {"global_steps": 1, "status": "finished", "expected_rollouts": 1},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
        },
    )

    batch = replay_buffer.sample(partition_id="val", global_steps=1)

    assert _get_transfer_queue_cleanup_keys(batch) == ["prompt_a_0_0", "prompt_a"]


def test_replay_buffer_allows_two_complete_val_batches_at_same_step_after_cleanup(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "val",
        {
            "prompt_a": {"global_steps": 1, "status": "finished", "expected_rollouts": 1},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
        },
    )
    batch = replay_buffer.sample(partition_id="val", global_steps=1)
    replay_buffer.remove(batch.partition_id, _get_transfer_queue_cleanup_keys(batch))

    replay_buffer.add(
        "val",
        {
            "prompt_b": {"global_steps": 1, "status": "finished", "expected_rollouts": 1},
            "prompt_b_0_0": {"global_steps": 1, "status": "success"},
        },
    )

    batch = replay_buffer.sample(partition_id="val", global_steps=1)

    assert batch.keys == ["prompt_b_0_0"]
    assert batch.extra_info["rollout_root_keys"] == ["prompt_b"]


def test_replay_buffer_fails_on_terminal_prompt_failure(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "train",
        {
            "prompt_a": {"global_steps": 1, "status": "failure", "expected_rollouts": 2},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
        },
    )

    with pytest.raises(RuntimeError, match="terminal failures: prompt_a=failure"):
        replay_buffer.sample(partition_id="train", global_steps=1)


def test_replay_buffer_fails_on_terminal_prompt_failure_without_expected_rollouts(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "train",
        {
            "prompt_a": {"global_steps": 1, "status": "failure"},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
        },
    )

    with pytest.raises(RuntimeError, match="terminal failures: prompt_a=failure"):
        replay_buffer.sample(partition_id="train", global_steps=1)


def test_replay_buffer_fails_when_finished_prompt_has_missing_rollout(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "train",
        {
            "prompt_a": {"global_steps": 1, "status": "finished", "expected_rollouts": 2},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
        },
    )

    with pytest.raises(RuntimeError, match=r"prompt_a: expected 2, got 1"):
        replay_buffer.sample(partition_id="train", global_steps=1)


def test_replay_buffer_counts_unique_sessions_not_output_rows(replay_buffer: ReplayBuffer):
    replay_buffer.add(
        "train",
        {
            "prompt_a": {"global_steps": 1, "status": "finished", "expected_rollouts": 2},
            "prompt_a_0_0": {"global_steps": 1, "status": "success"},
            "prompt_a_0_1": {"global_steps": 1, "status": "success"},
        },
    )

    with pytest.raises(RuntimeError, match=r"prompt_a: expected 2, got 1"):
        replay_buffer.sample(partition_id="train", global_steps=1)


def test_agent_loop_manager_records_default_expected_rollouts():
    manager = AgentLoopManagerTQ.__new__(AgentLoopManagerTQ)
    manager.rollout_config = SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=4))
    prompts = tu.get_tensordict({"uid": np.array(["prompt_a", "prompt_b"], dtype=object)})

    assert manager._get_expected_rollouts(prompts, partition_id="train") == [2, 2]
    assert manager._get_expected_rollouts(prompts, partition_id="val") == [4, 4]


def test_agent_loop_manager_records_per_prompt_expected_rollouts():
    manager = AgentLoopManagerTQ.__new__(AgentLoopManagerTQ)
    manager.rollout_config = SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=4))
    prompts = tu.get_tensordict(
        {
            "uid": np.array(["prompt_a", "prompt_b"], dtype=object),
            "__rollout_n__": np.array([1, 3], dtype=np.int64),
        }
    )

    assert manager._get_expected_rollouts(prompts, partition_id="train") == [1, 3]
