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
from types import SimpleNamespace

import pytest

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import _get_vllm_port_start, vLLMReplica


class _RemoteResult:
    def __init__(self, value):
        self.value = value

    def remote(self, *_args, **_kwargs):
        async def resolve():
            return self.value

        return resolve()


class _FakeWorker:
    def __init__(self, node_id: str, gpu_id: str):
        self.__ray_call__ = _RemoteResult((node_id, gpu_id))


class _FakeServerActor:
    def __init__(self):
        self.get_master_address = _RemoteResult(("10.0.0.1", 30000, 30001))
        self.launch_server = _RemoteResult(None)
        self.get_server_address = _RemoteResult(("10.0.0.1", 8000))


class _RecordingServerClass:
    def __init__(self):
        self.options_kwargs = []

    def options(self, **kwargs):
        self.options_kwargs.append(kwargs)
        return self

    def remote(self, **_kwargs):
        return _FakeServerActor()


def _launch_fake_tp2_replica(replica_rank: int) -> _RecordingServerClass:
    server_class = _RecordingServerClass()
    replica = object.__new__(vLLMReplica)
    replica.replica_rank = replica_rank
    replica.world_size = 2
    replica.workers = [_FakeWorker("same-node", "0"), _FakeWorker("same-node", "1")]
    replica.nnodes = 1
    replica.gpus_per_replica_node = 2
    replica.is_reward_model = False
    replica.is_teacher_model = False
    replica.name_suffix = ""
    replica.server_class = server_class
    replica.config = SimpleNamespace(max_num_seqs=128)
    replica.model_config = SimpleNamespace()
    replica.rollout_mode = RolloutMode.STANDALONE
    replica.servers = []

    asyncio.run(replica.launch_servers())
    return server_class


def test_tp2_server_actors_on_same_node_receive_unique_vllm_port_starts():
    first = _launch_fake_tp2_replica(replica_rank=0)
    second = _launch_fake_tp2_replica(replica_rank=1)

    first_env = first.options_kwargs[0]["runtime_env"]["env_vars"]
    second_env = second.options_kwargs[0]["runtime_env"]["env_vars"]

    assert first_env["VLLM_PORT"] == "25000"
    assert second_env["VLLM_PORT"] == "25032"
    assert first_env["VLLM_PORT"] != second_env["VLLM_PORT"]


@pytest.mark.parametrize("ranks", [(-1, 0, 1), (0, 0, 0), (0, 1, 1), (1300, 0, 1)])
def test_vllm_port_start_rejects_invalid_or_overflowing_partitions(ranks):
    with pytest.raises(ValueError):
        _get_vllm_port_start(*ranks)
