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
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("vllm")

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import _get_vllm_port_start, vLLMReplica


def test_tp2_server_actors_on_same_node_receive_unique_vllm_port_starts() -> None:
    ports = []
    for replica_rank in range(2):
        server = SimpleNamespace(
            get_master_address=SimpleNamespace(remote=AsyncMock(return_value=("10.0.0.1", 30000, 30001))),
            launch_server=SimpleNamespace(remote=AsyncMock()),
            get_server_address=SimpleNamespace(remote=AsyncMock(return_value=("10.0.0.1", 8000))),
        )
        server_class = Mock()
        server_class.options.return_value.remote.return_value = server
        replica = object.__new__(vLLMReplica)
        replica.replica_rank = replica_rank
        replica.world_size = 2
        replica.workers = [
            SimpleNamespace(__ray_call__=SimpleNamespace(remote=AsyncMock(return_value=("same-node", gpu_id))))
            for gpu_id in ("0", "1")
        ]
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

        server_class.options.assert_called_once()
        ports.append(server_class.options.call_args.kwargs["runtime_env"]["env_vars"]["VLLM_PORT"])

    assert ports == ["25000", "25032"]


@pytest.mark.parametrize("ranks", [(-1, 0, 1), (0, 0, 0), (0, 1, 1), (1300, 0, 1)])
def test_vllm_port_start_rejects_invalid_or_overflowing_partitions(ranks):
    with pytest.raises(ValueError):
        _get_vllm_port_start(*ranks)
