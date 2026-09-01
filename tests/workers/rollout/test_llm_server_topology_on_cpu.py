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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.topology import get_rollout_num_replicas, get_rollout_replica_world_size


def _rollout_config(
    *,
    name: str = "vllm",
    tp: int = 2,
    dp: int = 1,
    pp: int = 1,
    pd_enabled: bool = False,
    prefill_replicas: int = 1,
    decode_replicas: int = 1,
    decode_tp: int | None = None,
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
):
    return SimpleNamespace(
        name=name,
        tensor_model_parallel_size=tp,
        data_parallel_size=dp,
        pipeline_model_parallel_size=pp,
        disaggregation=SimpleNamespace(
            enabled=pd_enabled,
            prefill_replicas=prefill_replicas,
            decode_replicas=decode_replicas,
            decode_tensor_model_parallel_size=decode_tp,
        ),
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        prometheus=SimpleNamespace(enable=False),
        disable_log_stats=True,
    )


@pytest.mark.parametrize(
    "resource_world_size,config_kwargs,expected_replica_world_size,expected_num_replicas",
    [
        (16, {"tp": 2, "dp": 2, "pp": 1}, 4, 4),
        (4, {"tp": 2, "dp": 2, "pp": 1}, 4, 1),
        (
            12,
            {"tp": 2, "pd_enabled": True, "prefill_replicas": 1, "decode_replicas": 2, "decode_tp": 1},
            4,
            3,
        ),
        (
            12,
            {
                "tp": 2,
                "pd_enabled": True,
                "prefill_replicas": 1,
                "decode_replicas": 2,
                "n_gpus_per_node": 6,
            },
            6,
            2,
        ),
    ],
)
def test_valid_topologies(
    resource_world_size,
    config_kwargs,
    expected_replica_world_size,
    expected_num_replicas,
):
    config = _rollout_config(**config_kwargs)

    assert get_rollout_replica_world_size(config) == expected_replica_world_size
    assert get_rollout_num_replicas(config, resource_world_size) == expected_num_replicas


def test_dictconfig_topology():
    config = OmegaConf.create(
        {
            "name": "vllm",
            "tensor_model_parallel_size": 2,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "n_gpus_per_node": 8,
            "disaggregation": {
                "enabled": True,
                "prefill_replicas": 1,
                "decode_replicas": 2,
                "decode_tensor_model_parallel_size": 1,
            },
        }
    )

    assert get_rollout_replica_world_size(config) == 4
    assert get_rollout_num_replicas(config, 8) == 2


@pytest.mark.parametrize(
    "config_kwargs,field",
    [
        ({"tp": 0}, "tensor_model_parallel_size"),
        ({"tp": -1}, "tensor_model_parallel_size"),
        ({"dp": 0}, "data_parallel_size"),
        ({"dp": -1}, "data_parallel_size"),
        ({"pp": 0}, "pipeline_model_parallel_size"),
        ({"pp": -1}, "pipeline_model_parallel_size"),
        ({"pd_enabled": True, "prefill_replicas": 0}, "prefill_replicas"),
        ({"pd_enabled": True, "prefill_replicas": -1}, "prefill_replicas"),
        ({"pd_enabled": True, "decode_replicas": 0}, "decode_replicas"),
        ({"pd_enabled": True, "decode_replicas": -1}, "decode_replicas"),
        ({"pd_enabled": True, "decode_tp": 0}, "decode_tensor_model_parallel_size"),
        ({"pd_enabled": True, "decode_tp": -1}, "decode_tensor_model_parallel_size"),
    ],
)
def test_non_positive_parallel_sizes_are_rejected(config_kwargs, field):
    with pytest.raises(ValueError, match=field):
        get_rollout_replica_world_size(_rollout_config(**config_kwargs))


@pytest.mark.parametrize("resource_world_size", [0, -1])
def test_non_positive_resource_world_size_is_rejected(resource_world_size):
    with pytest.raises(ValueError, match="resource_world_size"):
        get_rollout_num_replicas(_rollout_config(), resource_world_size)


@pytest.mark.parametrize("n_gpus_per_node", [0, -1])
def test_non_positive_gpus_per_node_is_rejected(n_gpus_per_node):
    with pytest.raises(ValueError, match="rollout.n_gpus_per_node"):
        get_rollout_num_replicas(_rollout_config(n_gpus_per_node=n_gpus_per_node), 8)


@pytest.mark.parametrize(
    "resource_world_size,config_kwargs,message",
    [
        (3, {"tp": 2, "dp": 2}, "smaller than replica_world_size=4"),
        (10, {"tp": 2, "dp": 2}, r"replica_world_size=4 \(remainder=2\)"),
        (
            3,
            {"tp": 2, "pd_enabled": True, "decode_replicas": 2, "decode_tp": 1},
            "smaller than replica_world_size=4",
        ),
        (
            10,
            {"tp": 2, "pd_enabled": True, "decode_replicas": 2, "decode_tp": 1},
            r"replica_world_size=4 \(remainder=2\)",
        ),
        (24, {"tp": 6}, "replica_world_size=6 is not node-aligned"),
        (24, {"tp": 12}, "replica_world_size=12 is not node-aligned"),
        (
            24,
            {"tp": 2, "pd_enabled": True, "decode_replicas": 2, "decode_tp": 2},
            "replica_world_size=6 is not node-aligned",
        ),
    ],
)
def test_resource_pool_must_fit_complete_replicas(resource_world_size, config_kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_rollout_num_replicas(_rollout_config(**config_kwargs), resource_world_size)


def test_trtllm_allows_cross_placement_group_replica_slices():
    config = _rollout_config(name="trtllm", tp=6)

    assert get_rollout_num_replicas(config, 24) == 4


def test_allow_empty_only_permits_an_empty_resource_pool():
    config = _rollout_config(tp=2, dp=2)

    assert get_rollout_num_replicas(config, 0, allow_empty=True) == 0
    with pytest.raises(ValueError, match="resource_world_size"):
        get_rollout_num_replicas(config, -1, allow_empty=True)
    with pytest.raises(ValueError, match="smaller than replica_world_size=4"):
        get_rollout_num_replicas(config, 2, allow_empty=True)


def _build_manager(config, events, worker_world_size=None):
    class FakeReplica:
        def __init__(self, replica_rank, **_):
            self.replica_rank = replica_rank
            self._server_address = f"server-{replica_rank}"
            self._server_handle = f"handle-{replica_rank}"
            events.append(("created", replica_rank))

        async def init_hybrid(self, worker_group):
            events.append(("hybrid", self.replica_rank, worker_group.world_size))

        async def init_standalone(self):
            events.append(("standalone", self.replica_rank))

    manager = LLMServerManager.__new__(LLMServerManager)
    manager.rollout_config = config
    manager.model_config = object()
    manager.worker_group = SimpleNamespace(world_size=worker_world_size) if worker_world_size is not None else None
    manager.rollout_resource_pool = None
    manager.start_rank = 0
    manager.rollout_replica_class = FakeReplica
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_world_size,expected_mode", [(None, "standalone"), (8, "hybrid")])
async def test_manager_launches_exact_replica_count(worker_world_size, expected_mode):
    events = []
    manager = _build_manager(_rollout_config(tp=2), events, worker_world_size)

    await manager._initialize_llm_servers(start_rank=3)

    assert [replica.replica_rank for replica in manager.rollout_replicas] == [3, 4, 5, 6]
    assert [event[0] for event in events].count("created") == 4
    assert [event[0] for event in events].count(expected_mode) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_world_size,available_world_size,message",
    [
        (None, 2, "smaller than replica_world_size=4"),
        (2, 2, "smaller than replica_world_size=4"),
        (None, 6, r"replica_world_size=4 \(remainder=2\)"),
        (6, 6, r"replica_world_size=4 \(remainder=2\)"),
    ],
)
async def test_manager_rejects_invalid_topology_before_constructing_replicas(
    worker_world_size,
    available_world_size,
    message,
):
    events = []
    config = _rollout_config(
        tp=2,
        dp=2,
        nnodes=available_world_size // 2,
        n_gpus_per_node=2,
    )
    manager = _build_manager(config, events, worker_world_size)

    with pytest.raises(ValueError, match=message):
        await manager._initialize_llm_servers()

    assert events == []


@pytest.mark.asyncio
async def test_manager_allows_fully_async_empty_standalone_phase():
    events = []
    manager = _build_manager(_rollout_config(nnodes=0), events)

    await manager._initialize_llm_servers(allow_empty=True)

    assert manager.rollout_replicas == []
    assert manager.server_handles == []
    assert manager.server_addresses == []
    assert events == []
