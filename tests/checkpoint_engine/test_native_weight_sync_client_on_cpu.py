# Copyright 2026 Individual Contributor
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
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

# Loading the control-plane module alone avoids initializing the vLLM rollout package on CPU.
module_path = Path(__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/native_weight_sync.py"
spec = importlib.util.spec_from_file_location("native_weight_sync", module_path)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class FakeReplica:
    def __init__(self, index, size):
        self.replica_rank = index
        self.rollout_mode = sync.RolloutMode.STANDALONE
        self.config = SimpleNamespace(disaggregation=SimpleNamespace(enabled=False))
        self.world_size = size
        self.workers = [object() for _ in range(size)]
        self.server_handle = Mock(_actor_id=index)
        self.server_handle.set_global_steps.remote = AsyncMock()
        self.clear_kv_cache = AsyncMock()


def test_assigns_one_rank_interval_and_one_frontend_per_replica(monkeypatch):
    monkeypatch.setattr(sync, "get_rollout_replica_class", lambda _: FakeReplica)
    replicas = [FakeReplica(0, 2), FakeReplica(1, 4)]
    world_size, endpoints = sync.get_native_rollout_endpoints(replicas)
    assert world_size == 7
    assert endpoints == [(replicas[0].server_handle, 1), (replicas[1].server_handle, 3)]
    with pytest.raises(ValueError, match="Duplicate"):
        sync.get_native_rollout_endpoints([replicas[0], replicas[0]])
    replicas[0].workers.pop()
    with pytest.raises(ValueError, match="incomplete"):
        sync.get_native_rollout_endpoints(replicas)
    replicas[0].rollout_mode = sync.RolloutMode.HYBRID
    with pytest.raises(ValueError, match="standalone"):
        sync.get_native_rollout_endpoints(replicas)


def test_submit_all_replicas_before_waiting_and_preserve_rank_offsets(monkeypatch):
    servers = [FakeReplica(0, 2).server_handle, FakeReplica(1, 4).server_handle]
    get = Mock()
    monkeypatch.setattr(sync.ray, "get", get)
    client = sync.RayWeightSyncClient([(servers[0], 1), (servers[1], 3)])
    init_info = {"world_size": 7, "rank_offset": 1}
    update_info = {"names": ["w"], "dtype_names": ["float32"], "shapes": [[2]]}
    client.init_weight_transfer_engine(init_info)
    client.start_weight_update()
    client.update_weights(update_info)
    client.finish_weight_update("42")
    assert init_info["rank_offset"] == 1
    for server, rank_offset in zip(servers, (1, 3), strict=True):
        assert server.collective_rpc.remote.call_args_list == [
            call("init_weight_transfer_engine", kwargs={"init_info": {"world_size": 7, "rank_offset": rank_offset}}),
            call("start_weight_update"),
            call("update_weights", kwargs={"update_info": update_info}),
        ]
        server.finish_native_weight_update.remote.assert_called_once_with("42")
    collective_refs = [server.collective_rpc.remote.return_value for server in servers]
    finish_refs = [server.finish_native_weight_update.remote.return_value for server in servers]
    assert get.call_args_list == [call(collective_refs)] * 3 + [call(finish_refs)]


def test_version_advances_only_after_all_replicas_clear_caches():
    replicas = [FakeReplica(0, 2), FakeReplica(1, 4)]
    replicas[1].clear_kv_cache.side_effect = RuntimeError("cache reset failed")
    with pytest.raises(RuntimeError, match="cache reset failed"):
        asyncio.run(sync.finish_native_rollout_update(replicas, 42))
    for replica in replicas:
        replica.server_handle.set_global_steps.remote.assert_not_called()
    replicas[1].clear_kv_cache.side_effect = None
    asyncio.run(sync.finish_native_rollout_update(replicas, 42))
    for replica in replicas:
        assert replica.clear_kv_cache.await_count == 2
        replica.server_handle.set_global_steps.remote.assert_awaited_once_with(42)


def test_native_backend_opt_in_and_full_weight_validation():
    server = SimpleNamespace(
        rollout_mode=sync.RolloutMode.STANDALONE,
        config=SimpleNamespace(
            disaggregation=SimpleNamespace(enabled=False), quantization=SimpleNamespace(enabled=False), mtp=None
        ),
        model_config=SimpleNamespace(lora_rank=0, lora={}, hf_config=SimpleNamespace()),
    )
    kwargs = {}
    sync.configure_native_hccl(server, kwargs)
    assert kwargs == {"weight_transfer_config": {"backend": "hccl"}}
    with pytest.raises(ValueError, match="backend=hccl"):
        sync.configure_native_hccl(server, {"weight_transfer_config": {"backend": "nccl"}})
    server.model_config.lora = {"rank": 16, "merge": True}
    with pytest.raises(ValueError, match="without LoRA"):
        sync.configure_native_hccl(server, {})
    server.model_config.lora = {}
    server.config.quantization.enabled = True
    with pytest.raises(ValueError, match="unquantized"):
        sync.configure_native_hccl(server, {})
