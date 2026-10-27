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
from unittest.mock import AsyncMock

import pytest
import torch

pytest.importorskip("vllm")

from verl.checkpoint_engine import base
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout import vllm_rollout
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


@pytest.mark.parametrize(
    "mode,prefix_caching,multimodal,preserve,keep",
    [
        pytest.param(RolloutMode.STANDALONE, True, False, False, False, id="default"),
        (RolloutMode.STANDALONE, False, False, True, True),
        (RolloutMode.STANDALONE, True, False, True, True),
        (RolloutMode.HYBRID, False, False, True, False),
        (RolloutMode.STANDALONE, False, True, True, False),
    ],
)
def test_weight_sync_lifecycle(monkeypatch, mode, prefix_caching, multimodal, preserve, keep):
    events = []
    replicas = []
    for rank in range(2):
        server = object.__new__(vLLMHttpServer)
        server.node_rank = 0
        server.engine = SimpleNamespace(pause_generation=AsyncMock(), resume_generation=AsyncMock())
        server.open_submission_gate = AsyncMock()
        handle = SimpleNamespace(
            pause_generation=SimpleNamespace(remote=server.pause_generation),
            resume_engine_generation=SimpleNamespace(remote=server.resume_engine_generation),
            open_submission_gate=SimpleNamespace(remote=server.open_submission_gate),
        )
        replica = object.__new__(vLLMReplica)
        kwargs = {"preserve_kv_cache": True} if preserve else {}
        replica.config = RolloutConfig(name="vllm", enable_prefix_caching=prefix_caching, **kwargs)
        replica.model_config = SimpleNamespace(lora_rank=0, lora={})
        replica.is_multimodal = multimodal
        replica.rollout_mode = mode
        replica.workers = [rank]
        replica.servers = [handle]
        replica.abort_all_requests = AsyncMock(side_effect=lambda **kw: events.append("abort"))
        replica.release_kv_cache = AsyncMock(side_effect=lambda: events.append("release"))
        replica.resume_kv_cache = AsyncMock(side_effect=lambda: events.append("restore"))
        replicas.append((replica, server))

    class WorkerGroup:
        world_size = 2

        def __init__(self, **kwargs):
            pass

        def update_weights(self, **kwargs):
            if keep:
                assert all(s.engine.pause_generation.await_count == 1 for _, s in replicas)
                assert kwargs == {"global_steps": 8, "preserve_kv_cache": True}
            else:
                assert kwargs == {"global_steps": 8}
            events.append("transfer")
            return [None, None]

        def execute_checkpoint_engine(self, methods):
            assert methods == ["finalize", "finalize"]
            events.append("finalize")
            return [None, None]

    monkeypatch.setattr(base, "RayWorkerGroup", WorkerGroup)
    monkeypatch.setattr(base.ray, "get", lambda refs: refs)
    manager = object.__new__(base.CheckpointEngineManager)
    manager.backend = "nccl"
    manager.actor_wg = WorkerGroup()
    # Sender never receives the internal receiver-only option.
    manager.actor_wg.update_weights = lambda **kw: [dict(sent=kw)]
    manager.replicas = [r for r, _ in replicas]
    manager.build_process_group = lambda rollout: events.append("prepare")

    result = manager.update_weights(global_steps=8)

    assert result == {"sent": {"global_steps": 8, "mode": "nccl"}}
    if keep:
        assert events == ["prepare", "transfer", "finalize", "finalize"]
        for replica, server in replicas:
            replica.abort_all_requests.assert_not_awaited()
            replica.release_kv_cache.assert_not_awaited()
            replica.resume_kv_cache.assert_not_awaited()
            server.engine.pause_generation.assert_awaited_once_with(mode="keep", clear_cache=False)
            server.engine.resume_generation.assert_awaited_once_with()
    else:
        assert events == [
            "abort",
            "abort",
            "release",
            "release",
            "prepare",
            "transfer",
            "finalize",
            "finalize",
            "restore",
            "restore",
        ]


@pytest.mark.parametrize("preserve", [True, False])
@pytest.mark.parametrize("wire_format", ["named_tensors", "delta_flush"])
def test_checkpoint_worker_preserves_request_cache_during_weight_load(monkeypatch, preserve, wire_format):
    cache = torch.tensor([1.0, 2.0])
    request = SimpleNamespace(kv=cache, log_probs=[-0.2], routed_experts=[3], pending=True)
    loaded = []
    version = []
    receiver_kwargs = []
    weights = [("w", torch.tensor([8.0]))]

    async def receive_weights(global_steps=None):
        assert global_steps == 8
        if wire_format == "delta_flush":
            yield weights, True
        else:
            for item in weights:
                yield item

    class Sender:
        def __init__(self, **kwargs):
            pass

        async def async_send_weights(self, payload):
            from verl.workers.rollout.utils import ensure_async_iterator

            loaded.extend([item async for item in ensure_async_iterator(payload)])

    async def clear_cache():
        request.kv = None

    async def set_version(step):
        version.append(step)

    async def execute(method, **kwargs):
        receiver_kwargs.append(kwargs.get("kwargs", {}))

    monkeypatch.setattr(vllm_rollout, "BucketedWeightSender", Sender)
    adapter = object.__new__(vllm_rollout.ServerAdapter)
    adapter.config = RolloutConfig(name="vllm")
    adapter._has_server = True
    adapter.server_handle = SimpleNamespace(
        clear_kv_cache=SimpleNamespace(remote=clear_cache), set_global_steps=SimpleNamespace(remote=set_version)
    )
    adapter.use_shm = False
    adapter.zmq_handle = "unused"
    adapter.replica_rank = adapter.rollout_rank = 0
    adapter._delta_weight_transfer_engine_initialized = False
    adapter._execute_method = execute
    worker = object.__new__(base.CheckpointEngineWorker)
    worker.checkpoint_engine = SimpleNamespace(receive_weights=receive_weights, wire_format=wire_format)
    worker.server_adapter = adapter

    asyncio.run(
        base.CheckpointEngineWorker.update_weights.__wrapped__(worker, global_steps=8, preserve_kv_cache=preserve)
    )

    assert loaded == weights
    assert version == [8]
    assert all("preserve_kv_cache" not in kwargs for kwargs in receiver_kwargs)
    assert (request.kv is cache) if preserve else (request.kv is None)
    assert cache.tolist() == [1.0, 2.0]
    assert request.pending and request.log_probs == [-0.2] and request.routed_experts == [3]
