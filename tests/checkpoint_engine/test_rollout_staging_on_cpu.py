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

"""checkpoint_engine.rollout_staging: the rollout keeps generating during a weight update, so the manager must not
abort its requests, release its KV cache, or undo either afterwards."""

import asyncio
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine import CheckpointEngineManager
from verl.checkpoint_engine import base as checkpoint_engine_base

_QUIESCE = ["abort_replicas", "release_kv_cache_replicas"]
_RESUME = ["resume_kv_cache_replicas", "resume_generation_replicas"]


class _WorkerGroup:
    world_size = 0

    def update_weights(self, **kwargs):
        return []

    def execute_checkpoint_engine(self, *args, **kwargs):
        return []


def _manager(*, support_publish_weights: bool, rollout_staging: bool):
    manager = object.__new__(CheckpointEngineManager)
    manager.config = SimpleNamespace(rollout_staging=rollout_staging)
    manager.backend = "test"
    manager.backend_cls = SimpleNamespace(support_publish_weights=support_publish_weights)
    manager.actor_wg = _WorkerGroup()
    manager.replicas = []
    manager.build_process_group = lambda rollout: None
    manager._background_loop = asyncio.get_running_loop()
    calls = []
    for name in _QUIESCE + _RESUME:

        async def record(name=name):
            calls.append(name)

        setattr(manager, name, record)
    return manager, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("support_publish_weights", [False, True])
@pytest.mark.parametrize("rollout_staging", [False, True])
async def test_only_a_rollout_without_staging_is_quiesced(monkeypatch, support_publish_weights, rollout_staging):
    monkeypatch.setattr(checkpoint_engine_base, "RayWorkerGroup", lambda **kwargs: _WorkerGroup())
    monkeypatch.setattr(checkpoint_engine_base, "RayClassWithInitArgs", lambda **kwargs: None)
    manager, calls = _manager(support_publish_weights=support_publish_weights, rollout_staging=rollout_staging)

    ticket = await manager.submit_update(global_steps=1)
    await ticket.wait_committed()

    assert calls == ([] if rollout_staging else _QUIESCE + _RESUME)


def _replica(supports_weight_staging: bool):
    return type("Replica", (), {"supports_weight_staging": supports_weight_staging})()


def test_staging_is_refused_for_a_replica_that_loads_weights_in_place():
    """Never quiesced, it would be loaded into under running requests."""
    manager = object.__new__(CheckpointEngineManager)
    manager.config = SimpleNamespace(rollout_staging=True)

    manager._check_rollout_staging([_replica(True)])
    with pytest.raises(ValueError, match="rollout_staging=True is not supported by"):
        manager._check_rollout_staging([_replica(True), _replica(False)])


def test_without_staging_any_replica_is_accepted():
    manager = object.__new__(CheckpointEngineManager)
    manager.config = SimpleNamespace(rollout_staging=False)

    manager._check_rollout_staging([_replica(False)])


# ---------------------------------------------------------------- SGLang rollout: the staging session


class _Engine:
    """Records the server calls of SGLangRollout.update_weights."""

    def __init__(self, calls):
        self.calls = calls

    async def begin_weight_update(self, use_staging=False):
        self.calls.append(("begin", use_staging))

    async def end_weight_update(self, weight_version=None, abort=False):
        self.calls.append(("end", weight_version, abort))

    async def flush_cache(self):
        self.calls.append("flush_cache")


def _sglang_rollout(monkeypatch, *, rollout_staging: bool, is_leader: bool = True, fail_at_bucket: int | None = None):
    sglang_rollout = pytest.importorskip("verl.workers.rollout.sglang_rollout.sglang_rollout")
    calls = []

    async def sgl_update_weights(engine, params_batch, **kwargs):
        if calls.count("load") == fail_at_bucket:
            raise RuntimeError("load failed")
        calls.append("load")

    async def one_tensor_per_bucket(weights, bucket_bytes, fusion_groups=()):
        for item in weights:
            yield [item]

    async def ready():
        return None

    async def update_weights_flattened(self, params_batch):
        await sgl_update_weights(None, params_batch)

    monkeypatch.setattr(sglang_rollout, "sgl_update_weights", sgl_update_weights)
    monkeypatch.setattr(sglang_rollout.ServerAdapter, "_update_weights_flattened", update_weights_flattened)
    monkeypatch.setattr(sglang_rollout, "get_named_tensor_buckets", one_tensor_per_bucket)
    monkeypatch.setattr(sglang_rollout, "_uses_quantized_refit", lambda config, model_config: False)
    monkeypatch.setattr(sglang_rollout, "_to_ipc_device", lambda tensor: tensor)  # no GPU here
    rollout = object.__new__(sglang_rollout.ServerAdapter)
    rollout._init_server_adapter = ready
    rollout.config = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(rollout_staging=rollout_staging, update_weights_bucket_megabytes=1)
    )
    rollout.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3"))
    rollout.device_mesh = None
    rollout._engine = _Engine(calls) if is_leader else None
    rollout._is_server_tp_leader = lambda: is_leader
    rollout.server_actor = SimpleNamespace(
        set_global_steps=SimpleNamespace(remote=lambda steps: asyncio.sleep(0, calls.append(("steps", steps))))
    )
    return rollout, calls


def _weights():
    import torch

    return iter([("a", torch.zeros(2)), ("b", torch.zeros(2))])


@pytest.mark.asyncio
async def test_a_staging_update_opens_a_session_loads_every_bucket_and_installs_with_the_version(monkeypatch):
    """The server is generating: no flush, and the version goes with the call that installs the weights."""
    rollout, calls = _sglang_rollout(monkeypatch, rollout_staging=True)

    await rollout.update_weights(_weights(), global_steps=7)

    assert calls == [("begin", True), "load", "load", ("end", "7", False), ("steps", 7)]


@pytest.mark.asyncio
async def test_without_staging_the_update_loads_in_place_and_flushes(monkeypatch):
    rollout, calls = _sglang_rollout(monkeypatch, rollout_staging=False)

    await rollout.update_weights(_weights(), global_steps=7)

    assert calls == ["load", "load", "flush_cache", ("steps", 7)]


@pytest.mark.asyncio
async def test_a_failed_staging_update_is_aborted_so_the_server_keeps_its_weights(monkeypatch):
    rollout, calls = _sglang_rollout(monkeypatch, rollout_staging=True, fail_at_bucket=1)

    with pytest.raises(RuntimeError, match="load failed"):
        await rollout.update_weights(_weights(), global_steps=7)

    assert calls == [("begin", True), "load", ("end", None, True)]


@pytest.mark.asyncio
async def test_only_the_tp_leader_talks_to_the_server_session(monkeypatch):
    """Every rank iterates the weights (the bucket loads gather across ranks); the session is the leader's."""
    rollout, calls = _sglang_rollout(monkeypatch, rollout_staging=True, is_leader=False)

    await rollout.update_weights(_weights(), global_steps=7)

    assert calls == ["load", "load"]


@pytest.mark.asyncio
@pytest.mark.parametrize("nbytes,packed", [(16, True), (3, False)])
async def test_a_staging_bucket_is_packed_unless_that_would_misalign_a_tensor(monkeypatch, nbytes, packed):
    """The server views each tensor in place inside the packed bucket, which needs its offset aligned."""
    import torch

    rollout, _ = _sglang_rollout(monkeypatch, rollout_staging=True)
    ways = []

    async def per_tensor(engine, params_batch, **kwargs):
        ways.append("per tensor")

    async def flattened(params_batch):
        ways.append("packed")

    monkeypatch.setattr("verl.workers.rollout.sglang_rollout.sglang_rollout.sgl_update_weights", per_tensor)
    rollout._update_weights_flattened = flattened

    await rollout.update_weights(iter([("a", torch.zeros(nbytes, dtype=torch.uint8))]), global_steps=1)

    assert ways == ["packed" if packed else "per tensor"]


@pytest.mark.asyncio
async def test_a_flattened_bucket_reaches_the_server_as_one_tensor_per_tp_rank(monkeypatch):
    """Per tensor, a MoE update costs thousands of CUDA IPC handles; the server must still get every tensor."""
    import torch

    sglang_rollout = pytest.importorskip("verl.workers.rollout.sglang_rollout.sglang_rollout")
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    rollout = object.__new__(sglang_rollout.ServerAdapter)
    requests = []

    async def update_weights_from_tensor(request):
        requests.append(request)

    rollout._engine = SimpleNamespace(update_weights_from_tensor=update_weights_from_tensor)
    mesh = SimpleNamespace(
        get_local_rank=lambda: 0,
        mesh=torch.tensor([0, 1]),
        get_group=lambda: None,
    )
    rollout.device_mesh = {"infer_tp": mesh}

    def gather_object(obj, object_gather_list, dst, group):
        object_gather_list[:] = [obj, obj]  # the other rank's payload stands in as a copy of this one

    monkeypatch.setattr(torch.distributed, "gather_object", gather_object)
    # SGLang's reduction patch is for device tensors (it rewrites their CUDA IPC handle); these are on the CPU
    monkeypatch.setattr("sglang.srt.utils.patch_torch.monkey_patch_torch_reductions", lambda: None)
    batch = [("a", torch.arange(6.0).reshape(2, 3)), ("b", torch.tensor([1, 2, 3], dtype=torch.int16))]

    await rollout._update_weights_flattened(batch)

    (request,) = requests
    assert request.load_format == "flattened_bucket" and len(request.serialized_named_tensors) == 2
    payload = MultiprocessingSerializer.deserialize(request.serialized_named_tensors[1])
    rebuilt = dict(
        FlattenedTensorBucket(
            flattened_tensor=payload["flattened_tensor"], metadata=payload["metadata"]
        ).reconstruct_tensors()
    )
    assert list(rebuilt) == ["a", "b"]
    for name, tensor in batch:
        assert rebuilt[name].dtype == tensor.dtype and torch.equal(rebuilt[name], tensor)


@pytest.mark.asyncio
async def test_staging_refuses_updates_that_load_in_place(monkeypatch):
    rollout, _ = _sglang_rollout(monkeypatch, rollout_staging=True)

    with pytest.raises(ValueError, match="rollout_staging stages full bf16 weight tensors"):
        await rollout.update_weights(_weights(), global_steps=7, wire_format="delta_flush")


@pytest.mark.asyncio
async def test_the_http_adapter_does_not_ask_a_generating_server_to_flush(monkeypatch):
    """SGLang's update helper leaves flush_cache at True; the server refuses it in a staging session."""
    http_server_engine = pytest.importorskip("verl.workers.rollout.sglang_rollout.http_server_engine")
    adapter = object.__new__(http_server_engine.AsyncHttpServerAdapter)
    requests = []

    async def make_async_request(endpoint, payload=None, **kwargs):
        requests.append((endpoint, payload))
        return {}

    adapter._make_async_request = make_async_request
    load = SimpleNamespace(serialized_named_tensors=[b"x"], load_format=None, flush_cache=True)

    await adapter.begin_weight_update(use_staging=True)
    await adapter.update_weights_from_tensor(load)
    await adapter.end_weight_update(weight_version="7")
    await adapter.update_weights_from_tensor(load)

    assert [endpoint for endpoint, _ in requests] == [
        "begin_weight_update",
        "update_weights_from_tensor",
        "end_weight_update",
        "update_weights_from_tensor",
    ]
    assert requests[0][1] == {"selector": "all", "sync_base": True, "use_staging": True}
    assert requests[1][1]["flush_cache"] is False
    assert requests[2][1] == {"expected_lora_checksums": None, "weight_version": "7"}
    assert requests[3][1]["flush_cache"] is True
