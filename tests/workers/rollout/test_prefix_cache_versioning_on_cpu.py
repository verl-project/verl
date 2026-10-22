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
"""GPU-free tests for rollout.external_kv_cache_invalidation.

"salt" salts each request with the weight version it runs on. These tests pin the
salt format, that every replica of a run shares one namespace, and that SGLang salts a
request only after it clears the server's pause gate. The vLLM server is covered in
rollout_vllm/test_submitter_gate_on_cpu.py; TRT-LLM by its GPU test.
"""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.workers.config import RolloutConfig
from verl.workers.rollout import replica as replica_module
from verl.workers.rollout.replica import RolloutReplica, prefix_cache_salt


def test_external_kv_cache_invalidation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="external_kv_cache_invalidation"):
        RolloutConfig(name="vllm", external_kv_cache_invalidation="Salt")


@pytest.mark.parametrize("mode,salts", [("reset", [None, None]), ("salt", ["run:3", "run:4"])])
def test_prefix_cache_salt_combines_namespace_and_weight_version(mode, salts):
    config = RolloutConfig(name="vllm", external_kv_cache_invalidation=mode, prefix_cache_namespace="run")
    assert [prefix_cache_salt(config, step) for step in (3, 4)] == salts


class _Replica(RolloutReplica):
    async def launch_servers(self):
        pass


def test_replicas_created_by_one_process_share_a_namespace():
    config = OmegaConf.create(
        {"_target_": "verl.workers.config.RolloutConfig", "name": "vllm", "tensor_model_parallel_size": 1}
    )
    first = _Replica(replica_rank=0, config=config, model_config=None, gpus_per_node=1)
    second = _Replica(replica_rank=1, config=config, model_config=None, gpus_per_node=1)

    assert first.config is not second.config
    assert first.config.prefix_cache_namespace == second.config.prefix_cache_namespace
    assert first.config.prefix_cache_namespace == replica_module._PREFIX_CACHE_NAMESPACE


def _server_config(name, mode="salt"):
    return RolloutConfig(
        name=name,
        tensor_model_parallel_size=1,
        max_model_len=64,
        prompt_length=32,
        response_length=16,
        external_kv_cache_invalidation=mode,
        prefix_cache_namespace="run",
    )


# ---------------------------------------------------------------------------- SGLang


class _FakeTokenizerManager:
    """Mimics TokenizerManager's pause: generate_request waits while paused, then runs."""

    def __init__(self):
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
        self.salts = []

    async def pause_generation(self, obj):
        async with self.is_pause_cond:
            self.is_pause = True

    async def continue_generation(self, obj):
        async with self.is_pause_cond:
            self.is_pause = False
            self.is_pause_cond.notify_all()

    async def generate_request(self, obj, request):
        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)
        self.salts.append(obj.cache_salt)
        yield {"output_ids": [5], "meta_info": {"finish_reason": {"type": "stop"}}}


def _make_sglang_server(monkeypatch, config):
    sglang_server = pytest.importorskip("verl.workers.rollout.sglang_rollout.async_sglang_server")
    monkeypatch.setattr(
        sglang_server.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_name=lambda: "test")
    )
    server = object.__new__(sglang_server.SGLangHttpServer)
    server.config = config
    server.node_rank = 0
    server.global_steps = 7
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server._disaggregation_role = "null"
    server._pd_decode_peers = []
    server._generation_allowed = asyncio.Event()
    server._generation_allowed.set()
    server.tokenizer_manager = _FakeTokenizerManager()
    return server


def test_sglang_request_arriving_during_pause_is_salted_after_resume(monkeypatch):
    async def main():
        server = _make_sglang_server(monkeypatch, _server_config("sglang"))
        await server.generate([1, 2, 3], {}, "before")
        await server.abort_all_requests()

        task = asyncio.create_task(server.generate([1, 2, 3], {}, "parked"))
        await asyncio.sleep(0.05)
        assert not task.done()

        await server.set_global_steps(8)
        await server.resume_generation()
        await asyncio.wait_for(task, timeout=5)

        assert server.tokenizer_manager.salts == ["run:7", "run:8"]

    asyncio.run(main())


def test_sglang_failed_pause_reopens_the_gate(monkeypatch):
    async def main():
        server = _make_sglang_server(monkeypatch, _server_config("sglang"))

        async def failing_pause(obj):
            raise RuntimeError("pause failed")

        server.tokenizer_manager.pause_generation = failing_pause
        with pytest.raises(RuntimeError, match="pause failed"):
            await server.abort_all_requests()

        assert server._generation_allowed.is_set(), "a failed pause must not strand requests behind the gate"

    asyncio.run(main())


# ---------------------------------------------------------------------------- weight updates


@pytest.mark.parametrize("mode,raises", [("reset", False), ("salt", True)])
def test_salt_weight_update_requires_global_steps(mode, raises):
    from verl.checkpoint_engine.base import CheckpointEngineManager

    class _Reached(Exception):
        pass

    async def abort_replicas():
        raise _Reached

    manager = object.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.replicas = [SimpleNamespace(config=_server_config("vllm", mode=mode))]
    manager.abort_replicas = abort_replicas

    # The guard must fire before any replica is aborted; otherwise the update proceeds.
    with pytest.raises(ValueError if raises else _Reached):
        manager.update_weights(global_steps=None)


# ---------------------------------------------------------------------------- config validation


def test_salt_rejects_pd_disaggregation():
    with pytest.raises(ValueError, match="PD disaggregation"):
        RolloutConfig(name="vllm", external_kv_cache_invalidation="salt", disaggregation={"enabled": True})


@pytest.mark.parametrize(
    "transfer,ok",
    [
        ({"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}, True),
        ('{"kv_connector": "OffloadingConnector", "kv_role": "kv_both"}', True),
        ({"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}, False),
        (
            {
                "kv_connector": "MultiConnector",
                "kv_connector_extra_config": {
                    "connectors": [{"kv_connector": "LMCacheMPConnector"}, {"kv_connector": "LMCacheConnectorV1"}]
                },
            },
            False,
        ),
    ],
)
@pytest.mark.parametrize("key", ["kv_transfer_config", "kv-transfer-config"])
def test_salt_vllm_requires_salt_keyed_connectors(transfer, ok, key):
    def build():
        return RolloutConfig(
            name="vllm", external_kv_cache_invalidation="salt", engine_kwargs={"vllm": {key: transfer}}
        )

    if ok:
        build()
    else:
        with pytest.raises(ValueError, match="LMCacheConnectorV1"):
            build()
