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
"""CPU unit tests for vLLM native WeightTransferEngine integration."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineRegistry
from verl.checkpoint_engine.vllm_weight_transfer import (
    VLLM_NATIVE_BACKENDS,
    VerlEngineWeightSource,
    VerlRayVLLMWeightSyncClient,
    VLLMNCCLCheckpointEngine,
    VLLMSparseNCCLCheckpointEngine,
    resolve_vllm_weight_transfer_backend,
)
from verl.workers.config import CheckpointEngineConfig


class _FakeRemoteMethod:
    def __init__(self, log: list, server_id: str, name: str) -> None:
        self._log = log
        self._server_id = server_id
        self._name = name

    def remote(self, *args, **kwargs):
        self._log.append((self._server_id, self._name, args, kwargs))
        return _DoneAwaitable((self._server_id, self._name))


class _DoneAwaitable:
    def __init__(self, value=None) -> None:
        self._value = value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()


class _FakeServerHandle:
    def __init__(self, server_id: str, log: list) -> None:
        self._server_id = server_id
        self._log = log

    def __getattr__(self, name: str):
        return _FakeRemoteMethod(self._log, self._server_id, name)


@pytest.mark.parametrize("ce_backend,expected", list(VLLM_NATIVE_BACKENDS.items()))
def test_resolve_vllm_weight_transfer_backend_maps_all_native_backends(ce_backend, expected):
    assert resolve_vllm_weight_transfer_backend(ce_backend, {}) == expected
    assert (
        resolve_vllm_weight_transfer_backend(ce_backend, {"weight_transfer_config": {"backend": expected}}) == expected
    )
    assert CheckpointEngineRegistry.get(ce_backend).vllm_backend == expected


def test_resolve_vllm_weight_transfer_backend_passthrough_and_errors():
    assert resolve_vllm_weight_transfer_backend("naive", {}) is None
    assert (
        resolve_vllm_weight_transfer_backend("naive", {"weight_transfer_config": {"backend": "modelexpress"}})
        == "modelexpress"
    )
    with pytest.raises(ValueError, match="vllm_sparse_nccl"):
        resolve_vllm_weight_transfer_backend("delta_sharded", {})
    with pytest.raises(ValueError, match="requires vLLM weight_transfer_config"):
        resolve_vllm_weight_transfer_backend("vllm_nccl", {"weight_transfer_config": {"backend": "ipc"}})
    with pytest.raises(TypeError, match="must be a mapping"):
        resolve_vllm_weight_transfer_backend("vllm_nccl", {"weight_transfer_config": "nccl"})


def test_ray_weight_sync_client_multi_replica_offsets(monkeypatch):
    monkeypatch.setattr("verl.checkpoint_engine.vllm_weight_transfer.ray.get", lambda refs: list(refs))
    log: list = []
    s0 = _FakeServerHandle("r0", log)
    s1 = _FakeServerHandle("r1", log)
    s2 = _FakeServerHandle("r2", log)

    # Asymmetric PD-style world sizes for NCCL / sparse_nccl / nccl_m2n.
    client = VerlRayVLLMWeightSyncClient([s0, s1, s2], [2, 4, 4])
    assert client.total_workers == 10
    client.init_weight_transfer_engine({"master_address": "127.0.0.1", "master_port": 1234, "rank_offset": 1})
    init_calls = [(sid, args[0]["rank_offset"]) for sid, name, args, _ in log if name == "init_weight_transfer_engine"]
    assert init_calls == [("r0", 1), ("r1", 3), ("r2", 7)]

    # Uniform replicas for sharded_rdt inject replica_rank / num_replicas.
    log.clear()
    rdt_client = VerlRayVLLMWeightSyncClient([s0, s1], [4, 4])
    rdt_client.init_weight_transfer_engine({"num_consumers": 8})
    rdt_calls = [
        (sid, args[0]["replica_rank"], args[0]["num_replicas"])
        for sid, name, args, _ in log
        if name == "init_weight_transfer_engine"
    ]
    assert rdt_calls == [("r0", 0, 2), ("r1", 1, 2)]

    # Non-uniform replicas rejected for sharded_rdt.
    with pytest.raises(ValueError, match="uniform world sizes"):
        client.init_weight_transfer_engine({"num_consumers": 10})

    # IPC broadcasts unchanged init_info + full four-phase calls.
    log.clear()
    rdt_client.init_weight_transfer_engine({"packed": True})
    rdt_client.start_weight_update()
    rdt_client.update_weights({"names": ["w"]})
    rdt_client.finish_weight_update("step-1")
    rdt_client.shutdown_weight_transfer_engine()
    assert [name for _, name, _, _ in log] == [
        "init_weight_transfer_engine",
        "init_weight_transfer_engine",
        "start_weight_update",
        "start_weight_update",
        "update_weights",
        "update_weights",
        "finish_weight_update",
        "finish_weight_update",
        "shutdown_weight_transfer_engine",
        "shutdown_weight_transfer_engine",
    ]


def _install_fake_vllm_weight_transfer(monkeypatch):
    @dataclass(frozen=True)
    class _ParamMeta:
        name: str
        dtype: torch.dtype
        shape: tuple[int, ...]
        owners: tuple[int, ...] | None = None
        mesh_shape: tuple[int, ...] | None = None
        placements: tuple | None = None

    @dataclass
    class _SparseWeightPatch:
        name: str
        indices: torch.Tensor
        values: torch.Tensor
        full_shape: tuple[int, ...]

    @dataclass
    class _NCCLTrainerInitInfo:
        master_address: str
        master_port: int
        world_size: int
        rank: int
        packed: bool = True
        packed_buffer_size_bytes: int = 0
        packed_num_buffers: int = 2
        use_load_plan: bool = False

    @dataclass
    class _SparseNCCLTrainerInitInfo:
        master_address: str
        master_port: int
        world_size: int
        rank: int

    events: list = []

    class _FakeTrainerEngine:
        def __init__(self, init_info, client, source):
            self.init_info = init_info
            self.client = client
            self.source = source
            self.chunks: list = []
            events.append(("trainer_init", init_info))

        def send_weights(self):
            events.append(("send_weights", self.init_info))

        def send_weight_chunk(self, patches):
            self.chunks.append(list(patches))
            events.append(("send_weight_chunk", len(patches)))

        def shutdown(self):
            events.append(("shutdown", self.init_info))

    class _FakeFactory:
        @classmethod
        def trainer_init(cls, init_info, *, client, source=None):
            return _FakeTrainerEngine(init_info, client, source)

    pkg_vllm = types.ModuleType("vllm")
    pkg_dist = types.ModuleType("vllm.distributed")
    pkg_wt = types.ModuleType("vllm.distributed.weight_transfer")
    pkg_wt.WeightTransferTrainerFactory = _FakeFactory
    mod_base = types.ModuleType("vllm.distributed.weight_transfer.base")
    mod_base.ParamMeta = _ParamMeta
    mod_nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    mod_nccl.NCCLTrainerInitInfo = _NCCLTrainerInitInfo
    mod_sparse = types.ModuleType("vllm.distributed.weight_transfer.sparse_nccl_engine")
    mod_sparse.SparseNCCLTrainerInitInfo = _SparseNCCLTrainerInitInfo
    mod_sparse.SparseWeightPatch = _SparseWeightPatch

    for name, mod in [
        ("vllm", pkg_vllm),
        ("vllm.distributed", pkg_dist),
        ("vllm.distributed.weight_transfer", pkg_wt),
        ("vllm.distributed.weight_transfer.base", mod_base),
        ("vllm.distributed.weight_transfer.nccl_engine", mod_nccl),
        ("vllm.distributed.weight_transfer.sparse_nccl_engine", mod_sparse),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    return events


def test_engine_weight_source_metadata_and_lora_guard(monkeypatch):
    _install_fake_vllm_weight_transfer(monkeypatch)
    calls = {"full": 0}

    class _FakeEngine:
        def get_per_tensor_param(self, **kwargs):
            calls["full"] += 1
            return iter([("w0", torch.ones(2, 3, dtype=torch.bfloat16))]), None

    source = VerlEngineWeightSource(_FakeEngine())
    m1 = source.metadata
    m2 = source.metadata
    assert m1 is m2
    assert calls["full"] == 1
    assert [(m.name, m.dtype, m.shape) for m in m1] == [("w0", torch.bfloat16, (2, 3))]

    class _LoraEngine:
        def get_per_tensor_param(self, **kwargs):
            return iter([]), {"r": 8}

    with pytest.raises(NotImplementedError, match="unmerged LoRA"):
        _ = VerlEngineWeightSource(_LoraEngine()).metadata


def test_vllm_nccl_checkpoint_engine_reuse_and_reinit(monkeypatch):
    events = _install_fake_vllm_weight_transfer(monkeypatch)
    monkeypatch.setattr("verl.checkpoint_engine.vllm_weight_transfer.ray.get", lambda refs: list(refs))
    monkeypatch.setattr(
        "verl.checkpoint_engine.vllm_weight_transfer._broadcast_rendezvous_endpoint",
        lambda rank: ("127.0.0.1", 29500),
    )

    class _FakeActorEngine:
        def get_per_tensor_param(self, **kwargs):
            return iter([("w", torch.zeros(2, 2, dtype=torch.bfloat16))]), None

    log: list = []
    s0 = _FakeServerHandle("r0", log)
    s1 = _FakeServerHandle("r1", log)
    ce = VLLMNCCLCheckpointEngine(bucket_size=64 << 20, is_master=True, packed_num_buffers=3, use_load_plan=True)

    # First two syncs against the same target reuse trainer_init.
    ce.configure_targets([s0], [4])
    asyncio.run(ce.send_weights(_FakeActorEngine(), global_steps=1))
    ce.configure_targets([s0], [4])
    asyncio.run(ce.send_weights(_FakeActorEngine(), global_steps=2))
    assert [e[0] for e in events] == ["trainer_init", "send_weights", "send_weights"]
    init_info = events[0][1]
    assert init_info.world_size == 5
    assert init_info.packed_num_buffers == 3
    assert init_info.use_load_plan is True

    # Adding a replica triggers shutdown + fresh trainer_init.
    ce.configure_targets([s0, s1], [4, 4])
    asyncio.run(ce.send_weights(_FakeActorEngine(), global_steps=3))
    assert [e[0] for e in events] == [
        "trainer_init",
        "send_weights",
        "send_weights",
        "shutdown",
        "trainer_init",
        "send_weights",
    ]
    assert events[-2][1].world_size == 9
    assert ("r0", "shutdown_weight_transfer_engine", (), {}) in log


def test_vllm_sparse_nccl_seed_steady_and_empty_delta(monkeypatch):
    events = _install_fake_vllm_weight_transfer(monkeypatch)
    monkeypatch.setattr("verl.checkpoint_engine.vllm_weight_transfer.ray.get", lambda refs: list(refs))
    monkeypatch.setattr(
        "verl.checkpoint_engine.vllm_weight_transfer._broadcast_rendezvous_endpoint",
        lambda rank: ("127.0.0.1", 29501),
    )

    class _FakeDeltaActorEngine:
        def __init__(self):
            self.primed = 0
            self.delta_batches: list = []

        def get_per_tensor_param(self, **kwargs):
            return iter(
                [
                    ("w0", torch.arange(4, dtype=torch.float32).reshape(2, 2)),
                    ("int_buf", torch.arange(2, dtype=torch.int64)),
                ]
            ), None

        def prime_delta_snapshots(self):
            self.primed += 1

        def get_per_tensor_param_delta_shard(self):
            return iter(self.delta_batches), None

    log: list = []
    s0 = _FakeServerHandle("r0", log)
    actor_engine = _FakeDeltaActorEngine()
    ce = VLLMSparseNCCLCheckpointEngine(bucket_size=64 << 20, is_master=True)
    ce.configure_targets([s0], [2])

    # Step 0: dense seed via SparseWeightPatch + prime_delta_snapshots.
    m0 = asyncio.run(ce.send_weights(actor_engine, global_steps=0))
    assert actor_engine.primed == 1
    assert m0["checkpoint_engine/changed_ratio"] == 1.0
    assert m0["checkpoint_engine/changed_elems"] == 4.0
    assert [name for _, name, _, _ in log] == ["start_weight_update", "finish_weight_update"]
    assert len(ce._trainer_engine.chunks) == 1
    assert ce._trainer_engine.chunks[0][0].name == "w0"
    assert torch.equal(ce._trainer_engine.chunks[0][0].indices, torch.tensor([0, 1, 2, 3], dtype=torch.int32))

    # Step 1: steady sparse delta with 2 changed entries.
    log.clear()
    actor_engine.delta_batches = [
        (
            [("w0", (2, 2))],
            "bfloat16",
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([1, 3], dtype=torch.int32),
            torch.tensor([9.0, 10.0], dtype=torch.bfloat16),
            None,
        )
    ]
    m1 = asyncio.run(ce.send_weights(actor_engine, global_steps=1))
    assert m1["checkpoint_engine/changed_ratio"] == pytest.approx(0.5)
    assert m1["checkpoint_engine/changed_elems"] == 2.0
    assert [name for _, name, _, _ in log] == ["start_weight_update", "finish_weight_update"]
    assert len(ce._trainer_engine.chunks) == 2
    assert torch.equal(ce._trainer_engine.chunks[1][0].indices, torch.tensor([1, 3], dtype=torch.int32))

    # Step 2: zero changed entries skips start/finish entirely.
    log.clear()
    actor_engine.delta_batches = [
        (
            [("w0", (2, 2))],
            "bfloat16",
            torch.tensor([0], dtype=torch.int64),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.bfloat16),
            None,
        )
    ]
    m2 = asyncio.run(ce.send_weights(actor_engine, global_steps=2))
    assert m2["checkpoint_engine/changed_elems"] == 0.0
    assert log == []
    assert any(e[0] == "trainer_init" for e in events)


def test_checkpoint_engine_manager_vllm_native_colocated_and_disaggregated():
    class _FakeActorWG:
        def __init__(self, trace: list, world_size: int = 2) -> None:
            self.world_size = world_size
            self._trace = trace

        def execute_checkpoint_engine(self, method, **kwargs):
            self._trace.append(("actor.exec", method[0], kwargs))
            return [_DoneAwaitable(None) for _ in range(self.world_size)]

        def update_weights(self, global_steps=None, mode="auto"):
            self._trace.append(("actor.update_weights", mode, global_steps))
            return [_DoneAwaitable({"checkpoint_engine/sync_time_s": 0.1}) for _ in range(self.world_size)]

    class _FakeReplica:
        def __init__(self, trace: list, server_log: list, name: str, world_size: int, pd: bool = False) -> None:
            self._trace = trace
            self.world_size = world_size
            self.config = SimpleNamespace(free_cache_engine=True, external_kv_cache_invalidation="reset")
            self.servers = [_FakeServerHandle(f"{name}_s0", server_log)]
            self._pd = pd
            if pd:
                self.servers.append(_FakeServerHandle(f"{name}_d0", server_log))

        def get_weight_transfer_targets(self):
            if self._pd:
                return [(self.servers[0], 2), (self.servers[1], 4)]
            return [(self.servers[0], self.world_size)]

        async def abort_all_requests(self, reject_request: bool = False):
            self._trace.append("replica.abort")

        async def release_kv_cache(self):
            self._trace.append("replica.release_kv")

        async def resume_kv_cache(self):
            self._trace.append("replica.resume_kv")

        async def resume_generation(self):
            self._trace.append("replica.resume_gen")

    # 1. Colocated vllm_ipc lifecycle.
    trace, server_log = [], []
    mgr_ipc = CheckpointEngineManager(
        config=CheckpointEngineConfig(backend="vllm_ipc"),
        actor_wg=_FakeActorWG(trace),
        replicas=[_FakeReplica(trace, server_log, "r0", 2)],
    )
    try:
        metrics = mgr_ipc.update_weights(global_steps=7)
        assert metrics == {"checkpoint_engine/sync_time_s": 0.1}
        assert [t[0] for t in trace] == ["actor.exec", "actor.update_weights", "actor.exec"]
        assert [(name, kwargs) for _, name, _, kwargs in server_log] == [
            ("wake_up", {"tags": ["weights"]}),
            ("clear_kv_cache", {}),
            ("set_global_steps", {}),
            ("wake_up", {"tags": ["kv_cache"]}),
        ]
    finally:
        mgr_ipc.close()

    # 2. Disaggregated vllm_nccl lifecycle (with PD multi-target replica).
    trace, server_log = [], []
    mgr_nccl = CheckpointEngineManager(
        config=CheckpointEngineConfig(backend="vllm_nccl"),
        actor_wg=_FakeActorWG(trace),
        replicas=[_FakeReplica(trace, server_log, "r0", 6, pd=True)],
    )
    try:
        metrics = mgr_nccl.update_weights(global_steps=9)
        assert metrics == {"checkpoint_engine/sync_time_s": 0.1}
        assert trace[0:2] == ["replica.abort", "replica.release_kv"]
        assert trace[2][0:2] == ("actor.exec", "configure_targets")
        assert trace[2][2]["replica_world_sizes"][0] == [2, 4]
        assert trace[3] == ("actor.update_weights", "vllm_nccl", 9)
        assert trace[4][0:2] == ("actor.exec", "finalize")
        assert trace[5:7] == ["replica.resume_kv", "replica.resume_gen"]
        assert [name for _, name, _, _ in server_log] == [
            "clear_kv_cache",
            "clear_kv_cache",
            "set_global_steps",
            "set_global_steps",
        ]
    finally:
        mgr_nccl.close()


def _load_vllm_async_server_with_stubs(monkeypatch):
    """Import vllm_async_server in a CPU-only test environment without vllm installed."""
    import importlib.util
    from pathlib import Path

    stub_names = [
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.cli",
        "vllm.entrypoints.cli.serve",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.openai.parser",
        "vllm.entrypoints.openai.parser.harmony_utils",
        "vllm.engine",
        "vllm.engine.arg_utils",
        "vllm.inputs",
        "vllm.lora",
        "vllm.lora.request",
        "vllm.outputs",
        "vllm.usage",
        "vllm.usage.usage_lib",
        "vllm.utils",
        "vllm.utils.argparse_utils",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.async_llm",
        "vllm.v1.engine.exceptions",
        "vllm.v1.metrics",
        "vllm.v1.metrics.loggers",
        "verl.workers.rollout.vllm_rollout",
        "verl.workers.rollout.vllm_rollout.utils",
    ]
    for name in stub_names:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm"].__version__ = "0.34.0"
    sys.modules["vllm"].SamplingParams = object
    sys.modules["vllm.engine.arg_utils"].AsyncEngineArgs = object
    sys.modules["vllm.entrypoints.cli.serve"].run_headless = lambda *a, **k: None
    sys.modules["vllm.entrypoints.openai.api_server"].build_app = lambda *a, **k: None
    sys.modules["vllm.entrypoints.openai.api_server"].init_app_state = lambda *a, **k: None
    sys.modules["vllm.entrypoints.openai.parser.harmony_utils"].get_encoding = lambda *a, **k: None
    sys.modules["vllm.inputs"].TokensPrompt = object
    sys.modules["vllm.lora.request"].LoRARequest = object
    sys.modules["vllm.outputs"].STREAM_FINISHED = None
    sys.modules["vllm.outputs"].RequestOutput = object
    sys.modules["vllm.usage.usage_lib"].UsageContext = object
    sys.modules["vllm.utils.argparse_utils"].FlexibleArgumentParser = object
    sys.modules["vllm.v1.engine.async_llm"].AsyncLLM = object
    sys.modules["vllm.v1.engine.exceptions"].EngineDeadError = RuntimeError
    sys.modules["vllm.v1.metrics.loggers"].PrometheusStatLogger = object
    fake_utils = sys.modules["verl.workers.rollout.vllm_rollout.utils"]
    for attr, val in [
        ("VLLM_LORA_INT_ID", 123),
        ("VLLM_LORA_NAME", "123"),
        ("VLLM_LORA_PATH", "p"),
        ("SuppressSignalInThread", object),
        ("build_cli_args_from_config", lambda cfg: []),
        ("build_mtp_speculative_config", lambda *a, **k: {}),
        ("extract_prompt_logprobs", lambda *a, **k: None),
        ("get_vllm_max_lora_rank", lambda r: r),
    ]:
        setattr(fake_utils, attr, val)

    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_async_server.py"
    spec = importlib.util.spec_from_file_location("_test_vllm_async_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vllm_server_preprocess_engine_kwargs_and_sleep_level(monkeypatch):
    mod = _load_vllm_async_server_with_stubs(monkeypatch)
    monkeypatch.setattr(mod, "is_torch_npu_available", lambda check_device=False: False)

    def _make_server(ce_backend: str, engine_kwargs_vllm: dict | None = None):
        server = object.__new__(mod.vLLMHttpServer)
        cfg_dict = {"engine_kwargs": {"vllm": engine_kwargs_vllm or {}}}
        server.config = SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend=ce_backend),
            mtp=None,
            get=lambda k, default=None: cfg_dict.get(k, default),
        )
        server.model_config = SimpleNamespace(lora_rank=0, lora={})
        return server

    # vllm_nccl wires weight_transfer_config + VLLM_RELOAD_MODE=direct.
    monkeypatch.delenv("VLLM_RELOAD_MODE", raising=False)
    ek = {}
    _make_server("vllm_nccl")._preprocess_engine_kwargs(ek)
    assert ek["weight_transfer_config"] == {"backend": "nccl"}
    assert os.environ.get("VLLM_RELOAD_MODE") == "direct"

    # vllm_sharded_rdt sets distributed_executor_backend=ray and leaves VLLM_RELOAD_MODE unset.
    monkeypatch.delenv("VLLM_RELOAD_MODE", raising=False)
    ek = {}
    _make_server("vllm_sharded_rdt")._preprocess_engine_kwargs(ek)
    assert ek["weight_transfer_config"] == {"backend": "sharded_rdt"}
    assert ek["distributed_executor_backend"] == "ray"
    assert "VLLM_RELOAD_MODE" not in os.environ

    with pytest.raises(ValueError, match="distributed_executor_backend='ray'"):
        _make_server("vllm_sharded_rdt")._preprocess_engine_kwargs({"distributed_executor_backend": "mp"})

    # vllm_sparse_nccl uses sleep level 1 so live weights survive KV cache release.
    assert _make_server("vllm_sparse_nccl")._resolve_sleep_level() == 1
    assert _make_server("vllm_nccl")._resolve_sleep_level() == 2

    # delta_sharded on vLLM fails fast with a pointer to vllm_sparse_nccl.
    with pytest.raises(ValueError, match="vllm_sparse_nccl"):
        _make_server("delta_sharded")._preprocess_engine_kwargs({})
