# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Focused tests for VERL's vLLM DeltaFlush adapter."""

import json
from types import SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine.delta_sync.encode import checksum
from verl.workers.rollout.vllm_rollout import delta_weight_transfer
from verl.workers.rollout.vllm_rollout.delta_weight_transfer import (
    VerlDeltaIPCWeightTransferEngine,
)


@pytest.fixture(autouse=True)
def _stub_vllm_delta_capabilities(monkeypatch):
    """Stub delta APIs so these tests do not depend on the installed vLLM version."""

    monkeypatch.setattr(
        delta_weight_transfer,
        "_checkpoint_patch_api",
        lambda: (SimpleNamespace, lambda model, patches, **kwargs: None),
    )
    monkeypatch.setattr(delta_weight_transfer, "require_vllm_delta_support", lambda: None)


def _spec_tensor(spec: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)


def _delta_payloads():
    dense_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    empty_positions = torch.empty(0, dtype=torch.uint8)
    dense_spec = {
        "encoding": "dense",
        "params": [
            {
                "name": "model.weight",
                "dtype": "bfloat16",
                "shape": [2, 2],
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": 0,
                "val_end": 4,
            }
        ],
        "checksum": checksum(empty_positions, dense_values),
    }
    dense_payload = [("__delta_spec__", _spec_tensor(dense_spec)), ("__values__", dense_values)]

    positions = torch.tensor([0, 3], dtype=torch.int32).view(torch.uint8)
    sparse_values = torch.tensor([10.0, 30.0], dtype=torch.bfloat16)
    sparse_spec = {
        "encoding": "indices",
        "params": [
            {
                "name": "model.weight",
                "dtype": "bfloat16",
                "shape": [4],
                "pos_start": 0,
                "pos_end": 8,
                "pos_width": 4,
                "val_start": 0,
                "val_end": 2,
            },
        ],
        "checksum": checksum(positions, sparse_values),
    }
    payload = [
        ("__delta_spec__", _spec_tensor(sparse_spec)),
        ("__positions__", positions),
        ("__values__", sparse_values),
    ]
    corrupt_spec = {**sparse_spec, "checksum": sparse_spec["checksum"] + 1}
    corrupt_payload = [
        ("__delta_spec__", _spec_tensor(corrupt_spec)),
        ("__positions__", positions),
        ("__values__", sparse_values),
    ]
    return dense_payload, payload, corrupt_payload


def _delta_server(hf_config=None):
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.config = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(
            backend="delta_sharded",
            engine_kwargs={"delta_sharded": {}},
        ),
        data_parallel_size=1,
        disaggregation=SimpleNamespace(enabled=False),
    )
    server.model_config = SimpleNamespace(hf_config=hf_config or SimpleNamespace(num_experts=128))
    return server


def test_configures_vllm_delta_backend_for_moe():
    # MoE architectures name their expert count differently (Qwen/Jamba, Mixtral,
    # DeepSeek, Dbrx); all must be forced onto the Triton backend.
    for hf_config in (
        SimpleNamespace(num_experts=128),
        SimpleNamespace(num_local_experts=8),
        SimpleNamespace(n_routed_experts=64),
        SimpleNamespace(moe_num_experts=16),
    ):
        engine_kwargs = {}
        _delta_server(hf_config)._preprocess_engine_kwargs(engine_kwargs)
        assert engine_kwargs == {
            "moe_backend": "triton",
            "weight_transfer_config": {"backend": "verl_delta_ipc"},
        }

    # config-level PP is rejected globally; the engine_kwargs pass-through must be too.
    with pytest.raises(NotImplementedError, match="pipeline_parallel_size=1"):
        _delta_server(SimpleNamespace(hidden_size=1024))._preprocess_engine_kwargs({"pipeline_parallel_size": 2})


class _RemoteMethod:
    def __init__(self, name, calls, result=None):
        self.name = name
        self.calls = calls
        self.result = result

    async def _call(self, *args):
        self.calls.append((self.name, args))
        return self.result

    def remote(self, *args):
        return self._call(*args)


def _new_transfer_engine():
    engine = VerlDeltaIPCWeightTransferEngine.__new__(VerlDeltaIPCWeightTransferEngine)
    engine.model = torch.nn.Module()
    engine.model_config = SimpleNamespace()
    engine._session_encoding = None
    engine._update_failed = False
    return engine


def test_receive_payload_uses_current_bucket_callback_contract(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    first = torch.tensor([1.0], dtype=torch.bfloat16)
    second = torch.tensor([2.0], dtype=torch.bfloat16)

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("first", first)], False)
            on_bucket_received([("second", second)], True)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    engine = _new_transfer_engine()
    engine.device = torch.device("cpu")

    payload = engine._receive_payload(use_shm=False, zmq_handle="ipc:///tmp/test-verl-delta.sock")

    first.fill_(0)
    second.fill_(0)
    assert [name for name, _ in payload] == ["first", "second"]
    assert [tensor.item() for _, tensor in payload] == [1.0, 2.0]


def test_weight_transfer_engine_applies_dense_and_sparse_then_locks_after_failure(monkeypatch):
    import vllm.model_executor.model_loader.reload as reload_module

    lifecycle = []
    loaded = []
    engine = _new_transfer_engine()
    payloads = iter(_delta_payloads())
    monkeypatch.setattr(engine, "_receive_payload", lambda **kwargs: next(payloads))

    def load_patches(model, patches, **kwargs):
        lifecycle.append("load")
        loaded.append((patches, kwargs))

    monkeypatch.setattr(
        delta_weight_transfer,
        "_checkpoint_patch_api",
        lambda: (SimpleNamespace, load_patches),
    )
    monkeypatch.setattr(
        reload_module,
        "initialize_layerwise_reload",
        lambda model: lifecycle.append("initialize"),
    )
    monkeypatch.setattr(
        reload_module,
        "finalize_layerwise_reload",
        lambda model, model_config: lifecycle.append("finalize"),
    )

    update_info = SimpleNamespace(use_shm=False, zmq_handle="ipc:///tmp/test-verl-delta.sock")

    # The initial dense seed uses layerwise reload.
    engine.start_weight_update()
    engine.receive_weights(update_info)
    engine.finish_weight_update()

    # A steady sparse update applies its first patch without reinitializing the model.
    engine.start_weight_update()
    engine.receive_weights(update_info)

    assert lifecycle == ["initialize", "load", "finalize", "load"]
    dense_patch = loaded[0][0][0]
    assert (dense_patch.name, dense_patch.shape, dense_patch.indices) == ("model.weight", (2, 2), None)
    assert dense_patch.values.tolist() == [1.0, 2.0, 3.0, 4.0]
    sparse_patch = loaded[1][0][0]
    assert (sparse_patch.name, sparse_patch.shape) == ("model.weight", (4,))
    assert sparse_patch.indices.tolist() == [0, 3]
    assert sparse_patch.values.tolist() == [10.0, 30.0]
    assert [kwargs for _, kwargs in loaded] == [
        {"validate_unique_indices": False},
        {"validate_unique_indices": False},
    ]

    # A corrupt later flush leaves a partial update, so the worker must reject retries.
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        engine.receive_weights(update_info)
    with pytest.raises(RuntimeError, match="previous delta update"):
        engine.start_weight_update()


def _make_server_adapter():
    from verl.workers.rollout.vllm_rollout import vllm_rollout as rollout_module

    adapter = rollout_module.ServerAdapter.__new__(rollout_module.ServerAdapter)
    adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1))
    adapter.use_shm = False
    adapter.zmq_handle = "ipc:///tmp/test-verl-delta.sock"
    adapter._delta_weight_transfer_engine_initialized = False
    adapter._has_server = True
    adapter.replica_rank = 0
    adapter.rollout_rank = 0

    rpc_calls = []

    async def execute_method(method, **kwargs):
        rpc_calls.append((method, kwargs))
        return None

    adapter._execute_method = execute_method
    adapter._ensure_server_handle = lambda: True
    published = []
    adapter.server_handle = SimpleNamespace(
        clear_kv_cache=_RemoteMethod("clear_kv_cache", published),
        set_global_steps=_RemoteMethod("set_global_steps", published),
    )
    return rollout_module, adapter, rpc_calls, published


@pytest.mark.asyncio
async def test_server_adapter_streams_repeated_and_empty_updates(monkeypatch):
    rollout_module, adapter, rpc_calls, published = _make_server_adapter()
    sent = []

    class FakeSender:
        def __init__(self, **kwargs):
            pass

        async def async_send_weights(self, weights):
            sent.append(list(weights))

    monkeypatch.setattr(rollout_module, "BucketedWeightSender", FakeSender)
    payload_1 = [("__delta_spec__", torch.tensor([1], dtype=torch.uint8))]
    payload_2 = [("__values__", torch.tensor([2.0], dtype=torch.bfloat16))]

    await adapter._update_delta_weights(
        [(payload_1, False), (payload_2, True)],
        global_steps=7,
    )
    await adapter._update_delta_weights([(payload_1, True)], global_steps=8)
    await adapter._update_delta_weights([], global_steps=9)

    assert [method for method, _ in rpc_calls] == [
        "init_weight_transfer_engine",
        "start_weight_update",
        "update_verl_delta_weights",
        "update_verl_delta_weights",
        "finish_weight_update",
        "start_weight_update",
        "update_verl_delta_weights",
        "finish_weight_update",
    ]
    assert sent == [payload_1, payload_2, payload_1]
    assert published == [
        ("clear_kv_cache", ()),
        ("set_global_steps", (7,)),
        ("clear_kv_cache", ()),
        ("set_global_steps", (8,)),
        ("set_global_steps", (9,)),
    ]
