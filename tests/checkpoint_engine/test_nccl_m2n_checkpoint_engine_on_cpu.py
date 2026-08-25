# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""CPU-only tests for the NCCL M2N checkpoint backend."""

from __future__ import annotations

import asyncio

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

import verl.checkpoint_engine.nccl_m2n_checkpoint_engine as m2n_module
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_m2n_checkpoint_engine import (
    NCCLM2NCheckpointEngine,
    NCCLM2NMasterMetadata,
    _nccl_stream,
    _require_nccl,
)
from verl.models.transformers.hf_dense_decoder_tp import infer_dense_decoder_tp_shard_dim
from verl.workers.engine.spec import ShardSpec


class _Mesh:
    ndim = 2

    def __init__(self, source_shard_rank=3):
        self.source_shard_rank = source_shard_rank

    @staticmethod
    def size(mesh_dim=None):
        return 8 if mesh_dim is None else (2, 4)[mesh_dim]

    def get_local_rank(self, mesh_dim=None):
        return 7 if mesh_dim is None else (1, self.source_shard_rank)[mesh_dim]


class _Stream:
    def __init__(self):
        self.waited_for = []
        self.synchronize_calls = 0

    def wait_stream(self, stream):
        self.waited_for.append(stream)

    def synchronize(self):
        self.synchronize_calls += 1


class _Handle:
    def __init__(self):
        self.calls = []
        self.destroy_calls = 0

    def reshard(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def destroy(self):
        self.destroy_calls += 1


class _Communicator:
    def __init__(self):
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


def _engine(**overrides):
    kwargs = {
        "bucket_size": 256,
        "source_dp": 2,
        "source_shard_size": 4,
        "destination_dp": 3,
        "destination_shard_size": 2,
    }
    kwargs.update(overrides)
    return NCCLM2NCheckpointEngine(**kwargs)


def _metadata(**overrides):
    kwargs = {
        "unique_id": b"test-id",
        "zmq_ip": "127.0.0.1",
        "zmq_port": 12345,
        "source_dp": 2,
        "source_shard_size": 4,
        "destination_dp": 3,
        "destination_shard_size": 2,
    }
    kwargs.update(overrides)
    return NCCLM2NMasterMetadata(**kwargs)


def _exported_weight(source_shard_rank=3):
    return (
        "model.layers.0.mlp.down_proj.weight",
        torch.arange(12, dtype=torch.float32),
        ShardSpec(
            full_shape=(8, 6),
            mesh=_Mesh(source_shard_rank),
            placements=(Replicate(), Shard(0)),
        ),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.embed_tokens.weight", 0),
        ("model.layers.0.self_attn.q_proj.weight", 0),
        ("model.layers.0.self_attn.o_proj.weight", 1),
        ("model.layers.0.mlp.down_proj.weight", 1),
        ("model.layers.0.input_layernorm.weight", None),
    ],
)
def test_dense_decoder_tp_shard_dim(name, expected):
    assert infer_dense_decoder_tp_shard_dim(name) == expected


def test_dense_decoder_tp_shard_dim_rejects_unknown_weight():
    with pytest.raises(NotImplementedError, match="unvalidated dense-decoder"):
        infer_dense_decoder_tp_shard_dim("model.layers.0.unsupported.weight")


def test_backend_is_registered_without_requiring_nccl_at_import_time():
    assert CheckpointEngineRegistry.get("nccl_m2n") is NCCLM2NCheckpointEngine
    assert NCCLM2NCheckpointEngine.wire_format == "rank_local_named_tensors"


def test_constructor_requires_explicit_positive_topology():
    with pytest.raises(ValueError, match="explicit values.*source_dp"):
        NCCLM2NCheckpointEngine(bucket_size=256)
    with pytest.raises(ValueError, match="sizes must be positive"):
        _engine(source_dp=0)


@pytest.mark.parametrize(("is_lora", "qat_enabled"), [(True, False), (False, True)])
def test_fsdp_raw_export_rejects_lora_and_qat(is_lora, qat_enabled):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    engine = object.__new__(FSDPEngine)
    engine._is_lora = is_lora
    engine._qat_enabled = qat_enabled

    with pytest.raises(NotImplementedError, match="does not support LoRA or QAT"):
        engine.get_per_tensor_param_reshard()


def test_optional_nccl_dependency_has_a_clear_error(monkeypatch):
    import_error = ImportError("nccl is unavailable")
    monkeypatch.setattr(m2n_module, "_NCCL_IMPORT_ERROR", import_error)

    with pytest.raises(ImportError, match="requires NCCL4Py") as exc_info:
        _require_nccl()

    assert exc_info.value.__cause__ is import_error


def test_prepare_creates_and_caches_master_metadata(monkeypatch):
    class _UniqueId:
        def __bytes__(self):
            return b"unique-id"

    engine = _engine()
    engine.is_master = True
    engine._metadata_ip = "10.0.0.1"
    engine._metadata_port = 23456
    monkeypatch.setattr(m2n_module, "_require_nccl", lambda: None)
    monkeypatch.setattr(m2n_module, "get_unique_id", _UniqueId, raising=False)

    first = engine.prepare()
    second = engine.prepare()

    assert first is second
    assert first == NCCLM2NMasterMetadata(b"unique-id", "10.0.0.1", 23456, 2, 4, 3, 2)


def test_metadata_server_uses_xpub_subscription_notifications(monkeypatch):
    class _Socket:
        def __init__(self):
            self.options = []
            self.bound = None

        def setsockopt(self, option, value):
            self.options.append((option, value))

        def bind(self, address):
            self.bound = address

    socket = _Socket()

    class _Context:
        @staticmethod
        def socket(socket_type):
            assert socket_type == m2n_module.zmq.XPUB
            return socket

    engine = object.__new__(NCCLM2NCheckpointEngine)
    monkeypatch.setattr(m2n_module.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(m2n_module, "get_free_port", lambda _: (23456, None))
    monkeypatch.setattr(m2n_module.zmq, "Context", _Context)

    engine._start_metadata_server()

    assert (m2n_module.zmq.XPUB_VERBOSE, 1) in socket.options
    assert socket.bound == "tcp://127.0.0.1:23456"


def test_metadata_client_subscribes_to_common_topic_before_rank_readiness(monkeypatch):
    class _Socket:
        def __init__(self):
            self.subscriptions = []

        def connect(self, address):
            self.address = address

        def setsockopt_string(self, option, value):
            assert option == m2n_module.zmq.SUBSCRIBE
            self.subscriptions.append(value)

    socket = _Socket()

    class _Context:
        @staticmethod
        def socket(socket_type):
            assert socket_type == m2n_module.zmq.SUB
            return socket

    engine = _engine()
    engine.rank = 8
    monkeypatch.setattr(m2n_module.zmq, "Context", _Context)

    engine._connect_metadata_client(_metadata())

    assert socket.address == "tcp://127.0.0.1:12345"
    assert socket.subscriptions == [engine.topic, f"{engine.ready_topic_prefix}8"]
    engine._socket = engine._zmq_context = None


def test_master_waits_for_each_destination_readiness_subscription():
    engine = _engine()
    prefix = engine.ready_topic_prefix.encode()

    class _Socket:
        def __init__(self):
            self.events = iter(
                [
                    b"\x01" + engine.topic.encode(),
                    b"\x01" + prefix + b"10",
                    b"\x01" + prefix + b"8",
                    b"\x01" + prefix + b"10",
                    b"\x00" + prefix + b"10",
                    b"\x01" + prefix + b"9",
                    b"\x01" + prefix + b"11",
                    b"\x01" + prefix + b"12",
                    b"\x01" + prefix + b"13",
                    b"\x01" + prefix + b"10",
                ]
            )
            self.recv_calls = 0
            self.sent = []

        def recv(self):
            self.recv_calls += 1
            return next(self.events)

        def send_string(self, message):
            self.sent.append(message)

    engine._socket = _Socket()

    engine._wait_for_metadata_subscribers()

    assert engine._socket.recv_calls == 10
    assert engine._socket.sent == [f"{engine.ready_topic_prefix}{rank}" for rank in range(8, 14)]
    engine._socket = None


def test_master_rejects_readiness_from_a_non_destination_rank():
    engine = _engine()

    class _Socket:
        @staticmethod
        def recv():
            return b"\x01" + engine.ready_topic_prefix.encode() + b"7"

    engine._socket = _Socket()

    with pytest.raises(RuntimeError, match="unexpected.*readiness rank 7"):
        engine._wait_for_metadata_subscribers()
    engine._socket = None


def test_destination_waits_for_its_readiness_acknowledgement():
    engine = _engine()
    engine.rank = 8

    class _Socket:
        @staticmethod
        def recv_string():
            return f"{engine.ready_topic_prefix}8"

    engine._socket = _Socket()

    engine._wait_for_metadata_publisher()

    engine._socket = None


def test_destination_rejects_another_ranks_readiness_acknowledgement():
    engine = _engine()
    engine.rank = 8

    class _Socket:
        @staticmethod
        def recv_string():
            return f"{engine.ready_topic_prefix}9"

    engine._socket = _Socket()

    with pytest.raises(RuntimeError, match="unexpected.*acknowledgement"):
        engine._wait_for_metadata_publisher()
    engine._socket = None


def test_build_topology_assigns_all_trainers_and_destinations():
    master = _metadata()

    trainer, rollout = NCCLM2NCheckpointEngine.build_topology(8, 6, [master] + [None] * 13)

    assert trainer["rank"] == list(range(8))
    assert trainer["role"] == ["source"] * 8
    assert rollout["rank"] == list(range(8, 14))
    assert rollout["role"] == ["destination"] * 6
    assert trainer["world_size"] == [14] * 8
    assert rollout["world_size"] == [14] * 6
    assert trainer["master_metadata"] == [master] * 8
    assert rollout["master_metadata"] == [master] * 6


@pytest.mark.parametrize(
    ("trainer_world_size", "rollout_world_size"),
    [(7, 6), (9, 6), (8, 5), (8, 7)],
)
def test_build_topology_rejects_incompatible_world_sizes(trainer_world_size, rollout_world_size):
    with pytest.raises(ValueError, match="requires exactly 8 trainer and 6 rollout ranks"):
        NCCLM2NCheckpointEngine.build_topology(
            trainer_world_size,
            rollout_world_size,
            [_metadata()] + [None] * (trainer_world_size + rollout_world_size - 1),
        )


def test_build_topology_requires_exactly_one_master():
    with pytest.raises(ValueError, match="exactly one trainer master, got 0"):
        NCCLM2NCheckpointEngine.build_topology(8, 6, [None] * 14)
    with pytest.raises(ValueError, match="exactly one trainer master, got 2"):
        NCCLM2NCheckpointEngine.build_topology(8, 6, [_metadata(), _metadata()] + [None] * 12)


def test_process_group_rejects_invalid_rank_and_role_before_requiring_nccl():
    engine = _engine()

    with pytest.raises(ValueError, match="rank must be non-negative"):
        engine.init_process_group(-1, 14, _metadata(), "source")
    with pytest.raises(ValueError, match="role must be source or destination"):
        engine.init_process_group(0, 14, _metadata(), "inactive")


def test_active_rank_rejects_runtime_topology_mismatch_before_bootstrap(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(m2n_module, "_require_nccl", lambda: None)

    with pytest.raises(ValueError, match="runtime world_size=13, expected 14"):
        engine.init_process_group(0, 13, _metadata(), "source")
    with pytest.raises(ValueError, match="does not match local config"):
        engine.init_process_group(0, 14, _metadata(source_shard_size=2), "source")


def test_nccl_stream_uses_modern_torch_raw_handle_and_preserves_legacy_objects():
    class _ModernTorchStream:
        cuda_stream = 12345

    class _LegacyStream:
        def __cuda_stream__(self):
            return (0, 67890)

    legacy = _LegacyStream()
    assert _nccl_stream(_ModernTorchStream()) == 12345
    assert _nccl_stream(legacy) is legacy
    assert _nccl_stream(111) == 111


def test_raw_fsdp_export_is_normalized_into_rank_local_layouts():
    engine = _engine()

    weight = engine._coerce_weight(_exported_weight())
    source, destination = engine._layouts(weight)

    assert weight.name == "model.layers.0.mlp.down_proj.weight"
    assert weight.tensor.shape == (2, 6)
    assert weight.source_shard_size == 4
    assert weight.destination_shard_dim == 1
    assert (source.mesh_dims, source.placements, source.local_shape) == ((2, 4), (None, 0), (2, 6))
    assert (destination.mesh_dims, destination.placements, destination.local_shape) == (
        (3, 2),
        (None, 1),
        (8, 3),
    )


def test_source_rank_validates_device_mesh_to_m2n_ordering():
    engine = _engine()
    engine.rank = 2

    with pytest.raises(ValueError, match="DeviceMesh rank 3 does not match M2N mesh rank 2"):
        engine._coerce_weight(_exported_weight(source_shard_rank=3))


def test_layouts_are_converted_to_m2n_descriptors(monkeypatch):
    descriptors = []

    class _M2NMesh:
        def __init__(self, dims, start_rank):
            self.dims = dims
            self.start_rank = start_rank

    class _DistTensor:
        def __init__(self, tensor, **kwargs):
            self.tensor = tensor
            self.kwargs = kwargs
            descriptors.append(self)

    monkeypatch.setattr(m2n_module, "Mesh", _M2NMesh, raising=False)
    monkeypatch.setattr(m2n_module, "DistTensor", _DistTensor, raising=False)
    monkeypatch.setattr(m2n_module, "Replicate", lambda: ("replicate", None), raising=False)
    monkeypatch.setattr(m2n_module, "Shard", lambda dim: ("shard", dim), raising=False)

    engine = _engine()
    weight = engine._coerce_weight(_exported_weight())
    source_tensor = torch.empty(2, 6)
    destination_tensor = torch.empty(8, 3)

    source, destination = engine._descriptors(weight, source_tensor, destination_tensor)

    assert descriptors == [source, destination]
    assert source.tensor is source_tensor
    assert source.kwargs["mesh"].dims == (2, 4)
    assert source.kwargs["mesh"].start_rank == 0
    assert source.kwargs["placements"] == [("replicate", None), ("shard", 0)]
    assert destination.tensor is destination_tensor
    assert destination.kwargs["mesh"].dims == (3, 2)
    assert destination.kwargs["mesh"].start_rank == 8
    assert destination.kwargs["placements"] == [("replicate", None), ("shard", 1)]


def test_send_weights_passes_source_tensor_to_handle_reshard(monkeypatch):
    caller_stream = _Stream()
    transfer_stream = _Stream()
    handle = _Handle()
    published = []
    descriptor_tensors = []
    engine = _engine()
    engine.rank = 0
    engine.role = "source"
    engine._comm = _Communicator()
    engine._handle = handle
    engine._transfer_stream = transfer_stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)
    monkeypatch.setattr(engine, "_publish", published.append)

    def _descriptors(_weight, source, destination):
        descriptor_tensors.append((source, destination))
        return "source", "destination"

    monkeypatch.setattr(engine, "_descriptors", _descriptors)

    exported = _exported_weight(source_shard_rank=0)

    result = asyncio.run(engine.send_weights(iter([exported])))

    assert result == {}
    assert transfer_stream.waited_for == [caller_stream]
    assert caller_stream.waited_for == [transfer_stream]
    assert transfer_stream.synchronize_calls == 0
    assert handle.calls == [((engine._comm, "source", "destination"), {"stream": transfer_stream})]
    assert descriptor_tensors[0][0].data_ptr() == exported[1].data_ptr()
    assert descriptor_tensors[0][1] is None
    assert published[0]["kind"] == "weight"
    assert published[0]["name"] == "model.layers.0.mlp.down_proj.weight"
    assert published[0]["destination_shard_dim"] == 1
    assert "source_shard_rank" not in published[0]
    assert published[-1] == {"kind": "end"}


def test_finalize_waits_for_enqueued_transfer_and_consumer_work(monkeypatch):
    transfer_stream = _Stream()
    caller_stream = _Stream()
    engine = _engine()
    engine.rank = 0
    engine._transfer_stream = transfer_stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)

    engine.finalize()

    assert transfer_stream.synchronize_calls == 1
    assert caller_stream.synchronize_calls == 1


def test_receive_weights_rebuilds_layout_calls_reshard_and_yields_rank_local_tensor(monkeypatch):
    caller_stream = _Stream()
    transfer_stream = _Stream()
    handle = _Handle()
    descriptor_tensors = []
    messages = iter(
        [
            {
                "kind": "weight",
                "name": "model.layers.0.mlp.down_proj.weight",
                "global_shape": (8, 6),
                "dtype": torch.float32,
                "destination_shard_dim": 1,
                "source_shard_dim": 0,
                "source_shard_size": 4,
            },
            {"kind": "end"},
        ]
    )
    engine = _engine()
    engine.rank = 8
    engine.role = "destination"
    engine._comm = _Communicator()
    engine._handle = handle
    engine._transfer_stream = transfer_stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)
    monkeypatch.setattr(
        m2n_module,
        "_allocate_destination",
        lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    monkeypatch.setattr(engine, "_receive", lambda: next(messages))

    def _descriptors(_weight, source, destination):
        descriptor_tensors.append((source, destination))
        return "source", "destination"

    monkeypatch.setattr(engine, "_descriptors", _descriptors)

    async def _collect():
        return [item async for item in engine.receive_weights()]

    weights = asyncio.run(_collect())

    assert len(weights) == 1
    assert weights[0][0] == "model.layers.0.mlp.down_proj.weight"
    assert weights[0][1].shape == (8, 3)
    assert weights[0][1].dtype == torch.float32
    assert transfer_stream.waited_for == [caller_stream]
    assert caller_stream.waited_for == [transfer_stream]
    assert handle.calls == [((engine._comm, "source", "destination"), {"stream": transfer_stream})]
    assert descriptor_tensors == [(None, weights[0][1])]


def test_close_is_idempotent_and_releases_owned_resources():
    engine = _engine()
    stream = _Stream()
    cleanup_order = []

    class _OrderedHandle(_Handle):
        def destroy(self):
            cleanup_order.append("handle")
            super().destroy()

    class _OrderedCommunicator(_Communicator):
        def destroy(self):
            cleanup_order.append("communicator")
            super().destroy()

    handle = _OrderedHandle()
    communicator = _OrderedCommunicator()

    class _Socket:
        def __init__(self):
            self.linger = []

        def close(self, linger):
            self.linger.append(linger)

    class _Context:
        def __init__(self):
            self.term_calls = 0

        def term(self):
            self.term_calls += 1

    socket = _Socket()
    context = _Context()
    engine._transfer_stream = stream
    engine._handle = handle
    engine._comm = communicator
    engine._socket = socket
    engine._zmq_context = context

    engine.close()
    engine.close()

    assert stream.synchronize_calls == 1
    assert handle.destroy_calls == 1
    assert communicator.destroy_calls == 1
    assert cleanup_order == ["handle", "communicator"]
    assert socket.linger == [0]
    assert context.term_calls == 1
    assert engine._comm is engine._handle is engine._transfer_stream is None
    assert engine._socket is engine._zmq_context is None
