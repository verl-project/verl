# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Execute the actual manager/export method bodies without CUDA dependencies."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.utils.rollout_weight_dtype import stage_export_tensor, worker_weight_dtype_metadata
from verl.workers.config.rollout import CheckpointEngineConfig

ROOT = Path(__file__).parents[2]


def method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls)
    fn = next(f for f in node.body if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef) and f.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("tp_rank", range(4))
@pytest.mark.parametrize("replica", [0, 15])
def test_metadata_preflight_reaches_replica_owner_from_every_tp_rank(tp_rank, replica):
    calls = []
    expected = [{"rank": rank, "parameters": []} for rank in range(4)]

    async def metadata():
        return expected

    server = SimpleNamespace(get_weight_dtype_metadata=SimpleNamespace(remote=metadata))

    def get_actor(name):
        calls.append(name)
        return server

    leader_gate = Mock(side_effect=AssertionError("Read-only preflight must not use leader control gate"))
    worker = SimpleNamespace(
        _pd_role=None,
        _has_server=tp_rank == 0,
        server_handle=None,
        replica_rank=replica,
        _get_server_name_prefix=lambda: "vllm_",
        _ensure_server_handle=leader_gate,
    )
    fn = method(
        "verl/workers/rollout/vllm_rollout/vllm_rollout.py",
        "ServerAdapter",
        "get_weight_dtype_metadata",
        {"ray": SimpleNamespace(get_actor=get_actor)},
    )
    assert asyncio.run(fn(worker)) == expected
    assert calls == [f"vllm_server_{replica}_0"]
    leader_gate.assert_not_called()
    assert worker.server_handle is None and worker._has_server == (tp_rank == 0)


def test_metadata_preflight_propagates_missing_receiver_without_entering_sync():
    lookup = Mock(side_effect=ValueError("Named receiver does not exist"))
    worker = SimpleNamespace(_pd_role=None, replica_rank=3, _get_server_name_prefix=lambda: "vllm_")
    fn = method(
        "verl/workers/rollout/vllm_rollout/vllm_rollout.py",
        "ServerAdapter",
        "get_weight_dtype_metadata",
        {"ray": SimpleNamespace(get_actor=lookup)},
    )
    with pytest.raises(ValueError, match="Named receiver"):
        asyncio.run(fn(worker))
    lookup.assert_called_once_with("vllm_server_3_0")


@pytest.mark.parametrize("fault", [None, "missing", "disagree", "rpc_error"])
def test_all_rank_preflight_precedes_update_and_failure_blocks_update(fault):
    events = []
    recipients = [{"gate": "torch.bfloat16"}] * 2
    if fault == "missing":
        recipients = recipients[:1]
    elif fault == "disagree":
        recipients = [recipients[0], {"gate": "torch.float32"}]

    def preflight():
        events.append("preflight")
        if fault == "rpc_error":
            raise RuntimeError("rank 1 rejected receiver")
        return recipients

    def update(**kwargs):
        assert kwargs == {"global_steps": 7, "mode": "naive"}
        events.append("update")
        return []

    self = SimpleNamespace(
        backend="naive",
        config=CheckpointEngineConfig(export_receiver_dtype=True),
        actor_wg=SimpleNamespace(world_size=2, prepare_receiver_export_dtypes=preflight, update_weights=update),
    )
    fn = method(
        "verl/checkpoint_engine/base.py",
        "CheckpointEngineManager",
        "update_weights",
        {"ray": SimpleNamespace(get=lambda x: x)},
    )
    if fault is None:
        assert asyncio.run(fn(self, global_steps=7)) == {}
        assert events == ["preflight", "update"]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.run(fn(self, global_steps=7))
        assert events == ["preflight"]


def test_default_off_never_requests_receiver_metadata():
    update = Mock(return_value=[])
    self = SimpleNamespace(
        backend="naive",
        config=CheckpointEngineConfig(),
        actor_wg=SimpleNamespace(update_weights=update),
    )
    fn = method(
        "verl/checkpoint_engine/base.py",
        "CheckpointEngineManager",
        "update_weights",
        {"ray": SimpleNamespace(get=lambda x: x)},
    )
    assert asyncio.run(fn(self, global_steps=7)) == {}
    update.assert_called_once_with(global_steps=7, mode="naive")


@pytest.mark.parametrize("value", [1, "true", None])
def test_config_rejects_non_boolean(value):
    with pytest.raises(ValueError):
        CheckpointEngineConfig(export_receiver_dtype=value)


def test_non_naive_config_rejected():
    with pytest.raises(ValueError):
        CheckpointEngineConfig(backend="nccl", export_receiver_dtype=True)


def test_cpu_staging_preserves_fp32_source_and_buffer():
    source = torch.tensor([1.003, -0.0], dtype=torch.float32)
    original_bytes = source.view(torch.uint8).clone()
    staged = stage_export_tensor(source, "cpu", torch.bfloat16)
    assert staged.dtype == torch.bfloat16
    assert staged.data_ptr() != source.data_ptr()
    assert torch.equal(source.view(torch.uint8), original_bytes)
    assert stage_export_tensor(source, "cpu", None) is source


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_export_method_casts_before_gather(enabled):
    seen = []

    class Shard:
        def __init__(self, value):
            self.value = value

        def to(self, *args, dtype=None, **kwargs):
            return Shard(self.value.to(dtype=dtype)) if dtype is not None else self

        def full_tensor(self):
            seen.append(self.value.dtype)
            return self.value

    tensor = Shard(torch.tensor([1.003], dtype=torch.float32))
    self = SimpleNamespace(
        module=SimpleNamespace(state_dict=lambda: {"weight": tensor}, config=SimpleNamespace(model_type="qwen3_moe"))
    )
    namespace = {
        "get_checkpoint_tensor_converter": lambda module: None,
        "convert_weight_keys": lambda params, module: params,
        "parallel_state": SimpleNamespace(get_parallel_state=lambda: SimpleNamespace(ep_enabled=False)),
        "get_moe_param_handler": lambda *args: None,
        "get_device_id": lambda: "cpu",
        "DTensor": Shard,
        "torch": torch,
    }
    fn = method("verl/workers/engine/veomni/transformer_impl.py", "VeOmniEngine", "get_per_tensor_param", namespace)
    kwargs = {"receiver_export_dtypes": {"weight": torch.bfloat16}} if enabled else {}
    generator, _ = fn(self, **kwargs)
    output = list(generator)
    assert seen == [torch.bfloat16 if enabled else torch.float32]
    assert output[0][0] == "weight" and tensor.value.dtype == torch.float32


@pytest.mark.parametrize("quant", [None, "fp8"])
def test_worker_metadata_reads_real_named_parameters_without_tensor_copy(quant):
    model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    model.register_buffer("scale", torch.ones([], dtype=torch.float32))
    config = SimpleNamespace(
        quant_config=quant,
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=SimpleNamespace(model_type="qwen3_moe")),
    )
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(model=model, vllm_config=config))
    if quant:
        with pytest.raises(ValueError, match="Quantized"):
            worker_weight_dtype_metadata(worker)
    else:
        result = worker_weight_dtype_metadata(worker)
        assert result["parameters"][0]["dtype"] == "torch.bfloat16"
        assert result["buffers"] == [{"name": "scale", "dtype": "torch.float32", "shape": []}]
