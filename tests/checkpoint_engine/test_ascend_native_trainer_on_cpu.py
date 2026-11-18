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
import sys
import threading
from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.checkpoint_engine import ascend_native_checkpoint_engine as backend


@pytest.fixture
def native(monkeypatch):
    factory = Mock()
    meta = namedtuple("ParamMeta", ["name", "dtype", "shape"])
    bindings = {
        "vllm.distributed.weight_transfer.base": {"ParamMeta": meta, "WeightSource": object},
        "vllm.distributed.weight_transfer.factory": {"WeightTransferTrainerFactory": factory},
        "vllm.plugins": {"load_general_plugins": Mock()},
        "vllm_ascend.distributed.weight_transfer.hccl_engine": {"HCCLTrainerInitInfo": SimpleNamespace},
        "verl.workers.rollout.vllm_rollout.native_weight_sync": {"RayWeightSyncClient": tuple},
    }
    for module, attrs in bindings.items():
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**attrs))

    stream = object()
    device = SimpleNamespace(
        current_device=lambda: 0, current_stream=lambda: stream, device=nullcontext, stream=nullcontext
    )
    monkeypatch.setattr(backend, "get_torch_device", lambda: device)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda output, value: output.__setitem__(0, value))
    return factory


@pytest.mark.parametrize("rank", [0, 1])
def test_trainer_replays_current_export_and_rebuilds_session(native, monkeypatch, rank):
    actor_thread = threading.current_thread()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value: output.__setitem__(slice(None), [value, value])
    )
    model = Mock()
    model.get_weight_transfer_metadata.return_value = [("w", torch.float32, (1,))]
    values = []
    initializations = []

    def init(info, *, client, source):
        assert threading.current_thread() != actor_thread
        assert not torch.is_grad_enabled()
        assert info.rank == rank and info.packed_buffer_size_bytes == 512 and info.packed
        assert info.world_size == 2 and client == (("server", 1),)

        def send():
            assert source.metadata[0].name == "w"
            values.extend(tensor.item() for _, tensor in source.iter_full())

        session = Mock(send_weights=send)
        initializations.append(session)
        return session

    native.trainer_init.side_effect = init
    engine = backend.AscendNativeCheckpointEngine(512, is_master=rank == 0)
    for step in range(3):
        # A replacement manager may configure a new group with its own generation 1.
        if step in (0, 2):
            trainer_kwargs, relay_kwargs = engine.build_topology(2, 1, [("host", 5000 + step), None])
            assert relay_kwargs == {}
            engine.init_process_group(
                **{key: args[rank] for key, args in trainer_kwargs.items()}, endpoints=[("server", 1)], generation=1
            )
        model.get_per_tensor_param.return_value = (iter([("w", torch.tensor([float(step)]))]), None)
        asyncio.run(engine.send_weights(weights=model, global_steps=step))
    assert values == [0.0, 1.0, 2.0]
    assert len(initializations) == 2
    initializations[0].shutdown.assert_called_once_with()
    assert model.get_per_tensor_param.call_count == 3
    assert model.get_weight_transfer_metadata.call_count == 2


def test_trainer_preflight_rejects_bad_export_before_worker_rendezvous(native):
    engine = backend.AscendNativeCheckpointEngine(512)
    engine.init_process_group("host", 5123, 2, [("server", 1)], 1)
    model = Mock()
    model.get_weight_transfer_metadata.side_effect = NotImplementedError("unmerged adapter")
    with pytest.raises(RuntimeError, match="preflight failed on rank 0:.*unmerged adapter"):
        asyncio.run(engine.send_weights(model))
    native.trainer_init.assert_not_called()


def test_training_rank_schema_mismatch_does_not_start_worker_rendezvous(native, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, (None, "other")]),
    )
    engine = backend.AscendNativeCheckpointEngine(512)
    engine.init_process_group("host", 5123, 2, [("server", 1)], 1)
    model = Mock()
    model.get_weight_transfer_metadata.return_value = [("w", torch.float32, (1,))]
    with pytest.raises(ValueError, match="identical full-weight metadata"):
        asyncio.run(engine.send_weights(model))
    native.trainer_init.assert_not_called()


def test_failed_native_session_is_discarded(native):
    engine = backend.AscendNativeCheckpointEngine(512, packed=False)
    engine.init_process_group("host", 5123, 2, [("server", 1)], 1)
    model = Mock()
    model.get_weight_transfer_metadata.return_value = [("w", torch.float32, (1,))]
    broken = Mock()
    broken.send_weights.side_effect = RuntimeError("worker failed")
    good = Mock()
    native.trainer_init.side_effect = [broken, good]
    with pytest.raises(RuntimeError, match="worker failed"):
        asyncio.run(engine.send_weights(model))
    broken.shutdown.assert_called_once_with()
    engine.init_process_group("host", 5124, 2, [("replacement", 1)], 2)
    asyncio.run(engine.send_weights(model))
    good.send_weights.assert_called_once_with()
    assert native.trainer_init.call_args.args[0].packed is False
