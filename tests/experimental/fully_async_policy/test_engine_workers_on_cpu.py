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

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.utils import fsdp_utils


def _run_fsdp1_sharded_snapshot_roundtrip(rank, world_size, rendezvous_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7)
        model = FSDP(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3)), device_id=torch.device("cpu"))
        snapshot = fsdp_utils.fsdp1_sharded_save_to_cpu(model)
        local_snapshot = {name: value.local_tensor().clone() for name, value in snapshot.items()}

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)

        fsdp_utils.fsdp1_sharded_load_from_cpu(model, snapshot)
        restored = fsdp_utils.fsdp1_sharded_save_to_cpu(model)
        for name, value in restored.items():
            torch.testing.assert_close(value.local_tensor(), local_snapshot[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_fsdp1_sharded_snapshot_roundtrip(tmp_path):
    mp.spawn(
        _run_fsdp1_sharded_snapshot_roundtrip,
        args=(2, tmp_path / "rendezvous"),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    ("strategy", "handler_names"),
    [
        ("fsdp", ("fsdp1_sharded_save_to_cpu", "fsdp1_sharded_load_from_cpu")),
        ("fsdp2", ("fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu")),
        ("veomni", ("fsdp2_sharded_save_to_cpu", "fsdp2_sharded_load_from_cpu")),
    ],
)
def test_detach_actor_worker_selects_matching_fsdp_handlers(monkeypatch, strategy, handler_names):
    expected_handlers = (object(), object())
    for name, handler in zip(handler_names, expected_handlers, strict=True):
        monkeypatch.setattr(fsdp_utils, name, handler)

    worker = object.__new__(DetachActorWorker)
    worker._strategy_handlers = None
    worker.config = SimpleNamespace(actor=SimpleNamespace(strategy=strategy))

    assert worker._get_strategy_handlers() == expected_handlers


def test_fsdp1_sharded_snapshot_uses_cpu_state_dict(monkeypatch):
    snapshot = {"weight": object()}
    model = SimpleNamespace(state_dict=lambda: snapshot)
    captured = {}

    @contextmanager
    def fake_state_ctx(ctx_model, state_type, state_cfg, optim_cfg):
        captured.update(
            model=ctx_model,
            state_type=state_type,
            offload_to_cpu=state_cfg.offload_to_cpu,
            optim_cfg=optim_cfg,
        )
        yield

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 1)
    monkeypatch.setattr(fsdp_utils, "get_fsdp_state_ctx", fake_state_ctx)

    assert fsdp_utils.fsdp1_sharded_save_to_cpu(model) is snapshot
    assert captured == {
        "model": model,
        "state_type": StateDictType.SHARDED_STATE_DICT,
        "offload_to_cpu": True,
        "optim_cfg": None,
    }


def test_fsdp1_sharded_restore_uses_matching_state_dict_context(monkeypatch):
    snapshot = {"weight": object()}
    captured = {}

    class Model:
        def load_state_dict(self, state):
            captured["state"] = state

    model = Model()

    @contextmanager
    def fake_state_ctx(ctx_model, state_type, state_cfg, optim_cfg):
        captured.update(
            model=ctx_model,
            state_type=state_type,
            offload_to_cpu=state_cfg.offload_to_cpu,
            optim_cfg=optim_cfg,
        )
        yield

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 1)
    monkeypatch.setattr(fsdp_utils, "get_fsdp_state_ctx", fake_state_ctx)

    fsdp_utils.fsdp1_sharded_load_from_cpu(model, snapshot)
    assert captured == {
        "model": model,
        "state_type": StateDictType.SHARDED_STATE_DICT,
        "offload_to_cpu": True,
        "optim_cfg": None,
        "state": snapshot,
    }
