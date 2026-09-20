# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Scheduler-only tests of installed vLLM selector and real distributed stores."""

import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from vllm.v1.executor import multiproc_executor as mp


def config(**overrides):
    values = {"world_size": 1, "data_parallel_size": 1, "nnodes": 1, "enable_elastic_ep": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_single_rank_does_not_probe(monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_USE_AGENT_STORE", raising=False)
    monkeypatch.setattr(mp, "get_loopback_ip", lambda: "127.0.0.1")

    def forbidden():
        raise AssertionError("single rank must not probe/release a port")

    monkeypatch.setattr(mp, "get_open_port", forbidden)
    assert mp.VERL_SINGLE_RANK_ATOMIC_TCPSTORE == "20260915-v1"
    assert mp._get_mp_distributed_init_method(config()) == "tcp://127.0.0.1:0"


@pytest.mark.parametrize(
    "overrides",
    [
        {"world_size": 2},
        {"data_parallel_size": 2},
        {"nnodes": 2},
        {"enable_elastic_ep": True},
    ],
)
def test_other_topologies_keep_upstream(monkeypatch, overrides):
    monkeypatch.delenv("TORCHELASTIC_USE_AGENT_STORE", raising=False)
    monkeypatch.setattr(mp, "get_loopback_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mp, "get_open_port", lambda: 31001)
    assert mp._get_mp_distributed_init_method(config(**overrides)) == "tcp://127.0.0.1:31001"


def test_agent_store_keeps_concrete_port(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    monkeypatch.setattr(mp, "get_loopback_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(mp, "get_open_port", lambda: 31002)
    assert mp._get_mp_distributed_init_method(config()) == "tcp://127.0.0.1:31002"


def test_real_store_collision_and_atomic_listeners():
    # Deterministic old-path collision, then multiple simultaneously live port-0
    # stores. No multiprocessing pool; the store's own service threads are bounded.
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(dist.DistNetworkError, match="EADDRINUSE"):
            dist.TCPStore("127.0.0.1", port, 1, True, timeout=timedelta(seconds=10))
        stores = [dist.TCPStore("127.0.0.1", 0, 1, True) for _ in range(4)]
        assert len({store.port for store in stores}) == 4
        assert port not in {store.port for store in stores}
        for index, store in enumerate(stores):
            store.set("owner", str(index))
        for index, store in enumerate(stores):
            assert store.get("owner") == str(index).encode()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_real_process_group_lifecycle(monkeypatch, backend):
    monkeypatch.delenv("TORCHELASTIC_USE_AGENT_STORE", raising=False)
    assert not dist.is_initialized()
    if backend == "nccl":
        assert torch.cuda.is_available(), "GPU verification is mandatory; no skip"
        torch.cuda.set_device(0)
    for _ in range(3):
        try:
            dist.init_process_group(
                backend=backend,
                init_method=mp._get_mp_distributed_init_method(config()),
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=30),
            )
            value = torch.tensor([7.0], device="cuda" if backend == "nccl" else "cpu")
            dist.all_reduce(value)
            assert value.item() == 7.0
            group = dist.new_group(backend="gloo")
            dist.barrier(group=group)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
