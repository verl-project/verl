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
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from verl.workers.rollout.vllm_rollout import vllm_async_server
from verl.workers.rollout.vllm_rollout.race_free_multiproc_executor import RaceFreeMultiprocExecutor


def test_executor_uses_unique_file_rendezvous_and_restores_vllm(monkeypatch):
    original_get_distributed_init_method = multiproc_executor.get_distributed_init_method
    init_methods = []

    def fake_init_executor(self):
        init_methods.append(multiproc_executor.get_distributed_init_method("127.0.0.1", 12345))

    monkeypatch.setattr(MultiprocExecutor, "_init_executor", fake_init_executor)

    for _ in range(2):
        executor = object.__new__(RaceFreeMultiprocExecutor)
        executor._init_executor()

    assert all(init_method.startswith("file://") for init_method in init_methods)
    assert len(init_methods) == len(set(init_methods))
    assert multiproc_executor.get_distributed_init_method is original_get_distributed_init_method


def test_executor_cleans_up_rendezvous_file(monkeypatch, tmp_path):
    rendezvous_file = tmp_path / "vllm-rendezvous"
    rendezvous_file.touch()

    monkeypatch.setattr(MultiprocExecutor, "shutdown", lambda self: None)

    executor = object.__new__(RaceFreeMultiprocExecutor)
    executor._verl_rendezvous_path = str(rendezvous_file)
    executor.shutdown()

    assert not rendezvous_file.exists()
    assert executor._verl_rendezvous_path is None


def test_executor_class_is_spawn_serializable():
    assert pickle.loads(pickle.dumps(RaceFreeMultiprocExecutor)) is RaceFreeMultiprocExecutor


def test_single_node_mp_server_selects_race_free_executor(monkeypatch):
    class StopServerStartup(Exception):
        pass

    parallel_config = SimpleNamespace(data_parallel_master_port=None, distributed_executor_backend="mp")
    vllm_config = SimpleNamespace(parallel_config=parallel_config)
    engine_args = SimpleNamespace(create_engine_config=lambda usage_context: vllm_config)

    monkeypatch.setattr(
        vllm_async_server.AsyncEngineArgs,
        "from_cli_args",
        staticmethod(lambda args: engine_args),
    )

    def capture_vllm_config(vllm_config, usage_context):
        assert vllm_config.parallel_config.distributed_executor_backend is RaceFreeMultiprocExecutor
        raise StopServerStartup

    monkeypatch.setattr(
        vllm_async_server.AsyncLLM,
        "from_vllm_config",
        staticmethod(capture_vllm_config),
    )

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.nnodes = 1
    server._dp_master_port = 12345

    with pytest.raises(StopServerStartup):
        asyncio.run(server.run_server(SimpleNamespace()))
