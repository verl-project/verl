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

import importlib
import itertools
import sys
from types import ModuleType, SimpleNamespace

import pytest


class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive
        self.started = False
        self.terminated = False
        self.join_calls = []

    def start(self):
        self.started = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class FakeSession:
    def __init__(self, responses=None):
        self.responses = iter(responses or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, *args, **kwargs):
        return next(self.responses)


def _load_http_server_engine(monkeypatch):
    modules = {
        "sglang": ModuleType("sglang"),
        "sglang.srt": ModuleType("sglang.srt"),
        "sglang.srt.entrypoints": ModuleType("sglang.srt.entrypoints"),
        "sglang.srt.entrypoints.EngineBase": ModuleType("sglang.srt.entrypoints.EngineBase"),
        "sglang.srt.entrypoints.http_server": ModuleType("sglang.srt.entrypoints.http_server"),
        "sglang.srt.managers": ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.io_struct": ModuleType("sglang.srt.managers.io_struct"),
        "sglang.srt.server_args": ModuleType("sglang.srt.server_args"),
        "sglang.srt.utils": ModuleType("sglang.srt.utils"),
    }
    for name in ("sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.managers"):
        modules[name].__path__ = []

    modules["sglang.srt.entrypoints.EngineBase"].EngineBase = type("EngineBase", (), {})
    modules["sglang.srt.entrypoints.http_server"].launch_server = lambda _: None
    modules["sglang.srt.managers.io_struct"].UpdateWeightsFromTensorReqInput = type(
        "UpdateWeightsFromTensorReqInput", (), {}
    )
    modules["sglang.srt.server_args"].ServerArgs = type("ServerArgs", (), {})
    modules["sglang.srt.utils"].kill_process_tree = lambda _: None

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "verl.workers.rollout.sglang_rollout.http_server_engine"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _server_args():
    return SimpleNamespace(
        node_rank=0,
        api_key=None,
        is_embedding=False,
        url=lambda: "http://127.0.0.1:8000",
    )


def test_startup_timeout_terminates_and_joins_process(monkeypatch):
    module = _load_http_server_engine(monkeypatch)
    process = FakeProcess(alive=True)

    monkeypatch.setattr(module.multiprocessing, "Process", lambda **_: process)
    monkeypatch.setattr(module.requests, "Session", FakeSession)
    times = itertools.chain([0.0], itertools.repeat(2.0))
    monkeypatch.setattr(module.time, "time", lambda: next(times))

    with pytest.raises(TimeoutError, match="failed to become healthy"):
        module.launch_server_process(_server_args(), max_wait_time=1.0, first_rank_in_node=True)

    assert process.started
    assert process.terminated
    assert process.join_calls == [None]


def test_early_process_exit_is_joined_before_error(monkeypatch):
    module = _load_http_server_engine(monkeypatch)
    process = FakeProcess(alive=False)

    monkeypatch.setattr(module.multiprocessing, "Process", lambda **_: process)
    monkeypatch.setattr(module.requests, "Session", FakeSession)
    times = iter([0.0, 0.0])
    monkeypatch.setattr(module.time, "time", lambda: next(times))

    with pytest.raises(RuntimeError, match="terminated unexpectedly during startup"):
        module.launch_server_process(_server_args(), max_wait_time=1.0, first_rank_in_node=True)

    assert process.started
    assert not process.terminated
    assert process.join_calls == [None]


def test_cache_flush_timeout_terminates_and_joins_process(monkeypatch):
    module = _load_http_server_engine(monkeypatch)
    process = FakeProcess(alive=True)
    healthy = SimpleNamespace(status_code=200)

    monkeypatch.setattr(module.multiprocessing, "Process", lambda **_: process)
    monkeypatch.setattr(module.requests, "Session", lambda: FakeSession([healthy]))
    times = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(module.time, "time", lambda: next(times))

    with pytest.raises(TimeoutError, match="cache flush failed"):
        module.launch_server_process(_server_args(), max_wait_time=1.0, first_rank_in_node=True)

    assert process.terminated
    assert process.join_calls == [None]


def test_early_process_exit_during_cache_flush_is_joined(monkeypatch):
    module = _load_http_server_engine(monkeypatch)
    process = FakeProcess(alive=True)
    healthy = SimpleNamespace(status_code=200)
    alive_states = iter([True, False])
    process.is_alive = lambda: next(alive_states)

    monkeypatch.setattr(module.multiprocessing, "Process", lambda **_: process)
    monkeypatch.setattr(module.requests, "Session", lambda: FakeSession([healthy]))
    times = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(module.time, "time", lambda: next(times))

    with pytest.raises(RuntimeError, match="terminated unexpectedly during cache flush"):
        module.launch_server_process(_server_args(), max_wait_time=1.0, first_rank_in_node=True)

    assert not process.terminated
    assert process.join_calls == [None]
