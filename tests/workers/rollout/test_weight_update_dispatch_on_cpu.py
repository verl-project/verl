# Copyright 2026 Individual Contributor: yeyunu
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

# Copyright 2026 verl contributors
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

"""CPU-only dispatch contracts; no torch, vLLM, or accelerator imports required."""

import ast
import asyncio
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_ROLLOUT = _ROOT / "verl/workers/rollout/vllm_rollout"
_spec = importlib.util.spec_from_file_location("weight_update_dispatch", _ROLLOUT / "weight_update.py")
_dispatch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dispatch)
resolve_weight_update_mode = _dispatch.resolve_weight_update_mode


def _config(backend=None, mode=None):
    return SimpleNamespace(
        custom={"ascend_weight_update_mode": mode},
        engine_kwargs={"vllm": {"weight_transfer_config": {"backend": backend}}},
        checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=512),
    )


@pytest.mark.parametrize(
    "backend, mode, expected",
    [
        (None, None, None),
        ("nccl", None, None),
        ("ipc", None, None),
        ("custom_backend", None, None),
        ("npu_ipc", None, "npu_ipc"),
    ],
)
def test_mode_selection(backend, mode, expected):
    assert resolve_weight_update_mode(_config(backend, mode)) == expected


@pytest.mark.parametrize("config", [{}, {"engine_kwargs": {}}, {"engine_kwargs": {"vllm": None}}])
def test_missing_optional_config_preserves_legacy(config):
    assert resolve_weight_update_mode(config) is None


def _load_update_method(events, allow_native=False):
    """Execute the actual method body with only its external collaborators replaced.

    AST extraction avoids loading vLLM/Ray on CPU. It does not reimplement the
    dispatch branches being tested. CUDA IPC itself requires a separate GPU test.
    """
    tree = ast.parse((_ROLLOUT / "vllm_rollout.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ServerAdapter")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights")
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))

    class Sender:
        def __init__(self, **kwargs):
            events.append(("sender", kwargs))

        async def async_send_weights(self, weights):
            events.append(("send", list(weights)))

    async def native(adapter, weights, **kwargs):
        events.append(("native", list(weights), kwargs))

    def guarded_import(name, *args, **kwargs):
        if name.endswith("ascend_weight_transfer"):
            assert allow_native, "A legacy/CUDA path imported the Ascend adapter"
            return SimpleNamespace(update_ascend_weights=native)
        assert not name.startswith(("torch_npu", "vllm_ascend")), "Unexpected accelerator dependency"
        return builtins.__import__(name, *args, **kwargs)

    scope = {
        "__builtins__": {**vars(builtins), "__import__": guarded_import},
        "resolve_weight_update_mode": resolve_weight_update_mode,
        "BucketedWeightSender": Sender,
        "time": SimpleNamespace(time=lambda: 1),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(module, str(_ROLLOUT / "vllm_rollout.py"), "exec"), scope)
    return scope["update_weights"]


def _adapter(events, config, use_shm=False, has_server=True):
    class ReceiverFuture:
        def __await__(self):
            async def wait():
                events.append(("await_receiver",))

            return wait().__await__()

    async def execute(method, **kwargs):
        events.append(("receiver", method, kwargs))
        return ReceiverFuture() if has_server else None

    async def clear():
        events.append(("clear_cache",))

    async def steps(step):
        events.append(("steps", step))

    async def delta(weights, **kwargs):
        events.append(("delta", list(weights), kwargs))

    return SimpleNamespace(
        config=config,
        device="cuda:0",
        use_shm=use_shm,
        zmq_handle="ipc://dispatch-test",
        _has_server=has_server,
        _execute_method=execute,
        _update_delta_weights=delta,
        server_handle=SimpleNamespace(
            clear_kv_cache=SimpleNamespace(remote=clear), set_global_steps=SimpleNamespace(remote=steps)
        ),
        replica_rank=0,
        rollout_rank=0,
    )


@pytest.mark.parametrize("backend", [None, "nccl", "ipc"])
@pytest.mark.parametrize("use_shm", [False, True])
@pytest.mark.parametrize("has_server", [False, True])
def test_legacy_cuda_dispatch_contract(backend, use_shm, has_server):
    events = []
    adapter = _adapter(events, _config(backend), use_shm=use_shm, has_server=has_server)
    method = _load_update_method(events)
    asyncio.run(method(adapter, iter([("weight", 7)]), global_steps=5, base_sync_done=True))
    expected = [
        (
            "receiver",
            "update_weights_from_ipc",
            {"non_block": True, "kwargs": {"base_sync_done": True, "use_shm": use_shm}},
        ),
        ("sender", {"zmq_handle": "ipc://dispatch-test", "bucket_size_mb": 512, "use_shm": use_shm}),
        ("send", [("weight", 7)]),
    ]
    if has_server:
        expected.extend([("await_receiver",), ("clear_cache",), ("steps", 5)])
    assert events == expected


@pytest.mark.parametrize("backend, mode", [("npu_ipc", None)])
def test_native_mode_cannot_be_ignored_by_delta_flush(backend, mode):
    events = []
    adapter = _adapter(events, _config(backend, mode))
    with pytest.raises(ValueError, match="delta_flush"):
        asyncio.run(_load_update_method(events)(adapter, iter([]), wire_format="delta_flush"))
    assert events == []


def test_default_delta_flush_retains_existing_dispatch():
    events = []
    adapter = _adapter(events, _config())
    asyncio.run(_load_update_method(events)(adapter, iter([("weight", 7)]), global_steps=5, wire_format="delta_flush"))
    assert events == [("delta", [("weight", 7)], {"global_steps": 5})]


@pytest.mark.parametrize("backend, mode, expected", [("npu_ipc", None, "npu_ipc")])
def test_native_dispatch_preserves_kwargs_and_common_tail(backend, mode, expected):
    events = []
    adapter = _adapter(events, _config(backend, mode))
    method = _load_update_method(events, allow_native=True)
    asyncio.run(method(adapter, iter([("weight", 7)]), global_steps=5, base_sync_done=True, peft_config=None))
    assert events == [
        ("native", [("weight", 7)], {"mode": expected, "base_sync_done": True, "peft_config": None}),
        ("clear_cache",),
        ("steps", 5),
    ]
