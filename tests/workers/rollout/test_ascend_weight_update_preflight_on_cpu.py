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

"""Exercise real preflight/control-flow code without importing device runtimes."""

import ast
import asyncio
import builtins
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_ROLLOUT = _ROOT / "verl/workers/rollout/vllm_rollout"


def _load_functions(filename, names, namespace):
    """Compile the actual functions only, avoiding unrelated vLLM imports."""
    path = _ROLLOUT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def native_dependencies(monkeypatch):
    names = [
        "torch_npu",
        "torch_npu.multiprocessing",
        "torch_npu.multiprocessing.reductions",
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "vllm_ascend.distributed.weight_transfer.packed_tensor",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__path__ = []
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
        monkeypatch.setitem(sys.modules, name, module)
    modules["torch_npu.multiprocessing.reductions"].rebuild_npu_tensor = lambda *args: None
    modules["vllm_ascend.distributed.weight_transfer.npu_ipc_engine"].npu_generate_uuid = lambda *args: "npu-0"

    def packed_consumer(ipc_handle, physical_npu_id, names, shapes, dtype_names, tensor_sizes, device_index):
        raise AssertionError("Preflight must not reconstruct or copy tensors")

    modules["vllm_ascend.distributed.weight_transfer.packed_tensor"].packed_npu_ipc_consumer = packed_consumer
    return modules


@pytest.fixture
def worker_preflight():
    namespace = _load_functions(
        "utils.py",
        ["validate_ascend_weight_update"],
        {"get_device_name": lambda: "npu", "is_quantized_model": lambda config: False, "torch": torch},
    )

    class Worker:
        def __init__(self):
            self.model_runner = SimpleNamespace(
                vllm_config=SimpleNamespace(weight_transfer_config=SimpleNamespace(backend="npu_ipc"))
            )
            self.device = SimpleNamespace(type="npu", index=0)
            self._is_qat_model = False
            self._is_modelopt_qat = False
            self._ascend_weight_update_failed = False

        def _use_mtp_drafter_weight_sync(self):
            return False

        def init_weight_transfer_engine(self, init_info):
            raise AssertionError("Preflight must not initialize the engine")

        def start_weight_update(self):
            raise AssertionError("Preflight must not start an update")

        def update_weights(self, update_info):
            raise AssertionError("Preflight must not load weights")

        def finish_weight_update(self):
            raise AssertionError("Preflight must not finalize weights")

        def reload_weights(self, weights_iterator, is_checkpoint_format):
            raise AssertionError("Preflight must not reload weights")

    Worker.validate = namespace["validate_ascend_weight_update"]
    return Worker(), namespace


@pytest.mark.parametrize("mode", ["npu_ipc", "hccl"])
def test_cuda_rejected_before_importing_ascend(worker_preflight, monkeypatch, mode):
    worker, namespace = worker_preflight
    namespace["get_device_name"] = lambda: "cuda"
    original_import = builtins.__import__

    def no_ascend_import(name, *args, **kwargs):
        assert not name.startswith(("torch_npu", "vllm_ascend")), "CUDA path imported an Ascend runtime"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ascend_import)
    with pytest.raises(ValueError, match="requires Ascend NPU"):
        worker.validate(mode)


@pytest.mark.parametrize("mode", ["npu_ipc", "hccl"])
def test_valid_worker_preflight_does_not_call_loading_methods(worker_preflight, native_dependencies, mode):
    worker, _ = worker_preflight
    worker.model_runner.vllm_config.weight_transfer_config.backend = mode
    worker.validate(mode)


@pytest.mark.parametrize("mode", ["npu_ipc", "hccl"])
def test_receiver_backend_must_match_selected_mode(worker_preflight, native_dependencies, mode):
    worker, _ = worker_preflight
    worker.model_runner.vllm_config.weight_transfer_config.backend = "other"
    with pytest.raises(ValueError, match="backend must match"):
        worker.validate(mode)


@pytest.mark.parametrize("mode", ["npu_ipc"])
def test_shared_memory_is_rejected_for_native_ipc(worker_preflight, mode):
    worker, _ = worker_preflight
    with pytest.raises(ValueError, match="shared memory"):
        worker.validate(mode, use_shm=True)


@pytest.mark.parametrize("case", ["lora", "qat", "modelopt", "quantized", "mtp", "previous_failure"])
def test_worker_rejects_unsupported_model_state(worker_preflight, case):
    worker, namespace = worker_preflight
    kwargs = {}
    if case == "lora":
        kwargs["peft_config"] = {"rank": 8}
    elif case == "qat":
        worker._is_qat_model = True
    elif case == "modelopt":
        worker._is_modelopt_qat = True
    elif case == "quantized":
        namespace["is_quantized_model"] = lambda config: True
    elif case == "mtp":
        worker._use_mtp_drafter_weight_sync = lambda: True
    else:
        worker._ascend_weight_update_failed = True
    error = RuntimeError if case == "previous_failure" else ValueError
    with pytest.raises(error, match="Previous|unquantized|quantized|MTP"):
        worker.validate("npu_ipc", **kwargs)


@pytest.mark.parametrize(
    "method", ["init_weight_transfer_engine", "start_weight_update", "update_weights", "finish_weight_update"]
)
@pytest.mark.parametrize("failure", ["missing", "signature"])
def test_worker_method_capabilities_are_validated(worker_preflight, native_dependencies, method, failure):
    worker, _ = worker_preflight
    setattr(worker, method, None if failure == "missing" else lambda unknown_required_argument: None)
    with pytest.raises(RuntimeError, match=method):
        worker.validate("reload" if method == "reload_weights" else "npu_ipc")


@pytest.mark.parametrize("mode", ["npu_ipc"])
@pytest.mark.parametrize("failure", ["missing", "signature"])
def test_official_packed_consumer_capability_is_checked(worker_preflight, native_dependencies, mode, failure):
    worker, _ = worker_preflight
    worker.model_runner.vllm_config.weight_transfer_config.backend = mode
    module = native_dependencies["vllm_ascend.distributed.weight_transfer.packed_tensor"]
    module.packed_npu_ipc_consumer = None if failure == "missing" else lambda incompatible: None
    with pytest.raises(RuntimeError, match="packed|IPC"):
        worker.validate(mode)


@pytest.fixture
def coordinator(monkeypatch):
    events = []
    state = SimpleNamespace(
        prepare_failure=False, send_failure=False, receive_failure=False, rpc_failure=False, cancelled=False
    )
    dist = torch.distributed
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    namespace = _load_functions(
        "ascend_weight_transfer.py",
        ["_check_preflight_errors", "update_ascend_weights"],
        {"logger": logging.getLogger(__name__)},
    )

    class Sender:
        def __init__(self, **kwargs):
            events.append("create")

        def prepare(self):
            events.append("prepare")
            if state.prepare_failure:
                raise ValueError("buffer allocation failed")

        async def async_send_weights(self, weights):
            assert self.receiver_started
            events.append("send")
            if state.cancelled:
                raise asyncio.CancelledError("training task cancelled")
            if state.send_failure:
                raise RuntimeError("source iterator failed")
            events.extend(("weight", name) for name, value in weights)

        def close(self):
            events.append("close")

    class ReceiverResult:
        def __await__(self):
            async def wait():
                events.append("await_receiver")
                if state.receive_failure:
                    raise RuntimeError("receiver load failed")

            return wait().__await__()

    result = ReceiverResult()

    async def execute(method, **kwargs):
        events.append(method)
        if method == "validate_ascend_weight_update":
            if state.rpc_failure:
                raise ValueError("worker capability missing")
            return None
        assert kwargs["non_block"] is True
        return result

    adapter = SimpleNamespace(
        config=SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)),
        zmq_handle="unused",
        use_shm=False,
        _execute_method=execute,
    )
    namespace["validate_ascend_sender"] = lambda *args: None
    namespace["AscendNativeIPCWeightSender"] = Sender
    ray = types.ModuleType("ray")
    ray.cancel = lambda future, force: events.append(("cancel", future is result, force))
    monkeypatch.setitem(sys.modules, "ray", ray)
    return SimpleNamespace(events=events, state=state, namespace=namespace, adapter=adapter, ray=ray)


def _run_update(coordinator, mode="npu_ipc"):
    return asyncio.run(
        coordinator.namespace["update_ascend_weights"](
            coordinator.adapter,
            iter([("weight", torch.ones(1))]),
            mode=mode,
        )
    )


@pytest.mark.parametrize("failure", ["prepare", "worker", "previous"])
def test_preflight_failure_never_starts_receiver(coordinator, failure):
    if failure == "prepare":
        coordinator.state.prepare_failure = True
    elif failure == "worker":
        coordinator.state.rpc_failure = True
    else:
        coordinator.adapter._ascend_weight_update_failed = True
    with pytest.raises(RuntimeError, match="preflight failed"):
        _run_update(coordinator)
    assert "send" not in coordinator.events
    assert "update_weights_from_ascend_ipc" not in coordinator.events
    assert "update_weights_from_ascend_reload" not in coordinator.events
    if failure != "previous":
        assert coordinator.events[-1] == "close"


@pytest.mark.parametrize("failed_phase", [1, 2])
def test_peer_rank_failure_blocks_local_receiver(coordinator, monkeypatch, failed_phase):
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, local_error):
        calls.append(local_error)
        output[:] = [local_error, "peer preflight failed" if len(calls) == failed_phase else None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 1: peer preflight failed"):
        _run_update(coordinator)
    assert len(calls) == failed_phase
    assert "update_weights_from_ascend_ipc" not in coordinator.events
    assert coordinator.events[-1] == "close"


@pytest.mark.parametrize("failure", ["prepare", "previous"])
def test_local_preflight_error_still_participates_in_rank_coordination(coordinator, monkeypatch, failure):
    if failure == "prepare":
        coordinator.state.prepare_failure = True
    else:
        coordinator.adapter._ascend_weight_update_failed = True
    errors = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, error):
        errors.append(error)
        output[:] = [error, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank 0"):
        _run_update(coordinator)
    assert len(errors) == 1 and errors[0]
    assert "update_weights_from_ascend_ipc" not in coordinator.events


@pytest.mark.parametrize("failure", ["send", "receive", "cancel_also_fails"])
def test_runtime_failure_marks_invalid_and_cleans_up(coordinator, failure):
    coordinator.state.send_failure = failure != "receive"
    coordinator.state.receive_failure = failure == "receive"
    if failure == "cancel_also_fails":

        def cancel(*args, **kwargs):
            raise ValueError("cancellation failed")

        coordinator.ray.cancel = cancel
    expected = "receiver load failed" if failure == "receive" else "source iterator failed"
    with pytest.raises(RuntimeError, match=expected):
        _run_update(coordinator)
    assert coordinator.adapter._ascend_weight_update_failed is True
    assert coordinator.events[-1] == "close"
    if failure != "cancel_also_fails":
        assert ("cancel", True, False) in coordinator.events


def test_task_cancellation_marks_invalid_cancels_receiver_and_closes_sender(coordinator):
    coordinator.state.cancelled = True
    with pytest.raises(asyncio.CancelledError, match="training task cancelled"):
        _run_update(coordinator)
    assert coordinator.adapter._ascend_weight_update_failed is True
    assert ("cancel", True, False) in coordinator.events
    assert coordinator.events[-1] == "close"


@pytest.mark.parametrize("mode", ["npu_ipc"])
def test_success_prepares_before_rpc_then_awaits_receiver(coordinator, mode):
    _run_update(coordinator, mode)
    method = "update_weights_from_ascend_reload" if mode == "reload" else "update_weights_from_ascend_ipc"
    assert coordinator.events == [
        "create",
        "prepare",
        "validate_ascend_weight_update",
        method,
        "send",
        ("weight", "weight"),
        "await_receiver",
        "close",
    ]
    assert not getattr(coordinator.adapter, "_ascend_weight_update_failed", False)
