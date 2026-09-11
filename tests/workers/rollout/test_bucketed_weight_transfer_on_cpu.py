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
import importlib.util
import multiprocessing as mp
import os
import sys
import types
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import pytest
import torch
import zmq

PROCESS_TIMEOUT = 10
ACK_TIMEOUT_MS = 1_000
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _handle():
    return f"ipc:///tmp/verl-bwt-{uuid.uuid4().hex}.sock"


def _transfer_module():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
    spec = importlib.util.spec_from_file_location("bucketed_weight_transfer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.get_torch_device = lambda: _CpuDevice()
    module.is_support_ipc = lambda: False
    return module


class _CpuDevice:
    def synchronize(self):
        pass

    def empty_cache(self):
        pass


def _install_rollout_utils():
    module = types.ModuleType("verl.workers.rollout.utils")

    async def ensure_async_iterator(iterable):
        if hasattr(iterable, "__aiter__"):
            async for item in iterable:
                yield item
        else:
            for item in iterable:
                yield item

    module.ensure_async_iterator = ensure_async_iterator
    sys.modules[module.__name__] = module


def _send_weights(handle, ready_event, weights, result_queue):
    try:
        _install_rollout_utils()
        uuid.uuid4 = lambda: types.SimpleNamespace(hex=f"{os.getpid():x}")
        sender = _transfer_module().BucketedWeightSender(
            handle, bucket_size_mb=1, use_shm=True, ack_timeout_ms=ACK_TIMEOUT_MS
        )
        init_socket = sender._init_socket

        def signal_ready():
            init_socket()
            ready_event.set()

        sender._init_socket = signal_ready
        asyncio.run(sender.async_send_weights(iter(weights)))
    except Exception as exc:
        result_queue.put((type(exc).__name__, str(exc)))
        ready_event.set()
    else:
        result_queue.put(None)


def _receive_weights(handle, result_queue, failure):
    bucketed_weight_transfer = _transfer_module()

    if failure == "init":

        def fail_rebuild(*_args, **_kwargs):
            raise ValueError("failed to initialize receiver buffer")

        bucketed_weight_transfer.rebuild_shared_memory = fail_rebuild

    receiver = bucketed_weight_transfer.BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)
    received = []

    def on_bucket_received(weights, is_last):
        received.extend(name for name, _ in weights)
        if failure == "callback":
            raise KeyError("embed_tokens.weight")

    try:
        receiver.receive_weights(on_bucket_received)
    except Exception as exc:
        result_queue.put((type(exc).__name__, str(exc)))
    else:
        result_queue.put(("ok", received))


def _reply_then_exit(handle):
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(handle)
    socket.recv_pyobj()
    socket.send(b"")
    socket.recv_pyobj()
    socket.close()
    os._exit(0)


def _wait(process):
    process.join(PROCESS_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(f"process {process.pid} did not exit within {PROCESS_TIMEOUT}s")


def _stop(process):
    if process is not None and process.is_alive():
        process.terminate()
        process.join()


def _assert_no_transfer_artifacts(handle, sender_pid):
    assert not os.path.exists(handle.removeprefix("ipc://"))
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=f"verl_weights_{sender_pid:x}")


def _run_sender_and_receiver(weights, failure):
    ctx = mp.get_context("spawn")
    handle = _handle()
    sender_ready = ctx.Event()
    sender_result = ctx.Queue()
    receiver_result = ctx.Queue()
    sender = ctx.Process(target=_send_weights, args=(handle, sender_ready, weights, sender_result))
    receiver = None
    try:
        sender.start()
        assert sender_ready.wait(PROCESS_TIMEOUT), "sender did not bind its ZMQ socket"
        receiver = ctx.Process(target=_receive_weights, args=(handle, receiver_result, failure))
        receiver.start()
        _wait(sender)
        sender_result_value = sender_result.get(timeout=1)
        _wait(receiver)
        return sender.exitcode, receiver.exitcode, sender_result_value, receiver_result.get(timeout=1)
    finally:
        sender_pid = sender.pid
        _stop(sender)
        _stop(receiver)
        if sender_pid is not None:
            _assert_no_transfer_artifacts(handle, sender_pid)


class _ReceiveFailureSocket:
    def recv_pyobj(self):
        raise ValueError("invalid weight metadata")

    def send(self, message, flags=0):
        self.message = message


class _RecvOnlySocket:
    def recv(self):
        return b""


class _TimeoutSocket:
    def poll(self, _timeout, _flags):
        return 0

    def recv(self):
        raise AssertionError("recv must not be called after a timeout")


def test_cpu_sender_ack_supports_recv_only_socket():
    sender = _transfer_module().BucketedWeightSender("ipc:///tmp/test-bwt-unused.sock")
    sender.socket = _RecvOnlySocket()

    sender._receive_ack("test")


def test_cpu_sender_ack_timeout_raises():
    sender = _transfer_module().BucketedWeightSender("ipc:///tmp/test-bwt-unused.sock", ack_timeout_ms=1)
    sender.socket = _TimeoutSocket()

    with pytest.raises(RuntimeError, match="timed out waiting 1ms"):
        sender._receive_ack("test")


def test_cpu_receiver_preserves_decode_failure(monkeypatch):
    receiver = _transfer_module().BucketedWeightReceiver("ipc:///tmp/unused.sock", torch.device("cpu"), use_shm=True)
    socket = _ReceiveFailureSocket()

    monkeypatch.setattr(receiver, "_init_socket", lambda: setattr(receiver, "socket", socket))
    monkeypatch.setattr(receiver, "_init_buffer", lambda: None)
    monkeypatch.setattr(receiver, "_cleanup", lambda: None)

    with pytest.raises(ValueError, match="invalid weight metadata"):
        receiver.receive_weights(lambda _weights, _is_last: None)

    assert socket.message == b"error:ValueError: invalid weight metadata"


def test_cpu_shared_memory_transfer_acknowledges_success():
    sender_exit, receiver_exit, sender_result, receiver_result = _run_sender_and_receiver(
        [("weight", torch.ones(2))], failure=None
    )

    assert sender_exit == 0
    assert receiver_exit == 0
    assert sender_result is None
    assert receiver_result == ("ok", ["weight"])


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("init", "ValueError: failed to initialize receiver buffer"),
        ("callback", "KeyError: 'embed_tokens.weight'"),
    ],
)
def test_cpu_receiver_failure_is_sent_to_sender(failure, expected):
    sender_exit, receiver_exit, sender_result, receiver_result = _run_sender_and_receiver([], failure)

    assert sender_exit == 0
    assert receiver_exit == 0
    assert sender_result == ("RuntimeError", f"weight receiver failed: {expected}")
    assert receiver_result[0] in {"ValueError", "KeyError"}


def test_cpu_receiver_failure_on_intermediate_bucket_reaches_sender():
    weights = [("first", torch.ones(150_000)), ("second", torch.ones(150_000))]
    sender_exit, receiver_exit, sender_result, receiver_result = _run_sender_and_receiver(weights, failure="callback")

    assert sender_exit == 0
    assert receiver_exit == 0
    assert sender_result == ("RuntimeError", "weight receiver failed: KeyError: 'embed_tokens.weight'")
    assert receiver_result == ("KeyError", "'embed_tokens.weight'")


def test_cpu_sender_times_out_after_peer_loss():
    ctx = mp.get_context("spawn")
    handle = _handle()
    sender_ready = ctx.Event()
    sender_result = ctx.Queue()
    sender = ctx.Process(target=_send_weights, args=(handle, sender_ready, [], sender_result))
    peer = ctx.Process(target=_reply_then_exit, args=(handle,))
    try:
        sender.start()
        assert sender_ready.wait(PROCESS_TIMEOUT), "sender did not bind its ZMQ socket"
        peer.start()
        _wait(sender)
        _wait(peer)

        assert sender.exitcode == 0
        assert peer.exitcode == 0
        assert sender_result.get(timeout=1) == (
            "RuntimeError",
            f"timed out waiting {ACK_TIMEOUT_MS}ms for weight receiver acknowledgement (final bucket)",
        )
    finally:
        _stop(sender)
        _stop(peer)
        if sender.pid is not None:
            _assert_no_transfer_artifacts(handle, sender.pid)
