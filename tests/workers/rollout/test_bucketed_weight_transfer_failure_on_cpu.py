# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Real two-process failure tests for the bucketed weight-transfer IPC.

``tests/utils/test_bucketed_weight_transfer.py`` covers the HAPPY path across
two real processes (and needs an accelerator for the CUDA-IPC arm). This suite
covers the FAILURE path, which is what the FP8 layerwise-reload resync made
load-bearing: the reload lifecycle can raise from inside the receiver's
``on_bucket_received`` callback (a bad bucket, a quantization error, a
lifecycle violation), and the sender must not be left waiting forever.

The protocol is ZMQ REQ (sender, binds) / REP (receiver, connects) with strict
alternation, and the driver launches the receiver worker non-blocking and only
awaits its future AFTER sending completes. So a receiver that raises without
answering the outstanding request converts an intended fail-closed error into a
silent training hang whose real traceback is buried in a worker log. The two
mechanisms under test:

  * the receiver answers with an ACK ERROR FRAME before propagating its own
    exception, so the sender raises ``WeightTransferReceiverError`` carrying the
    receiver-side message;
  * every sender ``recv`` is BOUNDED, so a receiver that dies without reporting
    (SIGKILL, segfault) surfaces as ``WeightTransferAckTimeoutError`` instead
    of a hang.

Both run over a real ``ipc://`` socket between two real ``spawn``ed processes,
on the shared-memory (``use_shm=True``) path so the whole file is CPU-only and
collected by verl's ``*_on_cpu.py`` CI job. The success-path assertion in
``test_normal_transfer_still_succeeds`` is the regression guard that the error
frame and the bounded recv did not disturb the pre-existing BF16/LoRA
behaviour.
"""

import asyncio
import multiprocessing as mp
import os
import queue
import signal
import time
import uuid

import pytest
import torch

PROCESS_TIMEOUT_S = 120
# Deliberately short: these tests assert the sender gives up, so the bound has
# to be small enough to keep the suite fast while staying far above the
# microseconds a local ipc:// round trip actually takes.
ACK_TIMEOUT_S = 5.0


def _unique_zmq_handle():
    return f"ipc:///tmp/test-bwt-fail-{uuid.uuid4().hex}.sock"


def _patch_device_for_cpu():
    """Make the sender/receiver teardown accelerator-free inside a child process."""
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt

    class _CpuDeviceModule:
        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def ipc_collect():
            pass

        @staticmethod
        def empty_cache():
            pass

    bwt.get_torch_device = lambda: _CpuDeviceModule


# ---------------------------------------------------------------------------
# Process entry points (module level so `spawn` can pickle them)
# ---------------------------------------------------------------------------
def _sender_fn(zmq_handle, num_weights, result_queue, ack_timeout_s):
    """Send `num_weights` small tensors; report how the send ended."""
    _patch_device_for_cpu()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    weights = [(f"layer{i}.weight", torch.full((16, 16), float(i))) for i in range(num_weights)]
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=1,
        use_shm=True,
        ack_timeout_s=ack_timeout_s,
    )
    started = time.monotonic()
    try:
        asyncio.run(sender.async_send_weights(iter(weights)))
    except BaseException as exc:  # noqa: BLE001 - the outcome IS the assertion
        result_queue.put(("raised", type(exc).__name__, str(exc), time.monotonic() - started))
    else:
        result_queue.put(("completed", None, None, time.monotonic() - started))


def _receiver_fn_raises_on_bucket(zmq_handle, fail_on_bucket_index, result_queue):
    """Receiver whose per-bucket callback raises — the FP8 reload failure shape."""
    _patch_device_for_cpu()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=torch.device("cpu"), use_shm=True)
    seen = {"buckets": 0}

    def _on_bucket(weights):
        index = seen["buckets"]
        seen["buckets"] += 1
        if index == fail_on_bucket_index:
            raise RuntimeError("synthetic receiver failure while loading bucket weights")

    try:
        receiver.receive_weights(on_bucket_received=_on_bucket)
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("raised", type(exc).__name__, str(exc), seen["buckets"]))
    else:
        result_queue.put(("completed", None, None, seen["buckets"]))


def _receiver_fn_dies_silently(zmq_handle, result_queue):
    """Receiver that is SIGKILLed mid-transfer: no error frame can be sent.

    This is the case the error frame cannot cover and the bounded recv must:
    the process disappears between the sender's send and its ACK.
    """
    _patch_device_for_cpu()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=torch.device("cpu"), use_shm=True)

    def _on_bucket(weights):
        result_queue.put(("about_to_die", None, None, 0))
        os.kill(os.getpid(), signal.SIGKILL)

    receiver.receive_weights(on_bucket_received=_on_bucket)


def _receiver_fn_success(zmq_handle, result_queue):
    """Plain receiver: the pre-existing success path, unchanged."""
    _patch_device_for_cpu()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=torch.device("cpu"), use_shm=True)
    received = []
    receiver.receive_weights(on_bucket_received=lambda w: received.extend((n, t.clone()) for n, t in w))
    result_queue.put(("completed", None, [(n, float(t.flatten()[0])) for n, t in received], len(received)))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def _run_pair(receiver_target, receiver_args, num_weights, ack_timeout_s=ACK_TIMEOUT_S):
    """Run one sender + one receiver process; return (sender_result, receiver_result)."""
    zmq_handle = _unique_zmq_handle()
    ctx = mp.get_context("spawn")
    sender_queue = ctx.Queue()
    receiver_queue = ctx.Queue()

    sender_p = ctx.Process(target=_sender_fn, args=(zmq_handle, num_weights, sender_queue, ack_timeout_s))
    receiver_p = ctx.Process(target=receiver_target, args=(zmq_handle, *receiver_args, receiver_queue))

    # Sender binds first, then the receiver connects (matches production order).
    sender_p.start()
    receiver_p.start()
    try:
        sender_p.join(timeout=PROCESS_TIMEOUT_S)
        receiver_p.join(timeout=PROCESS_TIMEOUT_S)

        assert not sender_p.is_alive(), (
            f"sender still running after {PROCESS_TIMEOUT_S} s — it is HUNG waiting for an ACK, "
            "which is exactly the failure mode the error frame and the bounded recv exist to prevent"
        )

        try:
            sender_result = sender_queue.get(timeout=10)
        except queue.Empty:  # pragma: no cover - only on a genuinely stuck sender
            raise AssertionError("sender process reported no outcome") from None
        try:
            receiver_result = receiver_queue.get(timeout=10)
        except queue.Empty:
            receiver_result = None
        return sender_result, receiver_result
    finally:
        for proc in (sender_p, receiver_p):
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
        ipc_path = zmq_handle[len("ipc://") :]
        try:
            os.remove(ipc_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# (c) Regression guard: the success path is untouched.
# ---------------------------------------------------------------------------
def test_normal_transfer_still_succeeds():
    """The error frame and the bounded recv must not change the happy path.

    Sent values arrive intact across a real two-process shared-memory transfer,
    and both sides exit cleanly — the guard that BF16/LoRA/existing callers see
    no behaviour change.
    """
    sender_result, receiver_result = _run_pair(_receiver_fn_success, (), num_weights=4)

    assert sender_result[0] == "completed", f"sender did not finish cleanly: {sender_result}"
    assert receiver_result is not None and receiver_result[0] == "completed"
    names_values = receiver_result[2]
    assert names_values == [(f"layer{i}.weight", float(i)) for i in range(4)], names_values


# ---------------------------------------------------------------------------
# (a) A receiver that raises must not strand the sender.
# ---------------------------------------------------------------------------
def test_receiver_raise_surfaces_on_sender_without_hanging():
    """A raise inside on_bucket_received reaches the sender as an exception.

    The receiver fails on the FIRST bucket — the shape a bad weight or an
    immediate reload-lifecycle violation takes. Without the error frame the
    sender would sit in an un-timed ``recv()`` forever, because the driver only
    awaits the receiver's future after sending completes.
    """
    sender_result, receiver_result = _run_pair(
        _receiver_fn_raises_on_bucket,
        (0,),
        num_weights=2,
    )

    status, exc_name, message, elapsed = sender_result
    assert status == "raised", f"sender did NOT observe the receiver failure: {sender_result}"
    assert exc_name == "WeightTransferReceiverError", sender_result
    # The receiver-side message must travel, otherwise the operator is left
    # grepping worker logs for the real cause.
    assert "synthetic receiver failure while loading bucket weights" in message, message
    assert "RuntimeError" in message, message
    # Bounded by construction: an error frame is immediate, so this must be
    # far below the ACK timeout (which is itself only a deadlock backstop).
    assert elapsed < ACK_TIMEOUT_S, f"error frame took {elapsed:.1f} s, expected an immediate report"

    # The receiver must still propagate ITS OWN original exception. This is not
    # automatic: the per-bucket tensors are views into the shared-memory buffer
    # and the traceback keeps them alive, so an unguarded ``shm.close()`` in
    # _cleanup() raises `BufferError: cannot close exported pointers exist` from
    # the `finally` and REPLACES the real cause in the worker log.
    assert receiver_result is not None
    assert receiver_result[0] == "raised", "the receiver must still propagate its own exception"
    assert receiver_result[1] == "RuntimeError", (
        f"the receiver's original exception was masked by a teardown failure: "
        f"{receiver_result[1]}: {receiver_result[2]}"
    )
    assert "synthetic receiver failure" in receiver_result[2], receiver_result[2]


def test_receiver_raise_on_a_later_bucket_surfaces_on_sender():
    """Same mechanism, but the failure happens after a bucket was ACKed.

    Weights are sized so the transfer spans several buckets, and the receiver
    fails on the second one. This is the realistic FP8 shape: early buckets
    load fine and the lifecycle only breaks part-way through the sync.
    """
    zmq_handle = _unique_zmq_handle()
    ctx = mp.get_context("spawn")
    sender_queue = ctx.Queue()
    receiver_queue = ctx.Queue()

    # 20 x 256 KiB weights against a 1 MB bucket => ~5 buckets.
    sender_p = ctx.Process(
        target=_sender_fn_large,
        args=(zmq_handle, 20, sender_queue, ACK_TIMEOUT_S),
    )
    receiver_p = ctx.Process(
        target=_receiver_fn_raises_on_bucket,
        args=(zmq_handle, 1, receiver_queue),
    )
    sender_p.start()
    receiver_p.start()
    try:
        sender_p.join(timeout=PROCESS_TIMEOUT_S)
        receiver_p.join(timeout=PROCESS_TIMEOUT_S)
        assert not sender_p.is_alive(), "sender HUNG after a mid-stream receiver failure"

        status, exc_name, message, elapsed = sender_queue.get(timeout=10)
        assert status == "raised", f"sender did not observe the mid-stream failure: {status}"
        assert exc_name == "WeightTransferReceiverError", exc_name
        assert "synthetic receiver failure while loading bucket weights" in message, message
        assert elapsed < ACK_TIMEOUT_S, f"error frame took {elapsed:.1f} s"

        receiver_status, _, _, buckets_seen = receiver_queue.get(timeout=10)
        assert receiver_status == "raised"
        assert buckets_seen == 2, f"expected the failure on the 2nd bucket, saw {buckets_seen}"
    finally:
        for proc in (sender_p, receiver_p):
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)
        try:
            os.remove(zmq_handle[len("ipc://") :])
        except OSError:
            pass


def _sender_fn_large(zmq_handle, num_weights, result_queue, ack_timeout_s):
    """Sender with weights large enough to span multiple 1 MB buckets."""
    _patch_device_for_cpu()
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    # 256 KiB each (65536 float32 elements).
    weights = [(f"layer{i}.weight", torch.full((256, 256), float(i))) for i in range(num_weights)]
    sender = BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=1, use_shm=True, ack_timeout_s=ack_timeout_s)
    started = time.monotonic()
    try:
        asyncio.run(sender.async_send_weights(iter(weights)))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("raised", type(exc).__name__, str(exc), time.monotonic() - started))
    else:
        result_queue.put(("completed", None, None, time.monotonic() - started))


def test_sender_recv_is_bounded_when_the_receiver_dies_silently():
    """A receiver killed mid-bucket cannot send an error frame; the bound must fire.

    SIGKILL leaves no chance to answer the outstanding request, so this is the
    residual hang the error frame alone cannot close. The sender must raise
    ``WeightTransferAckTimeoutError`` shortly after the configured bound rather
    than block forever.
    """
    sender_result, _ = _run_pair(_receiver_fn_dies_silently, (), num_weights=2)

    status, exc_name, message, elapsed = sender_result
    assert status == "raised", f"sender did not give up on a dead receiver: {sender_result}"
    assert exc_name == "WeightTransferAckTimeoutError", sender_result
    assert "Timed out" in message, message
    # It waited for roughly the bound, then gave up — not forever, not instantly.
    assert ACK_TIMEOUT_S <= elapsed < ACK_TIMEOUT_S + 60, f"gave up after {elapsed:.1f} s, bound was {ACK_TIMEOUT_S} s"


# ---------------------------------------------------------------------------
# (b) Configuration errors are detected BEFORE any IPC resource exists.
# ---------------------------------------------------------------------------
def test_unsupported_configuration_is_rejected_before_the_receiver_is_built():
    """The pre-IPC gate must run before ``BucketedWeightReceiver`` is constructed.

    This is the structural half of the fix: the checks that used to raise from
    inside the worker after the socket existed (MTP drafter, an unvalidated
    vLLM version, a poisoned worker) now run at a point where raising can only
    fail the sync loudly. Asserted on the real source of
    ``update_weights_from_ipc`` so a future refactor that moves the validation
    back below the receiver fails here.
    """
    import inspect

    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    source = inspect.getsource(vLLMColocateWorkerExtension.update_weights_from_ipc)
    validate_at = source.find("validate_fp8_layerwise_reload_config(")
    receiver_at = source.find("BucketedWeightReceiver(")

    assert validate_at != -1, "update_weights_from_ipc no longer calls validate_fp8_layerwise_reload_config"
    assert receiver_at != -1, "update_weights_from_ipc no longer builds a BucketedWeightReceiver"
    assert validate_at < receiver_at, (
        "the FP8 configuration validation must run BEFORE the receiver (and therefore before any "
        "socket or shared buffer exists): the sender waits for an un-timed initial ACK, so raising "
        "once IPC is live turns a fail-closed error into a driver hang"
    )


def test_pre_ipc_validation_rejects_unsupported_config_without_touching_a_socket(monkeypatch):
    """The gate itself needs nothing but a config — no socket, no buffer, no model.

    That is what makes the ordering above possible: the checks are decidable
    from the rollout config plus the installed vLLM version.
    """
    from packaging import version

    from verl.utils.vllm import vllm_fp8_utils

    config = type("_Cfg", (), {"quant_config": None})()

    monkeypatch.setattr(vllm_fp8_utils, "_get_vllm_version", lambda: version.parse("0.23.0"))
    monkeypatch.setattr(vllm_fp8_utils, "_vllm_layerwise_reload_available", lambda: True)

    # Supported: silent.
    vllm_fp8_utils.validate_fp8_layerwise_reload_config(config, uses_mtp_drafter=False)

    # Unsupported: raises, with no IPC resource involved.
    with pytest.raises(NotImplementedError, match="MTP drafter"):
        vllm_fp8_utils.validate_fp8_layerwise_reload_config(config, uses_mtp_drafter=True)
