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
import sys
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


# ---------------------------------------------------------------------------
# (c) Teardown is bounded too: a finite LINGER on both sockets.
#
# The bounded ACK above stops the sender waiting forever for a *reply*, but
# ZMQ's default LINGER of -1 means ``socket.close()`` can then block forever
# flushing an outbound frame the dead peer will never read — and close() runs
# from ``finally: _cleanup()``, i.e. on exactly the paths the bound just made
# reachable. These two tests pin the invariant at the level it must hold:
# the option is set when the socket is CREATED (not only on the receiver's
# error path, which can itself fail before it gets there), and the whole
# send+cleanup sequence completes in bounded time with an unflushable frame
# queued.
# ---------------------------------------------------------------------------
def test_both_sockets_get_a_finite_linger_at_creation():
    """LINGER must be finite on the real sockets, set by ``_init_socket``."""
    import zmq

    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        SOCKET_LINGER_MS,
        BucketedWeightReceiver,
        BucketedWeightSender,
    )

    assert 0 <= SOCKET_LINGER_MS < 2**31 - 1, f"LINGER must be finite, got {SOCKET_LINGER_MS}"

    handle = _unique_zmq_handle()
    sender = BucketedWeightSender(zmq_handle=handle, bucket_size_mb=1, use_shm=True)
    sender._init_socket()
    try:
        assert sender.socket.getsockopt(zmq.LINGER) == SOCKET_LINGER_MS, (
            "sender socket kept ZMQ's default LINGER=-1: close() in _cleanup() could block forever"
        )
        receiver = BucketedWeightReceiver(zmq_handle=handle, device=torch.device("cpu"), use_shm=True)
        receiver._init_socket()
        try:
            assert receiver.socket.getsockopt(zmq.LINGER) == SOCKET_LINGER_MS, (
                "receiver socket's finite LINGER is only set on the error path"
            )
        finally:
            receiver.socket.close()
    finally:
        sender.socket.close(linger=0)
        try:
            os.remove(handle[len("ipc://") :])
        except OSError:
            pass


def _receiver_fn_connects_then_leaves(zmq_handle, result_queue):
    """Connect (so frames are queued to us) and exit without ever reading one."""
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(zmq_handle)
    time.sleep(1.0)
    socket.close()
    context.term()
    result_queue.put(("left", None, None, 0.0))


def test_send_and_cleanup_are_bounded_when_frames_cannot_be_flushed():
    """Fault-inject an unflushable queued frame; total time must stay bounded.

    Without a finite LINGER this test hangs until the harness timeout: the ACK
    bound fires as designed, and then ``close()`` blocks forever inside the
    ``finally``. The assertion is deliberately on the WHOLE sequence, not just
    on ``_recv_ack``, because a bound that the cleanup path can undo is not a
    deadlock bound.
    """
    sender_result, _ = _run_pair(_receiver_fn_connects_then_leaves, (), num_weights=1)

    status, exc_name, message, elapsed = sender_result
    assert status == "raised", f"expected the bounded ACK to fire, got {sender_result}"
    assert exc_name in ("WeightTransferAckTimeoutError", "WeightTransferReceiverError"), sender_result
    # ACK bound + LINGER (5 s) + slack. A LINGER=-1 regression blows straight
    # past this and is killed by PROCESS_TIMEOUT_S instead.
    bound_s = ACK_TIMEOUT_S + 30
    assert elapsed < bound_s, (
        f"send+cleanup took {elapsed:.1f} s (bound {bound_s:.0f} s): teardown is not bounded — "
        "check that both sockets set a finite ZMQ LINGER at creation"
    )
    # Whichever bound fired, the diagnostic must name the bound that was
    # actually armed for THIS sender (ack_timeout_s=5 s), not the module-level
    # send timeout. An operator who reads "1800 s" after a 5 s failure debugs
    # the wrong knob.
    if "trying to send" in message:
        assert f"Timed out after {ACK_TIMEOUT_S:.0f} s" in message, f"send timeout reported the wrong bound: {message}"


def test_send_timeout_message_reports_the_effective_bound():
    """The send-side diagnostic must quote the EFFECTIVE SNDTIMEO.

    ``_init_socket`` arms ``min(ack_timeout_s * 1000, SOCKET_SEND_TIMEOUT_MS)``,
    so with the caller-supplied bound below the module default the two differ by
    orders of magnitude (5 s vs 1800 s). Reporting the constant instead of the
    armed value sends the reader after the wrong timeout: single process, no
    peer ever connects, so the very first handshake send hits SNDTIMEO.
    """
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        SOCKET_SEND_TIMEOUT_MS,
        BucketedWeightSender,
        WeightTransferAckTimeoutError,
    )

    short_bound_s = 1.0
    assert int(short_bound_s * 1000) < SOCKET_SEND_TIMEOUT_MS, "the bound under test must win the min"

    handle = _unique_zmq_handle()
    sender = BucketedWeightSender(
        zmq_handle=handle,
        bucket_size_mb=1,
        use_shm=True,
        ack_timeout_s=short_bound_s,
    )
    started = time.monotonic()
    try:
        sender._init_socket()
        assert sender._snd_timeout_ms == int(short_bound_s * 1000), sender._snd_timeout_ms
        with pytest.raises(WeightTransferAckTimeoutError) as excinfo:
            sender._send_or_timeout({"probe": True}, "unit-test handshake")
    finally:
        if sender.socket is not None:
            sender.socket.close(linger=0)
        try:
            os.remove(handle[len("ipc://") :])
        except OSError:
            pass
    elapsed = time.monotonic() - started

    message = str(excinfo.value)
    assert f"Timed out after {short_bound_s:.0f} s" in message, message
    assert f"{SOCKET_SEND_TIMEOUT_MS / 1000:.0f} s" not in message, (
        f"the diagnostic still quotes the module default instead of the armed bound: {message}"
    )
    # And the wait really was the armed bound, so the message is not merely
    # consistent with itself.
    assert short_bound_s <= elapsed < short_bound_s + 30, f"send gave up after {elapsed:.1f} s"


def test_cleanup_preserves_the_primary_exception_when_the_accelerator_fails():
    """A failing accelerator teardown must not replace the original traceback.

    Observed in a 0.20 aggregate log: a CUDA init failure inside
    ``_init_buffer()`` was overwritten by a *second* CUDA failure raised from
    ``ipc_collect()`` in ``_cleanup()``, so the reported error named the wrong
    call. ``_cleanup()`` is best-effort by construction; this pins it.
    """
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    class _BrokenDevice:
        @staticmethod
        def synchronize():
            raise RuntimeError("secondary failure: device synchronize")

        @staticmethod
        def ipc_collect():
            raise RuntimeError("secondary failure: ipc_collect")

        @staticmethod
        def empty_cache():
            raise RuntimeError("secondary failure: empty_cache")

    original = bwt.get_torch_device
    bwt.get_torch_device = lambda: _BrokenDevice
    try:
        sender = BucketedWeightSender(zmq_handle=_unique_zmq_handle(), bucket_size_mb=1, use_shm=True)
        try:
            raise ValueError("primary failure: the error the caller must see")
        except ValueError:
            # Exactly the shape of async_send_weights: raise, then finally-cleanup.
            sender._cleanup()
            surfaced = sys.exc_info()[1]
        assert isinstance(surfaced, ValueError), f"cleanup replaced the primary exception with {surfaced!r}"
        assert "primary failure" in str(surfaced), str(surfaced)
    finally:
        bwt.get_torch_device = original
