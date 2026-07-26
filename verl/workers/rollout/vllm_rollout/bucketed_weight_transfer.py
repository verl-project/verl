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
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Marker prefix for a receiver -> sender ERROR frame, sent in place of a
# normal empty ACK when the receiver fails while handling a bucket (or while
# building the communication buffer). Both parties speak REQ/REP with strict
# alternation, so the failing receiver's last act is to answer the outstanding
# request with this frame; without it the sender blocks in recv() forever and
# an intended fail-closed error becomes a silent training hang, with the real
# traceback buried in the receiver worker's log.
ACK_ERROR_PREFIX = b"VERL_WEIGHT_TRANSFER_ERROR:"

# Upper bound (seconds) on how long the sender waits for any single ACK. This
# is a deadlock bound, not a performance knob: it only fires when the receiver
# died without answering (e.g. killed mid-bucket, or a failure before its
# first send). Keep it far above real per-bucket latency — an ACK covers one
# bucket's load_weights + quantization + device sync.
DEFAULT_ACK_TIMEOUT_S = float(os.getenv("VERL_WEIGHT_TRANSFER_ACK_TIMEOUT_S", "1800"))

# Upper bound (milliseconds) on how long socket.close() may block flushing
# queued outbound frames. ZMQ's default is LINGER=-1 ("block forever"), which
# makes teardown itself a deadlock site: after an ACK timeout or a dead peer,
# close() inside the `finally: _cleanup()` can wait indefinitely for a frame
# nobody will ever read, so the bounded-ACK guarantee above would be undone by
# the very cleanup that follows it. Set on BOTH sockets at creation time — not
# only on the receiver's error path — so the bound holds for every exit path,
# including the success path and failures that happen before any error frame
# could be sent. 5000 ms is generous for a local IPC/TCP flush while still
# finite.
SOCKET_LINGER_MS = int(os.getenv("VERL_WEIGHT_TRANSFER_LINGER_MS", "5000"))

# Upper bound (milliseconds) on how long a single socket.send() may block. ZMQ's
# default is -1 ("block forever"), and for a REQ socket that is a real deadlock
# site independent of the ACK bound: if the peer has disconnected, there is no
# pipe to queue the frame on, so send() blocks *before* any recv() the bound
# guards. Measured, not hypothetical — a receiver that connects and exits without
# reading wedges the sender inside
# ``_init_buffer -> socket.send_pyobj(comm_metadata)``. Bounding send turns that
# into an exception on the driver. Shares the ACK timeout's order of magnitude
# because both answer the same question ("is the peer still there?").
SOCKET_SEND_TIMEOUT_MS = int(os.getenv("VERL_WEIGHT_TRANSFER_SEND_TIMEOUT_MS", "1800000"))


class WeightTransferReceiverError(RuntimeError):
    """Raised on the sender when the receiver reported a failure via an ACK error frame."""


class WeightTransferAckTimeoutError(RuntimeError):
    """Raised on the sender when an ACK did not arrive within the bound."""


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        ack_timeout_s: float | None = None,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm
        self.ack_timeout_s = DEFAULT_ACK_TIMEOUT_S if ack_timeout_s is None else ack_timeout_s

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def _recv_ack(self, what: str):
        """Wait for one receiver ACK, bounded in time and aware of error frames.

        The receiver is launched non-blocking by the caller and its future is
        awaited only after sending completes, so a receiver that dies without
        answering would otherwise leave this sender blocked in an un-timed
        recv() forever. Both failure modes surface here as an exception instead:
        an explicit error frame (receiver raised and reported it) or a timeout
        (receiver died without reporting).
        """
        if self.ack_timeout_s and self.ack_timeout_s > 0:
            if not self.socket.poll(timeout=int(self.ack_timeout_s * 1000), flags=zmq.POLLIN):
                raise WeightTransferAckTimeoutError(
                    f"Timed out after {self.ack_timeout_s:.0f} s waiting for the weight-transfer "
                    f"receiver's ACK ({what}). The receiver worker is unresponsive or died without "
                    "reporting an error; check the rollout worker log for the original traceback."
                )
        ack = self.socket.recv()
        if ack.startswith(ACK_ERROR_PREFIX):
            raise WeightTransferReceiverError(
                f"Weight-transfer receiver failed ({what}): "
                f"{ack[len(ACK_ERROR_PREFIX) :].decode('utf-8', errors='replace')}"
            )
        return ack

    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    get_torch_device().synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self._recv_ack("intermediate bucket")
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                        f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                    )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].view(dtype=weight.dtype).view(weight.shape).copy_(
                    weight, non_blocking=True
                )
                offset += weight.nbytes

            # send the last bucket
            get_torch_device().synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self._recv_ack("final bucket")
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        # Finite LINGER at creation: _cleanup()'s close() must never block
        # forever on an unflushable frame (see SOCKET_LINGER_MS).
        self.socket.setsockopt(zmq.LINGER, SOCKET_LINGER_MS)
        # Finite SNDTIMEO: send() itself blocks forever on a REQ socket with no
        # live peer, which is upstream of every recv() the ACK bound protects.
        # A caller-provided ack_timeout_s bounds the send too (same failure
        # class: nobody is reading), otherwise SOCKET_SEND_TIMEOUT_MS applies.
        # zmq.Again then surfaces as a WeightTransferAckTimeoutError from the
        # send sites below.
        if self.ack_timeout_s and self.ack_timeout_s > 0:
            snd_timeout_ms = min(int(self.ack_timeout_s * 1000), SOCKET_SEND_TIMEOUT_MS)
        else:
            snd_timeout_ms = SOCKET_SEND_TIMEOUT_MS
        self.socket.setsockopt(zmq.SNDTIMEO, snd_timeout_ms)
        self.socket.bind(self.zmq_handle)

    def _send_or_timeout(self, payload, what: str):
        """Send one frame, converting ZMQ's ``Again`` into an explicit failure.

        The sender is the process the driver is waiting on, so an un-timed send
        is the same class of bug as an un-timed recv: the job hangs with no
        diagnosis. Raising here names the peer and the stage instead.
        """
        try:
            self.socket.send_pyobj(payload)
        except zmq.Again as exc:
            raise WeightTransferAckTimeoutError(
                f"Timed out after {SOCKET_SEND_TIMEOUT_MS / 1000:.0f} s trying to send the "
                f"weight-transfer frame ({what}) — the receiver is not reading from the socket "
                "(it died, or never connected). Check the rollout worker log for the original "
                "traceback."
            ) from exc

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self._send_or_timeout(handle, "initial buffer handshake (IPC handle)")
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self._send_or_timeout(comm_metadata, "initial buffer handshake (shm metadata)")

        self._recv_ack("initial buffer handshake")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """Release the socket, buffer and shm.

        Best-effort by construction: this runs from ``finally``, so it may be
        unwinding an in-flight exception (e.g. a CUDA init failure inside
        ``_init_buffer``). Every step is therefore individually guarded — an
        accelerator call that fails *because of* the original failure must not
        replace the traceback the caller needs. Observed in a 0.20 aggregate
        log: a CUDA init error in ``_init_buffer()`` was overwritten by a second
        CUDA error raised from ``ipc_collect()`` in this method.
        """
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception as exc:  # pragma: no cover - close() rarely raises
                logger.warning("Weight transfer sender: socket close failed during cleanup: %s", exc)
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception as exc:  # pragma: no cover - already-unlinked segment
                logger.warning("Weight transfer sender: shm teardown failed during cleanup: %s", exc)
            del self.shm
            self.shm = None
        gc.collect()
        # Accelerator cleanup is the step that historically masked the primary
        # exception: if the device context never came up, ipc_collect() /
        # empty_cache() raise the *same* CUDA failure again from inside this
        # finally-block and it becomes the exception the caller sees.
        for label, fn in (
            ("ipc_collect", get_torch_device().ipc_collect),
            ("empty_cache", get_torch_device().empty_cache),
        ):
            try:
                fn()
            except Exception as exc:
                logger.warning(
                    "Weight transfer sender: accelerator %s failed during cleanup and was "
                    "suppressed to preserve the original error: %s",
                    label,
                    exc,
                )

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor):
        """Send a weight larger than the bucket size via cuda ipc or share memory."""
        logger.debug(f"Direct sending large weight {name}({weight.shape}, {weight.dtype})")
        # TODO: support fallback to shared memory
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {}
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": 0,
            "handle": handle,
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self._recv_ack(f"direct large weight {name}")


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def _report_error_to_sender(self, exc: BaseException):
        """Answer the sender's outstanding request with an error frame.

        The sender waits for an ACK after every send, so a receiver that dies
        without answering strands it. Best-effort: the socket may already be
        unusable (or the failure may have happened while no request was
        outstanding), and this runs on the error path, so it must never mask
        the original exception.
        """
        socket = self.socket
        if socket is None:
            return
        detail = f"{type(exc).__name__}: {exc}"
        logger.error("Weight transfer receiver failed, reporting to sender: %s", detail)
        try:
            # Bound how long _cleanup()'s close() may block flushing this frame:
            # the default LINGER of -1 would wait forever if the sender has also
            # gone away, turning the receiver's own teardown into a hang.
            socket.setsockopt(zmq.LINGER, 5000)
            socket.send(ACK_ERROR_PREFIX + detail.encode("utf-8", errors="replace")[:4096], flags=zmq.NOBLOCK)
        except Exception as report_exc:  # pragma: no cover - socket already broken
            logger.error("Could not report receiver failure to sender: %s", report_exc)

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        except BaseException as exc:
            # on_bucket_received (weight loading, quantization) is the likely
            # raiser; its exception must reach the sender, which is otherwise
            # blocked waiting for this bucket's ACK.
            self._report_error_to_sender(exc)
            raise
        finally:
            self._cleanup()

    def iter_weights(self):
        """Yield received weights one-by-one while preserving bucket backpressure."""
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                metadata = self.socket.recv_pyobj()
                tensor = None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        yield name, tensor
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    yield name, tensor
                get_torch_device().synchronize()
                self.socket.send(b"")
                tensor = None
                if metadata["is_last"]:
                    break
        except GeneratorExit:
            # Normal early close by the consumer — not a receiver failure, and
            # the generator is already exhausted on the success path (the loop
            # `break`s after the last bucket). Keep the pre-existing behaviour.
            raise
        except BaseException as exc:
            # The consumer of this generator (vLLM's reload_weights) can raise
            # into the yield; report it so the sender does not hang.
            self._report_error_to_sender(exc)
            raise
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        # Finite LINGER at creation: the error frame written by
        # _report_error_to_sender() (and any queued normal ACK) must not be able
        # to wedge _cleanup()'s close() when the sender is already gone. Setting
        # it here rather than only on the error path means the bound also covers
        # failures that happen before _report_error_to_sender() can run.
        self.socket.setsockopt(zmq.LINGER, SOCKET_LINGER_MS)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """Release the socket and buffer.

        Best-effort by construction: called from ``finally``, so it may be
        unwinding an in-flight exception and must never replace it (see the
        BufferError note below and the matching guard in the sender).
        """
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception as exc:  # pragma: no cover - close() rarely raises
                logger.warning("Weight transfer receiver: socket close failed during cleanup: %s", exc)
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed. Guarded for the
        # same reason as the sender's accelerator cleanup: if the device context
        # is the thing that failed, synchronize() re-raises it from inside this
        # finally-block and masks the primary error.
        try:
            get_torch_device().synchronize()
        except Exception as exc:
            logger.warning(
                "Weight transfer receiver: device synchronize failed during cleanup and was "
                "suppressed to preserve the original error: %s",
                exc,
            )
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            # Best-effort: this runs from a `finally`, so a teardown failure here
            # would REPLACE the exception on its way out. On the shared-memory
            # path that is not hypothetical — the per-bucket tensors are views
            # into `shm.buf` (an exported memoryview), and when we unwind from an
            # exception the traceback keeps the raising frame's locals alive past
            # the gc.collect() below, so close() raises
            # `BufferError: cannot close exported pointers exist` and buries the
            # real cause (a weight-loading or reload-lifecycle error) in the
            # worker log. The mapping is released by SharedMemory.__del__ once
            # the traceback is dropped, and the sender owns unlink(), so logging
            # and moving on is the correct trade.
            try:
                self.shm.close()
            except BufferError as exc:
                logger.warning(
                    "Deferring shared-memory close during weight-transfer receiver cleanup: %s "
                    "(buffer views are still referenced, most likely by the traceback of an "
                    "in-flight exception; the mapping is released when they are dropped)",
                    exc,
                )
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
