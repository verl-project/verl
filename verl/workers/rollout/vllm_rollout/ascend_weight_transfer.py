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

"""Bridge bounded full-weight streams to opt-in vLLM-Ascend APIs."""

import logging
import os
import sys

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
    BucketedWeightSender,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _check_preflight_errors(error):
    """All trainer ranks must pass before any receiver starts waiting for data."""
    import torch.distributed as dist

    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError("Ascend weight update preflight failed: " + "; ".join(failures))


def validate_ascend_sender(adapter, mode, peft_config=None):
    """Validate the selected backend without allocating buffers or starting RPCs."""
    if mode not in ("npu_ipc",):
        raise ValueError(f"Unsupported Ascend weight update mode: {mode!r}")
    if get_device_name() != "npu":
        raise ValueError(f"Weight update mode {mode!r} requires Ascend NPU workers")
    if peft_config:
        raise ValueError("Ascend native weight updates require full weights, not LoRA adapters")
    if adapter.config.checkpoint_engine.update_weights_bucket_megabytes <= 0:
        raise ValueError("Weight update bucket size must be positive")
    if adapter.use_shm:
        raise ValueError("Ascend IPC/reload requires colocated NPU IPC, not shared memory")
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

    if not callable(npu_generate_uuid):
        raise RuntimeError("Installed vLLM-Ascend does not provide NPU IPC UUID support")


async def update_ascend_weights(adapter, weights, *, mode, **kwargs):
    """Optional Ascend path; default CUDA/ROCm updates never enter this module.

    Preflight is coordinated across trainer ranks. Runtime communication failure
    invalidates this adapter: a partially updated model must not be used again.
    IPC supports an explicit abort at request boundaries; process/collective
    failures still rely on transport timeouts and the training job supervisor.
    """
    sender = None
    error = None
    try:
        if getattr(adapter, "_ascend_weight_update_failed", False):
            raise RuntimeError("A previous Ascend weight update failed; restart the rollout workers before retrying")
        validate_ascend_sender(adapter, mode, kwargs.get("peft_config"))
        sender = AscendNativeIPCWeightSender(
            zmq_handle=adapter.zmq_handle,
            bucket_size_mb=adapter.config.checkpoint_engine.update_weights_bucket_megabytes,
            use_shm=adapter.use_shm,
        )
        sender.prepare()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        _check_preflight_errors(error)
        error = None
        try:
            await adapter._execute_method(
                "validate_ascend_weight_update",
                timeout=300,
                kwargs={"mode": mode, "peft_config": kwargs.get("peft_config"), "use_shm": adapter.use_shm},
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _check_preflight_errors(error)
    except BaseException:
        if sender is not None:
            sender.close()
        raise
    future = None
    try:
        method = "update_weights_from_ascend_ipc"
        future = await adapter._execute_method(method, non_block=True, kwargs={**kwargs, "use_shm": adapter.use_shm})
        sender.receiver_started = True
        await sender.async_send_weights(weights)
        if future is not None:
            await future
    except BaseException:
        adapter._ascend_weight_update_failed = True
        if future is not None:
            import ray

            try:
                ray.cancel(future, force=False)
            except Exception:
                logger.warning("Could not cancel failed Ascend receiver RPC", exc_info=True)
        raise
    finally:
        if sender is not None:
            sender.close()


class AscendNativeIPCWeightSender(BucketedWeightSender):
    """Send tightly packed bytes to the official NPU IPC consumer.

    The consumer owns independent device copies before acknowledging a bucket,
    so layerwise loaders can retain weights while the sender reuses its buffer.
    """

    async def async_send_weights(self, weights):
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()
            offset = 0
            bucket_meta = {}
            async for name, weight in ensure_async_iterator(weights):
                flat = weight.detach().contiguous().reshape(-1).view(torch.uint8)
                size = flat.numel()
                if bucket_meta and (offset + size > self.bucket_size or name in bucket_meta):
                    self._send_bucket(bucket_meta, is_last=False)
                    bucket_meta = {}
                    offset = 0
                if size > self.bucket_size:
                    self._direct_send_large_weight(name, weight)
                    continue
                # Packed IPC describes byte lengths, not dtype-aligned offsets.
                self.buffer[offset : offset + size].copy_(flat)
                bucket_meta[name] = {"shape": weight.shape, "dtype": weight.dtype, "offset": offset}
                offset += size
            self._send_bucket(bucket_meta, is_last=True)
        finally:
            self._cleanup()

    def _init_socket(self):
        if self.socket is not None:
            return
        super()._init_socket()
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 300_000)
        self.socket.setsockopt(zmq.SNDTIMEO, 300_000)

    def _init_buffer(self):
        if self.buffer is not None:
            return
        if self.use_shm or get_device_name() != "npu":
            raise ValueError("Native Ascend IPC requires colocated NPU workers with IPC support")
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            npu_generate_uuid,
        )

        self.npu_uuid = npu_generate_uuid(get_device_id())
        self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"npu:{get_device_id()}")

    def prepare(self):
        """Allocate locally before launching a receiver, without sending data."""
        self.receiver_started = False
        self._waiting_reply = False
        self._transfer_finished = False
        self._closed = False
        try:
            self._init_socket()
            self._init_buffer()
        except BaseException:
            self.close()
            raise

    def abort(self, error):
        """Notify a receiver between requests; never violate the ZMQ REQ state."""
        if (
            self.socket is None
            or not getattr(self, "receiver_started", False)
            or getattr(self, "_waiting_reply", False)
            or getattr(self, "_transfer_finished", False)
        ):
            return
        try:
            # Failure cleanup must not obscure the original error for minutes.
            self.socket.setsockopt(zmq.SNDTIMEO, 1000)
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.socket.send_pyobj({"abort": True, "error": str(error)})
            self._waiting_reply = True
            self.socket.recv_pyobj()
            self._waiting_reply = False
            self._transfer_finished = True
        except Exception:
            logger.warning("Could not acknowledge aborted IPC update; receiver timeout remains active", exc_info=True)

    def _cleanup(self):
        error = sys.exc_info()[1]
        if error is not None:
            self.abort(error)
        self.close()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        super()._cleanup()

    def _send_bucket(self, bucket_meta, is_last):
        # Each official consumer call rebuilds storage and consumes one IPC
        # export reference. Reuse the allocation, not the ref-counter handle:
        # replaying one reduce_tensor result makes those counters unbalanced.
        # An empty terminal message has no tensor for the receiver to rebuild.
        handle = reduce_tensor(self.buffer)[1] if bucket_meta else None
        self._send_packed(
            list(bucket_meta),
            [str(meta["dtype"]).split(".")[-1] for meta in bucket_meta.values()],
            [list(meta["shape"]) for meta in bucket_meta.values()],
            [meta["dtype"].itemsize * meta["shape"].numel() for meta in bucket_meta.values()],
            handle,
            is_last,
        )

    def _direct_send_large_weight(self, name, weight):
        # Oversized tensors bypass the bucket copy and must already be local.
        if weight.device != self.buffer.device:
            raise ValueError("Oversized IPC weights must reside on the sender buffer device")
        # Keep this allocation alive until the consumer has copied it on device.
        flat = weight.detach().contiguous().reshape(-1).view(torch.uint8)
        handle = reduce_tensor(flat)[1]
        self._send_packed(
            [name], [str(weight.dtype).split(".")[-1]], [list(weight.shape)], [flat.numel()], handle, False
        )

    def _send_packed(self, names, dtype_names, shapes, tensor_sizes, handle, is_last):
        get_torch_device().synchronize()
        self.socket.send_pyobj(
            {
                "update_info": {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    "ipc_handles": {self.npu_uuid: handle} if handle is not None else {},
                    "tensor_sizes": tensor_sizes,
                },
                "is_last": is_last,
            }
        )
        self._waiting_reply = True
        reply = self.socket.recv_pyobj()
        self._waiting_reply = False
        if not reply.get("ok"):
            self._transfer_finished = True
            raise RuntimeError(f"vLLM-Ascend native IPC weight update failed: {reply.get('error')}")
        self._transfer_finished = is_last


def receive_ascend_native_weights(worker, zmq_handle):
    """Drive the native init/start/update/finish lifecycle for one weight stream."""
    socket = zmq.Context.instance().socket(zmq.REP)
    ack_pending = False
    started = False
    buckets = tensor_count = 0
    try:
        socket.setsockopt(zmq.RCVTIMEO, 300_000)
        socket.setsockopt(zmq.SNDTIMEO, 300_000)
        socket.connect(zmq_handle)
        while True:
            message = socket.recv_pyobj()
            ack_pending = True
            if message.get("abort"):
                raise RuntimeError(f"Trainer aborted weight update: {message.get('error')}")
            if not started:
                worker.init_weight_transfer_engine({"packed": True})
                worker.start_weight_update()
                started = True
            info = message["update_info"]
            if info["names"]:
                worker.update_weights(info)
                buckets += 1
                tensor_count += len(info["names"])
            if message["is_last"]:
                worker.finish_weight_update()
            # Complete all model copies before the trainer reuses its storage.
            get_torch_device().synchronize()
            socket.send_pyobj({"ok": True})
            ack_pending = False
            if message["is_last"]:
                logger.info("Ascend native IPC weight update completed: buckets=%d tensors=%d", buckets, tensor_count)
                break
    except BaseException as exc:
        if ack_pending:
            try:
                socket.send_pyobj({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except zmq.ZMQError:
                pass
        # A failed native reload invalidates this worker; propagate the failure
        # instead of finalizing a partial parameter set.
        raise
    finally:
        socket.close(linger=0)
