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

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
    if mode not in ("npu_ipc", "hccl"):
        raise ValueError(f"Unsupported Ascend weight update mode: {mode!r}")
    if get_device_name() != "npu":
        raise ValueError(f"Weight update mode {mode!r} requires Ascend NPU workers")
    if peft_config:
        raise ValueError("Ascend native weight updates require full weights, not LoRA adapters")
    if adapter.config.checkpoint_engine.update_weights_bucket_megabytes <= 0:
        raise ValueError("Weight update bucket size must be positive")
    if mode == "hccl":
        import torch.distributed as dist
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerWeightTransferEngine

        for method in ("trainer_init", "send_weights", "shutdown"):
            if not callable(getattr(HCCLTrainerWeightTransferEngine, method, None)):
                raise RuntimeError(f"Installed vLLM-Ascend is missing HCCL {method}; check the pinned version")
        tp = adapter.config.tensor_model_parallel_size
        world = dist.get_world_size()
        if (
            tp <= 0
            or world != int(os.environ["RAY_LOCAL_WORLD_SIZE"])
            or world % tp
            or (world // tp < 2)
            or (adapter.config.data_parallel_size != 1)
            or (adapter.config.pipeline_model_parallel_size != 1)
            or (adapter._pd_role is not None)
        ):
            raise ValueError("Native HCCL colocated mode requires one node, at least two TP replicas, DP=PP=1")
    else:
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
        if mode != "hccl":
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
        if mode == "hccl":
            await send_ascend_hccl_weights(adapter, weights)
        else:
            method = "update_weights_from_ascend_ipc"
            future = await adapter._execute_method(
                method, non_block=True, kwargs={**kwargs, "use_shm": adapter.use_shm}
            )
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


class _HCCLBatchSource:
    """A replayable, materialized batch for the official trainer engine.

    Only this bounded batch is held. Advancing verl's async/FSDP iterator stays
    on the actor thread; the transport thread never performs FSDP gathers.
    """

    def __init__(self):
        self.batch = []

    def metadata(self):
        from vllm.distributed.weight_transfer.base import ParamMeta

        return [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self.batch]

    def __iter__(self):
        return iter(self.batch)


class _HCCLUpdateClient:
    """Adapt synchronous official RPCs to Ray and one whole-model transaction.

    Each engine send covers one complete batch source. Layerwise loading must
    span ALL batches, so forward start once and
    defer finish until the outer iterator has completed successfully.
    """

    def __init__(self, server):
        self.server = server

    def begin_round(self, loop):
        self.loop = loop
        self.started = False

    def _call(self, method, **kwargs):
        async def rpc():
            return await self.server.collective_rpc.remote(method, timeout=300, args=(), kwargs=kwargs)

        future = asyncio.run_coroutine_threadsafe(rpc(), self.loop)
        try:
            return future.result(timeout=305)
        except BaseException:
            future.cancel()
            raise

    def init_weight_transfer_engine(self, init_info):
        self._call("init_weight_transfer_engine", init_info=init_info)

    def start_weight_update(self):
        if not self.started:
            self._call("start_weight_update")
            self.started = True

    def update_weights(self, update_info):
        self._call("update_weights", update_info=update_info)

    def finish_weight_update(self, weight_version=None):
        # Called by the engine after each batch; finish_round owns the real tail.
        pass

    def finish_round(self):
        if not self.started:
            raise ValueError("Cannot finish an empty HCCL weight update")
        self._call("finish_weight_update")
        self.started = False


async def send_ascend_hccl_weights(adapter, weights):
    """Native HCCL broadcast for single-node replicated FSDP full-weight streams.

    Each replica leader sends to the next replica, so the source NPU is not
    also a receiver in its HCCL group. All other training ranks must still
    consume the stream to participate in FSDP parameter gathers.
    """
    import ray
    import torch.distributed as dist
    from vllm.utils.network_utils import get_ip, get_open_port
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerInitInfo,
        HCCLTrainerWeightTransferEngine,
    )

    from verl.workers.rollout.utils import ensure_async_iterator

    tp = adapter.config.tensor_model_parallel_size
    world = dist.get_world_size()
    local_world = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
    if (
        world != local_world
        or world % tp
        or world // tp < 2
        or adapter.config.data_parallel_size != 1
        or adapter.config.pipeline_model_parallel_size != 1
        or adapter._pd_role is not None
    ):
        raise ValueError("Native HCCL colocated mode requires one node, at least two TP replicas, DP=PP=1")
    is_sender = adapter.rollout_rank == 0
    if not is_sender:
        async for _ in ensure_async_iterator(weights):
            pass
        dist.barrier()
        return

    target = (adapter.replica_rank + 1) % (world // tp)
    server = ray.get_actor(f"{adapter._get_server_name_prefix()}server_{target}_0")

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verl-hccl-weights")
    device = get_device_id()
    state = getattr(adapter, "_ascend_hccl_state", None)
    if state is None:
        source = _HCCLBatchSource()
        client = _HCCLUpdateClient(server)
        engine = None
    else:
        engine, client, source = state
    # A cancelled await does not stop its executor thread. Retain an engine
    # created after cancellation so the queued cleanup can still close it.
    engine_holder = [engine]
    client.begin_round(loop)

    def on_device(function):
        # NPU device selection is thread-local. The official engine constructor
        # and all its communication/cleanup run with the trainer's device set.
        with get_torch_device().device(device):
            return function()

    async def run(function):
        return await loop.run_in_executor(executor, on_device, function)

    limit = adapter.config.checkpoint_engine.update_weights_bucket_megabytes * 1024 * 1024
    size = count = 0

    async def flush():
        nonlocal count
        # Finish producer copies before the transport thread uses another stream.
        get_torch_device().synchronize()
        await run(engine.send_weights)
        count += len(source.batch)
        source.batch.clear()

    try:
        if engine is None:
            info = HCCLTrainerInitInfo(
                rank=0, master_address=get_ip(), master_port=get_open_port(), world_size=tp + 1, packed=False
            )

            def initialize():
                engine_holder[0] = HCCLTrainerWeightTransferEngine.trainer_init(info, client=client, source=source)
                return engine_holder[0]

            engine = await run(initialize)
            adapter._ascend_hccl_state = (engine, client, source)

        async for name, tensor in ensure_async_iterator(weights):
            nbytes = tensor.numel() * tensor.element_size()
            if source.batch and size + nbytes > limit:
                await flush()
                size = 0
            # The exporter may reuse its storage on the next yield. Keep a
            # private contiguous copy, including for a single oversized tensor.
            source.batch.append((name, tensor.detach().to(device=engine.device, copy=True).contiguous()))
            size += nbytes
            if nbytes > limit:
                await flush()
                size = 0
        if source.batch:
            await flush()
        await run(client.finish_round)
    except BaseException:
        adapter._ascend_hccl_state = None

        def release():
            try:
                if engine_holder[0] is not None:
                    engine_holder[0].shutdown()
            except Exception:
                logger.warning("Could not close failed HCCL trainer engine", exc_info=True)
            finally:
                source.batch.clear()

        # Queue cleanup behind any operation still running after cancellation;
        # do not free its source storage or join a stuck native collective.
        executor.submit(on_device, release)
        raise
    finally:
        executor.shutdown(wait=False)
    dist.barrier()
    logger.info("Ascend native HCCL weight update completed: target_replica=%d tensors=%d", target, count)
