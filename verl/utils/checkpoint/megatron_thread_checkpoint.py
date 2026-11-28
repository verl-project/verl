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

"""Stage native Megatron checkpoints in the training rank and write them on a CPU thread."""

from __future__ import annotations

import gc
import inspect
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from unittest.mock import patch

from verl.utils.device import get_torch_device

# MCore suspends process-wide cyclic GC during planning/staging and native writes.
# Reference-count GC suspension across threads and serialize background writers.
_WRITE_LOCK = Lock()
_GC_LOCK = Lock()
_GC_DISABLE_DEPTH = 0
_GC_PRIOR_ENABLED = False
_RESULTS_QUEUE_PATCH_LOCK = Lock()


@contextmanager
def _gc_disabled_scope():
    """Keep process-wide cyclic GC disabled until the last active scope exits."""
    global _GC_DISABLE_DEPTH, _GC_PRIOR_ENABLED
    with _GC_LOCK:
        if _GC_DISABLE_DEPTH == 0:
            _GC_PRIOR_ENABLED = gc.isenabled()
            if _GC_PRIOR_ENABLED:
                gc.disable()
        _GC_DISABLE_DEPTH += 1
    try:
        yield
    finally:
        with _GC_LOCK:
            _GC_DISABLE_DEPTH -= 1
            if _GC_DISABLE_DEPTH == 0 and _GC_PRIOR_ENABLED:
                gc.enable()


class ThreadAsyncCaller:
    """Native request staging on the actor thread, CPU writes on one I/O thread.

    The caller never runs completion callbacks. Native AsyncCallsQueue keeps
    those callbacks and their collectives on the actor RPC thread. Completion
    waits do not raise a local Future exception before ranks agree on failure.
    """

    def __init__(self, executor, *, synchronize, sync_all_done, sync_failure):
        self._executor = executor
        self._synchronize = synchronize
        self._sync_all_done = sync_all_done
        self._sync_failure = sync_failure
        self._future: Future | None = None

    def schedule_async_call(self, request):
        ready = Event()
        agreed_write = None

        def write_after_agreement():
            ready.wait()
            if agreed_write is not None:
                agreed_write()

        try:
            try:
                args = list(request.async_fn_args)
                if request.preload_fn is not None:
                    args[1] = request.preload_fn()
                self._synchronize()
                staged_write = None
                if request.async_fn is None:
                    future = Future()
                    future.set_result(None)
                else:
                    staged_write = partial(request.async_fn, *args, **request.async_fn_kwargs)
                    future = self._executor.submit(write_after_agreement)
            except BaseException as error:
                self._sync_failure(error)
                raise
            # Executor submission can fail while starting a thread. Agree on
            # staging and submission before any rank starts checkpoint I/O.
            self._sync_failure(None)
            agreed_write = staged_write
            self._future = future
        finally:
            ready.set()

    def is_current_async_call_done(self, blocking=False, no_dist=False):
        if self._future is None:
            return True
        is_alive = int(not self._future.done())
        is_done = not is_alive if no_dist else self._sync_all_done(is_alive)
        if not (is_done or blocking):
            return False
        # wait(), unlike result(), leaves exceptions for the all-rank guard.
        self.close()
        error = self._future.exception()
        if no_dist:
            if error is not None:
                raise error
        else:
            self._sync_failure(error)
        return True

    def close(self, abort=False):
        # Running native file I/O cannot be safely cancelled. Supported normal
        # shutdown drains it before the executor or checkpoint owner exits.
        if self._future is not None:
            wait([self._future])

    def __del__(self):
        # No collectives or blocking work during Python object destruction.
        pass


def _raise_if_writer_failed(error):
    import torch
    from torch.distributed.checkpoint import CheckpointException
    from torch.distributed.checkpoint.api import _wrap_exception

    device = (
        "cpu"
        if torch.distributed.is_initialized() and torch.distributed.get_backend() == "gloo"
        else get_torch_device().current_device()
    )
    failed = torch.tensor([int(error is not None)], dtype=torch.int, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        messages = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(messages, repr(error) if error is not None else None)
        failures = {rank: _wrap_exception(RuntimeError(message)) for rank, message in enumerate(messages) if message}
        raise CheckpointException("Megatron async CPU checkpoint writer", failures)


def create_thread_checkpoint_queue(*, synchronize=None, sync_all_done=None, sync_failure=None):
    """Keep native queue/finalization with an injectable CPU-only caller gate."""
    from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCaller, AsyncCallsQueue

    class NativeThreadCaller(ThreadAsyncCaller, AsyncCaller):
        def __init__(self, executor):
            super().__init__(
                executor,
                synchronize=synchronize or get_torch_device().synchronize,
                sync_all_done=sync_all_done or self.sync_all_async_calls,
                sync_failure=sync_failure or _raise_if_writer_failed,
            )

    class NativeThreadQueue(AsyncCallsQueue):
        def __init__(self):
            super().__init__(persistent=False)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="megatron-checkpoint")

        def _get_async_caller(self):
            return NativeThreadCaller(self._executor)

        def close(self, abort=False):
            try:
                if abort:
                    while self.async_calls:
                        self.async_calls.popleft().async_caller.close(abort=True)
                super().close(abort=abort)
            finally:
                self._executor.shutdown(wait=True)

    return NativeThreadQueue()


def _isolate_cpu_snapshots(request, metrics=None):
    """Clone CPU sources that MCore's D2H preload would otherwise alias.

    GPU tensors already get independent host storage from native preload.
    CPU/offloaded tensors may be backed by shared storage, for which fork's
    copy-on-write does not isolate subsequent optimizer mutations.
    """
    if request.preload_fn is None:
        return request
    source_buckets = request.async_fn_args[1]
    native_preload = request.preload_fn

    def preload():
        staged_buckets = native_preload()
        result = []
        snapshot_bytes = 0
        cpu_copy_bytes = 0
        for source_bucket, staged_bucket in zip(source_buckets, staged_buckets, strict=True):
            _, _, (_, source_tensors) = source_bucket
            path, key, (byte_data, staged_tensors) = staged_bucket
            tensors = [
                (item, tensor.detach().clone() if source.device.type == "cpu" else tensor)
                for (_, source), (item, tensor) in zip(source_tensors, staged_tensors, strict=True)
            ]
            for (_, source), (_, tensor) in zip(source_tensors, tensors, strict=True):
                size_bytes = tensor.numel() * tensor.element_size()
                snapshot_bytes += size_bytes
                if source.device.type == "cpu":
                    cpu_copy_bytes += size_bytes
            result.append((path, key, (byte_data, tensors)))
        if metrics is not None:
            for key, value in (
                ("checkpoint/worker_snapshot_bytes", snapshot_bytes),
                ("checkpoint/worker_cpu_snapshot_copy_bytes", cpu_copy_bytes),
            ):
                metrics[key] = metrics.get(key, 0.0) + float(value)
        return result

    return request._replace(preload_fn=preload)


def _write_checkpoint_contents(_rank, contents, _queue):
    # Only staged CPU tensors/BytesIO reach the native writers. Reference-count
    # GC suspension across planning/staging and serialized background writes,
    # keep native fsync and serialization, and never change process priority.
    with _gc_disabled_scope(), _WRITE_LOCK:
        for write_fn, args, kwargs in contents:
            if write_fn is not None:
                write_fn(*args, **kwargs)


def _guard_rank0_serialization_callback(callback):
    """Propagate rank-0 metadata.json / integrity write errors across all ranks before barrier exit."""
    from megatron.core.dist_checkpointing import serialization

    def guarded():
        callback_error = None
        native_save_config = serialization.save_config

        def capture_save_config(*args, **kwargs):
            nonlocal callback_error
            try:
                return native_save_config(*args, **kwargs)
            except BaseException as error:
                callback_error = error

        with patch.object(serialization, "save_config", capture_save_config):
            callback()
        _raise_if_writer_failed(callback_error)

    return guarded


def _bundle_requests(requests, finalize_save_fn, metrics):
    """Stage all trees, then write and finalize them in matching FIFO order.

    One queue entry lets MCore agree on completion once for all content trees.
    """

    def preload():
        contents = []
        for request in requests:
            request = _isolate_cpu_snapshots(request, metrics)
            args = list(request.async_fn_args)
            if request.preload_fn is not None:
                args[1] = request.preload_fn()
            contents.append((request.async_fn, args, request.async_fn_kwargs))
        return contents

    def finalize():
        # A rank-local role manifest failure must reach peers before the
        # native queue enters its final all-reduce.
        try:
            finalize_save_fn()
        except BaseException as error:
            _raise_if_writer_failed(error)
            raise
        _raise_if_writer_failed(None)

    return requests[0]._replace(
        async_fn=_write_checkpoint_contents,
        async_fn_args=(None, None, None),
        async_fn_kwargs={},
        preload_fn=preload,
        finalize_fns=[callback for request in requests for callback in request.finalize_fns] + [finalize],
    )


class ThreadedCheckpointWriter:
    """One staged checkpoint per manager, with finalization on aligned ranks."""

    def __init__(self, *, cache_plans=False, queue=None):
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

        if "async_strategy" not in inspect.signature(dist_checkpointing.save).parameters or not hasattr(
            TorchDistSaveShardedStrategy, "_get_save_and_finalize_callbacks"
        ):
            raise NotImplementedError("Threaded checkpointing requires MCore 0.19.2's built-in async save API")

        class ThreadSaveStrategy(TorchDistSaveShardedStrategy):
            def async_save(self, sharded_state_dict, checkpoint_dir, async_strategy="mcore"):
                from megatron.core.dist_checkpointing.strategies import filesystem_async

                # Install a private in-process Queue during prepare_write_data so
                # threaded saves never spawn an unused multiprocessing.Manager or
                # toggle cyclic GC outside _gc_disabled_scope().
                with _RESULTS_QUEUE_PATCH_LOCK, patch.object(filesystem_async, "get_write_results_queue", Queue):
                    return super().async_save(sharded_state_dict, checkpoint_dir, async_strategy=async_strategy)

            def _get_save_and_finalize_callbacks(self, writer, save_state_dict_ret, async_strategy):
                if writer.results_queue is not None and not isinstance(writer.results_queue, Queue):
                    writer.results_queue = Queue()
                finish_error = None
                native_finish = writer.finish

                def guarded_finish(*args, **kwargs):
                    nonlocal finish_error
                    try:
                        return native_finish(*args, **kwargs)
                    except BaseException as error:
                        finish_error = error

                writer.finish = guarded_finish
                request = super()._get_save_and_finalize_callbacks(writer, save_state_dict_ret, async_strategy)
                native_finalize_fns = list(request.finalize_fns)

                def finalize_state_dict_and_check_finish():
                    for callback in native_finalize_fns:
                        callback()
                    _raise_if_writer_failed(finish_error)

                return request._replace(finalize_fns=[finalize_state_dict_and_check_finish])

        self._strategy_cls = ThreadSaveStrategy
        self._cache_plans = cache_plans
        self._strategies = {}
        self._queue = queue if queue is not None else create_thread_checkpoint_queue()
        self._pending = False
        self._failed_error: BaseException | None = None

    @property
    def pending(self) -> bool:
        return self._pending

    @property
    def failed(self) -> bool:
        return self._failed_error is not None

    def raise_if_failed(self) -> None:
        if self._failed_error is not None:
            raise self._failed_error

    def discard_failed(self) -> None:
        """Clear a previously surfaced finalization failure on all ranks before saving to a new path."""
        if self._pending:
            raise RuntimeError("Finalize the pending threaded checkpoint before discarding a failed save")
        self._failed_error = None

    def save_dist_checkpoint(self, path, state_dict, content_metadata=None):
        """Plan a tree and give its writer a private, in-process results queue."""
        from megatron.core import dist_checkpointing, mpu
        from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

        self.raise_if_failed()
        if self._pending:
            raise RuntimeError("Finalize the pending threaded checkpoint before starting another save")
        # The manager places each saved state tree in <content>/dist_ckpt.
        slot = (Path(path).parent.name, tuple(sorted(state_dict)))
        strategy = self._strategies.get(slot)
        first_save = strategy is None
        if first_save:
            strategy = FullyParallelSaveStrategyWrapper(
                self._strategy_cls(backend="torch_dist", version=1, cached_metadata=self._cache_plans),
                mpu.get_data_parallel_group(with_context_parallel=True),
                do_cache_distribution=self._cache_plans,
            )
        with _gc_disabled_scope():
            request = dist_checkpointing.save(
                state_dict,
                path,
                sharded_strategy=strategy,
                async_sharded_save=True,
                async_strategy="mcore",
                validate_access_integrity=first_save,
                content_metadata=content_metadata,
            )
        if len(request.finalize_fns) > 1:
            request = request._replace(
                finalize_fns=[request.finalize_fns[0]]
                + [_guard_rank0_serialization_callback(cb) for cb in request.finalize_fns[1:]]
            )
        if self._cache_plans and first_save:
            self._strategies[slot] = strategy
        return request

    def schedule(self, requests, finalize_fn):
        """Stage all trees before returning control to training."""
        self.raise_if_failed()
        if self._pending:
            raise RuntimeError("Finalize the pending threaded checkpoint before starting another save")
        metrics = {}
        with _gc_disabled_scope():
            self._queue.schedule_async_request(_bundle_requests(requests, finalize_fn, metrics))
        self._pending = True
        return metrics

    def finalize(self, blocking=True):
        """Advance native finalizers on aligned ranks. Failures remain sticky until discarded."""
        self.raise_if_failed()
        if not self._pending:
            return True
        try:
            if self._queue.maybe_finalize_async_calls(blocking=blocking):
                self._pending = False
        except BaseException as error:
            while self._queue.async_calls:
                self._queue.async_calls.popleft().async_caller.close(abort=True)
            self._pending = False
            self._failed_error = error
            raise
        return not self._pending

    def close(self):
        """Drain all ranks before shutting down the executor."""
        try:
            if self._pending:
                self.finalize()
        finally:
            self._queue.close(abort=True)
