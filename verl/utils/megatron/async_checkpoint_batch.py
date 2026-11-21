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
"""Combine compatible MCore async writes without changing their disk layout.

The MCore persistent writer waits for each request's preload before returning
from scheduling. Consecutive requests therefore wait for the preceding write
before they can preload. A combined request stages all eligible states first,
then writes and finalizes the original requests in order.
"""

from functools import partial

import torch


def _preload_batch(preloaders, cpu_masks=None):
    staged = [preload() if preload is not None else None for preload in preloaders]
    if cpu_masks is None:
        return staged
    frozen = []
    for buckets, masks in zip(staged, cpu_masks, strict=True):
        if buckets is None:
            frozen.append(None)
            continue
        frozen_buckets = []
        for (name, key, (byte_items, tensor_items)), tensor_mask in zip(buckets, masks, strict=True):
            # The trainer is still waiting for preload completion here. Clone
            # only tensors that originated on CPU; GPU tensors were copied by
            # MCore's preload and do not need a second host copy.
            frozen_tensors = [
                (item, tensor.detach().clone() if copy else tensor)
                for (item, tensor), copy in zip(tensor_items, tensor_mask, strict=True)
            ]
            frozen_buckets.append((name, key, (byte_items, frozen_tensors)))
        frozen.append(frozen_buckets)
    return frozen


def _native_mcore_functions():
    """Return the writer and preloader whose CPU snapshot layout is known."""
    try:
        from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

        return FileSystemWriterAsync.write_preloaded_data_multithread, FileSystemWriterAsync.preload_tensors
    except (ImportError, AttributeError):
        return None, None


def _cpu_snapshot_mask(request):
    """Describe live CPU tensors only for a recognized native MCore request."""
    if request.async_fn is None:
        return ()
    writer, preload = _native_mcore_functions()
    async_fn = request.async_fn
    if (
        writer is None
        or preload is None
        or not isinstance(async_fn, partial)
        or async_fn.func is not writer
        or not isinstance(request.preload_fn, partial)
        or request.preload_fn.func is not preload
        or not request.preload_fn.args
        or request.preload_fn.args[0] is not request.async_fn_args[1]
    ):
        return None
    masks = []
    for bucket in request.async_fn_args[1]:
        tensor_mask = []
        for _, tensor in bucket[2][1]:
            is_cpu = tensor.device.type == "cpu"
            # Unknown tensor subclasses or layouts require their own snapshot
            # contract. Keep the existing separate request path for those.
            if is_cpu and (type(tensor) is not torch.Tensor or tensor.layout != torch.strided):
                return None
            tensor_mask.append(is_cpu)
        masks.append(tuple(tensor_mask))
    return tuple(masks)


def _write_batch(writes, staged, _unused):
    if len(writes) != len(staged):
        raise RuntimeError("Checkpoint batch staging count does not match the write count")
    for (write, original_args, kwargs), data in zip(writes, staged, strict=True):
        if write is None:
            continue
        args = list(original_args)
        args[1] = data
        write(*args, **kwargs)


def _estimate_staging_bytes(request):
    """Conservatively count known tensor and serialized-byte payload sizes.

    This is an estimate of the staged payload, not a bound on process RSS or
    temporary serializer allocations. Unknown MCore request layouts are not
    eligible for coalescing.
    """
    if request.async_fn is None:
        return 0
    if not callable(getattr(request, "preload_fn", None)):
        return None
    args = getattr(request, "async_fn_args", None)
    if not isinstance(args, tuple | list) or len(args) != 3 or not isinstance(args[1], tuple | list):
        return None
    total = 0
    try:
        for bucket in args[1]:
            if not isinstance(bucket, tuple | list) or len(bucket) != 3:
                return None
            byte_items, tensor_items = bucket[2]
            for _, value in byte_items:
                if not hasattr(value, "getbuffer"):
                    return None
                total += value.getbuffer().nbytes
            for _, tensor in tensor_items:
                if not isinstance(tensor, torch.Tensor):
                    return None
                total += tensor.numel() * tensor.element_size()
    except (TypeError, ValueError, AttributeError, RuntimeError):
        return None
    return total


def coalesce_async_requests(requests, max_staging_bytes, agree, *, snapshot_cpu_tensors=False):
    """Batch requests only when every rank agrees that the local payload fits.

    ``agree`` performs an all-rank boolean AND. The on-disk directories and
    original finalize callbacks stay unchanged. With an unsupported request or
    over-budget rank, every rank retains the original scheduling path.
    """
    if not isinstance(max_staging_bytes, int) or isinstance(max_staging_bytes, bool) or max_staging_bytes < 0:
        raise ValueError("max_staging_bytes must be a non-negative integer")
    if type(snapshot_cpu_tensors) is not bool:
        raise ValueError("snapshot_cpu_tensors must be a boolean")
    if max_staging_bytes == 0:
        return requests, None
    fields = {"async_fn", "async_fn_args", "finalize_fns", "async_fn_kwargs", "preload_fn"}
    estimates = []
    cpu_masks = []
    local_ok = len(requests) > 1
    for request in requests:
        if (
            type(request) is not type(requests[0])
            or not hasattr(request, "_replace")
            or not fields.issubset(getattr(request, "_fields", ()))
            or getattr(request, "is_frozen", False)
            or not isinstance(request.finalize_fns, list)
            or not isinstance(request.async_fn_kwargs, dict)
        ):
            local_ok = False
            break
        size = _estimate_staging_bytes(request)
        if size is None:
            local_ok = False
            break
        estimates.append(size)
        if snapshot_cpu_tensors:
            mask = _cpu_snapshot_mask(request)
            if mask is None:
                local_ok = False
                break
            cpu_masks.append(mask)
    estimated_bytes = sum(estimates)
    local_ok = local_ok and estimated_bytes <= max_staging_bytes
    if not agree(local_ok):
        return requests, None
    if not local_ok:
        raise RuntimeError("Checkpoint batch agreement returned true for an ineligible rank")

    writes = [(req.async_fn, req.async_fn_args, req.async_fn_kwargs) for req in requests]
    preloaders = [req.preload_fn for req in requests]
    finalizers = [fn for req in requests for fn in req.finalize_fns]
    combined = requests[0]._replace(
        async_fn=_write_batch,
        async_fn_args=(writes, None, None),
        async_fn_kwargs={},
        preload_fn=partial(_preload_batch, preloaders, cpu_masks if snapshot_cpu_tensors else None),
        finalize_fns=finalizers,
    )
    return [combined], estimated_bytes
