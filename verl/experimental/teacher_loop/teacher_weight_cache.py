# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Memory-mapped weight cache shared by the on-demand multi-teacher fleet.

One TP shard set per teacher is written to a local directory and afterwards
mapped read-only by every replica worker. ``disk`` and ``shm`` backends reuse
the exact same mmap code path; only the base directory differs:

* ``disk`` -> ``<teacher_cache_dir>/<job_id>/<teacher_key>/tp<rank>/``
* ``shm``  -> ``/dev/shm/verl_teacher_cache/<job_id>/<teacher_key>/tp<rank>/``

The snapshot is taken *after* ``process_weights_after_loading`` so the cached
tensors already match the inference kernel layout; swapping teachers is then a
plain device copy without any layout transform.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_ALIGN = 64
_WEIGHTS_BIN = "weights.bin"
_META_JSON = "meta.json"
_SHM_ROOT = "/dev/shm/verl_teacher_cache"
_DISK_ROOT = "/tmp"
_PROTECTED_DIRS = {os.sep, "/tmp", "/dev/shm", _SHM_ROOT}

_CACHE_DIRS_TO_CLEAN: set[str] = set()
_ATEXIT_REGISTERED = False


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
def resolve_cache_root(backend: str, cache_dir: str | None = None) -> str:
    """Return the backend-specific base directory for teacher caches."""
    if backend == "disk":
        return os.path.abspath(cache_dir or os.environ.get("VERL_TEACHER_CACHE_DIR", _DISK_ROOT))
    if backend == "shm":
        if cache_dir:
            logger.warning("teacher_cache_dir=%r is ignored for the shm backend", cache_dir)
        return _SHM_ROOT
    raise ValueError(f"Unknown teacher_cache_backend {backend!r}; expected 'disk' or 'shm'.")


def resolve_job_id(job_id: str | None = None) -> str:
    return (
        job_id
        or os.environ.get("VERL_RAY_JOB_ID")
        or os.environ.get("RAY_JOB_ID")
        or os.environ.get("ARNOLD_TRIAL_ID")
        or "0"
    )


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def make_job_cache_dir(root: str, job_id: str | None = None) -> str:
    """``<root>/<job_id>``; per-job so cleanup never touches another job."""
    return os.path.join(os.path.abspath(root), _safe_name(resolve_job_id(job_id)))


def make_teacher_cache_dir(job_dir: str, teacher_key: str) -> str:
    return os.path.join(job_dir, _safe_name(teacher_key))


def tp_shard_dir(teacher_dir: str, tp_rank: int) -> str:
    return os.path.join(teacher_dir, f"tp{int(tp_rank)}")


def get_tp_rank(local_rank: int, parallel_config: Any | None) -> int:
    """Map a vLLM worker ``local_rank`` to its tensor-parallel rank."""
    tp_size = 1
    if parallel_config is not None:
        tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)
    return int(local_rank) % tp_size


# ---------------------------------------------------------------------------
# Capacity check / cleanup
# ---------------------------------------------------------------------------
def prepare_cache_dir(cache_dir: str, min_free_bytes: int = 0) -> dict[str, int | str]:
    """Create ``cache_dir`` and fail early when the backend has too little room."""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    usage = shutil.disk_usage(cache_dir)
    if min_free_bytes and usage.free < min_free_bytes:
        need_gib = min_free_bytes / (1 << 30)
        free_gib = usage.free / (1 << 30)
        raise RuntimeError(
            f"Shared teacher weight cache at {cache_dir} needs ~{need_gib:.1f} GiB free "
            f"but only {free_gib:.1f} GiB is available. Point distillation.teacher_cache_dir "
            f"at a larger local disk, or switch teacher_cache_backend=shm only when /dev/shm "
            f"is large enough."
        )
    logger.info(
        "[teacher_weight_cache] prepared %s free=%.1f GiB total=%.1f GiB",
        cache_dir,
        usage.free / (1 << 30),
        usage.total / (1 << 30),
    )
    return {"path": cache_dir, "free_bytes": int(usage.free), "total_bytes": int(usage.total)}


def cleanup_cache_dir(cache_dir: str | None) -> bool:
    """Delete a per-job cache directory. Safe to call more than once."""
    if not cache_dir:
        return False
    cache_dir = os.path.abspath(cache_dir)
    # Only ever delete a leaf job directory below a shared root such as /tmp or
    # /dev/shm, never the root itself.
    if len(cache_dir.strip(os.sep).split(os.sep)) < 2 or cache_dir in _PROTECTED_DIRS:
        logger.warning("[teacher_weight_cache] refuse to delete unsafe path %s", cache_dir)
        return False
    _CACHE_DIRS_TO_CLEAN.discard(cache_dir)
    if not os.path.isdir(cache_dir):
        return False
    shutil.rmtree(cache_dir, ignore_errors=True)
    deleted = not os.path.isdir(cache_dir)
    logger.info("[teacher_weight_cache] cleanup %s deleted=%s", cache_dir, deleted)
    return deleted


def _atexit_cleanup() -> None:
    for cache_dir in list(_CACHE_DIRS_TO_CLEAN):
        cleanup_cache_dir(cache_dir)


def register_cleanup_on_exit(cache_dir: str | None) -> None:
    """Delete ``cache_dir`` when this process exits (used for ``shm``)."""
    global _ATEXIT_REGISTERED
    if not cache_dir:
        return
    _CACHE_DIRS_TO_CLEAN.add(os.path.abspath(cache_dir))
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_cleanup)
        _ATEXIT_REGISTERED = True


# ---------------------------------------------------------------------------
# Snapshot / mmap load / device apply
# ---------------------------------------------------------------------------
def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _str_to_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype {name!r}")
    return dtype


def _uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    view.set_(tensor.untyped_storage())
    start = tensor.storage_offset() * tensor.element_size()
    return view[start : start + tensor.numel() * tensor.element_size()]


def _tensor_view_from_storage(
    storage: torch.UntypedStorage,
    *,
    offset: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    elem_size = torch.empty((), dtype=dtype).element_size()
    nbytes = elem_size
    for dim in shape:
        nbytes *= int(dim)
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype)
    if offset % elem_size != 0:
        raise ValueError(f"mmap offset {offset} is not aligned to {dtype} ({elem_size} bytes)")
    wrapped = torch.empty(0, dtype=torch.uint8)
    wrapped.set_(storage)
    return wrapped[offset : offset + nbytes].view(dtype).reshape(shape)


def _iter_named_tensors(models: Iterable[nn.Module]):
    """Yield ``(kind, prefixed_name, tensor)`` for every model in the fleet worker."""
    for model_idx, model in enumerate(models):
        prefix = f"__model{model_idx}__" if model_idx > 0 else ""
        for name, tensor in model.named_parameters():
            yield "params", f"{prefix}{name}", tensor
        for name, tensor in model.named_buffers():
            yield "buffers", f"{prefix}{name}", tensor


def estimate_model_bytes(models: Iterable[nn.Module]) -> int:
    """Bytes needed to cache this worker's TP shard for one teacher."""
    total = 0
    for _, _, tensor in _iter_named_tensors(models):
        total += tensor.numel() * tensor.element_size()
    return total


@torch.no_grad()
def write_shard_from_models(
    models: Iterable[nn.Module],
    shard_dir: str,
    *,
    min_free_bytes: int = 0,
    model_path: str | None = None,
) -> int:
    """Write one TP shard as ``weights.bin`` + ``meta.json``.

    ``model_path`` is recorded so a later map can reject a stale shard left by
    another checkpoint; it is the only guard against snapshotting the dummy boot
    weights of a different teacher.
    """
    os.makedirs(shard_dir, exist_ok=True)

    entries: list[dict[str, Any]] = []
    sources: list[torch.Tensor] = []
    offset = 0
    for kind, name, tensor in _iter_named_tensors(models):
        src = tensor.detach()
        if not src.is_contiguous():
            src = src.contiguous()
        nbytes = src.numel() * src.element_size()
        aligned = ((nbytes + _ALIGN - 1) // _ALIGN) * _ALIGN if nbytes else 0
        entries.append(
            {
                "kind": kind,
                "name": name,
                "shape": list(src.shape),
                "dtype": _dtype_to_str(src.dtype),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        sources.append(src)
        offset += aligned

    usage = shutil.disk_usage(shard_dir)
    if min_free_bytes and usage.free < min_free_bytes:
        raise RuntimeError(
            f"teacher weight cache at {shard_dir} needs ~{min_free_bytes / (1 << 30):.1f} GiB "
            f"but only {usage.free / (1 << 30):.1f} GiB is free"
        )

    bin_path = os.path.join(shard_dir, _WEIGHTS_BIN)
    meta_path = os.path.join(shard_dir, _META_JSON)
    with open(bin_path, "wb") as handle:
        handle.truncate(offset)
    if offset > 0:
        storage = torch.UntypedStorage.from_file(bin_path, shared=True, nbytes=offset)
        mapped = torch.empty(0, dtype=torch.uint8)
        mapped.set_(storage)
        for src, entry in zip(sources, entries, strict=True):
            nbytes = int(entry["nbytes"])
            if nbytes:
                mapped[entry["offset"] : entry["offset"] + nbytes].copy_(_uint8_view(src))
        del mapped, storage
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": 1,
                "nbytes": offset,
                "model_path": os.path.abspath(model_path) if model_path else None,
                "entries": entries,
            },
            handle,
        )

    logger.info(
        "[teacher_weight_cache] wrote shard %s (%.2f GiB, %d tensors)",
        os.path.abspath(shard_dir),
        offset / (1 << 30),
        len(entries),
    )
    return offset


def load_shard_mmap(shard_dir: str, *, model_path: str | None = None) -> dict[str, Any]:
    """Map a TP shard without copying tensors into a private CPU buffer.

    Raises when ``model_path`` is given and the shard was built for another
    checkpoint, so a stale or partially rebuilt cache fails loudly instead of
    serving the wrong teacher.
    """
    with open(os.path.join(shard_dir, _META_JSON), encoding="utf-8") as handle:
        meta = json.load(handle)
    if model_path is not None:
        cached_path = meta.get("model_path")
        expected_path = os.path.abspath(model_path)
        if cached_path != expected_path:
            raise RuntimeError(
                f"stale teacher weight cache at {shard_dir}: built for {cached_path!r}, expected {expected_path!r}"
            )
    total = int(meta["nbytes"])
    storage = (
        torch.UntypedStorage.from_file(os.path.join(shard_dir, _WEIGHTS_BIN), shared=True, nbytes=total)
        if total > 0
        else None
    )
    params: dict[str, torch.Tensor] = {}
    buffers: dict[str, torch.Tensor] = {}
    for entry in meta["entries"]:
        dtype = _str_to_dtype(entry["dtype"])
        shape = tuple(entry["shape"])
        if int(entry["nbytes"]) == 0 or storage is None:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            tensor = _tensor_view_from_storage(storage, offset=int(entry["offset"]), shape=shape, dtype=dtype)
        (params if entry["kind"] == "params" else buffers)[entry["name"]] = tensor
    # Keep the storage alive for as long as the views exist.
    return {"params": params, "buffers": buffers, "_storage": storage}


def _last2_dims_are_transpose(src_shape: tuple[int, ...], dst_shape: tuple[int, ...]) -> bool:
    if src_shape == dst_shape or len(src_shape) < 2 or len(src_shape) != len(dst_shape):
        return False
    return src_shape[:-2] == dst_shape[:-2] and src_shape[-2:] == dst_shape[-2:][::-1]


@torch.no_grad()
def _copy_into(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if tuple(source.shape) != tuple(target.shape):
        if _last2_dims_are_transpose(tuple(source.shape), tuple(target.shape)):
            source = source.transpose(-2, -1).contiguous()
        else:
            raise RuntimeError(
                f"teacher cache shape mismatch for {name!r}: cache={tuple(source.shape)} target={tuple(target.shape)}"
            )
    target.copy_(source.to(device=target.device, dtype=target.dtype, non_blocking=False))


@torch.no_grad()
def apply_shard_to_models(
    models: Iterable[nn.Module],
    shard: dict[str, Any],
) -> tuple[int, int]:
    """Copy a mapped shard into the live models in place."""
    cached = {"params": shard["params"], "buffers": shard["buffers"]}
    counts = {"params": 0, "buffers": 0}
    for kind, name, target in _iter_named_tensors(models):
        source = cached[kind].get(name)
        if source is None:
            continue
        _copy_into(target, source, name)
        counts[kind] += 1
    return counts["params"], counts["buffers"]
