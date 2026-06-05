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
"""NCCL communicator suspend/resume primitives (NCCL >= 2.29.7).

Ctypes shim over ``ncclCommSuspend`` / ``ncclCommResume`` plus batch helpers.
Callers supply ``[(name, handle)]`` to ``suspend_batch`` / ``resume_batch``.
Older NCCL: no-op via ``is_supported``.

RFC: https://github.com/verl-project/verl/issues/6266
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Bitmask passed to ncclCommSuspend: release dynamic GPU memory allocations
# while preserving topology / connection state.
NCCL_SUSPEND_MEM = 0x01
NCCL_SUCCESS = 0

_nccl_lib: Optional[ctypes.CDLL] = None
_lib_load_attempted: bool = False


@dataclass
class CommStat:
    """Per-communicator suspend or resume statistics."""

    name: str
    handle: int
    duration_ms: float
    # Memory delta in MB attributed to this communicator. Only populated when
    # ``measure_per_comm=True`` is passed to the public entry point. Default 0.
    delta_mb: float = 0.0
    success: bool = True


@dataclass
class SuspendResult:
    """Aggregate result of a suspend call."""

    success: bool = False
    skipped_reason: Optional[str] = None
    freed_mb: float = 0.0
    total_ms: float = 0.0
    comms: list[CommStat] = field(default_factory=list)


@dataclass
class ResumeResult:
    """Aggregate result of a resume call."""

    success: bool = False
    skipped_reason: Optional[str] = None
    reclaimed_mb: float = 0.0
    total_ms: float = 0.0
    comms: list[CommStat] = field(default_factory=list)


def _get_nccl_lib() -> Optional[ctypes.CDLL]:
    """Lazily load ``libnccl.so.2`` and resolve ``ncclCommSuspend`` / ``ncclCommResume``.

    Returns ``None`` (cached) if the library can't be loaded or the suspend/resume
    symbols aren't present (NCCL < 2.29.7).
    """
    global _nccl_lib, _lib_load_attempted
    if _lib_load_attempted:
        return _nccl_lib
    _lib_load_attempted = True

    try:
        lib = ctypes.CDLL("libnccl.so.2")
    except OSError:
        logger.warning("Failed to load libnccl.so.2; NCCL suspend/resume disabled.")
        return None

    if not hasattr(lib, "ncclCommSuspend") or not hasattr(lib, "ncclCommResume"):
        logger.warning(
            "libnccl.so.2 does not export ncclCommSuspend/ncclCommResume; "
            "NCCL suspend/resume disabled. Requires NCCL >= 2.29.7."
        )
        return None

    lib.ncclCommSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ncclCommSuspend.restype = ctypes.c_int
    lib.ncclCommResume.argtypes = [ctypes.c_void_p]
    lib.ncclCommResume.restype = ctypes.c_int

    _nccl_lib = lib
    return lib


def is_supported() -> bool:
    """Whether the loaded NCCL library supports ncclCommSuspend/Resume (>= 2.29.7)."""
    return _get_nccl_lib() is not None


def _gpu_used_mb() -> float:
    """Driver-level GPU memory used (MB)."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def _suspend_one(handle: int) -> bool:
    lib = _get_nccl_lib()
    if lib is None or handle == 0:
        return False
    rc = lib.ncclCommSuspend(ctypes.c_void_p(handle), NCCL_SUSPEND_MEM)
    if rc != NCCL_SUCCESS:
        logger.warning("ncclCommSuspend failed: handle=0x%x rc=%d", handle, rc)
        return False
    return True


def _resume_one(handle: int) -> bool:
    lib = _get_nccl_lib()
    if lib is None or handle == 0:
        return False
    rc = lib.ncclCommResume(ctypes.c_void_p(handle))
    if rc != NCCL_SUCCESS:
        logger.warning("ncclCommResume failed: handle=0x%x rc=%d", handle, rc)
        return False
    return True


def suspend_batch(handles: list[tuple[str, int]], *, measure_per_comm: bool = False) -> SuspendResult:
    """Suspend a batch of ``(name, handle)`` NCCL communicators.

    ``measure_per_comm=True`` attributes freed memory per communicator
    (intended for tests). The reported ``freed_mb`` is a lower bound — PyTorch's
    caching allocator may absorb some of the freed memory before measurement.
    """

    if not handles:
        return SuspendResult(success=False, skipped_reason="no_warm_comms")

    total_before = _gpu_used_mb()
    total_start = time.perf_counter()
    comms_stats: list[CommStat] = []

    for name, handle in handles:
        before_mb = _gpu_used_mb() if measure_per_comm else 0.0
        t0 = time.perf_counter()
        ok = _suspend_one(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        delta_mb = 0.0
        if measure_per_comm:
            delta_mb = before_mb - _gpu_used_mb()

        comms_stats.append(CommStat(name=name, handle=handle, duration_ms=elapsed_ms, delta_mb=delta_mb, success=ok))

    total_ms = (time.perf_counter() - total_start) * 1000
    freed_mb = total_before - _gpu_used_mb()

    n_ok = sum(1 for c in comms_stats if c.success)
    logger.info("NCCL suspend: %d/%d comms in %.0f ms, freed %.0f MB", n_ok, len(comms_stats), total_ms, freed_mb)
    return SuspendResult(success=n_ok > 0, freed_mb=freed_mb, total_ms=total_ms, comms=comms_stats)


def resume_batch(handles: list[tuple[str, int]], *, measure_per_comm: bool = False) -> ResumeResult:
    """Resume a batch of ``(name, handle)`` NCCL communicators."""
    if not handles:
        return ResumeResult(success=False, skipped_reason="no_warm_comms")

    total_before = _gpu_used_mb()
    total_start = time.perf_counter()
    comms_stats: list[CommStat] = []

    for name, handle in handles:
        before_mb = _gpu_used_mb() if measure_per_comm else 0.0
        t0 = time.perf_counter()
        ok = _resume_one(handle)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        delta_mb = 0.0
        if measure_per_comm:
            torch.cuda.synchronize()
            delta_mb = _gpu_used_mb() - before_mb

        comms_stats.append(CommStat(name=name, handle=handle, duration_ms=elapsed_ms, delta_mb=delta_mb, success=ok))

    total_ms = (time.perf_counter() - total_start) * 1000
    torch.cuda.synchronize()
    reclaimed_mb = _gpu_used_mb() - total_before

    n_ok = sum(1 for c in comms_stats if c.success)
    logger.info(
        "NCCL resume: %d/%d comms in %.0f ms, reclaimed %.0f MB",
        n_ok,
        len(comms_stats),
        total_ms,
        reclaimed_mb,
    )
    return ResumeResult(success=n_ok > 0, reclaimed_mb=reclaimed_mb, total_ms=total_ms, comms=comms_stats)


def log_aggregate_summary(action: str, results, *, size_attr: str, size_verb: str) -> None:
    """Aggregate per-rank Suspend/ResumeResult into one INFO line.

    Args:
        action: "suspend" or "resume", used as the log message label.
        results: One result per rank; ``None`` entries are dropped.
        size_attr: ``"freed_mb"`` for suspend, ``"reclaimed_mb"`` for resume.
        size_verb: ``"freed"`` for suspend, ``"reclaimed"`` for resume.
    """
    valid = [r for r in (results or []) if r is not None]
    if not valid:
        return
    skipped = [r for r in valid if r.skipped_reason]
    if skipped:
        reasons = sorted({r.skipped_reason for r in skipped})
        logger.info(
            "NCCL %s skipped on %d/%d ranks: %s",
            action,
            len(skipped),
            len(valid),
            ", ".join(reasons),
        )
    actual = [r for r in valid if not r.skipped_reason]
    if not actual:
        return
    sizes = [getattr(r, size_attr) for r in actual]
    durations = [r.total_ms for r in actual]
    n_ok = sum(1 for r in actual if r.success)
    logger.info(
        "NCCL %s: %d/%d ranks succeeded, %s %.0f MB avg (range %.0f-%.0f), %.0f ms avg (range %.0f-%.0f)",
        action,
        n_ok,
        len(actual),
        size_verb,
        sum(sizes) / len(sizes),
        min(sizes),
        max(sizes),
        sum(durations) / len(durations),
        min(durations),
        max(durations),
    )
