# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import gc
import inspect
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import psutil
import torch

from verl.utils.device import get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

GCSetting: TypeAlias = bool | int


def validate_gc_setting(value: object, *, name: str = "gc_setting") -> None:
    """Validate a GC switch without assuming a fixed set of generations."""
    if not isinstance(value, bool) and (not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a boolean or non-negative integer, got {value!r}")


def collect_garbage(
    gc_setting: GCSetting = True,
    *,
    diagnostics_point: str | None = None,
) -> int:
    """Run a configured collection, optionally printing resource diagnostics."""
    validate_gc_setting(gc_setting)
    if gc_setting is False:
        return 0

    if diagnostics_point is not None:
        process = psutil.Process()
        device = get_torch_device()
        device_available = device.is_available()
        gc_stats_before = gc.get_stats()
        rss_before = process.memory_info().rss
        device_before = (device.memory_allocated(), device.memory_reserved()) if device_available else None
        wall_start = time.perf_counter()
        cpu_start = time.thread_time()

    collected = gc.collect() if gc_setting is True else gc.collect(gc_setting)

    if diagnostics_point is None:
        return collected

    thread_cpu_s = time.thread_time() - cpu_start
    wall_s = time.perf_counter() - wall_start
    gc_stats_after = gc.get_stats()
    rss_after = process.memory_info().rss
    device_after = (device.memory_allocated(), device.memory_reserved()) if device_available else None
    uncollectable = sum(
        after["uncollectable"] - before["uncollectable"]
        for before, after in zip(gc_stats_before, gc_stats_after, strict=False)
    )
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.getenv("RANK", "0"))
    )

    fields = [
        f"point={diagnostics_point}",
        f"rank={rank}",
        f"generation={'full' if gc_setting is True else gc_setting}",
        f"wall_ms={wall_s * 1000:.3f}",
        f"thread_cpu_ms={thread_cpu_s * 1000:.3f}",
        f"collected={collected}",
        f"uncollectable={uncollectable}",
        f"rss_before_mib={rss_before / 1024**2:.3f}",
        f"rss_after_mib={rss_after / 1024**2:.3f}",
        f"rss_delta_mib={(rss_after - rss_before) / 1024**2:.3f}",
    ]
    if device_before is not None and device_after is not None:
        for metric, before, after in zip(("allocated", "reserved"), device_before, device_after, strict=False):
            fields.extend(
                [
                    f"cuda_{metric}_before_mib={before / 1024**2:.3f}",
                    f"cuda_{metric}_after_mib={after / 1024**2:.3f}",
                    f"cuda_{metric}_delta_mib={(after - before) / 1024**2:.3f}",
                ]
            )
    for generation, (before, after) in enumerate(zip(gc_stats_before, gc_stats_after, strict=False)):
        for name in ("collections", "collected", "uncollectable"):
            fields.append(f"generation_{generation}_{name}={after[name] - before[name]}")

    print("[gc_diagnostics] " + " ".join(fields), flush=True)
    return collected


def aggressive_empty_cache(
    force_sync: bool = True,
    max_retries: int = 3,
    *,
    gc_setting: GCSetting = True,
    gc_diagnostics_point: str | None = None,
) -> None:
    """
    More aggressive GPU memory cleanup function, tries to release PyTorch reserved
    but unallocated memory.

    Args:
        force_sync: Whether to force device synchronization
        max_retries: Maximum number of retries
        gc_setting: True for a full collection, False to skip GC, or a generation integer.
        gc_diagnostics_point: Print isolated GC resource deltas under this point name.
    """
    device = get_torch_device()
    if not device.is_available():
        return

    for attempt in range(max_retries):
        # Record memory status before cleanup
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        # Run garbage collection
        collect_garbage(
            gc_setting,
            diagnostics_point=gc_diagnostics_point,
        )

        # Clear PyTorch cache
        device.empty_cache()

        # Force synchronization (optional)
        if force_sync:
            device.synchronize()

        # Record memory status after cleanup
        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()

        # Calculate freed memory
        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated

        logger.info(
            f"Memory cleanup attempt {attempt + 1}: Freed {reserved_freed / 1024**3:.2f} GB reserved, "
            f"{allocated_freed / 1024**3:.2f} GB allocated"
        )

        # Stop retrying if little memory was freed
        if reserved_freed < 1024**3:  # less than 1GB
            break


def reset_memory_stats() -> None:
    """Reset GPU memory statistics"""
    if get_torch_device().is_available():
        device = get_torch_device()
        device.reset_peak_memory_stats()
        device.reset_accumulated_memory_stats()


def get_memory_info() -> dict:
    """Get detailed GPU memory information"""
    if not get_torch_device().is_available():
        return {}

    device = get_torch_device()
    device_id = device.current_device()

    return {
        "total_memory_gb": device.get_device_properties(device_id).total_memory / 1024**3,
        "reserved_memory_gb": device.memory_reserved() / 1024**3,
        "allocated_memory_gb": device.memory_allocated() / 1024**3,
        "cached_memory_gb": (device.memory_reserved() - device.memory_allocated()) / 1024**3,
        "max_memory_allocated_gb": device.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": device.max_memory_reserved() / 1024**3,
    }


def log_memory_usage(stage: str = "current") -> None:
    """Log GPU memory usage"""
    if not get_torch_device().is_available():
        return

    info = get_memory_info()
    logger.info(
        f"Memory usage [{stage}]: "
        f"Total: {info['total_memory_gb']:.2f} GB, "
        f"Allocated: {info['allocated_memory_gb']:.2f} GB, "
        f"Reserved: {info['reserved_memory_gb']:.2f} GB, "
        f"Cached: {info['cached_memory_gb']:.2f} GB"
    )


def optimize_memory_for_inference() -> None:
    """Optimize GPU memory usage for inference"""
    if not get_torch_device().is_available():
        return

    # Set a more aggressive memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=True)

    logger.info("Optimized GPU memory usage for inference")


def optimize_memory_for_training() -> None:
    """Optimize GPU memory usage for training"""
    if not get_torch_device().is_available():
        return

    # Set a moderate memory allocation policy
    get_torch_device().set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory

    # Clear cache
    aggressive_empty_cache(force_sync=False)

    logger.info("Optimized GPU memory usage for training")


def enable_memory_visualize(
    trace_alloc_max_entries: int = 200_000,
    stack_depth: int = 32,
    context: str = "all",
    stacks: str = "all",
    devices=None,
    record_context: bool = True,
):
    """
    Enables memory history recording for accelerator (CUDA/NPU) allocations.
    This function should be called before any large-scale allocations. For DDP
    or multi-process setups, it must be called on each rank.

    Args:
        trace_alloc_max_entries (int): Maximum number of allocation entries
            to record.
        stack_depth (int): The depth of the call stack to capture for each
            allocation. (Supported by some PyTorch versions).
        context (str): The type of memory events to record.
            'alloc': records only allocation events.
            'state': records memory state changes.
            'all': records both.
        stacks (str): The type of call stacks to record.
            'python': records Python stacks.
            'cpp': records C++ stacks (available in some versions).
            'all': records both.
        devices (Union[int, list[int], None]): The device for which to enable
            memory history. `None` enables it for the current default device.
        record_context (bool): Whether to record context information for
            allocations. Required by older PyTorch versions.
    """
    # Memory history recording is accelerator-specific functionality
    device = get_torch_device()
    if not device.is_available():
        logger.warning("[memory_visualize] Memory history recording is only available on accelerator devices")
        return

    f = device.memory._record_memory_history
    params = set(inspect.signature(f).parameters.keys())

    def _one_call(dev_kw=None):
        kwargs = {}
        if "context" in params:
            kwargs["context"] = context
        if "stacks" in params:
            kwargs["stacks"] = stacks
        if "max_entries" in params:
            kwargs["max_entries"] = trace_alloc_max_entries
        elif "trace_alloc_max_entries" in params:
            kwargs["trace_alloc_max_entries"] = trace_alloc_max_entries
        if "stack_depth" in params:
            kwargs["stack_depth"] = stack_depth
        if dev_kw is not None:
            if "device" in params:
                kwargs["device"] = dev_kw
            elif "devices" in params:
                kwargs["devices"] = dev_kw if isinstance(dev_kw, list) else [dev_kw]
        if "record_context" in params:
            kwargs["record_context"] = record_context

        try:
            f(**kwargs)
            return "native", kwargs
        except TypeError:
            try:
                if "trace_alloc_max_entries" in params and "record_context" in params:
                    f(enabled=True, trace_alloc_max_entries=trace_alloc_max_entries, record_context=True)
                    return "legacy", {
                        "enabled": True,
                        "trace_alloc_max_entries": trace_alloc_max_entries,
                        "record_context": True,
                    }
                else:
                    f(enabled=True)
                    return "legacy-min", {"enabled": True}
            except Exception:
                raise

    if devices is None or isinstance(devices, str | int | torch.device):
        mode, used = _one_call(devices if devices is not None else None)
    else:
        mode, used = "multi-device", {}
        for d in list(devices):
            _mode, _used = _one_call(d)
            used[f"dev{d}"] = _used

    device = get_torch_device()
    if device.is_available():
        device.reset_peak_memory_stats()
        device.synchronize()

    rank = int(os.environ.get("RANK", "0") or 0)
    logger.info(f"[memory_visualize][rank {rank}] recording enabled ({mode}); args={used}")


def clear_memory_history(trace_alloc_max_entries: int = 200_000, stack_depth: int = 32):
    device = get_torch_device()
    if not device.is_available():
        logger.warning("[memory_visualize] Memory history recording is only available on accelerator devices")
        return
    try:
        device.memory._record_memory_history(enabled=None)
        enable_memory_visualize(trace_alloc_max_entries=trace_alloc_max_entries, stack_depth=stack_depth)
    except Exception as e:
        logger.warning(f"[memory_visualize] Failed to reset memory history: {e}")


class MemorySnapshotSampler:
    """
    A utility class that dumps GPU memory snapshots.
    This is useful for monitoring memory usage over a long-running process.

    The dumped files can be visualized with https://docs.pytorch.org/memory_viz

    Args:
        out_dir (str): The directory where the snapshots will be saved.
        tag (str): A tag for the snapshot filenames.
    """

    def __init__(self, out_dir: str = "./mem_snapshots", tag: str = "periodic"):
        self.out_dir = out_dir
        self.tag = tag

    def dump_memory_snapshot(self, out_dir: str = "./mem_snapshots", tag: str = "snapshot", sub_dir: str = None):
        """
        Generates a memory snapshot and saves it as a pickle file in a specified directory.
        The files are organized by timestamp in subdirectories, with all ranks' files
        placed in the same timestamp subdirectory.

        Args:
            out_dir (str): The directory where the snapshot file will be saved.
                The directory is created if it does not exist.
            tag (str): A string tag to prepend to the filename for easier identification.
            sub_dir (str): A subdirectory to place the snapshot file in.
        """
        if sub_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            out_path = Path(out_dir) / timestamp
        else:
            out_path = Path(out_dir) / sub_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # get the GPU rank on the current process
        rank = os.environ.get("RANK", "0")
        pid = os.getpid()
        # todo(chenyang): check wether we need to sync all ranks before dump
        fname = f"{tag}_rank{rank}_pid{pid}.pickle"
        path = out_path / fname

        device = get_torch_device()
        if not device.is_available():
            logger.warning("[memory_visualize] is only available on CUDA devices.")
            return
        try:
            device.synchronize()
            # Memory snapshot is CUDA-specific functionality
            device.memory._dump_snapshot(str(path))
            logger.info(f"[memory_visualize] dumped: {path}")
        except Exception as e:
            logger.info(f"[memory_visualize][warn] dump failed: {e}")
