# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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

"""Reclaim host memory leaked by Megatron-Core dist_checkpointing.

Root cause: ``FileSystemWriterAsync.write_buckets`` holds CPU tensor copies of
optimizer state in anonymous mmap regions.  After the async write completes,
glibc does not return these pages to the OS because the allocations were
serviced by ``mmap(2)`` (not ``sbrk``), and the C library never calls
``munmap`` on them — it keeps them in its free-list for future reuse.

The fix snapshots ``/proc/self/maps`` after the first checkpoint (iteration 0)
to record *permanent* regions (NCCL buffers, CUDA contexts, model weights,
etc.).  On every subsequent checkpoint the module identifies *new* large
anonymous regions — which are the leaked write-bucket pages — and reclaims
them with ``munmap`` + ``mmap(MAP_FIXED)``.
"""

import ctypes
import gc
import logging

logger = logging.getLogger(__name__)


def _get_anon_region_addrs(threshold_mb: float = 50) -> set:
    """Return address-range strings for large anonymous writable mmap regions.

    Each element is a string like ``"7f0000000000-7f0100000000"`` taken
    directly from ``/proc/self/maps``.
    """
    addrs: set = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                addr_range = parts[0]
                lo, hi = addr_range.split("-")
                size_mb = (int(hi, 16) - int(lo, 16)) / (1024 * 1024)
                perms = parts[1]
                is_anon = len(parts) <= 5 or parts[-1] == "[heap]"
                is_writable = "w" in perms
                if size_mb >= threshold_mb and is_anon and is_writable:
                    addrs.add(addr_range)
    except Exception:
        pass
    return addrs


def _munmap_remap_new_regions(
    permanent_addrs: set,
    threshold_mb: float = 50,
    rank: int | None = None,
) -> tuple[float, float]:
    """``munmap`` + ``mmap(MAP_FIXED)`` anonymous regions **not** in *permanent_addrs*.

    Unlike ``MADV_DONTNEED``, ``munmap`` actually frees the virtual mapping.
    ``mmap(MAP_FIXED)`` recreates it at the same address with fresh zero pages
    so that any stale pointers still land on valid (uncommitted) virtual memory
    rather than segfaulting.

    Returns ``(freed_mb, skipped_mb)``.
    """
    PROT_READ_WRITE = 0x1 | 0x2  # PROT_READ | PROT_WRITE
    MAP_PRIVATE_ANON_FIXED = 0x02 | 0x20 | 0x10  # MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED
    freed_mb = 0.0
    skipped_mb = 0.0

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]

        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 5:
                    continue  # file-backed — skip
                if len(parts) < 2:
                    continue
                addr_range = parts[0]
                lo_s, hi_s = addr_range.split("-")
                start = int(lo_s, 16)
                end = int(hi_s, 16)
                size_mb = (end - start) / (1024 * 1024)
                perms = parts[1]

                if not (size_mb >= threshold_mb and "w" in perms):
                    continue

                if addr_range in permanent_addrs:
                    skipped_mb += size_mb
                    continue

                size = end - start
                ret = libc.munmap(ctypes.c_void_p(start), ctypes.c_size_t(size))
                if ret == 0:
                    new_addr = libc.mmap(
                        ctypes.c_void_p(start),
                        ctypes.c_size_t(size),
                        PROT_READ_WRITE,
                        MAP_PRIVATE_ANON_FIXED,
                        -1,
                        0,
                    )
                    if new_addr == start:
                        freed_mb += size_mb

        print(f"[rank {rank}] munmap_remap: freed={freed_mb:.0f}MB skipped_permanent={skipped_mb:.0f}MB")
    except Exception as e:
        print(f"[rank {rank}] munmap_remap error: {e}")

    return freed_mb, skipped_mb


def _trim_memory():
    """Ask glibc to return free heap pages to the OS."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module-level state — one instance per process (each rank is a process).
# ---------------------------------------------------------------------------
_permanent_regions: set | None = None
_save_count: int = 0


def reclaim_checkpoint_memory(rank: int | None = None) -> None:
    """Reclaim leaked anonymous mmap regions after a checkpoint save.

    Call this **once** at the end of every ``save_checkpoint`` invocation,
    **after** all distributed saves and bridge HF-weight saves have finished.

    * First call (iteration 0): captures the set of permanent regions so that
      NCCL buffers, CUDA contexts, and model weights are never touched.
    * Subsequent calls: ``munmap`` + ``mmap(MAP_FIXED)`` on every new large
      anonymous region, then ``gc.collect()`` + ``malloc_trim``.
    """
    global _permanent_regions, _save_count

    print(f"[rank {rank}] reclaim_checkpoint_memory: save_count={_save_count}")

    if _save_count == 0:
        # First save — record permanent regions; do NOT reclaim.
        _permanent_regions = _get_anon_region_addrs(threshold_mb=50)
        print(f"[rank {rank}] Captured {len(_permanent_regions)} permanent anon regions (first checkpoint).")
    else:
        permanent = _permanent_regions or set()
        freed_mb, _ = _munmap_remap_new_regions(
            permanent_addrs=permanent,
            threshold_mb=50,
            rank=rank,
        )
        gc.collect()
        _trim_memory()
        print(f"[rank {rank}] Reclaimed {freed_mb:.0f}MB after checkpoint {_save_count}.")

    _save_count += 1
