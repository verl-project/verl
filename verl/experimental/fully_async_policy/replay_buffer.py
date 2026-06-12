# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""ReplayBuffer: Ray Actor for metadata channel + slot-based flow control in TQ fully async training.

Architecture:
- TQFullyAsyncRollouter: Calls acquire_slot() in _feed_samples, writes results to TQ with status=finish,
  then calls release_slot() to release the slot after successful TQ write
- TQFullyAsyncTrainer: Consumes finished samples via sample(), reads data from TQ

Status Flow:
    (Rollouter writes status=finish) -> finish -> (Trainer consumes & removes)

Slot Control (Dual-Layer):
    Layer 1 (Physical):  acquire_slot() / release_slot()
        Limits simultaneous in-flight samples to prevent OOM/GPU overload.
        Maps to max_concurrent_samples (e.g. TP * PP * 16).
    Layer 2 (Version):  acquire_slot() blocks / reset_staleness() unblocks
        Limits total slots per model version to control staleness.
        Maps to required_samples * trigger_parameter_sync_step.

Usage:
    from verl.experimental.fully_async_policy.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.remote(max_pending_slots=256, max_version_slots=2176)
    # Rollouter side:
    acquired = await asyncio.wrap_future(rb.acquire_slot.remote(timeout=None).future())
    # ... write to TQ ...
    await rb.release_slot.remote()  # release after successful TQ write
    # Trainer side:
    sampled = await rb.sample.remote(partition_id="train", sample_size=32)
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from pprint import pformat

import ray

try:
    import transfer_queue as tq
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.8` and try again.")
    from verl.utils.transferqueue_utils import tq

from verl.experimental.fully_async_policy.detach_utils import safe_create_task

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_DEFAULT_MONITOR_INTERVAL_S = 60.0


@ray.remote(max_concurrency=100)
class ReplayBuffer:
    """Ray Actor: metadata channel + dual-layer slot-based flow control for TQ fully async training.

    Replaces MessageQueue (data channel) in the original fully_async_policy.

    Key responsibilities:
      1. **Slot-based backpressure** – ``acquire_slot()`` blocks rollouter at dataloader source
      2. **Metadata storage** – tracks status of each sample (synced from TQ via background poll)
      3. **Consumer interface** – ``sample()`` for trainer to get finished samples
      4. **Version tracking** – ``reset_staleness()`` for parameter sync coordination
    """

    # ------------------------------------------------------------------
    # Construction & lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_version_slots: int,
        max_pending_slots: int = 256,
        poll_interval: float = 1.0,
    ) -> None:
        # --- Timing ---
        self.idle_start_time: float = time.time()
        self.step_start_time: float = time.time()

        # --- Metadata store ---
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.poll_interval = poll_interval
        self._finished: bool = False

        # --- Layer 1: Physical slot control (concurrency / OOM guard) ---
        self.max_pending_slots: int = max_pending_slots
        self._pending_slots: int = 0

        # --- Layer 2: Version window control (staleness guard) ---
        self.max_version_slots: int = max_version_slots
        self._version_slots: int = 0

        # --- Condition for slot availability ---
        self._slot_available: asyncio.Condition = asyncio.Condition()

        # --- Condition for data availability ---
        self._data_available: asyncio.Condition = asyncio.Condition()

        # --- TQ init ---
        self._init_tq()

        # --- Background tasks ---
        self._monitor_task = safe_create_task(self._monitor_loop(), name="monitor_task")

        print(
            f"[ReplayBuffer] initialized  max_pending={max_pending_slots}, max_version={max_version_slots}, ",
            flush=True,
        )
        print("[ReplayBuffer] Background monitor task started", flush=True)

    async def _monitor_loop(self) -> None:
        """Background task: periodically log buffer statistics."""
        while True:
            try:
                stats = await self.get_statistics()
                print(f"[ReplayBuffer][Monitor] {pformat(stats)}")
                await asyncio.sleep(_DEFAULT_MONITOR_INTERVAL_S)
            except Exception as exc:
                logger.error("[ReplayBuffer] _monitor_loop error: %s", exc)

    async def acquire_slot(self, timeout: float | None = None, uid: str = "") -> bool:
        """Acquire a slot before processing a dataloader sample.

        Both layer conditions must hold:

        - **Layer 1 (Physical)**: ``_pending_slots < max_pending_slots``
          Prevents OOM / GPU overload from too many in-flight samples.
        - **Layer 2 (Version)**: ``_version_slots < max_version_slots``
          Controls staleness; blocks until ``reset_staleness()`` after param sync.

        Returns ``True`` if acquired, ``False`` if timed out or ``_finished``.
        """
        wait_forever = timeout is None
        deadline = None if wait_forever else asyncio.get_event_loop().time() + timeout

        async with self._slot_available:
            while not self._finished:
                physical_ok = self._pending_slots < self.max_pending_slots
                version_ok = self._version_slots < self.max_version_slots

                if physical_ok and version_ok:
                    self._pending_slots += 1
                    self._version_slots += 1
                    logger.debug(
                        "[acquire_slot] ok  pending=%d  version=%d  uid=%s",
                        self._pending_slots,
                        self._version_slots,
                        uid,
                    )
                    return True

                # Block until slot released, staleness reset, or signal_finish
                if not wait_forever:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        return False
                    try:
                        await asyncio.wait_for(self._slot_available.wait(), timeout=remaining)
                    except TimeoutError:
                        return False
                else:
                    await self._slot_available.wait()
            return False

    async def release_slot(self) -> None:
        """Release a slot after processing (success or failure).

        When all slots are released (``_pending_slots == 0``), records
        ``idle_start_time`` so ``reset_staleness()`` can compute idle ratio.
        """
        async with self._slot_available:
            self._pending_slots = max(0, self._pending_slots - 1)
            logger.debug(
                "[release_slot]  pending=%d  version=%d",
                self._pending_slots,
                self._version_slots,
            )
            if self._pending_slots == 0:
                self.idle_start_time = time.time()
            self._slot_available.notify_all()

    async def sample(
        self,
        partition_id: str,
        sample_size: int,
    ) -> list[tuple[str, dict]] | None:
        """Block until *sample_size* complete UIDs are ready, or return ``None`` on termination.

        Args:
            partition_id: Partition to consume (e.g. ``'train'``, ``'val'``).
            sample_size:   Number of **UIDs** (not response keys) to collect.

        Returns:
            A list of ``(key, meta)`` tuples for the selected response keys, or
            ``None`` when ``_finished`` is True and no more data will arrive.
        """
        async with self._data_available:
            while True:
                # Always pull fresh TQ snapshot — we are the sole reader of TQ metadata.
                # No background poll task; this loop IS the sync point.
                self._refresh_partitions_from_tq()

                # Early exit: finished + empty → signal trainer to stop
                part = self.partitions.get(partition_id)
                result = self._try_sample_partition(part, partition_id, sample_size)

                if result is not None:
                    return result

                # Not enough data yet — check termination before waiting
                if self._finished:
                    return None

                await asyncio.sleep(self.poll_interval)

    async def reset_staleness(self) -> dict:
        """Reset the version window after parameter synchronization.

        Computes timing metrics (active_time, version_time, idle_ratio) that
        mirror ``FullyAsyncRollouter.reset_staleness()``, then wakes up any
        ``acquire_slot()`` callers blocked on the version window.
        """
        async with self._slot_available:
            partition_stats = await self.compute_partition_stats()
            prev_version_slots = self._version_slots

            data = tq.kv_list()
            tq_keys = sum(len(items) for items in data.values()) if data else 0

            train_stats = partition_stats.get("train", {})
            train_finished_slots = train_stats.get("finished", 0)

            # Recompute version slots: in-flight + unconsumed backlog
            self._version_slots = self._pending_slots + train_finished_slots

            # Timing metrics
            timing = self._compute_timing_metrics()

            print(
                f"[ReplayBuffer][reset_staleness] "
                f"version_slots: {prev_version_slots} -> {self._version_slots}  "
                f"pending={self._pending_slots}, "
                f"train_finished={train_finished_slots}, "
                f"tq_keys={tq_keys}, "
                f"idle_ratio: {timing['fully_async/rollouter/idle_ratio']:.4f}, "
                f"partition_stats: {partition_stats}",
                flush=True,
            )

            # Reset timers for next cycle
            now = time.time()
            self.step_start_time = now
            self.idle_start_time = now

            # Unblock acquire_slot waiters on version window
            self._slot_available.notify_all()

        return timing

    async def signal_finish(self) -> None:
        """Signal that production is fully complete — no more samples will arrive."""
        self._finished = True
        async with self._slot_available:
            self._slot_available.notify_all()
        async with self._data_available:
            self._data_available.notify_all()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def compute_partition_stats(self) -> dict[str, dict[str, int]]:
        """Per-partition status counts (lock-free read).

        Returns e.g. ``{"train": {"success": 64, "finished": 2}}``
        """
        async with self._data_available:
            self._refresh_partitions_from_tq()
            partition_stats: dict[str, dict[str, int]] = {}
            for pid, part in self.partitions.items():
                stats: dict[str, int] = {
                    "success": 0,
                    "finished": 0,
                    "failure": 0,
                    "unknown": 0,
                }
                for v in part.values():
                    status = v.get("status", "unknown")
                    if status in stats:
                        stats[status] += 1
                partition_stats[pid] = stats
            return partition_stats

    async def get_statistics(self) -> dict:
        """Full buffer-state snapshot (lock-free read)."""
        partition_stats = await self.compute_partition_stats()
        return {
            "partitions": partition_stats,
            # Layer 1
            "pending_slots": self._pending_slots,
            "max_pending_slots": self.max_pending_slots,
            "available_physical_slots": max(0, self.max_pending_slots - self._pending_slots),
            # Layer 2
            "version_slots": self._version_slots,
            "max_version_slots": self.max_version_slots,
            "available_version_slots": max(0, self.max_version_slots - self._version_slots),
        }

    # ==================================================================
    # Private helpers  (pure logic, no I/O / locks)
    # ==================================================================

    # -- TQ snapshot -------------------------------------------------------

    def _init_tq(self) -> None:
        """Initialize TransferQueue in this actor process."""
        try:
            tq.init()
            print("[ReplayBuffer] TQ initialized in RB actor process", flush=True)
        except Exception as exc:
            print(f"[ReplayBuffer] TQ init warning: {exc}", flush=True)

    def _refresh_partitions_from_tq(self):
        """Atomically replace ``self.partitions`` with a fresh TQ snapshot."""
        data = tq.kv_list()
        if data is None:
            self.partitions = None
        else:
            new_partitions: dict[str, dict[str, dict]] = defaultdict(dict)
            for pid, items in data.items():
                for key, meta in items.items():
                    new_partitions[pid][key] = meta
            self.partitions = new_partitions

    # -- Sample logic -----------------------------------------------------

    def _try_sample_partition(
        self,
        part: dict[str, dict],
        partition_id: str,
        sample_size: int,
    ) -> list[tuple[str, dict]] | None:
        """Try to extract *sample_size* complete UIDs from *part*.

        Returns the list of ``(key, meta)`` tuples on success, or ``None`` if
        there aren't enough complete UIDs yet (caller should wait/retry).
        """
        if not part:
            return None

        finished_uids = list({key for key, meta in part.items() if meta.get("status") == "finished"})

        if len(finished_uids) < sample_size:
            print(
                f"[ReplayBuffer][sample][{partition_id}] ready: {len(finished_uids)} uids, need={sample_size}",
            )
            return None

        uid_response_keys = self._build_uid_response_key_map(part, finished_uids)

        selected_uids = finished_uids[:sample_size]
        all_response_keys: list[tuple[str, dict]] = []
        for uid in selected_uids:
            all_response_keys.extend(uid_response_keys[uid])

        print(
            f"[ReplayBuffer][sample][{partition_id}] Returning {len(all_response_keys)} keys "
            f"from {len(selected_uids)} uids "
            f"(sample_size={sample_size}, total_finished={len(finished_uids)}, ",
            flush=True,
        )

        return all_response_keys

    @staticmethod
    def _build_uid_response_key_map(
        part: dict[str, dict],
        finished_uids: list[str],
    ) -> dict[str, list[tuple[str, dict]]]:
        """Map each finished UID → its response-key entries from *part*."""
        mapping: dict[str, list[tuple[str, dict]]] = {}
        for key, meta in part.items():
            uid = meta.get("uid", "")
            if uid and uid in finished_uids:
                mapping.setdefault(uid, []).append((key, meta))
        return mapping

    # -- Timing -------------------------------------------------------------
    def _compute_timing_metrics(self) -> dict[str, float]:
        """Compute rollouter timing metrics for the just-completed version window.

        |step_start_time          |idle_start_time          |
        |<----- active_time ----->|<------ idle time ------>|
        |<------------- version_time ---------------------->|
        """

        now = time.time()
        if self.step_start_time is None:
            self.step_start_time = now
            self.idle_start_time = now

        version_time = max(now - self.step_start_time, 1e-6)
        if self.idle_start_time > self.step_start_time:
            active_time = self.idle_start_time - self.step_start_time
            idle_ratio = 1.0 - active_time / version_time
        elif self.idle_start_time == self.step_start_time:
            active_time = 0
            idle_ratio = 1.0
        else:
            active_time = version_time
            idle_ratio = 0.0

        return {
            "fully_async/rollouter/active_time": active_time,
            "fully_async/rollouter/version_time": version_time,
            "fully_async/rollouter/idle_ratio": idle_ratio,
        }


def tq_kv_clear(batch):
    """Cleanup consumed batch from TQ and RB.

    Two-phase cleanup:
    1. Clear uid-level keys (deduplicated by uid from tags): removes uid status entries
       like {'uid': {'status': 'finished'}} from both TQ and RB partitions.
    2. Clear all sampled response keys: full cleanup of {uid}_{resp_idx} keys from TQ and RB.
    """
    uid_keys: set[str] = set()
    for key, tag in zip(batch.keys, batch.tags, strict=False):
        uid = tag.get("uid", "") if isinstance(tag, dict) else ""
        if uid and uid not in uid_keys:
            uid_keys.add(uid)

    tq.kv_clear(keys=list(uid_keys), partition_id=batch.partition_id)
    tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
