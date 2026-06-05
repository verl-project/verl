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
- TQFullyAsyncTrainer: Consumes finished samples via wait_and_sample(), reads data from TQ

Status Flow:
    (Rollouter writes status=finish) -> finish -> (Trainer consumes & removes)

Slot Control:
- acquire_slot(): Rollouter calls in _feed_samples BEFORE putting to pending_queue (blocking)
- release_slot(): Called by Rollouter after successfully writing sample to TQ (normal path),
  or on error/drop path (sample never written to TQ)
- This replaces _should_pause_generation() + MessageQueue.queue_size backpressure

Usage:
    from verl.experimental.fully_async_policy_tq.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.remote(max_pending_slots=256)
    # Rollouter side:
    acquired = await asyncio.wrap_future(rb.acquire_slot.remote(timeout=None).future())
    # ... write to TQ ...
    await rb.release_slot.remote()  # release after successful TQ write
    # Trainer side:
    sampled = await asyncio.wrap_future(rb.wait_and_sample.remote("train", batch_size=64).future())
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
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import tq

from verl.experimental.fully_async_policy.detach_utils import safe_create_task

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote(max_concurrency=100)
class ReplayBuffer:
    """Ray Actor: metadata channel + slot-based flow control for TQ fully async training.

    Replaces MessageQueue (data channel) in the original fully_async_policy.
    Key responsibilities:
    1. Slot-based backpressure: acquire_slot() blocks rollouter at dataloader source
    2. Metadata storage: tracks status of each sample (updated by caller via update_metadata)
    3. Consumer interface: wait_and_sample() for trainer to get finished samples
    4. Version tracking: reset_staleness() for parameter sync coordination
    """

    def __init__(
        self,
        max_version_slots: int,
        max_pending_slots: int = 256,
        poll_interval: float = 1.0,
    ):
        # Partition -> {key: tags_dict}
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.poll_interval = poll_interval
        self._finished = False

        # ======== Layer 1: Physical slot control (concurrency / OOM guard) ========
        # Limits simultaneous in-flight samples.
        # Acquired in _feed_samples (Rollouter), released by release_slot() after TQ write.
        # Maps to: max_concurrent_samples (e.g. TP * PP * 16)
        self.max_pending_slots = max_pending_slots
        self._pending_slots = 0  # acquired but not yet finish
        # Condition for slot flow control: acquire_slot waits, release_slot/reset_staleness/signal_finish notify
        self._slot_available = asyncio.Condition()
        # Condition for data availability: wait_and_sample waits, _poll_from_tq/signal_finish notify
        self._data_available = asyncio.Condition()

        # ======== Layer 2: Version window control (staleness guard) ========
        # Limits total samples (slots) per model version.
        # When _version_slots >= max_version_slots, acquire_slot() blocks until
        # reset_staleness() is called (after param sync).
        # _version_slots is in SLOT granularity: 1 slot = 1 sample (prompt).
        self.max_version_slots = max_version_slots
        self._version_slots = 0  # cumulative slots issued in current version

        # Initialize TQ in this actor process so _poll_from_tq can call tq.kv_list()
        try:
            import transfer_queue as tq

            tq.init()
            print("[ReplayBuffer] TQ initialized in RB actor process", flush=True)
        except Exception as e:
            print(f"[ReplayBuffer] TQ init warning: {e}", flush=True)

        print(
            f"[ReplayBuffer] initialized with "
            f"max_pending_slots={max_pending_slots}, "
            f"max_version_slots={max_version_slots}, "
            f"poll_interval={poll_interval}"
        )

        # Start background tasks immediately
        self._poll_task = safe_create_task(self._poll_from_tq(), name="poll_tq_task")
        self._monitor_task = safe_create_task(self._monitor_loop(), name="monitor_task")
        print("[ReplayBuffer] Background poll & monitor tasks started (asyncio)", flush=True)

    async def _poll_from_tq(self):
        """Background asyncio task that polls TQ for metadata updates.

        Each poll replaces self.partitions with a fresh TQ snapshot (not in-place mutation).
        This ensures self.partitions is always consistent with TQ's current state,
        avoiding stale/corrupted data from incremental merge + concurrent remove().

        UID integrity check:
            When TQ deletion errors occur, orphaned keys may remain whose meta.uid
            does not match the key's prefix. For example:
            - Normal:   key='sample_1_13247_10_0', meta.uid='sample_1_13247'  → key starts with uid ✓
            - Orphaned: key='sample_1_10885_13_0', meta.uid='sample_1_11114' → key does NOT start with uid ✗
            Such entries are detected, logged as ABNORMAL, and deleted from TQ to prevent
            corrupting the uid→response_keys mapping used by wait_and_sample().
        """
        try:
            while not self._finished:
                data = tq.kv_list()
                if data is not None:
                    # Build a fresh snapshot from TQ, then atomically replace self.partitions
                    new_partitions: dict[str, dict[str, dict]] = defaultdict(dict)
                    for partition_id, items in data.items():
                        for key, meta in items.items():
                            new_partitions[partition_id][key] = meta
                    # Update self.partitions atomically
                    async with self._data_available:
                        self.partitions = new_partitions
                        self._data_available.notify_all()
                await asyncio.sleep(self.poll_interval)
        except Exception as e:
            print(f"[ReplayBuffer] _poll_from_tq error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            os._exit(1)

    async def _monitor_loop(self):
        """Background asyncio task that periodically logs buffer statistics."""

        monitor_interval = 60.0
        while not self._finished:
            await asyncio.sleep(monitor_interval)
            if self._finished:
                break
            try:
                stats = await self.get_statistics()
                print(f"[ReplayBuffer][Monitor] {pformat(stats)}")
            except Exception as e:
                logger.error(f"[ReplayBuffer] _monitor_loop error: {e}")

    # ======== Public API ========

    async def acquire_slot(self, timeout: float | None = None, uid="") -> bool:
        """Acquire a slot before processing a dataloader sample.

        Layer 1 (Physical): ``_pending_slots < max_pending_slots``
            Limits simultaneous in-flight samples to prevent OOM / GPU overload.
            Slot is released by calling release_slot() after writing to TQ.

        Layer 2 (Version window): ``_version_slots < max_version_slots``
            Limits total slots issued per model version to control staleness.
            When the version window is full, acquire_slot() blocks until
            reset_staleness() is called after parameter synchronization.
            Only enforced if max_version_slots is set (via set_version_config()).

        Both conditions must be satisfied for a slot to be issued.
        """
        _wait_forever = timeout is None
        _deadline_abs = None if _wait_forever else (asyncio.get_event_loop().time() + timeout)

        async with self._slot_available:
            while True:
                # Check termination first
                if self._finished:
                    return False

                # Layer 1: Physical concurrency limit
                physical_ok = self._pending_slots < self.max_pending_slots

                # Layer 2: Version window limit (staleness control)
                version_ok = self.max_version_slots is None or self._version_slots < self.max_version_slots

                if physical_ok and version_ok:
                    self._pending_slots += 1
                    self._version_slots += 1
                    logger.debug(
                        f"[ReplayBuffer][acquire_slot] Acquiring slot, "
                        f"pending_slots={self._pending_slots}, "
                        f"version_slots={self._version_slots}, "
                        f"uid={uid}"
                    )
                    return True

                # Wait for notification (slot release, reset_staleness, or signal_finish)
                if not _wait_forever:
                    remaining = _deadline_abs - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        return False
                    try:
                        await asyncio.wait_for(
                            self._slot_available.wait(),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        return False
                else:
                    await self._slot_available.wait()

    async def release_slot(self):
        """Release a slot after processing.

        When all slots are released (_pending_slots == 0), record the idle start time.
        This means the rollouter has finished all in-flight work but may be blocked
        from acquiring new slots (e.g., version window full, dataloader exhausted).
        Mirrors fully_async_rollouter.py:886 idle tracking.
        """
        async with self._slot_available:
            self._pending_slots = max(0, self._pending_slots - 1)
            logger.debug(
                f"[ReplayBuffer][release_slot] Releasing slot, "
                f"pending_slots={self._pending_slots}, "
                f"version_slots={self._version_slots}"
            )
            # Record idle start when all slots are released (rollouter has no work)
            if self._pending_slots == 0:
                self.idle_start_time = time.time()
            self._slot_available.notify_all()

    async def sample(
        self,
        partition_id: str,
        sample_size: int,
        rollout_n: int,
    ) -> list[tuple[str, dict]] | None:
        """Block until enough finish samples (uids) are ready or production is fully complete.

        Args:
            partition_id: Partition to sample from (e.g. 'train').
            sample_size: Number of **samples (uids)** to wait for, not number of response keys.
                        Each sample/uid may have multiple response keys (e.g. n=16 responses per prompt).
            rollout_n: Number of response keys per uid (e.g. n=16 responses per prompt).

        Strategy:
        1. Find uid-level keys with status 'finished' (written by _run_prompt after all n responses done)
        2. Extract uids from these uid-level keys
        3. For each uid, find ALL matching response keys (e.g. 'sample_0_190_0', ..., 'sample_0_190_15')
           from the partition — integrity check: skip uids whose response keys haven't been
           fully synced by _poll_from_tq yet (race between uid "finished" and response key arrival)
        4. Return all collected response keys for up to sample_size *complete* uids
        """
        async with self._data_available:
            while True:
                # Check termination first
                if self._finished:
                    return None

                # Check if enough finish samples (uids) are ready
                part = self.partitions.get(partition_id)
                if part is not None:
                    # Step 1: Find finished uids from uid-level keys with status 'finished'
                    finished_uids: set[str] = set()
                    for key, meta in part.items():
                        status = meta.get("status", "")
                        if status == "finished":
                            finished_uids.add(key)

                    # Step 2: Check if we have enough finished uids
                    if len(finished_uids) >= sample_size:
                        # Step 3: Count response keys per uid (integrity check)
                        # Build uid -> [response_keys] mapping from ALL non-uid-level entries
                        uid_response_keys: dict[str, list[tuple[str, dict]]] = {}
                        for key, meta in part.items():
                            uid = meta.get("uid", "")
                            if uid and uid in finished_uids:
                                # This is a response key (has uid tag) belonging to a finished uid
                                uid_response_keys.setdefault(uid, []).append((key, meta))

                        normal_eq_uids = [uid for uid, keys in uid_response_keys.items() if len(keys) == rollout_n]
                        abnormal_gt_uids = [uid for uid, keys in uid_response_keys.items() if len(keys) > rollout_n]
                        abnormal_lt_uids = [uid for uid, keys in uid_response_keys.items() if len(keys) < rollout_n]

                        # For abnormal_gt_uids: check key vs meta.uid consistency,
                        # remove mismatched keys from in-memory partition (not TQ),
                        # then promote to normal_eq_uids if remaining count == rollout_n
                        if abnormal_gt_uids:
                            promoted_uids: list[str] = []
                            detail_lines: list[str] = []
                            for uid in abnormal_gt_uids:
                                keys_and_metas = uid_response_keys[uid]
                                matched_keys: list[tuple[str, dict]] = []
                                removed_keys: list[str] = []
                                for key, meta in keys_and_metas:
                                    meta_uid = meta.get("uid", "") if isinstance(meta, dict) else ""
                                    if meta_uid and key.startswith(meta_uid):
                                        matched_keys.append((key, meta))
                                    else:
                                        part.pop(key, None)
                                        removed_keys.append(key)

                                if matched_keys and len(matched_keys) == rollout_n:
                                    promoted_uids.append(uid)
                                    uid_response_keys[uid] = matched_keys
                                    detail_lines.append(
                                        f"  uid='{uid}': {len(keys_and_metas)}->{len(matched_keys)} keys, "
                                        f"removed={removed_keys}"
                                    )

                            if detail_lines:
                                print(
                                    f"[ReplayBuffer][sample] ✂️ Cleaned {len(abnormal_gt_uids)} abnormal_gt uids "
                                    f"(promoted {len(promoted_uids)} to normal_eq, "
                                    f"remaining abnormal={len(abnormal_gt_uids) - len(promoted_uids)})\n"
                                    + "\n".join(detail_lines),
                                    flush=True,
                                )

                            normal_eq_uids.extend(promoted_uids)
                            abnormal_gt_uids = [u for u in abnormal_gt_uids if u not in set(promoted_uids)]

                        if len(normal_eq_uids) >= sample_size:
                            selected_uids = normal_eq_uids[:sample_size]
                            all_response_keys: list[tuple[str, dict]] = []
                            for uid in selected_uids:
                                all_response_keys.extend(uid_response_keys[uid])

                            expected_keys = sample_size * rollout_n

                            if len(all_response_keys) != expected_keys:
                                print(
                                    f"len(all_response_keys)={len(all_response_keys)} != expected_keys={expected_keys}"
                                )
                                continue

                            print(
                                f"[ReplayBuffer][wait_and_sample][{partition_id}] Returning {len(all_response_keys)} "
                                f"response keys from {len(selected_uids)} uids "
                                f"(sample_size={sample_size}, "
                                f"total_finished={len(finished_uids)}, "
                                f"normal_eq_uids={len(normal_eq_uids)}, "
                                f"abnormal_gt_uids={len(abnormal_gt_uids)}, "
                                f"abnormal_lt_uids={len(abnormal_lt_uids)})",
                                flush=True,
                            )

                            return all_response_keys
                    else:
                        print(
                            f"[ReplayBuffer][wait_and_sample][{partition_id}] ready: "
                            f"{len(finished_uids)} uids, need={sample_size}",
                        )

                # Wait for _poll_from_tq to write new metadata or signal_finish
                await self._data_available.wait()

    async def remove(self, partition_id: str, keys: list[str]):
        """Remove consumed samples from the metadata store."""
        async with self._data_available:
            part = self.partitions.get(partition_id)
            size_before = len(part) if part is not None else 0
            if part is not None:
                for key in keys:
                    part.pop(key, None)
            size_after = len(part) if part is not None else 0

        logger.debug(
            f"[ReplayBuffer][remove] partition={partition_id}: "
            f"size {size_before} -> {size_after} "
            f"(removed {len(keys)} keys, actually_deleted={size_before - size_after})",
        )

    async def reset_staleness(self) -> dict:
        """Reset the version window after parameter synchronization.

        Mirrors FullyAsyncRollouter.reset_staleness() timing logic:
        - version_time: wall-clock time since last reset (i.e., this param sync cycle duration)
        - active_time: actual work time within the cycle (version_time minus idle periods)
        - idle_ratio: fraction of time the rollouter was idle during the cycle
        """
        async with self._slot_available:
            # Compute current partition status counts for state reset
            partition_stats = await self.compute_partition_stats()
            prev_version_slots = self._version_slots

            data = tq.kv_list()
            tq_keys = sum(len(items) for items in data.values()) if data else 0

            # Use partition_stats["train"]["success"] as the unconsumed backlog count
            # This reflects samples written to TQ (status=success) but not yet consumed by trainer
            train_stats = partition_stats.get("train", {})
            train_finished_slots = train_stats.get("finished", 0)

            # _version_slots = pending_slots (in-flight) + success_backlog (unconsumed in partitions)
            # Uses partition_stats which is the source of truth for what trainer has not yet consumed
            self._version_slots = self._pending_slots + train_finished_slots

            # Timing metrics
            # |step_start_time          |idle_start_time
            #
            # |<----- active_time ----->|<------ idle time ------>|
            # |<------------- version_time ---------------------->|
            now = time.time()
            if self.step_start_time is None:
                self.step_start_time = now
                self.idle_start_time = now

            rollout_version_time = max(now - self.step_start_time, 1e-6)
            if self.idle_start_time is not None and self.idle_start_time > self.step_start_time:
                rollout_active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1.0 - rollout_active_time / rollout_version_time
            else:
                rollout_active_time = rollout_version_time
                idle_ratio = 0.0

            timing_raw = {
                "fully_async/rollouter/active_time": rollout_active_time,
                "fully_async/rollouter/version_time": rollout_version_time,
                "fully_async/rollouter/idle_ratio": idle_ratio,
            }

            print(
                f"[ReplayBuffer][reset_staleness] "
                f"version_slots: {prev_version_slots} -> {self._version_slots} "
                f"pending_slots={self._pending_slots}, "
                f"train_finished_slots={train_finished_slots}, "
                f"tq_keys={tq_keys}), "
                f"idle_ratio: {idle_ratio:.4f}, ",
                f"partition_stats: {partition_stats}",
                flush=True,
            )

            # Reset timers for next cycle (mirrors fully_async_rollouter.py:600)
            self.step_start_time = now
            self.idle_start_time = now

            # Wake up acquire_slot waiters blocked on version window
            self._slot_available.notify_all()

        return timing_raw

    async def signal_finish(self):
        """Signal that production is fully complete — all samples are done."""
        # Wake up acquire_slot waiters (return False)
        async with self._slot_available:
            self._finished = True
            self._slot_available.notify_all()
        # Wake up wait_and_sample waiters (drain remaining)
        async with self._data_available:
            self._data_available.notify_all()

    # ======== Statistics ========
    async def compute_partition_stats(self) -> dict[str, dict[str, int]]:
        """Compute per-partition status counts. Lock-free read.

        Returns:
            dict mapping partition_id -> {status_type: count, ...}
            e.g. {"train": {"success": 64, "finished": 2}}
        """
        async with self._data_available:
            partition_stats: dict[str, dict[str, int]] = {}
            for pid, part in self.partitions.items():
                stats: dict[str, int] = {"success": 0, "finished": 0, "failure": 0, "unknown": 0}
                for v in part.values():
                    status = v.get("status", "unknown")
                    if status in stats:
                        stats[status] += 1
                partition_stats[pid] = stats
            return partition_stats

    async def get_statistics(self) -> dict:
        """Return statistics about the buffer state. Lock-free read."""
        partition_stats = await self.compute_partition_stats()

        return {
            "partitions": partition_stats,
            # Layer 1: Physical slot control
            "pending_slots": self._pending_slots,
            "max_pending_slots": self.max_pending_slots,
            "available_physical_slots": max(0, self.max_pending_slots - self._pending_slots),
            # Layer 2: Version window control
            "version_slots": self._version_slots,
            "max_version_slots": self.max_version_slots,
            "available_version_slots": max(0, (self.max_version_slots or 0) - self._version_slots),
        }
