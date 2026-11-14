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
"""Occupancy / donate for the on-demand teacher fleet.

One replica is not stolen while its teacher still has remaining samples this
step. When that teacher finishes, the idle replica is donated to the next
unstarted key (script order), then to the longest remaining queue.

``TeacherSlotScheduler`` is the CPU plan. ``SharedTeacherLeases`` is the async
wrapper the fleet actor calls. No ray/vLLM imports, so occupancy is unit
tested on CPU.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

AcquireAction = str  # "use" | "load" | "wait"


def occupancy_spawn_keys(key_priority, key_replicas, num_replicas, large_num_replicas=None) -> list:
    """First-wave keys for replica 0..N-1: script order, one slot per key, truncated to fleet size."""
    n = int(num_replicas)
    keys = [str(key) for key in list(key_priority or []) if key is not None and str(key)]
    if n < 1 or not keys:
        return []
    quotas: dict[str, int] = {}
    if key_replicas is not None:
        replicas = [int(v) for v in list(key_replicas)]
        if len(replicas) != len(keys):
            raise ValueError(f"key_replicas length {len(replicas)} != key_priority length {len(keys)}")
        for key, count in zip(keys, replicas):
            if count < 1:
                raise ValueError(f"replica occupancy for {key!r} must be >= 1, got {count}")
            quotas[key] = count
    else:
        head = int(large_num_replicas) if large_num_replicas is not None else 1
        quotas = {keys[0]: head, **{key: 1 for key in keys[1:]}}
    out: list[str] = []
    for key in keys:
        out.extend([key] * int(quotas.get(key, 1)))
        if len(out) >= n:
            return out[:n]
    i = 0
    while len(out) < n:
        out.append(keys[i % len(keys)])
        i += 1
    return out


@dataclass
class TeacherSlotScheduler:
    """Assign teacher keys onto a fixed set of replica slots."""

    num_replicas: int
    key_priority: list = field(default_factory=list)
    key_replicas: list | dict | None = None
    current_key: list = field(init=False)
    inflight: list = field(init=False)

    def __post_init__(self) -> None:
        if self.num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {self.num_replicas}")
        self.key_priority = [str(key) for key in self.key_priority if key is not None and str(key)]
        if isinstance(self.key_replicas, dict):
            replicas_list = [int(self.key_replicas[key]) for key in self.key_priority] if self.key_priority else None
        else:
            replicas_list = self.key_replicas
        if replicas_list is not None:
            replicas_list = [int(n) for n in replicas_list]
            if len(replicas_list) != len(self.key_priority):
                raise ValueError(
                    f"key_replicas length {len(replicas_list)} != key_priority length {len(self.key_priority)}"
                )
            self._quotas = dict(zip(self.key_priority, replicas_list))
        else:
            self._quotas = {key: 1 for key in self.key_priority}
        self._priority_rank = {key: i for i, key in enumerate(self.key_priority)}
        self.current_key = [None] * self.num_replicas
        self.loading_key = [None] * self.num_replicas
        self.inflight = [0] * self.num_replicas
        self.n_leases = [0] * self.num_replicas
        self.step_outstanding: dict[str, int] = {}
        self.started_keys: set[str] = set()
        self._cross_key_swaps = [0] * self.num_replicas
        self._load_started_new = [False] * self.num_replicas
        self._fw_hold_donate = False
        self._fw_pin_released = False
        self._fw_need_keys: set[str] = set()
        self._fw_need_replicas: set[int] = set()
        self._fw_got_keys: set[str] = set()
        self._fw_got_replicas: set[int] = set()

    def priority_rank(self, key: str) -> int:
        return self._priority_rank.get(key, len(self._priority_rank))

    def quota(self, key: str) -> int:
        return int(self._quotas.get(key, 1))

    def remaining_keys(self) -> set[str]:
        return {key for key, n in self.step_outstanding.items() if n > 0}

    def begin_generate_step(self, counts: dict | None) -> None:
        """Reset leftover remaining so step N cannot deadlock step N+1 donate."""
        self.step_outstanding.clear()
        self.started_keys.clear()
        self._cross_key_swaps = [0] * self.num_replicas
        for key, n in dict(counts or {}).items():
            add = int(n)
            if add > 0:
                name = str(key)
                self.step_outstanding[name] = self.step_outstanding.get(name, 0) + add

    def note_sample_done(self, key: str) -> None:
        name = str(key)
        remaining = self.step_outstanding.get(name)
        if remaining is None:
            return
        remaining -= 1
        if remaining <= 0:
            self.step_outstanding.pop(name, None)
            if not self.occupying(name):
                self.started_keys.discard(name)
        else:
            self.step_outstanding[name] = remaining

    def mark_key_started(self, key: str) -> None:
        if key:
            self.started_keys.add(str(key))

    def has_started(self, key: str) -> bool:
        name = str(key)
        return name in self.started_keys or bool(self.occupying(name))

    def begin_load(self, replica_idx: int, key: str | None = None, mark_started: bool = False) -> None:
        newly = False
        if mark_started and key:
            newly = str(key) not in self.started_keys
            self.mark_key_started(key)
        self.loading_key[replica_idx] = key
        self.current_key[replica_idx] = None
        self.inflight[replica_idx] += 1
        self._load_started_new[replica_idx] = newly

    def complete_load(self, replica_idx: int, key: str, keep_inflight: bool) -> None:
        self.loading_key[replica_idx] = None
        self.current_key[replica_idx] = key
        if not keep_inflight and self.inflight[replica_idx] > 0:
            self.inflight[replica_idx] -= 1
        if keep_inflight:
            self.n_leases[replica_idx] += 1
            self.mark_key_started(key)

    def abort_load(self, replica_idx: int) -> None:
        key = self.loading_key[replica_idx]
        newly = self._load_started_new[replica_idx]
        self._load_started_new[replica_idx] = False
        self.loading_key[replica_idx] = None
        if self.inflight[replica_idx] > 0:
            self.inflight[replica_idx] -= 1
        self.current_key[replica_idx] = None
        if newly and key and not self.occupying(str(key)):
            self.started_keys.discard(str(key))

    def holders(self, key: str) -> list[int]:
        return [
            i
            for i, (slot_key, n) in enumerate(zip(self.current_key, self.inflight))
            if slot_key == key and n > 0
        ]

    def loading_indices(self, key: str) -> list[int]:
        return [i for i, slot_key in enumerate(self.loading_key) if slot_key == key]

    def occupying(self, key: str) -> list[int]:
        seen = set(self.holders(key))
        seen.update(self.loading_indices(key))
        return sorted(seen)

    def assignable_indices(self, key: str) -> list[int]:
        remaining = self.remaining_keys()
        out: list[int] = []
        for i in range(self.num_replicas):
            if self.inflight[i] != 0:
                continue
            tagged = self.current_key[i]
            if tagged is not None and tagged in remaining and tagged != key:
                continue
            out.append(i)
        return out

    def unstarted_keys(self) -> list[str]:
        keys = [key for key in self.remaining_keys() if not self.has_started(key)]
        return sorted(keys, key=self.priority_rank)

    def _slots_reserved_for_earlier(self, key: str) -> int:
        reserved = 0
        for other in self.unstarted_keys():
            if self.priority_rank(other) < self.priority_rank(key):
                reserved += self.quota(other)
        for other in self.remaining_keys():
            if self.priority_rank(other) < self.priority_rank(key) and self.occupying(other):
                need = self.quota(other) - len(self.occupying(other))
                if need > 0:
                    reserved += need
        return reserved

    def can_scale_out(self, key: str) -> bool:
        n = len(self.occupying(key))
        if n >= self.num_replicas:
            return False
        if self.unstarted_keys():
            if self.has_started(key) and n == 0:
                return False
            return n < self.quota(key)
        return key == self.longest_queue_key()

    def _pick_free(self, free: list[int], key: str) -> int:
        tagged = [i for i in free if self.current_key[i] == key]
        return tagged[0] if tagged else free[0]

    def try_acquire(self, key: str) -> tuple[AcquireAction, int | None]:
        """Pick a replica for ``key`` without mutating inflight."""
        holders = self.holders(key)
        loading = self.loading_indices(key)
        tagged_idle = [i for i in range(self.num_replicas) if self.inflight[i] == 0 and self.current_key[i] == key]
        if tagged_idle:
            return "load", tagged_idle[0]
        if self._fw_hold_donate:
            return "wait", None
        free = self.assignable_indices(key)
        if holders:
            if free and self.can_scale_out(key):
                return "load", self._pick_free(free, key)
            idx = min(holders, key=lambda i: (self.n_leases[i], self.inflight[i], i))
            return "use", idx
        if loading:
            if free and self.can_scale_out(key):
                return "load", self._pick_free(free, key)
            return "wait", None
        if not free or not self.can_scale_out(key):
            return "wait", None
        if len(free) <= self._slots_reserved_for_earlier(key):
            return "wait", None
        return "load", self._pick_free(free, key)

    def commit_acquire(self, replica_idx: int, key: str, new_lease: bool = False) -> None:
        self.current_key[replica_idx] = key
        self.inflight[replica_idx] += 1
        self.mark_key_started(key)
        if new_lease:
            self.n_leases[replica_idx] += 1

    def release(self, replica_idx: int, lease_complete: bool = False) -> int:
        if self.inflight[replica_idx] > 0:
            self.inflight[replica_idx] -= 1
        if lease_complete and self.n_leases[replica_idx] > 0:
            self.n_leases[replica_idx] -= 1
        return self.inflight[replica_idx]

    def should_reassign(self, replica_idx: int) -> bool:
        if self.inflight[replica_idx] != 0:
            return False
        current = self.current_key[replica_idx]
        if current is None:
            return True
        return current not in self.remaining_keys()

    def arm_first_wave(self) -> None:
        spawn = occupancy_spawn_keys(
            self.key_priority,
            [self._quotas.get(key, 1) for key in self.key_priority] or None,
            self.num_replicas,
        )
        needed_keys = {key for key in spawn if int(self.step_outstanding.get(key, 0)) > 0}
        needed_replicas = {i for i, key in enumerate(spawn) if key in needed_keys}
        self._fw_need_keys = needed_keys
        self._fw_need_replicas = needed_replicas
        self._fw_got_keys = set()
        self._fw_got_replicas = set()
        self._fw_pin_released = False
        self._fw_hold_donate = bool(needed_keys and needed_replicas)

    def note_first_wave_pin(self, replica_idx: int, key: str) -> None:
        if self._fw_pin_released or not self._fw_need_replicas:
            self._fw_hold_donate = False
            return
        name = str(key)
        if name in self._fw_need_keys and int(replica_idx) in self._fw_need_replicas:
            self._fw_got_keys.add(name)
            self._fw_got_replicas.add(int(replica_idx))
        if self._fw_need_keys <= self._fw_got_keys and self._fw_need_replicas <= self._fw_got_replicas:
            self._fw_pin_released = True
            self._fw_hold_donate = False

    def occupancy_wave(self, keys=None) -> list[tuple[int, str]]:
        if keys is None:
            key_set = set(self.key_priority)
        else:
            key_set = {str(key) for key in keys if key is not None and str(key)}
        if not key_set:
            return []
        extra = [key for key in key_set if key not in self._priority_rank]
        ordered = [key for key in self.key_priority if key in key_set] + extra
        free = [i for i in range(self.num_replicas) if self.inflight[i] == 0]
        used: set[int] = set()
        assigned: list[tuple[int, str]] = []
        for key in ordered:
            need = max(self.quota(key) - len(self.occupying(key)), 0)
            if need <= 0:
                continue
            tagged = [i for i in free if i not in used and self.current_key[i] == key]
            others = [i for i in free if i not in used and i not in tagged]
            for replica_idx in (tagged + others)[:need]:
                assigned.append((replica_idx, key))
                used.add(replica_idx)
            if len(used) >= len(free):
                break
        return assigned

    def initial_wave_targets(self) -> list[tuple[int, str]]:
        return self.occupancy_wave(self.remaining_keys())

    def longest_queue_key(self) -> str | None:
        keys: set[str] = set(self.remaining_keys())
        for slot_key, n in zip(self.current_key, self.inflight):
            if slot_key and n > 0:
                keys.add(slot_key)
        if not keys:
            return None
        return max(
            keys,
            key=lambda key: (
                int(self.step_outstanding.get(key, 0)) / max(len(self.occupying(key)), 1),
                int(self.step_outstanding.get(key, 0)),
                -self.priority_rank(key),
            ),
        )

    def next_key_for_idle_slot(self, replica_idx: int, waiting_keys: list[str]) -> str | None:
        if self.inflight[replica_idx] != 0:
            return None
        seen: set[str] = set()
        unstarted: list[str] = []
        for key in list(self.unstarted_keys()) + [str(k) for k in waiting_keys]:
            name = str(key)
            if not name or name in seen:
                continue
            seen.add(name)
            if not self.has_started(name):
                unstarted.append(name)
        unstarted.sort(key=self.priority_rank)
        pick = unstarted[0] if unstarted else self.longest_queue_key()
        if pick is None or pick == self.current_key[replica_idx]:
            return None
        return pick

    def take_idle_donations(self, waiting_keys: list[str]) -> list[tuple[int, str, str | None]]:
        planned: list[tuple[int, str, str | None]] = []
        for replica_idx in range(self.num_replicas):
            if not self.should_reassign(replica_idx):
                continue
            if self.current_key[replica_idx] is not None and self._cross_key_swaps[replica_idx] >= 1:
                continue
            next_key = self.next_key_for_idle_slot(replica_idx, waiting_keys)
            if next_key is None or next_key == self.current_key[replica_idx]:
                continue
            if self.has_started(next_key) and (
                self._cross_key_swaps[replica_idx] >= 1 or int(self.step_outstanding.get(next_key, 0)) <= 0
            ):
                continue
            from_key = self.current_key[replica_idx]
            self.begin_load(replica_idx, next_key, mark_started=True)
            planned.append((replica_idx, next_key, from_key))
        return planned

    def note_swap(self, replica_idx: int, from_key: str | None, to_key: str, seconds: float, *, count_cross_key: bool = False) -> None:
        if count_cross_key and from_key is not None and from_key != to_key:
            self._cross_key_swaps[replica_idx] += 1


class SharedTeacherLeases:
    """Async occupancy leases. ``apply_fn(slot, teacher_key)`` loads weights onto a slot."""

    def __init__(self, apply_fn, num_slots: int, key_priority: list[str], key_replicas=None):
        if num_slots <= 0:
            raise ValueError(f"shared teacher fleet needs at least one replica, got {num_slots}")
        keys = [str(key) for key in key_priority if key]
        if not keys:
            raise ValueError("shared teacher fleet needs at least one teacher key")
        self._apply = apply_fn
        self._scheduler = TeacherSlotScheduler(
            num_replicas=num_slots,
            key_priority=keys,
            key_replicas=key_replicas,
        )
        spawn = occupancy_spawn_keys(keys, key_replicas, num_slots)
        self._scheduler.current_key = list(spawn)
        self.failed: dict[int, str] = {}
        self.swaps = 0
        self.swap_seconds = 0.0
        self._waiters: list[asyncio.Future] = []

    @property
    def num_slots(self) -> int:
        return self._scheduler.num_replicas

    @property
    def keys(self) -> list[Optional[str]]:
        return self._scheduler.current_key

    @property
    def inflight(self) -> list[int]:
        return self._scheduler.inflight

    def _wake_all(self) -> None:
        waiters, self._waiters = self._waiters, []
        for future in waiters:
            if not future.done():
                future.set_result(None)

    async def set_step_key_counts(self, counts: dict) -> None:
        self._scheduler.begin_generate_step(counts)
        self._scheduler.arm_first_wave()
        to_swap: list[tuple[int, str]] = []
        for replica_idx, key in self._scheduler.initial_wave_targets():
            if int(self._scheduler.inflight[replica_idx]) > 0 or replica_idx in self.failed:
                continue
            if self._scheduler.current_key[replica_idx] == key:
                continue
            self._scheduler.begin_load(replica_idx, key)
            to_swap.append((replica_idx, key))
        logger.warning(
            "[shared_teacher_fleet] occupancy outstanding=%s wave=%s",
            dict(self._scheduler.step_outstanding),
            to_swap,
        )
        self._wake_all()
        if to_swap:
            await self._apply_loads(to_swap, keep_inflight=False, count_cross_key=False)

    async def _apply_loads(
        self,
        planned: list[tuple[int, str]],
        *,
        keep_inflight: bool,
        count_cross_key: bool,
    ) -> None:
        for replica_idx, key in planned:
            try:
                await self._run_apply(replica_idx, key, count_cross_key=count_cross_key)
            except Exception:
                self._scheduler.abort_load(replica_idx)
                self._retire_slot(replica_idx, "apply failed")
                self._wake_all()
                raise
            self._scheduler.complete_load(replica_idx, key, keep_inflight=keep_inflight)
        self._wake_all()

    async def _run_apply(self, slot: int, teacher_key: str, *, count_cross_key: bool) -> None:
        started = asyncio.get_running_loop().time()
        await self._apply(slot, teacher_key)
        elapsed = asyncio.get_running_loop().time() - started
        self.swaps += 1
        self.swap_seconds += elapsed
        self._scheduler.note_swap(slot, None, teacher_key, elapsed, count_cross_key=count_cross_key)
        logger.warning("[shared_teacher_fleet] slot %d -> %r in %.1fs", slot, teacher_key, elapsed)

    def _retire_slot(self, slot: int, reason: str) -> None:
        self.failed[slot] = reason
        if self._scheduler.inflight[slot] <= 0:
            self._scheduler.inflight[slot] = 1

    def _waiting_keys(self) -> list[str]:
        return list(self._scheduler.remaining_keys())

    async def _donate_idle(self) -> bool:
        planned = [
            (idx, key)
            for idx, key, _from in self._scheduler.take_idle_donations(self._waiting_keys())
            if idx not in self.failed
        ]
        if not planned:
            return False
        await self._apply_loads(planned, keep_inflight=False, count_cross_key=True)
        return True

    def _already_tagged_idle(self, slot: int, teacher_key: str) -> bool:
        return (
            self._scheduler.inflight[slot] == 0
            and self._scheduler.loading_key[slot] is None
            and self._scheduler.current_key[slot] == teacher_key
        )

    async def acquire(self, teacher_key: str) -> int:
        teacher_key = str(teacher_key)
        while True:
            if len(self.failed) == self.num_slots:
                raise RuntimeError(
                    f"No usable teacher replica left to serve {teacher_key!r}: {dict(self.failed)}"
                )
            action, slot = self._scheduler.try_acquire(teacher_key)
            if action == "use" and slot is not None and slot not in self.failed:
                self._scheduler.commit_acquire(slot, teacher_key, new_lease=True)
                return slot
            if action == "load" and slot is not None and slot not in self.failed:
                if self._already_tagged_idle(slot, teacher_key):
                    self._scheduler.commit_acquire(slot, teacher_key, new_lease=True)
                    self._scheduler.note_first_wave_pin(slot, teacher_key)
                    self._wake_all()
                    return slot
                self._scheduler.begin_load(slot, teacher_key, mark_started=True)
                try:
                    await self._run_apply(slot, teacher_key, count_cross_key=True)
                except Exception as exc:
                    self._scheduler.abort_load(slot)
                    self._retire_slot(slot, repr(exc))
                    logger.exception(
                        "[shared_teacher_fleet] slot %d failed to load %r; marking it unusable",
                        slot,
                        teacher_key,
                    )
                    self._wake_all()
                    raise
                self._scheduler.complete_load(slot, teacher_key, keep_inflight=True)
                self._scheduler.note_first_wave_pin(slot, teacher_key)
                self._wake_all()
                return slot
            donated = await self._donate_idle()
            if donated:
                continue
            future = asyncio.get_running_loop().create_future()
            self._waiters.append(future)
            # Enqueue first so a concurrent release cannot drop the wakeup.
            action, slot = self._scheduler.try_acquire(teacher_key)
            if action != "wait" or await self._donate_idle():
                if future in self._waiters:
                    self._waiters.remove(future)
                continue
            await future

    async def release(self, slot: int) -> bool:
        slot = int(slot)
        if not 0 <= slot < self.num_slots:
            logger.warning("[shared_teacher_fleet] release on unknown slot %s", slot)
            return False
        if self._scheduler.inflight[slot] <= 0:
            logger.warning("[shared_teacher_fleet] release on idle slot %d", slot)
            return False
        key = self._scheduler.current_key[slot]
        self._scheduler.release(slot, lease_complete=True)
        if key:
            self._scheduler.note_sample_done(str(key))
        await self._donate_idle()
        self._wake_all()
        return True

    def snapshot(self) -> dict[str, Any]:
        scheduler = self._scheduler
        return {
            "swaps": self.swaps,
            "swap_seconds": self.swap_seconds,
            "inflight": sum(scheduler.inflight),
            "waiters": len(self._waiters),
            "outstanding": dict(scheduler.step_outstanding),
            "started": sorted(scheduler.started_keys),
            "slots": [
                {
                    "slot": slot,
                    "key": scheduler.current_key[slot],
                    "inflight": scheduler.inflight[slot],
                    "loading": scheduler.loading_key[slot],
                    "failed": self.failed.get(slot),
                }
                for slot in range(self.num_slots)
            ],
        }
