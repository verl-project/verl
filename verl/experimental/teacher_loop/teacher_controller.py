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
"""On-demand teacher wake/sleep for multi-teacher shared GPU group placement.

When ``distillation.share_gpu_group`` is enabled, all teachers' replicas are placed on
the same GPU bundles, so keeping every teacher engine resident may not fit in GPU
memory. This module provides:

- ``TeacherSleepState``: pure-Python LRU/pin state machine deciding which teacher to
  put to sleep when a new teacher must be woken. Unit-testable without Ray.
- ``TeacherSleepController``: a Ray actor wrapping ``TeacherSleepState`` and issuing
  the actual ``sleep``/``wake_up`` RPCs to teacher server actors. A plain sync actor,
  so Ray serializes method calls, giving mutual exclusion across AgentLoopWorker
  processes.
- ``TeacherLLMServerClient``: an ``LLMServerClient`` subclass that acquires (wakes) its
  teacher before each ``generate`` call and releases it afterwards.
"""

import asyncio
import logging
import os
from typing import Any, Optional

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["TeacherSleepState", "TeacherSleepController", "TeacherLLMServerClient"]


class TeacherSleepState:
    """LRU/pin state machine for on-demand teacher wake/sleep.

    All teachers start asleep. ``awake`` is an LRU list (least-recently-used first,
    most-recent last); ``pins`` counts in-flight acquisitions per teacher. A pinned
    teacher is never evicted.
    """

    def __init__(self, keys: list[str], max_awake: int):
        if max_awake < 1:
            raise ValueError(f"max_awake must be >= 1, but got {max_awake}.")
        self._keys = set(keys)
        self._max_awake = max_awake
        self._awake: list[str] = []
        self._pins: dict[str, int] = {key: 0 for key in keys}

    @property
    def awake(self) -> list[str]:
        """Awake teachers, least-recently-used first."""
        return list(self._awake)

    def try_acquire(self, key: str) -> tuple[bool, list[str], bool]:
        """Try to pin ``key`` for use, evicting LRU unpinned teachers if needed.

        Returns:
            (success, to_sleep, need_wake):
                success: True if the caller may use the teacher now.
                to_sleep: teachers evicted from the awake set (must be slept by the caller).
                need_wake: True if ``key`` was asleep and must be woken by the caller.
        """
        if key not in self._keys:
            raise KeyError(f"Unknown teacher key {key!r}; known keys: {sorted(self._keys)}.")

        if key in self._awake:
            # Refresh LRU position (most-recent last).
            self._awake.remove(key)
            self._awake.append(key)
            self._pins[key] += 1
            return True, [], False

        to_sleep = []
        while len(self._awake) >= self._max_awake:
            victim = next((k for k in self._awake if self._pins[k] == 0), None)
            if victim is None:
                return False, [], False
            self._awake.remove(victim)
            to_sleep.append(victim)

        self._pins[key] += 1
        self._awake.append(key)
        return True, to_sleep, True

    def release(self, key: str) -> None:
        """Release one pin on ``key`` previously taken by ``try_acquire``."""
        if self._pins.get(key, 0) <= 0:
            raise ValueError(
                f"release({key!r}) called without a matching acquire "
                f"(pins={self._pins.get(key, 0)}). Acquire/release must be paired."
            )
        self._pins[key] -= 1

    def keys_to_sleep_all(self) -> list[str]:
        """Return all awake teachers and mark them asleep."""
        keys = list(self._awake)
        self._awake.clear()
        return keys


@ray.remote
class TeacherSleepController:
    """Ray actor serializing teacher wake/sleep across all AgentLoopWorker processes.

    A plain (sync) actor, so Ray serializes method calls: the state-machine update and
    the wake/sleep RPCs inside ``acquire`` are mutually exclusive with any other call.
    Wake/sleep RPCs complete before ``acquire`` returns True, so a subsequent
    ``generate`` never hits a sleeping engine.
    """

    def __init__(self, server_handles: dict[str, list[ActorHandle]], max_awake: int):
        self._server_handles = server_handles
        self._state = TeacherSleepState(keys=list(server_handles), max_awake=max_awake)

    def acquire(self, key: str) -> bool:
        """Pin ``key``, sleeping LRU victims and waking ``key`` as needed.

        Returns True once the teacher is awake and pinned; False if all awake teachers
        are pinned and no victim can be evicted (caller should retry later).
        """
        success, to_sleep, need_wake = self._state.try_acquire(key)
        if not success:
            return False
        for victim in to_sleep:
            logger.info(f"[TeacherSleepController] sleeping teacher {victim!r} to make room for {key!r}")
            ray.get([handle.sleep.remote() for handle in self._server_handles[victim]])
        if need_wake:
            logger.info(f"[TeacherSleepController] waking teacher {key!r}")
            ray.get([handle.wake_up.remote() for handle in self._server_handles[key]])
        return True

    def release(self, key: str) -> None:
        self._state.release(key)

    def sleep_all(self) -> None:
        """Sleep every currently-awake teacher."""
        for key in self._state.keys_to_sleep_all():
            ray.get([handle.sleep.remote() for handle in self._server_handles[key]])


class TeacherLLMServerClient(LLMServerClient):
    """LLMServerClient that wakes its teacher on demand in shared GPU group mode.

    With ``controller_handle=None`` this behaves exactly like ``LLMServerClient``.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ActorHandle = None,
        teacher_key: Optional[str] = None,
        controller_handle: Optional[ActorHandle] = None,
        **kwargs,
    ):
        """Initialize the TeacherLLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor.
            teacher_key (str): routing key of the teacher this client serves.
            controller_handle (ray.actor.ActorHandle): TeacherSleepController actor.
                Optional; when None, generate() skips wake/sleep entirely.
        """
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._teacher_key = teacher_key
        self._controller_handle = controller_handle

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if self._controller_handle is None:
            return await super().generate(
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )

        # Wake the teacher (sleeping an LRU victim if the awake budget is full) before
        # generating. acquire() returns False only when every awake teacher is pinned,
        # in which case retry until one is released.
        while not await self._controller_handle.acquire.remote(self._teacher_key):
            await asyncio.sleep(1.0)
        try:
            return await super().generate(
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )
        finally:
            await self._controller_handle.release.remote(self._teacher_key)
