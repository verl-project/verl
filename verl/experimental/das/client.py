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
"""DAS on top of verl's ``LLMServerClient``: problem identity, request ids, trajectory pushes.

:func:`wrap_llm_client` re-types an existing client (``LLMServerClient`` or
``FullyAsyncLLMServerClient``) as a subclass that

- derives a problem hash from the first-turn prompt at ``_acquire_server`` time
  and pins it to verl's trajectory-sticky ``request_id`` (later turns and
  resumed partial rollouts keep the hash even though their prompts grew),
- names every engine request ``dasp{hash}-{uuid}`` through ``_vllm_request_id``
  so the patched proposer can find the problem's tree, and
- after ``generate`` returns, pushes the response tokens to the store
  (fire-and-forget, gated by :data:`DAS_PUSH_ENABLED` so eval batches are
  never collected).

For the fully-async client, the wrapper sits outside the resume loop: one push
per trajectory with the concatenated response; each resumed segment still gets
a fresh engine id carrying the same hash.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from verl.experimental.das.problem_id import encode_request_id, hash_problem_id
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Per-task gate for trajectory collection. Agent-loop workers set it around
# ``generate_sequences`` from the batch's ``validate`` flag; asyncio copies the
# context into every task spawned underneath, so fire-and-forget trajectories
# of a training batch keep pushing even if a validation batch arrives later.
DAS_PUSH_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar("verl_das_push_enabled", default=True)

_TRAJ_PROBLEM_LRU_CAPACITY = 65536


@contextlib.contextmanager
def push_gate(enabled: bool):
    """Set :data:`DAS_PUSH_ENABLED` for the duration of a ``generate_sequences`` call.

    Agent-loop workers call this instead of touching the ContextVar directly: Ray pickles
    actor classes by value when it cannot resolve them by name, and a ContextVar referenced
    from a method's globals is not picklable, whereas this function is.
    """
    token = DAS_PUSH_ENABLED.set(enabled)
    try:
        yield
    finally:
        DAS_PUSH_ENABLED.reset(token)


@dataclass
class _RequestState:
    phash: str | None


class DASClientMixin:
    """DAS behaviour layered over any ``LLMServerClient`` subclass; see :func:`wrap_llm_client`."""

    def _das_init(self, cfg: DASConfig, track_weight_version: bool = False) -> None:
        self._das_cfg = cfg
        self._das_requests: dict[str, _RequestState] = {}
        self._das_traj_problems: OrderedDict[str, str] = OrderedDict()
        self._das_service = None
        self._das_push_client = None
        self._das_track_weight_version = track_weight_version
        self._das_seen_version: int | None = None
        self._das_errors = 0
        rollout_cfg = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        self._das_full_determinism = bool(getattr(rollout_cfg, "full_determinism", False))
        if self._das_full_determinism:
            logger.warning(
                "DAS: rollout.full_determinism=True keeps verl's stable request ids, so problem-hash request "
                "ids are disabled and engines run stock suffix decoding"
            )

    # ------------------------------------------------------------ identity

    def _das_problem_for(self, request_id: str, prompt_ids) -> str | None:
        phash = self._das_traj_problems.get(request_id)
        if phash is not None:
            self._das_traj_problems.move_to_end(request_id)
            return phash
        if not prompt_ids:
            return None
        phash = hash_problem_id(prompt_ids)
        self._das_traj_problems[request_id] = phash
        while len(self._das_traj_problems) > _TRAJ_PROBLEM_LRU_CAPACITY:
            self._das_traj_problems.popitem(last=False)
        return phash

    async def _acquire_server(self, request_id: str, **extra):
        phash = None if self._das_full_determinism else self._das_problem_for(request_id, extra.get("prompt_ids"))
        self._das_requests[request_id] = _RequestState(phash=phash)
        return await super()._acquire_server(request_id, **extra)

    def _vllm_request_id(self, request_id: str) -> str:
        state = self._das_requests.get(request_id)
        if state is not None and state.phash is not None:
            # Fresh per engine call (turn or resumed segment), like verl's per-turn uuid.
            return encode_request_id(state.phash)
        return super()._vllm_request_id(request_id)

    # ------------------------------------------------------------ generate

    async def generate(self, request_id, *, prompt_ids: list[int], sampling_params: dict[str, Any], **kwargs):
        try:
            output = await super().generate(
                request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, **kwargs
            )
        finally:
            state = self._das_requests.pop(request_id, None)
        if state is not None and state.phash is not None and DAS_PUSH_ENABLED.get():
            self._das_push(state.phash, getattr(output, "token_ids", None))
        if self._das_track_weight_version:
            self._das_note_weight_version(output)
        return output

    # --------------------------------------------------------------- store

    def _das_get_service(self):
        """The job's store actor handle, or None while it does not exist yet."""
        if self._das_service is None:
            try:
                import ray

                self._das_service = ray.get_actor(self._das_cfg.service_name)
            except Exception:
                return None
        return self._das_service

    def _das_push(self, phash: str, token_ids) -> None:
        """Enqueue one trajectory's response tokens. Never raises into the rollout path."""
        try:
            if not token_ids:
                return
            if self._das_push_client is None:
                service = self._das_get_service()
                if service is None:
                    return
                from verl.experimental.das.push_client import DASPushClient

                self._das_push_client = DASPushClient(service, f"verl-das-{uuid.uuid4().hex[:8]}", self._das_cfg)
            from verl.experimental.das.contracts import TrajectoryPush

            self._das_push_client.enqueue(TrajectoryPush(phash, tuple(int(t) for t in token_ids)))
        except Exception as e:
            self._das_errors += 1
            if self._das_errors == 1 or self._das_errors % 1000 == 0:
                logger.warning("DAS: trajectory push failed (%d so far): %s", self._das_errors, e)

    def _das_note_weight_version(self, output) -> None:
        """Fully-async step boundary: first sighting of a newer weight version opens the next iteration.

        Trajectories generated under weight version ``v`` are logged while iteration ``v + 1`` is in
        progress; the first response seen from version ``v + 1`` makes them servable. The store's
        ``begin_iteration`` is monotone and idempotent, so racing workers are harmless.
        """
        try:
            extra = getattr(output, "extra_fields", None) or {}
            version = extra.get("max_global_steps")
            if version is None:
                version = extra.get("global_steps")
            if version is None:
                return  # no weight sync yet
            version = int(version)
            if self._das_seen_version is not None and version <= self._das_seen_version:
                return
            self._das_seen_version = version
            service = self._das_get_service()
            if service is not None:
                service.begin_iteration.remote(version + 1)
        except Exception as e:
            logger.warning("DAS: weight-version boundary signal failed: %s", e)


_WRAPPED_CLASSES: dict[type, type] = {}


def wrap_llm_client(client, cfg: DASConfig, track_weight_version: bool = False):
    """Return ``client`` re-typed as a DAS subclass of its own class.

    Subclassing the concrete class keeps that class's behaviour (for example the
    fully-async client's retry while the balancer is empty during weight sync).
    The wrapped instance shares the original's state and must be built inside
    the agent-loop worker: the push client it lazily creates is not picklable.
    """
    if isinstance(client, DASClientMixin):
        return client
    base_cls = type(client)
    cls = _WRAPPED_CLASSES.get(base_cls)
    if cls is None:
        cls = type(
            f"DAS{base_cls.__name__}",
            (DASClientMixin, base_cls),
            {"__module__": __name__, "__doc__": f"{base_cls.__name__} with DAS problem ids and trajectory pushes."},
        )
        _WRAPPED_CLASSES[base_cls] = cls
    wrapped = cls.__new__(cls)
    wrapped.__dict__.update(client.__dict__)
    wrapped._das_init(cfg, track_weight_version=track_weight_version)
    return wrapped
