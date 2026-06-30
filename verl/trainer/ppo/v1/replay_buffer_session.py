# Copyright 2026 Tencent Inc. and/or its affiliates
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
"""Rollout-level (session-counting) replay buffer.

Opt-in alternative to the default prompt-level :class:`~verl.trainer.ppo.v1.replay_buffer.
ReplayBuffer`. Select it via::

    trainer.v1.sampler.custom_sampler.path = "verl/trainer/ppo/v1/replay_buffer_session.py"
    trainer.v1.sampler.custom_sampler.name = "SessionReplayBuffer"

Pair it with the rollout-level worker (:class:`~verl.trainer.ppo.v1.agent_loop_tq_rollout.
RolloutAgentLoopManagerTQ`), which writes the per-session ``{uid}_sess{session_id}`` markers this
buffer reads. Readiness is derived from per-session completion (not a worker-owned prompt status),
so rollouts can be dispatched one session at a time, and an all-failed prompt is detected and
discarded by the streaming feeder (:meth:`dead_prompt_keys`) so a fresh prompt can replace it.
"""

import logging
import os
import time
from collections import defaultdict

import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))


def compute_complete_uids(prompt_n: dict, session_done: dict) -> set:
    """Return uids whose all ``n`` GRPO sessions have completed (success or failure).

    Pure function (stdlib only) so session-counting readiness can be unit-tested on a CPU-only
    machine without TransferQueue. ``prompt_n`` maps uid -> expected session count; ``session_done``
    maps uid -> set of completed session ids.
    """
    complete = set()
    for uid, n in prompt_n.items():
        if len(session_done.get(uid, ())) >= n:
            complete.add(uid)
    return complete


class SessionReplayBuffer(ReplayBuffer):
    """ReplayBuffer whose readiness is derived from per-session completion markers.

    ### [GRPO group sampling control]
    Readiness is **derived from per-session completion**, not a worker-owned prompt status, so
    rollouts can be dispatched at the granularity of a single session (decoupled from any worker):
    - Each prompt is registered with key ``{uid}``, tag ``{"is_prompt": True, "global_steps", "n"}``,
      where ``n`` is the number of GRPO sessions the prompt expects.
    - Each rollout writes a per-session completion marker ``{uid}_sess{session_id}``, tag
      ``{"is_session": True, "session_id", "status"}`` where status is ``success`` or ``failure``.
    A prompt becomes sampleable once all ``n`` of its session markers are present **and at least
    one succeeded**; its trajectories are then collected by ``{uid}`` prefix. Markers are written
    after the session's trajectory data, so "all markers present" implies "all available data
    present". A prompt whose every session failed is reported by :meth:`dead_prompt_keys`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # partition_id => {prompt_uid: n}, number of GRPO sessions the prompt expects.
        self.prompt_n: dict[str, dict[str, int]] = defaultdict(dict)
        # partition_id => {prompt_uid: set(session_id)}, completed sessions (success or failure).
        self.session_done: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # partition_id => {prompt_uid: set(session_id)}, completed sessions that SUCCEEDED. A prompt
        # with an empty success set (once complete) is all-failed -> discarded + replaced.
        self.session_success: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # partition_id => {prompt_uid: set(marker_key)}, per-session markers to clear on sample().
        self.session_marker_keys: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue (prompt ``n`` + per-session completion markers)."""
        self.partitions.clear()
        self.prompt_global_steps.clear()
        self.prompt_n.clear()
        self.session_done.clear()
        self.session_success.clear()
        self.session_marker_keys.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control] — prompt metadata (global_steps + n).
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    self.prompt_n[partition_id][key] = tag["n"]
                elif tag.get("is_session", False):
                    # Per-session completion marker `{uid}_sess{session_id}`.
                    uid = key.split("_")[0]
                    self.session_done[partition_id][uid].add(tag["session_id"])
                    if tag.get("status") == "success":
                        self.session_success[partition_id][uid].add(tag["session_id"])
                    self.session_marker_keys[partition_id][uid].add(key)
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    def _complete_uids(self, partition_id: str) -> set:
        """uids whose all ``n`` GRPO sessions have completed (success or failure)."""
        return compute_complete_uids(self.prompt_n[partition_id], self.session_done[partition_id])

    def _usable_uids(self, partition_id: str) -> set:
        """Complete uids with >=1 successful session — the only ones sampleable for training.

        (An all-failed prompt is complete but carries no usable trajectory; the streaming feeder
        discards it via :meth:`dead_prompt_keys` and feeds a replacement.)
        """
        success = self.session_success[partition_id]
        return {uid for uid in self._complete_uids(partition_id) if success.get(uid)}

    def dead_prompt_keys(self, partition_id: str = "train") -> list[str]:
        """The TransferQueue keys of all-failed prompts (complete but no successful session).

        A failed session writes no trajectory data, only a completion marker, so an all-failed
        prompt has only control-plane keys: its prompt key ``{uid}`` and its session markers.
        Returned read-only for the streaming feeder to ``kv_clear`` (the buffer never auto-discards
        — it only serves the trainer).
        """
        success = self.session_success[partition_id]
        keys: list[str] = []
        for uid in self._complete_uids(partition_id):
            if not success.get(uid):  # complete, yet every session failed
                keys.append(uid)
                keys.extend(self.session_marker_keys[partition_id].get(uid, ()))
        return keys

    def count_inflight(self, partition_id: str = "train") -> dict[str, int]:
        """Return the current un-consumed prompt counts, for throttling the streaming feeder.

        - incomplete: prompts fed but whose ``n`` sessions are not all done yet (still generating).
        - complete: prompts whose sessions are all done — ready to sample, not yet consumed.

        The streaming feeder bounds the sum of these (total un-consumed prompts) to keep the
        rollouter from running arbitrarily far ahead of training. (The feeder discards all-failed
        prompts each tick, so they leave this count instead of occupying the budget.)
        """
        self._sync_metadata_from_transfer_queue()
        total = len(self.prompt_global_steps[partition_id])
        complete = len(self._complete_uids(partition_id))
        return {"incomplete": total - complete, "complete": complete}

    def _has_enough_samples(self, global_steps: int, partition_id: str, batch_size: int) -> bool:
        # "none" applies no staleness gate: just wait for batch_size usable prompts and sample the
        # oldest (streaming bounds staleness via the feeder budget; TIS corrects off-policyness).
        # For wait strategy, we must wait for all still-incomplete prompts that have reached the
        # staleness threshold to finish before sampling.
        if self.max_off_policy_strategy == "wait":
            complete = self._complete_uids(partition_id)
            for uid, prompt_global_steps in self.prompt_global_steps[partition_id].items():
                if uid in complete:  # complete = usable + all-failed; neither is still generating
                    continue
                if (global_steps - prompt_global_steps + 1) / self.parameter_sync_step >= self.max_off_policy_threshold:
                    return False

        # Only prompts with >=1 successful session count toward the batch; all-failed ones are
        # discarded and effectively replaced by waiting for more usable prompts.
        return len(self._usable_uids(partition_id)) >= batch_size

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> KVBatchMeta:
        """Sample a batch of usable prompts (>=1 successful session), oldest first."""
        last_debug_time = time.time()
        self._sync_metadata_from_transfer_queue()
        while not self._has_enough_samples(global_steps, partition_id, batch_size):
            time.sleep(self.poll_interval)
            self._sync_metadata_from_transfer_queue()

            if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                total = len(self.prompt_global_steps[partition_id])
                usable = len(self._usable_uids(partition_id))
                logger.info(f"prompts in-flight: {total}, usable(ready): {usable}, incomplete: {total - usable}")
                last_debug_time = time.time()

        # Sample only prompts with >=1 successful session (all-failed prompts are never sampleable;
        # the feeder discards + replaces them). Prioritize the oldest to cut staleness.
        usable = self._usable_uids(partition_id)
        prompt_global_steps = self.prompt_global_steps[partition_id]
        sampleable_keys = sorted(usable, key=lambda uid: prompt_global_steps.get(uid, 0))
        selected_prompt_uids = sampleable_keys[:batch_size]
        selected = set(selected_prompt_uids)

        # Clear the prompt metadata keys and their per-session completion markers. The trajectory
        # data keys are cleared by the trainer after the batch is consumed (kv_clear(batch.keys)).
        marker_keys = []
        for uid in selected_prompt_uids:
            marker_keys.extend(self.session_marker_keys[partition_id].get(uid, ()))
        tq.kv_clear(partition_id=partition_id, keys=list(selected_prompt_uids) + marker_keys)

        # Collect the prompt's trajectory rows. Failed sessions wrote no data, so every data key of
        # a usable prompt belongs to a successful session — no per-row filtering needed.
        keys, tags = [], []
        for key, tag in self.partitions[partition_id].items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        return self._drop_max_off_policy_samples(global_steps, partition_id, batch)
