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
import logging
import os
import time
from collections import defaultdict

import numpy as np
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))


# TODO: Pass custom sampler to TransferQueue:
# https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py


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


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "success", "seq_len": ..., ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    ### [GRPO group sampling control]
    Readiness is **derived from per-session completion**, not a worker-owned prompt status, so
    rollouts can be dispatched at the granularity of a single session (decoupled from any worker):
    - Each prompt is registered with key `{uid}`, tag `{"is_prompt": True, "global_steps", "n"}`,
      where `n` is the number of GRPO sessions the prompt expects.
    - Each rollout writes a per-session completion marker `{uid}_sess{session_id}`, tag
      `{"is_session": True, "session_id", "status"}` where status is `success` or `failure`.
    A prompt becomes sampleable once all `n` of its session markers are present (success or
    failure); its trajectories are then collected by `{uid}` prefix. Markers are written after the
    session's trajectory data, so "all markers present" implies "all available data present".

    Args:
        trainer_mode (str): Trainer mode.
        trainer_config (DictConfig): Trainer configuration.
        max_off_policy_threshold (int): Maximum number of model versions that trajectory can span.
        max_off_policy_strategy (str): How to handle trajectory that exceeds the maximum number of model versions.
        sampler_kwargs (dict): Additional kwargs for the custom sampler.
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig,
        poll_interval: float = 2.0,
        rollout_level_dispatch: bool = False,
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.parameter_sync_step = trainer_config.get("parameter_sync_step", 1)
        # Opt-in: rollout-level dispatch uses session-counting readiness; default False keeps the
        # legacy prompt-status readiness (pending/running/finished/failure buckets).
        self.session_counting = bool(rollout_level_dispatch)

        assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
            f"Invalid max off policy threshold: {self.max_off_policy_threshold}, must be an integer greater than 0"
        )
        assert self.max_off_policy_strategy in ["drop", "wait", "none"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait', 'none']"
        )

        # partition_id => {key: tag} for trajectory (data) keys only.
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        # partition_id => {prompt_uid: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)
        # Legacy prompt-status buckets (used when session_counting=False).
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)
        # partition_id => {prompt_uid: n}, number of GRPO sessions the prompt expects.
        self.prompt_n: dict[str, dict[str, int]] = defaultdict(dict)
        # partition_id => {prompt_uid: set(session_id)}, completed sessions (success or failure).
        self.session_done: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # partition_id => {prompt_uid: set(marker_key)}, per-session markers to clear on sample().
        self.session_marker_keys: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue.

        Routes prompt tags by scheme: legacy tags carry a ``status`` (-> pending/running/
        finished/failure buckets); rollout-level tags carry ``n`` + per-session ``is_session``
        markers (-> prompt_n / session_done). Both are populated so either readiness model works.
        """
        self.partitions.clear()
        self.prompt_global_steps.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.prompt_n.clear()
        self.session_done.clear()
        self.session_marker_keys.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    if "n" in tag:
                        # rollout-level scheme: readiness derived from per-session markers.
                        self.prompt_n[partition_id][key] = tag["n"]
                    else:
                        # legacy scheme: GRPO-group status set by the worker.
                        match tag["status"]:
                            case "pending":
                                self.pending_keys[partition_id].add(key)
                            case "running":
                                self.running_keys[partition_id].add(key)
                            case "finished":
                                self.finished_keys[partition_id].add(key)
                            case "failure":
                                self.failure_keys[partition_id].add(key)
                            case _:
                                raise ValueError(f"Unknown status: {tag['status']}")
                elif tag.get("is_session", False):
                    # rollout-level per-session completion marker `{uid}_sess{session_id}`.
                    uid = key.split("_")[0]
                    self.session_done[partition_id][uid].add(tag["session_id"])
                    self.session_marker_keys[partition_id][uid].add(key)
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    def _complete_uids(self, partition_id: str) -> set:
        """uids whose all ``n`` GRPO sessions have completed (ready to sample)."""
        return compute_complete_uids(self.prompt_n[partition_id], self.session_done[partition_id])

    def _has_enough_samples(self, global_steps: int, partition_id: str, batch_size: int) -> bool:
        # "none" applies no staleness gate: just wait for batch_size ready prompts and sample the
        # oldest (off-policyness is corrected downstream, e.g. via importance sampling / TIS).
        # For wait strategy, we must wait for all still-unready prompts that have reached the
        # staleness threshold to finish before sampling.
        if not self.session_counting:
            if self.max_off_policy_strategy == "wait":
                for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                    prompt_global_steps = self.prompt_global_steps[partition_id][key]
                    if (
                        global_steps - prompt_global_steps + 1
                    ) / self.parameter_sync_step >= self.max_off_policy_threshold:
                        return False
            return len(self.finished_keys[partition_id]) + len(self.failure_keys[partition_id]) >= batch_size

        if self.max_off_policy_strategy == "wait":
            complete = self._complete_uids(partition_id)
            for uid, prompt_global_steps in self.prompt_global_steps[partition_id].items():
                if uid in complete:
                    continue
                if (global_steps - prompt_global_steps + 1) / self.parameter_sync_step >= self.max_off_policy_threshold:
                    return False

        return len(self._complete_uids(partition_id)) >= batch_size

    def _drop_max_off_policy_samples(
        self, global_steps: int, partition_id: str, batch: KVBatchMeta
    ) -> tuple[KVBatchMeta, dict]:
        if not self.max_off_policy_strategy == "drop":
            return batch, {}

        kept_keys, kept_tags = [], []
        dropped_keys, dropped_tags = [], []
        for key, tag in zip(batch.keys, batch.tags, strict=False):
            prompt_global_steps = tag["global_steps"]
            if (global_steps - prompt_global_steps + 1) / self.parameter_sync_step > self.max_off_policy_threshold:
                dropped_keys.append(key)
                dropped_tags.append(tag)
            else:
                kept_keys.append(key)
                kept_tags.append(tag)

        # Remove dropped keys from TransferQueue
        metrics = {}
        if len(dropped_keys) > 0:
            # TODO: should we drop the entire GRPO group if any of its sessions exceeds the threshold?
            tq.kv_clear(partition_id=batch.partition_id, keys=dropped_keys)
            logger.warning(f"Dropped {len(dropped_keys)} max off policy samples from partition {batch.partition_id}")
            dropped_global_steps = np.array([tag["global_steps"] for tag in dropped_tags])
            trajectory_staleness = (global_steps - dropped_global_steps + 1) / self.parameter_sync_step
            prefix = "training" if partition_id == "train" else "validation"
            metrics[f"{prefix}/off_policy/dropped_samples"] = len(dropped_keys)
            metrics[f"{prefix}/off_policy/dropped_samples_staleness/mean"] = trajectory_staleness.mean()
            metrics[f"{prefix}/off_policy/dropped_samples_staleness/max"] = trajectory_staleness.max()
            metrics[f"{prefix}/off_policy/dropped_samples_staleness/min"] = trajectory_staleness.min()

        return KVBatchMeta(partition_id=batch.partition_id, keys=kept_keys, tags=kept_tags), metrics

    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> KVBatchMeta:
        """Sample a batch of data from the replay buffer.

        NOTE: user can customize sampling strategy by setting:
        ```bash
        trainer.v1.sampler.custom_sampler.path = "path/to/your/sampler.py"
        trainer.v1.sampler.custom_sampler.name = "UserCustomReplayBuffer"
        ```

        Args:
            global_steps (int): Global steps of the current training.
            partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
            batch_size (int, optional): Batch size.

        Returns:
            KVBatchMeta: A batch of data.
            dict: Auxiliary metrics.
        """
        last_debug_time = time.time()
        self._sync_metadata_from_transfer_queue()
        while not self._has_enough_samples(global_steps, partition_id, batch_size):
            time.sleep(self.poll_interval)
            self._sync_metadata_from_transfer_queue()

            if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                if self.session_counting:
                    total = len(self.prompt_global_steps[partition_id])
                    complete = len(self._complete_uids(partition_id))
                    logger.info(
                        f"prompts in-flight: {total}, complete(ready): {complete}, incomplete: {total - complete}"
                    )
                else:
                    logger.info(
                        f"pending: {len(self.pending_keys[partition_id])}, "
                        f"running: {len(self.running_keys[partition_id])}, "
                        f"finished: {len(self.finished_keys[partition_id])}, "
                        f"failure: {len(self.failure_keys[partition_id])}"
                    )
                last_debug_time = time.time()

        # Select the oldest ready prompts (smallest global_steps first) to reduce staleness.
        # TODO: should we filter out samples with some of their sessions failed?
        prompt_global_steps = self.prompt_global_steps[partition_id]
        if self.session_counting:
            ready = self._complete_uids(partition_id)
        else:
            ready = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        sampleable_keys = sorted(ready, key=lambda uid: prompt_global_steps.get(uid, 0))
        selected_prompt_uids = sampleable_keys[:batch_size]
        selected = set(selected_prompt_uids)

        # Clear the prompt metadata keys (and, in session-counting mode, their per-session markers).
        # The trajectory data keys are cleared by the trainer after the batch is consumed.
        clear_keys = list(selected_prompt_uids)
        if self.session_counting:
            for uid in selected_prompt_uids:
                clear_keys.extend(self.session_marker_keys[partition_id].get(uid, ()))
        tq.kv_clear(partition_id=partition_id, keys=clear_keys)

        keys, tags = [], []
        for key, tag in self.partitions[partition_id].items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        return self._drop_max_off_policy_samples(global_steps, partition_id, batch)
