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


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "running", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompt with status `finished` or `failure`, its trajectories can be sampled by replay buffer.

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
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.parameter_sync_step = trainer_config.get("parameter_sync_step", 1)

        assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
            f"Invalid max off policy threshold: {self.max_off_policy_threshold}, must be an integer greater than 0"
        )
        assert self.max_off_policy_strategy in ["drop", "wait"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait']"
        )

        # partition_id => {key: tag}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)
        # partition_id => {prompt_key: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue."""
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.prompt_global_steps.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
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
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    def _has_enough_samples(self, global_steps: int, partition_id: str, batch_size: int) -> bool:
        # For wait strategy, we need to wait all trajectories that reach threshold to finish
        if self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) / self.parameter_sync_step >= self.max_off_policy_threshold:
                    return False

        return len(self.finished_keys[partition_id]) + len(self.failure_keys[partition_id]) >= batch_size

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
                logger.info(
                    f"pending: {len(self.pending_keys[partition_id])}, "
                    f"running: {len(self.running_keys[partition_id])}, "
                    f"finished: {len(self.finished_keys[partition_id])}, "
                    f"failure: {len(self.failure_keys[partition_id])}"
                )
                last_debug_time = time.time()

        # TODO: should we filter out samples with some of their sessions failed?
        finished_keys = self.finished_keys[partition_id]
        failure_keys = self.failure_keys[partition_id]
        # Prioritize sampling the oldest prompts (smallest global_steps first) to reduce staleness.
        prompt_global_steps = self.prompt_global_steps[partition_id]
        sampleable_keys = sorted(finished_keys.union(failure_keys), key=lambda key: prompt_global_steps.get(key, 0))
        selected_prompt_uids = sampleable_keys[:batch_size]
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in self.partitions[partition_id].items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        return self._drop_max_off_policy_samples(global_steps, partition_id, batch)
