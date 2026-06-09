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
import asyncio
import logging
import os
import time
from collections import defaultdict

import transfer_queue as tq
from transfer_queue import KVBatchMeta

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tags and value. The tags are arbitrary metadata:
    `{"status": "running", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tags and value separately, the tags are stored in meta server, while the value is stored
    in storage units. ReplayBuffer periodically polls metadata from meta server to get the latest status of the
    trajectories.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompt with status `finished` or `failure`, its trajectories can be sampled by replay buffer.

    Args:
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
    """

    def __init__(self, poll_interval: float = 2.0):
        # partition_id => {key: tags}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)

        self.poll_interval = poll_interval
        self.poll_cond = asyncio.Condition()
        self.stop_event = asyncio.Event()
        self.background_task = asyncio.create_task(self._poll_from_transfer_queue())
        self.background_error = None

        # Off-policy training used only
        self.available_slots = 0
        self.slot_cond = asyncio.Condition()

    async def _poll_from_transfer_queue(self):
        """Periodically poll metadata from TransferQueue."""
        try:
            while not self.stop_event.is_set():
                async with self.poll_cond:
                    data = await tq.async_kv_list()
                    self._update_date(data)
                    self.poll_cond.notify_all()
                await asyncio.sleep(self.poll_interval)
        except Exception as e:
            if not self.stop_event.is_set():
                logger.exception(f"Error in _poll_from_transfer_queue: {e}")
                self.background_error = e

    def _update_date(self, data: dict[str, dict[str, dict]] | None):
        # TODO: Poll full metadata from TransferQueue for now, for better performance
        # 1. Delta polling: poll only the keys that are changed since last poll.
        # 2. Pass custom sampler to TransferQueue:
        #   https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()

        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tags in items.items():
                if tags.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    match tags["status"]:
                        case "pending":
                            self.pending_keys[partition_id].add(key)
                        case "running":
                            self.running_keys[partition_id].add(key)
                        case "finished":
                            self.finished_keys[partition_id].add(key)
                        case "failure":
                            self.failure_keys[partition_id].add(key)
                        case _:
                            raise ValueError(f"Unknown status: {tags['status']}")
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tags)

    async def close(self):
        """Stop the background polling task."""
        self.stop_event.set()
        await self.background_task
        if self.background_error is not None:
            raise self.background_error

    async def sample(self, partition_id: str, batch_size: int) -> KVBatchMeta:
        """Sample a batch of data from the replay buffer.

        NOTE: This method is meant to be overridden by user to customize sampling strategy.

        Args:
            partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
            batch_size (int, optional): Batch size.

        Returns:
            KVBatchMeta: A batch of data.
        """
        last_debug_time = time.time()
        async with self.poll_cond:
            while True:
                finished_keys = self.finished_keys[partition_id]
                failure_keys = self.failure_keys[partition_id]
                if len(finished_keys) + len(failure_keys) >= batch_size:
                    break
                if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                    logger.info(
                        f"pending: {len(self.pending_keys[partition_id])}, "
                        f"running: {len(self.running_keys[partition_id])}, "
                        f"finished: {len(self.finished_keys[partition_id])}, "
                        f"failure: {len(self.failure_keys[partition_id])}"
                    )
                    last_debug_time = time.time()
                await self.poll_cond.wait()

            # TODO: should we filter out samples with some of their sessions failed?
            selected_prompt_uids = list(finished_keys.union(failure_keys))[:batch_size]
            await tq.async_kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

            keys, tags = [], []
            for key, tag in self.partitions[partition_id].items():
                uid = key.split("_")[0]
                if uid in selected_prompt_uids:
                    keys.append(key)
                    tags.append(tag)
            return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)

    async def add_slot(self, num_slots: int):
        """Trainer add available slots to the replay buffer.

        Args:
            num_slots (int): Number of slots to add.
        """
        async with self.slot_cond:
            self.available_slots += num_slots
            self.slot_cond.notify_all()

    async def acquire_slot(self, max_slots: int) -> int:
        """Rollouter acquire available slots from the replay buffer.

        Args:
            max_slots (int): Maximum number of slots to acquire.

        Returns:
            int: The number of slots acquired.
        """
        async with self.slot_cond:
            if self.available_slots <= 0:
                await self.slot_cond.wait()
            slot = min(self.available_slots, max_slots)
            self.available_slots -= slot
            return slot
