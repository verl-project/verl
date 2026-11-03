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
"""Rollout-side dispatcher that starts generation for prompts the trainer enqueued in TransferQueue.

The trainer writes prompt groups to TransferQueue as ``pending`` (see ReplayBuffer for the key and tag
schema) and never calls rollout. The dispatcher claims them and hands them to the agent loop workers,
which then follow the usual path: ``running`` → trajectories → ``finished`` / ``failure``.
"""

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any

import transfer_queue as tq

from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ACTIVE_STATUSES = ("pending", "running")


class PromptDispatcher:
    """Claims pending prompt groups and starts their generation.

    A prompt group is claimed when every replica has loaded at least its ``min_weight_version`` tag, so
    it never starts on weights that a replica is about to replace. Claims go oldest ``global_steps``
    first, and at most ``max_inflight_prompts`` claimed groups are pending or running at once.

    There is one dispatcher per rollout service: TransferQueue tags are last-writer-wins, so claimers
    working independently could start the same prompt twice. Run it as a Ray actor next to the agent
    loop workers: ``ray.remote(PromptDispatcher).remote(...)``.

    Args:
        agent_loop_workers: Handles of the AgentLoopWorkerTQ actors that run generation.
        update_agents: Handles of the replicas' WeightUpdateAgent actors, for their loaded versions.
        partition_id: TransferQueue partition holding the prompts.
        max_inflight_prompts: Cap on claimed groups that are not finished yet; None for no cap.
        poll_interval: Seconds between TransferQueue polls.
    """

    def __init__(
        self,
        agent_loop_workers: list,
        update_agents: list,
        partition_id: str = "train",
        max_inflight_prompts: int | None = None,
        poll_interval: float = 0.2,
    ):
        assert agent_loop_workers, "need at least one agent loop worker"
        assert max_inflight_prompts is None or max_inflight_prompts > 0
        tq.init()
        self.agent_loop_workers = agent_loop_workers
        self.update_agents = update_agents
        self.partition_id = partition_id
        self.max_inflight_prompts = max_inflight_prompts
        self.poll_interval = poll_interval

        self._inflight: set[str] = set()
        self._next_worker = 0
        self._stopped = False
        self._error: BaseException | None = None
        self.stats: dict[str, Any] = {"dispatched": 0, "waiting_for_version": 0, "loaded_version": -1}

    async def run(self):
        """Dispatch until `stop`. Re-raises the error that stopped it, if any."""
        try:
            while not self._stopped:
                await self.dispatch_once()
                await asyncio.sleep(self.poll_interval)
        except BaseException as e:
            self._error = e
            logger.exception("Prompt dispatcher stopped")
            raise

    async def _loaded_version(self) -> int:
        states = await asyncio.gather(*[agent.get_state.remote() for agent in self.update_agents])
        if any(state["failed"] for state in states):
            raise RuntimeError("A weight update agent failed; its replica cannot reach new versions")
        return min(state["version"] for state in states)

    async def dispatch_once(self) -> int:
        """Claim and start whatever is eligible now; returns the number of prompt groups started."""
        items = (tq.kv_list(partition_id=self.partition_id) or {}).get(self.partition_id, {})
        prompts = {key: tag for key, tag in items.items() if tag.get("is_prompt", False)}
        # A claimed group stops counting once it is finished, failed, or cleared from the queue.
        self._inflight = {key for key in self._inflight if prompts.get(key, {}).get("status") in _ACTIVE_STATUSES}

        loaded = await self._loaded_version()
        pending = [key for key, tag in prompts.items() if tag["status"] == "pending" and key not in self._inflight]
        eligible = sorted(
            (key for key in pending if prompts[key].get("min_weight_version", -1) <= loaded),
            key=lambda key: (prompts[key].get("global_steps", 0), key),
        )
        self.stats.update(loaded_version=loaded, waiting_for_version=len(pending) - len(eligible))
        if self.max_inflight_prompts is not None:
            eligible = eligible[: max(0, self.max_inflight_prompts - len(self._inflight))]
        if not eligible:
            return 0

        # The agent loop reads global_steps (the enqueue step) as one value per batch.
        by_step: dict[int, list[str]] = defaultdict(list)
        for key in eligible:
            by_step[prompts[key]["global_steps"]].append(key)
        for global_steps, keys in by_step.items():
            batch = tq.kv_batch_get(keys=keys, partition_id=self.partition_id)
            tu.assign_non_tensor_data(batch, "global_steps", global_steps)
            self._inflight.update(keys)
            await self._submit(batch)
        self.stats["dispatched"] += len(eligible)
        return len(eligible)

    async def _submit(self, batch):
        """Spread a batch over the workers, rotating so small batches do not all land on the first."""
        workers = self.agent_loop_workers
        start = self._next_worker % len(workers)
        workers = workers[start:] + workers[:start]
        chunks = batch.chunk(min(len(workers), len(batch)))
        self._next_worker += len(chunks)
        # generate_sequences only spawns the agent loops and returns.
        await asyncio.gather(
            *[worker.generate_sequences.remote(chunk) for worker, chunk in zip(workers, chunks, strict=False)]
        )

    def get_state(self) -> dict[str, Any]:
        return {"inflight": len(self._inflight), "failed": self._error is not None, **self.stats}

    def stop(self):
        self._stopped = True
