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

"""A manager-owned concurrency budget for cooperative async reward scorers."""

import asyncio
import uuid
from contextlib import asynccontextmanager, suppress

import ray


def run_concurrency_limit(reward_config) -> int | None:
    """Validate the optional cross-worker limit without changing legacy defaults."""
    limit = reward_config.get("run_max_concurrent")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("reward.run_max_concurrent must be a positive integer or null")
    return limit


class RewardConcurrencyBudget:
    """Serialize nonblocking grants in one Ray actor; never reset active grants.

    Calls are idempotent by token. A worker that dies while holding a grant
    leaves it occupied: automatically expiring it could admit new work while
    the original external request is still running. Restart the job to recover.
    """

    def __init__(self, limit: int):
        self.limit = run_concurrency_limit({"run_max_concurrent": limit})
        if self.limit is None:
            raise ValueError("A shared reward budget requires a limit")
        self.active: set[str] = set()

    def try_acquire(self, token: str) -> bool:
        """Grant immediately when capacity is available; do not queue actor calls."""
        if token in self.active:
            return True
        if len(self.active) >= self.limit:
            return False
        self.active.add(token)
        return True

    def release(self, token: str) -> None:
        """Release a grant, including an unknown/already released token."""
        self.active.discard(token)


def create_reward_budget(config):
    """Create one unnamed, non-restarting coordinator for this RewardLoopManager."""
    limit = run_concurrency_limit(config.reward)
    if limit is None:
        return None
    manager = config.reward.reward_manager
    if manager.source != "register" or manager.name != "rate_limited":
        raise ValueError("reward.run_max_concurrent requires the registered rate_limited reward manager")
    from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager
    from verl.trainer.ppo.reward import resolve_reward_manager_cls

    if resolve_reward_manager_cls(config) is not RateLimitedRewardManager:
        raise ValueError("reward.run_max_concurrent requires the built-in RateLimitedRewardManager")
    # Import and validate the scorer in reward workers, where its runtime context is available.
    if not (config.reward.get("custom_reward_function") or {}).get("path"):
        raise ValueError("reward.run_max_concurrent requires a custom async reward function")
    return ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)(RewardConcurrencyBudget).remote(limit)


async def _complete_rpc(awaitable):
    """Delay caller cancellation until an in-flight RPC has a known outcome.

    In particular, release must not overtake an acquire whose result has not
    arrived. Keep a strong reference and handle repeated cancellation while
    draining the RPC. RPC failures propagate instead of becoming zero rewards.
    """
    pending = asyncio.ensure_future(awaitable)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(pending)
            break
        except asyncio.CancelledError as error:
            if pending.cancelled():
                raise
            cancellation = error
    if cancellation is not None:
        raise cancellation
    return result


async def await_budgeted_reward(awaitable, timeout):
    """Cancel the scorer once, then drain cleanup despite repeated caller cancellation."""
    scorer = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(scorer), timeout)
    except (Exception, asyncio.CancelledError):
        scorer.cancel()
        cancellation = None
        while not scorer.done():
            try:
                await asyncio.shield(scorer)
            except asyncio.CancelledError as error:
                if not scorer.cancelled():
                    cancellation = error
            except Exception:
                break
        # Retrieve cleanup exceptions; preserve the original scorer failure/timeout.
        with suppress(Exception, asyncio.CancelledError):
            scorer.result()
        if cancellation is not None:
            raise cancellation from None
        raise


@asynccontextmanager
async def shared_reward_slot(budget, timeout=None):
    """Hold a cross-worker grant until the scorer and its cancellation cleanup finish."""
    if budget is None:
        yield
        return
    token = uuid.uuid4().hex

    async def acquire():
        while not await _complete_rpc(budget.try_acquire.remote(token)):
            await asyncio.sleep(0.05)

    try:
        try:
            await asyncio.wait_for(acquire(), timeout)
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                "Timed out waiting for the shared reward budget; a failed worker may still hold capacity. "
                "Restart the job after checking outstanding scorer requests."
            ) from error
        yield
    finally:
        await _complete_rpc(budget.release.remote(token))
