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

"""Dynamic resource controller for fully-async training.

State machine:  STANDALONE_ONLY  <->  HYBRID_ACTIVE

Activate (after weight sync):
  1. add_replicas               – register hybrid replicas in the load balancer.
  2. resume_generation_replicas – allow hybrid replicas to accept requests.

Deactivate (order is critical):
  1. remove_replicas  – cut routing; prevents retry loop re-routing to dying replicas.
  2. abort_replicas   – abort in-flight requests; partial-rollout retries to standalone.
  3. sleep_replicas   – release KV cache + offload weights, return GPU to training.
"""

import time

import ray

from verl.experimental.fully_async_policy.dynamic_schedule.base import (
    DynamicSchedulePolicyBase,
    build_policy,
    register_policy,
)
from verl.experimental.fully_async_policy.dynamic_schedule.default_policy import DefaultDynamicSchedulePolicy
from verl.experimental.fully_async_policy.dynamic_schedule.static_fully_async_policy import StaticFullyAsyncPolicy

__all__ = [
    "DynamicResourceController",
    "DynamicSchedulePolicyBase",
    "DefaultDynamicSchedulePolicy",
    "StaticFullyAsyncPolicy",
    "build_policy",
    "register_policy",
]


class DynamicResourceController:
    """Manages hybrid rollout replica lifecycle (STANDALONE_ONLY <-> HYBRID_ACTIVE).

    FullyAsyncTrainer drives transitions via activate_hybrid_replicas /
    deactivate_hybrid_replicas after each weight-sync step.  The *policy*
    (DynamicSchedulePolicyBase) decides when to deactivate.

    Args:
        rollouter: Ray actor handle for FullyAsyncRollouter.
        hybrid_checkpoint_manager: CheckpointEngineManager with naive backend.
        num_standalone_replicas: Number of standalone replicas (for logging).
        num_hybrid_replicas: Number of hybrid replicas (for logging).
        policy: Scheduling policy; defaults to DefaultDynamicSchedulePolicy.
    """

    def __init__(
        self,
        rollouter,
        hybrid_checkpoint_manager,
        num_standalone_replicas: int,
        num_hybrid_replicas: int,
        policy: DynamicSchedulePolicyBase | None = None,
    ):
        self.rollouter = rollouter
        self.hybrid_checkpoint_manager = hybrid_checkpoint_manager
        self.num_standalone = num_standalone_replicas
        self.num_hybrid = num_hybrid_replicas
        self._hybrid_active: bool = False
        self._only_hybrid: bool = num_standalone_replicas == 0
        self.activate_count: int = 0
        self.deactivate_count: int = 0
        self.policy: DynamicSchedulePolicyBase = (
            policy if policy is not None else DefaultDynamicSchedulePolicy(only_hybrid=self._only_hybrid)
        )
        # Inject rollouter reference so the policy can perform request rebalancing.
        self.policy._rollouter = rollouter

    @property
    def is_hybrid_active(self) -> bool:
        return self._hybrid_active

    @property
    def has_hybrid_replicas(self) -> bool:
        """Whether any hybrid replica has been registered with the hybrid checkpoint manager."""
        return bool(self.hybrid_checkpoint_manager.replicas)

    async def sync_hybrid_weights(self, global_steps: int) -> None:
        """Push the latest trainer weights to hybrid rollout replicas (naive backend).

        Wraps the weight sync with abort/resume of in-flight requests so that
        hybrid replicas never receive weight updates while serving a request,
        mirroring the guard that CheckpointEngineManager.update_weights() applies
        for the non-naive (standalone) path. No-ops when no hybrid replicas have
        been registered yet (e.g. still initializing, or purely standalone setup).

        Args:
            global_steps: Current global step (parameter version) to tag the sync with.
        """
        if not self.has_hybrid_replicas:
            print("[DynamicResourceController] No hybrid replicas registered, skipping weight sync")
            return

        await self.hybrid_checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.update_weights(global_steps=global_steps)
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

    async def activate_hybrid_replicas(self, global_steps: int) -> None:
        """Add hybrid replicas to the LB and resume generation (weight sync must be done first)."""
        print(f"[DynamicResourceController] Activating hybrid replicas at step {global_steps}")
        start = time.time()

        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        if not hybrid_resource_ids:
            print("[DynamicResourceController] No hybrid replicas found, skipping activation")
            return

        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        self._hybrid_active = True
        self.activate_count += 1
        print(
            f"[DynamicResourceController] Activated {len(hybrid_resource_ids)} replicas "
            f"in {time.time() - start:.2f}s (count={self.activate_count})"
        )

    async def deactivate_hybrid_replicas(self, global_steps: int) -> None:
        """Remove hybrid replicas from LB, abort in-flight requests, release GPU memory."""
        print(f"[DynamicResourceController] Deactivating hybrid replicas at step {global_steps}")
        start = time.time()

        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        if not hybrid_resource_ids:
            print("[DynamicResourceController] No hybrid replicas found, skipping deactivation")
            self._hybrid_active = False
            return

        # Order is critical: remove from LB first so retry loop can't re-route to dying replicas.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.sleep_replicas()

        self._hybrid_active = False
        self.deactivate_count += 1
        print(
            f"[DynamicResourceController] Deactivated {len(hybrid_resource_ids)} replicas "
            f"in {time.time() - start:.2f}s (count={self.deactivate_count})"
        )
