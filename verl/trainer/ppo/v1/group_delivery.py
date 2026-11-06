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
"""Fixed-batch sampling for synchronous, group-streamed GRPO."""

import time

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class GroupDeliveryReplayBuffer(ReplayBuffer):
    """Deliver complete groups from exactly one fixed batch, without refill or dropping."""

    def begin_batch(self, group_ids: list[str], global_steps: int, responses_per_group: int) -> None:
        """Record the identities and version of the batch before dispatching rollout."""
        if getattr(self, "remaining", None):
            raise RuntimeError("The previous group batch is incomplete")
        if not group_ids or len(set(group_ids)) != len(group_ids):
            raise ValueError("A synchronous batch requires unique group IDs")
        self.remaining = set(group_ids)
        self.batch_step = global_steps
        self.responses_per_group = responses_per_group

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        """Wait for any complete group; fail the whole batch if a sibling fails."""
        if partition_id != "train":
            return super().sample(global_steps, partition_id, batch_size)
        if batch_size != 1 or not getattr(self, "remaining", None) or global_steps != self.batch_step:
            raise ValueError("Group streaming must consume one group from the active synchronous batch")
        while True:
            self._sync_metadata_from_transfer_queue()
            versions = self.prompt_global_steps[partition_id]
            if set(versions) != self.remaining or any(step != global_steps for step in versions.values()):
                raise RuntimeError("TransferQueue contains missing, foreign, or wrong-version prompt groups")
            if self.failure_keys[partition_id]:
                raise RuntimeError(f"Rollout failed for synchronous groups: {sorted(self.failure_keys[partition_id])}")
            ready = self.finished_keys[partition_id] & self.remaining
            wait_for_all = self.trainer_config.get("wait_for_full_batch", False)
            if ready and (not wait_for_all or ready == self.remaining):
                uid = sorted(ready)[0]
                trajectories = {
                    key: tag for key, tag in self.partitions[partition_id].items() if key.split("_", 1)[0] == uid
                }
                expected = {f"{uid}_{session}_0" for session in range(self.responses_per_group)}
                if set(trajectories) != expected:
                    raise RuntimeError(
                        f"Group {uid} must have exactly one trajectory for each of {self.responses_per_group} sessions"
                    )
                for tag in trajectories.values():
                    if (
                        tag.get("global_steps") != global_steps
                        or tag.get("min_global_steps") != global_steps - 1
                        or tag.get("max_global_steps") != global_steps - 1
                    ):
                        raise RuntimeError(f"Group {uid} contains stale or cross-version trajectories")
                batch = self._materialize_batch(partition_id, [uid], trajectories)
                batch.extra_info["group_delivery_id"] = uid
                self.remaining.remove(uid)
                return batch, {}
            time.sleep(self.poll_interval)

    def finish_batch(self) -> None:
        """Check the barrier before committing an optimizer update."""
        if getattr(self, "remaining", None):
            raise RuntimeError(f"Still waiting for {len(self.remaining)} prompt groups")
