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

from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))


def _accumulate_drop_metrics(acc: dict, new: dict, dropped: int) -> None:
    """Merge one poll iteration's staleness-drop metrics into ``acc`` in place.

    ``dropped`` weights the staleness mean so it stays a true per-sample average across iterations.
    """
    count_key = next((k for k in new if k.endswith("/dropped_samples")), None)
    prev_total = acc.get(count_key, 0) if count_key else 0

    for key, value in new.items():
        if key.endswith("/dropped_samples"):
            acc[key] = acc.get(key, 0) + value
        elif key.endswith("/dropped_samples_staleness/mean"):
            denom = prev_total + dropped
            acc[key] = (acc.get(key, 0.0) * prev_total + value * dropped) / denom if denom else value
        elif key.endswith("/dropped_samples_staleness/max"):
            acc[key] = max(acc.get(key, value), value)
        elif key.endswith("/dropped_samples_staleness/min"):
            acc[key] = min(acc.get(key, value), value)
        else:
            # DAPO/failure counters are simple sums.
            acc[key] = acc.get(key, 0) + value


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
    Only prompts with status `finished` or `failure` enter terminal-group handling.

    ### [Terminal-group handling matrix]
    ``drop`` means off-policy staleness dropping; ``DAPO`` means filtering groups whose configured reward
    metric is identical across all trajectories. The matrix applies to the training partition:

    | trainer mode | drop | DAPO | failure |
    | --- | --- | --- | --- |
    | sync | Keep; no refill. | Drop ``k``; submit ``2k``; wait; trim surplus. | Keep/sample; no refill. |
    | async | Drop ``k``; refill ``k``. | Drop ``k``; refill ``k``. | Drop ``k``; refill ``k``. |

    DAPO filtering requires its metric to be available before a prompt becomes ``finished``. This is the
    async reward-computation path used by rule-based rewards (or a reward model with its own resource pool),
    regardless of whether the trainer itself is sync or async. A colocated reward model computes rewards only
    after ``sample`` returns and therefore cannot use this filtering path.
    ``algorithm.filter_groups.max_num_gen_batches`` is not enforced by this v1 path.

    Args:
        trainer_mode (str): Trainer mode.
        trainer_config (DictConfig): Trainer configuration.
        max_off_policy_threshold (int): Maximum number of model versions that trajectory can span.
        max_off_policy_strategy (str): How to handle trajectory that exceeds the maximum number of model versions.
        sampler_kwargs (dict): Additional kwargs for the custom sampler.
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
        refill_fn (callable, optional): Trainer-injected function that submits an exact number of fresh prompts.
        filter_groups_metric (str, optional): DAPO group-filtering metric read from each trajectory's
            ``extra_fields.reward_extra_info``. ``None`` disables DAPO filtering.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig,
        poll_interval: float = 2.0,
        refill_fn=None,
        filter_groups_metric: str | None = None,
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.refill_fn = refill_fn
        self.filter_groups_metric = filter_groups_metric

        assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
            f"Invalid max off policy threshold: {self.max_off_policy_threshold}, must be an integer greater than 0"
        )
        assert self.max_off_policy_strategy in ["drop", "wait"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait']"
        )
        if self.filter_groups_metric is not None and self.refill_fn is None:
            raise ValueError("DAPO group filtering requires refill_fn")

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

    @staticmethod
    def _metrics_prefix(partition_id: str) -> str:
        return "training" if partition_id == "train" else "validation"

    def _sync_dapo_enabled(self, partition_id: str) -> bool:
        return self.trainer_mode == "sync" and partition_id != "val" and self.filter_groups_metric is not None

    def _traj_keys_of(self, partition_id: str, uids: set[str]) -> list[str]:
        """Trajectory keys "{uid}_..." belonging to the given prompt uids (kv_clear does not cascade)."""
        return [key for key in self.partitions[partition_id] if key.split("_")[0] in uids]

    def _clear_groups(self, partition_id: str, uids: set[str]) -> None:
        """Remove the given prompt groups (prompt keys + their trajectory keys) from TransferQueue."""
        tq.kv_clear(partition_id=partition_id, keys=list(uids) + self._traj_keys_of(partition_id, uids))

    def _stale_terminal_keys(self, global_steps: int, partition_id: str) -> set[str]:
        """Stale terminal prompts dropped only by async trainers using the ``drop`` strategy."""
        if partition_id == "val" or self.trainer_mode == "sync" or self.max_off_policy_strategy != "drop":
            return set()
        prompt_global_steps = self.prompt_global_steps[partition_id]
        terminal_keys = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        return {
            uid
            for uid in terminal_keys
            if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
        }

    def _dapo_filtered_keys(self, partition_id: str) -> set[str]:
        """Finished groups whose configured DAPO metric is identical across all trajectories."""
        if partition_id == "val" or self.filter_groups_metric is None:
            return set()

        finished_uids = self.finished_keys[partition_id]
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in finished_uids]
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        missing_metric_uids = finished_uids - {key.split("_")[0] for key in trajectory_keys}

        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields_list = data["extra_fields"].tolist()
        else:
            extra_fields_list = []

        for key, extra_fields in zip(trajectory_keys, extra_fields_list, strict=True):
            uid = key.split("_")[0]
            extra_fields = getattr(extra_fields, "data", extra_fields)
            reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
            if self.filter_groups_metric not in reward_extra_info:
                missing_metric_uids.add(uid)
            else:
                metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))

        if missing_metric_uids:
            raise RuntimeError(
                f"Finished groups are missing DAPO metric {self.filter_groups_metric!r}: "
                f"{sorted(missing_metric_uids)[:5]}"
            )

        return {uid for uid, values in metrics_by_uid.items() if len(values) > 1 and float(np.std(values)) == 0.0}

    def _terminal_drop_reasons(self, global_steps: int, partition_id: str) -> tuple[set[str], set[str], set[str]]:
        """Return stale, DAPO-filtered, and failed groups for this metadata snapshot.

        The sets may overlap. Callers clear and refill their union, so one prompt is never handled twice.
        """
        if partition_id == "val":
            return set(), set(), set()

        stale_uids = self._stale_terminal_keys(global_steps, partition_id)
        dapo_uids = self._dapo_filtered_keys(partition_id)
        failed_uids = set() if self.trainer_mode == "sync" else set(self.failure_keys[partition_id])
        return stale_uids, dapo_uids, failed_uids

    def _sampleable_terminal_keys(
        self,
        global_steps: int,
        partition_id: str,
        drop_reasons: tuple[set[str], set[str], set[str]] | None = None,
    ) -> set[str]:
        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        if drop_reasons is None:
            drop_reasons = self._terminal_drop_reasons(global_steps, partition_id)
        return terminal_uids - set().union(*drop_reasons)

    def _has_enough_samples(
        self,
        global_steps: int,
        partition_id: str,
        batch_size: int,
        sampleable_keys: set[str] | None = None,
    ) -> bool:
        # For wait strategy, we need to wait all trajectories that reach threshold to finish
        if self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) >= self.max_off_policy_threshold:
                    return False

        if sampleable_keys is None:
            sampleable_keys = self._sampleable_terminal_keys(global_steps, partition_id)
        return len(sampleable_keys) >= batch_size

    def _drop_terminal_groups(
        self,
        global_steps: int,
        partition_id: str,
        drop_reasons: tuple[set[str], set[str], set[str]] | None = None,
    ) -> tuple[set[str], int, int, dict]:
        """Clear terminal groups rejected by any active policy exactly once."""
        if drop_reasons is None:
            drop_reasons = self._terminal_drop_reasons(global_steps, partition_id)
        stale_uids, dapo_uids, failed_uids = drop_reasons
        dropped_uids = stale_uids | dapo_uids | failed_uids
        if not dropped_uids:
            return set(), 0, 0, {}

        prefix = self._metrics_prefix(partition_id)
        metrics: dict = {}
        if stale_uids:
            prompt_global_steps = self.prompt_global_steps[partition_id]
            spans = np.array(
                [global_steps - prompt_global_steps.get(uid, global_steps) + 1 for uid in stale_uids],
                dtype=float,
            )
            metrics.update(
                {
                    f"{prefix}/off_policy/dropped_samples": len(stale_uids),
                    f"{prefix}/off_policy/dropped_samples_staleness/mean": spans.mean(),
                    f"{prefix}/off_policy/dropped_samples_staleness/max": spans.max(),
                    f"{prefix}/off_policy/dropped_samples_staleness/min": spans.min(),
                }
            )
        if dapo_uids:
            metrics[f"{prefix}/filter_groups/dropped_samples"] = len(dapo_uids)
        if failed_uids:
            metrics[f"{prefix}/rollout_failure/dropped_samples"] = len(failed_uids)

        self._clear_groups(partition_id, dropped_uids)
        return dropped_uids, len(stale_uids), len(dapo_uids), metrics

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
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
        drop_metrics: dict = {}
        selected_prompt_uids: list[str] = []
        partition_snapshot: dict[str, dict] = {}
        prompt_global_steps_snapshot: dict[str, int] = {}

        while True:
            # Drop, gating, and selection below must all use this snapshot.
            self._sync_metadata_from_transfer_queue()

            # Sync DAPO is round-based: do not inspect/select a partial 2k refill, otherwise early
            # completions recreate the long-tail problem that over-generation is intended to avoid.
            if self._sync_dapo_enabled(partition_id) and (
                self.pending_keys[partition_id] or self.running_keys[partition_id]
            ):
                time.sleep(self.poll_interval)
                continue

            drop_reasons = self._terminal_drop_reasons(global_steps, partition_id)
            dropped_uids, stale_count, dapo_count, metrics = self._drop_terminal_groups(
                global_steps, partition_id, drop_reasons
            )
            if dropped_uids:
                _accumulate_drop_metrics(drop_metrics, metrics, stale_count)

                # Async has a buffer and replaces every rejected group exactly. Sync only rejects DAPO
                # groups and deliberately over-generates 2x to avoid a new generation tail.
                refill_count = len(dropped_uids) if self.trainer_mode != "sync" else 2 * dapo_count
                if refill_count > 0 and self.refill_fn is not None:
                    self.refill_fn(refill_count)
                continue

            sampleable_keys = self._sampleable_terminal_keys(global_steps, partition_id, drop_reasons)
            if self._has_enough_samples(global_steps, partition_id, batch_size, sampleable_keys):
                prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
                partition_snapshot = dict(self.partitions[partition_id])
                sampleable_keys = sorted(
                    sampleable_keys,
                    key=lambda key: prompt_global_steps_snapshot.get(key, 0),
                )
                selected_prompt_uids = sampleable_keys[:batch_size]

                # Legacy sync DAPO truncates an over-generated batch. Clearing the equivalent TQ
                # surplus keeps sync mode bufferless instead of carrying extra groups into later steps.
                if self._sync_dapo_enabled(partition_id):
                    surplus_uids = set(sampleable_keys[batch_size:])
                    if surplus_uids:
                        self._clear_groups(partition_id, surplus_uids)
                        key = f"{self._metrics_prefix(partition_id)}/filter_groups/dropped_surplus_samples"
                        drop_metrics[key] = drop_metrics.get(key, 0) + len(surplus_uids)
                break

            time.sleep(self.poll_interval)
            if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                logger.info(
                    f"pending: {len(self.pending_keys[partition_id])}, "
                    f"running: {len(self.running_keys[partition_id])}, "
                    f"finished: {len(self.finished_keys[partition_id])}, "
                    f"failure: {len(self.failure_keys[partition_id])}"
                )
                last_debug_time = time.time()

        if self.trainer_mode != "sync" and self.max_off_policy_strategy == "drop":
            selected_spans = [
                global_steps - prompt_global_steps_snapshot.get(uid, global_steps) + 1 for uid in selected_prompt_uids
            ]
            assert all(span <= self.max_off_policy_threshold for span in selected_spans), (
                f"drop strategy selected stale prompts: spans={selected_spans}, "
                f"threshold={self.max_off_policy_threshold}"
            )

        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        return batch, drop_metrics
