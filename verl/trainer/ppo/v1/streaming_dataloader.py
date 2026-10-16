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

import asyncio
import logging
import math
import os
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.v1 import tq_schema
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.import_utils import load_class_from_fqn, load_extern_type
from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))

DAPO_FILTERED_REWARD_COUNTS_KEY = "_dapo_filtered_reward_counts"
FILTER_GROUPS_REWARD_METRIC = "reward"


def _accumulate_eviction_metrics(acc: dict, new: dict) -> None:
    """Merge one poll iteration's eviction metrics into ``acc`` in place.

    The stale eviction count weights the staleness mean so it stays a true per-sample average across iterations.
    """
    stale_count_key = next((k for k in new if k.endswith("/off_policy/evicted_samples")), None)
    prev_stale_total = acc.get(stale_count_key, 0) if stale_count_key else 0
    stale_count = new[stale_count_key] if stale_count_key else 0

    for key, value in new.items():
        if key.endswith("/evicted_samples_staleness/mean"):
            denom = prev_stale_total + stale_count
            acc[key] = (acc.get(key, 0.0) * prev_stale_total + value * stale_count) / denom if denom else value
        elif key.endswith("/evicted_samples_staleness/max"):
            acc[key] = max(acc.get(key, value), value)
        elif key.endswith("/evicted_samples_staleness/min"):
            acc[key] = min(acc.get(key, value), value)
        elif key == DAPO_FILTERED_REWARD_COUNTS_KEY:
            # Dict-valued diagnostic: merge {metric_value: count} across poll iterations.
            merged = Counter(acc.get(key, {}))
            merged.update(value)
            acc[key] = dict(merged)
        else:
            acc[key] = acc.get(key, 0) + value


class _RequestKind(Enum):
    # Consume ``batch_size`` groups.
    SAMPLE = "sample"
    # Report how many groups are sampleable now without consuming them.
    COUNT = "count"
    # Block until ``target_count`` groups are sampleable without consuming them.
    WAIT = "wait"


@dataclass
class _SampleRequest:
    kind: _RequestKind
    global_steps: int
    partition_id: str
    future: asyncio.Future
    batch_size: int = 0
    target_count: int = 0
    # Sync-only state carried across coordinator passes of one request.
    metrics: dict = field(default_factory=dict)
    refill_credit: int = 0
    draining: bool = False


class StreamingDataloader:
    """Sync rollout data actor: dataset iteration, rollout dispatch, and TransferQueue replay in one place.

    TransferQueue is the replay storage; this actor owns its control plane. Key and tag formats are defined in
    :mod:`verl.trainer.ppo.v1.tq_schema`. The actor registers prompts, AgentLoop writes session markers and
    trajectories, and the trainer reads trajectory fields directly from TransferQueue.

    ### [Trajectories storage format]
    Each trajectory is stored under `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "success", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    AgentLoop stores a metadata-only `{uid}_{session_id}` key with `status=running` when a session starts, then
    updates it to `finished` or `failure` after all trajectory writes settle.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompts with status `finished` or `failure` enter terminal-group handling.

    ### [Coordinator]
    Both modes run one asyncio coordinator that polls TransferQueue metadata, evicts terminal groups, refills
    prompts, and fulfills the pending ``sample`` request. RPC handlers contain no ``await`` between state
    transitions, so they never interleave with a coordinator pass. The modes differ only in refill policy:
    - sync refills only while a request is pending and leaves nothing in flight once it is fulfilled.
    - async keeps a session watermark in flight at all times (see :class:`StreamingDataloaderAsync`).

    ### [Terminal-group eviction/refill matrix]
    ``drop`` means off-policy staleness dropping. Both off-policy strategies (``drop`` and the dropless
    ``wait``, which blocks until stale in-flight prompts finish) are only for async trainers; sync sampling
    is on-policy, so ``max_off_policy_strategy`` is a NO-OP there.
    ``DAPO`` means filtering groups whose configured reward metric is identical across all trajectories,
             for async reward-computation path.
    ``failure`` is the group status described above.
    The matrix applies to the training partition, in which ``k`` is the number of prompts to evict.
    Validation treats all terminal groups as sampleable

    |   trainer mode   |   drop   |               DAPO               |            failure            |
    | ---------------- | -------- | -------------------------------- | ----------------------------- |
    |       sync       |   NO-OP  |    Evict ``k``; refill ``2k``    |  NO-OP or opt-in refill ``k`` |
    |      async       |                      Auto-fill to the target watermark                      |

    In sync mode, DAPO is opt-in and trades generation time for training stability. Each ``k`` evictions add
    ``2k`` logical refill credits, but prompts are fetched only as bounded pending/running slots become available,
    rounded up to whole ``gen_batch_size`` batches (``gen_batch_size`` is 1 or ``train_batch_size``).
    Terminal groups are filtered while other requests remain in flight. Once enough groups are sampleable, inflight
    requests are drained and discarded. By default, failed groups remain sampleable and missing trajectories are
    padded downstream. Setting ``sync_refill_failed_groups=True`` allows refilling failed samples.

    ### [Customization]
    ``trainer.v1.sampler.custom_sampler`` may point to a subclass of the dataloader class for the trainer mode.
    Override ``_terminal_eviction_reasons`` to change which terminal groups are evicted and
    ``_select_prompt_uids`` to change which sampleable groups form a batch.
    """

    # Whether train prompt fields are stored with their prompt keys, e.g. for reissue after restore.
    _persist_prompt_fields = False

    def __init__(
        self,
        *,
        config,
        tokenizer,
        processor,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        filter_groups_metric: str | None = None,
    ):
        tq.init()
        SkipManager.init(config)
        self.config = config
        self.trainer_mode = config.trainer.v1.trainer_mode
        self.global_steps = 0

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        self.gen_batch_size = resolve_gen_batch_size(config)
        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.gen_batch_size,
            num_workers=config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(config.data, train_dataset),
        )
        self.train_dataloader_it = None

        manager_class_fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_cls = (
            load_class_from_fqn(manager_class_fqn, "AgentLoopManager") if manager_class_fqn else AgentLoopManagerTQ
        )
        self.agent_loop_manager = manager_cls.create(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        sampler_config = config.trainer.v1.sampler
        self._init_replay_state(
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            filter_groups_metric=filter_groups_metric,
            poll_interval=sampler_config.poll_interval,
        )
        self._init_refill_policy()

    def _init_replay_state(
        self,
        *,
        max_off_policy_threshold: int | None,
        max_off_policy_strategy: str,
        sampler_kwargs,
        poll_interval: float,
        filter_groups_metric: str | None = None,
    ) -> None:
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.filter_groups_metric = filter_groups_metric
        self.poll_interval = poll_interval
        if self.max_off_policy_threshold is not None:
            assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
                f"Invalid max off policy threshold: {self.max_off_policy_threshold}, "
                "must be an integer greater than 0, or null to disable off-policy version control"
            )
        else:
            logger.info(
                "[StreamingDataloader] off-policy version control disabled (max_off_policy_threshold is null): "
                "no samples will be dropped or awaited for staleness"
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
        self.running_session_keys: dict[str, set[str]] = defaultdict(set)
        self.finished_session_keys: dict[str, set[str]] = defaultdict(set)
        self.failure_session_keys: dict[str, set[str]] = defaultdict(set)
        # partition_id => {prompt_key: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)
        # Finished groups are immutable, so their DAPO classification can be reused across polling iterations.
        self._dapo_classification_cache: dict[str, dict[str, float | None]] = defaultdict(dict)

        self._pending_sample: _SampleRequest | None = None
        # Wake local requests immediately. If sample batch is not ready at once, wait for next poll.
        self._wakeup = asyncio.Event()
        self._coordinator_task: asyncio.Task | None = None
        self._last_debug_time = time.time()

    def _init_refill_policy(self) -> None:
        sampler_config = self.config.trainer.v1.sampler
        filter_groups = self.config.algorithm.get("filter_groups", None)
        self._set_sync_refill_policy(
            train_batch_size=self.config.data.train_batch_size,
            max_inflight_gen_batches=(
                filter_groups.get("max_inflight_gen_batches", 1) if self.filter_groups_metric is not None else 1
            ),
            sync_refill_failed_groups=bool(sampler_config.get("sync_refill_failed_groups", False)),
        )

    def _set_sync_refill_policy(
        self, *, train_batch_size: int, max_inflight_gen_batches: int, sync_refill_failed_groups: bool
    ) -> None:
        self.train_batch_size = train_batch_size
        self.max_inflight_gen_batches = max_inflight_gen_batches
        self.sync_refill_failed_groups = sync_refill_failed_groups
        if self.filter_groups_metric is not None:
            if not isinstance(self.max_inflight_gen_batches, int) or self.max_inflight_gen_batches <= 0:
                raise ValueError("max_inflight_gen_batches must be a positive integer")

    # ------------------------------------------------------------------ #
    # Trainer-facing RPCs
    # ------------------------------------------------------------------ #

    async def sample(
        self,
        global_steps: int,
        batch_size: int,
        partition_id: str = "train",
        prompts: TensorDict | None = None,
    ) -> tuple[Any, dict, int]:
        """Consume ``batch_size`` prompt groups, returning ``(batch, eviction_metrics, len(batch))``.

        ``prompts`` are dispatched before waiting, e.g. for a validation batch.
        """
        return await self._submit_request(
            _RequestKind.SAMPLE, global_steps, partition_id, batch_size=batch_size, prompts=prompts
        )

    async def _submit_request(
        self,
        kind: _RequestKind,
        global_steps: int,
        partition_id: str,
        *,
        batch_size: int = 0,
        target_count: int = 0,
        prompts: TensorDict | None = None,
    ) -> tuple[Any, dict, int]:
        """Register one request and wait until the coordinator fulfills it."""
        self._ensure_coordinator()
        if self._pending_sample is not None:
            raise RuntimeError("Concurrent sample requests are not supported")
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        self._on_sample_request(batch_size, partition_id, prompts)
        future = asyncio.get_running_loop().create_future()
        self._pending_sample = _SampleRequest(
            kind=kind,
            global_steps=global_steps,
            partition_id=partition_id,
            future=future,
            batch_size=batch_size,
            target_count=target_count,
        )
        self._wakeup.set()
        return await future

    def _on_sample_request(self, batch_size: int, partition_id: str, prompts: TensorDict | None) -> None:
        if prompts is not None:
            self._submit_batch_to_rollout(prompts, partition_id)
        elif partition_id == "train":
            self._refill_exact(batch_size)

    def save_checkpoint(self, local_global_step_folder: str, global_steps: int) -> None:
        """Persist the dataloader cursor."""
        del global_steps
        os.makedirs(local_global_step_folder, exist_ok=True)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))

    def load_checkpoint(self, local_global_step_folder: str | None, global_steps: int) -> dict:
        """Restore the dataloader cursor."""
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        if local_global_step_folder is not None:
            dataloader_path = os.path.join(local_global_step_folder, "data.pt")
            if os.path.exists(dataloader_path):
                self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
            else:
                logger.warning("No dataloader checkpoint found at %s", dataloader_path)
        self.train_dataloader_it = None
        return {}

    # ------------------------------------------------------------------ #
    # Coordinator
    # ------------------------------------------------------------------ #

    def _ensure_coordinator(self) -> None:
        if self._coordinator_task is None or self._coordinator_task.done():
            self._wakeup = asyncio.Event()
            self._coordinator_task = asyncio.create_task(
                self._coordinator_loop(),
                name="streaming-dataloader-coordinator",
            )
            self._coordinator_task.add_done_callback(self._on_coordinator_done)

    def _on_coordinator_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception() or RuntimeError("Streaming dataloader coordinator exited unexpectedly")
        request, self._pending_sample = self._pending_sample, None
        if request is not None and not request.future.done():
            request.future.set_exception(error)
        logger.critical(
            "Streaming dataloader coordinator failed; terminating the actor",
            exc_info=(type(error), error, error.__traceback__),
        )
        # The coordinator exclusively advances replay state, so the actor cannot recover without it.
        ray.actor.exit_actor()

    async def _coordinator_loop(self) -> None:
        while True:
            # Clear before doing work so a concurrent producer cannot have its wakeup erased.
            self._wakeup.clear()
            self._coordinator_step()
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    def _coordinator_step(self) -> None:
        request = self._pending_sample
        if request is None:
            return
        try:
            result = self._advance_request(request)
        except BaseException as error:
            self._pending_sample = None
            if not request.future.done():
                request.future.set_exception(error)
            return
        if result is not None:
            self._pending_sample = None
            if not request.future.done():
                request.future.set_result(result)

    def _advance_request(self, request: _SampleRequest) -> tuple[KVBatchMeta, dict, int] | None:
        """Run one sync polling pass; return the result once the request can be fulfilled.

        Refill dispatches wake the coordinator immediately so newly submitted prompts are observed without
        waiting for the poll interval.
        """
        global_steps, partition_id, batch_size = request.global_steps, request.partition_id, request.batch_size
        dapo_enabled = partition_id != "val" and self.filter_groups_metric is not None

        # Eviction, gating, and selection below must all use this snapshot.
        self._sync_metadata_from_transfer_queue()
        eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
        failed_count = len(eviction_reasons[2])
        evicted_uids, _stale_count, dapo_count, metrics = self._evict_terminal_groups(
            global_steps, partition_id, eviction_reasons
        )
        if evicted_uids:
            _accumulate_eviction_metrics(request.metrics, metrics)

        sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
        has_enough_samples = len(sampleable_keys) >= batch_size
        inflight_count = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])

        if not dapo_enabled and failed_count > 0 and not has_enough_samples:
            self._refill_exact(failed_count)
            self._wakeup.set()
            return None

        if dapo_enabled:
            if has_enough_samples:
                # Stop speculative dispatch, then drain requests already running under this policy version.
                request.draining = True
                request.refill_credit = 0
            elif not request.draining:
                request.refill_credit += 2 * dapo_count + failed_count

            if not request.draining and request.refill_credit > 0:
                max_inflight_prompts = self.max_inflight_gen_batches * self.train_batch_size
                available_slots = max(0, max_inflight_prompts - inflight_count)
                # Round up: rounding credit down stalls when credit < gen_batch_size and nothing is in flight.
                dispatch_count = min(
                    math.ceil(request.refill_credit / self.gen_batch_size) * self.gen_batch_size,
                    available_slots - available_slots % self.gen_batch_size,
                )
                if dispatch_count > 0:
                    self._refill_exact(dispatch_count)
                    request.refill_credit -= dispatch_count
                    self._wakeup.set()
                    return None

        if not (has_enough_samples and (not dapo_enabled or inflight_count == 0)):
            self._maybe_log_progress(partition_id)
            return None

        selected_prompt_uids, partition_snapshot, _prompt_global_steps_snapshot = self._select_prompt_uids(
            partition_id, sampleable_keys, batch_size
        )
        # Sync remains bufferless: all speculative requests are drained, then surplus is discarded.
        if dapo_enabled:
            surplus_uids = sampleable_keys - set(selected_prompt_uids)
            if surplus_uids:
                self._clear_groups(partition_id, surplus_uids)
                key = f"{self._metrics_prefix(partition_id)}/filter_groups/discarded_surplus_samples"
                request.metrics[key] = request.metrics.get(key, 0) + len(surplus_uids)

        selected_uids = set(selected_prompt_uids)
        if partition_id != "val" and not any(tq_schema.uid_of(key) in selected_uids for key in partition_snapshot):
            message = "Sync sampling selected terminal groups with no materializable trajectories."
            if not self.sync_refill_failed_groups:
                message += " Enable trainer.v1.sampler.sync_refill_failed_groups to replace failed groups."
            raise RuntimeError(message)
        batch = self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)
        return batch, request.metrics, len(batch)

    def _maybe_log_progress(self, partition_id: str) -> None:
        now = time.time()
        if now - self._last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
            logger.info(
                f"pending: {len(self.pending_keys[partition_id])}, "
                f"running: {len(self.running_keys[partition_id])}, "
                f"finished: {len(self.finished_keys[partition_id])}, "
                f"failure: {len(self.failure_keys[partition_id])}"
            )
            self._last_debug_time = now

    # ------------------------------------------------------------------ #
    # Prompt source
    # ------------------------------------------------------------------ #

    def _fetch_one_gen_batch(self) -> TensorDict:
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int) -> TensorDict:
        if num_prompts <= 0 or num_prompts % self.gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size ({self.gen_batch_size})"
            )
        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // self.gen_batch_size)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _default_rollout_n(self, partition_id: str) -> int:
        rollout_config = self.config.actor_rollout_ref.rollout
        return int(rollout_config.n if partition_id == "train" else rollout_config.val_kwargs.n)

    def _rollout_n_values(self, batch: TensorDict, partition_id: str) -> list[int]:
        if "__rollout_n__" not in batch:
            return [self._default_rollout_n(partition_id)] * len(batch)

        values = batch["__rollout_n__"]
        if isinstance(values, NonTensorData):
            values = [values.data] * len(batch)
        rollout_ns = []
        for i in range(len(batch)):
            value = values[i]
            if isinstance(value, NonTensorData):
                value = value.data
            if isinstance(value, torch.Tensor):
                value = value.item()
            rollout_n = int(value)
            if rollout_n <= 0:
                raise ValueError(f"rollout_n must be positive, got {rollout_n}")
            rollout_ns.append(rollout_n)
        return rollout_ns

    def _submit_batch_to_rollout(self, batch: TensorDict, partition_id: str) -> int:
        self._register_prompts(batch, partition_id)
        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _refill_exact(self, num_prompts: int) -> int:
        if num_prompts <= 0:
            return 0
        return self._submit_batch_to_rollout(self._next_train_batch(num_prompts), "train")

    # ------------------------------------------------------------------ #
    # TransferQueue control plane
    # ------------------------------------------------------------------ #

    def _register_prompts(self, batch: TensorDict, partition_id: str) -> None:
        """Register freshly sampled prompts as pending groups before they are dispatched to rollout."""
        keys = list(batch["uid"])
        tags = [tq_schema.prompt_tag(tq_schema.PROMPT_PENDING, self.global_steps) for _ in keys]
        if partition_id == "train" and self._persist_prompt_fields:
            fields = batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)])
            tq.kv_batch_put(keys=keys, partition_id=partition_id, tags=tags, fields=fields)
        else:
            tq.kv_batch_put(keys=keys, partition_id=partition_id, tags=tags)

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue."""
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.running_session_keys.clear()
        self.finished_session_keys.clear()
        self.failure_session_keys.clear()
        self.prompt_global_steps.clear()

        data = tq.kv_list()
        if data is None:
            return

        prompt_status_keys = {
            tq_schema.PROMPT_PENDING: self.pending_keys,
            tq_schema.PROMPT_RUNNING: self.running_keys,
            tq_schema.PROMPT_FINISHED: self.finished_keys,
            tq_schema.PROMPT_FAILURE: self.failure_keys,
        }
        session_status_keys = {
            tq_schema.SESSION_RUNNING: self.running_session_keys,
            tq_schema.SESSION_FINISHED: self.finished_session_keys,
            tq_schema.SESSION_FAILURE: self.failure_session_keys,
        }
        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tq_schema.is_prompt_tag(tag):
                    # see: [GRPO group sampling control]
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    if tag["status"] not in prompt_status_keys:
                        raise ValueError(f"Unknown status: {tag['status']}")
                    prompt_status_keys[tag["status"]][partition_id].add(key)
                elif tq_schema.is_session_tag(tag):
                    if tag["status"] not in session_status_keys:
                        raise ValueError(f"Unknown session status: {tag['status']}")
                    session_status_keys[tag["status"]][partition_id].add(key)
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    @staticmethod
    def _metrics_prefix(partition_id: str) -> str:
        return "training" if partition_id == "train" else "validation"

    def _clear_groups(self, partition_id: str, uids: set[str]) -> None:
        """Remove prompt groups from TransferQueue and the active metadata snapshot."""
        if not uids:
            return

        trajectory_keys = {key for key in self.partitions[partition_id] if tq_schema.uid_of(key) in uids}
        session_status_sets = (
            self.running_session_keys,
            self.finished_session_keys,
            self.failure_session_keys,
        )
        session_keys = {
            key
            for status_keys in session_status_sets
            for key in status_keys[partition_id]
            if tq_schema.uid_of(key) in uids
        }
        tq.kv_clear(
            partition_id=partition_id,
            keys=[*uids, *trajectory_keys, *session_keys],
        )

        # Keep same-poll decisions consistent with tq.
        for key in trajectory_keys:
            del self.partitions[partition_id][key]
        for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
            status_keys[partition_id].difference_update(uids)
        for status_keys in session_status_sets:
            status_keys[partition_id].difference_update(session_keys)
        for uid in uids:
            self.prompt_global_steps[partition_id].pop(uid, None)
            self._dapo_classification_cache[partition_id].pop(uid, None)

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        """Finished groups whose configured DAPO metric is identical across all trajectories.

        Returns the filtered uids and a ``{shared_metric_value: group_count}`` breakdown built in the
        same scope, so the diagnostic (which reward level the no-signal groups collapse to) travels
        with the uids through the return value instead of via hidden instance state.
        """
        if partition_id == "val" or self.filter_groups_metric is None:
            return set(), Counter()

        finished_uids = self.finished_keys[partition_id]
        classification_cache = self._dapo_classification_cache[partition_id]
        for uid in classification_cache.keys() - finished_uids:
            del classification_cache[uid]

        new_finished_uids = finished_uids - classification_cache.keys()
        trajectory_keys = [key for key in self.partitions[partition_id] if tq_schema.uid_of(key) in new_finished_uids]
        use_canonical_reward = self.filter_groups_metric == FILTER_GROUPS_REWARD_METRIC
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        missing_metric_uids = new_finished_uids - {tq_schema.uid_of(key) for key in trajectory_keys}

        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["rm_scores"] if use_canonical_reward else ["extra_fields"],
            )
            metric_data = list(data["rm_scores"] if use_canonical_reward else data["extra_fields"])
        else:
            metric_data = []

        for key, value in zip(trajectory_keys, metric_data, strict=True):
            uid = tq_schema.uid_of(key)
            if use_canonical_reward:
                metrics_by_uid[uid].append(float(value.sum().item()))
            else:
                extra_fields = getattr(value, "data", value)
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

        for uid in new_finished_uids:
            values = metrics_by_uid[uid]
            classification_cache[uid] = float(values[0]) if len(values) > 1 and float(np.std(values)) == 0.0 else None

        filtered_rewards = {uid: reward for uid, reward in classification_cache.items() if reward is not None}
        return set(filtered_rewards), Counter(filtered_rewards.values())

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        """Return stale, DAPO-filtered, and failed groups (plus the DAPO value->count breakdown).

        The three sets may overlap. Callers clear and refill their union, so one prompt is never handled
        twice. ``dapo_counts`` is the {shared_metric_value: group_count} diagnostic for ``dapo_uids``; it
        rides along in the return value so no hidden state is needed between production and consumption.
        """
        if partition_id == "val":
            return set(), set(), set(), Counter()

        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        failed_uids = set()
        if self.sync_refill_failed_groups:
            materializable_uids = {tq_schema.uid_of(key) for key in self.partitions[partition_id]}
            failed_uids = self.failure_keys[partition_id] - materializable_uids
        return set(), dapo_uids, failed_uids, dapo_counts

    def _sampleable_terminal_keys(
        self,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> set[str]:
        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        stale_uids, dapo_uids, failed_uids, _dapo_counts = eviction_reasons
        return terminal_uids - (stale_uids | dapo_uids | failed_uids)

    def _evict_terminal_groups(
        self,
        global_steps: int,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> tuple[set[str], int, int, dict]:
        """Evict terminal groups selected by any active policy exactly once."""
        stale_uids, dapo_uids, failed_uids, dapo_counts = eviction_reasons
        evicted_uids = stale_uids | dapo_uids | failed_uids
        if not evicted_uids:
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
                    f"{prefix}/off_policy/evicted_samples": len(stale_uids),
                    f"{prefix}/off_policy/evicted_samples_staleness/mean": spans.mean(),
                    f"{prefix}/off_policy/evicted_samples_staleness/max": spans.max(),
                    f"{prefix}/off_policy/evicted_samples_staleness/min": spans.min(),
                }
            )
        if dapo_uids:
            metrics[f"{prefix}/filter_groups/evicted_samples"] = len(dapo_uids)
            # Non-scalar diagnostic: how many filtered (no-signal) groups collapsed to each metric value.
            metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] = dict(dapo_counts)
        if failed_uids:
            metrics[f"{prefix}/rollout_failure/evicted_samples"] = len(failed_uids)

        self._clear_groups(partition_id, evicted_uids)
        return evicted_uids, len(stale_uids), len(dapo_uids), metrics

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
        partition_snapshot = dict(self.partitions[partition_id])
        ordered_keys = sorted(
            sampleable_keys,
            key=lambda key: prompt_global_steps_snapshot.get(key, 0),
        )
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot

    def _materialize_batch(
        self, partition_id: str, selected_prompt_uids: list[str], partition_snapshot: dict[str, dict]
    ) -> KVBatchMeta:
        selected = set(selected_prompt_uids)
        # Trajectory keys are cleared by the trainer after training; session markers have no other owner.
        session_keys = [
            key
            for status_keys in (self.running_session_keys, self.finished_session_keys, self.failure_session_keys)
            for key in status_keys[partition_id]
            if tq_schema.uid_of(key) in selected
        ]
        tq.kv_clear(partition_id=partition_id, keys=[*selected_prompt_uids, *session_keys])
        # Keep same-poll decisions consistent with tq, e.g. the outstanding-group cap checked by async refill.
        for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
            status_keys[partition_id].difference_update(selected)
        for status_keys in (self.running_session_keys, self.finished_session_keys, self.failure_session_keys):
            status_keys[partition_id].difference_update(session_keys)

        keys, tags = [], []
        for key, tag in partition_snapshot.items():
            if tq_schema.uid_of(key) in selected:
                keys.append(key)
                tags.append(tag)
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)


class StreamingDataloaderAsync(StreamingDataloader):
    """Async rollout data actor: keeps a session watermark in flight and serves requests without blocking.

    It also owns in-flight recovery: TransferQueue checkpointing and reissuing prompts that were pending or
    running when the checkpoint was taken. Custom subclasses may additionally override ``poll_once`` and
    ``materialize_sampleable``.
    """

    # Restore reissues in-flight prompts from TransferQueue, so their fields must be stored there.
    _persist_prompt_fields = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validation may start the coordinator before load_checkpoint establishes train state.
        self._train_state_ready = False
        self._validation_active = False
        self._pending_metrics: dict = {}

    def _init_refill_policy(self) -> None:
        # Refill keeps prompts in flight while a request waits, and every in-flight prompt has staleness >= 1,
        # so ``wait`` with threshold 1 would block sampling forever.
        if self.max_off_policy_strategy == "wait" and self.max_off_policy_threshold == 1:
            raise ValueError(
                "max_off_policy_strategy=wait requires max_off_policy_threshold >= 2 in async trainers; "
                "use the drop strategy or the sync trainer for strictly on-policy sampling"
            )
        mode_config = self.config.trainer.v1.get(self.trainer_mode, {})
        train_batch_size = self.config.data.train_batch_size
        watermark_batches = float(mode_config.get("watermark_batches", 2))
        if watermark_batches <= 0:
            raise ValueError(f"watermark_batches must be positive, got {watermark_batches}")
        self.default_rollout_n = self._default_rollout_n("train")
        self.watermark_sessions = round(watermark_batches * train_batch_size * self.default_rollout_n)
        if self.watermark_sessions <= 0:
            raise ValueError(
                "watermark_batches resolves to zero in-flight sessions; increase it for the configured "
                f"train_batch_size={train_batch_size} and rollout.n={self.default_rollout_n}"
            )

        # Colocated rollout runs only while the trainer blocks in sample, so unconsumed groups stay below two
        # watermarks and the cap has nothing to bound.
        if self.trainer_mode != "separate_async":
            self.max_outstanding_groups = None
            return
        # When rollout outpaces training, unconsumed groups pile up and their staleness grows with the pile.
        max_outstanding_batches = mode_config.get("max_outstanding_batches", 4)
        if max_outstanding_batches is None:
            self.max_outstanding_groups = None
            return
        max_outstanding_batches = float(max_outstanding_batches)
        if max_outstanding_batches < watermark_batches:
            raise ValueError(
                f"max_outstanding_batches ({max_outstanding_batches}) must be >= watermark_batches "
                f"({watermark_batches}), otherwise the in-flight watermark is never reached"
            )
        self.max_outstanding_groups = round(max_outstanding_batches * train_batch_size)
        if self.max_off_policy_threshold is not None and max_outstanding_batches + 1 > self.max_off_policy_threshold:
            logger.warning(
                f"max_outstanding_batches={max_outstanding_batches} lets steady-state staleness reach "
                f"max_off_policy_threshold={self.max_off_policy_threshold} when rollout outpaces training; "
                "expect off-policy drops or waits"
            )

    # ------------------------------------------------------------------ #
    # Trainer-facing RPCs
    # ------------------------------------------------------------------ #

    async def count_sampleable(self, global_steps: int) -> int:
        """Return how many train groups are sampleable after this pass's evictions, consuming none."""
        _, _, count = await self._submit_request(_RequestKind.COUNT, global_steps, "train")
        return count

    async def wait_sampleable(self, global_steps: int, target_count: int) -> dict:
        """Block until ``target_count`` train groups are sampleable, consuming none.

        Returns the eviction metrics accrued since the previous metrics-draining request.
        """
        _, metrics, _ = await self._submit_request(_RequestKind.WAIT, global_steps, "train", target_count=target_count)
        return metrics

    def _on_sample_request(self, batch_size: int, partition_id: str, prompts: TensorDict | None) -> None:
        self._validation_active = partition_id == "val"
        if prompts is not None:
            # only for validation
            self._submit_batch_to_rollout(prompts, partition_id)

    def save_checkpoint(self, local_global_step_folder: str, global_steps: int) -> None:
        super().save_checkpoint(local_global_step_folder, global_steps)
        self._save_tq_checkpoint(os.path.join(local_global_step_folder, "transfer_queue"), global_steps)

    def load_checkpoint(self, local_global_step_folder: str | None, global_steps: int) -> dict:
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        if local_global_step_folder is not None:
            dataloader_path = os.path.join(local_global_step_folder, "data.pt")
            if not os.path.exists(dataloader_path):
                raise RuntimeError(f"StreamingDataloader restore requires {dataloader_path}")
            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
            self._load_tq_checkpoint(os.path.join(local_global_step_folder, "transfer_queue"))
        self.train_dataloader_it = None
        reissued = self._reissue_inflight_prompts()
        self._train_state_ready = True
        self._validation_active = False
        self._ensure_coordinator()
        self._wakeup.set()
        return {"streaming_dataloader/reissued_groups": float(reissued)}

    # ------------------------------------------------------------------ #
    # Coordinator
    # ------------------------------------------------------------------ #

    def _coordinator_step(self) -> None:
        if self._advance_once():
            # poll_once already refreshed replay metadata for this coordinator pass.
            self._refill_to_watermark()

    def _advance_once(self) -> bool:
        request = self._pending_sample
        if request is not None:
            self._serve_request(request)
            return self._train_state_ready and request.partition_id == "train"
        if self._train_state_ready and not self._validation_active:
            self._serve_request()
            return True
        return False

    def _serve_request(self, request: _SampleRequest | None = None) -> None:
        if request is None:
            global_steps = self.global_steps
            partition_id = "train"
            target_count = 0
        else:
            global_steps = request.global_steps
            partition_id = request.partition_id
            target_count = {
                _RequestKind.SAMPLE: request.batch_size,
                _RequestKind.WAIT: request.target_count,
                _RequestKind.COUNT: 0,
            }[request.kind]

        sampleable, metrics, ready = self.poll_once(global_steps, partition_id, target_count)
        if partition_id == "train":
            _accumulate_eviction_metrics(self._pending_metrics, metrics)
        if request is None:
            return
        if request.kind is _RequestKind.COUNT:
            ready = True
        if not ready:
            return

        try:
            if request.kind is _RequestKind.SAMPLE:
                batch = self.materialize_sampleable(
                    request.global_steps,
                    request.partition_id,
                    sampleable,
                    request.batch_size,
                )
                result_count = len(batch)
            else:
                batch = None
                result_count = len(sampleable)
            if request.kind is _RequestKind.COUNT:
                # Counts carry no metrics, so keep accrued evictions for the next sample/wait request.
                result_metrics = {}
            elif request.partition_id == "train":
                result_metrics = dict(self._pending_metrics)
                self._pending_metrics.clear()
            else:
                result_metrics = metrics
            result = (batch, result_metrics, result_count)
        except BaseException as error:
            self._pending_sample = None
            if not request.future.done():
                request.future.set_exception(error)
            return

        self._pending_sample = None
        if not request.future.done():
            request.future.set_result(result)

    def poll_once(
        self,
        global_steps: int,
        partition_id: str,
        target_count: int,
    ) -> tuple[set[str], dict, bool]:
        """Run one metadata/eviction/readiness pass without sleeping."""
        self._sync_metadata_from_transfer_queue()
        eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
        _evicted_uids, _stale_count, _dapo_count, metrics = self._evict_terminal_groups(
            global_steps, partition_id, eviction_reasons
        )
        sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
        ready = self._has_enough_samples(global_steps, partition_id, target_count, sampleable_keys)
        return sampleable_keys, metrics, ready

    def materialize_sampleable(
        self,
        global_steps: int,
        partition_id: str,
        sampleable_keys: set[str],
        batch_size: int,
    ) -> KVBatchMeta:
        """Materialize a batch from a sampleable set produced by the latest poll."""
        selected_prompt_uids, partition_snapshot, prompt_global_steps_snapshot = self._select_prompt_uids(
            partition_id, sampleable_keys, batch_size
        )
        if (
            partition_id != "val"
            and self.max_off_policy_threshold is not None
            and self.max_off_policy_strategy == "drop"
        ):
            selected_spans = [
                global_steps - prompt_global_steps_snapshot.get(uid, global_steps) + 1 for uid in selected_prompt_uids
            ]
            assert all(span <= self.max_off_policy_threshold for span in selected_spans), (
                f"drop strategy selected stale prompts: spans={selected_spans}, "
                f"threshold={self.max_off_policy_threshold}"
            )
        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)

    # ------------------------------------------------------------------ #
    # Async refill and eviction policy
    # ------------------------------------------------------------------ #

    def _submit_batch_to_rollout(self, batch: TensorDict, partition_id: str) -> int:
        # TODO: support custom `__rollout_n__`
        if partition_id == "train":
            rollout_ns = self._rollout_n_values(batch, partition_id)
            if any(rollout_n != self.default_rollout_n for rollout_n in rollout_ns):
                raise ValueError("Async trainers require a fixed train rollout_n")
        return super()._submit_batch_to_rollout(batch, partition_id)

    def _refill_to_watermark(self) -> int:
        inflight_sessions = self.unfinished_session_count("train", self.default_rollout_n)
        free_slots = max(0, self.watermark_sessions - inflight_sessions)
        num_prompts = free_slots // self.default_rollout_n
        if self.max_outstanding_groups is not None:
            num_prompts = min(num_prompts, max(0, self.max_outstanding_groups - self.outstanding_group_count("train")))
        return self._refill_exact(num_prompts) if num_prompts > 0 else 0

    def outstanding_group_count(self, partition_id: str) -> int:
        """Count prompt groups dispatched but not yet consumed: in flight or terminal and awaiting sampling."""
        return sum(
            len(status_keys[partition_id])
            for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys)
        )

    def unfinished_session_count(self, partition_id: str, default_rollout_n: int) -> int:
        """Count session slots still occupied by pending or running prompt groups."""
        if default_rollout_n <= 0:
            raise RuntimeError(f"Invalid rollout_n={default_rollout_n}")
        terminal_session_keys = self.finished_session_keys[partition_id] | self.failure_session_keys[partition_id]
        completed_by_uid = Counter(tq_schema.uid_of(key) for key in terminal_session_keys)

        total = len(self.pending_keys[partition_id]) * default_rollout_n

        for uid in self.running_keys[partition_id]:
            completed = completed_by_uid[uid]
            if completed > default_rollout_n:
                raise RuntimeError(
                    f"Prompt {uid} has inconsistent session progress: "
                    f"rollout_n={default_rollout_n}, completed={completed}"
                )
            total += default_rollout_n - completed
        return total

    def _stale_terminal_keys(self, global_steps: int, partition_id: str) -> set[str]:
        if partition_id == "val" or self.max_off_policy_threshold is None or self.max_off_policy_strategy != "drop":
            return set()
        prompt_global_steps = self.prompt_global_steps[partition_id]
        terminal_keys = self.finished_keys[partition_id]
        return {
            uid
            for uid in terminal_keys
            if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
        }

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        if partition_id == "val":
            return set(), set(), set(), Counter()

        stale_uids = self._stale_terminal_keys(global_steps, partition_id)
        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        return stale_uids, dapo_uids, set(self.failure_keys[partition_id]), dapo_counts

    def _has_enough_samples(
        self,
        global_steps: int,
        partition_id: str,
        batch_size: int,
        sampleable_keys: set[str],
    ) -> bool:
        # Dropless off-policy control: block sampling while any in-flight prompt has reached the staleness
        # threshold, so it can finish and be trained on instead of dropped.
        if self.max_off_policy_threshold is not None and self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) >= self.max_off_policy_threshold:
                    return False

        return len(sampleable_keys) >= batch_size

    # ------------------------------------------------------------------ #
    # In-flight recovery
    # ------------------------------------------------------------------ #

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Restart pending/running prompts from their stored prompt fields; return how many were reissued.

        Partial trajectories and session markers of those prompts are discarded, so every reissued group
        restarts all of its sessions from scratch.
        """
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tq_schema.is_prompt_tag(tag) and tag.get("status") in tq_schema.PROMPT_INFLIGHT_STATUSES
        ]
        if not inflight_uids:
            return 0

        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        if "__rollout_n__" in batch:
            batch.pop("__rollout_n__")
        inflight_uid_set = set(inflight_uids)
        stale_keys = [
            key
            for key, tag in items.items()
            if not tq_schema.is_prompt_tag(tag) and tq_schema.uid_of(key) in inflight_uid_set
        ]
        if stale_keys:
            tq.kv_clear(keys=stale_keys, partition_id=partition_id)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [tq_schema.prompt_tag(tq_schema.PROMPT_PENDING, self.global_steps) for _ in inflight_uids]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        return len(inflight_uids)

    def _save_tq_checkpoint(self, path: str, global_steps: int) -> None:
        tq.save_checkpoint(path, metadata={"global_steps": global_steps})

    def _load_tq_checkpoint(self, path: str) -> bool:
        """Restore TransferQueue state; return ``False`` when there is nothing restorable."""
        if not os.path.exists(path):
            logger.warning(
                f"Skipping TransferQueue restore because {path} does not exist; "
                "in-flight rollout state will not be recovered"
            )
            return False
        tq.load_checkpoint(path)
        return True


def resolve_gen_batch_size(config) -> int:
    """Resolve ``data.gen_batch_size``, the number of prompts per train dataloader fetch.

    Async watermark refill and sync DAPO/failure refill dispatch exact prompt counts, so they fetch one prompt at a
    time and other configured values are overridden to 1. Plain sync sampling dispatches whole train batches.
    """
    configured = config.data.get("gen_batch_size", None)
    train_batch_size = config.data.train_batch_size
    filter_groups = config.algorithm.get("filter_groups", None)
    dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
    sync_refill_failed_groups = bool(config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
    if config.trainer.v1.trainer_mode != "sync" or dapo_enabled or sync_refill_failed_groups:
        if configured not in (None, 1):
            logger.warning(f"data.gen_batch_size={configured} is overridden to 1.")
        return 1
    if configured is None:
        return int(train_batch_size)
    if configured not in (1, train_batch_size):
        raise ValueError(
            f"Sync trainers require data.gen_batch_size to be 1 or train_batch_size ({train_batch_size}), "
            f"got {configured}"
        )
    return int(configured)


def resolve_streaming_dataloader_cls(config) -> type[StreamingDataloader]:
    """Return the dataloader class for the trainer mode, honoring ``trainer.v1.sampler.custom_sampler``."""
    is_sync = config.trainer.v1.trainer_mode == "sync"
    base_cls = StreamingDataloader if is_sync else StreamingDataloaderAsync
    custom_sampler = config.trainer.v1.sampler.get("custom_sampler", None)
    if custom_sampler is None or not (custom_sampler.get("path") and custom_sampler.get("name")):
        return base_cls
    cls = load_extern_type(custom_sampler.path, custom_sampler.name)
    mismatched_mode = is_sync and isinstance(cls, type) and issubclass(cls, StreamingDataloaderAsync)
    if not (isinstance(cls, type) and issubclass(cls, base_cls)) or mismatched_mode:
        raise TypeError(
            f"custom_sampler {custom_sampler.name!r} must subclass {base_cls.__name__} "
            f"for trainer_mode={config.trainer.v1.trainer_mode!r}"
        )
    return cls


def create_streaming_dataloader_actor_cls(config):
    """Wrap the resolved dataloader class as a single-concurrency Ray actor."""
    return ray.remote(max_concurrency=1)(resolve_streaming_dataloader_cls(config))


StreamingDataloaderActor = ray.remote(max_concurrency=1)(StreamingDataloader)
StreamingDataloaderAsyncActor = ray.remote(max_concurrency=1)(StreamingDataloaderAsync)
