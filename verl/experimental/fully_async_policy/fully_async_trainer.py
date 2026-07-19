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

import asyncio
import logging
import os
import time
from datetime import datetime

import torch
from typing import Any

import numpy as np
import ray
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.teacher_routing import ExclusiveTeacherScheduler
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)


def _select_teacher_logprobs_for_sample(rollout_sample, teacher_keys, routing_field):
    """nonrouter: pick this sample's data_source-matched teacher log_prob and write it
    to the canonical `teacher_logprobs` field of shape (sample_N, seqlen, 1).

    Two storage conventions:
      - separate mode: `teacher_logprobs` holds the multi tensor (sample_N, seqlen, T).
      - fused mode: per-teacher keys `__teacher_lp_{idx}` each (sample_N, seqlen, 1).

    Each RolloutSample carries a single data_source value (asserted by the existing
    routing logic), so a single trailing-dim gather over the whole sample is correct.
    """
    values = rollout_sample.full_batch.non_tensor_batch.get(routing_field)
    if values is None:
        raise ValueError(
            f"nonrouter selection requires non_tensor_batch[{routing_field!r}], "
            f"but sample {rollout_sample.sample_id!r} does not contain it."
        )
    normalized = []
    for v in np.asarray(values, dtype=object).reshape(-1):
        normalized.append(v.item() if hasattr(v, "item") else v)
    unique = set(normalized)
    if len(unique) != 1:
        raise ValueError(
            f"A RolloutSample must route all responses to one teacher, but sample "
            f"{rollout_sample.sample_id!r} has {routing_field} values {sorted(unique)!r}."
        )
    ds = next(iter(unique))
    if ds not in teacher_keys:
        raise ValueError(
            f"data_source {ds!r} not in configured teacher_keys {list(teacher_keys)}."
        )
    teacher_idx = list(teacher_keys).index(ds)

    batch = rollout_sample.full_batch.batch
    tl = batch.get("teacher_logprobs", None)
    if tl is not None and tl.ndim == 3 and tl.shape[-1] == len(teacher_keys):
        # separate mode: (sample_N, seqlen, T) dense multi
        selected = tl[..., teacher_idx].unsqueeze(-1).clone()
        del batch["teacher_logprobs"]
    else:
        # fused mode: per-teacher keys __teacher_lp_{idx}
        per_teacher = [batch.pop(f"__teacher_lp_{i}") for i in range(len(teacher_keys))]
        stacked = torch.cat(per_teacher, dim=-1)  # (sample_N, seqlen, T)
        selected = stacked[..., teacher_idx].unsqueeze(-1).clone()
    batch["teacher_logprobs"] = selected  # (sample_N, seqlen, 1)




class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None
        self.fused_teacher_enabled = bool(
            self.distillation_config is not None and self.distillation_config.teacher_execution == "trainer"
        )

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self._init_dump_executor()
        self.validation_generations_logger = None
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self._fused_teacher_queue_finished = False
        self._resident_teacher_key = None
        self._teacher_scheduler = None
        if self.fused_teacher_enabled:
            teacher_keys = list(self.distillation_config.teacher_models)
            scoring_batch_size = (
                self.distillation_config.teacher_scoring_batch_size or self.required_samples
            )
            self._teacher_scheduler = ExclusiveTeacherScheduler(
                teacher_keys=teacher_keys,
                scoring_batch_size=scoring_batch_size,
                max_consecutive_batches=self.distillation_config.teacher_max_consecutive_batches,
                max_wait_seconds=self.distillation_config.teacher_max_wait_seconds,
            )
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

        print(f"Total training steps: {self.total_train_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_train_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_train_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        if self.fused_teacher_enabled:
            return await self._get_fused_teacher_samples_from_queue()

        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(ray.cloudpickle.loads(sample))

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        # nonrouter (separate mode): each sample carries all teachers' log_probs stacked
        # in `teacher_logprobs` (shape (sample_N, seqlen, T)). Select the data_source-matched
        # teacher per sample before assembly so downstream sees (sample_N, seqlen, 1).
        if self.distillation_config is not None and getattr(self.distillation_config, "nonrouter", False):
            teacher_keys = tuple(self.distillation_config.teacher_models)
            routing_field = self.distillation_config.teacher_key
            for rs in queue_samples:
                _select_teacher_logprobs_for_sample(rs, teacher_keys, routing_field)

        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _get_rollout_sample_teacher_key(self, rollout_sample) -> str:
        if self._teacher_scheduler.teacher_keys == ("default",):
            return "default"

        routing_field = self.distillation_config.teacher_key
        values = rollout_sample.full_batch.non_tensor_batch.get(routing_field)
        if values is None:
            raise ValueError(
                f"Fused multi-teacher routing requires non_tensor_batch[{routing_field!r}], "
                f"but sample {rollout_sample.sample_id!r} does not contain it."
            )

        normalized_values = []
        for value in np.asarray(values, dtype=object).reshape(-1):
            normalized_values.append(value.item() if hasattr(value, "item") else value)
        unique_values = set(normalized_values)
        if len(unique_values) != 1:
            raise ValueError(
                f"A RolloutSample must route all responses to one teacher, but sample "
                f"{rollout_sample.sample_id!r} has {routing_field} values {sorted(unique_values)!r}."
            )
        return next(iter(unique_values))

    def _score_fused_teacher_samples(self, teacher_key: str, pending_items) -> None:
        rollout_samples = [item.sample for item in pending_items]
        teacher_batch = DataProto.concat([sample.full_batch for sample in rollout_samples])
        batch_td = left_right_2_no_padding(teacher_batch.to_tensordict())
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            compute_loss=False,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )

        output = self.actor_rollout_wg.compute_teacher_log_prob(batch_td)
        teacher_log_probs = tu.get(output, "log_probs")
        teacher_log_probs = no_padding_2_padding(teacher_log_probs, batch_td)

        offset = 0
        trajectory_count = 0
        for rollout_sample in rollout_samples:
            sample_size = len(rollout_sample.full_batch)
            next_offset = offset + sample_size
            sample_log_probs = teacher_log_probs[offset:next_offset]
            if sample_log_probs.shape[0] != sample_size:
                raise RuntimeError(
                    f"Teacher {teacher_key!r} returned an invalid batch size while scoring "
                    f"sample {rollout_sample.sample_id!r}: expected {sample_size}, "
                    f"got {sample_log_probs.shape[0]}."
                )
            rollout_sample.full_batch.batch["teacher_logprobs"] = (
                sample_log_probs.float().unsqueeze(-1).clone()
            )
            offset = next_offset
            trajectory_count += sample_size

        if offset != teacher_log_probs.shape[0]:
            raise RuntimeError(
                f"Teacher {teacher_key!r} produced {teacher_log_probs.shape[0]} rows, "
                f"but only {offset} rows were assigned to rollout samples."
            )

        teacher_metrics = tu.get(output, "metrics")
        if teacher_metrics and "mfu" in teacher_metrics:
            self.metrics[f"perf/mfu/teacher_infer/{teacher_key}"] = teacher_metrics["mfu"]
        metric_key = f"distillation/teacher_route/{teacher_key}/trajectories"
        self.metrics[metric_key] = self.metrics.get(metric_key, 0) + trajectory_count

    async def _get_fused_teacher_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        if getattr(self.distillation_config, "nonrouter", False):
            return await self._get_fused_teacher_samples_nonrouter()
        scheduler = self._teacher_scheduler
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} routed samples from queue; "
            f"ready={scheduler.ready_count}, pending={scheduler.pending_count}",
            flush=True,
        )

        consumer_start = time.time()
        queue_len = 0
        collected = 0
        while scheduler.available_count < self.required_samples and not self._fused_teacher_queue_finished:
            sample, queue_len = await self.message_queue_client.get_sample()
            if sample is None:
                self._fused_teacher_queue_finished = True
                print(
                    "[FullyAsyncTrainer] Detected termination signal while collecting routed samples; "
                    f"ready={scheduler.ready_count}, pending={scheduler.pending_count}."
                )
                break

            rollout_sample = ray.cloudpickle.loads(sample)
            teacher_key = self._get_rollout_sample_teacher_key(rollout_sample)
            scheduler.add_sample(teacher_key, rollout_sample)
            collected += 1

            if collected % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {collected} new routed samples; "
                    f"ready={scheduler.ready_count}, pending={scheduler.pending_count}, mq_len={queue_len}."
                )

        total_wait_time = time.time() - consumer_start
        if scheduler.available_count < self.required_samples:
            self._activate_fused_actor()
            print(
                "[FullyAsyncTrainer] Not enough routed samples remain for a training batch: "
                f"available={scheduler.available_count}, required={self.required_samples}."
            )
            return None, None

        scoring_start = time.time()
        try:
            while scheduler.ready_count < self.required_samples:
                teacher_key = scheduler.choose_teacher(resident_teacher=self._resident_teacher_key)
                if teacher_key is None:
                    raise RuntimeError(
                        "Fused teacher scheduler has insufficient ready samples but no pending teacher work."
                    )
                pending_items = scheduler.pop_scoring_batch(teacher_key)
                self._activate_fused_teacher(teacher_key)
                self._score_fused_teacher_samples(teacher_key, pending_items)
                scheduler.mark_scored(teacher_key, pending_items)
        finally:
            self._activate_fused_actor()

        teacher_scoring_time = time.time() - scoring_start
        queue_samples = scheduler.take_ready(self.required_samples)
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, self._balance_batch
            )
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/teacher_scoring_time"] = teacher_scoring_time
        # Record teacher scoring time separately so it can be subtracted from gen
        # (gen otherwise bundles queue-wait + teacher scoring + batch assembly).
        self.timing_raw["teacher_scoring"] = teacher_scoring_time
        batch.meta_info["fully_async/teacher_pending_samples"] = scheduler.pending_count
        batch.meta_info["fully_async/teacher_ready_samples"] = scheduler.ready_count
        print(
            f"[FullyAsyncTrainer] Routed batch ready: collected={collected}, "
            f"teacher_scoring_time={teacher_scoring_time:.2f}s, "
            f"pending={scheduler.pending_count}, ready={scheduler.ready_count}.",
            flush=True,
        )
        return 0, batch

    def _score_fused_teacher_samples_multi(self, teacher_key: str, teacher_idx: int, batch_items) -> None:
        """nonrouter fused: score ONE teacher for a batch of samples and write the
        per-token log_probs to a per-teacher key `__teacher_lp_{teacher_idx}` (shape
        (sample_N, seqlen, 1)). Called once per teacher; selection merges them later."""
        # batch_items are RolloutSample objects (passed directly from the nonrouter loop)
        rollout_samples = batch_items
        teacher_batch = DataProto.concat([sample.full_batch for sample in rollout_samples])
        batch_td = teacher_batch.to_tensordict()
        # Strip per-teacher keys from prior teachers BEFORE left_right_2_no_padding,
        # which only knows input_ids/teacher_logprobs/etc. and would leave unknown
        # dense keys (__teacher_lp_*) in the tensordict sent to the engine.
        for k in list(batch_td.keys()):
            if k.startswith("__teacher_lp_"):
                del batch_td[k]
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            compute_loss=False,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )

        output = self.actor_rollout_wg.compute_teacher_log_prob(batch_td)
        teacher_log_probs = tu.get(output, "log_probs")
        teacher_log_probs = no_padding_2_padding(teacher_log_probs, batch_td)

        offset = 0
        trajectory_count = 0
        for rollout_sample in rollout_samples:
            sample_size = len(rollout_sample.full_batch)
            next_offset = offset + sample_size
            sample_log_probs = teacher_log_probs[offset:next_offset]
            if sample_log_probs.shape[0] != sample_size:
                raise RuntimeError(
                    f"Teacher {teacher_key!r} returned an invalid batch size while scoring "
                    f"sample {rollout_sample.sample_id!r}: expected {sample_size}, "
                    f"got {sample_log_probs.shape[0]}."
                )
            rollout_sample.full_batch.batch[f"__teacher_lp_{teacher_idx}"] = (
                sample_log_probs.float().unsqueeze(-1).clone()
            )
            offset = next_offset
            trajectory_count += sample_size

        if offset != teacher_log_probs.shape[0]:
            raise RuntimeError(
                f"Teacher {teacher_key!r} produced {teacher_log_probs.shape[0]} rows, "
                f"but only {offset} rows were assigned to rollout samples."
            )

        teacher_metrics = tu.get(output, "metrics")
        if teacher_metrics and "mfu" in teacher_metrics:
            self.metrics[f"perf/mfu/teacher_infer/{teacher_key}"] = teacher_metrics["mfu"]
        metric_key = f"distillation/teacher_route/{teacher_key}/trajectories"
        self.metrics[metric_key] = self.metrics.get(metric_key, 0) + trajectory_count

    async def _get_fused_teacher_samples_nonrouter(self) -> tuple[None, None] | tuple[int, Any]:
        """nonrouter fused: every sample is forwarded by ALL teachers (no routing at
        forward time). Two teacher CPU<->GPU swaps per scoring batch. Per-teacher
        log_probs are stored under `__teacher_lp_{idx}`, then the data_source-matched
        teacher is selected per sample before assembly. gen excludes teacher_scoring."""
        teacher_keys = list(self.distillation_config.teacher_models)
        scoring_batch_size = (
            self.distillation_config.teacher_scoring_batch_size or self.required_samples
        )
        print(
            f"[FullyAsyncTrainer] [nonrouter] Requesting {self.required_samples} samples; "
            f"teachers={teacher_keys}",
            flush=True,
        )

        consumer_start = time.time()
        queue_len = 0
        pending: list = []  # collected, not yet scored
        ready: list = []    # scored
        while (len(pending) + len(ready)) < self.required_samples and not self._fused_teacher_queue_finished:
            sample, queue_len = await self.message_queue_client.get_sample()
            if sample is None:
                self._fused_teacher_queue_finished = True
                print(
                    "[FullyAsyncTrainer] [nonrouter] Detected termination signal while collecting; "
                    f"pending={len(pending)}, ready={len(ready)}."
                )
                break
            pending.append(ray.cloudpickle.loads(sample))

        total_wait_time = time.time() - consumer_start
        if (len(pending) + len(ready)) < self.required_samples:
            self._activate_fused_actor()
            print(
                f"[FullyAsyncTrainer] [nonrouter] Not enough samples: "
                f"available={len(pending)+len(ready)}, required={self.required_samples}."
            )
            return None, None

        scoring_start = time.time()
        try:
            while len(ready) < self.required_samples:
                batch_items = pending[:scoring_batch_size]
                pending = pending[scoring_batch_size:]
                # Run EVERY teacher on this batch (sequential CPU<->GPU swaps).
                for idx, teacher_key in enumerate(teacher_keys):
                    self._activate_fused_teacher(teacher_key)
                    self._score_fused_teacher_samples_multi(teacher_key, idx, batch_items)
                ready.extend(batch_items)
        finally:
            self._activate_fused_actor()

        teacher_scoring_time = time.time() - scoring_start

        # Per-sample selection by data_source (before assemble -> avoids nested gather).
        routing_field = self.distillation_config.teacher_key
        for rs in ready:
            _select_teacher_logprobs_for_sample(rs, tuple(teacher_keys), routing_field)

        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                ready, self.tokenizer, self.config, self._balance_batch
            )
        else:
            batch = assemble_batch_from_rollout_samples(ready, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/teacher_scoring_time"] = teacher_scoring_time
        # Record teacher scoring time separately so gen can subtract it.
        self.timing_raw["teacher_scoring"] = teacher_scoring_time
        print(
            f"[FullyAsyncTrainer] [nonrouter] Batch ready: "
            f"teacher_scoring_time={teacher_scoring_time:.2f}s, ready={len(ready)}.",
            flush=True,
        )
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        self._activate_fused_actor()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()
        self._activate_fused_teacher()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        # In fused mode, gen bundles teacher scoring time. Subtract it so gen
        # reflects only queue-wait + batch assembly (comparable to separate mode).
        # teacher_scoring is now recorded separately as timing_s/teacher_scoring.
        if "teacher_scoring" in timing_raw:
            timing_raw["gen"] -= timing_raw["teacher_scoring"]
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _activate_fused_actor(self):
        if self.fused_teacher_enabled:
            self.actor_rollout_wg.activate_actor()
            self._resident_teacher_key = None

    def _activate_fused_teacher(self, teacher_key: str | None = None):
        if self.fused_teacher_enabled:
            if teacher_key is None:
                teacher_key = self._teacher_scheduler.choose_teacher(
                    resident_teacher=self._resident_teacher_key
                )
            if teacher_key is None:
                return
            self.actor_rollout_wg.activate_teacher(teacher_key)
            self._resident_teacher_key = teacher_key

    def prepare_fused_teacher(self):
        """Place the fused node in teacher-resident state before queue consumption."""
        self._activate_fused_teacher()

    def _fit_collect_metrics(self, batch):
        """Merge worker-side fused CPU<->GPU swap timings into timing_raw before
        computing metrics, so they surface as timing_s/<name> like other timers."""
        if getattr(self, "fused_teacher_enabled", False):
            try:
                swap_timings = self.actor_rollout_wg.collect_swap_timing()
                # ONE_TO_ALL returns one entry per worker; workers are TP-synchronized,
                # so take the max across workers as the wall-clock time.
                if swap_timings:
                    merged: dict = {}
                    for entry in swap_timings:
                        if isinstance(entry, dict):
                            for k, v in entry.items():
                                try:
                                    fv = float(v)
                                except (TypeError, ValueError):
                                    continue
                                merged[k] = max(merged.get(k, 0.0), fv)
                    for k, v in merged.items():
                        self.timing_raw[k] = self.timing_raw.get(k, 0.0) + v
            except Exception as e:
                print(f"[FullyAsyncTrainer] collect_swap_timing failed: {e}")
        super()._fit_collect_metrics(batch)

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        # CPU<->GPU swaps for old-policy student recompute. Keys are semantically distinct:
        #   old_student_gpu_to_cpu : step 1 only - archive theta_v1 to CPU[1] (once per 4-step cycle)
        #   cur_student_gpu_to_cpu : step 2/3/4 - stash current student to free GPU for theta_v1
        #   old_student_cpu_to_gpu : step 2/3/4 - load theta_v1 onto GPU to compute old_log_prob
        #   cur_student_cpu_to_gpu : step 2/3/4 - restore current student after old_log_prob
        if self.local_trigger_step == 1:
            with marked_timer("old_student_gpu_to_cpu", self.timing_raw):
                self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            with marked_timer("cur_student_gpu_to_cpu", self.timing_raw):
                self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            try:
                with marked_timer("old_student_cpu_to_gpu", self.timing_raw):
                    self.actor_rollout_wg.restore_model_from_cpu(1)
                old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            finally:
                with marked_timer("cur_student_cpu_to_gpu", self.timing_raw):
                    self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
                self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        steps = self.config.global_profiler.steps
        last_profiler_step = self.current_param_version
        if steps is not None and last_profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._stop_profiling.remote().future())

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        profiler_step = last_profiler_step + 1

        if steps is not None and profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._start_profiling.remote().future())

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            if force:
                print("Train finish, final addition checkpoint saving...")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                try:
                    self._save_checkpoint()
                except Exception as e:
                    print(f"When save ckpt, error: {e}")
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")
        print(f"[FullyAsyncTrainer] Training progress set to {self.current_param_version}/{self.total_train_steps}")
        self.progress_bar.update(self.current_param_version)

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

        from verl.experimental.agent_loop.agent_loop import agent_loop_dump, agent_loop_metrics

        async_metrics = agent_loop_metrics(batch)
        metrics.update(async_metrics)
        if self.config.trainer.get("save_error_query", False):
            from verl.utils.fs import local_mkdir_safe

            local_mkdir_safe(self.config.trainer.default_local_dir)
            agent_loop_dump(
                batch,
                f"{self.config.trainer.default_local_dir}/error_query.txt",
                self.tokenizer,
            )
