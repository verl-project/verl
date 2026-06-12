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

"""TQFullyAsyncTrainer: Multi-inheritance trainer combining PPOTrainer's KVBatchMeta pipeline
with FullyAsyncTrainer's async infrastructure.

MRO: TQFullyAsyncTrainer → PPOTrainer → FullyAsyncTrainer → SeparateRayPPOTrainer → ...

Data flow:
    TQFullyAsyncRollouter --(tq.kv_batch_put)--> TransferQueue (status=finish)
        |
    TQFullyAsyncTrainer <-(RB.wait_and_sample)--+--(KVBatchMeta)--> [PPOTrainer pipeline]
                                                    |
                                              update_actor(KVBatchMeta)
"""

import logging
import os
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer,
    TrainingStopException,
)
from verl.experimental.fully_async_policy.replay_buffer import tq_kv_clear
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking, ValidationGenerationsLogger

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.8` and try again.")
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncTrainerTQ(PPOTrainer, FullyAsyncTrainer):
    """
    Fully async PPO trainer via multi-inheritance.

    - PPOTrainer: provides KVBatchMeta-native training pipeline (_compute_*, _update_*, etc.)
    - FullyAsyncTrainer: provides async infrastructure (fit loop, param sync, validate, checkpoint)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Any = None,
        device_name=None,
    ):
        # ======== 1. PPOTrainer.__init__: config, dataloader, local replay_buffer, worker groups ========
        PPOTrainer.__init__(
            self,
            config=config,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
        )
        # PPOTrainer doesn't accept device_name/ray_worker_group_cls, set them manually
        # These are required by FullyAsyncTrainer(SeparateRayPPOTrainer) init_workers pipeline:
        #   _init_resource_pools → _create_worker_classes → _init_worker_groups → _init_models
        self.device_name = device_name
        self.ray_worker_group_cls = ray_worker_group_cls or RayWorkerGroup
        self.tokenizer = tokenizer

        # Additional attributes from SeparateRayPPOTrainer/RayPPOTrainer.__init__
        # that PPOTrainer.__init__ doesn't set but _create_worker_classes / _init_models need:
        from verl.trainer.ppo.utils import need_reward_model

        self.use_rm = need_reward_model(self.config)
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # ======== 2. FullyAsyncTrainer state fields ========
        # (mirrors FullyAsyncTrainer.__init__ lines 108-163)
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}

        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        self.rollouter = None
        self.checkpoint_manager = None
        self.hybrid_checkpoint_manager = None

        # ======== 3. TQ-specific: ReplayBuffer Ray Actor handle ========
        self.replay_buffer = None  # Set via set_replay_buffer()

        # Logger
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        print("[TQFullyAsyncTrainer] initialized (multi-inherit: PPOTrainer + FullyAsyncTrainer)")

    async def set_replay_buffer(self, replay_buffer):
        """Set ReplayBuffer Ray Actor handle."""
        self.replay_buffer = replay_buffer

    async def init_workers(self):
        await FullyAsyncTrainer.init_workers(self)
        tq.init()

    async def _get_keys_from_rb(self) -> KVBatchMeta | None:
        """
        Get a KVBatchMeta from TQ via ReplayBuffer.

        Replaces both:
        - PPOTrainer's generate_sequences() + replay_buffer.sample()
        - FullyAsyncTrainer's message_queue_client.get_sample() + assemble

        Returns KVBatchMeta compatible with PPOTrainer's entire step() pipeline.
        """
        sampled_keys_meta = await self.replay_buffer.sample.remote(
            partition_id="train",
            sample_size=self.required_samples,
        )

        if sampled_keys_meta is None or len(sampled_keys_meta) == 0:
            print("[TQFullyAsyncTrainer] RB returned None (termination signal)")
            return None

        rollout_n = self.config.actor_rollout_ref.rollout.n
        expected = self.required_samples * rollout_n
        if len(sampled_keys_meta) != expected:
            # TODO: when multi-trajectory output support is added, this will need to change
            raise ValueError(
                f"[ReplayBuffer][sample] BUG: len(all_response_keys)={len(sampled_keys_meta)} != expected={expected}"
            )

        keys = [k for k, _ in sampled_keys_meta]
        tags = [meta for _, meta in sampled_keys_meta]

        return KVBatchMeta(partition_id="train", keys=keys, tags=tags)

    async def fit(self):
        """Main training loop: async RB consumption + PPOTrainer step() pipeline."""
        print("[TQFullyAsyncTrainer] Starting fit ...", flush=True)
        print(
            f"[TQTrainer] fit(): rb={self.replay_buffer is not None}, rollouter={self.rollouter is not None}",
            flush=True,
        )
        if self.replay_buffer is None:
            raise ValueError("ReplayBuffer not set. Call set_replay_buffer() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[TQFullyAsyncTrainer] Training stopped by termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """Single training step: get KVBatchMeta from RB → run PPOTrainer pipeline."""
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}

        self._start_profiling()

        with marked_timer("step", self.timing_raw):
            # Run PPOTrainer's full KVBatchMeta pipeline (steps 2-10)
            metrics = self.metrics
            timing_raw = self.timing_raw

            # ★ CORE: Get KVBatchMeta from RB (replaces generate_sequences + replay_buffer.sample)

            # 2. sample batch from replay buffer
            with marked_timer("gen", timing_raw, color="red"):
                batch = await self._get_keys_from_rb()
                if batch is None:
                    raise TrainingStopException("Training terminated: RB returned None")
                batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            # 3. [OPTIONAL] compute reward score with colocated reward model
            # TODO colocate reward not implemented

            # 4. balance batch across data parallel groups
            batch = self._balance_batch(batch, metrics=metrics)

            # 5. compute old_log_prob
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                batch = self._compute_old_log_prob(batch, metrics=metrics)

            # 6. [OPTIONAL] compute ref_log_prob
            if self.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    batch = self._compute_ref_log_prob(batch, metrics=metrics)

            # 7. [OPTIONAL] compute critic values
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    batch = self._compute_values(batch, metrics=metrics)

            # 8. compute advantage and return
            with marked_timer("adv", timing_raw, color="brown"):
                batch = self._compute_advantage(batch, metrics=metrics)

            # 9. [OPTIONAL] update critic
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    batch = self._update_critic(batch, metrics=metrics)

            # 10. update actor
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch = self._update_actor(batch, metrics=metrics)

            self._fit_update_local_step()
            await self._fit_update_weights()

        await self._fit_collect_metrics(batch)
        tq_kv_clear(batch)
        await self._fit_reset_staleness()

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._stop_profiling()
        self._fit_postprocess_step()

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )
        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_reset_staleness(self):
        # Reset staleness in rollouter
        if self.local_trigger_step != 1:
            return
        timing_raw = await self.rollouter.reset_staleness.remote()
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

    async def _fit_collect_metrics(self, batch: KVBatchMeta):
        """Collect metrics using PPOTrainer's _compute_metrics (expects KVBatchMeta).

        Also merges rollouter statistics and TQ timing tags to produce the same set of
        ``fully_async/*`` and ``timing_s/*`` metrics as the non-TQ path's batch assembly.
        """
        # 1. Base PPOTrainer metrics (data, timing, throughput, variance proxy)
        self._compute_metrics(batch, self.metrics, self.timing_raw, global_steps=self.global_steps, epoch=self.epoch)

        # 2. Collect per-sample timing stats from tags for aggregation
        processing_times = []
        tool_calls_list = []
        compute_score_list = []
        param_versions = []  # max_global_steps per sample (mirrors trajectory_param_versions in detach_utils)
        param_version_starts = []  # min_global_steps per sample
        if batch.tags:
            for tag in batch.tags:
                if not isinstance(tag, dict):
                    continue
                # Collect raw values for aggregation
                if "timing_s/gen" in tag:
                    processing_times.append(tag["timing_s/gen"])
                if "timing_s/agent_loop/tool_calls" in tag:
                    tool_calls_list.append(tag["timing_s/agent_loop/tool_calls"])
                if "timing_s/compute_score" in tag:
                    compute_score_list.append(tag["timing_s/compute_score"])
                if "max_global_steps" in tag:
                    param_versions.append(tag["max_global_steps"])
                if "min_global_steps" in tag:
                    param_version_starts.append(tag["min_global_steps"])
                # Merge all fully_async/* and timing_s/* tags directly
                for key, value in tag.items():
                    if key.startswith("fully_async") or key.startswith("timing_s"):
                        self.metrics[key] = value

        # 3. Compute aggregated processing_time stats (mirrors assemble_batch_from_rollout_samples lines 134-149)
        if processing_times:
            processing_time_stats = {
                "fully_async/processing_time/avg": float(np.mean(processing_times)),
                "fully_async/processing_time/max": float(np.max(processing_times)),
                "fully_async/processing_time/min": float(np.min(processing_times)),
                "fully_async/processing_time/tp50": float(np.percentile(processing_times, 50)),
                "fully_async/processing_time/tp99": float(np.percentile(processing_times, 99)),
                "fully_async/processing_time/tp95": float(np.percentile(processing_times, 95)),
            }
            self.metrics.update(processing_time_stats)

        # 4. Compute tool_calls stats
        if tool_calls_list:
            tool_calls_stats = {
                "timing_s/agent_loop/tool_calls/max": float(np.max(tool_calls_list)),
                "timing_s/agent_loop/tool_calls/min": float(np.min(tool_calls_list)),
                "timing_s/agent_loop/tool_calls/mean": float(np.mean(tool_calls_list)),
            }
            self.metrics.update(tool_calls_stats)

        # 4b. Compute compute_score stats
        if compute_score_list:
            compute_score_stats = {
                "timing_s/compute_score/max": float(np.max(compute_score_list)),
                "timing_s/compute_score/min": float(np.min(compute_score_list)),
                "timing_s/compute_score/mean": float(np.mean(compute_score_list)),
            }
            self.metrics.update(compute_score_stats)

        # 5. Compute partial stats from param versions (mirrors assemble_batch_from_rollout_samples lines 151-159)
        if param_versions and param_version_starts:
            param_version_diff = [abs(a - b) for a, b in zip(param_versions, param_version_starts, strict=False)]
            num_diff0 = param_version_diff.count(0)
            partial_stats = {
                "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
                "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff)
                if param_version_diff
                else 0.0,
                "fully_async/partial/max_partial_span": max(param_version_diff) if param_version_diff else 0,
            }
            self.metrics.update(partial_stats)

            # 6. stale_trajectory_processed count
            # Use max_global_steps as trajectory_param_versions
            stale_traj_count = sum(1 for v in param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            self.metrics["fully_async/count/stale_trajectory_processed"] = self.stale_trajectory_processed
            self.metrics["fully_async/count/current_param_version"] = self.current_param_version

        # 7. Merge rollouter monitor/count/static stats
        rollouter_stats = await self.replay_buffer.get_statistics.remote()
        for k, v in rollouter_stats.items():
            if isinstance(v, int | float):
                self.metrics[f"fully_async/{k}"] = v
