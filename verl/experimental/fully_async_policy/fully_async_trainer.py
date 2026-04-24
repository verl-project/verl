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
from pprint import pprint
from typing import Any

import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.intermediate_trajectory_utils import (
    assert_batch_schema,
    expand_intermediate_trajectories_pre_log_prob,
    scatter_advantage_to_intermediate_and_normalize,
    zero_out_padding_rows,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)


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
        processor=None,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
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
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
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
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # use trainer to do validation
        if self.config.async_training.use_trainer_do_validate:
            from verl.trainer.main_ppo import create_rl_dataset
            from verl.utils.dataset.rl_dataset import collate_fn

            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            print(f"[FullyAsyncTrainer] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[rollout_gpus:]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncTrainer] split after val_dataset total len: {len(val_dataset)}")
            self.val_dataset = val_dataset
            # update val_dataloader
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            from torchdata.stateful_dataloader import StatefulDataLoader

            print(f"[FullyAsyncTrainer] create val_dataloader with batch_size: {val_batch_size}")
            self.val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # when use_trainer_do_validate == Ture, use colocate_checkpoint_manager to sync params
        self.colocate_checkpoint_manager = None

    def _setup_checkpoint_manager(self, rollouter):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_rollouter(self, rollouter):
        """Set rollouter reference for parameter synchronization"""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        self._setup_checkpoint_manager(rollouter)

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
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
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

            queue_samples.append(sample)

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

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                queue_samples,
                self.tokenizer,
                self.config,
                self._balance_batch,
                processor=self.processor,
            )
        else:
            batch = assemble_batch_from_rollout_samples(
                queue_samples,
                self.tokenizer,
                self.config,
                None,
                processor=self.processor,
            )

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor
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
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _init_reward_loop(self):
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init reward loop")
            super()._init_reward_loop()

    async def _init_async_rollout_manager(self):
        # use async rollout do validate
        print(f"[FullyAsyncTrainer] use_trainer_do_validate: {self.config.async_training.use_trainer_do_validate}")
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init async rollout manager")

            # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
            # agent_reward_loop: streaming reward computation with actor rollout
            # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
            enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

            # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
            # to stream reward computation with actor rollout
            reward_loop_worker_handles = (
                self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
            )

            # create async rollout manager and request scheduler
            assert self.config.actor_rollout_ref.rollout.mode == "async"

            self.async_rollout_mode = True
            from verl.experimental.agent_loop import AgentLoopManager

            self.llm_server_manager = await LLMServerManager.create(
                config=self.config, worker_group=self.actor_rollout_wg
            )
            self.async_rollout_manager = await AgentLoopManager.create(
                config=self.config,
                llm_client=self.llm_server_manager.get_client(),
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            print("[FullyAsyncTrainer] async_rollout_manager initialized")

            # Modify checkpoint_engine config to use naive backend
            checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
            original_backend = checkpoint_engine_cfg.backend
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = "naive"
            checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

            print(f"[FullyAsyncTrainer] checkpoint_engine_config: {checkpoint_engine_config}")

            self.colocate_checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=self.actor_rollout_wg,
                replicas=self.llm_server_manager.get_replicas(),
            )

            # sleep all replicas to load checkpoint
            await self.colocate_checkpoint_manager.sleep_replicas()

            # Restore original backend value
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = original_backend

            print("[FullyAsyncTrainer] colocate_checkpoint_manager initialized")

        else:
            print("[FullyAsyncTrainer] Skip async rollout manager (use_trainer_do_validate=False)")

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
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
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

        self._fit_start_profile()

        _CORE_KEYS = {"input_ids", "attention_mask", "position_ids", "responses", "response_mask"}

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)

            # Detect position_ids ndim from the assembled batch for consistent
            # assertion throughout the pipeline.
            _pos_ndim = (
                batch.batch["position_ids"].ndim
                if batch.batch is not None and "position_ids" in batch.batch.keys()
                else None
            )

            batch = self._fit_compute_reward(batch)
            assert_batch_schema(batch, "fit_step.after_reward",
                                expected_tensor_keys=_CORE_KEYS,
                                require_position_ids_ndim=_pos_ndim,
                                has_processor=self.processor is not None)
            self._log_training_diagnostics("after_reward", batch)
            # Expand intermediate trajectories (and pad to actor mini-batch
            # multiple) BEFORE any per-token forward pass, so that
            # log_prob / ref_log_prob / critic are computed over every
            # trajectory row that will participate in the actor update.
            batch = self._fit_expand_and_pad(batch)
            assert_batch_schema(batch, "fit_step.after_expand_and_pad",
                                expected_tensor_keys=_CORE_KEYS,
                                require_position_ids_ndim=_pos_ndim)
            self._log_training_diagnostics("after_expand_and_pad", batch)
            batch = self._fit_compute_log_prob(batch)
            assert_batch_schema(batch, "fit_step.after_log_prob",
                                require_position_ids_ndim=_pos_ndim)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            assert_batch_schema(batch, "fit_step.after_critic",
                                require_position_ids_ndim=_pos_ndim)
            self._log_training_diagnostics("after_log_prob_and_ref", batch)
            # Advantage is computed on the FINAL subset only (GRPO group
            # stats must not see intermediate rows), then the scalar is
            # broadcast to sibling intermediate rows and scaled by 1/T_rollout
            # so that every rollout contributes equally under token-mean.
            batch = self._fit_compute_advantage(batch)
            assert_batch_schema(batch, "fit_step.after_advantage",
                                require_position_ids_ndim=_pos_ndim)
            self._log_training_diagnostics("after_advantage_scatter_normalize", batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _log_training_diagnostics(
        self,
        stage: str,
        batch: DataProto,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Dump a FULL per-field snapshot of the batch for eyeballing.

        For every entry in ``batch.batch`` (tensors), ``batch.non_tensor_batch``
        (numpy object arrays / lists) and ``batch.meta_info``, one line is
        emitted with:
          * value type,
          * shape / length / dtype when applicable,
          * full content if the element count fits within a small threshold,
          * otherwise a head/tail preview.

        A few derived summaries (role distribution, rollout-group sizes,
        advantage distribution per role, masked-mean of log_probs) are
        appended at the end because they are not readable from the raw
        per-field dump.

        Output goes only to a per-run log file (``FULLY_ASYNC_DIAG_LOG`` or
        ``<trainer.default_local_dir>/training_diagnostics.log``).
        """
        import numpy as np

        # ---------------- configurable thresholds ----------------
        # Max number of elements printed in full for a tensor / ndarray /
        # sequence. Beyond this we only show shape + head preview.
        MAX_ELEMS_FULL = 64
        # Max length of a repr preview string (per field).
        MAX_PREVIEW_LEN = 240

        def _fmt_tensor(t: torch.Tensor) -> str:
            shape = tuple(t.shape)
            numel = t.numel()
            head = f"shape={shape} dtype={t.dtype} device={t.device}"
            try:
                if numel == 0:
                    return head + " value=<empty>"
                if numel <= MAX_ELEMS_FULL:
                    return head + f" value={t.detach().cpu().tolist()}"
                flat = t.detach().reshape(-1).cpu()
                head_preview = flat[: min(8, numel)].tolist()
                tail_preview = flat[-min(4, numel) :].tolist()
                return head + f" head={head_preview} tail={tail_preview}"
            except Exception as exc:
                return head + f" <repr failed: {exc!r}>"

        def _fmt_ndarray(a: np.ndarray) -> str:
            shape = tuple(a.shape)
            numel = int(a.size)
            head = f"shape={shape} dtype={a.dtype}"
            try:
                if numel == 0:
                    return head + " value=<empty>"
                if numel <= MAX_ELEMS_FULL:
                    return head + " value=" + np.array2string(a, threshold=MAX_ELEMS_FULL)[:MAX_PREVIEW_LEN]
                flat = a.reshape(-1)
                return (
                    head
                    + " head="
                    + np.array2string(flat[:8], threshold=MAX_ELEMS_FULL)[:MAX_PREVIEW_LEN]
                    + " tail="
                    + np.array2string(flat[-4:], threshold=MAX_ELEMS_FULL)[:MAX_PREVIEW_LEN]
                )
            except Exception as exc:
                return head + f" <repr failed: {exc!r}>"

        def _fmt_sequence(v) -> str:
            try:
                ln = len(v)
            except TypeError:
                return f"type={type(v).__name__} value={repr(v)[:MAX_PREVIEW_LEN]}"
            head = f"len={ln} type={type(v).__name__}"
            if ln == 0:
                return head + " value=[]"
            if ln <= MAX_ELEMS_FULL:
                return head + f" value={repr(v)[:MAX_PREVIEW_LEN]}"
            # dump first 3 + last 2 to show schema without flooding
            preview = list(v[:3]) + ["..."] + list(v[-2:])
            return head + f" preview={repr(preview)[:MAX_PREVIEW_LEN]}"

        def _fmt_any(v) -> str:
            if isinstance(v, torch.Tensor):
                return _fmt_tensor(v)
            if isinstance(v, np.ndarray):
                return _fmt_ndarray(v)
            if isinstance(v, (list, tuple)):
                return _fmt_sequence(v)
            if isinstance(v, dict):
                keys = list(v.keys())
                preview_keys = keys[:10]
                return f"type=dict len={len(keys)} keys[:10]={preview_keys} repr={repr(v)[:MAX_PREVIEW_LEN]}"
            if isinstance(v, (int, float, bool, str)) or v is None:
                return f"type={type(v).__name__} value={repr(v)[:MAX_PREVIEW_LEN]}"
            return f"type={type(v).__name__} repr={repr(v)[:MAX_PREVIEW_LEN]}"

        # ==================== build the dump ====================
        n = len(batch)
        lines: list[str] = [f"[FullyAsyncTrainer][DIAG][{stage}] batch_size={n}"]

        # --- batch.batch (tensors) ---
        tensor_keys = sorted(batch.batch.keys()) if batch.batch is not None else []
        lines.append(f"  [batch.batch] num_keys={len(tensor_keys)}")
        for k in tensor_keys:
            try:
                v = batch.batch[k]
            except Exception as exc:
                lines.append(f"    - {k!r}: <fetch failed: {exc!r}>")
                continue
            lines.append(f"    - {k!r}: {_fmt_any(v)}")

        # --- batch.non_tensor_batch ---
        nt = batch.non_tensor_batch or {}
        lines.append(f"  [batch.non_tensor_batch] num_keys={len(nt)}")
        for k in sorted(nt.keys()):
            v = nt[k]
            lines.append(f"    - {k!r}: {_fmt_any(v)}")

        # --- batch.meta_info ---
        meta = batch.meta_info or {}
        lines.append(f"  [batch.meta_info] num_keys={len(meta)}")
        cache_key = "__intermediate_trajectories_cache__"
        for k in sorted(meta.keys(), key=lambda x: str(x)):
            v = meta[k]
            if k == cache_key and isinstance(v, dict):
                # Expand intermediate trajectories cache with per-row detail.
                interm_col = v.get("intermediate_col")
                main_bsz = v.get("main_batch_size")
                n_rows_with = 0
                total_trajs = 0
                per_row_counts = []
                per_traj_mm_status: list[str] = []
                if interm_col is not None:
                    for row_idx, row_list in enumerate(interm_col):
                        cnt = len(row_list) if row_list else 0
                        per_row_counts.append(cnt)
                        if cnt > 0:
                            n_rows_with += 1
                            total_trajs += cnt
                        for traj in (row_list or []):
                            mm = traj.get("multi_modal_data") if isinstance(traj, dict) else None
                            if mm is None:
                                per_traj_mm_status.append("None")
                            elif not mm:
                                per_traj_mm_status.append("empty")
                            else:
                                img_count = len(mm.get("images") or [])
                                vid_count = len(mm.get("videos") or [])
                                per_traj_mm_status.append(f"img={img_count},vid={vid_count}")
                lines.append(
                    f"    - {k!r}: main_batch_size={main_bsz} "
                    f"rows_with_intermediates={n_rows_with}/{len(interm_col) if interm_col else 0} "
                    f"total_intermediate_trajs={total_trajs} "
                    f"per_row_counts={per_row_counts}"
                )
                if per_traj_mm_status:
                    lines.append(
                        f"      multi_modal_data_per_traj: [{', '.join(per_traj_mm_status)}]"
                    )
            else:
                lines.append(f"    - {k!r}: {_fmt_any(v)}")

        # ============ derived summaries (not in raw dump) ============
        lines.append("  [summary]")

        roles = nt.get("trajectory_role")
        role_arr = None
        if roles is not None:
            role_arr = np.asarray(roles)
            n_final = int((role_arr == "final").sum())
            n_inter = int((role_arr == "intermediate").sum())
            n_other = n - n_final - n_inter
            lines.append(f"    roles: final={n_final} intermediate={n_inter} other={n_other}")

        gids = nt.get("rollout_group_id")
        if gids is not None:
            gids_np = np.asarray(gids, dtype=np.int64)
            unique_gids, counts = np.unique(gids_np, return_counts=True)
            lines.append(
                f"    rollout_groups: n_groups={len(unique_gids)} "
                f"rows_per_group(min/median/max)="
                f"{int(counts.min())}/{int(np.median(counts))}/{int(counts.max())}"
            )

        if "response_mask" in batch.batch.keys():
            rm = batch.batch["response_mask"]
            per_row = rm.to(torch.float32).sum(dim=-1)
            lines.append(
                f"    response_mask: total_valid_tokens={float(per_row.sum()):.0f} "
                f"per_row(min/mean/max)="
                f"{float(per_row.min()):.0f}/{float(per_row.mean()):.2f}/{float(per_row.max()):.0f}"
            )

        if "rm_scores" in batch.batch.keys():
            rs = batch.batch["rm_scores"]
            rs_row = rs.to(torch.float32).sum(dim=-1)
            nonzero = int((rs_row != 0).sum().item())
            lines.append(
                f"    rm_scores: rows_with_nonzero={nonzero}/{n} "
                f"per_row_sum(min/mean/max)="
                f"{float(rs_row.min()):.4f}/{float(rs_row.mean()):.4f}/{float(rs_row.max()):.4f}"
            )

        if "advantages" in batch.batch.keys():
            adv = batch.batch["advantages"].to(torch.float32)
            if "response_mask" in batch.batch.keys():
                mask = batch.batch["response_mask"].to(torch.float32)
                denom = mask.sum(dim=-1).clamp_min(1.0)
                adv_scalar = (adv * mask).sum(dim=-1) / denom
            else:
                adv_scalar = adv.mean(dim=-1)
            n_zero_adv = int((adv_scalar.abs() < 1e-12).sum().item())
            lines.append(
                f"    advantages(per_row_masked_mean): "
                f"min/mean/max={float(adv_scalar.min()):.6f}/"
                f"{float(adv_scalar.mean()):.6f}/{float(adv_scalar.max()):.6f} "
                f"nonzero_rows={n - n_zero_adv}/{n}"
            )
            if role_arr is not None:
                for role_name in ("final", "intermediate"):
                    idx = np.where(role_arr == role_name)[0]
                    if len(idx) == 0:
                        continue
                    sub = adv_scalar[torch.as_tensor(idx, dtype=torch.long)]
                    lines.append(
                        f"      {role_name}: advantages "
                        f"min/mean/max={float(sub.min()):.6f}/"
                        f"{float(sub.mean()):.6f}/{float(sub.max()):.6f}"
                    )

        for lp_key in ("old_log_probs", "ref_log_prob"):
            if lp_key in batch.batch.keys():
                lp = batch.batch[lp_key].to(torch.float32)
                if "response_mask" in batch.batch.keys():
                    mask = batch.batch["response_mask"].to(torch.float32)
                    denom = mask.sum().clamp_min(1.0)
                    masked_mean = float((lp * mask).sum() / denom)
                else:
                    masked_mean = float(lp.mean())
                lines.append(f"    {lp_key}(masked_mean): {masked_mean:.4f}")

        n_pad = batch.meta_info.get("fully_async/pad/num_padding_rows")
        if n_pad is not None:
            lines.append(f"    pad: num_padding_rows={int(n_pad)}")

        if extra:
            for k, v in extra.items():
                lines.append(f"    {k}: {v}")

        msg = "\n".join(lines)

        # Write diagnostics ONLY to a per-run log file (not stdout) to avoid
        # spamming Ray's log aggregation when the batch is large.
        # First write of each run truncates the file; subsequent writes append.
        try:
            log_path = os.environ.get("FULLY_ASYNC_DIAG_LOG")
            if not log_path:
                default_dir = self.config.trainer.get("default_local_dir", "outputs/fully_async")
                log_path = os.path.join(default_dir, "training_diagnostics.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            header = f"[{ts}] step={self.global_steps} local_trigger_step={self.local_trigger_step}"
            with open(log_path, "a" if getattr(self, "_diag_file_initialized", False) else "w", encoding="utf-8") as fh:
                fh.write(header + "\n")
                fh.write(msg + "\n")
                fh.write("-" * 80 + "\n")
            self._diag_file_initialized = True
        except Exception as exc:
            # Diagnostic logging must never crash training. Fall back to a
            # SINGLE terminal line so the failure is visible without dumping
            # the whole snapshot to console.
            print(
                f"[FullyAsyncTrainer][DIAG] failed to write log file: {exc!r}",
                flush=True,
            )

    def _fit_expand_and_pad(self, batch: DataProto) -> DataProto:
        """Expand intermediate trajectories and pad the batch.

        This runs between ``_fit_compute_reward`` and ``_fit_compute_log_prob``
        so that log_prob / ref_log_prob / critic are computed on every row
        (final + intermediate) with their own independent forward passes.

        Steps:
          1. Expand cached intermediate trajectories into independent DataProto
             rows matching the final rows' tensor schema (via
             ``expand_intermediate_trajectories_pre_log_prob``). Each row is
             tagged with ``trajectory_role`` and ``rollout_group_id`` in the
             non-tensor batch for later advantage scatter / normalization.
          2. Pad the expanded batch to a multiple of the global actor
             mini-batch size (``ppo_mini_batch_size * rollout.n``) and zero
             out the training signal on the padded tail so padding rows do
             not contribute to the actor loss.
        """
        timing_raw = self.timing_raw
        with marked_timer("expand_intermediate", timing_raw, color="magenta"):
            rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 1) or 1)
            rollout_cfg = self.config.actor_rollout_ref.rollout

            batch = expand_intermediate_trajectories_pre_log_prob(
                batch,
                tokenizer=self.tokenizer,
                processor=self.processor,
                rollout_config=rollout_cfg,
                rollout_n=rollout_n,
            )

        # Dump the full batch (tensors + non_tensor + meta_info) BEFORE pad so
        # that if ``pad_dataproto_to_divisor`` (which internally calls
        # ``DataProto.concat``) hits a conflicting meta_info key, the offending
        # field is already captured in ``training_diagnostics.log``.
        self._log_training_diagnostics("pre_pad", batch)

        # Temporarily move meta_info fields that are per-row ndarrays out of
        # ``batch.meta_info`` before pad. ``DataProto.concat`` inside
        # ``pad_dataproto_to_divisor`` asserts equality (``==``) on
        # overlapping meta_info values, and an ndarray value raises
        # "truth value of an array is ambiguous". We stash the offending
        # values aside and restore them after pad so downstream consumers
        # (e.g. ``_collect_metrics_from_samples``, which reads
        # ``trajectory_param_versions``) keep working unchanged.
        _NDARRAY_META_KEYS = ("trajectory_param_versions",)
        _stashed_meta: dict[str, Any] = {}
        for _bad_key in _NDARRAY_META_KEYS:
            if _bad_key in batch.meta_info:
                _stashed_meta[_bad_key] = batch.meta_info.pop(_bad_key)

        # Pad to actor mini-batch multiple. ``train_mini_batch`` requires
        # ``mini_batch_size = ppo_mini_batch_size * rollout.n`` to evenly
        # divide the per-DP batch; padding the global batch guarantees that
        # dispatching by DP size preserves divisibility.
        ppo_mini_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        global_mini_batch = ppo_mini_batch_size * rollout_n
        if global_mini_batch > 0 and len(batch) % global_mini_batch != 0:
            with marked_timer("pad_mini_batch", timing_raw, color="magenta"):
                batch, pad_size = pad_dataproto_to_divisor(batch, global_mini_batch)
                zero_out_padding_rows(batch, pad_size)
                batch.meta_info["fully_async/pad/num_padding_rows"] = int(pad_size)
        else:
            batch.meta_info["fully_async/pad/num_padding_rows"] = 0

        # Restore stashed ndarray meta_info fields onto the (possibly new)
        # batch object. These are batch-global monitoring arrays (one entry
        # per rollout), so they stay valid across pad.
        if _stashed_meta:
            batch.meta_info.update(_stashed_meta)

        return batch

    def _fit_compute_advantage(self, batch: DataProto) -> DataProto:
        """Compute GRPO advantage on the final subset only, then scatter.

        The batch at this point is the expanded+padded batch with mixed
        ``trajectory_role`` rows. Running GRPO on the whole thing would
        pollute group mean/std statistics with intermediate rows (which
        share the same ``uid`` and ``reward`` as their final siblings),
        collapsing advantage to zero.

        We therefore:
          1. Slice out the ``trajectory_role == "final"`` subset and run
             the standard ``_fit_compute_advantage`` (parent class) on it.
          2. Copy the resulting advantages/returns back into the full batch
             at the same indices.
          3. Scatter final-row advantages to sibling intermediate rows
             (grouped by ``rollout_group_id``) and apply ``1 / T_rollout``
             normalization so every rollout contributes equally to the loss.
        """
        import numpy as np  # local import to avoid polluting top-level namespace

        nt = batch.non_tensor_batch or {}
        roles = nt.get("trajectory_role")

        # Fast path: no intermediate rows (e.g. rollouts all finished in one
        # turn). Fall back to the standard advantage pipeline, followed only
        # by the optional 1/T_rollout normalization.
        if roles is None:
            batch = super()._fit_compute_advantage(batch)
            if bool(self.config.async_training.get("normalize_rollout_weight", True)):
                batch = scatter_advantage_to_intermediate_and_normalize(batch, normalize_rollout_weight=True)
            return batch

        final_idx = np.where(np.asarray(roles) == "final")[0]
        if len(final_idx) == 0:
            # Degenerate case: pad-only batch. Skip advantage.
            return batch

        # ------------------------------------------------------------------
        # (1) Slice the final-only subset while keeping row order stable.
        # ------------------------------------------------------------------
        final_subset = batch.select_idxs(torch.as_tensor(final_idx, dtype=torch.long))

        # Transfer reward tensors the parent expects. ``_fit_compute_advantage``
        # reads ``self.reward_tensor`` (set by ``_fit_compute_reward``) which
        # has shape (len(batch_before_expand), response_length). We need to
        # re-align it to the final subset: the first ``n_final`` rows of the
        # expanded batch are exactly the original (pre-expansion) final rows,
        # in the same order. ``final_idx`` recovers them.
        saved_reward_tensor = self.reward_tensor
        if self.reward_tensor is not None:
            # reward_tensor was produced before expansion, so its rows align
            # with the final rows now sitting at positions [0, n_final).
            # ``final_idx`` should therefore be ``arange(n_final)`` in the
            # canonical pipeline; we still index defensively in case
            # expander reorders in the future.
            if self.reward_tensor.shape[0] == len(final_subset):
                reward_tensor_final = self.reward_tensor
            else:
                reward_tensor_final = self.reward_tensor[: len(final_subset)]
            self.reward_tensor = reward_tensor_final

        # ------------------------------------------------------------------
        # (2) Run the standard advantage pipeline on the final subset.
        # ------------------------------------------------------------------
        final_subset = super()._fit_compute_advantage(final_subset)

        # Restore original reward_tensor reference for any downstream hooks.
        self.reward_tensor = saved_reward_tensor

        # ------------------------------------------------------------------
        # (3) Write the computed advantages/returns back into the full batch
        #     (only on final rows). Intermediate rows remain untouched here;
        #     ``scatter_advantage_to_intermediate_and_normalize`` will copy
        #     from the final rows in the next step.
        # ------------------------------------------------------------------
        if "advantages" in final_subset.batch.keys():
            if "advantages" not in batch.batch.keys():
                batch.batch["advantages"] = torch.zeros(
                    (len(batch), final_subset.batch["advantages"].shape[-1]),
                    dtype=final_subset.batch["advantages"].dtype,
                )
            batch.batch["advantages"][final_idx] = final_subset.batch["advantages"]
        if "returns" in final_subset.batch.keys():
            if "returns" not in batch.batch.keys():
                batch.batch["returns"] = torch.zeros_like(batch.batch["advantages"])
            batch.batch["returns"][final_idx] = final_subset.batch["returns"]
        # Bring along any other token-level fields the parent may have added
        # (e.g. ``token_level_rewards``) so downstream metrics are consistent.
        for extra_key in ("token_level_rewards", "token_level_scores"):
            if extra_key in final_subset.batch.keys():
                if extra_key not in batch.batch.keys():
                    batch.batch[extra_key] = torch.zeros(
                        (len(batch), final_subset.batch[extra_key].shape[-1]),
                        dtype=final_subset.batch[extra_key].dtype,
                    )
                batch.batch[extra_key][final_idx] = final_subset.batch[extra_key]

        # ------------------------------------------------------------------
        # (4) Scatter final advantages to intermediate siblings and apply
        #     1 / T_rollout normalization so every rollout contributes
        #     equally under loss_agg_mode="token-mean".
        # ------------------------------------------------------------------
        normalize_rollout_weight = bool(self.config.async_training.get("normalize_rollout_weight", True))
        batch = scatter_advantage_to_intermediate_and_normalize(
            batch, normalize_rollout_weight=normalize_rollout_weight
        )

        return batch

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

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

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

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Capture validation generations for deferred logging in _fit_validate.

        When use_trainer_do_validate=True, the trainer also runs _validate(True) which
        calls this method. We capture instead of logging immediately so that we can
        merge with rollouter-side generations and log once with the correct step.
        """
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            self._captured_val_generations = []
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        self._captured_val_generations = samples[:generations_to_log]

    async def _validate_process(self):
        """Run trainer-side validation using async rollout manager"""
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] _validate_process")
            from verl.utils.profiler import marked_timer

            # Wake up rollouter replicas and sync weights
            print("[FullyAsyncTrainer] wake up replicas before validation")
            await self.colocate_checkpoint_manager.update_weights(global_steps=self.current_param_version)

            with marked_timer("trainer/validate_time", self.timing_raw):
                train_val_metrics = self._validate(True)

            # Sleep rollouter replicas to free GPU memory for validation
            print("[FullyAsyncTrainer] sleep replicas after validation")
            await self.colocate_checkpoint_manager.sleep_replicas()

            print(f"[FullyAsyncTrainer] validate timing: {self.timing_raw['trainer/validate_time']}")
            return train_val_metrics
        else:
            print("[FullyAsyncTrainer] _validate_process without async_rollout_manager")
            return None

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

        # Trigger rollouter validation and get future
        val_future = self.rollouter.do_validate.remote()

        # Run trainer-side validation
        self._captured_val_generations = []
        train_val_metrics = await self._validate_process()

        # Wait for rollouter validation result and log
        val_metrics: ValidateMetrics = await asyncio.wrap_future(val_future.future())
        if train_val_metrics:
            # Merge trainer and rollouter validation results
            with marked_timer("timing_s/merge_val", self.timing_raw):
                new_metrics = self._merge_validation_results(train_val_metrics, val_metrics.metrics)
            if new_metrics:
                self.logger.log(data=new_metrics, step=self.current_param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {new_metrics}, timing: {self.timing_raw['timing_s/merge_val']}"
                )
        else:
            if val_metrics.metrics:
                self.logger.log(data=val_metrics.metrics, step=self.current_param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {val_metrics.metrics}"
                )
        self.logger.log(data=val_metrics.timing_raw, step=self.current_param_version)

        # Merge and log validation generations from rollouter (and trainer if applicable)
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log > 0:
            import numpy as np

            all_generations = list(self._captured_val_generations)
            if val_metrics.val_generations:
                all_generations.extend(val_metrics.val_generations)
            if all_generations:
                all_generations.sort(key=lambda x: x[0])
                rng = np.random.RandomState(42)
                rng.shuffle(all_generations)
                all_generations = all_generations[:generations_to_log]
                self.validation_generations_logger.log(
                    self.config.trainer.logger, all_generations, self.current_param_version
                )

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
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
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

        if self.colocate_checkpoint_manager:
            await self.colocate_checkpoint_manager.update_weights(self.current_param_version)
            await self.colocate_checkpoint_manager.sleep_replicas()

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
