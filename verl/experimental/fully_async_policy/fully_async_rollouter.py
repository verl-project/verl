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
from pprint import pformat
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.utils import need_reward_model
from verl.utils import normalize_token_ids
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import RolloutReplica, TokenOutput
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break

            await asyncio.sleep(1)

        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


class FullyAsyncLLMServerManager(LLMServerManager):
    """Extension of :class:`LLMServerManager` for fully async training with hybrid scaling."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        super().__init__(config, worker_group, rollout_resource_pool)
        # Pre-registered hybrid replicas: bound at init time but still sleeping.
        # Keyed by resource_id; populated during _initialize_llm_servers().
        self.hybrid_replicas: dict[str, RolloutReplica] = {}
        # Currently active (awake + in LB) subset of hybrid replicas.
        self.alive_replicas: dict[str, RolloutReplica] = {}
        # resource_id → server_address for alive hybrid replicas.
        self.alive_addresses: dict[str, str] = {}
        # Prometheus server addresses
        self.prometheus_server_addresses = []

        # Timing / counters
        self.last_hybrid_add_time: float = 0.0
        self.last_hybrid_remove_time: float = 0.0

    async def _initialize_llm_servers(self, start_rank: int = 0):
        # ── Step 1: hybrid replicas first (replica_rank 0 … N_e-1) ──────────
        # Use parent class to create + init_hybrid all hybrid replicas, then
        # migrate them from rollup_replicas → hybrid_replicas (sleeping, not
        # yet in the load balancer).  Starting from rank 0 gives hybrid actors
        # the lowest-numbered placement-group bundles which are co-located with
        # the training engine, maximising GPU affinity on multi-node deployments.
        num_hybrid = 0
        if self.worker_group is not None:
            await super()._initialize_llm_servers(start_rank=0)
            num_hybrid = len(self.rollout_replicas)
            # Migrate hybrid replicas out of the parent's tracking lists.
            for i, replica in enumerate(self.rollout_replicas):
                resource_id = f"hybrid_{i}"
                self.hybrid_replicas[resource_id] = replica
                print(
                    f"[FullyAsyncAgentLoopManager] Hybrid replica '{resource_id}' "
                    f"(rank={i}) initialised at {replica._server_address} "
                )
            self.prometheus_server_addresses.extend(self.server_addresses)
            print(f"AgentLoopManager Hybrid: {self.server_addresses}")
            # Clear parent state so Step 2 starts clean.
            self.rollout_replicas = []
            self.server_handles = []
            self.server_addresses = []

        # ── Step 2: standalone replicas via parent class ─────────────────────
        # Temporarily clear worker_group so that super()._initialize_llm_servers()
        # takes the standalone branch (init_standalone).  Pass start_rank=num_hybrid
        # so that Ray actor names remain globally unique and never collide with the
        # hybrid actors created above.
        saved_worker_group = self.worker_group
        self.worker_group = None
        try:
            await super()._initialize_llm_servers(start_rank=num_hybrid)
        finally:
            self.worker_group = saved_worker_group

        # Update Prometheus with the final (standalone) addresses.
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            all_addresses = self.prometheus_server_addresses + self.server_addresses
            update_prometheus_config(self.rollout_config.prometheus, all_addresses, self.rollout_config.name)

        print(
            f"[FullyAsyncLLMServerManager] Created: "
            f"{len(self.rollout_replicas)} standalone replicas (rank {num_hybrid}+), "
            f"{num_hybrid} hybrid replicas registered (sleeping, rank 0-{num_hybrid - 1})"
        )

    async def add_replicas(self, resource_ids: list[str]) -> int:
        """Activate multiple pre-registered hybrid replicas in a single batch RPC.

        Uses ``batch_add_servers`` on the GlobalRequestLoadBalancer for atomic
        bulk registration, which is more efficient than calling :meth:`add_replica`
        in a loop.

        Args:
            resource_ids: List of resource identifiers to activate.

        Returns:
            Number of successfully activated replicas.
        """
        # Filter out already-active and missing replicas.
        servers_to_add: dict[str, ray.actor.ActorHandle] = {}
        valid_resource_ids: list[str] = []
        for rid in resource_ids:
            if rid in self.alive_replicas:
                logger.warning("[FullyAsyncLLMServerManager] Replica '%s' already active, skipping", rid)
                continue
            replica = self.hybrid_replicas.get(rid)
            if replica is None:
                logger.error(
                    "[FullyAsyncLLMServerManager] Replica '%s' is not registered, skipping",
                    rid,
                )
                continue
            servers_to_add[replica._server_address] = replica._server_handle
            valid_resource_ids.append(rid)

        if not servers_to_add:
            return 0

        try:
            # Single atomic batch RPC: register all handles + add all to LB pool.
            await self.global_load_balancer.add_servers.remote(servers=servers_to_add)

            # Track locally for introspection / Prometheus.
            for rid in valid_resource_ids:
                replica = self.hybrid_replicas[rid]
                server_address = replica._server_address
                server_handle = replica._server_handle
                if server_address not in self.server_addresses:
                    self.server_handles.append(server_handle)
                    self.server_addresses.append(server_address)
                if replica not in self.rollout_replicas:
                    self.rollout_replicas.append(replica)
                self.alive_replicas[rid] = replica
                self.alive_addresses[rid] = server_address

            self.last_hybrid_add_time = time.time()

            print(
                f"[FullyAsyncLLMServerManager] added {len(valid_resource_ids)} replicas: {valid_resource_ids}. "
                f"Active hybrid replicas ({len(self.alive_replicas)}): {list(self.alive_replicas.keys())}"
            )
            return len(valid_resource_ids)

        except Exception as e:
            logger.error("[FullyAsyncLLMServerManager] Failed to batch activate replicas: %s", e)
            return 0

    async def remove_replicas(self, resource_ids: list[str]) -> int:
        """Deactivate multiple active hybrid replicas in a single batch RPC.

        Uses ``batch_remove_servers`` on the GlobalRequestLoadBalancer for atomic
        bulk removal, which is more efficient than calling :meth:`remove_replica`
        in a loop.

        Args:
            resource_ids: List of resource identifiers to deactivate.

        Returns:
            Number of successfully deactivated replicas.
        """
        # Filter out missing replicas and collect server addresses.
        server_ids_to_remove: list[str] = []
        valid_resource_ids: list[str] = []
        for rid in resource_ids:
            if rid not in self.alive_replicas:
                logger.warning("[FullyAsyncLLMServerManager] Replica '%s' not active, skipping", rid)
                continue
            server_ids_to_remove.append(self.alive_addresses[rid])
            valid_resource_ids.append(rid)

        if not server_ids_to_remove:
            return 0

        try:
            # Single atomic batch RPC: remove all from LB pool + purge handles.
            await self.global_load_balancer.remove_servers.remote(server_ids=server_ids_to_remove)

            # Clean up local tracking lists.
            for rid in valid_resource_ids:
                server_address = self.alive_addresses[rid]
                replica = self.alive_replicas[rid]
                if server_address in self.server_addresses:
                    idx = self.server_addresses.index(server_address)
                    self.server_addresses.pop(idx)
                    self.server_handles.pop(idx)
                if replica in self.rollout_replicas:
                    self.rollout_replicas.remove(replica)
                self.alive_replicas.pop(rid)
                self.alive_addresses.pop(rid)

            self.last_hybrid_remove_time = time.time()

            print(
                f"[FullyAsyncLLMServerManager] removed {len(valid_resource_ids)} replicas: {valid_resource_ids}. "
                f"Remaining hybrid replicas ({len(self.alive_replicas)}): {list(self.alive_replicas.keys())}"
            )
            return len(valid_resource_ids)

        except Exception as e:
            logger.error("[FullyAsyncLLMServerManager] Failed to batch remove replicas: %s", e)
            return 0

    # -------------------------------------------------------------------------
    # Statistics / introspection
    # -------------------------------------------------------------------------
    def get_num_hybrid_replicas(self) -> int:
        """Return the number of currently active hybrid replicas."""
        return len(self.alive_replicas)

    def get_hybrid_replicas_info(self) -> list[dict]:
        """Return metadata for all active hybrid replicas."""
        return [{"resource_id": rid, "server_address": addr} for rid, addr in self.alive_addresses.items()]

    def get_hybrid_statistics(self) -> dict:
        """Return hybrid-specific counters for monitoring."""
        return {
            "hybrid/num_hybrid_replicas": len(self.alive_replicas),
            "hybrid/last_add_time": self.last_hybrid_add_time,
            "hybrid/last_remove_time": self.last_hybrid_remove_time,
        }

    def get_active_server_count(self) -> int:
        """Total active rollout servers (standalone + hybrid)."""
        return len(self.rollout_replicas) + len(self.alive_replicas)


class FullyAsyncAgentLoopManager(AgentLoopManager):
    async def generate_sequences_single(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch. Single sample data
        Returns:
            DataProto: Output batch.
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(prompts)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(SeparateRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        self.use_reference_policy = False

        self.use_rm = need_reward_model(self.config)
        if self.use_rm:
            assert self.config.reward.reward_model.enable_resource_pool, (
                "GenRM/DisRM in fully async mode requires standalone mode (enable_resource_pool=True). "
                "Colocate mode is not supported because async rollout never pauses."
            )

        self.use_critic = False
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== fully async config ====================

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        self.async_rollout_manager = None

        # Elastic worker group (injected via set_hybrid_worker_group before init_workers)
        # When set, its GPUs back hybrid replicas for trainer-side validation.
        self._hybrid_worker_group = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # `lock` protects shared state: paused / active_tasks / staleness_samples / timing fields.
        self.lock = asyncio.Lock()
        # `_resume_event` signals that the rollouter is currently running (paused == False).
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = len(self.llm_server_manager.get_replicas()) * 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_replicas(self):
        """Get rollout worker group"""
        return self.llm_server_manager.get_replicas()

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def reset_staleness(self):
        """
        Reset staleness samples after parameter update.
        Returns timing_raw dictionary for metrics.
        """
        async with self.lock:
            self.paused = False
            # Wake the drain loop in _processor_worker so it can exit early and resume submitting
            # new samples to idle replicas instead of waiting for long-tail in-flight tasks.
            self._resume_event.set()
            # every time param change, reset staleness_samples
            self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
            timing_raw = {}
            rollout_version_time = max(time.time() - self.step_start_time, 1e-6)
            if self.idle_start_time > self.step_start_time:
                rollout_active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
            else:
                rollout_active_time = rollout_version_time
                idle_ratio = 0
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            print(
                f"[FullyAsyncRollouter][Public][reset_staleness] "
                f"reset staleness_samples to: {self.staleness_samples} "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f}"
            )
            self.step_start_time = time.time()

        return timing_raw

    def do_validate(self):
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return timing_raw | val_metrics

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._create_worker_classes()
        await self._create_reward_loop_manager()
        await self._create_teacher_model_manager()
        await self._init_async_rollout_manager()

    async def _create_reward_loop_manager(self):
        """Create RewardLoopManager for the rollouter.

        TODO: RewardModelManager.__init__ uses asyncio.run() which forces us to use
        run_in_executor here. Upstream should provide an async init method so this
        can be a simple await call instead.
        """
        import asyncio

        from verl.experimental.reward_loop import RewardLoopManager

        loop = asyncio.get_running_loop()
        self.reward_loop_manager = await loop.run_in_executor(
            None,
            lambda: RewardLoopManager(config=self.config, rm_resource_pool=None),
        )

    async def _create_teacher_model_manager(self):
        """Create MultiTeacherModelManager for distillation if enabled.

        Allocates a big resource pool for all teachers and passes it to
        MultiTeacherModelManager, which splits it internally per teacher.

        NOTE: MultiTeacherModelManager.__init__ calls _run_all internally which uses
        asyncio.run(), conflicting with the already-running event loop. Run in a thread executor.
        """
        from verl.trainer.distillation.losses import is_distillation_enabled
        from verl.trainer.ppo.utils import Role

        self.teacher_model_manager = None
        if is_distillation_enabled(self.config.get("distillation")):
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            resource_pool_spec = {}
            mapping = {}
            distillation_cfg = self.config.get("distillation", {})
            n_gpus = distillation_cfg.get("n_gpus_per_node", 0)
            nnodes = distillation_cfg.get("nnodes", 1)
            assert n_gpus > 0, "distillation.n_gpus_per_node must be greater than 0 for TeacherModel"
            teacher_pool = [n_gpus] * nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            mapping[Role.TeacherModel] = "teacher_pool"

            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
            resource_pool_manager.create_resource_pool()
            teacher_resource_pool = resource_pool_manager.get_resource_pool(Role.TeacherModel)

            loop = asyncio.get_running_loop()
            self.teacher_model_manager = await loop.run_in_executor(
                None,
                lambda: MultiTeacherModelManager(config=self.config, resource_pool=teacher_resource_pool),
            )

    def _create_actor_rollout_classes(self):
        # Skip rollout creation and let agentloop handle it
        pass

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        """
        Create the server manager and agent loop manager for fully async training.

        Uses :class:`FullyAsyncLLMServerManager` which supports two-phase init:
        - Phase 1: hybrid replicas on trainer GPUs (sleeping)
        - Phase 2: standalone replicas on rollout GPUs

        The ``GlobalRequestLoadBalancer`` (which also holds the server-handle
        registry) serves as the single source of truth for handle mapping and
        routing.  Clients look up handles atomically — no per-worker notification
        needed on hybrid add/remove.
        """
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        self.async_rollout_mode = True
        # Use FullyAsyncLLMServerManager for two-phase (hybrid + standalone) init.
        # It creates GlobalRequestLoadBalancer (with merged handle registry) internally.
        self.llm_server_manager = await FullyAsyncLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
        )
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                    self._resume_event.clear()

                resume_future = asyncio.ensure_future(self._resume_event.wait())
                try:
                    # Drain: wait for either (a) at least one active task to finish, or
                    # (b) a resume signal (reset_staleness / monitor flipping paused=False) to
                    # break the drain early so new samples can be submitted to free replicas.
                    # We do NOT hold the lock during the wait, so publishers can acquire it to
                    # update paused / staleness_samples concurrently.
                    while self.active_tasks and not resume_future.done():
                        wait_set = set(self.active_tasks) | {resume_future}
                        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {resume_future}
                        if actual_done:
                            async with self.lock:
                                for task in actual_done:
                                    self.active_tasks.discard(task)
                                    await task
                        if resume_future in done:
                            print(
                                "[FullyAsyncRollouter][Processor] "
                                "Drain interrupted by resume signal, resuming generation early "
                                f"(active tasks remaining: {len(self.active_tasks)})"
                            )
                            break

                    # block until resuming
                    if not resume_future.done():
                        self.idle_start_time = time.time()
                        await resume_future
                finally:
                    if not resume_future.done():
                        resume_future.cancel()
                        await asyncio.gather(resume_future, return_exceptions=True)
                continue
            # Get sample from appropriate queue and immediately mark task as done
            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()
            self.staleness_samples += 1

            if rollout_sample is None:
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                break

            # Check whether the number of concurrent tasks exceeds the limit
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task

            # Submit single sample processing
            if self.paused:
                await self._resume_event.wait()
            async with self.lock:
                task = safe_create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        rollout_sample.full_batch = ret
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

            await self.pending_queue.join()
            print("[FullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # Send a finish signal
            await self.message_queue_client.put_sample(sample=None)

            async with self.lock:
                self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True
            self._resume_event.set()

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[FullyAsyncRollouter][ShouldPause] resume rollouter.")
                    self._resume_event.set()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = await self.message_queue_client.get_statistics()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        queue_stats = await self.message_queue_client.get_statistics()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats

    # -------------------------------------------------------------------------
    # Elastic worker group injection
    # -------------------------------------------------------------------------
    def set_hybrid_worker_group(self, worker_group: RayWorkerGroup):
        """Inject the hybrid worker group."""
        self._hybrid_worker_group = worker_group

    def get_hybrid_worker_group(self):
        """Return the worker group for hybrid replicas."""
        return self._hybrid_worker_group

    async def add_replicas(self, resource_ids: list[str]) -> int:
        return await self.llm_server_manager.add_replicas(resource_ids)

    async def remove_replicas(self, resource_ids: list[str]) -> int:
        return await self.llm_server_manager.remove_replicas(resource_ids)

    def get_hybrid_replica(self, resource_id: str):
        """Return the RolloutReplica object for a registered hybrid resource."""
        return self.llm_server_manager.hybrid_replicas.get(resource_id)

    def get_all_hybrid_replicas(self) -> dict:
        """Return all registered hybrid replicas (sleeping + active)."""
        return dict(self.llm_server_manager.hybrid_replicas)

    # -------------------------------------------------------------------------
    # Statistics / introspection – delegate to llm_server_manager
    # -------------------------------------------------------------------------
    async def get_hybrid_statistics(self) -> dict:
        """Combined rollout + hybrid statistics."""
        base_stats = await self.get_statistics()
        hybrid_stats = self.llm_server_manager.get_hybrid_statistics()
        return {**base_stats, **hybrid_stats}

    def get_num_active_replicas(self) -> int:
        """Total active rollout replicas (standalone + hybrid)."""
        return self.llm_server_manager.get_active_server_count()

    def get_hybrid_replicas_info(self) -> list[dict]:
        """Metadata for all active hybrid replicas."""
        return self.llm_server_manager.get_hybrid_replicas_info()

    def get_total_produced_samples(self) -> int:
        """Total samples produced (uses base class counter)."""
        return self.total_generated_samples
