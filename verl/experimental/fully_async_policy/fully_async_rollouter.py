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
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import numpy as np
import ray
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.image_refs import (
    attach_image_bank_ref,
    attach_image_refs_to_dataproto,
    image_refs_enabled,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


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

    async def generate_sequence_row(self, prompt: DataProto, rollout_n: int) -> DataProto:
        """Dispatch one rollout row to an agent loop worker."""
        if len(prompt) != 1:
            raise ValueError(f"generate_sequence_row expects exactly one row, got {len(prompt)}")

        row = prompt.select(deepcopy=True)
        row.meta_info = dict(row.meta_info)
        row.meta_info["rollout_n"] = [rollout_n]
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(row)
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
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.image_refs_enabled = image_refs_enabled(config)
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = False

        self.use_rm = need_reward_model(self.config)
        if self.use_rm:
            assert self.config.reward.reward_model.enable_resource_pool, (
                "GenRM/DisRM in fully async mode requires standalone mode (enable_resource_pool=True). "
                "Colocate mode is not supported because async rollout never pauses."
            )

        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== fully async config ====================

        logger.info("[FullyAsyncRollouter] Creating datasets...")
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
        if self.config.async_training.use_trainer_do_validate:
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            train_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
            total_gpus = rollout_gpus + train_gpus
            logger.info("[FullyAsyncRollouter] split before val_dataset total len: %d", len(val_dataset))
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[:rollout_gpus]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            logger.info("[FullyAsyncRollouter] split after val_dataset total len: %d", len(val_dataset))
        logger.info(
            "[FullyAsyncRollouter] Rollouter _create_dataloader...\n%s\n%s",
            train_dataset,
            val_dataset,
        )

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        logger.info("[FullyAsyncRollouter] Total rollout steps: %d", self.total_rollout_steps)
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        self.max_concurrent_rollouts = None
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
        self.postprocess_tasks = set()
        self.rollout_semaphore = None

        cpu_cores = multiprocessing.cpu_count()
        # cpu case use cpu_cores; io case use cpu_cores*2
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.validate_task = None

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # `lock` protects shared state: paused / active_tasks / staleness_samples / timing fields.
        self.lock = asyncio.Lock()
        # `_resume_event` signals that the rollouter is currently running (paused == False).
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self.image_refs_postprocess_semaphore = asyncio.Semaphore(1)

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

            rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 1) or 1)
            self.max_concurrent_samples = self.max_required_samples
            # Optional user-provided hard cap on in-flight rollout trajectories,
            # e.g. to respect an external desktop-env service limit. Sample-level
            # batching is still preserved for MQ/trainer; only env acquisition is
            # gated at trajectory level.
            user_cap = int(self.config.async_training.get("max_concurrent_rollouts", 0) or 0)
            if user_cap > 0:
                self.max_concurrent_rollouts = user_cap
            else:
                self.max_concurrent_rollouts = max(1, self.max_concurrent_samples * rollout_n)
            self.rollout_semaphore = asyncio.Semaphore(self.max_concurrent_rollouts)
            self.max_queue_size = self.max_required_samples

            logger.info(
                "[FullyAsyncRollouter] required_samples : %d "
                "max_required_samples: %s "
                "max_queue_size: %s "
                "total_train_steps: %s "
                "total_rollout_steps: %s "
                "max_concurrent_samples: %s "
                "max_concurrent_rollouts: %s "
                "user_cap(rollout-level): %d "
                "rollout.n: %d ",
                self.required_samples,
                self.max_required_samples,
                self.max_queue_size,
                self.total_train_steps,
                self.total_rollout_steps,
                self.max_concurrent_samples,
                self.max_concurrent_rollouts,
                user_cap,
                rollout_n,
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
            self.staleness_samples = (
                len(self.active_tasks) + len(self.postprocess_tasks) + await self.message_queue_client.get_queue_size()
            )
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

            logger.info(
                "[FullyAsyncRollouter][Public][reset_staleness] reset staleness_samples to: %d idle_ratio: %.4f",
                self.staleness_samples,
                timing_raw["fully_async/rollouter/idle_ratio"],
            )
            self.step_start_time = time.time()

        return timing_raw

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Capture validation generations to send back to trainer instead of logging directly.

        The rollouter process does not have an active wandb session, so we capture the
        sampled generations and return them via ValidateMetrics to the trainer for logging.
        """
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            self._captured_val_generations = []
            return

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        self._captured_val_generations = samples[:generations_to_log]

    def do_validate(self) -> ValidateMetrics:
        """Run validation and return metrics"""
        timing_raw = {}
        self._captured_val_generations = []
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return ValidateMetrics(
            timing_raw=timing_raw,
            metrics=val_metrics,
            val_generations=self._captured_val_generations,
        )

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
        logger.info("[FullyAsyncRollouter] Saved dataloader checkpoint to %s", dataloader_local_path)

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            logger.info("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
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
                logger.info("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
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

        logger.info("[FullyAsyncRollouter] Loading checkpoint from: %s", global_step_folder)

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        logger.info("[FullyAsyncRollouter] Setting global_steps to %d", self.global_steps)

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            logger.info("[FullyAsyncRollouter] Loaded dataloader state from %s", dataloader_local_path)
        else:
            logger.warning(
                "[FullyAsyncRollouter] No dataloader state found at %s, will start from scratch",
                dataloader_local_path,
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

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
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
        self.llm_server_manager = await LLMServerManager.create(config=self.config)
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(fully_async=True),
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
                logger.info(
                    "[FullyAsyncRollouter][Feed] Maximum count has been reached, stop adding new samples: %d >= %d",
                    self.global_steps,
                    self.total_rollout_steps,
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        logger.info(
            "[FullyAsyncRollouter][Feed] Sample addition is complete, %d samples have been added",
            self.global_steps,
        )

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                logger.info(
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
                            logger.info(
                                "[FullyAsyncRollouter][Processor] "
                                "Drain interrupted by resume signal, resuming generation early "
                                "(active tasks remaining: %d)",
                                len(self.active_tasks),
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
                logger.info(
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
                while self.postprocess_tasks:
                    done_tasks, _ = await asyncio.wait(self.postprocess_tasks, return_when=asyncio.FIRST_COMPLETED)
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
        """Process a single sample streamingly.

        ``staleness_samples`` is incremented in ``_processor_worker`` the moment
        a sample is taken off ``pending_queue``. Its physical meaning is
        "samples currently in flight or sitting in the MQ for the trainer".
        So whenever this coroutine ends WITHOUT the sample reaching the MQ
        (env failure, all rollouts discarded, MQ put rejected, ...), we must
        decrement ``staleness_samples`` immediately. Otherwise dropped
        samples accumulate against ``max_required_samples`` and the rollouter
        can wedge into ``paused`` even though no real samples are in flight.
        """

        def _drop_sample():
            self.dropped_stale_samples += 1
            self.processed_sample_count += 1
            self.staleness_samples = max(0, self.staleness_samples - 1)

        try:
            await self._process_single_sample_rollouts(rollout_sample, _drop_sample)
        except Exception as exc:
            logger.exception(
                "[POTENTIAL ERROR][FullyAsyncRollouter] _process_single_sample_streaming dropped "
                "sample_id=%s due to unexpected aggregation failure: %r",
                rollout_sample.sample_id,
                exc,
            )
            _drop_sample()

    async def _process_single_sample_rollouts(
        self,
        rollout_sample: RolloutSample,
        _drop_sample,
    ):
        """Run rollout rows independently, then aggregate successful rows back into one sample."""

        if self.rollout_semaphore is None:
            cap = self.max_concurrent_rollouts or len(rollout_sample.full_batch)
            self.rollout_semaphore = asyncio.Semaphore(max(1, int(cap)))

        async def _run_one_rollout(row_idx: int) -> DataProto:
            async with self.rollout_semaphore:
                row_batch = rollout_sample.full_batch.slice(row_idx, row_idx + 1)
                return await self.async_rollout_manager.generate_sequence_row(row_batch, rollout_n=row_idx)

        try:
            rollout_tasks = []
            for row_idx in range(len(rollout_sample.full_batch)):
                task = asyncio.create_task(
                    _run_one_rollout(row_idx),
                    name=f"{rollout_sample.sample_id}_rollout_{row_idx}",
                )
                rollout_tasks.append(task)
            raw_outputs = await asyncio.gather(*rollout_tasks, return_exceptions=True)
        except Exception as exc:
            # Entire sample failed (e.g. every rollout hit a fatal tool error
            # and was discarded). Skip put_sample so the downstream trainer
            # does not see an empty/inconsistent batch. Do NOT re-raise: a
            # single failed sample must not kill the whole rollouter.
            logger.error(
                "[POTENTIAL ERROR][FullyAsyncRollouter] _process_single_sample_streaming dropped "
                "sample_id=%s due to row rollout dispatch failure: %r",
                rollout_sample.sample_id,
                exc,
            )
            _drop_sample()
            return

        valid_outputs = []
        n_exceptions = 0
        n_empty = 0
        for output in raw_outputs:
            if isinstance(output, BaseException):
                n_exceptions += 1
                logger.error(
                    "[POTENTIAL ERROR][FullyAsyncRollouter] sample_id=%s rollout task failed: %r",
                    rollout_sample.sample_id,
                    output,
                )
                continue
            if output.batch is None or len(output) == 0 or output.meta_info.get("all_rollouts_failed", False):
                n_empty += 1
                continue
            valid_outputs.append(output)

        if not valid_outputs:
            logger.error(
                "[POTENTIAL ERROR][FullyAsyncRollouter] _process_single_sample_streaming dropped "
                "sample_id=%s: empty batch (all rollouts failed; exceptions=%d, empty=%d)",
                rollout_sample.sample_id,
                n_exceptions,
                n_empty,
            )
            _drop_sample()
            return

        ret = DataProto.concat(valid_outputs)
        metrics = []
        for output in valid_outputs:
            item_metrics = output.meta_info.get("metrics", [])
            if isinstance(item_metrics, dict):
                metrics.append(item_metrics)
            else:
                metrics.extend(item_metrics)
        ret.meta_info["metrics"] = metrics
        rollout_sample.full_batch = ret

        # If all rollouts in this sample were discarded (e.g. env creation
        # failures), do not put the empty batch into the message queue.
        if ret.batch is None or len(ret) == 0 or ret.meta_info.get("all_rollouts_failed", False):
            logger.error(
                "[POTENTIAL ERROR][FullyAsyncRollouter] _process_single_sample_streaming dropped "
                "sample_id=%s: empty batch (all rollouts failed)",
                rollout_sample.sample_id,
            )
            _drop_sample()
            return

        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        task = safe_create_task(
            self._postprocess_and_publish_sample(rollout_sample),
            name=f"{rollout_sample.sample_id}_postprocess",
            task_set=self.postprocess_tasks,
        )
        task.add_done_callback(self.postprocess_tasks.discard)

    def _attach_image_refs_for_sample(self, rollout_sample: RolloutSample) -> RolloutSample:
        if not self.image_refs_enabled:
            return rollout_sample

        image_ref_start = time.time()
        rollout_sample.full_batch, image_bank, image_bank_stats = attach_image_refs_to_dataproto(
            rollout_sample.full_batch,
            processor=self.processor,
            sample_id=rollout_sample.sample_id,
        )
        image_bank_stats["build_ms"] = (time.time() - image_ref_start) * 1000.0
        bank_put_start = time.time()
        image_bank_ref = ray.put(image_bank) if image_bank else None
        image_bank_stats["bank_ref_put_ms"] = (time.time() - bank_put_start) * 1000.0
        rollout_sample.full_batch = attach_image_bank_ref(rollout_sample.full_batch, image_bank_ref)
        image_bank_stats["total_ms"] = (time.time() - image_ref_start) * 1000.0
        rollout_sample.image_bank_ref = image_bank_ref
        rollout_sample.image_bank_stats = image_bank_stats
        return rollout_sample

    async def _postprocess_and_publish_sample(self, rollout_sample: RolloutSample):
        try:
            image_ref_start = time.time()
            async with self.image_refs_postprocess_semaphore:
                rollout_sample = await asyncio.to_thread(self._attach_image_refs_for_sample, rollout_sample)
            postprocess_ms = (time.time() - image_ref_start) * 1000.0
            if self.image_refs_enabled and hasattr(rollout_sample, "image_bank_stats"):
                rollout_sample.image_bank_stats["postprocess_task_ms"] = postprocess_ms

            rollout_sample.rollout_status = await self.get_statistics()

            sample_ref = await asyncio.to_thread(ray.put, rollout_sample)
            # Wrap the ObjectRef so Ray does not auto-dereference it when passing it
            # as a top-level actor method argument to MessageQueue.put_sample.
            success = await self.message_queue_client.put_sample(
                sample=[sample_ref],
            )
            if success:
                self.total_generated_samples += 1
            else:
                # MQ rejected the sample (e.g. queue full + reject policy). The
                # sample never reaches the trainer, so it must not keep occupying
                # a staleness slot.
                self.dropped_stale_samples += 1
                self.staleness_samples = max(0, self.staleness_samples - 1)
            self.processed_sample_count += 1
        except Exception as exc:
            logger.error(
                "[POTENTIAL ERROR][FullyAsyncRollouter] postprocess dropped sample_id=%s: %r",
                rollout_sample.sample_id,
                exc,
            )
            self.dropped_stale_samples += 1
            self.staleness_samples = max(0, self.staleness_samples - 1)
            self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        logger.info(
            "[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: %s, "
            "maximum concurrent rollouts: %s",
            self.max_concurrent_samples,
            self.max_concurrent_rollouts,
        )

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

            logger.info("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            logger.info("[FullyAsyncRollouter] Streaming process completed")

            await self.pending_queue.join()
            logger.info("[FullyAsyncRollouter] pending_queue joined")

            while self.postprocess_tasks:
                done_tasks, _ = await asyncio.wait(self.postprocess_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    await task
            logger.info("[FullyAsyncRollouter] postprocess tasks drained")

        except Exception:
            logger.exception("[FullyAsyncRollouter] Streaming process exception")
            raise

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            if self.postprocess_tasks:
                await asyncio.gather(*list(self.postprocess_tasks), return_exceptions=True)

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

        logger.info("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

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
        except Exception:
            logger.exception("[FullyAsyncRollouter] Asynchronous task execution error")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        logger.info("[FullyAsyncRollouter] Rollouter fit completed")

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
                logger.info("[FullyAsyncRollouter][MonitorLoop][Statistics] %s", pformat(stats))
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    logger.info("[FullyAsyncRollouter][ShouldPause] resume rollouter.")
                    self._resume_event.set()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = await self.message_queue_client.get_statistics()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                logger.info(
                    "[FullyAsyncRollouter][ShouldPause] due to full queue: size=%d, max=%s",
                    queue_size,
                    self.max_queue_size,
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                logger.info(
                    "[FullyAsyncRollouter][ShouldPause] due to staleness_samples %d >= max_required_samples %s ",
                    self.staleness_samples,
                    self.max_required_samples,
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        queue_stats = await self.message_queue_client.get_statistics()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/postprocess_tasks_size": len(self.postprocess_tasks),
            "monitor/total_inflight_tasks_size": len(self.active_tasks) + len(self.postprocess_tasks),
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
            "static/max_concurrent_rollouts": self.max_concurrent_rollouts,
        }

        return stats
