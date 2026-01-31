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
import functools
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from pprint import pformat

import numpy as np
import ray
import torch
from ray import ObjectRef

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


class RollouterState(Enum):
    """Rollouter state"""

    IDLE = auto()  # created but not started
    INIT = auto()  # initializing and starting helpers
    ADD_TASK = auto()  # enqueue and schedule sample tasks
    WAIT_CONCURRENCY = auto()  # wait for available slots
    PAUSED = auto()  # paused by internal/external signal
    FINISHING = auto()  # drain remaining tasks
    FINISHED = auto()  # completed and shutdown


@ray.remote(num_cpus=10, max_concurrency=100, concurrency_groups={"monitor": 1})
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
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
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        if self.config.async_training.use_trainer_do_validate:
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            train_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
            total_gpus = rollout_gpus + train_gpus
            print(f"[FullyAsyncRollouter] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[:rollout_gpus]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncRollouter] split after val_dataset total len: {len(val_dataset)}")
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # Fully async configuration

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameters
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Async training config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples = ppo_mini_batch_size * require_batches
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # Trajectories generated between parameter syncs (before filter)
        self.sync_sampled_trajectory_count = 0
        # Total trajectories sampled (before filter)
        self.total_sampled_trajectory_count = 0
        # Trajectories generated between parameter syncs (after filter)
        self.sync_kept_trajectory_count = 0
        # Total trajectories after filter (after filter)
        self.total_kept_trajectory_count = 0
        # Trajectories generated between parameter syncs (used for throttling)
        self.generated_trajectory_count = 0

        # Trajectories requested by trainer between parameter syncs
        self.request_trajectories_per_sync = (
            (
                self.config.filter.trajectories_per_mini_batch
                or self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            )
            * self.require_batches
            * self.config.async_training.trigger_parameter_sync_step
        )
        # Max trajectories generated between parameter syncs
        self.max_generate_trajectories_per_sync = int(
            self.request_trajectories_per_sync * (self.staleness_threshold + 1)
        )
        # Avg trajectories per sample after filter
        self.avg_trajectories_per_sample = (
            self.config.filter.avg_trajectories_per_sample or self.config.actor_rollout_ref.rollout.n
        )

        # Queue size
        self.max_queue_size = None

        # Statistics counters
        self.current_param_version = 0

        # total_processed_sample_count =
        #   total_generated_samples + total_dropped_samples + total_filtered_out_sample_count
        # total_generated_samples + total_dropped_samples = count of non-empty outputs (len(ret) > 0)
        self.total_generated_samples = 0  # samples successfully enqueued to MQ
        self.total_dropped_samples = 0  # samples rejected by MQ
        self.total_processed_sample_count = 0  # samples processed (non-cancel)
        self.total_filtered_out_sample_count = 0  # samples fully filtered out

        # Global steps start at 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        # State machine (initialized in _init_async_objects)
        self.state = RollouterState.IDLE
        self.feed_finished = False

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Async queues and task set
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.cancel_queue = asyncio.Queue()

        cpu_cores = multiprocessing.cpu_count()
        # cpu case use cpu_cores; io case use cpu_cores*2
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.parallel_validate_and_rollout = config.async_training.get("parallel_validate_and_rollout", False)
        self.validate_task = None

    def _init_async_objects(self):
        """Initialize async synchronization primitives"""

        # pause_event: internal pause signal
        # set=paused, clear=running
        # Use: check state / wait for pause / asyncio.wait pause signal
        self.pause_event = asyncio.Event()
        self.pause_event.clear()  # initial state: running

        # external_pause_event: external pause signal, called by pause(), only resume() can recover
        # set=external pause, clear=not external pause (internal pause can auto-resume)
        self.external_pause_event = asyncio.Event()
        self.external_pause_event.clear()  # initial state: not external pause

        # resume_signal: resume signal
        # Used by PAUSED state to wait for resume
        self.resume_signal = asyncio.Event()
        self.resume_signal.clear()

        # running_event: set=running, clear=stopped
        self.running_event = asyncio.Event()
        self.running_event.set()  # initial state: running

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_max_required_samples(self):
        self.max_required_samples = int(
            self.required_samples
            * (self.staleness_threshold + 1)
            * self.config.async_training.trigger_parameter_sync_step
        )
        self.total_train_steps = int(
            self.total_rollout_steps / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
        )

        self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
        self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)

        print(
            f"[FullyAsyncRollouter] required_samples : {self.required_samples}, "
            f"max_required_samples: {self.max_required_samples}, "
            f"max_generate_trajectories_per_sync: {self.max_generate_trajectories_per_sync}, "
            f"request_trajectories_per_sync: {self.request_trajectories_per_sync}, "
            f"max_queue_size: {self.max_queue_size}, "
            f"total_train_steps: {self.total_train_steps}, "
            f"total_rollout_steps: {self.total_rollout_steps}, "
            f"max_concurrent_samples: {self.max_concurrent_samples}"
        )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(
        self, version: int, validate: bool = False, global_steps: int = 0, use_trainer_do_validate: bool = False
    ):
        """Update current parameter version"""
        old_version = self.current_param_version
        self.current_param_version = version
        timing_raw = {}
        idle_ratio = None
        if self.idle_start_time is not None and self.version_start_time is not None:
            rollout_active_time = self.idle_start_time - self.version_start_time
            rollout_version_time = time.time() - self.version_start_time
            idle_ratio = 1 - rollout_active_time / rollout_version_time
            timing_raw["rollouter/active_time"] = rollout_active_time
            timing_raw["rollouter/version_time"] = rollout_version_time
            timing_raw["rollouter/idle_ratio"] = idle_ratio
            self.idle_start_time = None
        generated_trajectory_count_tmp = self.generated_trajectory_count
        # reset generated_trajectory_count to number of extra trajectories generated
        # warning, When validate=true, it occurs outside of the training process, and 0 should be subtracted.
        if self.config.async_training.force_cleanup_stale_samples:
            self.generated_trajectory_count = 0
            while not self.cancel_queue.empty():
                try:
                    _ = self.cancel_queue.get_nowait()
                    self.task_done()
                except asyncio.QueueEmpty:
                    break
            print("[FullyAsyncRollouter]: clear cancel_queue(drop partial task)")
        else:
            self.generated_trajectory_count -= (
                self.request_trajectories_per_sync if not validate and global_steps > 0 else 0
            )

        self.sync_sampled_trajectory_count = 0
        self.sync_kept_trajectory_count = 0

        print(
            f"[FullyAsyncRollouter][Public][update_param_version] "
            f"Parameter version updated from {old_version} to {version} "
            f", reset generated_trajectory_count to: {self.generated_trajectory_count}"
            f", exceeding the specified limit: "
            f"{generated_trajectory_count_tmp - self.max_generate_trajectories_per_sync}"
            f",idle_ratio: {idle_ratio}"
        )
        need_validate = (
            (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and self.current_param_version % self.config.rollout.test_freq == 0
                and self.current_param_version > 0
            )  # don't test here in the initial parameter sync
            or (validate and self.val_reward_fn is not None)
        )
        print(
            f"[FullyAsyncRollouter] need_validate: {need_validate},"
            f"parallel_validate_and_rollout: {self.parallel_validate_and_rollout}"
        )
        if not need_validate:
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=None, global_steps=global_steps, param_version=version
            )
        elif need_validate and not self.parallel_validate_and_rollout:
            data = self._validate_wrapper(timing_raw, version, global_steps, use_trainer_do_validate)

        if not need_validate or not self.parallel_validate_and_rollout:
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

        self.version_start_time = time.time()

        if need_validate and self.parallel_validate_and_rollout:
            if self.validate_task and not self.validate_task.done():
                print("[FullyAsyncRollouter] validate_task is running, wait last validate_task to finish")
                self.validate_task.get()
            self.validate_task = asyncio.create_task(
                self.do_validate_async(timing_raw, version, global_steps, use_trainer_do_validate)
            )

    def _validate_wrapper(
        self, timing_raw: dict, version: int, global_steps: int = 0, use_trainer_do_validate: bool = False
    ):
        val_metrics = None
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate(use_trainer_do_validate)
        data = ValidateMetrics(
            timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
        )
        return data

    async def do_validate_async(
        self,
        timing_raw: dict,
        version: int,
        global_steps: int = 0,
        use_trainer_do_validate: bool = False,
    ):
        loop = asyncio.get_running_loop()

        data = await loop.run_in_executor(
            self.validate_executor,
            functools.partial(
                self._validate_wrapper,
                timing_raw=timing_raw,
                version=version,
                global_steps=global_steps,
                use_trainer_do_validate=use_trainer_do_validate,
            ),
        )
        await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

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
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            sample_id = f"sample_{epoch}_{self.global_steps}"
            full_batch.non_tensor_batch["uid"] = np.array([f"uid_{sample_id}"] * len(full_batch), dtype=object)

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[],
                sample_id=sample_id,
                epoch=epoch,
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples"
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put("DONE")
        self.feed_finished = True
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    def _on_task_done(self, task: asyncio.Task):
        """Task completion callback to handle results and cleanup."""
        try:
            task.result()
        except asyncio.CancelledError:
            print(f"[FullyAsyncRollouter] Task {task.get_name()} was cancelled")
        except Exception as e:
            print(f"[FullyAsyncRollouter] Task {task.get_name()} failed with exception: {e}")
            raise e
        finally:
            self.active_tasks.discard(task)

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        ret, is_cancel = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )

        if not is_cancel:
            sampled_trajectory_count = ret.meta_info.pop("sampled_trajectory_count", len(ret))
            self.sync_sampled_trajectory_count += sampled_trajectory_count
            self.total_sampled_trajectory_count += sampled_trajectory_count
            if len(ret) > 0:
                self.sync_kept_trajectory_count += len(ret)
                self.total_kept_trajectory_count += len(ret)
                self.generated_trajectory_count += len(ret)

                rollout_sample.agent_loop_output_list = []
                ret.meta_info["rollout_param_versions"] = self.current_param_version
                ret.meta_info["rollout_status"] = await self.get_statistics()
                success = await self.message_queue_client.put_sample(
                    sample=ray.cloudpickle.dumps(ret),
                    size=len(ret),
                    param_version=self.current_param_version,
                )
                if success:
                    self.total_generated_samples += 1
                else:
                    self.total_dropped_samples += 1
                    print("[FullyAsyncRollouter][_process_single_sample_streaming] Dropped stale sample")

                if self.generated_trajectory_count > self.max_generate_trajectories_per_sync:
                    print(
                        f"[FullyAsyncRollouter][_process_single_sample_streaming] "
                        f"Excess data detected: generated_trajectory_count={self.generated_trajectory_count} > "
                        f"max_generate_trajectories_per_sync={self.max_generate_trajectories_per_sync}. "
                    )
            else:
                # Sample was completely filtered out
                self.total_filtered_out_sample_count += 1

            self.avg_trajectories_per_sample = 0.1 * len(ret) + 0.9 * self.avg_trajectories_per_sample

            # Internal auto-resume: if paused internally and
            # expected_trajectory_count < max_generate_trajectories_per_sync
            if (
                self.pause_event.is_set()
                and not self.external_pause_event.is_set()
                and not self._should_pause_generation(True)
            ):
                # Rollouter has stopped adding new tasks due to internal pause (expected_trajectory_count),
                # but the expected number of generated trajectories is now insufficient,
                # so Rollouter can be automatically resumed to add new tasks.
                print(
                    "[FullyAsyncRollouter][AutoResume] Internal pause condition cleared, "
                    f"auto-resuming generation (is_external_pause={self.external_pause_event.is_set()})"
                )
                self.pause_event.clear()  # internal auto-resume (clear paused state)
                # Trigger resume signal to wake waiters
                self.resume_signal.set()
            self.total_processed_sample_count += 1
        else:
            rollout_sample.agent_loop_output_list = ret
            await self.cancel_queue.put(rollout_sample)

    async def fit(self):
        """Main loop scheduler that dispatches by state."""
        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        feed_task = None
        try:
            # ===== INIT state =====
            feed_task = await self._handle_init_state()

            # ===== State machine main loop =====
            while True:
                if self.state == RollouterState.ADD_TASK:
                    # Handle ADD_TASK state
                    await self._handle_add_task_state()

                elif self.state == RollouterState.WAIT_CONCURRENCY:
                    # Wait for concurrency slots
                    await self._handle_wait_concurrency_state()

                elif self.state == RollouterState.PAUSED:
                    # Paused state, wait for resume
                    await self._handle_paused_state()

                elif self.state == RollouterState.FINISHING:
                    # Finishing state
                    finished = await self._handle_finishing_state()
                    if finished:
                        break

                else:
                    # Unexpected state
                    print(f"[Rollouter] Unexpected state: {self.state}")
                    raise RuntimeError(f"Unexpected state: {self.state}")

            # ===== FINISHED state =====
            await self._handle_finished_state()
        finally:
            # Unified cleanup
            if feed_task and not feed_task.done():
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

        print("[FullyAsyncRollouter] Rollouter fit completed")

    @ray.method(concurrency_group="monitor")
    async def run_monitor(self):
        """Standalone monitoring loop."""

        print("[Rollouter] Monitor loop started")
        stats_interval = 60.0

        while self.running_event.is_set():  # check running_event
            await asyncio.sleep(stats_interval)
            stats = await self.get_statistics()
            print(f"[MonitorLoop][Statistics] {pformat(stats)}")

        print("[Rollouter] Monitor loop stopped")

    async def _handle_init_state(self):
        """Handle INIT state: initialize and start helper tasks."""
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set")

        self.state = RollouterState.INIT
        print("[Rollouter] Entering INIT state")

        # Ensure initial state is running
        self.pause_event.clear()  # clear = running
        self.running_event.set()
        self.feed_finished = False

        # Start helper task
        feed_task = asyncio.create_task(self._feed_samples())

        # Transition to ADD_TASK state
        self.state = RollouterState.ADD_TASK

        return feed_task

    async def _handle_add_task_state(self):
        """Handle ADD_TASK state: fetch samples, wait for slots, create tasks."""
        # Check whether to pause
        external_pause_active = self.external_pause_event.is_set()
        internal_pause_needed = self._should_pause_generation()

        if external_pause_active or internal_pause_needed:
            if internal_pause_needed and not external_pause_active:
                # Internal pause: only set pause_event
                print("[FullyAsyncRollouter][InternalPause] Pausing due to expected_trajectory_count threshold")
                self.pause_event.set()  # set paused state
                # Do not set external_pause_event for internal pause
            # For external_pause_active, no extra action

            self.state = RollouterState.PAUSED
            return

        # Wait for concurrency slots
        if len(self.active_tasks) >= self.max_concurrent_samples:
            self.state = RollouterState.WAIT_CONCURRENCY
            return

        # Get sample (prefer cancel_queue)
        simple_from_cancel_queue = False
        if not self.cancel_queue.empty():
            rollout_sample = await self.cancel_queue.get()
            simple_from_cancel_queue = True
        else:
            if self.feed_finished and self.pending_queue.empty():
                self.state = RollouterState.FINISHING
                return
            rollout_sample = await self.pending_queue.get()

        # End signal
        if rollout_sample == "DONE":
            self.state = RollouterState.FINISHING
            return

        # Check pause signal again before creating task
        if self.pause_event.is_set():
            # Pause signal is set; put sample back and enter PAUSED state
            await self.cancel_queue.put(rollout_sample)
            self.state = RollouterState.PAUSED
            return

        # Create a task
        task = asyncio.create_task(
            self._process_single_sample_streaming(rollout_sample),
            name=rollout_sample.sample_id,
        )
        self.active_tasks.add(task)
        task.add_done_callback(self._on_task_done)

        if simple_from_cancel_queue:
            self.cancel_queue.task_done()
        else:
            self.pending_queue.task_done()
        # Stay in ADD_TASK state for next loop

    async def _handle_wait_concurrency_state(self):
        """Handle WAIT_CONCURRENCY state: wait for slots and pause signals."""
        print(f"[Rollouter] Waiting for concurrency slot (current: {len(self.active_tasks)})")

        # Check whether to pause
        if self.external_pause_event.is_set():
            self.state = RollouterState.PAUSED
            return

        # Wait for external pause
        pause_wait_task = asyncio.create_task(self.external_pause_event.wait())

        try:
            while len(self.active_tasks) >= self.max_concurrent_samples:
                # Check if paused
                if self.external_pause_event.is_set():
                    self.state = RollouterState.PAUSED
                    return

                all_tasks = self.active_tasks | {pause_wait_task}
                done_tasks, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

                # Check if pause signal triggered
                if pause_wait_task in done_tasks:
                    # External pause signal received
                    print("[Rollouter] Pause signal received during wait")
                    self.state = RollouterState.PAUSED
                    return

            # Slot released; return to ADD_TASK
            self.state = RollouterState.ADD_TASK
        finally:
            # Ensure pause_wait_task is cancelled
            if not pause_wait_task.done():
                pause_wait_task.cancel()
                try:
                    await pause_wait_task
                except asyncio.CancelledError:
                    pass

    async def _handle_paused_state(self):
        """Handle PAUSED state: wait for resume signal."""
        print("[Rollouter] Entering PAUSED state")

        # Ensure paused state is set
        self.pause_event.set()  # set = paused

        # Record idle time
        self.idle_start_time = time.time()

        # Wait for resume signal
        print("[Rollouter][PAUSED] Waiting for resume...")
        await self.resume_signal.wait()
        print("[Rollouter][PAUSED] Resumed")

        # Clear resume signal for next pause
        self.resume_signal.clear()

        # Return to WAIT_CONCURRENCY (check slots before ADD_TASK)
        self.state = RollouterState.WAIT_CONCURRENCY

    async def _handle_finishing_state(self) -> bool:
        """Handle FINISHING state: wait for all tasks to complete."""
        print("[Rollouter] Entering FINISHING state")
        if self.external_pause_event.is_set():
            self.state = RollouterState.PAUSED
            return False

        if self.active_tasks:
            pause_wait_task = asyncio.create_task(self.external_pause_event.wait())
            try:
                while self.active_tasks:
                    all_tasks = set(self.active_tasks) | {pause_wait_task}
                    done_tasks, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

                    if pause_wait_task in done_tasks:
                        print("[Rollouter] Pause signal received during finishing")
                        self.state = RollouterState.PAUSED
                        return False
            finally:
                if not pause_wait_task.done():
                    pause_wait_task.cancel()
                    try:
                        await pause_wait_task
                    except asyncio.CancelledError:
                        pass

        print("[Rollouter] All active tasks completed")
        # Note: main loop will break and call _handle_finished_state
        return True

    async def _handle_finished_state(self):
        """Handle FINISHED state: cleanup and send completion signal."""
        self.state = RollouterState.FINISHED
        print("[Rollouter] Entering FINISHED state")

        # Send completion signal to message queue
        await self.message_queue_client.put_sample(
            sample=None,
            size=0,
            param_version=self.current_param_version,
        )

        # Clear running_event
        self.running_event.clear()

        print("[Rollouter] Shutdown complete")

    def _should_pause_generation(self, in_process_single_sample: bool = False) -> bool:
        """Determine whether generation should pause"""
        if in_process_single_sample:
            expected_trajectory_count = (
                self.generated_trajectory_count + (len(self.active_tasks) - 1) * self.avg_trajectories_per_sample
            )
        else:
            expected_trajectory_count = (
                self.generated_trajectory_count + len(self.active_tasks) * self.avg_trajectories_per_sample
            )
        if expected_trajectory_count >= self.max_generate_trajectories_per_sync:
            # Use pause_event.is_set() to avoid duplicate logs
            if not self.pause_event.is_set():  # not set = running, log once
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"expected_trajectory_count: {expected_trajectory_count} >= "
                    f"max_generate_trajectories_per_sync: {self.max_generate_trajectories_per_sync} "
                )
            return True

        return False

    async def pause(self):
        """External pause; only resume() can recover."""
        print("[FullyAsyncRollouter][Public][Pause] External pause triggered")
        self.external_pause_event.set()  # mark external pause
        self.pause_event.set()  # set paused state

        if self.config.async_training.partial_rollout:
            await self.async_rollout_manager.cancel()

        # Wait for all active_tasks to complete
        print(f"[FullyAsyncRollouter][Public][Pause] Waiting for {len(self.active_tasks)} active tasks to complete...")
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.active_tasks.clear()
        print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")

        # Clear KV cache after all tasks complete
        await self.async_rollout_manager.clear_kv_cache()
        print("[FullyAsyncRollouter][Public][Pause] KV cache cleared, pause complete")

    async def resume(self, dependency_ref: ObjectRef = None):
        """External resume to recover from an external pause."""
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume] Resuming from external pause")

        if self.config.async_training.partial_rollout:
            await self.async_rollout_manager.resume()

        self.external_pause_event.clear()  # clear external pause flag
        self.pause_event.clear()  # clear paused state and resume

        # Trigger resume signal to wake waiters
        self.resume_signal.set()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()
        total_sampled_trajectories = self.total_sampled_trajectory_count
        total_kept_trajectories = self.total_kept_trajectory_count
        sync_sampled_trajectories = self.sync_sampled_trajectory_count
        sync_kept_trajectories = self.sync_kept_trajectory_count
        total_filtered_trajectory_rate = (
            (total_sampled_trajectories - total_kept_trajectories) / total_sampled_trajectories
            if total_sampled_trajectories > 0
            else 0.0
        )
        sync_filtered_trajectory_rate = (
            (sync_sampled_trajectories - sync_kept_trajectories) / sync_sampled_trajectories
            if sync_sampled_trajectories > 0
            else 0.0
        )

        stats = {
            # state stats
            "state/current_state": self.state.name,
            "state/is_paused": self.pause_event.is_set(),
            "state/is_external_pause": self.external_pause_event.is_set(),
            "state/is_running": self.running_event.is_set(),
            "state/feed_finished": self.feed_finished,
            "state/resume_signal": self.resume_signal.is_set(),
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/avg_trajectories_per_sample": self.avg_trajectories_per_sample,
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # count: samples
            "count/total_generated_samples": self.total_generated_samples,
            "count/total_dropped_samples": self.total_dropped_samples,
            "count/total_processed_sample_count": self.total_processed_sample_count,
            "count/total_filtered_out_sample_count": self.total_filtered_out_sample_count,
            # count: trajectories (sync)
            "count/sync_sampled_trajectory_count": self.sync_sampled_trajectory_count,
            "count/sync_kept_trajectory_count": self.sync_kept_trajectory_count,
            "count/generated_trajectory_count": self.generated_trajectory_count,
            # count: trajectories (total)
            "count/total_sampled_trajectory_count": self.total_sampled_trajectory_count,
            "count/total_kept_trajectory_count": self.total_kept_trajectory_count,
            "count/total_filtered_trajectory_rate": total_filtered_trajectory_rate,
            "count/sync_filtered_trajectory_rate": sync_filtered_trajectory_rate,
            # count: other
            "count/current_param_version": self.current_param_version,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/max_generate_trajectories_per_sync": self.max_generate_trajectories_per_sync,
            "static/request_trajectories_per_sync": self.request_trajectories_per_sync,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats
