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
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


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
        self.use_reference_policy = False

        self.use_rm = False
        self.use_reward_loop = self.config.reward_model.use_reward_loop

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
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== fully async config ====================

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

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
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
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.cancel_queue = asyncio.Queue()

        cpu_cores = multiprocessing.cpu_count()
        # cpu case use cpu_cores; io case use cpu_cores*2
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.parallel_validate_and_rollout = config.async_training.get("parallel_validate_and_rollout", False)
        self.validate_task = None

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

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

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
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
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = (
                len(self.active_tasks) + self.cancel_queue.qsize() + await self.message_queue_client.get_queue_size()
            )
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
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            need_validate = (
                (
                    self.config.rollout.test_freq > 0
                    and self.current_param_version % self.config.rollout.test_freq == 0
                    and self.current_param_version > 0
                )  # don't test here in the initial parameter sync
                or validate
            )
            print(
                f"[FullyAsyncRollouter] need_validate: {need_validate}, "
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
        self._init_reward_loop()
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
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = self.use_reward_loop and (
            not self.use_rm or self.config.reward_model.enable_resource_pool
        )
        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config, worker_group=self.rollout_wg, reward_loop_worker_handles=reward_loop_worker_handles
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
                agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                tool_calls=[],
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
        await self.pending_queue.put("DONE")
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
                while self.active_tasks:
                    async with self.lock:
                        # After acquiring the lock, the number of active_tasks may change, need to be verified again
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task

                async with self.lock:
                    while self.paused:
                        self.idle_start_time = time.time()
                        await self.condition.wait()
                continue

            simple_from_cancel_queue = False
            if not self.cancel_queue.empty():
                rollout_sample = await self.cancel_queue.get()
                simple_from_cancel_queue = True
            else:
                rollout_sample = await self.pending_queue.get()
                self.staleness_samples += 1

            if rollout_sample == "DONE":
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
            async with self.lock:
                # After the pause is over, the lock is acquired and it is necessary
                # to determine whether it is the pause phase, otherwise continue to wait
                while self.paused:
                    await self.condition.wait()
                task = asyncio.create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                )
                self.active_tasks.add(task)

            if simple_from_cancel_queue:
                self.cancel_queue.task_done()
            else:
                self.pending_queue.task_done()

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        ret, is_cancel = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        if not is_cancel:
            rollout_sample.full_batch = ret
            rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
                [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
            )
            rollout_sample.param_version = self.current_param_version
            rollout_sample.rollout_status = await self.get_statistics()
            rollout_sample.agent_loop_output_list = []

            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1
        else:
            rollout_sample.agent_loop_output_list = ret
            await self.cancel_queue.put(rollout_sample)

        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.processor_task = asyncio.create_task(self._processor_worker())

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

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception:{e}")

        finally:
            if self.processor_task:
                self.processor_task.cancel()

            await asyncio.gather(self.processor_task, return_exceptions=True)

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

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

        # Create the main asynchronous task
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

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
            if self.monitor_loop_trigger:
                if not await self._should_pause_generation():
                    async with self.lock:
                        self.paused = False
                        self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
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

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause] partial rollout:", self.config.async_training.partial_rollout)
        async with self.lock:
            self.paused = True
            # Cancel all rollout tasks
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.cancel()
                print("[FullyAsyncRollouter][Public][Pause] Unfinished rollout tasks canceled")
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
                self.active_tasks.clear()
                print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")

            # TODO use checkpoint engine for rollout clear_kv_cache
            # print("[FullyAsyncRollouter][Public][Pause] clear kv cache")
            # # Always clear KV cache to release GPU memory during weight synchronization,
            # # regardless of partial_rollout setting.
            # await self.async_rollout_manager.clear_kv_cache()
            self.monitor_loop_trigger = False

    async def resume(self, dependency_ref: ObjectRef = None):
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume()
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/current_param_version": self.current_param_version,
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
