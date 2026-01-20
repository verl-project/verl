# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import time
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from collections import defaultdict
import uuid
import random
from torch.utils.data import Subset

from recipe.gkd.teacher import TeacherClient
from recipe.gkd.teacher_utils import get_teacher_knowledge
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


WorkerType = type[Worker]


class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.teacher_batch_output = None

    def set_teacher_batch_output(self, teacher_batch_output):
        """Set the teacher batch output for this generation batch.

        Args:
            teacher_batch_output: The teacher model's output (DataProtoFuture or raw output)
                to be associated with this generation batch. This will be used for
                distillation or guidance during training.
        """
        self.teacher_batch_output = teacher_batch_output

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (batch, gen_batch_result)
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
            self.gen_batch_output = gen_batch_result

        if self.teacher_batch_output is None:
            return self.epoch, self.batch, self.gen_batch_output

        if hasattr(self.teacher_batch_output, "get"):
            try:
                teacher_batch_result = self.teacher_batch_output.get()
            except Exception as e:
                # set result to empty
                teacher_batch_result = None
                print(f"{e}")
        else:
            teacher_batch_result = self.teacher_batch_output

        return self.epoch, self.batch, self.gen_batch_output, teacher_batch_result


class OnPolicyDistillTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
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
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        # [Claude]
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_critic = False

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.teacher_config = self.config.actor_rollout_ref.teacher
        self.n_server_workers = self.teacher_config.n_server_workers
        self.teacher_client = TeacherClient(
            self.teacher_config.server_ip, self.teacher_config.server_port, n_server_workers=self.n_server_workers
        )

        self.params_dtype = PrecisionType.to_dtype("bfloat16")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_sampler

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # Will be used in for GKD scenario for seperate dataloaders
        self._collate_fn = collate_fn
        self._num_workers = num_workers
        self._on_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        self._off_batch_size = self.config.data.get("off_policy_batch_size") or self.config.data.train_batch_size
        self._drop_last = True 

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        # Create off-policy dataloader if enabled
        self.off_policy_dataloader = None
        if self.config.trainer.get("enable_off_policy", False):
            from recipe.gkd.off_policy_dataset import GKDOffPolicyDataset

            if self.config.data.get("off_policy_files") is not None:
                print("Creating off-policy dataloader with ground truth answers...")
                off_policy_dataset = GKDOffPolicyDataset(
                    self.config.data.off_policy_files, self.tokenizer, self.config.data
                )
                self.off_policy_dataset = off_policy_dataset
                off_policy_batch_size = self.config.data.get("off_policy_batch_size")
                if off_policy_batch_size is None:
                    off_policy_batch_size = self.config.data.train_batch_size

                self.off_policy_dataloader = StatefulDataLoader(
                    dataset=off_policy_dataset,
                    batch_size=off_policy_batch_size,
                    num_workers=num_workers,
                    drop_last=True,
                    collate_fn=collate_fn,
                    shuffle=self.config.data.get("shuffle", True),
                )
                print(f"Size of off-policy dataloader: {len(self.off_policy_dataloader)}")
            else:
                raise ValueError(
                    "enable_off_policy is True but off_policy_files is not set in config.data.off_policy_files"
                )



        if self.val_dataset:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(self.config.trainer.total_training_steps, total_training_steps)

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        # Build Ray classes per pool
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Rollout group
        rollout_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
        )
        resource_pool_to_cls[rollout_pool]["rollout"] = rollout_cls

        # Actor group
        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        resource_pool_to_cls[actor_pool]["actor"] = actor_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            time.sleep(20)  # avoid port conflict

        self.rollout_wg = all_wg["rollout"]
        self.actor_wg = all_wg["actor"]

        # Initialize both groups
        self.rollout_wg.init_model()
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)
        from ray.util.collective import collective

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend="nccl",
            group_name="actor_rollout",
        )

    def sync_rollout_weights(self):
        assert not self.hybrid_engine
        self.actor_wg.sync_rollout_weights()
        ray.get(self.rollout_wg.sync_rollout_weights())

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch.
        If off-policy is enabled, we do *disjoint partition* of sample indices per epoch:
        - shuffle all sample indices once
        - split into on_ids / off_ids by gkd_lambda
        - build per-epoch dataloaders on Subset(train_dataset) and Subset(off_policy_dataset)
        - then shuffle the *source sequence* (on vs off) so training sees a mixed stream
        This guarantees: within an epoch, a sample is used by at most one of {on, off}.
        """
        gkd_lambda = float(self.config.trainer.get("gkd_lambda", 1.0))
        enable_off_policy = bool(self.config.trainer.get("enable_off_policy", False))

        # clamp
        if gkd_lambda < 0.0:
            gkd_lambda = 0.0
        if gkd_lambda > 1.0:
            gkd_lambda = 1.0

        # If off-policy disabled or lambda==1 -> original pure on-policy behavior
        if (not enable_off_policy) or (self.off_policy_dataloader is None) or (gkd_lambda >= 1.0 - 1e-12):
            for epoch in range(self.config.trainer.total_epochs):
                on_it = iter(self.train_dataloader)
                for _ in range(len(self.train_dataloader)):
                    batch_dict = next(on_it)
                    batch_dict["is_on_policy"] = True
                    yield epoch, batch_dict
            return

        # If lambda==0 -> pure off-policy behavior
        if gkd_lambda <= 1e-12:
            for epoch in range(self.config.trainer.total_epochs):
                off_it = iter(self.off_policy_dataloader)
                for _ in range(len(self.off_policy_dataloader)):
                    batch_dict = next(off_it)
                    batch_dict["is_on_policy"] = False
                    yield epoch, batch_dict
            return

        # Disjoint partition mode (0<lambda<1)
        base_seed = int(self.config.trainer.get("data_partition_seed", 12345))

        train_ds = self.train_dataset
        off_ds = getattr(self, "off_policy_dataset", None) or self.off_policy_dataloader.dataset

        N = len(train_ds)
        assert len(off_ds) == N, (
            f"Disjoint partition requires on/off datasets aligned with same length. "
            f"Got len(train)={len(train_ds)} len(off)={len(off_ds)}"
        )

        for epoch in range(self.config.trainer.total_epochs):
            # 1) global shuffle of indices (deterministic per epoch)
            rng = random.Random(base_seed + epoch)
            indices = list(range(N))
            rng.shuffle(indices)

            # 2) split into on/off ids (mutually exclusive)
            n_on = int(round(gkd_lambda * N))
            on_ids = indices[:n_on]
            off_ids = indices[n_on:]

            # 3) build per-epoch subset dataloaders
            on_subset = Subset(train_ds, on_ids)
            off_subset = Subset(off_ds, off_ids)

            on_loader = StatefulDataLoader(
                dataset=on_subset,
                batch_size=self._on_batch_size,
                num_workers=self._num_workers,
                drop_last=self._drop_last,
                collate_fn=self._collate_fn,
                shuffle=False,  # order is defined by on_ids
            )

            off_loader = StatefulDataLoader(
                dataset=off_subset,
                batch_size=self._off_batch_size,
                num_workers=self._num_workers,
                drop_last=self._drop_last,
                collate_fn=self._collate_fn,
                shuffle=False,  # order is defined by off_ids
            )

            on_steps = len(on_loader)
            off_steps = len(off_loader)

            # 4) shuffle the source schedule so batches are mixed
            #    (This shuffles only whether next batch comes from on or off; samples remain disjoint.)
            schedule = [True] * on_steps + [False] * off_steps
            rng.shuffle(schedule)

            on_it = iter(on_loader)
            off_it = iter(off_loader)

            for use_on in schedule:
                if use_on:
                    batch_dict = next(on_it)
                    batch_dict["is_on_policy"] = True
                else:
                    batch_dict = next(off_it)
                    batch_dict["is_on_policy"] = False
                yield epoch, batch_dict

    def _async_gen_next_batch(self, epoch, batch_dict, sync_before_generation=True):
        """
        Call parameter synchronization and asynchronous sequence generation.
        For off-policy data, skip generation as responses (ground truth answers) are already in batch.
        """
        is_on_policy = batch_dict.get("is_on_policy", True)
        # DataProto doesn't support bool; also avoid side effects of pop()
        if "is_on_policy" in batch_dict:
            batch_dict = dict(batch_dict)  # shallow copy
            batch_dict.pop("is_on_policy", None)

        if not is_on_policy:
            # Off-policy data: already has full sequence (prompt + ground truth answer)
            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["global_steps"] = self.global_steps
            batch.meta_info["is_on_policy"] = False

            # For off-policy, we pop everything to gen_batch (similar to on-policy structure)
            # This keeps batch mostly empty, and gen_batch contains all data
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids", "responses"],
                non_tensor_batch_keys=["raw_prompt_ids"]
            )
            # IMPORTANT: metrics read from gen_batch_output.meta_info
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch.meta_info["is_on_policy"] = False
            
            # Create future with separated batch and gen_batch_output
            future = GenerationBatchFuture(epoch, batch, gen_batch)
            future.gen_batch_output = gen_batch

            return future


        # On-policy data: normal generation flow
        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["is_on_policy"] = True
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info["global_steps"] = self.global_steps
        # sync weights from actor to rollout
        if sync_before_generation:
            self.sync_rollout_weights()
        # Call non-blocking rollout (worker method registered with blocking=False)
        gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)
        return GenerationBatchFuture(epoch, batch, gen_batch_output)

    def _async_get_teacher_knowledge(self, future: GenerationBatchFuture):
        """Asynchronously obtain teacher model knowledge for generated sequences.

        This method retrieves generated sequences from the future object, adds response length metadata,
        and asynchronously queries the teacher model for knowledge distillation. The teacher model's output
        is set in the future object for subsequent processing.

        For off-policy data, we still query teacher but on the full sequence (prompt + ground truth answer).

        Both on-policy and off-policy need teacher knowledge!
        The difference is:
            - On-policy: teacher sees (prompt + generated response)
            - Off-policy: teacher sees (prompt + ground truth answer)

        Args:
            future (GenerationBatchFuture): Future object containing generated sequences and metadata

        Returns:
            GenerationBatchFuture: The same future object with teacher knowledge set

        Raises:
            RuntimeError: If teacher client initialization fails or knowledge retrieval fails
        """
        _, _, gen_batch_output = future.get()
        gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        future.set_teacher_batch_output(
            get_teacher_knowledge(gen_batch_output, self.teacher_client, self.n_server_workers, is_async=True)
        )
        return future


    def one_step_off_scheduler(self, continuous_iterator):
        """One-step-off scheduler implementation (version 1) for GKD training with improved pipeline.

        This scheduler optimizes the training pipeline by:
        1. Overlapping rollout weight synchronization with teacher knowledge processing
        2. Maintaining consistent timing measurement across iterations
        3. Reducing idle time between generation and knowledge distillation phases

        The scheduler maintains the following timing metrics:
        - sync_rollout_weights: Time taken to synchronize rollout weights
        - wait_prev_gen: Time waiting for previous generation to complete
        - wait_prev_teacher: Time waiting for teacher knowledge to be ready

        Args:
            continuous_iterator: Iterator providing (epoch, batch_dict) tuples for training

        Yields:
            tuple: Contains (batch, gen_batch_output, teacher_batch_output, timing_metrics)
                - batch: Original input batch data
                - gen_batch_output: Generated sequences from main model
                - teacher_batch_output: Knowledge distillation from teacher model
                - timing_metrics: Dictionary of timing measurements
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            if i == 0:
                # sync weights and start first async rollout
                with marked_timer("sync_rollout_weights", timing):
                    fut = self._async_gen_next_batch(epoch, batch_dict)
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
                # no yield here, so we will continue to the next loop and enter `else` block
            if i == 1:
                # we don't need to sync weights here because we have not trained the actor yet
                # start second async rollout
                fut = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                # wait for generating teacher knowledge finish
                # and get previous result including rollout and teacher knowledge
                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                # start next step from here
                timing = {}
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
            else:
                # sync weights and start next async rollout
                with marked_timer("sync_rollout_weights", timing):
                    fut = self._async_gen_next_batch(epoch, batch_dict)
                # wait for generating teacher knowledge finish
                # and get previous result including rollout and teacher knowledge
                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                # start next step from here
                timing = {}
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)

        # for last step
        with marked_timer("wait_prev_teacher", timing):
            prev_result = prev_fut.get()
        yield *prev_result, timing

    def two_step_off_scheduler(self, continuous_iterator):
        """Two-step-off scheduler implementation for GKD training with optimized pipeline.

        This scheduler implements a double-buffered pipeline that overlaps:
        1. Sequence generation with teacher knowledge distillation
        2. Weight synchronization with previous batch processing

        Key features:
        - Maintains two parallel processing streams (current and previous batches)
        - Overlaps computation and communication where possible
        - Provides consistent timing metrics across iterations

        Pipeline stages:
        1. Initialization: Start first generation without teacher processing
        2. Steady state: Alternate between processing teacher knowledge and starting new generation
        3. Final state: Process last batch of teacher knowledge

        Timing metrics collected:
        - sync_rollout_weights: Time for weight synchronization between actor and rollout workers
        - wait_prev_prev_teacher: Time waiting for teacher knowledge from two batches ago
        - wait_prev_gen: Time waiting for previous generation to complete

        Args:
            continuous_iterator: Iterator providing (epoch, batch_dict) tuples for training

        Yields:
            tuple: Contains (batch, gen_batch_output, teacher_batch_output, timing_metrics)
                - batch: Original input batch data
                - gen_batch_output: Generated sequences from main model
                - teacher_batch_output: Knowledge distillation from teacher model
                - timing_metrics: Dictionary of timing measurements
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            if i == 0:
                with marked_timer("sync_rollout_weights", timing):
                    rollout_future = self._async_gen_next_batch(epoch, batch_dict)
                continue
            elif i == 1:
                teacher_future = self._async_get_teacher_knowledge(rollout_future)
                rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                continue
            elif i == 2:
                with marked_timer("wait_prev_prev_teacher", timing):
                    result = teacher_future.get()
                with marked_timer("wait_prev_gen", timing):
                    teacher_future = self._async_get_teacher_knowledge(rollout_future)
                rollout_future = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                yield *result, timing
                timing = {}
            else:
                with marked_timer("wait_prev_prev_teacher", timing):
                    result = teacher_future.get()
                with marked_timer("wait_prev_gen", timing):
                    teacher_future = self._async_get_teacher_knowledge(rollout_future)
                with marked_timer("sync_rollout_weights", timing):
                    rollout_future = self._async_gen_next_batch(epoch, batch_dict)
                yield *result, timing
                timing = {}

        # for second to last step
        with marked_timer("wait_prev_prev_teacher", timing):
            result = teacher_future.get()
        with marked_timer("wait_prev_gen", timing):
            teacher_future = self._async_get_teacher_knowledge(rollout_future)
        yield *result, timing

        # for last step
        with marked_timer("wait_prev_prev_teacher", timing):
            result = teacher_future.get()
        yield *result, timing

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # ============ [Claude] ADD VALIDATION BEFORE TRAINING ============
        # Perform validation before training (same as PPO)
        if hasattr(self, 'val_reward_fn') and self.val_reward_fn is not None:
            if self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                if val_metrics:
                    print(f"Initial validation metrics: {val_metrics}")
                    logger.log(data=val_metrics, step=self.global_steps)
        # =========================================================

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        max_steps_duration = 0

        # Pre-warm: submit the first rollout
        continuous_iterator = self._create_continuous_iterator()

        scheduler_type = self.config.trainer.scheduler

        if scheduler_type == "one_step_off":
            scheduler = self.one_step_off_scheduler(continuous_iterator)
        elif scheduler_type == "two_step_off":
            scheduler = self.two_step_off_scheduler(continuous_iterator)
        else:
            raise TypeError(f"unrecognized scheduler type: {scheduler_type}")

        # Main loop
        while True:
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            if do_profile:
                self.rollout_wg.start_profile()
                self.actor_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                _, batch, gen_batch_output, teacher_batch_output, schedule_timing = next(scheduler)
                if teacher_batch_output is None:
                    # save model
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        self._save_checkpoint()
                    print("Error in getting teacher knowledge. Skip this batch.")
                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        progress_bar.close()
                        return
                    continue

                timing_raw.update(schedule_timing)

                gen_timing = gen_batch_output.meta_info.pop("timing", {})
                for k, v in gen_timing.items():
                    if isinstance(v, list):
                        array_v = np.array(v)
                        timing_raw[k + "_mean"] = array_v.mean().item()
                        timing_raw[k + "_min"] = array_v.min().item()
                        timing_raw[k + "_max"] = array_v.max().item()
                        timing_raw[k] = array_v.max().item()
                    else:
                        timing_raw[k] = v

                timing_raw.update(teacher_batch_output.meta_info.pop("timing"))
                
                # Track on-policy vs off-policy samples
                if "is_on_policy" not in gen_batch_output.meta_info:
                    print("WARN: gen_batch_output.meta_info missing is_on_policy; keys=", gen_batch_output.meta_info.keys())
                is_on_policy_batch = gen_batch_output.meta_info.get("is_on_policy", True)
                metrics["data/is_on_policy"] = 1.0 if is_on_policy_batch else 0.0

                # Compute statistics of generated response lengths distribution
                response_lens = (
                    (gen_batch_output.batch["responses"] != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
                )
                metrics.update(
                    {
                        "response_seq_len/average": sum(response_lens) / len(response_lens),
                        "response_seq_len/max": max(response_lens),
                        "response_seq_len/min": min(response_lens),
                        "response_seq_len/max_count": response_lens.count(max(response_lens)),
                        "response_seq_len/min_count": response_lens.count(min(response_lens)),
                    }
                )

                # Merge generated outputs back
                batch = batch.union(gen_batch_output)

                # Debug print
                one_attention_mask = batch.batch["attention_mask"][0].to(torch.bool)
                one_sentence = batch.batch["input_ids"][0]
                print("INFO:", "generate text done.")
                print("DEBUG:", self.tokenizer.decode(one_sentence[one_attention_mask].tolist()))

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                batch = batch.union(teacher_batch_output)

                # # update actor
                # with marked_timer("send_teacher_knowledge", timing_raw, color="red"):
                #     self.actor_wg.send_teacher_knowledge(teacher_batch_output)

                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self.actor_wg.update_actor(batch)

                print("INFO:", "update actor done.")
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                # save model
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # ============ [Claude] ADD PERIODIC VALIDATION ============
                # Validate periodically (same as PPO)
                if hasattr(self, 'val_reward_fn') and self.val_reward_fn is not None:
                    if self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate()
                        if val_metrics:
                            metrics.update(val_metrics)
                # =================================================


            # Metrics and bookkeeping
            steps_duration = timing_raw["step"]
            max_steps_duration = max(max_steps_duration, steps_duration)
            # training metrics
            metrics["training/global_step"] = self.global_steps
            # collect metrics
            # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.rollout_wg.stop_profile()
                self.actor_wg.stop_profile()

            if is_last_step:
                progress_bar.close()
                return
            
    
    # [Claude] Add new validate function
    def _validate(self):
        """Validate the model on validation dataset (rule-based RM).

        Key fixes vs current version:
        - sync rollout weights BEFORE any generation (rollout worker holds its own model copy)
        - align with PPO validate: repeat val batch, pad/unpad, val_kwargs.do_sample
        - avoid mutating the reward batch via _get_gen_batch() (which pops tensor keys)
        """
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            print("No validation dataset available, skipping validation")
            return {}

        if not hasattr(self, "val_reward_fn") or self.val_reward_fn is None:
            print("No val_reward_fn provided, skipping validation")
            return {}

        # 🔥 CRITICAL: rollout engine is a separate model instance; sync weights before generating
        if not self.hybrid_engine:
            self.sync_rollout_weights()

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        # PPO-style validation kwargs
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        repeat_n = int(getattr(val_kwargs, "n", 1))
        do_sample = bool(getattr(val_kwargs, "do_sample", False))

        for test_data in self.val_dataloader:
            # Keep a pristine batch for reward computation / logging
            reward_batch = DataProto.from_single_dict(test_data)

            if "uid" not in reward_batch.non_tensor_batch:
                reward_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(reward_batch.batch))], dtype=object
                )

            # Build a separate batch for generation because _get_gen_batch() pops tensor keys
            rollout_batch = DataProto.from_single_dict(test_data)
            rollout_batch.non_tensor_batch["uid"] = reward_batch.non_tensor_batch["uid"]

            # repeat test batch (for best@k/maj@k etc.)
            if repeat_n > 1:
                reward_batch = reward_batch.repeat(repeat_times=repeat_n, interleave=True)
                rollout_batch = rollout_batch.repeat(repeat_times=repeat_n, interleave=True)

            # we only do validation on rule-based rm (same guard as PPO)
            try:
                if (
                    getattr(self.config, "reward_model", None) is not None
                    and self.config.reward_model.enable
                    and reward_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
                ):
                    return {}
            except Exception:
                pass

            # Store original inputs and ground truths
            input_ids = reward_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(reward_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in reward_batch
            ]
            sample_gts.extend(ground_truths)

            # Prepare generation batch (PPO-style)
            gen_batch = self._get_gen_batch(rollout_batch)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            # pad to be divisible by rollout dp size (safe fallback to 1)
            size_divisor = getattr(self.rollout_wg, "world_size", 1) or 1
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)

            # Generate sequences using rollout worker
            output_gen_batch_padded = self.rollout_wg.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            # Store outputs
            output_ids = output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # validate-only: avoid union() asserting on duplicated prompt tensors
            # Delete input_ids and position_ids from output_gen_batch (keep attention_mask!)
            for k in ("input_ids", "position_ids"):
                if k in output_gen_batch.batch:
                    output_gen_batch.batch.pop(k)
            
            # CRITICAL: Delete old attention_mask from reward_batch before union
            # output_gen_batch has the correct full-sequence attention_mask (prompt + response)
            # reward_batch only has prompt-only attention_mask, which would cause:
            # 1. union() assertion failure (two different attention_masks)
            # 2. BatchRewardManager to compute wrong valid_response_lengths
            if "attention_mask" in reward_batch.batch:
                reward_batch.batch.pop("attention_mask")
            
            # Also ensure prompts exist for reward computation
            if "prompts" not in output_gen_batch.batch:
                output_gen_batch.batch["prompts"] = rollout_batch.batch["input_ids"]

            # Merge batches
            reward_batch = reward_batch.union(output_gen_batch)

            reward_batch.meta_info["validate"] = True

            # Compute rewards
            result = self.val_reward_fn(reward_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)

            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(reward_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # log generations table
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        from verl.trainer.ppo.metric_utils import process_validation_metrics

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict
    
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """
        Build a generation-only DataProto WITHOUT mutating the input batch.

        Why:
        - batch.pop(...) mutates `batch.batch` in-place, deleting input_ids/attn/pos.
        - That can silently break reward/metrics/logging/dump paths that assume these exist.
        """
        gen_dict = {}

        # tensor keys needed by rollout
        for k in ["input_ids", "attention_mask", "position_ids"]:
            if k in batch.batch:
                gen_dict[k] = batch.batch[k]

        # non-tensor keys needed by rollout (agent/tool/MM)
        for k in [
            "raw_prompt_ids",
            "raw_prompt",
            "multi_modal_data",
            "tools_kwargs",
            "interaction_kwargs",
        ]:
            if k in batch.non_tensor_batch:
                gen_dict[k] = batch.non_tensor_batch[k]

        # optional: keep uid if you want easier tracing (harmless)
        if "uid" in batch.non_tensor_batch:
            gen_dict["uid"] = batch.non_tensor_batch["uid"]

        return DataProto.from_single_dict(gen_dict)