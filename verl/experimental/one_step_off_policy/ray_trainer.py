# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
This trainer supports model-agonistic model initialization with huggingface
"""

import asyncio
import uuid
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.collective import collective
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.one_step_off_policy.utils import need_critic
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import ValidationGenerationsLogger


class OneStepOffRayTrainer(SeparateRayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
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
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

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
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

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
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== SeparateRayPPOTrainer config ====================

        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
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

    def _validate(self):
        self.actor_rollout_wg = self.rollout_wg
        ret = super()._validate()
        self.actor_rollout_wg = self.actor_wg
        return ret

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor, Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.actor_wg.init_model()
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)
        self._create_weight_sync_group()

    def _init_async_rollout_manager(self):
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
        from verl.experimental.one_step_off_policy.agent_loop import OneStepOffAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = OneStepOffAgentLoopManager(
            config=self.config, worker_group=self.rollout_wg, reward_loop_worker_handles=reward_loop_worker_handles
        )

    def _create_weight_sync_group(self):
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(actor_rollout_workers)

        if self.device_name == "npu":
            master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
            master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
            self.actor_wg.create_weight_sync_group(
                master_address,
                master_port,
                0,
                n_workers,
            )
            ray.get(
                self.rollout_wg.create_weight_sync_group(
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                )
            )
        else:
            # Create Ray collective group for fallback communication
            collective.create_collective_group(
                actor_rollout_workers,
                n_workers,
                list(range(0, n_workers)),
                backend=get_nccl_backend(),
                group_name="actor_rollout",
            )

    def sync_rollout_weights(self):
        self.actor_wg.sync_rollout_weights()
        ray.get(self.rollout_wg.sync_rollout_weights())

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            gen_batch_output = await self.async_rollout_manager.generate_sequences_async(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Launch individual reward computations as each generation completes
        future_reward = None

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, batch, future_reward

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        reward_tensor, reward_extra_info = extract_reward(batch)
        return reward_tensor, reward_extra_info

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._fit_update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()
        # Start the first asynchronous generation task.
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
        while batch_data_future is not None:
            batch_data_future = await self.fit_step(batch_data_future, continuous_iterator)
            if self.is_last_step:
                return

    async def fit_step(self, batch_data_future, continuous_iterator):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_data_future: batch future
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)

            # await asyncio.sleep(0) ensures:
            # Asynchronous tasks can start executing immediately
            # The event loop can handle other pending coroutines
            # Prevents computations in a certain phase from blocking the entire asynchronous workflow
            #
            # The purpose here is to ensure that after triggering
            # `self.async_rollout_manager.generate_sequences_async(gen_batch_output)`,
            # the subsequent relevant logic can proceed in a timely manner
            await asyncio.sleep(0)
            batch = self._fit_compute_reward(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_ref_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_advantage(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_actor(batch)
            await asyncio.sleep(0)
            self._fit_update_weights()
            await asyncio.sleep(0)
            self._fit_dump_data(batch)
            await asyncio.sleep(0)

        self._fit_validate()
        await asyncio.sleep(0)
        self._fit_save_checkpoint()
        await asyncio.sleep(0)
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_torch_memory()
        self._fit_experimental(batch)
        self._fit_postprocess_step()

        return batch_data_future

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("gen", timing_raw, color="red"):
            _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
            timing_raw.update(batch.meta_info["timing"])
            timing_raw.update(_timing_raw)
            metrics.update(_metrics)
            batch.meta_info.pop("timing", None)

        # sync weights from actor to rollout
        with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
            self._fit_update_weights()
            await self.async_rollout_manager.clear_kv_cache()

        # async next generation
        if not self.is_last_step:
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
            await asyncio.sleep(0)
        else:
            batch_data_future = None

        return batch, batch_data_future

    def _fit_update_weights(self):
        # TODO: use checkpoint engine to update weight
        # self.sync_rollout_weights()
        pass
