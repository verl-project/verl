# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Synchronous PPO trainer with colocated actor and rollout.

Differs from original PPO trainer in main_ppo.py:
1. Use TransferQueue to zero-padding and zero-copy data transfer.
2. Use ReplayBuffer to sample data from TransferQueue.
3. Support different `n` sampling for each prompt.
4. Support multiple outputs for each agent loop.
"""

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from functools import partial
from pprint import pprint

import hydra
import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import NonTensorData, NonTensorStack, TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager, AgentLoopOutput, AgentLoopWorker, get_trajectory_info
from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.tracking import Tracking
from verl.workers.config import CriticConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import extract_response_from_unpad_output

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ======================================= USER SECTION BEGIN =======================================


class ReplayBuffer:
    """Replay buffer periodically polls metadata from transfer queue.

    Args:
        poll_interval (float, optional): Poll interval in seconds. Defaults to 1.0.
    """

    def __init__(self, poll_interval: float = 1.0):
        # partition_id => {key: tags}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)

        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        self.poll_thread = threading.Thread(target=self._poll_from_transfer_queue, daemon=True)
        self.poll_thread.start()

    def _poll_from_transfer_queue(self):
        """Periodically poll metadata from transfer queue."""
        try:
            while True:
                data = tq.kv_list()
                if data is not None:
                    for partition_id, items in data.items():
                        self.add(partition_id, items)
                time.sleep(self.poll_interval)
        except Exception as e:
            logger.error(f"Error in _poll_from_transfer_queue: {e}")
            os._exit(1)

    def add(self, partition_id: str, items: dict[str, dict]):
        """Add items to the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            items (dict[str, dict]): Items to add, e.g. {"key": {"tag": "value"}}.
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key, tags in items.items():
                if key not in partition:
                    partition[key] = {}
                partition[key].update(tags)

    def sample(self, partition_id: str, global_steps: int = None, batch_size: int = None) -> KVBatchMeta:
        """Sample a batch of data from the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            global_steps (int, optional): Global training steps. If not None, wait until all prompts of
                this global steps have finished.
            batch_size (int, optional): Batch size. Defaults to None.

        Returns:
            KVBatchMeta: A batch of data.
        """
        assert (global_steps or batch_size) and (not (global_steps and batch_size)), (
            "Either global_steps or batch_size must be specified, but not both."
        )

        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                keys, tags = [], []
                should_wait = False
                partition = self.partitions[partition_id]
                for key, tag in partition.items():
                    if tag["global_steps"] == global_steps:
                        if tag["status"] == "running":
                            should_wait = True
                            break
                        elif tag["status"] == "success":
                            keys.append(key)
                            tags.append(tag)
                        else:
                            logger.warning(f"Unknown status {tag['status']} for key {key}")
                logger.info("partitions", self.partitions)
                if not should_wait:
                    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not prompt.get("validate", False) else "val"
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = config.n if not prompt.get("validate", False) else config.val_kwargs.n

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt)
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(self, output: AgentLoopOutput | list[AgentLoopOutput], **kwargs) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        # NOTE: only use the last output to compute reward score, then assign reward score to all agent loop outputs.
        # User can customize the reward score assignment strategy.
        final_output = outputs[-1]
        prompts = torch.tensor(final_output.prompt_ids, dtype=torch.int64)
        responses = torch.tensor(final_output.response_ids, dtype=torch.int64)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        ).squeeze(0)
        await self._compute_score(
            final_output,
            prompts=prompts.unsqueeze(0),  # [1, prompt_length]
            responses=responses.unsqueeze(0),  # [1, response_length]
            attention_mask=attention_mask.unsqueeze(0),  # [1, seq_len]
            input_ids=input_ids.unsqueeze(0),  # [1, seq_len]
            position_ids=position_ids.unsqueeze(0),  # [1, seq_len] or [1, 4, seq_len]
            kwargs=kwargs,
        )
        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "global_steps": kwargs["global_steps"],
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not kwargs.get("validate", False) else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, replay_buffer: ReplayBuffer, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        # mark prompts as pending in replay buffer
        global_steps = prompts["global_steps"]
        partition_id = "train" if not prompts.get("validate", False) else "val"
        items = {uid: {"global_steps": global_steps, "status": "running"} for uid in prompts["uid"]}
        self.replay_buffer.add(partition_id, items)

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )


# ======================================= USER SECTION END =======================================


class PPOTrainer:
    """PPO Trainer with TransferQueue and ReplayBuffer.

    Args:
        config: DictConfig from yaml config file.
        role_worker_mapping: dict[Role, WorkerType]
        resource_pool_manager: ResourcePoolManager
    """

    def __init__(
        self,
        config: DictConfig,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
    ):
        self.config = config
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_critic = need_critic(self.config)
        self.use_reference_policy = need_reference_policy(self.config)
        self.replay_buffer = ReplayBuffer()

        self._init_tokenizer()
        self._init_dataloader()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.actor_rollout_ref.model.path, use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _init_dataloader(self):
        """Initialize train and validate dataloader."""
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=self.config.data.get("val_max_samples", -1),
        )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size or len(self.val_dataset),
            num_workers=self.config.data["dataloader_num_workers"],
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        logger.info(
            f"train and validate dataloader initialized, train dataset size: "
            f"{len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
        )

        # adjust total_training_steps
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        logger.info(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 1. define actor and rollout class
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # 2. define critic class
        if self.use_critic:
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=critic_cfg.model_config,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # 3. create worker group for actor rollout and critic
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device
        logger.info(f"worker group kwargs: {wg_kwargs}")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            logger.info(f"create worker group {spawn_wg.keys()}")

        # 5. initiliaze critic model engine
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = partial(value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)
            logger.info("critic model engine initialized")

        # 6. initialize actor and ref model engine
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        logger.info("actor and ref model engine initialized")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # 7. initialize reward loop manager
        resource_pool = (
            self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        logger.info("reward loop manager initialized")

        # 8. initialize agent loop manager
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ
        self.agent_loop_manager = agent_loop_manager_cls(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=self.reward_loop_manager.reward_loop_worker_handles,
            replay_buffer=self.replay_buffer,
        )
        logger.info("agent loop manager initialized")

        # 9. initialize checkpoint engine manager
        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.agent_loop_manager.rollout_replicas,
        )
        logger.info("checkpoint engine manager initialized")

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

        logger.info("all initialize finished, ready to fit")

    def _load_checkpoint(self):
        self.global_steps = 0

        # 1. find latest checkpoint folder
        if self.config.trainer.resume_mode == "disable":
            return
        elif self.config.trainer.resume_mode == "auto":
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                logger.info("Training from scratch")
                return
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

        # 2. load actor checkpoint
        self.actor_rollout_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # 3. load critic checkpoint
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # 4. load dataloader checkpoint
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logger.warning(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        do_profile = (
            not self.prev_step_profile and self.curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )

        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        self.next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        do_profile = (
            self.curr_step_profile and not self.next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else self.curr_step_profile
        )
        self.prev_step_profile = self.curr_step_profile
        self.curr_step_profile = self.next_step_profile

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reward with colocate reward model."""
        # TODO: add reward model
        raise NotImplementedError

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens."""
        global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # get actor dp size
        role, worker_group = "actor", self.actor_rollout_wg
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        dp_size = max(dp_rank_mapping) + 1

        # TODO: up sampling if batch is not divisible by dp_size
        if len(batch) % dp_size != 0:
            raise ValueError(f"Batch size {len(batch)} is not divisible by dp_size {dp_size}")

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        batch.reorder([j for partition in global_partition_lst for j in partition])
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, fields=["rollout_log_probs"])
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
            return

        # 1. compute log probs
        batch.extra_info.update({"calculate_entropy": True, "compute_loss": False})
        output: KVBatchMeta = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)

        fields = ["entropy", "log_probs", "response_mask"]
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.extend(["responses", "old_log_probs", "rollout_log_probs"])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, fields=fields)

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = extract_response_from_unpad_output(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = extract_response_from_unpad_output(data.pop("entropy"), data["response_mask"])
        tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=data.select("old_log_probs", "entropy")
        )

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu,
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch)

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, fields=["log_probs", "response_mask"])
        data["ref_log_prob"] = extract_response_from_unpad_output(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def _compute_values(self, batch: KVBatchMeta) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        output = self.critic_wg.infer_batch(batch)
        # TODO: DataProtoFuture support KVBatchMeta
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, fields=["values", "response_mask"])
        data["value"] = extract_response_from_unpad_output(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("value"))

        return batch

    def fit(self):
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # TODO(wuxibin): validate before train

        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics, timing_raw = {}, {}
                self._start_profiling()
                with marked_timer("step", timing_raw):
                    metrics = self.step(batch_dict, metrics, timing_raw)
                self._stop_profiling()
                self.logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

    def step(self, batch_dict: dict, metrics: dict, timing_raw: dict) -> dict:
        # 1. put batch to agent loop manager
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        self.agent_loop_manager.generate_sequences(batch)

        # 2. get batch from replay buffer
        with marked_timer("gen", timing_raw, color="red"):
            batch = self.replay_buffer.sample(partition_id="train", global_steps=self.global_steps)
        batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self.checkpoint_manager.sleep_replicas()

        # 3. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch)

        # 4. balance batch across data parallel groups
        self._balance_batch(batch, metrics=metrics)

        # 5. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch)

        # 7. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch)

        # 8. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch)


@ray.remote
class TaskRunner:
    def __init__(self) -> None:
        # role => worker class
        self.role_worker_mapping = {}
        # role => resource pool
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker to mapping."""
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

    def add_critic_worker(self, config):
        """Add critic worker to mapping."""
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        # Global resource pool is used for actor, rollout, critic, ref
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # Add separate resource pool for reward model if enabled
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # initialize transfer queue
        tq.init(config.transfer_queue)

        self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.init_resource_pool_mgr(config)

        trainer = PPOTrainer(
            config=config,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    config.transfer_queue.enable = True

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
