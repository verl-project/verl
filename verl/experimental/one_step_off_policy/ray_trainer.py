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
import logging
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager
from verl.utils.config import omega_conf_to_dataclass


logger = logging.getLogger(__name__)


@contextmanager
def _stage_scheduler_lease(config, stage: str, pool: str, global_step: int, task_id: str | None = None):
    scheduler_url = os.environ.get("STAGE_SCHEDULER_URL", "").strip()
    if not scheduler_url:
        yield None
        return

    try:
        from scheduler.stage_client import stage_lease
    except Exception:
        yield None
        return

    project = getattr(config.trainer, "project_name", "verl")
    experiment = getattr(config.trainer, "experiment_name", "default")
    job_prefix = os.environ.get("VERL_JOB_ID", f"{project}:{experiment}")
    job_id = f"{job_prefix}:{task_id}" if task_id else job_prefix
    lease_stage = f"{stage}:{task_id}:step-{global_step}" if task_id else f"{stage}:step-{global_step}"
    with stage_lease(job_id=job_id, stage=lease_stage, pool=pool):
        yield None


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

        # Skip rollout worker mapping and let agentloop create it.
        role_worker_mapping.pop(Role.Rollout, None)
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

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

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._adjust_phase_interleave_total_steps()

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
        self.current_phase_task_id = None
        self._phase_train_lease_active = False
        self._phase_loaded_actor_task_id = None
        self._phase_job_steps = {}
        self._phase_job_state_initialized = False
        self._phase_rollout_resources_active = True
        self._phase_train_resources_active = True
        self._phase_train_checkpoint_step = 0
        self._phase_actor_dp_size = None
        self.phase_rollout_ready_hook = None

    def _phase_skip_initial_rollout_sync_enabled(self) -> bool:
        return bool(self.config.trainer.get("phase_skip_initial_rollout_sync", False))

    def init_workers(self):
        """Initialize train-side workers, optionally deferring rollout startup.

        Independent phase jobs should not acquire the rollout node while they
        only need to initialize training state. Otherwise one job can hold A10
        during init while another job holds L20 during rollout, creating a
        cross-phase wait. With lazy rollout enabled, vLLM is started only after
        the job has acquired the rollout phase lease.
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()

        lazy_rollout = bool(self.config.trainer.get("phase_lazy_rollout_init", False))
        if lazy_rollout:
            self.llm_server_manager = None
            self.async_rollout_manager = None
            self._phase_rollout_resources_active = False
            replicas = []
        else:
            self._init_async_rollout_manager()
            replicas = self.llm_server_manager.get_replicas()

        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager

        self.checkpoint_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
            trainer=self.actor_rollout_wg,
            replicas=replicas,
        )

    def _adjust_phase_interleave_total_steps(self) -> None:
        if not self.config.trainer.get("phase_interleave_tasks", False):
            return
        if self.config.trainer.total_training_steps is not None:
            return

        task_count = len(self._phase_task_ids())
        if task_count <= 1:
            return

        self.total_training_steps *= task_count
        logger.info(
            "phase interleave enabled for %s tasks; total training steps adjusted to %s",
            task_count,
            self.total_training_steps,
        )
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            print(f"Warning: Could not set phase interleave total_training_steps in config. Error: {e}")

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor]:
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
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _get_dp_size(self, worker_group, role: str) -> int:
        if worker_group is None and role == "actor" and self._phase_actor_dp_size is not None:
            return self._phase_actor_dp_size

        dp_size = super()._get_dp_size(worker_group, role)
        if role == "actor":
            self._phase_actor_dp_size = dp_size
        return dp_size

    def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.llm_server_manager = LLMServerManager.create(config=self.config)
        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

    def _phase_task_ids(self) -> list[str]:
        raw_ids = self.config.trainer.get("phase_task_ids", None)
        if raw_ids:
            return [str(task_id) for task_id in raw_ids]
        num_tasks = int(self.config.trainer.get("phase_num_tasks", 1))
        return [f"task-{idx}" for idx in range(num_tasks)]

    def _phase_independent_jobs_enabled(self) -> bool:
        return bool(self.config.trainer.get("phase_independent_jobs", False))

    def _phase_release_rollout_enabled(self) -> bool:
        return bool(self.config.trainer.get("phase_release_rollout_after_generate", False))

    def _phase_release_train_enabled(self) -> bool:
        return bool(self.config.trainer.get("phase_release_train_after_update", False))

    def _phase_job_state_root(self) -> str:
        root = str(self.config.trainer.get("phase_job_state_dir", "/dev/shm/verl-phase-jobs"))
        if self._phase_independent_jobs_enabled():
            job_id = os.environ.get("VERL_JOB_ID", self.config.trainer.get("experiment_name", "verl-job"))
            root = os.path.join(root, self._phase_safe_task_id(job_id))
        return root

    def _phase_safe_task_id(self, task_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id))

    def _phase_actor_state_path(self, task_id: str) -> str:
        return os.path.join(self._phase_job_state_root(), self._phase_safe_task_id(task_id), "actor")

    def _phase_single_job_actor_state_path(self) -> str:
        job_id = os.environ.get("VERL_JOB_ID", self.config.trainer.get("experiment_name", "verl-job"))
        return os.path.join(self._phase_job_state_root(), self._phase_safe_task_id(job_id), "actor")

    def _init_phase_job_states(self) -> None:
        if not self._phase_independent_jobs_enabled() or self._phase_job_state_initialized:
            return

        self._phase_job_steps = {task_id: 0 for task_id in self._phase_task_ids()}
        state_root = self._phase_job_state_root()
        if os.path.exists(state_root):
            shutil.rmtree(state_root)
        for task_id in self._phase_task_ids():
            actor_path = self._phase_actor_state_path(task_id)
            logger.info("initializing phase job state task=%s path=%s", task_id, actor_path)
            self.actor_wg.save_checkpoint(actor_path, None, 0, max_ckpt_to_keep=None)

        # The actor currently contains the same initial state for all jobs, but
        # force the first real phase to declare which job it is loading.
        self._phase_loaded_actor_task_id = None
        self._phase_job_state_initialized = True
        if self._phase_release_train_enabled():
            self._release_phase_train_resources()

    def _load_phase_job_state(self, task_id: str) -> None:
        if not self._phase_independent_jobs_enabled():
            return
        if self._phase_loaded_actor_task_id == task_id:
            return

        actor_path = self._phase_actor_state_path(task_id)
        logger.info("loading phase job state task=%s path=%s", task_id, actor_path)
        self.actor_wg.load_checkpoint(actor_path, del_local_after_load=False)
        self._phase_loaded_actor_task_id = task_id

    def _save_phase_job_state(self, task_id: str) -> None:
        if not self._phase_independent_jobs_enabled():
            return

        actor_path = self._phase_actor_state_path(task_id)
        job_step = self._phase_job_steps.get(task_id, 0)
        logger.info("saving phase job state task=%s job_step=%s path=%s", task_id, job_step, actor_path)
        if os.path.exists(actor_path):
            shutil.rmtree(actor_path)
        self.actor_wg.save_checkpoint(actor_path, None, job_step, max_ckpt_to_keep=None)
        self._phase_loaded_actor_task_id = task_id

    def _prepare_phase_rollout_state(self, task_id: str, phase_step: int, timing_raw: dict) -> None:
        if self._phase_release_rollout_enabled() and not self._phase_rollout_resources_active:
            self._restore_phase_rollout_resources()

        if not self._phase_independent_jobs_enabled():
            if self._phase_release_rollout_enabled():
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    train_pool = os.environ.get("VERL_TRAIN_POOL", "a10-train")
                    with _stage_scheduler_lease(
                        self.config,
                        stage="sync_rollout_weights",
                        pool=train_pool,
                        global_step=phase_step,
                        task_id=task_id,
                    ):
                        self._restore_phase_train_resources()
                        self._fit_update_weights()
                        if self._phase_release_train_enabled():
                            self._save_phase_train_state()
                            self._release_phase_train_resources()
            return

        train_pool = os.environ.get("VERL_TRAIN_POOL", "a10-train")
        if self._phase_skip_initial_rollout_sync_enabled() and self._phase_job_steps.get(task_id, 0) == 0:
            logger.info("skipping initial rollout weight sync task=%s step=%s", task_id, phase_step)
            return

        with marked_timer("load_rollout_state", timing_raw, color="purple"):
            with _stage_scheduler_lease(
                self.config,
                stage="load_rollout_state",
                pool=train_pool,
                global_step=phase_step,
                task_id=task_id,
            ):
                self._restore_phase_train_resources()
                self._load_phase_job_state(task_id)
                self._fit_update_weights()
                if self._phase_release_train_enabled():
                    self._save_phase_job_state(task_id)
                    self._release_phase_train_resources()

    def _restore_phase_rollout_resources(self) -> None:
        logger.info("restoring phase rollout resources")
        self._init_async_rollout_manager()
        self.checkpoint_manager.replicas = self.llm_server_manager.get_replicas()
        self._phase_rollout_resources_active = True
        if self.phase_rollout_ready_hook is not None:
            self.phase_rollout_ready_hook(self)

    def _release_phase_rollout_resources(self) -> None:
        if not self._phase_release_rollout_enabled() or not self._phase_rollout_resources_active:
            return
        logger.info("releasing phase rollout resources")
        try:
            self.llm_server_manager.shutdown(release_resource_pool=True)
        finally:
            self.async_rollout_manager = None
            self.llm_server_manager = None
            self.checkpoint_manager.replicas = []
            self._phase_rollout_resources_active = False

    def _phase_train_resource_pool_name(self) -> str:
        return "trainer_pool"

    def _save_phase_train_state(self) -> None:
        if not self._phase_release_train_enabled():
            return
        if self._phase_independent_jobs_enabled():
            task_id = self.current_phase_task_id or "task-0"
            self._save_phase_job_state(task_id)
            return
        actor_path = self._phase_single_job_actor_state_path()
        logger.info("saving phase train state path=%s step=%s", actor_path, self.global_steps)
        if os.path.exists(actor_path):
            shutil.rmtree(actor_path)
        self.actor_wg.save_checkpoint(actor_path, None, self.global_steps, max_ckpt_to_keep=None)
        self._phase_train_checkpoint_step = self.global_steps

    def _release_phase_train_resources(self) -> None:
        if not self._phase_release_train_enabled() or not self._phase_train_resources_active:
            return
        logger.info("releasing phase train resources")
        if self.actor_rollout_wg is not None:
            try:
                self._phase_actor_dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
            except Exception as e:
                logger.warning("failed to cache actor dp size before train resource release: %s", e)
        for role_name in (str(Role.Actor), str(Role.Critic), str(Role.RefPolicy), str(Role.RewardModel)):
            worker_group = self.all_wg.get(role_name) if hasattr(self, "all_wg") else None
            if worker_group is not None:
                worker_group.shutdown(release_resource_pool=False)
        if hasattr(self, "resource_pool_manager"):
            self.resource_pool_manager.release_resource_pool(self._phase_train_resource_pool_name())
        self.all_wg = {}
        self.actor_wg = None
        self.actor_rollout_wg = None
        self.checkpoint_manager.trainer = None
        self._phase_train_resources_active = False

    def _restore_phase_train_resources(self) -> None:
        if not self._phase_release_train_enabled() or self._phase_train_resources_active:
            return
        logger.info("restoring phase train resources")
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self.checkpoint_manager.trainer = self.actor_wg
        if self._phase_independent_jobs_enabled():
            task_id = self.current_phase_task_id or "task-0"
            self._load_phase_job_state(task_id)
            self._phase_train_resources_active = True
            return
        actor_path = self._phase_single_job_actor_state_path()
        if os.path.exists(actor_path):
            self.actor_wg.load_checkpoint(actor_path, del_local_after_load=False)
        else:
            logger.warning("phase train checkpoint path does not exist; using freshly initialized actor: %s", actor_path)
        self._phase_train_resources_active = True

    def _create_single_continuous_iterator(self, task_id: str):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, task_id, batch_dict

    def _create_continuous_iterator(self):
        if not self.config.trainer.get("phase_interleave_tasks", False):
            yield from self._create_single_continuous_iterator("task-0")
            return

        task_ids = self._phase_task_ids()
        logger.info("phase interleave task order: %s", task_ids)
        iterators = {
            task_id: self._create_single_continuous_iterator(task_id)
            for task_id in task_ids
        }
        while iterators:
            for task_id in list(iterators):
                try:
                    yield next(iterators[task_id])
                except StopIteration:
                    iterators.pop(task_id, None)

    async def _async_gen_next_batch(self, continuous_iterator, phase_step: int):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, task_id, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["phase_task_id"] = task_id
        batch.meta_info["phase_step"] = phase_step
        logger.info("phase rollout start task=%s step=%s epoch=%s", task_id, phase_step, epoch)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = phase_step
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        rollout_pool = os.environ.get("VERL_ROLLOUT_POOL", "l20-rollout")

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            with _stage_scheduler_lease(
                self.config,
                stage="rollout",
                pool=rollout_pool,
                global_step=phase_step,
                task_id=task_id,
            ):
                if self._phase_release_rollout_enabled() and not self._phase_rollout_resources_active:
                    self._restore_phase_rollout_resources()
                self._prepare_phase_rollout_state(task_id, phase_step, timing_raw)
                gen_batch_output = await self.async_rollout_manager.generate_sequences(gen_batch_output)

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
        self._release_phase_rollout_resources()

        # Launch individual reward computations as each generation completes
        future_reward = None

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, phase_step, task_id, batch, future_reward

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

        # Load checkpoint before doing anything. In lazy-rollout mode the
        # rollout/vLLM replicas do not exist yet, so the first weight sync is
        # deferred until the job enters a rollout lease and restores rollout
        # resources.
        self._load_checkpoint()
        if self._phase_independent_jobs_enabled():
            self._init_phase_job_states()
        elif self.config.trainer.get("phase_lazy_rollout_init", False):
            logger.info("deferring initial rollout weight sync until lazy rollout init")
        else:
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
        batch_data_future = asyncio.create_task(
            self._async_gen_next_batch(continuous_iterator, phase_step=self.global_steps)
        )
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
            if batch is None:
                self.is_last_step = True
                return None

            # await asyncio.sleep(0) ensures:
            # Asynchronous tasks can start executing immediately
            # The event loop can handle other pending coroutines
            # Prevents computations in a certain phase from blocking the entire asynchronous workflow
            #
            # The purpose here is to ensure that after triggering
            # `self.async_rollout_manager.generate_sequences(gen_batch_output)`,
            # the subsequent relevant logic can proceed in a timely manner
            await asyncio.sleep(0)
            batch = self._fit_compute_reward(batch)
            await asyncio.sleep(0)
            train_pool = os.environ.get("VERL_TRAIN_POOL", "a10-train")
            with _stage_scheduler_lease(
                self.config,
                stage="train",
                pool=train_pool,
                global_step=self.global_steps,
                task_id=self.current_phase_task_id,
            ):
                logger.info("phase train start task=%s step=%s", self.current_phase_task_id, self.global_steps)
                self._phase_train_lease_active = True
                try:
                    self._restore_phase_train_resources()
                    if self._phase_independent_jobs_enabled():
                        self._load_phase_job_state(self.current_phase_task_id)
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
                    if self._phase_independent_jobs_enabled():
                        task_id = self.current_phase_task_id
                        self._phase_job_steps[task_id] = self._phase_job_steps.get(task_id, 0) + 1
                        self.metrics[f"phase/{task_id}/job_step"] = self._phase_job_steps[task_id]
                        self._save_phase_job_state(task_id)
                    if self._phase_release_train_enabled():
                        self._save_phase_train_state()
                        self._release_phase_train_resources()
                finally:
                    self._phase_train_lease_active = False
            self._fit_dump_data(batch)
            await asyncio.sleep(0)

        self._fit_validate()
        await asyncio.sleep(0)
        self._fit_save_checkpoint()
        await asyncio.sleep(0)
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_experimental(batch)
        self._fit_postprocess_step()

        return batch_data_future

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("gen", timing_raw, color="red"):
            batch_data = await batch_data_future
            if batch_data is None:
                return None, None
            _metrics, _timing_raw, epoch, phase_step, task_id, batch, future_reward = batch_data
            self.current_phase_task_id = task_id
            self.epoch = epoch
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            timing_raw.update(batch.meta_info["timing"])
            timing_raw.update(_timing_raw)
            metrics.update(_metrics)
            batch.meta_info.pop("timing", None)

        if not self._phase_independent_jobs_enabled():
            # sync weights from actor to rollout
            with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                rollout_pool = os.environ.get("VERL_ROLLOUT_POOL", "l20-rollout")
                with _stage_scheduler_lease(
                    self.config,
                    stage="sync_rollout_weights",
                    pool=rollout_pool,
                    global_step=self.global_steps,
                    task_id=task_id,
                ):
                    self._fit_update_weights()

        # async next generation
        if not self.is_last_step:
            batch_data_future = asyncio.create_task(
                self._async_gen_next_batch(continuous_iterator, phase_step=self.global_steps + 1)
            )
            await asyncio.sleep(0)
        else:
            batch_data_future = None

        return batch, batch_data_future
