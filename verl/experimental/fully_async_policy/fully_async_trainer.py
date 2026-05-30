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
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import psutil
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
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    _debug_actor_eval_alignment_snapshot,
    _debug_dataproto_summary,
    _debug_shape,
    _rollout_corr_debug_limit,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


def _bytes_to_gb(n: int | float) -> float:
    return float(n) / (1024**3)


def _tensor_bytes(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(_tensor_bytes(v) for v in obj)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return sum(_tensor_bytes(v) for v in obj.flat)
    return 0


def _object_bytes(obj: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> int:
    """Best-effort recursive host-memory estimator for Python/numpy/torch objects."""
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, np.ndarray):
        total = int(obj.nbytes)
        if obj.dtype == object and _depth < 8:
            total += sum(_object_bytes(v, _seen=_seen, _depth=_depth + 1) for v in obj.flat)
        return total

    size = sys.getsizeof(obj, 0)
    if _depth >= 8:
        return size

    # PIL.Image does not expose nbytes, but raw RGB/RGBA storage dominates.
    if hasattr(obj, "size") and hasattr(obj, "mode"):
        try:
            width, height = obj.size
            channels = len(obj.getbands()) if hasattr(obj, "getbands") else 3
            size += int(width) * int(height) * int(channels)
        except Exception:
            pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _object_bytes(k, _seen=_seen, _depth=_depth + 1)
            size += _object_bytes(v, _seen=_seen, _depth=_depth + 1)
    elif isinstance(obj, list | tuple | set | frozenset):
        for v in obj:
            size += _object_bytes(v, _seen=_seen, _depth=_depth + 1)
    return int(size)


def _intermediate_summary(batch: DataProto) -> dict[str, Any]:
    cache_key = "__intermediate_trajectories_cache__"
    cache = (batch.meta_info or {}).get(cache_key)
    interm_col = None
    if isinstance(cache, dict):
        interm_col = cache.get("intermediate_col")
    elif batch.non_tensor_batch and "intermediate_trajectories" in batch.non_tensor_batch:
        interm_col = batch.non_tensor_batch["intermediate_trajectories"]

    if interm_col is None:
        return {"rows_with_intermediate": 0, "num_intermediate": 0, "per_row_counts": []}
    counts = [len(x) if x else 0 for x in interm_col]
    return {
        "rows_with_intermediate": sum(1 for c in counts if c),
        "num_intermediate": int(sum(counts)),
        "per_row_counts": counts,
        "payload_gb": _bytes_to_gb(_object_bytes(interm_col)),
    }


def _sort_dataproto_by_sample_key(batch: DataProto) -> tuple[str | None, int, bool]:
    """Stable-sort rows so rows from the same sample/image bank are contiguous.

    Worker dispatch still chunks by row count; this reorder improves image bank
    locality before dispatch without changing row contents or loss weights.
    """
    nt = batch.non_tensor_batch or {}
    key_name = None
    for candidate in ("uid", "image_bank_ref", "rollout_group_id"):
        values = nt.get(candidate)
        if values is not None and len(values) == len(batch):
            key_name = candidate
            break
    if key_name is None:
        return None, 0, False

    values = nt[key_name]
    keys = ["<none>" if value is None else str(value) for value in values]
    group_count = len(set(keys))
    order = sorted(range(len(keys)), key=lambda idx: (keys[idx], idx))
    changed = any(idx != original for original, idx in enumerate(order))
    if changed:
        batch.reorder(torch.tensor(order, dtype=torch.long))
    return key_name, group_count, changed


def _dataproto_storage_summary(batch: DataProto | None) -> dict[str, Any]:
    if batch is None:
        return {"batch_len": 0}
    tensor_bytes = 0
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    if batch.batch is not None:
        for k, v in batch.batch.items():
            tensor_bytes += _tensor_bytes(v)
            if isinstance(v, torch.Tensor):
                tensor_shapes[k] = tuple(v.shape)

    non_tensor_bytes = _object_bytes(batch.non_tensor_batch or {})
    meta_bytes = _object_bytes(batch.meta_info or {})
    mm_bytes = 0
    nt = batch.non_tensor_batch or {}
    if "multi_modal_inputs" in nt:
        mm_bytes = _object_bytes(nt["multi_modal_inputs"])

    return {
        "batch_len": len(batch),
        "tensor_gb": _bytes_to_gb(tensor_bytes),
        "non_tensor_gb": _bytes_to_gb(non_tensor_bytes),
        "meta_gb": _bytes_to_gb(meta_bytes),
        "multi_modal_inputs_gb": _bytes_to_gb(mm_bytes),
        "tensor_shapes": tensor_shapes,
        **_intermediate_summary(batch),
    }


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
        self._actor_eval_logprob_debug_count = 0
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
            logger.info("[FullyAsyncTrainer] split before val_dataset total len: %d", len(val_dataset))
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[rollout_gpus:]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            logger.info("[FullyAsyncTrainer] split after val_dataset total len: %d", len(val_dataset))
            self.val_dataset = val_dataset
            # update val_dataloader
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            from torchdata.stateful_dataloader import StatefulDataLoader

            logger.info("[FullyAsyncTrainer] create val_dataloader with batch_size: %s", val_batch_size)
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
        logger.info("[FullyAsyncTrainer] Checkpoint manager initialized")

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
            logger.warning("Could not set total_training_steps in config. Structure missing? Error: %r", e)

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
        logger.info(
            "[FullyAsyncTrainer] Requesting %d samples from queue",
            self.required_samples,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            queue_result = await self.message_queue_client.get_sample()
            if queue_result is None:
                sample, queue_len = None, 0
            else:
                sample, queue_len = queue_result

            if sample is None:
                logger.info(
                    "[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    "Collected %d/%d samples",
                    len(queue_samples),
                    self.required_samples,
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                logger.info(
                    "[FullyAsyncTrainer] Collected %d/%d samples. mq_len: %s",
                    len(queue_samples),
                    self.required_samples,
                    queue_len,
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            logger.warning("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        logger.info(
            "[FullyAsyncTrainer] Loop collection completed: %d/%d samples, total wait time: %.2f seconds. mq_len: %s",
            len(queue_samples),
            self.required_samples,
            total_wait_time,
            queue_len,
        )

        materialized_samples = []
        sample_refs = []
        sample_ref_positions = []
        for sample in queue_samples:
            if isinstance(sample, bytes | bytearray):
                # Backward-compatible path for samples produced by older rollouters.
                materialized_samples.append(ray.cloudpickle.loads(sample))
            elif isinstance(sample, ray.ObjectRef):
                sample_ref_positions.append(len(materialized_samples))
                materialized_samples.append(None)
                sample_refs.append(sample)
            elif isinstance(sample, list | tuple) and len(sample) == 1 and isinstance(sample[0], ray.ObjectRef):
                sample_ref_positions.append(len(materialized_samples))
                materialized_samples.append(None)
                sample_refs.append(sample[0])
            else:
                # Compatibility for queues populated by code that passed ObjectRef
                # as a top-level actor argument and was auto-dereferenced by Ray.
                materialized_samples.append(sample)
        if sample_refs:
            resolved_samples = ray.get(sample_refs)
            for position, resolved_sample in zip(sample_ref_positions, resolved_samples, strict=True):
                materialized_samples[position] = resolved_sample
        queue_samples = materialized_samples
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
            logger.info("[FullyAsyncTrainer] Init reward loop")
            super()._init_reward_loop()

    async def _init_async_rollout_manager(self):
        # use async rollout do validate
        logger.info(
            "[FullyAsyncTrainer] use_trainer_do_validate: %s",
            self.config.async_training.use_trainer_do_validate,
        )
        if self.config.async_training.use_trainer_do_validate:
            logger.info("[FullyAsyncTrainer] Init async rollout manager")

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
            logger.info("[FullyAsyncTrainer] async_rollout_manager initialized")

            # Modify checkpoint_engine config to use naive backend
            checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
            original_backend = checkpoint_engine_cfg.backend
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = "naive"
            checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

            logger.info("[FullyAsyncTrainer] checkpoint_engine_config: %s", checkpoint_engine_config)

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

            logger.info("[FullyAsyncTrainer] colocate_checkpoint_manager initialized")

        else:
            logger.info("[FullyAsyncTrainer] Skip async rollout manager (use_trainer_do_validate=False)")

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger.info("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
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
                logger.info("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    def _log_batch_storage(self, stage: str, batch: DataProto | None) -> None:
        try:
            summary = _dataproto_storage_summary(batch)
            rss_gb = _bytes_to_gb(psutil.Process(os.getpid()).memory_info().rss)
            ts = datetime.now().isoformat(timespec="milliseconds")
            message = (
                f"[FullyAsyncTrainer][Storage][{stage}] "
                f"ts={ts} pid={os.getpid()} global_step={getattr(self, 'global_steps', None)} "
                f"local_trigger_step={getattr(self, 'local_trigger_step', None)} "
                f"param_version={getattr(self, 'current_param_version', None)} "
                f"rss={rss_gb:.3f}GB batch_len={summary.get('batch_len')} "
                f"tensor={summary.get('tensor_gb', 0.0):.3f}GB "
                f"non_tensor={summary.get('non_tensor_gb', 0.0):.3f}GB "
                f"meta={summary.get('meta_gb', 0.0):.3f}GB "
                f"multi_modal_inputs={summary.get('multi_modal_inputs_gb', 0.0):.3f}GB "
                f"intermediate_rows={summary.get('num_intermediate', 0)} "
                f"rows_with_intermediate={summary.get('rows_with_intermediate', 0)} "
                f"per_row_counts={summary.get('per_row_counts', [])} "
                f"tensor_shapes={summary.get('tensor_shapes', {})}"
            )
            print(message, flush=True)
            if "payload_gb" in summary:
                print(
                    f"[FullyAsyncTrainer][Storage][{stage}] ts={ts} intermediate_payload={summary['payload_gb']:.3f}GB",
                    flush=True,
                )
        except Exception as exc:
            logger.exception("[FullyAsyncTrainer][Storage][%s] failed to collect storage log", stage)
            print(
                f"[FullyAsyncTrainer][Storage][{stage}] failed to collect storage log: {exc!r}",
                flush=True,
            )

    def _log_rollout_corr_actor_update(
        self,
        stage: str,
        batch: DataProto | None,
        actor_metrics: dict[str, Any] | None = None,
    ) -> None:
        if str(os.getenv("VERL_ROLLOUT_CORR_UPDATE_DEBUG", "1")).lower() in {"0", "false", "no"}:
            return

        def _as_float(value: Any) -> float | None:
            try:
                if isinstance(value, torch.Tensor):
                    return float(value.detach().float().mean().cpu().item())
                if isinstance(value, np.ndarray):
                    return float(np.asarray(value, dtype=np.float64).mean())
                return float(value)
            except Exception:
                return None

        def _count_values(values: Any, limit: int = 8) -> dict[str, int]:
            if values is None:
                return {}
            try:
                arr = np.asarray(values).reshape(-1).tolist()
            except Exception:
                arr = list(values) if isinstance(values, list | tuple) else [values]
            counts: dict[str, int] = {}
            for raw in arr:
                key = str(int(raw)) if isinstance(raw, int | np.integer) else str(raw)
                counts[key] = counts.get(key, 0) + 1
            return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])

        def _min_max(values: Any) -> tuple[Any, Any]:
            if values is None:
                return None, None
            try:
                arr = np.asarray(values).reshape(-1)
                if arr.size == 0:
                    return None, None
                return arr.min().item(), arr.max().item()
            except Exception:
                return None, None

        def _as_list(values: Any) -> list[Any]:
            if values is None:
                return []
            try:
                if hasattr(values, "tolist"):
                    values = values.tolist()
                else:
                    values = list(values)
            except Exception:
                values = [values]
            return [getattr(item, "data", item) for item in values]

        def _image_ref_count(value: Any) -> int:
            if not isinstance(value, dict):
                return 0
            return len(value.get("image_ids") or []) + len(value.get("video_ids") or [])

        try:
            summary: dict[str, Any] = {
                "stage": stage,
                "global_steps": self.global_steps,
                "local_trigger_step": self.local_trigger_step,
                "trigger_parameter_sync_step": self.trigger_parameter_sync_step,
                "current_param_version": self.current_param_version,
            }
            if batch is not None:
                pad_rows = int(batch.meta_info.get("fully_async/pad/num_padding_rows", 0) or 0)
                rollout_group_ids = (batch.non_tensor_batch or {}).get("rollout_group_id")
                rollout_groups = (
                    len(set(np.asarray(rollout_group_ids).reshape(-1).tolist()))
                    if rollout_group_ids is not None
                    else None
                )
                summary.update(
                    {
                        "batch_len": len(batch),
                        "valid_batch_len": max(len(batch) - pad_rows, 0),
                        "pad_rows": pad_rows,
                        "trajectory_param_versions": _count_values(batch.meta_info.get("trajectory_param_versions")),
                        "trajectory_roles": _count_values((batch.non_tensor_batch or {}).get("trajectory_role")),
                        "rollout_groups": rollout_groups,
                    }
                )
                sample_steps = (batch.non_tensor_batch or {}).get("global_steps")
                sample_min, sample_max = _min_max(sample_steps)
                summary["sample_global_steps_min"] = sample_min
                summary["sample_global_steps_max"] = sample_max

                if batch.batch is not None and "response_mask" in batch.batch.keys():
                    response_mask = batch.batch["response_mask"]
                    valid = response_mask.bool()
                    row_lengths = response_mask.detach().float().sum(dim=-1).cpu()
                    summary.update(
                        {
                            "valid_tokens": int(valid.sum().detach().cpu().item()),
                            "response_len_mean": float(row_lengths.mean().item()) if row_lengths.numel() else None,
                            "response_len_min": float(row_lengths.min().item()) if row_lengths.numel() else None,
                            "response_len_max": float(row_lengths.max().item()) if row_lengths.numel() else None,
                        }
                    )
                    for key in ("rollout_log_probs", "old_log_probs", "ref_log_prob"):
                        if key in batch.batch.keys():
                            values = batch.batch[key][valid].detach().float()
                            if values.numel() > 0:
                                summary[f"{key}_mean"] = float(values.mean().cpu().item())
                                summary[f"{key}_min"] = float(values.min().cpu().item())
                                summary[f"{key}_max"] = float(values.max().cpu().item())
                    diff_pairs = (
                        ("old_log_probs", "rollout_log_probs", "old_minus_rollout"),
                        ("rollout_log_probs", "ref_log_prob", "rollout_minus_ref"),
                        ("old_log_probs", "ref_log_prob", "old_minus_ref"),
                    )
                    for left_key, right_key, out_key in diff_pairs:
                        if left_key in batch.batch.keys() and right_key in batch.batch.keys():
                            diff = (batch.batch[left_key] - batch.batch[right_key])[valid].detach().float()
                            if diff.numel() > 0:
                                summary[f"{out_key}_mean"] = float(diff.mean().cpu().item())
                                summary[f"{out_key}_min"] = float(diff.min().cpu().item())
                                summary[f"{out_key}_max"] = float(diff.max().cpu().item())

                    nt = batch.non_tensor_batch or {}
                    roles = _as_list(nt.get("trajectory_role"))
                    if roles:
                        role_stats: dict[str, dict[str, Any]] = {}
                        image_refs = _as_list(nt.get("multi_modal_refs"))
                        turn_numbers = _as_list(nt.get("turn_number"))
                        prompt_lens = None
                        if "attention_mask" in batch.batch.keys() and "prompts" in batch.batch.keys():
                            prompt_width = batch.batch["prompts"].shape[-1]
                            prompt_lens = (
                                batch.batch["attention_mask"][:, :prompt_width].detach().float().sum(dim=-1).cpu()
                            )

                        for role in sorted(set(str(role) for role in roles)):
                            row_indices = [idx for idx, value in enumerate(roles) if str(value) == role]
                            row_indices = [idx for idx in row_indices if idx < response_mask.shape[0]]
                            if not row_indices:
                                continue
                            idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=response_mask.device)
                            role_valid = valid.index_select(0, idx_tensor)
                            stat: dict[str, Any] = {
                                "rows": len(row_indices),
                                "valid_tokens": int(role_valid.sum().detach().cpu().item()),
                            }
                            role_response_lens = row_lengths.index_select(0, idx_tensor.cpu())
                            if role_response_lens.numel() > 0:
                                stat["response_len_mean"] = float(role_response_lens.mean().item())
                                stat["response_len_max"] = float(role_response_lens.max().item())
                            if prompt_lens is not None:
                                role_prompt_lens = prompt_lens.index_select(0, idx_tensor.cpu())
                                stat["prompt_len_mean"] = float(role_prompt_lens.mean().item())
                                stat["prompt_len_min"] = float(role_prompt_lens.min().item())
                                stat["prompt_len_max"] = float(role_prompt_lens.max().item())
                            if image_refs:
                                counts = [
                                    _image_ref_count(image_refs[idx]) if idx < len(image_refs) else 0
                                    for idx in row_indices
                                ]
                                if counts:
                                    stat["image_refs_mean"] = float(np.mean(counts))
                                    stat["image_refs_max"] = int(max(counts))
                            if turn_numbers:
                                turns = []
                                for idx in row_indices:
                                    if idx >= len(turn_numbers):
                                        continue
                                    try:
                                        turns.append(int(turn_numbers[idx]))
                                    except Exception:
                                        pass
                                if turns:
                                    stat["turn_min"] = min(turns)
                                    stat["turn_max"] = max(turns)

                            for key in ("rollout_log_probs", "old_log_probs", "ref_log_prob"):
                                if key in batch.batch.keys():
                                    values = batch.batch[key].index_select(0, idx_tensor)[role_valid].detach().float()
                                    if values.numel() > 0:
                                        stat[f"{key}_mean"] = float(values.mean().cpu().item())
                            for left_key, right_key, out_key in diff_pairs:
                                if left_key in batch.batch.keys() and right_key in batch.batch.keys():
                                    diff = (
                                        (
                                            batch.batch[left_key].index_select(0, idx_tensor)
                                            - batch.batch[right_key].index_select(0, idx_tensor)
                                        )[role_valid]
                                        .detach()
                                        .float()
                                    )
                                    if diff.numel() > 0:
                                        stat[f"{out_key}_mean"] = float(diff.mean().cpu().item())
                                        stat[f"{out_key}_max"] = float(diff.max().cpu().item())
                            role_stats[role] = stat
                        summary["role_stats"] = role_stats

            if actor_metrics:
                metric_keys = (
                    "actor/rollout_corr/kl",
                    "actor/rollout_corr/k3_kl",
                    "actor/rollout_corr/training_log_ppl",
                    "actor/rollout_corr/rollout_log_ppl",
                    "actor/rollout_corr/rollout_rs_masked_fraction",
                    "actor/rollout_corr/rollout_rs_seq_masked_fraction",
                    "actor/ppo_kl",
                    "actor/pg_loss",
                    "actor/entropy_loss",
                    "actor/kl_loss",
                    "actor/grad_norm",
                )
                summary["actor_metrics"] = {
                    key: value for key in metric_keys if (value := _as_float(actor_metrics.get(key))) is not None
                }

            print(f"[RolloutCorrDebug][actor_update_{stage}] {summary}", flush=True)
        except Exception as exc:
            print(f"[RolloutCorrDebug][actor_update_{stage}] failed: {exc!r}", flush=True)

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
            self._log_batch_storage("after_generate", batch)

            # Detect position_ids ndim from the assembled batch for consistent
            # assertion throughout the pipeline.
            _pos_ndim = (
                batch.batch["position_ids"].ndim
                if batch.batch is not None and "position_ids" in batch.batch.keys()
                else None
            )

            batch = self._fit_compute_reward(batch)
            self._log_batch_storage("after_reward", batch)
            assert_batch_schema(
                batch,
                "fit_step.after_reward",
                expected_tensor_keys=_CORE_KEYS,
                require_position_ids_ndim=_pos_ndim,
                has_processor=self.processor is not None,
            )
            # Expand intermediate trajectories (and pad for even worker
            # dispatch) BEFORE any per-token forward pass, so that
            # log_prob / ref_log_prob / critic are computed over every
            # trajectory row that will participate in the actor update.
            batch = self._fit_expand_and_pad(batch)
            self._log_batch_storage("after_expand_and_pad", batch)
            assert_batch_schema(
                batch,
                "fit_step.after_expand_and_pad",
                expected_tensor_keys=_CORE_KEYS,
                require_position_ids_ndim=_pos_ndim,
            )
            batch = self._fit_compute_log_prob(batch)
            self._log_batch_storage("after_log_prob", batch)
            assert_batch_schema(batch, "fit_step.after_log_prob", require_position_ids_ndim=_pos_ndim)
            self._log_batch_storage("before_ref_log_prob", batch)
            batch = self._fit_compute_ref_log_prob(batch)
            self._log_batch_storage("after_ref_log_prob", batch)
            batch = self._fit_compute_critic(batch)
            assert_batch_schema(batch, "fit_step.after_critic", require_position_ids_ndim=_pos_ndim)
            # Advantage is computed on the FINAL subset only (GRPO group
            # stats must not see intermediate rows), then the scalar is
            # broadcast to sibling intermediate rows and scaled by 1/T_rollout
            # so that every rollout contributes equally under token-mean.
            batch = self._fit_compute_advantage(batch)
            self._log_batch_storage("after_advantage", batch)
            assert_batch_schema(batch, "fit_step.after_advantage", require_position_ids_ndim=_pos_ndim)
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

    def _debug_actor_eval_log_prob_before_update(self, batch: DataProto) -> None:
        limit = _rollout_corr_debug_limit()
        if limit <= 0 or self._actor_eval_logprob_debug_count >= limit:
            return
        self._actor_eval_logprob_debug_count += 1
        try:
            print(
                "[FullyAsyncTrainer][actor_eval_logprob_debug][enter] "
                f"ts={time.time():.3f} count={self._actor_eval_logprob_debug_count - 1} "
                f"summary={_debug_dataproto_summary(batch)}",
                flush=True,
            )
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=False,
                compute_loss=False,
            )
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            log_probs = no_padding_2_padding(log_probs, batch_td).float()
            _debug_actor_eval_alignment_snapshot(batch, log_probs, "fully_async_before_update_actor_eval")
            print(
                "[FullyAsyncTrainer][actor_eval_logprob_debug][exit] "
                f"ts={time.time():.3f} log_probs_shape={_debug_shape(log_probs)}",
                flush=True,
            )
        except Exception as exc:
            print(
                "[RolloutCorrDebug][actor_eval_logprob_alignment] "
                f"stage=fully_async_before_update_actor_eval failed={type(exc).__name__}: {exc}",
                flush=True,
            )

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Update actor using the expanded rows as one PPO mini-batch."""
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["temperature"] = rollout_config.temperature
        self._debug_actor_eval_log_prob_before_update(batch)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        pad_size = int(batch.meta_info.get("fully_async/pad/num_padding_rows", 0) or 0)
        valid_batch_size = max(len(batch) - pad_size, 1)
        global_rollout_count = int(
            batch.meta_info.get("fully_async/rollout_weight/num_groups", valid_batch_size) or valid_batch_size
        )
        expanded_mini_batch_size = len(batch)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=valid_batch_size,
            global_rollout_count=global_rollout_count,
            mini_batch_size=expanded_mini_batch_size,
            epochs=self.config.actor_rollout_ref.actor.ppo_epochs,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _fit_update_actor(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        if self.config.trainer.critic_warmup <= self.global_steps:
            self._log_rollout_corr_actor_update("begin", batch)
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)

            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            self._log_rollout_corr_actor_update("end", batch, actor_metrics=actor_output_metrics)
            metrics.update(actor_output_metrics)
        return batch

    def _update_critic(self, batch: DataProto) -> DataProto:
        """Update critic using the expanded rows as one PPO mini-batch."""
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        pad_size = int(batch.meta_info.get("fully_async/pad/num_padding_rows", 0) or 0)
        valid_batch_size = max(len(batch) - pad_size, 1)
        expanded_mini_batch_size = len(batch)
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=valid_batch_size,
            mini_batch_size=expanded_mini_batch_size,
            epochs=self.config.critic.ppo_epochs,
            seed=self.config.critic.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.critic.shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": output})

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
          2. Pad the expanded batch to a multiple of the training DP size so
             worker dispatch can split rows evenly. Padding rows have their
             training signal zeroed out and do not contribute to the actor
             loss.
        """
        timing_raw = self.timing_raw
        self._log_batch_storage("expand.before", batch)
        with marked_timer("expand_intermediate", timing_raw, color="magenta"):
            rollout_cfg = self.config.actor_rollout_ref.rollout
            rollout_n = int(rollout_cfg.get("n", 1) or 1)

            batch = expand_intermediate_trajectories_pre_log_prob(
                batch,
                tokenizer=self.tokenizer,
                processor=self.processor,
                rollout_config=rollout_cfg,
                rollout_n=rollout_n,
            )
        self._log_batch_storage("expand.after_expand", batch)
        sort_key, sort_groups, sort_changed = _sort_dataproto_by_sample_key(batch)
        print(
            "[FullyAsyncTrainer][SampleSort][after_expand] "
            f"key={sort_key} groups={sort_groups} changed={sort_changed} rows={len(batch)}",
            flush=True,
        )
        if sort_changed:
            self._log_batch_storage("expand.after_sample_sort", batch)

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

        # After intermediate expansion, the global update batch is the expanded
        # row count. Pad only so worker dispatch can split it evenly by DP size.
        training_dp_size = int(self.config.trainer.nnodes) * int(self.config.trainer.n_gpus_per_node)
        if training_dp_size > 0 and len(batch) % training_dp_size != 0:
            with marked_timer("pad_mini_batch", timing_raw, color="magenta"):
                batch, pad_size = pad_dataproto_to_divisor(batch, training_dp_size)
                zero_out_padding_rows(batch, pad_size)
                batch.meta_info["fully_async/pad/num_padding_rows"] = int(pad_size)
                if batch.batch is not None and "attention_mask" in batch.batch.keys():
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                self._log_batch_storage("expand.after_pad", batch)
        else:
            batch.meta_info["fully_async/pad/num_padding_rows"] = 0

        # Restore stashed ndarray meta_info fields onto the (possibly new)
        # batch object. These are batch-global monitoring arrays (one entry
        # per rollout), so they stay valid across pad.
        if _stashed_meta:
            batch.meta_info.update(_stashed_meta)

        if batch.batch is not None and "attention_mask" in batch.batch.keys():
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return batch

    def _fit_compute_advantage(self, batch: DataProto) -> DataProto:
        """Compute one GRPO advantage per rollout, then broadcast to its rows.

        The expanded batch contains multiple rows for the same rollout
        trajectory (intermediate steps plus the final step). All those rows have
        the same outcome reward, so only one representative row per
        ``rollout_group_id`` should enter GRPO group mean/std. The resulting
        rollout-level advantage is then broadcast to every row in that rollout
        using each row's own ``response_mask``.
        """
        import numpy as np  # local import to avoid polluting top-level namespace

        nt = batch.non_tensor_batch or {}
        roles = nt.get("trajectory_role")

        # Fast path: no intermediate rows (e.g. rollouts all finished in one
        # turn). Fall back to the standard advantage pipeline; optional rollout
        # weight normalization is disabled by default.
        if roles is None:
            saved_reward_tensor = self.reward_tensor
            saved_reward_extra_infos_dict = self.reward_extra_infos_dict
            if batch.batch is not None and "rm_scores" in batch.batch.keys():
                self.reward_tensor = batch.batch["rm_scores"]
            if saved_reward_extra_infos_dict:
                self.reward_extra_infos_dict = {
                    k: batch.non_tensor_batch[k]
                    for k in saved_reward_extra_infos_dict
                    if k in (batch.non_tensor_batch or {}) and len(batch.non_tensor_batch[k]) == len(batch)
                }
            try:
                batch = super()._fit_compute_advantage(batch)
            finally:
                self.reward_tensor = saved_reward_tensor
                self.reward_extra_infos_dict = saved_reward_extra_infos_dict
            if bool(self.config.async_training.get("normalize_rollout_weight", False)):
                batch = scatter_advantage_to_intermediate_and_normalize(batch, normalize_rollout_weight=True)
            return batch

        group_ids = nt.get("rollout_group_id")
        if group_ids is None:
            batch = super()._fit_compute_advantage(batch)
            if bool(self.config.async_training.get("normalize_rollout_weight", False)):
                batch = scatter_advantage_to_intermediate_and_normalize(batch, normalize_rollout_weight=True)
            return batch

        roles_np = np.asarray(roles, dtype=object)
        group_ids_np = np.asarray(group_ids, dtype=np.int64)
        group_order: list[int] = []
        first_by_group: dict[int, int] = {}
        final_by_group: dict[int, int] = {}
        for idx, gid_raw in enumerate(group_ids_np):
            gid = int(gid_raw)
            if gid < 0 or roles_np[idx] == "padding":
                continue
            if gid not in first_by_group:
                first_by_group[gid] = idx
                group_order.append(gid)
            if roles_np[idx] == "final":
                final_by_group[gid] = idx

        if not group_order:
            return batch

        # Pick one representative per rollout group. Prefer the final step for
        # stability, but any row in the group has the same outcome reward.
        rep_idx = np.asarray([final_by_group.get(gid, first_by_group[gid]) for gid in group_order], dtype=np.int64)
        rep_subset = batch.select_idxs(torch.as_tensor(rep_idx, dtype=torch.long))

        # The parent implementation expects rewards from ``self.reward_tensor``.
        # Use row-attached ``rm_scores`` from the representative subset so
        # rewards follow DataProto reorder/sort naturally.
        saved_reward_tensor = self.reward_tensor
        saved_reward_extra_infos_dict = self.reward_extra_infos_dict
        if rep_subset.batch is not None and "rm_scores" in rep_subset.batch.keys():
            self.reward_tensor = rep_subset.batch["rm_scores"]
        if saved_reward_extra_infos_dict:
            self.reward_extra_infos_dict = {
                k: rep_subset.non_tensor_batch[k]
                for k in saved_reward_extra_infos_dict
                if k in (rep_subset.non_tensor_batch or {}) and len(rep_subset.non_tensor_batch[k]) == len(rep_subset)
            }

        try:
            rep_subset = super()._fit_compute_advantage(rep_subset)
        finally:
            self.reward_tensor = saved_reward_tensor
            self.reward_extra_infos_dict = saved_reward_extra_infos_dict

        if "advantages" in rep_subset.batch.keys():
            if "advantages" not in batch.batch.keys():
                batch.batch["advantages"] = torch.zeros(
                    (len(batch), rep_subset.batch["advantages"].shape[-1]),
                    dtype=rep_subset.batch["advantages"].dtype,
                )
            batch.batch["advantages"][rep_idx] = rep_subset.batch["advantages"]
        if "returns" in rep_subset.batch.keys():
            if "returns" not in batch.batch.keys():
                batch.batch["returns"] = torch.zeros_like(batch.batch["advantages"])
            batch.batch["returns"][rep_idx] = rep_subset.batch["returns"]

        if "rm_scores" in batch.batch.keys():
            batch.batch["token_level_scores"] = batch.batch["rm_scores"]
            if not self.config.algorithm.use_kl_in_reward:
                batch.batch["token_level_rewards"] = batch.batch["rm_scores"]

        normalize_rollout_weight = bool(self.config.async_training.get("normalize_rollout_weight", False))
        batch = scatter_advantage_to_intermediate_and_normalize(
            batch,
            normalize_rollout_weight=normalize_rollout_weight,
            source_indices=rep_idx,
        )

        return batch

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        old_local_trigger_step = self.local_trigger_step
        old_param_version = self.current_param_version
        logger.info(
            "[FullyAsyncTrainer] global_steps: %d local_trigger_step: %d trigger_parameter_sync_step: %s %s",
            self.global_steps,
            self.local_trigger_step,
            self.trigger_parameter_sync_step,
            time_str,
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1
        if str(os.getenv("VERL_ROLLOUT_CORR_UPDATE_DEBUG", "1")).lower() not in {"0", "false", "no"}:
            print(
                "[RolloutCorrDebug][actor_update_local_step] "
                f"global_steps={self.global_steps} "
                f"local_trigger_step_before={old_local_trigger_step} "
                f"local_trigger_step_after={self.local_trigger_step} "
                f"current_param_version_before={old_param_version} "
                f"current_param_version_after={self.current_param_version} "
                f"trigger_parameter_sync_step={self.trigger_parameter_sync_step} "
                f"will_sync={self.local_trigger_step == 1}",
                flush=True,
            )

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        logger.info(
            "[FullyAsyncTrainer] _fit_update_weights, timing_s/param_sync: %.4f seconds self.current_param_version: %s",
            self.timing_raw["timing_s/param_sync"],
            self.current_param_version,
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
            logger.info("[FullyAsyncTrainer] _validate_process")
            from verl.utils.profiler import marked_timer

            # Wake up rollouter replicas and sync weights
            logger.info("[FullyAsyncTrainer] wake up replicas before validation")
            await self.colocate_checkpoint_manager.update_weights(global_steps=self.current_param_version)

            with marked_timer("trainer/validate_time", self.timing_raw):
                train_val_metrics = self._validate(True)

            # Sleep rollouter replicas to free GPU memory for validation
            logger.info("[FullyAsyncTrainer] sleep replicas after validation")
            await self.colocate_checkpoint_manager.sleep_replicas()

            logger.info(
                "[FullyAsyncTrainer] validate timing: %s",
                self.timing_raw["trainer/validate_time"],
            )
            return train_val_metrics
        else:
            logger.info("[FullyAsyncTrainer] _validate_process without async_rollout_manager")
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
                logger.info(
                    "[FullyAsyncTrainer] parameter version: %s Validation metrics: %s, timing: %s",
                    self.current_param_version,
                    new_metrics,
                    self.timing_raw["timing_s/merge_val"],
                )
        else:
            if val_metrics.metrics:
                self.logger.log(data=val_metrics.metrics, step=self.current_param_version)
                logger.info(
                    "[FullyAsyncTrainer] parameter version: %s Validation metrics: %s",
                    self.current_param_version,
                    val_metrics.metrics,
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
                logger.info("Force saving checkpoint: ESI instance expiration approaching.")
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

        logger.info("[FullyAsyncTrainer] local_global_step_folder: %s", local_global_step_folder)
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
            logger.warning(
                "[FullyAsyncTrainer] remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
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
        logger.info("[FullyAsyncTrainer] Load from checkpoint folder: %s", global_step_folder)
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        logger.info(
            "[FullyAsyncTrainer] Setting global step to %d, current_param_version to %s",
            self.global_steps,
            self.current_param_version,
        )
        logger.info("[FullyAsyncTrainer] Resuming from %s", global_step_folder)

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
