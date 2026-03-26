      
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

import json
import os
import time
import uuid
import threading
import concurrent.futures
import queue
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
# from verl.utils.dataset.gfpo_sampler import AdaptiveGFPOSampler
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GFPO:
        # GFPO 特殊处理
        # 从配置中获取 G 和 k 参数
        G = config.get("G", 16)  # 每组生成的响应数量
        k = config.get("k", 8)   # 每组保留的响应数量
        # Initialize the mask for GFPO calculation
        gfpo_calculation_mask = data.batch["response_mask"]
        # compute_gfpo_outcome_advantage
        # 支持 adaptive_k: 如果调用处（如 trainer）把 adaptive_k 放到 data.non_tensor_batch 中则传入
        adaptive_k = None
        try:
            adaptive_k = data.non_tensor_batch.get("adaptive_k", None)
        except Exception:
            adaptive_k = None
        advantages, returns = core_algos.compute_gfpo_outcome_advantage_wrapper(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=gfpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            G=G,
            k=k,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            adaptive_k=adaptive_k,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
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
        self.processor = processor
        self.config = config
        if reward_fn is not None or val_reward_fn is not None:
            raise ValueError(
                "Function-based reward injection is not supported in ray_trainer_pipeline. "
                "Configure reward through Reward Loop settings instead of reward_fn/val_reward_fn."
            )
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
            AdvantageEstimator.GFPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        reward_model_cfg = OmegaConf.select(config, "reward.reward_model")
        if reward_model_cfg is None:
            reward_model_cfg = OmegaConf.select(config, "reward_model")

        # Only validate legacy RM worker batch-size knobs when they exist.
        # Reward Loop's newer reward.reward_model schema may omit these fields entirely.
        if reward_model_cfg is not None:
            rm_use_dynamic_bsz = OmegaConf.select(reward_model_cfg, "use_dynamic_bsz")
            rm_micro_batch_size = OmegaConf.select(reward_model_cfg, "micro_batch_size")
            rm_micro_batch_size_per_gpu = OmegaConf.select(reward_model_cfg, "micro_batch_size_per_gpu")
            if reward_model_cfg.enable and rm_use_dynamic_bsz is not None and not rm_use_dynamic_bsz:
                check_mutually_exclusive(
                    rm_micro_batch_size,
                    rm_micro_batch_size_per_gpu,
                    "reward_model",
                )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        # 如果传入的 sampler 是 AbstractCurriculumSampler 的子类，则强制 num_workers=0，避免
        # DataLoader worker 缓存，因此 sampler 在主进程上只有一个实例（集中）
        # 那么就可以集中追踪每个 sample 的 reward，实现 adaptive k
        if isinstance(train_sampler, AbstractCurriculumSampler):
            assert num_workers == 0, (
                "If using curriculum, num_workers must be 0 to prevent data caching. "
                "If the dataloader caches data before the batch is done the "
                "curriculum sampler won't have the opportunity to reorder it. "
            )

        # 如果 trainer 有保存的 next_iter_state，先验证其格式是否包含 StatefulDataLoader 期望的键
        next_iter_state = getattr(self, "next_iter_state", None)
        if isinstance(next_iter_state, dict):
            # StatefulDataLoader 单进程状态至少应包含 '_num_yielded'
            if "_num_yielded" not in next_iter_state:
                print("Warning: invalid dataloader state detected, ignoring and starting fresh.")
                next_iter_state = None

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

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

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

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

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            raw_prompt_for_rm = None
            if "raw_prompt" in test_batch.non_tensor_batch:
                raw_prompt_for_rm = test_batch.non_tensor_batch["raw_prompt"].copy()
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            elif "full_prompts" in test_batch.non_tensor_batch:
                raw_prompt_for_rm = test_batch.non_tensor_batch["full_prompts"].copy()
                non_tensor_batch_keys_to_pop.append("full_prompts")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            with self._reward_batch_context():
                test_batch = self._compute_reward_batch(test_batch, raw_prompt_for_rm=raw_prompt_for_rm)
            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])
                print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            # print(lst)
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
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

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        
        For mini-pipeline mode, creates separate worker groups for actor_rollout and actor_update
        to avoid GPU resource conflicts during parallel processing.
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Simplified approach: use unified workers for prompt-level pipeline
        print("[Worker Init] Using unified worker approach for prompt-level pipeline within batch")

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

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

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        # Initialize actor workers - use unified approach for simplicity
        print("[Worker Init] Using unified actor worker (prompt-level pipeline within batch)")
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.actor_update_wg = self.actor_rollout_wg  # Same worker for both
        self.has_separate_actor_workers = False

        # Use RewardLoopManager as the single reward execution entrypoint.
        from verl.experimental.reward_loop import RewardLoopManager, migrate_legacy_reward_impl

        reward_loop_config = deepcopy(self.config)
        if (
            OmegaConf.select(reward_loop_config, "reward.reward_model") is None
            and OmegaConf.select(reward_loop_config, "reward_model") is not None
        ):
            reward_loop_config = migrate_legacy_reward_impl(reward_loop_config)

        reward_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=reward_loop_config,
            rm_resource_pool=reward_resource_pool,
        )

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,  # Always use rollout worker for generation
            )

    def _compute_reward_colocate(self, batch: DataProto) -> DataProto:
        """Compute reward through the unified Reward Loop entrypoint."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        return self.reward_loop_manager.compute_rm_score(batch, manage_lifecycle=False)

    def _get_reward_model_cfg(self, config=None):
        cfg = self.config if config is None else config
        reward_model_cfg = OmegaConf.select(cfg, "reward.reward_model")
        if reward_model_cfg is not None:
            return reward_model_cfg
        return cfg.reward_model

    def _get_reward_manager_name(self, config=None):
        cfg = self.config if config is None else config
        reward_manager_name = OmegaConf.select(cfg, "reward.reward_manager.name")
        if reward_manager_name is not None:
            return reward_manager_name
        reward_model_cfg = OmegaConf.select(cfg, "reward_model")
        return reward_model_cfg.get("reward_manager") if reward_model_cfg is not None else None

    @contextmanager
    def _reward_batch_context(self):
        reward_model_manager = getattr(self.reward_loop_manager, "reward_model_manager", None)
        if reward_model_manager is not None:
            self.reward_loop_manager.wake_up()
        try:
            yield
        finally:
            if reward_model_manager is not None:
                self.reward_loop_manager.sleep()

    def _ensure_reward_inputs_ready(self, batch: DataProto, raw_prompt_for_rm=None):
        if "raw_prompt" in batch.non_tensor_batch:
            return batch

        if raw_prompt_for_rm is None and "full_prompts" in batch.non_tensor_batch:
            raw_prompt_for_rm = batch.non_tensor_batch["full_prompts"]

        if raw_prompt_for_rm is not None:
            expected_size = len(batch)
            current_size = len(raw_prompt_for_rm)
            if current_size > 0 and expected_size > current_size:
                repeat_factor = expected_size // current_size
                batch.non_tensor_batch["raw_prompt"] = np.repeat(raw_prompt_for_rm, repeat_factor, axis=0)
            else:
                batch.non_tensor_batch["raw_prompt"] = raw_prompt_for_rm
            return batch

        batch.non_tensor_batch["raw_prompt"] = np.array([[] for _ in range(len(batch))], dtype=object)
        return batch

    def _compute_reward_batch(self, batch: DataProto, raw_prompt_for_rm=None) -> DataProto:
        self._ensure_reward_inputs_ready(batch, raw_prompt_for_rm=raw_prompt_for_rm)
        if "rm_scores" in batch.batch.keys():
            return batch

        size_divisor = len(self.reward_loop_manager.reward_loop_workers)
        padded_batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        batch_reward = self._compute_reward_colocate(padded_batch)
        padded_batch = padded_batch.union(batch_reward)
        return unpad_dataproto(padded_batch, pad_size=pad_size)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # Save actor checkpoint - use update worker if available (has the most recent model state)
        # Both workers share the same model parameters, so we only need to save once
        actor_worker_for_save = self.actor_update_wg if self.has_separate_actor_workers else self.actor_rollout_wg
        print(f"[Checkpoint] Saving actor using {'update' if self.has_separate_actor_workers else 'unified'} worker")
        
        actor_worker_for_save.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
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
                print("Training from scratch")
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
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # Load actor checkpoint to both workers if they are separate
        print(f"[Checkpoint] Loading actor \
              using {'separate' if self.has_separate_actor_workers else 'unified'} workers")
        
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        
        # Load to update worker as well if separate workers are used
        if self.has_separate_actor_workers:
            print("[Checkpoint] Loading checkpoint to actor_update_wg as well")
            self.actor_update_wg.load_checkpoint(
                actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            if isinstance(dataloader_state_dict, dict) and "_num_yielded" in dataloader_state_dict:
                try:
                    self.train_dataloader.load_state_dict(dataloader_state_dict)
                except Exception as e:
                    print(f"Warning: failed to load dataloader state (will start fresh): {e}")
            else:
                print(f"Warning: ignoring invalid/legacy dataloader state file (missing '_num_yielded'), \
                      starting from scratch")
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        # Safety check: ensure batch has the required keys
        if "attention_mask" not in batch.batch:
            print(f"[Balance Warning] No attention_mask in batch.batch. Available keys: {list(batch.batch.keys())}")
            return
            
        attention_mask = batch.batch["attention_mask"]
        tensor_batch_size = attention_mask.shape[0]
        
        # Debug: Check sizes before computing balancing - check BOTH tensor and non_tensor sizes  
        debug_balance = hasattr(self.config, 'debug_level') and self.config.debug_level > 2
        if debug_balance:
            print(f"[Balance DEBUG] tensor batch_size={tensor_batch_size}")
        non_tensor_sizes = {}
        min_non_tensor_size = float('inf')
        
        if batch.non_tensor_batch:
            for key, val in batch.non_tensor_batch.items():
                try:
                    size = len(val)
                    non_tensor_sizes[key] = size
                    min_non_tensor_size = min(min_non_tensor_size, size)
                except Exception as e:
                    print(f"[Balance Warning] Error getting size for non_tensor_batch[{key}]: {e}")
                    non_tensor_sizes[key] = "unknown"
            
            if debug_balance:
                print(f"[Balance DEBUG] non_tensor_batch sizes: {non_tensor_sizes}")
                print(f"[Balance DEBUG] min_non_tensor_size: {min_non_tensor_size}")
            
            # Check for size mismatches
            if tensor_batch_size != min_non_tensor_size:
                print(f"[Balance ERROR] Size mismatch! tensor_batch_size={tensor_batch_size}, \
                      min_non_tensor_size={min_non_tensor_size}")
                print(f"[Balance FIX] Skipping reorder due to size mismatch - avoiding IndexError")
                return
        else:
            if debug_balance:
                print(f"[Balance DEBUG] No non_tensor_batch data")
            return
        
        # Use the minimum size between tensor and non_tensor for safety
        effective_batch_size = min(tensor_batch_size, min_non_tensor_size) \
            if min_non_tensor_size != float('inf') else tensor_batch_size
        if debug_balance:
            print(f"[Balance DEBUG] Using effective_batch_size={effective_batch_size} for partitioning")
        
        # (train_batch_size,)
        global_seqlen_lst = batch.batch["attention_mask"].view(tensor_batch_size, -1).sum(-1).tolist()
        
        # Truncate seqlen_lst to effective_batch_size if needed
        if len(global_seqlen_lst) > effective_batch_size:
            print(f"[Balance FIX] Truncating seqlen_lst from {len(global_seqlen_lst)} to {effective_batch_size}")
            global_seqlen_lst = global_seqlen_lst[:effective_batch_size]
        
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        
        # Enhanced defensive check: ensure indices are within BOTH tensor and non_tensor bounds
        try:
            max_idx = int(global_idx.max().item()) if global_idx.numel() > 0 else -1
            min_idx = int(global_idx.min().item()) if global_idx.numel() > 0 else -1
        except Exception:
            print(f"[Balance Warning] failed to compute min/max of global_idx. global_idx={global_idx}")
            return
            
        # Check against effective_batch_size (the minimum safe size)
        if max_idx >= effective_batch_size or min_idx < 0:
            print(
                f"[Balance Warning] computed global_idx out of range: max_idx={max_idx}, min_idx={min_idx}, \
                    effective_batch_size={effective_batch_size}, world_size={world_size}"
            )
            print(f"[Balance Warning] tensor_batch_size={tensor_batch_size}, min_non_tensor_size={min_non_tensor_size}")
            if debug_balance:
                try:
                    sample_partitions = global_partition_lst[: min(5, len(global_partition_lst))]
                    print(f"[Balance Debug] sample_partitions={sample_partitions}")
                    print(f"[Balance Debug] global_idx[:10]={global_idx[:10].tolist() if global_idx.numel() > 0 else 'empty'}")
                except Exception:
                    pass
            # Skip reordering to avoid crashing; caller will continue without DP balancing
            print(f"[Balance FIX] Skipping reorder to prevent IndexError")
            return

        if debug_balance:
            print(f"[Balance DEBUG] Proceeding with reorder: max_idx={max_idx}, \
                  effective_batch_size={effective_batch_size}")
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO with pipeline parallelism.
        Implements rollout->reward->update pipeline where batch_i does reward API calls 
        while batch_{i+1} performs rollout concurrently.
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

        # perform validation before training using the same Reward Loop path as training
        if getattr(self, "val_dataloader", None) is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, 
                            initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Use prompt-level pipeline when reward is unified through Reward Loop and
        # the algorithm semantics are compatible with mini-batch rollout/reward overlap.
        pipeline_requested = self.config.trainer.get("enable_prompt_pipeline", True)
        reward_model_cfg = self._get_reward_model_cfg()
        rm_enabled = reward_model_cfg.enable
        supports_rm_overlap = (not rm_enabled) or reward_model_cfg.enable_resource_pool
        pipeline_supported = supports_rm_overlap and self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX
        use_prompt_pipeline = pipeline_requested and pipeline_supported

        print(f"[Pipeline] Pipeline conditions: "
              f"enable_prompt_pipeline={pipeline_requested}, "
              f"rm_enabled={rm_enabled}, "
              f"supports_rm_overlap={supports_rm_overlap}, "
              f"pipeline_supported={pipeline_supported}, "
              f"adv_estimator={self.config.algorithm.adv_estimator}")
        if not supports_rm_overlap:
            print("[Pipeline] Disabling mini pipeline because reward model is colocated with rollout "
                  "and reward.reward_model.enable_resource_pool is not enabled.")
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            print("[Pipeline] Disabling mini pipeline for REMAX because reward_baselines "
                  "are only implemented in the sequential path.")
        
        if use_prompt_pipeline:
            print("[Pipeline] Using prompt level pipeline mode: rollout/reward vs update overlap")
            self._fit_with_mini_pipeline(logger, progress_bar, last_val_metrics)
        else:
            print("[Pipeline] Using sequential mode: no overlap between rollout and reward")
            self._fit_sequential(logger, progress_bar, last_val_metrics)

    def _fit_sequential(self, logger, progress_bar, last_val_metrics):
        """Original sequential training loop."""
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                debug_verbose = hasattr(self.config, 'debug_level') and self.config.debug_level > 1
                if debug_verbose:
                    print(f"[DEBUG] Original batch: batch.batch.keys()={list(batch.batch.keys())[:10]}")
                    print(f"[DEBUG] Original batch: non_tensor_batch.keys()={list(batch.non_tensor_batch.keys())[:5]}")
                    try:
                        original_tensor_size = batch.batch.batch_size[0] if \
                            hasattr(batch.batch, 'batch_size') else len(batch.batch.get('input_ids', []))
                        print(f"[DEBUG] Original tensor batch_size: {original_tensor_size}")
                    except Exception as e:
                        print(f"[DEBUG] Error getting original tensor size: {e}")
                        
                    try:
                        original_non_tensor_size = len(
                            list(batch.non_tensor_batch.values())[0]
                        ) if batch.non_tensor_batch else 0
                        print(f"[DEBUG] Original non_tensor batch_size: {original_non_tensor_size}")
                    except Exception as e:
                        print(f"[DEBUG] Error getting original non_tensor size: {e}")
                        
                    print(f"[DEBUG] GFPO_G (rollout.n): {self.config.actor_rollout_ref.rollout.n}")

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                # Keep raw_prompt in batch for reward model, only remove from gen_batch
                raw_prompt_for_rm = None
                if "raw_prompt" in batch.non_tensor_batch:
                    raw_prompt_for_rm = batch.non_tensor_batch["raw_prompt"].copy()
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                elif "full_prompts" in batch.non_tensor_batch:
                    # Some datasets provide full text under 'full_prompts' when raw_prompt not present
                    raw_prompt_for_rm = batch.non_tensor_batch["full_prompts"].copy()
                    non_tensor_batch_keys_to_pop.append("full_prompts")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                    
                if debug_verbose:
                    print(f"[DEBUG] About to pop batch_keys: {batch_keys_to_pop}")
                    print(f"[DEBUG] About to pop non_tensor_batch_keys: {non_tensor_batch_keys_to_pop}")
                
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                if debug_verbose:
                    print(f"[DEBUG] After pop - remaining batch.batch.keys()={list(batch.batch.keys())[:10]}")
                    print(f"[DEBUG] After pop - \
                          remaining batch.non_tensor_batch.keys()={list(batch.non_tensor_batch.keys())[:5]}")
                    print(f"[DEBUG] After pop - gen_batch.batch.keys()={list(gen_batch.batch.keys())[:10]}")
                
                try:
                    remaining_tensor_size = batch.batch.batch_size[0] \
                        if hasattr(batch.batch, 'batch_size') else 0
                    if debug_verbose:
                        print(f"[DEBUG] After pop - remaining tensor batch_size: {remaining_tensor_size}")
                except Exception as e:
                    if debug_verbose:
                        print(f"[DEBUG] After pop - error getting remaining tensor size: {e}")
                    
                try:
                    remaining_non_tensor_size = len(
                        list(batch.non_tensor_batch.values())[0]
                    ) if batch.non_tensor_batch else 0
                    if debug_verbose:
                        print(f"[DEBUG] After pop - remaining non_tensor batch_size: {remaining_non_tensor_size}")
                except Exception as e:
                    if debug_verbose:
                        print(f"[DEBUG] After pop - error getting remaining non_tensor size: {e}")
                    
                try:
                    gen_tensor_size = gen_batch.batch.batch_size[0] if \
                        hasattr(gen_batch.batch, 'batch_size') else 0
                    if debug_verbose:
                        print(f"[DEBUG] After pop - gen_batch tensor batch_size: {gen_tensor_size}")
                except Exception as e:
                    if debug_verbose:
                        print(f"[DEBUG] After pop - error getting gen_batch tensor size: {e}")

                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, 
                    interleave=True
                )
                
                try:
                    gen_repeated_size = gen_batch.batch.batch_size[0] if \
                        hasattr(gen_batch.batch, 'batch_size') else 0
                    if debug_verbose:
                        print(f"[DEBUG] After gen_batch.repeat({self.config.actor_rollout_ref.rollout.n}) \
                              - size: {gen_repeated_size}")
                except Exception as e:
                    if debug_verbose:
                        print(f"[DEBUG] Error getting gen_batch repeated size: {e}")

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        
                        # Debug generation output size (only if high debug level)
                        if hasattr(self.config, 'debug_level') and self.config.debug_level > 2:
                            try:
                                gen_output_size = gen_batch_output.batch.batch_size[0] \
                                    if hasattr(gen_batch_output.batch, 'batch_size') else 0
                                print(f"[DEBUG] Generation output batch_size: {gen_output_size}")
                            except Exception as e:
                                print(f"[DEBUG] Error getting generation output size: {e}")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            with self._reward_batch_context():
                                batch = self._compute_reward_batch(batch, raw_prompt_for_rm=raw_prompt_for_rm)
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()) + ["rm_scores"])
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    # NOTE: We need to repeat the non_tensor_batch to match the generation output size
                    # but the tensor batch is empty after pop, so we only repeat non_tensor_batch
                    debug_verbose = hasattr(self.config, 'debug_level') and self.config.debug_level > 2
                    if debug_verbose:
                        print(f"[DEBUG] Before repeat: batch.batch.keys()={list(batch.batch.keys())[:5]}")
                        try:
                            batch_size_before = getattr(batch.batch, 'batch_size', None)
                            print(f"[DEBUG] Before repeat: batch.batch.batch_size={batch_size_before}")
                        except Exception as e:
                            print(f"[DEBUG] Before repeat: error getting batch_size: {e}")
                        print(f"[DEBUG] Before repeat: non_tensor_batch keys={list(batch.non_tensor_batch.keys())[:3]}")
                        if batch.non_tensor_batch:
                            sample_key = list(batch.non_tensor_batch.keys())[0]
                            print(f"[DEBUG] Before repeat: \
                                  non_tensor_batch[{sample_key}].shape={getattr(
                                      batch.non_tensor_batch[sample_key], 'shape', 'no shape')}")
                    
                    # Check if batch.batch is empty before repeat
                    if len(batch.batch.keys()) == 0:
                        if debug_verbose:
                            print(f"[DEBUG FIX] batch.batch is empty after pop - \
                                  this is expected. Only repeating non_tensor_batch.")
                        # Since batch.batch is empty, we only need to repeat the non_tensor_batch manually
                        repeated_non_tensor_batch = {}
                        for key, val in batch.non_tensor_batch.items():
                            repeated_non_tensor_batch[key] = np.repeat(
                                val, 
                                self.config.actor_rollout_ref.rollout.n, 
                                axis=0
                            )
                        batch.non_tensor_batch = repeated_non_tensor_batch
                        
                        # Create a new empty TensorDict with the correct batch size for union compatibility
                        from tensordict import TensorDict
                        expected_batch_size = len(
                            list(repeated_non_tensor_batch.values())[0]
                        ) if repeated_non_tensor_batch else 0
                        batch.batch = TensorDict({}, batch_size=(expected_batch_size,))
                        
                        print(f"[DEBUG FIX] After manual repeat: \
                              non_tensor_batch keys={list(batch.non_tensor_batch.keys())[:3]}")
                        print(f"[DEBUG FIX] After manual repeat: \
                              batch.batch.batch_size={batch.batch.batch_size}")
                        if batch.non_tensor_batch:
                            sample_key = list(batch.non_tensor_batch.keys())[0]
                            print(f"[DEBUG FIX] After manual repeat: non_tensor_batch\
                                  [{sample_key}].shape=\
                                    {getattr(batch.non_tensor_batch[sample_key], 'shape', 'no shape')}")
                    else:
                        # Normal case: batch has tensor keys, can use standard repeat
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        print(f"[DEBUG] After repeat: batch.batch.keys()={list(batch.batch.keys())[:5]}")
                        try:
                            batch_size_after = getattr(batch.batch, 'batch_size', None)
                            print(f"[DEBUG] After repeat: batch.batch.batch_size={batch_size_after}")
                        except Exception as e:
                            print(f"[DEBUG] After repeat: error getting batch_size: {e}")
                        if batch.non_tensor_batch:
                            sample_key = list(batch.non_tensor_batch.keys())[0]
                            print(f"[DEBUG] After repeat: \
                                  non_tensor_batch[{sample_key}].shape=\
                                    {getattr(batch.non_tensor_batch[sample_key], 'shape', 'no shape')}")
                    
                    batch = batch.union(gen_batch_output)
                    print(f"[DEBUG] After union: batch.batch.keys()={list(batch.batch.keys())[:5]}")
                    try:
                        batch_size_union = getattr(batch.batch, 'batch_size', None)
                        print(f"[DEBUG] After union: batch.batch.batch_size={batch_size_union}")
                    except Exception as e:
                        print(f"[DEBUG] After union: error getting batch_size: {e}")
                    if batch.non_tensor_batch:
                        sample_key = list(batch.non_tensor_batch.keys())[0]
                        print(f"[DEBUG] After union: \
                              non_tensor_batch[{sample_key}].shape=\
                                {getattr(batch.non_tensor_batch[sample_key],'shape', 'no shape')}")
                    try:
                        attention_mask_size = batch.batch["attention_mask"].shape[0] if "attention_mask" \
                            in batch.batch else "no attention_mask"
                        print(f"[DEBUG] After union: attention_mask size={attention_mask_size}")
                    except Exception as e:
                        print(f"[DEBUG] Error getting attention_mask size: {e}")

                    # Restore raw_prompt for reward model if it was saved
                    if raw_prompt_for_rm is not None:
                        # Check if we need to repeat raw_prompt to match the repeated batch
                        expected_size = len(
                            list(batch.non_tensor_batch.values())[0]
                        ) if batch.non_tensor_batch else 0
                        current_raw_prompt_size = len(raw_prompt_for_rm)
                        
                        print(f"[DEBUG] raw_prompt restore: \
                              expected_size={expected_size}, current_size={current_raw_prompt_size}")
                        print(f"[DEBUG] raw_prompt_for_rm type: {type(raw_prompt_for_rm)}, \
                              shape: {getattr(raw_prompt_for_rm, 'shape', 'no shape')}")
                        
                        if expected_size > current_raw_prompt_size:
                            # Need to repeat raw_prompt to match the batch size
                            repeat_factor = expected_size // current_raw_prompt_size
                            print(f"[DEBUG] Repeating raw_prompt by factor {repeat_factor}")
                            
                            # Debug: test the repeat operation 
                            # (consistent with DataProto.repeat interleave=True logic)
                            print(f"[DEBUG] Before repeat: len={len(raw_prompt_for_rm)}")
                            repeated_raw_prompt = np.repeat(raw_prompt_for_rm, repeat_factor, axis=0)
                            print(f"[DEBUG] After repeat: \
                                  len={len(repeated_raw_prompt)}, type={type(repeated_raw_prompt)}")
                            
                            batch.non_tensor_batch["raw_prompt"] = repeated_raw_prompt
                            print(f"[DEBUG] After assignment: \
                                  raw_prompt size = {len(batch.non_tensor_batch['raw_prompt'])}")
                        else:
                            # Direct assignment for same size or downsampling case
                            batch.non_tensor_batch["raw_prompt"] = raw_prompt_for_rm
                            print(f"[DEBUG] Direct assignment: \
                                  raw_prompt size = {len(batch.non_tensor_batch['raw_prompt'])}")
                    
                    # Debug: Check final consistency before proceeding
                    if batch.non_tensor_batch and len(batch.batch.keys()) > 0:
                        tensor_size = batch.batch.batch_size[0] \
                            if hasattr(batch.batch, 'batch_size') else "unknown"
                        non_tensor_sample = list(batch.non_tensor_batch.values())[0]
                        non_tensor_size = len(non_tensor_sample) \
                            if hasattr(non_tensor_sample, '__len__') else "unknown"
                        print(f"[DEBUG] Final consistency \
                              check: tensor_size={tensor_size}, non_tensor_size={non_tensor_size}")
                        
                        # Check raw_prompt specifically
                        if "raw_prompt" in batch.non_tensor_batch:
                            raw_prompt_size = len(batch.non_tensor_batch["raw_prompt"])
                            print(f"[DEBUG] raw_prompt final size: {raw_prompt_size}")
                            if raw_prompt_size != tensor_size:
                                print(f"[DEBUG ERROR] \
                                      raw_prompt size mismatch: {raw_prompt_size} vs tensor_size {tensor_size}")
                                # Force fix the raw_prompt size
                                if raw_prompt_size < tensor_size:
                                    print(f"[DEBUG FORCE FIX] \
                                          Extending raw_prompt from {raw_prompt_size} to {tensor_size}")
                                    extend_factor = tensor_size // raw_prompt_size
                                    batch.non_tensor_batch["raw_prompt"] = np.repeat(
                                        batch.non_tensor_batch["raw_prompt"], 
                                        extend_factor, axis=0
                                    )
                                    print(f"[DEBUG FORCE FIX] \
                                          After extension: {len(batch.non_tensor_batch['raw_prompt'])}")
                        
                        if tensor_size != "unknown" and non_tensor_size != "unknown" \
                            and tensor_size != non_tensor_size:
                            print(f"[DEBUG ERROR] \
                                  Size mismatch detected before RM call: {tensor_size} vs {non_tensor_size}")
                            # Try to fix by trimming non_tensor_batch to match tensor size
                            if non_tensor_size > tensor_size:
                                print(f"[DEBUG FIX] \
                                      Trimming non_tensor_batch from {non_tensor_size} to {tensor_size}")
                                for key, val in batch.non_tensor_batch.items():
                                    if hasattr(val, '__len__') and len(val) > tensor_size:
                                        batch.non_tensor_batch[key] = val[:tensor_size]

                    # batch.non_tensor_batch["uid"] = np.array(
                    #     [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    # )
                    # prefer stable id from dataset if present (GFPO for calculating adaptive k)
                    if "raw_prompt_ids" in batch.non_tensor_batch:
                        # Ensure raw_prompt_ids are converted to hashable strings
                        raw_ids = batch.non_tensor_batch["raw_prompt_ids"]
                        uid_list = []
                        for raw_id in raw_ids:
                            if isinstance(raw_id, (list, tuple, np.ndarray)):
                                # Convert non-hashable types to strings
                                uid_list.append(str(raw_id))
                            else:
                                # Keep hashable types as strings
                                uid_list.append(str(raw_id))
                        batch.non_tensor_batch["uid"] = np.array(uid_list, dtype=object)
                    elif "data_idx" in batch.non_tensor_batch:
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(idx) for idx in batch.non_tensor_batch["data_idx"]], 
                            dtype=object
                        )
                    elif "raw_prompt" in batch.non_tensor_batch:
                        # hashed prompt text as stable id
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(hash(s)) for s in batch.non_tensor_batch["raw_prompt"]], dtype=object
                        )
                    else:
                        # fallback to old UUID when no stable id exists
                        print("Warning: using unstable uid for this dataset, which might affect \
                              advantage calculation if prompts are repeated!")
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        # Debug: print batch sizes before balance to track the issue
                        try:
                            batch_size_before_balance = len(batch.batch.get('attention_mask', []))
                            print(f"[DEBUG] Before balance: batch_size={batch_size_before_balance}, "
                                  f"non_tensor keys={list(batch.non_tensor_batch.keys())[:5]}")
                        except Exception as e:
                            print(f"[DEBUG] Error getting batch info before balance: {e}")
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # Compute reward through Reward Loop so sequential and
                        # mini-pipeline paths share the same reward semantics.
                        with self._reward_batch_context():
                            batch = self._compute_reward_batch(batch, raw_prompt_for_rm=raw_prompt_for_rm)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Continue with rest of training loop...
                    # [The rest continues with old_log_prob, ref, values, adv, updates, etc.]
                    # This is just the sequential version, keeping the original behavior

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if self._get_reward_manager_name() == 'gfpo_rm':
                            # GFPO 难度自适应采样
                            # 注入动态 k 值
                            try:
                                sampler = getattr(self.train_dataloader, "sampler", None)
                                underlying = sampler
                                # unwrap common wrapper attributes
                                while hasattr(underlying, "sampler"):
                                    underlying = getattr(underlying, "sampler")
                                print(f"[GFPO] dataloader.sampler={type(sampler)}, underlying={type(underlying)}, \
                                    has_method={hasattr(underlying, 'get_adaptive_k_for_batch')}")
                            except Exception as e:
                                print(f"[GFPO] failed to inspect/unwrap sampler: {e}")
                                underlying = None

                            if underlying is not None and hasattr(underlying, "get_adaptive_k_for_batch"):
                                try:
                                    adaptive_k = underlying.get_adaptive_k_for_batch(batch)
                                    # normalize return to numpy array
                                    if isinstance(adaptive_k, torch.Tensor):
                                        adaptive_k = adaptive_k.cpu().numpy()
                                    elif isinstance(adaptive_k, list):
                                        adaptive_k = np.array(adaptive_k)
                                    # small summary for debug
                                    sample_view = None
                                    try:
                                        sample_view = adaptive_k[: min(5, len(adaptive_k))]
                                    except Exception:
                                        sample_view = str(type(adaptive_k))
                                    print(f"[GFPO] trainer computed adaptive_k type={type(adaptive_k)}, \
                                        shape={getattr(adaptive_k, 'shape', None)}, sample={sample_view}")
                                except Exception as e:
                                    print(f"[GFPO] get_adaptive_k_for_batch failed: {e}")
                                    adaptive_k = None
                                batch.non_tensor_batch["adaptive_k"] = adaptive_k
                            else:
                                print(f"[GFPO] underlying sampler does not provide get_adaptive_k_for_batch. \
                                    underlying={type(underlying)}")

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        getattr(self, "val_dataloader", None) is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

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
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _fit_with_mini_pipeline(self, logger, progress_bar, last_val_metrics):
        """
        Simplified prompt-level pipeline implementation.
        Uses existing pipeline functions but removes one-step-off complexity.
        Processes each batch with concurrent prompts but updates immediately.
        """
        
        print("[Mini-batch Pipeline] Starting simplified prompt-level pipeline mode")
        
        # Get pipeline configuration
        max_concurrent_prompts = self.config.trainer.get("max_concurrent_prompts", 8)
        
        print(f"[Mini-batch Pipeline] Config: max_concurrent_prompts={max_concurrent_prompts}")
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps > self.total_training_steps:
                    break
                    
                print(f"[Mini-batch Pipeline] Step {self.global_steps}: Processing batch with prompt pipeline")
                
                # Process current batch with simplified pipeline (no one-step-off)
                self._process_batch_with_simplified_pipeline(
                    batch_dict, self.global_steps, epoch, logger, progress_bar, 
                    last_val_metrics, max_concurrent_prompts
                )
                
                self.global_steps += 1
                progress_bar.update(1)
                
                # Check termination condition
                if self.global_steps > self.total_training_steps:
                    print(f"[Mini-batch Pipeline] \
                          Reached total training steps ({self.total_training_steps}), finishing...")
                    break
            
            # Break outer epoch loop if we've reached the limit
            if self.global_steps > self.total_training_steps:
                break
        
        progress_bar.close()
        print("[Mini-batch Pipeline] Simplified prompt-level pipeline training completed")
        
    def _process_batch_with_simplified_pipeline(self, batch_dict, global_steps, epoch, logger, progress_bar, 
                                               last_val_metrics, max_concurrent_prompts):
        """
        Process a batch with simplified pipeline.
        """
        
        # Initialize timing and profiling
        do_profile = (
            global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        if do_profile:
            self.actor_rollout_wg.start_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()

        metrics = {}
        timing_raw = {}
        
        with marked_timer("step", timing_raw):
            # Process current batch with pipeline functions
            print(f"[Mini-batch Pipeline] Processing batch rollout/reward (step {global_steps})")
            training_batch = self._process_current_batch_rollout_reward(
                batch_dict, global_steps, max_concurrent_prompts, metrics, timing_raw
            )
            
            # Update model immediately (one-step-off 目前搞不定) 
            if training_batch is not None:
                print(f"[Mini-batch Pipeline] \
                      Updating model immediately with batch size: {len(training_batch.batch)}")
                
                # Update critic if used
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(training_batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # Update actor (with critic warmup check)
                if self.config.trainer.critic_warmup <= global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self.actor_update_wg.update_actor(training_batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
            
            # Handle validation and checkpointing
            is_last_step = global_steps >= self.total_training_steps
            if (getattr(self, "val_dataloader", None) is not None and self.config.trainer.test_freq > 0 and 
                (is_last_step or global_steps % self.config.trainer.test_freq == 0)):
                with marked_timer('testing', timing_raw):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if (self.config.trainer.save_freq > 0 and 
                (is_last_step or global_steps % self.config.trainer.save_freq == 0)):
                with marked_timer('save_checkpoint', timing_raw):
                    self._save_checkpoint()
        
        # Update sampler with training batch (GFPO adaptive k calculation)
        if training_batch is not None and isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=training_batch)
        
        # Log metrics for current step
        if training_batch is not None:
            metrics.update(compute_data_metrics(batch=training_batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=training_batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(
                compute_throughout_metrics(batch=training_batch, timing_raw=timing_raw, n_gpus=n_gpus))
        
        metrics.update({
            "training/global_step": global_steps,
            "training/epoch": epoch,
        })
        
        # Reduce metrics lists to scalar values before logging
        # This handles metrics accumulated from multiple mini-batches
        list_metrics = {k: v for k, v in metrics.items() if isinstance(v, list)}
        if list_metrics:
            reduced_metrics = reduce_metrics(list_metrics)
            metrics.update(reduced_metrics)
        
        logger.log(data=metrics, step=global_steps)
        
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
        
    def _process_current_batch_rollout_reward(self, batch_dict, global_steps, max_concurrent_prompts, 
                                              metrics, timing_raw):
        """Process current batch: rollout + reward computation in pipeline."""
        
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        print(f"[Mini-Batch Pipeline] Processing rollout/reward for batch size: {len(batch.batch)}")
        
        # Add UIDs for stable tracking
        if "uid" not in batch.non_tensor_batch:
            if "raw_prompt_ids" in batch.non_tensor_batch:
                # Ensure raw_prompt_ids are converted to hashable strings
                raw_ids = batch.non_tensor_batch["raw_prompt_ids"]
                uid_list = []
                for raw_id in raw_ids:
                    if isinstance(raw_id, (list, tuple, np.ndarray)):
                        # Convert non-hashable types to strings
                        uid_list.append(str(raw_id))
                    else:
                        # Keep hashable types as strings
                        uid_list.append(str(raw_id))
                batch.non_tensor_batch["uid"] = np.array(uid_list, dtype=object)
            elif "data_idx" in batch.non_tensor_batch:
                batch.non_tensor_batch["uid"] = np.array(
                    [str(idx) for idx in batch.non_tensor_batch["data_idx"]], dtype=object)
            elif "raw_prompt" in batch.non_tensor_batch:
                import hashlib
                batch.non_tensor_batch["uid"] = np.array([
                    hashlib.md5(prompt.encode()).hexdigest()[:8] 
                    for prompt in batch.non_tensor_batch["raw_prompt"]
                ], dtype=object)
            else:
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        
        # Phase 1: Split into mini-batches instead of individual prompts
        mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        mini_batches = self._split_batch_into_mini_batches(batch, mini_batch_size)
        print(f"[Mini-Batch Pipeline] Split into {len(mini_batches)} mini-batches of size ~{mini_batch_size}")
        
        # Phase 2: Overlap rollout with reward computation only.
        max_concurrent_mini_batches = min(max_concurrent_prompts // mini_batch_size + 1, len(mini_batches))
        processed_mini_batches = self._run_rollout_reward_pipeline(
            mini_batches, max_concurrent_mini_batches, timing_raw
        )
        
        # Phase 3: Recombine first, then prepare once on the full batch so
        # group/full-batch statistics are preserved.
        if processed_mini_batches:
            training_batch = DataProto.concat(processed_mini_batches)
            with marked_timer("prep", timing_raw):
                training_batch = self._prepare_batch_for_update(training_batch, metrics)
            print(f"[Mini-Batch Pipeline] \
                  Recombined into training batch of size: {len(training_batch.batch)}")
            if self.config.trainer.balance_batch:
                # Balance and prepare the batch
                self._balance_batch(training_batch, metrics=metrics, logging_prefix="rollout_reward_seqlen")
            training_batch.meta_info['global_token_num'] = torch.sum(
                training_batch.batch['attention_mask'], dim=-1
            ).tolist()
            
            return training_batch
        else:
            return None
    
    def _run_rollout_reward_pipeline(self, mini_batches, max_concurrent_mini_batches, timing_raw):
        """
        Run rollout and reward computation pipeline with mini-batch granularity.

        This stage intentionally stops after reward computation so actor/ref/critic
        pre-update work can run outside the overlap window.
        """
        
        import queue
        import threading
        import time
        import concurrent.futures
        
        print(f"[Mini-Batch Pipeline] Processing {len(mini_batches)} \
              mini-batches with max_concurrent={max_concurrent_mini_batches}")
        
        # Configuration for mini-batch processing
        max_reward_concurrent = self.config.trainer.get("max_reward_concurrent", 16)  # For external APIs
        
        # Queues and tracking
        generation_queue = queue.Queue()
        pending_rewards = {}  # mini_batch_id -> {'future': future, 'generated': data, 'start_time': time}
        pending_rewards_lock = threading.RLock()  # Thread-safe lock for pending_rewards dictionary
        completed_mini_batches = []
        generation_completed = threading.Event()
        
        # Submit mini-batches to generation queue
        for i, mini_batch in enumerate(mini_batches):
            generation_queue.put((i, mini_batch))
        
        # Executor for reward API calls - separate from generation threads
        reward_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_reward_concurrent)
        
        def generation_worker():
            """
            Worker thread for mini-batch generation -
              processes entire mini-batches for better GPU utilization.
            """
            while True:
                try:
                    item = generation_queue.get(timeout=1.0)
                    if item is None:  # Poison pill
                        break
                    
                    mini_batch_id, mini_batch = item
                    print(f"[Mini-Batch Pipeline] \
                          Processing mini-batch {mini_batch_id} with size {len(mini_batch.batch)}")
                    
                    # Generate response for entire mini-batch (much more efficient)
                    with marked_timer(f"gen_mb_{mini_batch_id}", timing_raw, color="red"):
                        generated_mini_batch = self._generate_mini_batch(mini_batch)
                    
                    # Immediately submit reward computation (non-blocking, high concurrency)
                    reward_future = reward_executor.submit(
                        self._compute_reward_for_mini_batch,
                        mini_batch_id,
                        generated_mini_batch,
                        timing_raw,
                    )
                    
                    # Thread-safe write to pending_rewards
                    with pending_rewards_lock:
                        pending_rewards[mini_batch_id] = {
                            'future': reward_future,
                            'start_time': time.time()
                        }
                        pending_count = len(pending_rewards)
                    
                    print(f"[Mini-Batch Pipeline] Mini-batch {mini_batch_id}: \
                          Generated and submitted for reward (pending: {pending_count})")
                    generation_queue.task_done()
                    
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[Mini-Batch Pipeline] Generation error for mini-batch {mini_batch_id}: {e}")
                    import traceback
                    print(f"[Mini-Batch Pipeline] Full traceback:\n{traceback.format_exc()}")
                    generation_queue.task_done()
        
        def reward_collection_worker():
            """Worker thread to collect completed rewards - runs continuously."""
            while True:
                # Thread-safe check for empty dictionary
                with pending_rewards_lock:
                    is_empty = len(pending_rewards) == 0
                    if is_empty and generation_completed.is_set():
                        break
                    
                if is_empty:
                    time.sleep(0.05)
                    continue
                
                # 100% safe: Create snapshot inside lock
                # Lock protects the entire dictionary, other threads cannot write during snapshot creation
                with pending_rewards_lock:
                    # In this protected scope, we can safely traverse and create snapshot
                    safe_snapshot = list(pending_rewards.items())
                
                # Outside lock scope, snapshot is a safe static list, can be used freely
                completed_ids = []
                for mini_batch_id, reward_info in safe_snapshot:
                    if reward_info['future'].done():
                        try:
                            reward_ready_mini_batch = reward_info['future'].result()
                            completed_mini_batches.append((mini_batch_id, reward_ready_mini_batch))
                            
                            elapsed = time.time() - reward_info['start_time']
                            print(f"[Mini-Batch Pipeline] Mini-batch {mini_batch_id}: Reward completed in {elapsed:.2f}s")
                            completed_ids.append(mini_batch_id)
                            
                        except Exception as e:
                            print(f"[Mini-Batch Pipeline] Reward completion error for mini-batch {mini_batch_id}: {e}")
                            completed_ids.append(mini_batch_id)
                
                # Thread-safe removal of completed rewards
                if completed_ids:
                    with pending_rewards_lock:
                        for mini_batch_id in completed_ids:
                            # Use pop with default to avoid KeyError if already deleted
                            pending_rewards.pop(mini_batch_id, None)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.02)
        
        with self._reward_batch_context():
            # Start workers only after reward resources are ready, otherwise a
            # fast rollout path can submit reward work before RM wake_up finishes.
            generation_threads = []
            for i in range(min(max_concurrent_mini_batches, len(mini_batches))):
                gen_thread = threading.Thread(target=generation_worker, name=f"gen_worker_{i}")
                gen_thread.start()
                generation_threads.append(gen_thread)

            # Start reward collection worker
            collection_thread = threading.Thread(target=reward_collection_worker, name="reward_collector")
            collection_thread.start()
            
            print(f"[Mini-Batch Pipeline] Started {len(generation_threads)} \
                  generation workers, max {max_reward_concurrent} reward concurrent")
            
            # Wait for generation to complete
            generation_queue.join() # Wait until all mini-batches are processed
            generation_completed.set() # Signal that generation is complete

            # Stop generation workers
            for _ in range(len(generation_threads)):
                generation_queue.put(None)
            
            for thread in generation_threads:
                thread.join()
            
            # Thread-safe check for pending rewards count
            with pending_rewards_lock:
                pending_count = len(pending_rewards)
            print(f"[Mini-Batch Pipeline] Generation completed. \
                  Waiting for {pending_count} pending rewards...")
            
            # Wait for all rewards to complete
            collection_thread.join()
        
        # Cleanup
        reward_executor.shutdown(wait=True)
        
        # Sort completed mini-batches by original order and return
        completed_mini_batches.sort(key=lambda x: x[0])
        print(f"[Mini-Batch Pipeline] All mini-batches completed. \
              Generated: {len(mini_batches)}, Processed: {len(completed_mini_batches)}")
        return [mini_batch for _, mini_batch in completed_mini_batches]
    
    def _generate_mini_batch(self, mini_batch):
        """Generate response for an entire mini-batch - much more efficient than single prompts."""
        debug_gen = hasattr(self.config, 'debug_level') and self.config.debug_level > 2
        
        # Prepare for generation (similar to single prompt but for entire batch)
        gen_mini_batch = self._prepare_batch_for_generation(mini_batch)
        
        # Generate using existing batch generation logic
        gen_output = self.actor_rollout_wg.generate_sequences(gen_mini_batch)
        
        # Restore original batch data using the same logic as original ray_trainer.py
        original_batch_info = gen_mini_batch.meta_info['original_batch']
        raw_prompt_for_rm = gen_mini_batch.meta_info['raw_prompt_for_rm']
        
        # Reconstruct original_mini_batch from stored info (avoiding empty DataProto serialization)
        from tensordict import TensorDict
        
        if original_batch_info['has_empty_batch']:
            # Batch was empty after pop, reconstruct appropriately
            # Get the correct batch size from non_tensor_batch
            non_tensor_batch = original_batch_info['non_tensor_batch']
            if non_tensor_batch:
                correct_batch_size = len(list(non_tensor_batch.values())[0])
            else:
                correct_batch_size = 0
                
            original_mini_batch = DataProto(
                batch=TensorDict({}, batch_size=(correct_batch_size,)),
                non_tensor_batch=non_tensor_batch,
                meta_info=original_batch_info['meta_info']
            )
        else:
            # This shouldn't happen in current pipeline logic, but handle it gracefully
            raise RuntimeError("Unexpected: original_batch was not empty after pop in mini-pipeline mode")
        
        # Clean timing info from gen_output
        gen_output.meta_info.pop("timing", None)
        
        # Follow original logic: check if original_mini_batch.batch is empty after pop
        repeat_times = self.config.actor_rollout_ref.rollout.n
        
        if len(original_mini_batch.batch.keys()) == 0:
            # Since batch.batch is empty, we only need to repeat the non_tensor_batch manually
            repeated_non_tensor_batch = {}
            for key, val in original_mini_batch.non_tensor_batch.items():
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            original_mini_batch.non_tensor_batch = repeated_non_tensor_batch
            
            # Create a new empty TensorDict with the correct batch size for union compatibility
            from tensordict import TensorDict
            expected_batch_size = len(list(repeated_non_tensor_batch.values())[0]) if repeated_non_tensor_batch else 0
            original_mini_batch.batch = TensorDict({}, batch_size=(expected_batch_size,))
            
            if debug_gen:
                print(f"[DEBUG] After manual repeat: batch.batch.batch_size={original_mini_batch.batch.batch_size}")
        else:
            # Normal case: batch has tensor keys, can use standard repeat
            original_mini_batch = original_mini_batch.repeat(repeat_times=repeat_times, interleave=True)
            if debug_gen:
                print(f"[DEBUG] After repeat: batch.batch.batch_size={original_mini_batch.batch.batch_size}")
        
        # Now union should work
        full_mini_batch = original_mini_batch.union(gen_output)
        
        # Restore raw_prompt for reward model if it was saved - CRITICAL for GFPO RM
        if raw_prompt_for_rm is not None:
            # Check if we need to repeat raw_prompt to match the repeated batch
            expected_size = len(list(full_mini_batch.non_tensor_batch.values())[0]) if full_mini_batch.non_tensor_batch else 0
            current_raw_prompt_size = len(raw_prompt_for_rm)
            
            if debug_gen:
                print(f"[DEBUG] raw_prompt restore: expected_size={expected_size}, current_size={current_raw_prompt_size}")
            
            if expected_size > current_raw_prompt_size:
                # Need to repeat raw_prompt to match the batch size
                repeat_factor = expected_size // current_raw_prompt_size
                if debug_gen:
                    print(f"[DEBUG] Repeating raw_prompt by factor {repeat_factor}")
                
                repeated_raw_prompt = np.repeat(raw_prompt_for_rm, repeat_factor, axis=0)
                full_mini_batch.non_tensor_batch["raw_prompt"] = repeated_raw_prompt
            else:
                # Direct assignment for same size or downsampling case
                full_mini_batch.non_tensor_batch["raw_prompt"] = raw_prompt_for_rm
            
            if debug_gen:
                print(f"[DEBUG] After raw_prompt restore: size = {len(full_mini_batch.non_tensor_batch['raw_prompt'])}")
        
        return full_mini_batch

    def _prepare_batch_for_generation(self, batch):
        """Prepare an entire batch for generation."""
        debug_prep = hasattr(self.config, 'debug_level') and self.config.debug_level > 2
        if debug_prep:
            print(f"[DEBUG] Preparing batch: batch_size={len(batch.batch)}")
        
        # Same logic as original ray_trainer.py
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        
        # Save raw_prompt for reward model - CRITICAL for GFPO RM
        raw_prompt_for_rm = None
        if "raw_prompt" in batch.non_tensor_batch:
            raw_prompt_for_rm = batch.non_tensor_batch["raw_prompt"].copy()
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        elif "full_prompts" in batch.non_tensor_batch:
            raw_prompt_for_rm = batch.non_tensor_batch["full_prompts"].copy()
            non_tensor_batch_keys_to_pop.append("full_prompts")
        
        # Note: DO NOT pop data_source - it's needed by GFPO reward manager
        # Only pop fields that are specifically for generation
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")
        
        # Pop for generation
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        
        # Validate
        required_keys = ["input_ids", "attention_mask", "position_ids"]
        missing_keys = [key for key in required_keys if key not in gen_batch.batch]
        if missing_keys:
            raise ValueError(f"Missing required keys for VLLM generation: {missing_keys}")
        
        # Repeat for multiple generations per prompt
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # Store for later restoration - avoid storing empty DataProto to prevent serialization issues
        # Only store the non_tensor_batch and meta_info to avoid Ray serializing empty TensorDict
        # gen_batch.meta_info['original_batch'] = batch
        gen_batch.meta_info['original_batch'] = {
            'non_tensor_batch': batch.non_tensor_batch,
            'meta_info': batch.meta_info,
            'has_empty_batch': len(batch.batch.keys()) == 0  # Flag for restoration logic
        }
        gen_batch.meta_info['raw_prompt_for_rm'] = raw_prompt_for_rm
        
        return gen_batch

    def _compute_reward_for_mini_batch(self, mini_batch_id, generated_mini_batch, timing_raw):
        """Compute reward for an entire mini-batch without pre-update work."""
        try:
            if hasattr(self.config, 'debug_level') and self.config.debug_level > 1:
                print(f"[DEBUG] Computing reward for mini-batch {mini_batch_id}: \
                      size={len(generated_mini_batch.batch)}")
            
            # Compute reward for entire mini-batch through Reward Loop.
            with marked_timer(f"reward_mb_{mini_batch_id}", timing_raw):
                generated_mini_batch = self._compute_reward_batch(generated_mini_batch)
                reward_tensor, _ = extract_reward(generated_mini_batch)
                generated_mini_batch.batch["token_level_scores"] = reward_tensor
            
            return generated_mini_batch
            
        except Exception as e:
            print(f"[Mini-Batch Pipeline] Reward computation failed for mini-batch {mini_batch_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback with zero rewards
            batch_size = len(generated_mini_batch.batch)
            response_len = generated_mini_batch.batch["responses"].shape[-1]
            generated_mini_batch.batch["token_level_scores"] = torch.zeros(batch_size, response_len)
            return generated_mini_batch

    def _prepare_batch_for_update(self, batch, metrics):
        """Prepare an entire batch for model update - batch version of single prompt preparation."""
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        # Compute old log probs if not using rollout log probs
        if not self.config.actor_rollout_ref.rollout.get("enable_log_prob", False):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            for k, v in old_log_prob_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)
            
            if "rollout_log_probs" in batch.batch.keys():
                batch.batch.pop("rollout_log_probs")
        elif "old_log_probs" not in batch.batch.keys() and "rollout_log_probs" in batch.batch.keys():
            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        
        # Compute reference log probs if needed
        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)
        
        # Compute values if using critic
        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)

        # Apply KL penalty only after old/ref log-probs are ready.
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            for k, v in kl_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        
        # Compute advantages
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        if self._get_reward_manager_name() == 'gfpo_rm':
            # GFPO 难度自适应采样
            # 注入动态 k 值
            try:
                sampler = getattr(self.train_dataloader, "sampler", None)
                underlying = sampler
                # unwrap common wrapper attributes
                while hasattr(underlying, "sampler"):
                    underlying = getattr(underlying, "sampler")
                print(f"[GFPO] _prepare_batch_for_update: \
                    dataloader.sampler={type(sampler)}, underlying={type(underlying)}, \
                        has_method={hasattr(underlying, 'get_adaptive_k_for_batch')}")
            except Exception as e:
                print(f"[GFPO] failed to inspect/unwrap sampler: {e}")
                underlying = None

            if underlying is not None and hasattr(underlying, "get_adaptive_k_for_batch"):
                try:
                    adaptive_k = underlying.get_adaptive_k_for_batch(batch)
                    # normalize return to numpy array
                    if isinstance(adaptive_k, torch.Tensor):
                        adaptive_k = adaptive_k.cpu().numpy()
                    elif isinstance(adaptive_k, list):
                        adaptive_k = np.array(adaptive_k)
                    # small summary for debug
                    sample_view = None
                    try:
                        sample_view = adaptive_k[: min(5, len(adaptive_k))]
                    except Exception:
                        sample_view = str(type(adaptive_k))
                    print(f"[GFPO] _prepare_batch_for_update: \
                        computed adaptive_k type={type(adaptive_k)}, \
                            shape={getattr(adaptive_k, 'shape', None)}, sample={sample_view}")
                except Exception as e:
                    print(f"[GFPO] get_adaptive_k_for_batch failed: {e}")
                    adaptive_k = None
                batch.non_tensor_batch["adaptive_k"] = adaptive_k
            else:
                print(f"[GFPO] underlying sampler does not provide get_adaptive_k_for_batch. \
                    underlying={type(underlying)}")
        
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )
        
        return batch
        
    def _split_batch_into_mini_batches(self, batch, mini_batch_size):
        """Split a DataProto batch into mini-batches for better GPU utilization."""
        mini_batches = []
        batch_size = len(batch.batch)
        
        print(f"[DEBUG] Splitting batch into mini-batches: \
              total_size={batch_size}, mini_batch_size={mini_batch_size}")
        
        # Calculate number of mini-batches
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
        
        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, batch_size)
            indices = list(range(start_idx, end_idx))
            
            try:
                # Create mini-batch using list indexing
                mini_batch = batch[indices]  # This uses __getitem__ with list, returns DataProto
                
                # Debug: check if mini-batch is valid
                if len(mini_batch.batch) == 0:
                    print(f"[DEBUG] Warning: Mini-batch {i} is empty, skipping")
                    continue
                
                print(f"[DEBUG] Mini-batch {i}: size={len(mini_batch.batch)}, indices={start_idx}:{end_idx}")
                mini_batches.append(mini_batch)
                
            except Exception as e:
                print(f"[DEBUG] Error creating mini-batch {i}: {e}")
                raise
        
        return mini_batches

    