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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import math
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import tensordict
import torch
from omegaconf import OmegaConf, open_dict
from packaging.version import parse as parse_version
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transfer_queue import (
    BatchMeta,
    SimpleStorageUnit,
    TransferQueueController,
    get_placement_group,
    process_zmq_server_info,
)

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
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
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.transferqueue_utils import create_transferqueue_client, get_transferqueue_client, tqbridge


@tqbridge(put_data=False)
def compute_reward_decorated(data, reward_fn):
    return compute_reward(data, reward_fn)


@tqbridge(put_data=False)
def compute_reward_async_decorated(data, reward_fn):
    return compute_reward_async.remote(data, reward_fn)


@tqbridge(put_data=False)
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

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

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return token_level_rewards, metrics


def compute_response_mask(batch_meta: BatchMeta, tq_client):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        batch_meta (BatchMeta): The data containing batched model outputs and inputs.

    Returns:
        BatchMeta: The BatchMeta of attention mask for the response tokens.
    """
    data = tq_client.get_data(batch_meta)

    responses = data["responses"]
    response_length = responses.size(1)
    attention_mask = data["attention_mask"]
    response_mask = attention_mask[:, -response_length:]
    output = TensorDict({"response_mask": response_mask}, batch_size=response_mask.size(0))

    batch_meta = tq_client.put(data=output, metadata=batch_meta)

    return batch_meta


@tqbridge(put_data=False)
def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[Any, Any]:
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
        tuple: A tuple containing:
            - advantages: The computed advantage estimates.
            - returns: The computed returns.
    """
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
        # TODO (TQ): adapt core_algos.compute_pf_ppo_reweight_data function to support transfer queue
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
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
    return advantages, returns


@tqbridge(put_data=False)
def compute_data_metrics_decorated(batch, use_critic: bool = True):
    return compute_data_metrics(batch, use_critic)


@tqbridge(put_data=False)
def compute_timing_metrics_decorated(batch, timing_raw: dict[str, float]) -> dict[str, Any]:
    return compute_timing_metrics(batch, timing_raw)


@tqbridge(put_data=False)
def compute_throughout_metrics_decorated(batch, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    return compute_throughout_metrics(batch, timing_raw, n_gpus)


@tqbridge(put_data=False)
def calculate_debug_metrics_decorated(data):
    from verl.utils.debug.metrics import calculate_debug_metrics

    return calculate_debug_metrics(data)


@tqbridge(put_data=False)
def compute_val_reward_decorated(reward_fn, data, return_dict):
    return reward_fn(data, return_dict)


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
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
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
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
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = lora_rank > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.tq_client = self._initialize_transferqueue()

    def _initialize_transferqueue(self):
        # 1. initialize TransferQueueStorage
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            train_data_size = (
                self.config.data.train_batch_size
                * self.config.transfer_queue.num_global_batch
                * self.config.actor_rollout_ref.rollout.n
            )
            val_data_size = self.val_dataset_size * self.config.actor_rollout_ref.rollout.val_kwargs.n

            total_storage_size = train_data_size + val_data_size
            self.data_system_storage_units = {}
            storage_placement_group = get_placement_group(
                self.config.transfer_queue.num_data_storage_units, num_cpus_per_actor=1
            )
            for storage_unit_rank in range(self.config.transfer_queue.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.config.transfer_queue.num_data_storage_units)
                )
                self.data_system_storage_units[storage_unit_rank] = storage_node
                logging.info(f"SimpleStorageUnit #{storage_unit_rank} has been created.")
        else:
            raise NotImplementedError("Currently only support AsyncSimpleStorageManager backend in TransferQueue")

        # 2. Initialize TransferQueueController (single controller only)

        # Sampler usage instructions:
        # For GRPO grouped sampling, you can initialize the controller with GRPOGroupNSampler:
        # Option 1: Pass sampler class (will be instantiated automatically)
        # self.data_system_controller = TransferQueueController.remote(sampler=GRPOGroupNSampler)

        # Option 2: Pass sampler instance (if you need custom configuration)
        # grpo_sampler = GRPOGroupNSampler()
        # self.data_system_controller = TransferQueueController.remote(sampler=grpo_sampler)

        # Then use sampling_config in get_meta calls:
        # sampling_config={"n_samples_per_prompt": 4}
        self.data_system_controller = TransferQueueController.remote()
        logging.info("TransferQueueController has been created.")

        # 3. register controller & storage and prepare necessary information
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        # Note: Need to generate a new DictConfig with allow_objects=True to preserve ZMQServerInfo instances
        # (which contain socket connection details). Without this flag, OmegaConf would flatten these objects to dicts,
        # breaking the transfer queue client initialization.
        tq_config = OmegaConf.create({"transfer_queue": {}}, flags={"allow_objects": True})
        tq_config.transfer_queue.controller_info = self.data_system_controller_info

        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            tq_config.transfer_queue.storage_unit_infos = self.data_system_storage_unit_infos

        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. create client
        create_transferqueue_client(client_id="Trainer", config=self.config.transfer_queue, sync=True)
        tq_client = get_transferqueue_client()
        return tq_client

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

        self.val_dataset_size = len(val_dataset)

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

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_batch_size = val_batch_size

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

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
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

    def _log_rollout_data(
        self, log_rollout_meta: BatchMeta, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """
        Log rollout data to disk.

        Args:
            log_rollout_meta (BatchMeta): The batch_meta of rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            data = self.tq_client.get_data(log_rollout_meta)

            inputs = self.tokenizer.batch_decode(data["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(data["responses"], skip_special_tokens=True)
            scores = data["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.get("ground_truth", None) for item in data.get("reward_model", {})]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in log_rollout_meta.field_names:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    data["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            if "uid" not in test_data.keys():
                test_data["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_data["raw_prompt"]))], dtype=object
                )

            # repeat test data
            repeated_test_data = self.repeat_dict(
                test_data, repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            test_batch: TensorDict = self.dict_to_tensordict(repeated_test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0]["reward_model"]["style"] == "model":
                return {}

            batch_meta = self.tq_client.put(data=test_batch, partition_id=f"val_{self.global_steps - 1}")

            batch_meta.update_extra_info(
                {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }
            )
            print(f"batch_meta extra_info: {batch_meta.extra_info}")

            # TODO: (TQ) Support padding and unpadding to make DataProto divisible by dp_size with TransferQueue
            if not self.async_rollout_mode:
                test_output_gen_meta = self.actor_rollout_wg.generate_sequences(batch_meta)
            else:
                test_output_gen_meta = self.async_rollout_manager.generate_sequences(batch_meta)

            batch_meta = batch_meta.union(test_output_gen_meta)

            print("validation generation end")

            # Store generated outputs
            test_response_meta = batch_meta.select_fields(["prompts", "responses", "uid", "reward_model"])
            data = self.tq_client.get_data(test_response_meta)
            output_ids = data["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # TODO: Can we keep special tokens except for padding tokens?
            input_ids = data["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(data["uid"])

            ground_truths = [item.get("ground_truth", None) for item in data.get("reward_model", {})]
            sample_gts.extend(ground_truths)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")

            compute_reward_fields = [
                "responses",
                "prompts",
                "attention_mask",
                "reward_model",
                "data_source",
            ]
            if "rm_scores" in batch_meta.field_names:
                compute_reward_fields = ["rm_scores", *set(batch_meta.extra_info["reward_extra_keys"])]

            val_reward_meta = batch_meta.select_fields(compute_reward_fields)
            result = compute_val_reward_decorated(self.val_reward_fn, val_reward_meta, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in batch_meta.field_names:
                data = self.tq_client.get_data(batch_meta.select_fields(["__num_turns__"]))
                sample_turns.append(data["__num_turns__"])

            data_source = ["unknown"] * reward_tensor.shape[0]
            if "data_source" in batch_meta.field_names:
                data_source_meta = batch_meta.select_fields(["data_source"])
                data = self.tq_client.get_data(data_source_meta)
                data_source = data["data_source"]

            data_source_lst.append(data_source)

            self.tq_client.clear_samples(batch_meta)

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
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

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
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

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
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
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
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
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

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # set transferqueue server info for each worker
        for _, wg in all_wg.items():
            wg.create_transferqueue_client(self.config)

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from .agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

            self.checkpoint_manager = CheckpointEngineManager(
                backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
                trainer=self.actor_rollout_wg,
                replicas=self.async_rollout_manager.rollout_replicas,
            )

            # sleep all replicas to load checkpoint
            self.checkpoint_manager.sleep_replicas()

            # TODO (TQ): initialize tq during worker init when enable TQ switch is stable
            self.async_rollout_manager.create_transferqueue_client_for_workers()

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

        self.actor_rollout_wg.save_checkpoint(
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
        # load actor
        self.actor_rollout_wg.load_checkpoint(
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
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(
        self, batch: BatchMeta, tq_client, metrics, logging_prefix="global_seqlen", keep_minibatch=False
    ):
        """Reorder the batchmeta on single controller such that each dp rank gets similar total tokens"""
        data = tq_client.get_data(batch)

        attention_mask = data["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = data["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", None)
            if minibatch_size is None:
                raise ValueError("'ppo_mini_batch_size' must be set in actor config when 'keep_minibatch' is True.")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = [j for partition in global_partition_lst for j in partition]
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return global_idx

    @classmethod
    def repeat_dict(
        cls, batch_dict: dict[str, torch.Tensor | np.ndarray], repeat_times=2, interleave=True
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Repeat the batch dict a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            dict: A new dict with repeated data.
        """
        if repeat_times == 1:
            return batch_dict

        repeated_batch_dict = {}
        if batch_dict:
            if interleave:
                # Interleave the data
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        repeated_batch_dict[key] = val.repeat_interleave(repeat_times, dim=0)
                    elif isinstance(val, np.ndarray):
                        repeated_batch_dict[key] = np.repeat(val, repeat_times, axis=0)
                    else:
                        raise ValueError(f"Unsupported type in data {type(val)}")
            else:
                # Stack the data
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        repeated_batch_dict[key] = (
                            val.unsqueeze(0).expand(repeat_times, *val.shape).reshape(-1, *val.shape[1:])
                        )
                    elif isinstance(val, np.ndarray):
                        repeated_batch_dict[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))
                    else:
                        raise ValueError(f"Unsupported type in data {type(val)}")
        return repeated_batch_dict

    @classmethod
    def dict_to_tensordict(cls, data: dict[str, torch.Tensor | np.ndarray]) -> TensorDict:
        """
        Create a TensorDict from a dict of tensors and non_tensors.
        Note that this requires tensordict version at least 0.10
        """
        assert parse_version(tensordict.__version__) >= parse_version("0.10"), (
            "Storing non-tensor data in TensorDict at least requires tensordict version 0.10"
        )
        tensors_batch = {}
        batch_size = None

        for key, val in data.items():
            if isinstance(val, torch.Tensor | np.ndarray):
                tensors_batch[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

            if batch_size is None:
                batch_size = len(val)
            else:
                assert len(val) == batch_size

        if batch_size is None:
            batch_size = []
        else:
            batch_size = [batch_size]

        return TensorDict(tensors_batch, batch_size=batch_size)

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

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                base_get_meta_kwargs = dict(
                    batch_size=self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n,
                    partition_id=f"train_{self.global_steps - 1}",  # self.global_steps starts from 1
                )

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # add uid to batch
                batch_dict["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
                )
                # When n > 1, repeat input data before putting to data system, simulating DataProto repeat.
                repeated_batch_dict = self.repeat_dict(
                    batch_dict, repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                batch: TensorDict = self.dict_to_tensordict(repeated_batch_dict)
                gen_meta = self.tq_client.put(data=batch, partition_id=f"train_{self.global_steps - 1}")

                # pass global_steps to trace
                gen_meta.set_extra_info("global_steps", self.global_steps)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_output_meta = self.actor_rollout_wg.generate_sequences(gen_meta)
                        else:
                            gen_output_meta = self.async_rollout_manager.generate_sequences(gen_meta)
                        self.checkpoint_manager.sleep_replicas()
                        timing_raw.update(gen_output_meta.extra_info["timing"])
                        gen_output_meta.extra_info.pop("timing", None)

                    # TODO (TQ): support transfer queue
                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     if self.reward_fn is None:
                    #         raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                    #
                    #     with marked_timer("gen_max", timing_raw, color="purple"):
                    #         gen_baseline_meta = deepcopy(gen_meta)
                    #         gen_baseline_meta.extra_info["do_sample"] = False
                    #         if not self.async_rollout_mode:
                    #             gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_meta)
                    #         else:
                    #             gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_meta)
                    #         batch = batch.union(gen_baseline_output)
                    #         reward_baseline_tensor = self.reward_fn(batch)
                    #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                    #
                    #         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                    #
                    #         batch.batch["reward_baselines"] = reward_baseline_tensor
                    #
                    #         del gen_baseline_batch, gen_baseline_output

                    batch_meta: BatchMeta = gen_meta.union(gen_output_meta)

                    if "response_mask" not in batch_meta.field_names:
                        response_mask_meta = self.tq_client.get_meta(
                            data_fields=["responses", "attention_mask"],
                            task_name="compute_response_mask",
                            **base_get_meta_kwargs,
                        )
                        response_mask_output_meta = compute_response_mask(response_mask_meta, self.tq_client)
                        batch_meta = batch_meta.union(response_mask_output_meta)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.

                    attention_mask_meta = batch_meta.select_fields(["attention_mask"])
                    balanced_idx = None
                    if self.config.trainer.balance_batch:
                        balanced_idx = self._balance_batch(attention_mask_meta, self.tq_client, metrics=metrics)
                        batch_meta.reorder(balanced_idx)

                    # compute global_valid tokens
                    data = self.tq_client.get_data(attention_mask_meta)
                    batch_meta.extra_info["global_token_num"] = torch.sum(data["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch_meta.field_names:
                            reward_meta = self.rm_wg.compute_rm_score(batch_meta)
                            batch_meta = batch_meta.union(reward_meta)

                        compute_reward_fields = [
                            "responses",
                            "prompts",
                            "attention_mask",
                            "reward_model",
                            "data_source",
                        ]
                        if "rm_scores" in batch_meta.field_names:
                            compute_reward_fields.extend(
                                ["rm_scores", *set(batch_meta.extra_info["reward_extra_keys"])]
                            )

                        compute_reward_meta = batch_meta.select_fields(compute_reward_fields)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async_decorated(
                                data=compute_reward_meta,
                                reward_fn=self.reward_fn,
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward_decorated(
                                compute_reward_meta, self.reward_fn
                            )
                        batch_meta = batch_meta.union(compute_reward_meta)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob_meta_fields = [
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                            "prompts",
                            "responses",
                            "response_mask",
                            "data_source",
                            "reward_model",
                            "extra_info",
                            "uid",
                            "index",
                            "tools_kwargs",
                            "interaction_kwargs",
                            "ability",
                        ]
                        old_log_prob_meta = batch_meta.select_fields(old_log_prob_meta_fields)
                        old_log_prob_output_meta = self.actor_rollout_wg.compute_log_prob(old_log_prob_meta)
                        batch_meta = batch_meta.union(old_log_prob_output_meta)

                        data = self.tq_client.get_data(old_log_prob_output_meta)
                        entropys = data["entropys"]
                        response_masks = data["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        if "rollout_log_probs" in batch_meta.field_names:
                            # TODO: we may want to add diff of probs too.
                            calculate_debug_metrics_fields = ["rollout_log_probs", "old_log_probs", "responses"]

                            if "response_mask" in batch_meta.field_names:
                                calculate_debug_metrics_fields.append("response_mask")
                            if "attention_mask" in batch_meta.field_names:
                                calculate_debug_metrics_fields.append("attention_mask")

                            calculate_debug_metrics_meta = batch_meta.select_fields(calculate_debug_metrics_fields)
                            metrics.update(calculate_debug_metrics_decorated(calculate_debug_metrics_meta))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        ref_log_prob_fields = [
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                            "prompts",
                            "responses",
                            "response_mask",
                            "old_log_probs",
                            "data_source",
                            "reward_model",
                            "extra_info",
                            "uid",
                            "index",
                            "tools_kwargs",
                            "interaction_kwargs",
                            "ability",
                        ]
                        ref_log_prob_meta = batch_meta.select_fields(ref_log_prob_fields)

                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob_output_meta = self.ref_policy_wg.compute_ref_log_prob(ref_log_prob_meta)
                            else:
                                ref_log_prob_output_meta = self.actor_rollout_wg.compute_ref_log_prob(ref_log_prob_meta)
                            batch_meta = batch_meta.union(ref_log_prob_output_meta)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values_meta = self.critic_wg.compute_values(batch_meta)
                            batch_meta = batch_meta.union(values_meta)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        reward_td = TensorDict({"token_level_scores": reward_tensor}, batch_size=reward_tensor.size(0))
                        batch_meta = self.tq_client.put(data=reward_td, metadata=batch_meta)

                        if reward_extra_infos_dict:
                            reward_extra_infos_dict_new = {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            reward_extra_infos_td = self.dict_to_tensordict(reward_extra_infos_dict_new)
                            batch_meta = self.tq_client.put(data=reward_extra_infos_td, metadata=batch_meta)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            apply_kl_penalty_fields = [
                                "response_mask",
                                "token_level_scores",
                                "old_log_probs",
                                "ref_log_prob",
                            ]

                            apply_kl_penalty_meta = batch_meta.select_fields(apply_kl_penalty_fields)

                            token_level_rewards, kl_metrics = apply_kl_penalty(
                                apply_kl_penalty_meta,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            token_level_rewards_td = TensorDict(
                                {"token_level_rewards": token_level_rewards}, batch_size=token_level_rewards.size(0)
                            )
                            apply_kl_penalty_meta = self.tq_client.put(
                                data=token_level_rewards_td, metadata=apply_kl_penalty_meta
                            )

                            metrics.update(kl_metrics)
                            batch_meta = batch_meta.union(apply_kl_penalty_meta)
                        else:
                            token_level_scores_meta = batch_meta.select_fields(["token_level_scores"])

                            data = self.tq_client.get_data(token_level_scores_meta)
                            token_level_rewards_td = TensorDict(
                                {"token_level_rewards": data["token_level_scores"]},
                                batch_size=data["token_level_scores"].size(0),
                            )
                            token_level_scores_meta = self.tq_client.put(
                                data=token_level_rewards_td, metadata=token_level_scores_meta
                            )
                            batch_meta = batch_meta.union(token_level_scores_meta)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        assert "response_mask" in batch_meta.field_names, (
                            f"`response_mask` must be in batch_meta {batch_meta.field_names} for advantage computation"
                        )
                        compute_advantage_fields = [
                            "response_mask",
                            "token_level_rewards",
                        ]
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
                            compute_advantage_fields.append("values")
                        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                            compute_advantage_fields.append("uid")
                        else:
                            if "uid" in batch_meta.field_names:
                                compute_advantage_fields.append("uid")
                            if "reward_baselines" in batch_meta.field_names:
                                compute_advantage_fields.append("reward_baselines")

                        compute_advantage_meta = batch_meta.select_fields(compute_advantage_fields)

                        advantages, returns = compute_advantage(
                            compute_advantage_meta,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        advantages_td = TensorDict(
                            {"advantages": advantages, "returns": returns}, batch_size=advantages.size(0)
                        )
                        compute_advantage_meta = self.tq_client.put(data=advantages_td, metadata=compute_advantage_meta)
                        batch_meta = batch_meta.union(compute_advantage_meta)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output_meta = self.critic_wg.update_critic(batch_meta)
                            batch_meta = batch_meta.union(critic_output_meta)
                        critic_output_metrics = reduce_metrics(critic_output_meta.extra_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch_meta.extra_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )

                            update_actor_fields = [
                                "input_ids",
                                "attention_mask",
                                "position_ids",
                                "prompts",
                                "responses",
                                "response_mask",
                                "old_log_probs",
                                "ref_log_prob",
                                "advantages",
                                "returns",
                                "token_level_rewards",
                                "token_level_scores",
                                "data_source",
                                "reward_model",
                                "extra_info",
                                "uid",
                                "index",
                                "tools_kwargs",
                                "interaction_kwargs",
                                "ability",
                            ]
                            update_actor_meta = batch_meta.select_fields(update_actor_fields)

                            update_actor_meta.set_extra_info(
                                "global_token_num", batch_meta.get_extra_info("global_token_num")
                            )
                            update_actor_meta.set_extra_info("temperature", batch_meta.get_extra_info("temperature"))

                            actor_output_meta = self.actor_rollout_wg.update_actor(update_actor_meta)
                            batch_meta = batch_meta.union(actor_output_meta)

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output_meta.extra_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        log_rollout_fields = ["prompts", "responses", "token_level_scores", "reward_model"]
                        if "request_id" in batch_meta.field_names:
                            log_rollout_fields.append("request_id")
                        log_rollout_meta = batch_meta.select_fields(log_rollout_fields)
                        self._log_rollout_data(log_rollout_meta, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # TODO: validate
                if (
                    self.val_reward_fn is not None
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
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

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
                compute_data_metrics_fields = [
                    "token_level_rewards",
                    "token_level_scores",
                    "advantages",
                    "returns",
                    "responses",
                    "attention_mask",
                    "response_mask",
                ]
                if "__num_turns__" in batch_meta.field_names:
                    compute_data_metrics_fields.append("__num_turns__")
                if "tool_call_counts" in batch_meta.field_names:
                    compute_data_metrics_fields.append("tool_call_counts")
                compute_data_metrics_meta = batch_meta.select_fields(compute_data_metrics_fields)
                compute_data_metrics_meta.reorder(balanced_idx)
                metrics.update(
                    compute_data_metrics_decorated(batch=compute_data_metrics_meta, use_critic=self.use_critic)
                )

                compute_timing_metrics_fields = ["responses", "attention_mask"]
                compute_timing_metrics_meta = batch_meta.select_fields(compute_timing_metrics_fields)
                compute_timing_metrics_meta.reorder(balanced_idx)
                metrics.update(
                    compute_timing_metrics_decorated(batch=compute_timing_metrics_meta, timing_raw=timing_raw)
                )

                compute_throughout_metrics_meta = BatchMeta(
                    samples=[],
                    extra_info={"global_token_num": batch_meta.get_extra_info("global_token_num")},
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics_decorated(
                        batch=compute_throughout_metrics_meta, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    # TODO (TQ) :support transfer queue
                    self.train_dataloader.sampler.update(batch=batch)

                self.tq_client.clear_samples(batch_meta)
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    # TODO (TQ): support transfer queue
                    self.train_dataset.on_batch_end(batch=batch)
