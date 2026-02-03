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
PPO Trainer with TransferQueue support.
"""

import math
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from transfer_queue import (
    BatchMeta,
    SimpleStorageUnit,
    TransferQueueController,
    get_placement_group,
    process_zmq_server_info,
)

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, apply_kl_penalty
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.transferqueue_utils import create_transferqueue_client, get_transferqueue_client, repeat_dict, tqbridge
from verl.workers.config import FSDPEngineConfig


@tqbridge(put_data=True)
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

        if config.get("use_pf_ppo", False):
            # the below code will resample the full data, for TQ adaption, we will return the resampled index
            # which will be read and used for resample later in fit func
            pf_ppo_reweight_idx = core_algos.compute_pf_ppo_reweight_data_tq(
                data.batch["token_level_rewards"],
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )

            advantages_td = TensorDict(
                {
                    "advantages": advantages,
                    "returns": returns,
                },
                batch_size=advantages.size(0),
            )
            non_tensor_batch = {"pf_ppo_reweight_idx": pf_ppo_reweight_idx}
            return DataProto(batch=advantages_td, non_tensor_batch=non_tensor_batch)
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

    advantages_td = TensorDict({"advantages": advantages, "returns": returns}, batch_size=advantages.size(0))
    return DataProto(batch=advantages_td)


# TODO: dispatch these decorated functions from single-controller
@tqbridge(put_data=False)
def compute_reward_decorated(data, reward_fn):
    return compute_reward(data, reward_fn)


@tqbridge(put_data=False)
def compute_reward_async_decorated(data, reward_fn):
    return compute_reward_async.remote(data, reward_fn)


@tqbridge(put_data=False, convert_type="TensorDict")
def compute_data_metrics_decorated(batch, use_critic: bool = True):
    return compute_data_metrics(batch, use_critic)


@tqbridge(put_data=False, convert_type="TensorDict")
def compute_timing_metrics_decorated(batch, timing_raw: dict[str, float]) -> dict[str, Any]:
    return compute_timing_metrics(batch, timing_raw)


@tqbridge(put_data=False, convert_type="TensorDict")
def compute_throughout_metrics_decorated(batch, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    return compute_throughout_metrics(batch, timing_raw, n_gpus)


@tqbridge(put_data=False, convert_type="TensorDict")
def calculate_debug_metrics_decorated(data):
    from verl.utils.debug.metrics import calculate_debug_metrics

    return calculate_debug_metrics(data)


@tqbridge(put_data=False)
def compute_val_reward_decorated(reward_fn, data, return_dict):
    return reward_fn(data, return_dict)


class RayPPOTrainerTransferQueue(RayPPOTrainer):
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
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )

        # Initialize TransferQueue client
        self.tq_client = self._initialize_transferqueue()

    def _initialize_transferqueue(self):
        # 1. initialize TransferQueueStorage
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            train_data_size = (
                self.config.data.train_batch_size
                * self.config.transfer_queue.num_global_batch
                * self.config.actor_rollout_ref.rollout.n
            )
            # val_data_size = self.val_dataset_size * self.config.actor_rollout_ref.rollout.val_kwargs.n
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)
            val_data_size = val_batch_size * self.config.actor_rollout_ref.rollout.val_kwargs.n

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
                print(f"SimpleStorageUnit #{storage_unit_rank} has been created.")
        else:
            raise NotImplementedError("Currently only support AsyncSimpleStorageManager backend in TransferQueue.")

        # 2. Initialize TransferQueueController
        self.data_system_controller = TransferQueueController.remote()
        print("TransferQueueController has been created.")

        # 3. register controller & storage and prepare necessary information
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        # Note: Need to generate a new DictConfig with allow_objects=True to preserve ZMQServerInfo instances
        tq_config = OmegaConf.create({"transfer_queue": {}}, flags={"allow_objects": True})
        tq_config.transfer_queue.controller_info = self.data_system_controller_info

        if self.config.transfer_queue.storage_backend == "AsyncSimpleStorageManager":
            tq_config.transfer_queue.storage_unit_infos = self.data_system_storage_unit_infos

        self.config = OmegaConf.merge(tq_config, self.config)

        # 4. create client
        create_transferqueue_client(client_id="Trainer", config=self.config.transfer_queue)
        tq_client = get_transferqueue_client()
        return tq_client

    # Note: Now in these functions, we remove the unpadding/padding process. When TransferQueue becomes the
    #       default option, we can let dataloader to directly produce unpadded NestedTensor batches.
    def _compute_values(self, batch_meta: BatchMeta) -> BatchMeta:
        if self.use_legacy_worker_impl == "disable":
            batch_meta.extra_info["compute_loss"] = False
            values_meta = self.critic_wg.infer_batch(batch_meta)
        else:
            values_meta = self.critic_wg.compute_values(batch_meta)
        return values_meta

    def _compute_ref_log_prob(self, batch_meta: BatchMeta) -> BatchMeta:
        if self.use_legacy_worker_impl == "disable":
            batch_meta.extra_info["compute_loss"] = False
            batch_meta.extra_info["calculate_entropy"] = False
            if self.ref_in_actor:
                batch_meta.extra_info["no_lora_adapter"] = True

            if self.ref_in_actor:
                # output contains log_probs and ref_log_prob
                output_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            else:
                output_meta = self.ref_policy_wg.compute_ref_log_prob(batch_meta)
        else:
            output_meta = self.ref_policy_wg.compute_ref_log_prob(batch_meta)
        return output_meta

    def _compute_old_log_prob(self, batch_meta: BatchMeta) -> tuple[BatchMeta, Any]:
        if self.use_legacy_worker_impl == "disable":
            batch_meta.extra_info["compute_loss"] = False
            batch_meta.extra_info["calculate_entropy"] = True
            output_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            # metrics originally is saved in tensordict as a NonTensorData
            # after tqbridge, metrics should be turned into batchmeta extra_info
            old_log_prob_mfu = output_meta.extra_info.get("metrics")["mfu"]
        else:
            output_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            old_log_prob_mfu = 0
        return output_meta, old_log_prob_mfu

    def _update_actor(self, batch_meta: BatchMeta) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch_meta.extra_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch_meta.extra_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            extra_meta = {
                "calculate_entropy": calculate_entropy,
                "global_batch_size": ppo_mini_batch_size,
                "mini_batch_size": ppo_mini_batch_size,
                "epochs": ppo_epochs,
                "seed": seed,
                "dataloader_kwargs": {"shuffle": shuffle},
            }
            batch_meta.update_extra_info(extra_meta)
            output = self.actor_rollout_wg.update_actor(batch_meta)
            output = tu.get(output, "metrics")
            output = rename_dict(output, "actor/")
            # modify key name
            output["perf/mfu/actor"] = output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch_meta)
        return actor_output

    def _update_critic(self, batch_meta: BatchMeta) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            extra_meta = {
                "global_batch_size": ppo_mini_batch_size,
                "mini_batch_size": ppo_mini_batch_size,
                "epochs": ppo_epochs,
                "seed": seed,
                "dataloader_kwargs": {"shuffle": shuffle},
            }
            batch_meta.update_extra_info(extra_meta)
            output = self.critic_wg.train_mini_batch(batch_meta)
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch_meta)
        return critic_output

    def _get_gen_batch_fields(self, non_tensor_batch_keys: set) -> set:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & non_tensor_batch_keys

        # pop those keys for generation
        tensor_batch_keys_to_pop = set()
        non_tensor_batch_keys_to_pop = non_tensor_batch_keys - reward_model_keys
        gen_batch_field_names = tensor_batch_keys_to_pop | non_tensor_batch_keys_to_pop

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch_field_names = gen_batch_field_names | non_tensor_batch_keys

        return gen_batch_field_names

    def _validate(self, merged: bool = False):
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
            repeated_test_data = repeat_dict(
                test_data, repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            test_batch: TensorDict = tu.dict_to_tensordict(repeated_test_data)

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

            test_gen_fields = self._get_gen_batch_fields(tu.get_non_tensor_keys(test_batch))
            del test_batch
            test_gen_meta = batch_meta.select_fields(list(test_gen_fields))

            if not self.async_rollout_mode:
                test_output_gen_meta = self.actor_rollout_wg.generate_sequences(test_gen_meta)
            else:
                test_output_gen_meta = self.async_rollout_manager.generate_sequences(test_gen_meta)

            batch_meta = batch_meta.union(test_output_gen_meta)

            print("validation generation end")

            # Store generated outputs
            test_response_meta = batch_meta.select_fields(["prompts", "responses", "uid", "reward_model"])
            data = self.tq_client.get_data(test_response_meta)
            output_ids = data["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Store original inputs
            input_ids = data["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(data["uid"])

            # Store ground truths
            ground_truths = [item.get("ground_truth", None) for item in data.get("reward_model", {})]
            sample_gts.extend(ground_truths)

            # evaluate using reward_function
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

            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                val_reward_meta, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in batch_meta.field_names:
                data = self.tq_client.get_data(batch_meta.select_fields(["__num_turns__"]))
                sample_turns.append(data["__num_turns__"])

            data_source = ["unknown"] * reward_tensor.shape[0]
            if "data_source" in batch_meta.field_names:
                data = self.tq_client.get_data(batch_meta.select_fields(["data_source"]))
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

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout

        # pass tq_config to workers if enable TQ
        actor_rollout_ref_config = self.config.actor_rollout_ref
        reward_config = self.config.reward_model
        tq_config = OmegaConf.select(self.config, "transfer_queue", default=None)
        assert tq_config is not None and tq_config["enable"], (
            "Must have TQ related configs and set to enable when running RayPPOTrainerTransferQueue"
        )
        OmegaConf.set_struct(actor_rollout_ref_config, False)
        actor_rollout_ref_config.transfer_queue = tq_config
        OmegaConf.set_struct(actor_rollout_ref_config, True)
        OmegaConf.set_struct(reward_config, False)
        reward_config.transfer_queue = tq_config
        OmegaConf.set_struct(reward_config, True)
        from verl.workers.config.engine import TransferQueueConfig

        transferqueue_config = TransferQueueConfig.from_dict(tq_config)

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=actor_rollout_ref_config,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                    transfer_queue=transferqueue_config,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=actor_rollout_ref_config,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

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
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

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

        current_epoch = self.global_steps // len(self.train_dataloader)

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

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

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

                repeated_batch_dict = repeat_dict(
                    batch_dict, repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                batch: TensorDict = tu.dict_to_tensordict(repeated_batch_dict)
                del repeated_batch_dict

                batch_meta = self.tq_client.put(data=batch, partition_id=f"train_{self.global_steps - 1}")
                batch_meta.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch_meta.extra_info["global_steps"] = self.global_steps  # pass global_steps to trace

                gen_batch_fields = self._get_gen_batch_fields(tu.get_non_tensor_keys(batch))
                del batch
                gen_meta = batch_meta.select_fields(list(gen_batch_fields))

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_output_meta = self.actor_rollout_wg.generate_sequences(gen_meta)
                        else:
                            gen_output_meta = self.async_rollout_manager.generate_sequences(gen_meta)
                        timing_raw.update(gen_output_meta.extra_info["timing"])
                        gen_output_meta.extra_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_meta = deepcopy(gen_meta)
                            gen_baseline_meta.extra_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output_meta = self.actor_rollout_wg.generate_sequences(gen_baseline_meta)
                            else:
                                gen_baseline_output_meta = self.async_rollout_manager.generate_sequences(
                                    gen_baseline_meta
                                )
                            batch_meta = batch_meta.union(gen_baseline_output_meta)
                            # compute reward model score on batch
                            rm_scores_output_meta = None
                            if self.use_rm and "rm_scores" not in batch_meta.field_names:
                                if not self.use_reward_loop:
                                    rm_scores_output_meta = self.rm_wg.compute_rm_score(batch_meta)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores_output_meta = self.reward_loop_manager.compute_rm_score(batch_meta)
                                batch_meta = batch_meta.union(rm_scores_output_meta)

                            # Compute or extract reward for REMAX baseline
                            compute_reward_fields = [
                                "responses",
                                "prompts",
                                "attention_mask",
                                "reward_model",
                                "data_source",
                                "rm_scores",
                            ]
                            if "rm_scores" in batch_meta.field_names:
                                compute_reward_fields.extend(
                                    ["rm_scores", *set(batch_meta.extra_info["reward_extra_keys"])]
                                )

                            compute_reward_meta = batch_meta.select_fields(compute_reward_fields)

                            reward_baseline_tensor = self._compute_or_extract_reward(
                                compute_reward_meta, reward_fn=self.reward_fn, sum_reward=True
                            )

                            reward_baseline_tensor_td = TensorDict(
                                {"reward_baselines": reward_baseline_tensor}, batch_size=reward_baseline_tensor.size(0)
                            )
                            batch_meta = self.tq_client.put(data=reward_baseline_tensor_td, metadata=batch_meta)
                            del reward_baseline_tensor_td

                            keys_to_pop = set(gen_baseline_output_meta.field_names)
                            if rm_scores_output_meta is not None:
                                keys_to_pop.update(rm_scores_output_meta.field_names)
                            keys_to_retain = list(set(batch_meta.field_names) - keys_to_pop)
                            batch_meta = batch_meta.select_fields(keys_to_retain)
                            del rm_scores_output_meta, gen_baseline_meta, gen_baseline_output_meta

                    batch_meta = batch_meta.union(gen_output_meta)
                    del gen_output_meta, gen_meta

                    assert "response_mask" in batch_meta.field_names

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        attention_mask_meta = batch_meta.select_fields(["attention_mask"])
                        balanced_idx = self._balance_batch(attention_mask_meta, metrics=metrics)
                        batch_meta = batch_meta.select_samples(balanced_idx)

                    # compute global_valid tokens
                    data = self.tq_client.get_data(batch_meta.select_fields(["attention_mask"]))
                    # tu.strided_to_jagged(data)
                    batch_meta.extra_info["global_token_num"] = data["attention_mask"].sum(-1).tolist()
                    del data

                    # get images_seqlens
                    images_seqlens_all = []
                    images_seqlens_meta = batch_meta.select_fields(["multi_modal_inputs"])
                    data = self.tq_client.get_data(images_seqlens_meta)  # non tensor
                    for multi_modal_input in data:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch_meta.extra_info["images_seqlens"] = images_seqlens_all
                    del data

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch_meta.field_names:
                            if not self.use_reward_loop:
                                reward_tensor_meta = self.rm_wg.compute_rm_score(batch_meta)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor_meta = self.reward_loop_manager.compute_rm_score(batch_meta)
                            batch_meta = batch_meta.union(reward_tensor_meta)

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

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async_decorated(
                                data=compute_reward_meta, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                compute_reward_meta, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    # TODO: simplify this workflow when integrating with TransferQueue
                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        old_log_prob_bypass_meta = batch_meta.select_fields(["rollout_log_probs"])

                        old_log_prob_bypass_output_meta = apply_bypass_mode(
                            batch=old_log_prob_bypass_meta,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                        batch_meta = batch_meta.union(old_log_prob_bypass_output_meta)
                    else:  # Recompute old_log_probs
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
                            old_log_prob_output_meta, old_log_prob_mfu = self._compute_old_log_prob(old_log_prob_meta)

                            data = self.tq_client.get_data(
                                old_log_prob_output_meta.select_fields(["log_probs", "response_mask", "entropys"])
                            )
                            old_log_probs = TensorDict(
                                {"old_log_probs": data["log_probs"]},
                                batch_size=data["log_probs"].size(0),
                            )
                            old_log_prob_output_meta = self.tq_client.put(
                                data=old_log_probs, metadata=old_log_prob_output_meta
                            )
                            del old_log_probs

                            old_log_prob_output_meta.field_names.remove("log_probs")
                            old_log_prob_output_meta.field_names.remove("entropys")

                            # tu.strided_to_jagged(data)
                            entropys = data["entropys"]
                            response_masks = data["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob_output_meta = old_log_prob_output_meta.select_fields(["old_log_probs"])
                            batch_meta = batch_meta.union(old_log_prob_output_meta)
                            del data, entropys, response_masks

                            if "rollout_log_probs" in batch_meta.field_names:
                                # TODO: we may want to add diff of probs too.
                                calculate_debug_metrics_fields = ["rollout_log_probs", "old_log_probs", "responses"]

                                if "response_mask" in batch_meta.field_names:
                                    calculate_debug_metrics_fields.append("response_mask")
                                if "attention_mask" in batch_meta.field_names:
                                    calculate_debug_metrics_fields.append("attention_mask")

                                calculate_debug_metrics_meta = batch_meta.select_fields(calculate_debug_metrics_fields)
                                metrics.update(calculate_debug_metrics_decorated(calculate_debug_metrics_meta))

                    assert "old_log_probs" in batch_meta.field_names, (
                        f'"old_log_probs" not in {batch_meta.field_names=}'
                    )
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
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob_output_meta = self._compute_ref_log_prob(ref_log_prob_meta)
                            if self.ref_in_actor:
                                data = self.tq_client.get_data(ref_log_prob_output_meta.select_fields(["log_probs"]))
                                ref_log_probs = TensorDict(
                                    {"ref_log_prob": data["log_probs"]},
                                    batch_size=data["log_probs"].size(0),
                                )
                                ref_log_prob_output_meta = self.tq_client.put(
                                    data=ref_log_probs, metadata=ref_log_prob_output_meta
                                )
                                ref_log_prob_output_meta.field_names.remove("log_probs")
                                del data, ref_log_probs
                            batch_meta = batch_meta.union(ref_log_prob_output_meta)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values_meta = self._compute_values(batch_meta)
                            batch_meta = batch_meta.union(values_meta)

                    # TODO: dispatch adv computation to workers, or simplify the workflow if needed
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        reward_td = TensorDict({"token_level_scores": reward_tensor}, batch_size=reward_tensor.size(0))
                        batch_meta = self.tq_client.put(data=reward_td, metadata=batch_meta)

                        if reward_extra_infos_dict:
                            reward_extra_infos_dict_new = {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            reward_extra_infos_td = tu.dict_to_tensordict(reward_extra_infos_dict_new)
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
                            del data, token_level_rewards_td

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch_meta.field_names
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            rollout_correction_meta_fields = ["old_log_probs", "rollout_log_probs", "response_mask"]
                            rollout_correction_meta = batch_meta.select_fields(rollout_correction_meta_fields)
                            data, is_metrics = compute_rollout_correction_and_add_to_batch(
                                rollout_correction_meta, rollout_corr_config
                            )  # data is a dataproto
                            correction_td = TensorDict(
                                {
                                    "response_mask": data.batch["response_mask"],
                                    "rollout_is_weights": data.batch["rollout_is_weights"],
                                },
                                batch_size=data.batch["rollout_is_weights"].size(0),
                            )
                            batch_meta = self.tq_client.put(data=correction_td, metadata=batch_meta)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        assert "response_mask" in batch_meta.field_names

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
                        compute_advantage_output_meta = compute_advantage(
                            compute_advantage_meta,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        batch_meta = batch_meta.union(compute_advantage_output_meta)

                        full_batch_meta = deepcopy(batch_meta)
                        if "pf_ppo_reweight_idx" in batch_meta.field_names and self.config.transfer_queue.enable:
                            resample_idx_meta = batch_meta.select_fields(["pf_ppo_reweight_idx"])
                            resampled_idx = self.tq_client.get_data(resample_idx_meta)  # list of int
                            batch_meta = full_batch_meta.select_samples(resampled_idx)
                            del resampled_idx

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch_meta)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
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
                            actor_output = self._update_actor(update_actor_meta)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        log_rollout_fields = ["prompts", "responses", "token_level_scores", "reward_model"]
                        if "request_id" in batch_meta.field_names:
                            log_rollout_fields.append("request_id")
                        log_rollout_meta = batch_meta.select_fields(log_rollout_fields)
                        self._log_rollout_data(log_rollout_meta, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
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
                if self.use_critic:
                    compute_data_metrics_fields.append("values")

                compute_data_metrics_meta = batch_meta.select_fields(compute_data_metrics_fields)
                metrics.update(
                    compute_data_metrics_decorated(batch=compute_data_metrics_meta, use_critic=self.use_critic)
                )

                compute_timing_metrics_fields = ["responses", "attention_mask"]

                compute_timing_metrics_meta = batch_meta.select_fields(compute_timing_metrics_fields)
                metrics.update(
                    compute_timing_metrics_decorated(batch=compute_timing_metrics_meta, timing_raw=timing_raw)
                )

                compute_throughout_metrics_meta = BatchMeta(
                    samples=[],
                    extra_info={"global_token_num": batch_meta.extra_info.get("global_token_num")},
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics_decorated(
                        batch=compute_throughout_metrics_meta, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    # TODO (TQ) :support this feature
                    print("Currently TransferQueue does not support this!")
                    # self.train_dataloader.sampler.update(batch=batch)

                self.tq_client.clear_samples(full_batch_meta)
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
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    # TODO (TQ) :support this feature
                    print("Currently TransferQueue does not support this!")
                    # self.train_dataset.on_batch_end(batch=batch)
