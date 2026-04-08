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
MAPPO Trainer with Ray-based single controller.
Provides RayMAPPOTrainer and RayRiskAverseTrainer for multi-agent PPO training.
"""

import csv
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ResourcePoolManager
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
try:
    from verl.trainer.ppo.reward import compute_reward, compute_reward_async
except ImportError:
    compute_reward = None  # type: ignore[assignment]
    compute_reward_async = None  # type: ignore[assignment]
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
import verl.utils.torch_functional as verl_F


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset for a single agent.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer: The tokenizer.
        processor: The processor.

    Returns:
        dataset: The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.import_utils import load_extern_type

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset: The dataset.

    Returns:
        sampler: The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    from verl.experimental.dataset.sampler import AbstractSampler
    from verl.utils.import_utils import load_extern_type

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching."
        )
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler
class RayMAPPOTrainer:
    """Distributed MAPPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizers,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processors=None,
        reward_fns=None,
        val_reward_fns=None,
        train_datasets: Optional[Dataset] = None,
        val_datasets: Optional[Dataset] = None,
        collate_fn=None,
        train_samplers: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed MAPPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizers: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processors: Optional data processor, used for multimodal data
            reward_fns: Function for computing rewards during training.
            val_reward_fns: Function for computing rewards during validation.
            train_datasets (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_datasets (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_samplers (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizers = tokenizers
        self.processors = processors
        self.config = config
        self.reward_fns = reward_fns
        self.val_reward_fns = val_reward_fns

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = self.config.algorithm.use_kl_in_reward
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_datasets, val_datasets, collate_fn, train_samplers)

    def _create_dataloader(self, train_datasets, val_datasets, collate_fn, train_samplers: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        num_agents = int(ma.get("num_agents", 1))
        if train_datasets is None:
            train_datasets={}
            for i in range(num_agents):
                train_datasets[f"model_{i}"] = create_rl_dataset(
                    self.config.data.train_files, self.config.data, self.tokenizers[f"model_{i}"], self.processors[f"model_{i}"]
                )
        if val_datasets is None:
            val_datasets={}
            for i in range(num_agents):
                val_datasets[f"model_{i}"] = create_rl_dataset(
                    self.config.data.val_files, self.config.data, self.tokenizers[f"model_{i}"], self.processors[f"model_{i}"]
                )
        self.train_datasets, self.val_datasets = train_datasets, val_datasets

        if train_samplers is None:
            train_samplers={}
            for i in range(num_agents):
                train_samplers[f"model_{i}"] = create_rl_sampler(self.config.data, self.train_datasets[f"model_{i}"])
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]
        self.train_dataloaders={}
        for i in range(num_agents):
            self.train_dataloaders[f"model_{i}"] = StatefulDataLoader(
                dataset=self.train_datasets[f"model_{i}"],
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_samplers[f"model_{i}"],
            )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_datasets["model_0"])

        self.val_dataloaders={}
        for i in range(num_agents):
            self.val_dataloaders[f"model_{i}"] = StatefulDataLoader(
                dataset=self.val_datasets[f"model_{i}"],
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloaders["model_0"]) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloaders["model_0"]) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloaders['model_0'])}, Size of val dataloader: "
            f"{len(self.val_dataloaders['model_0'])}"
        )

        total_training_steps = len(self.train_dataloaders["model_0"]) * self.config.trainer.total_epochs

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
    
    def _build_validation_rollout_id(
        self,
        *,
        global_step: int,
        batch_idx: int,
        sample_uid: str,
        sample_idx: int,
    ) -> str:
        return f"gs{global_step}_b{batch_idx}_i{sample_idx}_{sample_uid}"

    def _extract_validation_correctness(self, reward_extra_info, sample_idx: int, score: float):
        if reward_extra_info:
            for key in ("acc", "correct", "is_correct"):
                values = reward_extra_info.get(key)
                if values is None or sample_idx >= len(values):
                    continue

                value = values[sample_idx]
                if hasattr(value, "item"):
                    value = value.item()
                return bool(value), key

        return score > 0, "score>0"

    def _append_validation_rollout_record(
        self,
        *,
        rollout_records: dict,
        rollout_id: str,
        global_step: int,
        batch_idx: int,
        sample_idx: int,
        round_idx: int,
        agent_idx: int,
        agent_key: str,
        question,
        ground_truth,
        output_text: str,
        score: float,
        reward_extra_info,
        sample_uid: str,
        data_source,
    ):
        is_correct, correctness_source = self._extract_validation_correctness(reward_extra_info, sample_idx, score)
        record = rollout_records.setdefault(
            rollout_id,
            {
                "rollout_id": rollout_id,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "uid": sample_uid,
                "data_source": data_source,
                "question": question,
                "ground_truth": ground_truth,
                "rows": [],
            },
        )
        record["rows"].append(
            {
                "rollout_id": rollout_id,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "round_idx": round_idx,
                "agent_idx": agent_idx,
                "agent_key": agent_key,
                "question": question,
                "ground_truth": ground_truth,
                "output_text": output_text,
                "score": score,
                "is_correct": is_correct,
                "correctness_source": correctness_source,
                "uid": sample_uid,
                "data_source": data_source,
            }
        )

    def _save_multi_agent_validation_csv(self, rollout_records: dict) -> Optional[str]:
        csv_dir = self.config.trainer.get("validation_rollout_csv_dir", None)
        if not csv_dir or not rollout_records:
            return None

        fieldnames = [
            "rollout_id",
            "global_step",
            "data_source",
            "question",
            "ground_truth",
            "round_idx",
            "agent_idx",
            "agent_key",
            "output_text",
            "score",
            "is_correct",
            "correctness_source",
            "uid",
        ]
        csv_rows = []
        for rollout_id in sorted(rollout_records):
            record = rollout_records[rollout_id]
            for row in record.get("rows", []):
                csv_rows.append(
                    {
                        "rollout_id": row.get("rollout_id", record.get("rollout_id")),
                        "global_step": row.get("global_step", record.get("global_step")),
                        "data_source": row.get("data_source", record.get("data_source")),
                        "question": row.get("question", record.get("question")),
                        "ground_truth": row.get("ground_truth", record.get("ground_truth")),
                        "round_idx": row.get("round_idx"),
                        "agent_idx": row.get("agent_idx"),
                        "agent_key": row.get("agent_key"),
                        "output_text": row.get("output_text"),
                        "score": row.get("score"),
                        "is_correct": row.get("is_correct"),
                        "correctness_source": row.get("correctness_source"),
                        "uid": row.get("uid", record.get("uid")),
                    }
                )

        if not csv_rows:
            return None

        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"mappo_val_rollouts_step_{self.global_steps}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        return csv_path

    # TODO:Revise for multi-agent PPO
    def _multi_agent_validate(self):
        metric_dict = {}
        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        agent_keys = list(self.val_dataloaders.keys())
        num_agents = int(ma.get("num_agents", 1))
        num_rounds = int(ma.get("num_rounds", 1))
        sample_scores_lsts=[[[] for _ in range(num_agents)] for _ in range(num_rounds)]
        sample_uids_lsts=[[[] for _ in range(num_agents)] for _ in range(num_rounds)]
        sample_outputs_lsts= [[[] for _ in range(num_agents)] for _ in range(num_rounds)]
        sample_inputs_lsts=[[[] for _ in range(num_agents)] for _ in range(num_rounds)]
        sample_turns_lsts = [[[] for _ in range(num_agents)] for _ in range(num_rounds)]
        sample_gts=[]
        data_source_lsts=[[[] for _ in range(num_agents)] for _ in range(num_rounds)]
        reward_extra_info_list= [[defaultdict(list) for _ in range(num_agents)] for _ in range(num_rounds)]
        rollout_records = {}
        for batch_idx, batch_tuple in enumerate(zip(*(self.val_dataloaders[k] for k in agent_keys))):
            batch_size = len(DataProto.from_single_dict(batch_tuple[0]))
            histories=[""]*batch_size
            for r in range(num_rounds):
                this_round = [""] * batch_size
                for agent_idx,agent_key in enumerate(agent_keys):
                    sample_inputs = sample_inputs_lsts[r][agent_idx]
                    sample_uids = sample_uids_lsts[r][agent_idx]
                    sample_outputs=sample_outputs_lsts[r][agent_idx]
                    sample_scores = sample_scores_lsts[r][agent_idx]
                    sample_turns = sample_turns_lsts[r][agent_idx]
                    data_source_lst = data_source_lsts[r][agent_idx]
                    batch_dict=batch_tuple[agent_idx]
                    reward_extra_infos_dict=reward_extra_info_list[r][agent_idx]
                    test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    questions, _ =self._extract_prompts_and_questions(test_batch,agent_key)
                    chat_prompts=[]
                    if r==0:
                        chat_prompts=questions
                    if r>0:
                        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please collaborate with the other agent to help the user. Provide a well−reasoned response that not only considers your own previous solution but also takes into account answers from other agents. If you believe your previous answer was incorrect, feel free to revise it. Ensure that your explanation clearly justifies your final answer. Please maintain your answer with very simple reasoning."
                        discussion_prompt=f"The discussion history is as follows: "
                        chat_prompts = self._build_input_ids_from_histories(system_prompt,discussion_prompt,questions,histories,test_batch,agent_key,max_history_tokens=4096)
                    if "uid" not in test_batch.non_tensor_batch:
                        test_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                        )
                    sample_inputs.extend(chat_prompts)
                    sample_uids.extend(test_batch.non_tensor_batch["uid"])

                    if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                        return {}
                    ground_truths = [
                        item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
                    ]
                    if agent_idx==0 and r==0:
                        sample_gts.extend(ground_truths)
                    test_gen_batch = self._get_gen_batch(test_batch)
                    test_gen_batch.meta_info = {
                        "eos_token_id": self.tokenizers[agent_key].eos_token_id,
                        "pad_token_id": self.tokenizers[agent_key].pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                        "validate": True,
                        "global_steps": self.global_steps,
                    }
                    print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
                    size_divisor = (
                        self.actor_rollout_wgs[agent_key].world_size
                        if not self.async_rollout_mode
                        else self.config.actor_rollout_ref.rollout.agent.num_workers
                    )
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                    if not self.async_rollout_mode:
                        test_output_gen_batch_padded = self.actor_rollout_wgs[agent_key].generate_sequences(test_gen_batch_padded)
                    else:
                        test_output_gen_batch_padded = self.async_rollout_managers[agent_key].generate_sequences(test_gen_batch_padded)

                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                    print("validation generation end")

                    output_ids = test_output_gen_batch.batch["responses"]
                    output_texts = [self.tokenizers[agent_key].decode(ids, skip_special_tokens=True) for ids in output_ids]
                    this_round = [
                        old + f"\nAgent {agent_idx}: {new}" 
                        for old, new in zip(this_round, output_texts)
                    ]
                    sample_outputs.extend(output_texts)

                    test_batch = test_batch.union(test_output_gen_batch)
                    test_batch.meta_info["validate"] = True

                    if self.val_reward_fns is None:
                        raise ValueError("val_reward_fn must be provided for validation.")
                    result = self.val_reward_fns[agent_key](test_batch, return_dict=True)
                    reward_tensor = result["reward_tensor"]
                    scores = reward_tensor.sum(-1).cpu().tolist()
                    sample_scores.extend(scores)

                    reward_extra_infos_dict["reward"].extend(scores)
                    print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
                    reward_extra_info = result.get("reward_extra_info", {})
                    if "reward_extra_info" in result:
                        for key, lst in result["reward_extra_info"].items():
                            reward_extra_infos_dict[key].extend(lst)
                            print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

                    if "__num_turns__" in test_batch.non_tensor_batch:
                        sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

                    batch_uids = [str(uid) for uid in test_batch.non_tensor_batch["uid"]]
                    batch_data_sources = list(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
                    data_source_lst.append(batch_data_sources)

                    for sample_idx, (sample_uid, question, ground_truth, output_text, score, data_source) in enumerate(
                        zip(batch_uids, questions, ground_truths, output_texts, scores, batch_data_sources, strict=True)
                    ):
                        rollout_id = self._build_validation_rollout_id(
                            global_step=self.global_steps,
                            batch_idx=batch_idx,
                            sample_uid=sample_uid,
                            sample_idx=sample_idx,
                        )
                        self._append_validation_rollout_record(
                            rollout_records=rollout_records,
                            rollout_id=rollout_id,
                            global_step=self.global_steps,
                            batch_idx=batch_idx,
                            sample_idx=sample_idx,
                            round_idx=r,
                            agent_idx=agent_idx,
                            agent_key=agent_key,
                            question=question,
                            ground_truth=ground_truth,
                            output_text=output_text,
                            score=score,
                            reward_extra_info=reward_extra_info,
                            sample_uid=sample_uid,
                            data_source=data_source,
                        )

                histories[:] = [
                    f"[Last round]: {new}"
                    for new in this_round
                ]
                if r==num_rounds-1:
                    print(histories[0])
        for r in range(num_rounds):
            for agent_idx,agent_key in enumerate(agent_keys):
                sample_inputs = sample_inputs_lsts[r][agent_idx]
                sample_uids = sample_uids_lsts[r][agent_idx]
                sample_turns = []
                sample_outputs=sample_outputs_lsts[r][agent_idx]
                sample_scores = sample_scores_lsts[r][agent_idx]
                data_source_lst = data_source_lsts[r][agent_idx]
                reward_extra_infos_dict=reward_extra_info_list[r][agent_idx]
                self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

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
                            pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}/round_{r}/{agent_key}"
                            metric_dict[pfx] = metric_val

                if len(sample_turns) > 0:
                    sample_turns = np.concatenate(sample_turns)
                    metric_dict[f"val-aux/num_turns/min/round_{r}/{agent_key}"] = sample_turns.min()
                    metric_dict[f"val-aux/num_turns/max/round_{r}/{agent_key}"] = sample_turns.max()
                    metric_dict[f"val-aux/num_turns/mean/round_{r}/{agent_key}"] = sample_turns.mean()

        save_validation_csv = getattr(self, "_save_multi_agent_validation_csv", None)
        if callable(save_validation_csv):
            save_validation_csv(rollout_records)

        return metric_dict
    def _validate(self):
        metric_dict = {}
        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        num_agents = int(ma.get("num_agents", 1))
        num_rounds = int(ma.get("num_rounds", 1))
        for i in range(num_agents):
            data_source_lst = []
            reward_extra_infos_dict: dict[str, list] = defaultdict(list)

            # Lists to collect samples for the table
            sample_inputs = []
            sample_outputs = []
            sample_gts = []
            sample_scores = []
            sample_turns = []
            sample_uids = []

            for test_data in self.val_dataloaders[f"model_{i}"]:
                test_batch = DataProto.from_single_dict(test_data)

                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )

                # repeat test batch
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])

                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
                ]
                sample_gts.extend(ground_truths)

                test_gen_batch = self._get_gen_batch(test_batch)
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizers[f"model_{i}"].eos_token_id,
                    "pad_token_id": self.tokenizers[f"model_{i}"].pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
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
                output_texts = [self.tokenizers[f"model_{i}"].decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True

                # evaluate using reward_function
                if self.val_reward_fns is None:
                    raise ValueError("val_reward_fn must be provided for validation.")
                result = self.val_reward_fns[f"model_{i}"](test_batch, return_dict=True)
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
                if "__num_turns__" in test_batch.non_tensor_batch:
                    sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            
            # TODO: update the multi-agent log
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
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}/model_{i}"
                        metric_dict[pfx] = metric_val

            if len(sample_turns) > 0:
                sample_turns = np.concatenate(sample_turns)
                metric_dict[f"val-aux/num_turns/min/model_{i}"] = sample_turns.min()
                metric_dict[f"val-aux/num_turns/max/model_{i}"] = sample_turns.max()
                metric_dict[f"val-aux/num_turns/mean/model_{i}"] = sample_turns.mean()

        return metric_dict
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
    # TODO: update for multi-agent
    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizers.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizers.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
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
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch


    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Per-agent Ray worker groups for ACTORs (actor_pool_0..actor_pool_{A-1})
        3. Worker groups for each role (actor, critic, etc.)
        """
        # TODO: recreate resource pool to manage multi agent workers
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # read multi-agent config
        ma = OmegaConf.select(self.config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))

        # 公共 RayWorkerGroup kwargs
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for i in range(num_agents):
            # 1.1 取资源池：优先 actor_pool_i，若没配则回退到单体 ActorRollout 的池
            pool = self.resource_pool_manager.get_resource_pool(f"agent_pool_{i}")
            if pool is None:
                print(f"Could NOT find agent_pool_{i}!")
                pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # 1.2 拼该 agent 的 actor 配置（在 base 上叠加 multi_agent.agents[i] 覆盖项）
            per_agent_override = OmegaConf.select(self.config, f"multi_agent.agents.{i}") or {}
            per_agent_actor_override = per_agent_override.get("actor", {})
            actor_cfg_i = OmegaConf.merge(deepcopy(self.config.actor_rollout_ref), per_agent_actor_override)
            print(f"actor_model_{i}_path:", actor_cfg_i.model.path)

            # 1.3 构造并 spawn 该 agent 的 WG
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=actor_cfg_i,
                role="actor_rollout",     # 注意：每个组内名字可相同，因为我们不把它们塞进 all_wg 的同一字典
            )
            self.resource_pool_to_cls[pool][f"actor_rollout_{i}"] = actor_rollout_cls

            # initialize critic
            if self.use_critic:
                pool_critic = self.resource_pool_manager.get_resource_pool(f"agent_pool_{i}")
                if pool_critic is None:
                    pool_critic = self.resource_pool_manager.get_resource_pool(Role.Critic)
                per_agent_critic_override = per_agent_override.get("critic", {})
                critic_cfg_i = OmegaConf.merge(deepcopy(self.config.critic), per_agent_critic_override)
                print(f"critic_model_{i}_path:", critic_cfg_i.model.path)

                critic_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.Critic],
                    config=critic_cfg_i 
                )
                self.resource_pool_to_cls[pool_critic][f"critic_{i}"] = critic_cls

            # create reference policy if needed
            if self.use_reference_policy:
                pool_ref = self.resource_pool_manager.get_resource_pool(f"ref_pool_{i}")
                ref_policy_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RefPolicy],
                    config=actor_cfg_i,
                    role="ref",
                )
                self.resource_pool_to_cls[pool_ref][f"ref_{i}"] = ref_policy_cls
            # worker_dict_cls = create_colocated_worker_cls({"actor_rollout": actor_rollout_cls})
            # wg = self.ray_worker_group_cls(
            #     resource_pool=pool,
            #     ray_cls_with_init=worker_dict_cls,
            #     **wg_kwargs,
            # )
            # spawned = wg.spawn(prefix_set=["actor_rollout"])
            # actor_wg = spawned["actor_rollout"]
            # actor_wg.init_model()
            # self.actor_rollout_wgs[f"model_{i}"]=actor_wg


        # create actor and rollout
        # if self.hybrid_engine:
        #     resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        #     actor_rollout_cls = RayClassWithInitArgs(
        #         cls=self.role_worker_mapping[Role.ActorRollout],
        #         config=self.config.actor_rollout_ref,
        #         role="actor_rollout",
        #     )
        #     self.resource_pool_to_cls[resource_pool][f"model_{i}"] = actor_rollout_cls
        # else:
        #     raise NotImplementedError

        # create critic
        # if self.use_critic:
        #     resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        #     critic_cfg = omega_conf_to_dataclass(self.config.critic)
        #     critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
        #     self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls


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

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # print(resource_pool)
            # print(class_dict)
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        
        # get agent_wg
        self.actor_rollout_wgs = {}
        for i in range(num_agents):
            actor_wg = all_wg[f"actor_rollout_{i}"]
            actor_wg.init_model()
            self.actor_rollout_wgs[f"model_{i}"] = actor_wg

        self.critic_wgs = {}
        if self.use_critic:
            for i in range(num_agents):
                critic_wg = all_wg[f"critic_{i}"]
                critic_wg.init_model()
                self.critic_wgs[f"model_{i}"] = critic_wg

        # TODO: ref and rm for each agent
        self.ref_policy_wgs={}
        if self.use_reference_policy and not self.ref_in_actor:
            for i in range(num_agents):
                ref_policy_wg = all_wg[f"ref_{i}"]
                ref_policy_wg.init_model()
                self.ref_policy_wgs[f"model_{i}"] = ref_policy_wg

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_managers={}
            for i in range(num_agents):
                self.async_rollout_managers[f"model_{i}"] = AgentLoopManager(
                    config=self.config, worker_group=self.actor_rollout_wgs[f"model_{i}"], rm_wg=self.rm_wg
                )
    
    # multi-agent
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
        # multi-agent logic
        ma = OmegaConf.select(self.config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))
        for i in range(num_agents):
            agent_path=f"{i}"
            agent_local_path=os.path.join(actor_local_path,agent_path)
            if actor_remote_path is None:
                agent_remote_path = None
            else:
                agent_remote_path=os.path.join(actor_remote_path,agent_path)
            self.actor_rollout_wgs[f"model_{i}"].save_checkpoint(
                agent_local_path, agent_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
            )

            if self.use_critic:
                critic_local_path = os.path.join(local_global_step_folder, "critic",agent_path)
                critic_remote_path = (
                    None
                    if self.config.trainer.default_hdfs_dir is None
                    else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic",agent_path)
                )
                self.critic_wgs[f"model_{i}"].save_checkpoint(
                    critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
                )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        for i in range(num_agents):
            dataloader_local_path = os.path.join(local_global_step_folder, f"data_{i}.pt")
            dataloader_state_dict = self.train_dataloaders[f"model_{i}"].state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
    
    # multi-agent
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
        # multi-agent logic
        ma = OmegaConf.select(self.config, "multi_agent") or {}
        num_agents = int(ma.get("num_agents", 1))
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        for i in range(num_agents):
            agent_path=f"{i}"
            agent_local_path=os.path.join(actor_path, agent_path)
            self.actor_rollout_wgs[f"model_{i}"].load_checkpoint(
                agent_local_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        #load critic
        if self.use_critic:
            for i in range(num_agents):
                agent_path = f"{i}"
                critic_local_path=os.path.join(critic_path, agent_path)
                self.critic_wgs[f"model_{i}"].load_checkpoint(
                    critic_local_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                )

        # load dataloader
        for i in range(num_agents):
            dataloader_local_path = os.path.join(global_step_folder, f"data_{i}.pt")
            if os.path.exists(dataloader_local_path):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloaders[f"model_{i}"].load_state_dict(dataloader_state_dict)
            else:
                print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            ma = OmegaConf.select(self.config, "multi_agent") or {}
            num_agents = int(ma.get("num_agents", 1))
            for i in range(num_agents):
                self.actor_rollout_wgs[f"model_{i}"].start_profile(role="e2e", profile_step=self.global_steps)
                if self.use_critic:
                    self.critic_wgs[f"model_{i}"].start_profile(profile_step=self.global_steps)
                if self.use_reference_policy:
                    self.ref_policy_wgs[f"model_{i}"].start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            ma = OmegaConf.select(self.config, "multi_agent") or {}
            num_agents = int(ma.get("num_agents", 1))
            for i in range(num_agents):
                self.actor_rollout_wgs[f"model_{i}"].stop_profile()
                if self.use_critic:
                    self.critic_wgs[f"model_{i}"].stop_profile()
                if self.use_reference_policy:
                    self.ref_policy_wgs[f"model_{i}"].stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wgs["model_0"].world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _extract_prompts_and_questions(self,batch: DataProto,agent_key):
        import re
        # Step 1: 一次性 decode，避免循环里 decode
        input_ids=batch.batch["input_ids"]
        tokenizer=self.tokenizers[agent_key]
        prompts=tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        questions = []
        system_prompt=""
        # Step 2: 提取 user 和 assistant 之间的内容
        for prompt in prompts:
            # 尝试匹配 role 标记：user ... assistant
            # 不区分大小写，用非贪婪 .*? 提取中间内容
            match = re.search(r"user(.*?)assistant", prompt, flags=re.DOTALL | re.IGNORECASE)
            if match:
                q = match.group(1)
            else:
                # 如果没找到，就返回整个 prompt
                q = prompt
            
            # Step 3: 清理换行符和多余空格
            q = q.replace("\n", " ").strip()
            q = re.sub(r"\s+", " ", q)  # 合并多个空格为1个
            
            questions.append(q)
        # Step 3: extract system prompts
        match = re.search(r"system(.*?)user", prompt, flags=re.DOTALL | re.IGNORECASE)
        if match:
                system_prompt = match.group(1)
        else:
            # 如果没找到，就返回整个 prompt
            system_prompt = prompt
            
            # Step 3: 清理换行符和多余空格
            system_prompt = system_prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
            system_prompt = re.sub(r"\s+", " ", system_prompt)  # 合并多个空格为1个
        return questions, system_prompt
    
    def _build_input_ids_from_histories(self,system_prompt,discussion_prompt,questions,histories,batch:DataProto,agent_key,max_history_tokens):
        prompts = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "user","content": discussion_prompt+hist}
            ]
            for q,hist in zip(questions,histories)
        ]
        tokenizer=self.tokenizers[agent_key]
        raw_prompts = [
            tokenizer.apply_chat_template(
                p,
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]
        input_ids_list = []
        attention_mask_list = []
        for rp in raw_prompts:
            model_inputs = tokenizer(rp, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            truncation=self.config.data.get("truncation", "error")
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_history_tokens,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        batch.batch["input_ids"] = input_ids
        batch.batch["attention_mask"] = attention_mask
        batch.non_tensor_batch.pop("raw_prompt_ids")
        return raw_prompts
    
    def _build_input_ids_adversary(self,system_prompt,questions,batch:DataProto,agent_key,max_history_tokens):
        prompts = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            for q in questions
        ]
        tokenizer=self.tokenizers[agent_key]
        raw_prompts = [
            tokenizer.apply_chat_template(
                p,
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]
        input_ids_list = []
        attention_mask_list = []
        for rp in raw_prompts:
            model_inputs = tokenizer(rp, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            truncation=self.config.data.get("truncation", "error")
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_history_tokens,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        batch.batch["input_ids"] = input_ids
        batch.batch["attention_mask"] = attention_mask
        batch.non_tensor_batch.pop("raw_prompt_ids")
        return raw_prompts
    def _single_agent_rollout(self,agent_idx,agent_key,batch_dict,histories,r,timing_raw,round_agent_metrics):
        # Per-agent bookkeeping and metrics
        metrics=round_agent_metrics[r][agent_idx]
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # if not the first round
        if r>0:
            questions, _ =self._extract_prompts_and_questions(batch,agent_key)
            system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please collaborate with the other agent to help the user. "
            discussion_prompt=f"The discussion history is as follows:"
            chat_prompts=self._build_input_ids_from_histories(system_prompt,discussion_prompt,questions,histories,batch,agent_key,max_history_tokens=4096)
        # add uid to batch
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
        
        gen_batch = self._get_gen_batch(batch)
        
        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        reward_extra_infos_dict: dict = {}
        with marked_timer("step", timing_raw):
            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wgs[agent_key].generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_managers[agent_key].generate_sequences(gen_batch)

                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

            # TODO: figure out the meaning of this step (calculate adv)
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                reward_fn = None
                if isinstance(self.reward_fns, dict):
                    reward_fn = self.reward_fns.get(agent_key, None)
                elif hasattr(self, "reward_fn"):
                    reward_fn = self.reward_fn
                if reward_fn is None:
                    raise ValueError("A reward_fn is required for REMAX advantage estimation in MAPPO.")
            
                with marked_timer("gen_max", timing_raw, color="purple"):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    if not self.async_rollout_mode:
                        gen_baseline_output = self.actor_rollout_wgs[agent_key].generate_sequences(
                            gen_baseline_batch
                        )
                    else:
                        gen_baseline_output = self.async_rollout_managers[agent_key].generate_sequences(
                            gen_baseline_batch
                        )
                    batch = batch.union(gen_baseline_output)
                    # TODO: why do we need a baseline
                    reward_baseline_tensor = reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            if self.config.trainer.balance_batch:
                self._balance_batch(
                    batch,
                    metrics=metrics,
                    logging_prefix=f"global_seqlen_agent{agent_idx}",
                )

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("reward", timing_raw, color="yellow"):
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_fn = None
                if isinstance(self.reward_fns, dict):
                    reward_fn = self.reward_fns.get(agent_key, None)
                elif hasattr(self, "reward_fn"):
                    reward_fn = self.reward_fn

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)

            # recompute old_log_probs
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wgs[agent_key].compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(
                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                )
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with marked_timer("ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wgs[agent_key].compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wgs[agent_key].compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wgs[agent_key].compute_values(batch)
                    batch = batch.union(values)          

            with marked_timer("adv", timing_raw, color="brown"):
                # we combine with rule-based rm
                reward_extra_infos_dict = reward_extra_infos_dict or {}
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            
                # if self.config.algorithm.use_kl_in_reward:
                #     batch, kl_metrics = apply_kl_penalty(
                #         batch,
                #         kl_ctrl=self.kl_ctrl_in_reward,
                #         kl_penalty=self.config.algorithm.kl_penalty,
                #     )
                #     metrics.update(kl_metrics)
                # else:
                #     batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        # update round history
        resp_ids = batch.batch["responses"]
        resp_texts = self.tokenizers[agent_key].batch_decode(resp_ids,skip_special_tokens=True)
        del gen_batch_output
        return resp_texts,batch
    def _run_single_agent(self,agent_idx,agent_key,batch_dict,histories,r,metrics,timing_raw,round_agent_batches,step_durations):
        def _with_prefix(prefix: str, metric_dict: dict) -> dict:
            """Prefix metrics with agent identifier to avoid collisions."""
            return {f"{prefix}/{k}": v for k, v in metric_dict.items()}
        # Per-agent bookkeeping and metrics
        agent_metrics = {}
        agent_timing: dict[str, float] = {}
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # if not the first round
        if r>0:
            questions, _ =self._extract_prompts_and_questions(batch,agent_key)
            system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please collaborate with the other agent to help the user. Provide a well−reasoned response that not only considers your own previous solution but also takes into account answers from other agents. If you believe your previous answer was incorrect, feel free to revise it. Ensure that your explanation clearly justifies your final answer. Please maintain your answer with very simple reasoning."
            discussion_prompt=f"The discussion history is as follows:"
            chat_prompts=self._build_input_ids_from_histories(system_prompt,discussion_prompt,questions,histories,batch,agent_key,max_history_tokens=4096)
        # add uid to batch
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
        
        gen_batch = self._get_gen_batch(batch)
        
        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        reward_extra_infos_dict: dict = {}
        with marked_timer("step", agent_timing):
            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wgs[agent_key].generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_managers[agent_key].generate_sequences(gen_batch)

                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

            # TODO: figure out the meaning of this step (calculate adv)
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                reward_fn = None
                if isinstance(self.reward_fns, dict):
                    reward_fn = self.reward_fns.get(agent_key, None)
                elif hasattr(self, "reward_fn"):
                    reward_fn = self.reward_fn
                if reward_fn is None:
                    raise ValueError("A reward_fn is required for REMAX advantage estimation in MAPPO.")
            
                with marked_timer("gen_max", agent_timing, color="purple"):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    if not self.async_rollout_mode:
                        gen_baseline_output = self.actor_rollout_wgs[agent_key].generate_sequences(
                            gen_baseline_batch
                        )
                    else:
                        gen_baseline_output = self.async_rollout_managers[agent_key].generate_sequences(
                            gen_baseline_batch
                        )
                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            if self.config.trainer.balance_batch:
                self._balance_batch(
                    batch,
                    metrics=agent_metrics,
                    logging_prefix=f"global_seqlen_agent{agent_idx}",
                )

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("reward", agent_timing, color="yellow"):
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_fn = None
                if isinstance(self.reward_fns, dict):
                    reward_fn = self.reward_fns.get(agent_key, None)
                elif hasattr(self, "reward_fn"):
                    reward_fn = self.reward_fn

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)

            # recompute old_log_probs
            with marked_timer("old_log_prob", agent_timing, color="blue"):
                old_log_prob = self.actor_rollout_wgs[agent_key].compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(
                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                )
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                agent_metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with marked_timer("ref", agent_timing, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wgs[agent_key].compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wgs[agent_key].compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with marked_timer("values", agent_timing, color="cyan"):
                    values = self.critic_wgs[agent_key].compute_values(batch)
                    batch = batch.union(values)          

            with marked_timer("adv", agent_timing, color="brown"):
                # we combine with rule-based rm
                reward_extra_infos_dict = reward_extra_infos_dict or {}
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl_in_reward,
                        kl_penalty=self.config.algorithm.kl_penalty,
                    )
                    agent_metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                
                # compute advantages, executed on the driver process
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
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
                    critic_output = self.critic_wgs[agent_key].update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                agent_metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with marked_timer("update_actor", agent_timing, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wgs[agent_key].update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                agent_metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with marked_timer("dump_rollout_generations", agent_timing, color="green"):
                    agent_tokenizer = (
                        self.tokenizers[agent_key] if isinstance(self.tokenizers, dict) else self.tokenizers
                    )
                    inputs = agent_tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = agent_tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    sample_gts = [
                        item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                        for item in batch
                    ]
                    reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
                    if "request_id" in batch.non_tensor_batch:
                        reward_extra_infos_to_dump.setdefault(
                            "request_id",
                            batch.non_tensor_batch["request_id"].tolist(),
                        )
                    self._dump_generations(
                        inputs=inputs,
                        outputs=outputs,
                        gts=sample_gts,
                        scores=scores,
                        reward_extra_infos_dict=reward_extra_infos_to_dump,
                        dump_path=rollout_data_dir,
                    )
                    

        # collect metrics per agent with prefix
        agent_metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        agent_metrics.update(compute_timing_metrics(batch=batch, timing_raw=agent_timing))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        agent_metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=agent_timing, n_gpus=n_gpus)
        )
        metrics.update(_with_prefix(f"agent{agent_idx}", agent_metrics))
        round_agent_batches[agent_key] = batch

        if "step" in agent_timing:
            step_durations.append(agent_timing["step"])
        # update round history
        resp_ids = batch.batch["responses"]
        resp_texts = self.tokenizers[agent_key].batch_decode(resp_ids,skip_special_tokens=True)
        del gen_batch_output, batch

        return resp_texts, agent_metrics
    def back_propogate_reward(self,num_rounds,num_agents,round_agent_batches,gamma):
        # ---- Step 0: 预先把所有 reward.sum(-1) 缓存起来（避免重复计算） ----
        # rewards[r][a] = [B] reward vector
        rewards = [
            [
                round_agent_batches[r][a].batch["token_level_scores"].sum(-1)
                for a in range(num_agents)
            ]
            for r in range(num_rounds)
        ]

        # ---- Step 1: backward pass ----
        coef = gamma / num_agents    # constant
        for r in range(num_rounds - 2, 0, -1):
            # precompute the future-sum across all agents ONCE
            # future_sum = Σ_a reward_{r+1}^{(a)}
            future_sum = sum(rewards[r+1])       # shape [B]

            # for each agent, compute reward_to_go (no more inner sum loop)
            for a in range(num_agents):
                base_reward = rewards[r][a]      # shape [B]
                rewards[r][a] = base_reward + coef * future_sum

                # write back only to last token 
                batch = round_agent_batches[r][a]
                scores = batch.batch["token_level_scores"]                 # [B, T]
                rmask  = batch.batch["response_mask"].bool()               # [B, T]

                B, T = scores.shape
                rows = torch.arange(B, device=scores.device)

                # last index where response_mask == 1
                idx = torch.arange(T, device=scores.device).unsqueeze(0).expand(B, T)
                last_resp_idx = (idx * rmask.long()).max(dim=1).values  # [B]

                # safety: if some sample has no response tokens, do nothing for that sample
                has_resp = rmask.any(dim=1)
                scores[rows[has_resp], last_resp_idx[has_resp]] = rewards[r][a][has_resp].to(scores.dtype)
    
    def _update_critic(self,r,agent_idx,agent_key,round_agent_batches,timing_raw,round_agent_metrics):
        with marked_timer("update_critic", timing_raw, color="pink"):
            batch=round_agent_batches[r][agent_idx]
            critic_output = self.critic_wgs[agent_key].update_critic(batch)
        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
        round_agent_metrics[r][agent_idx].update(critic_output_metrics)

    def _update_actor(self,r,agent_idx,agent_key,round_agent_batches,timing_raw,round_agent_metrics):
        with marked_timer("update_actor", timing_raw, color="red"):
            batch=round_agent_batches[r][agent_idx]
            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
            actor_output = self.actor_rollout_wgs[agent_key].update_actor(batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        round_agent_metrics[r][agent_idx].update(actor_output_metrics)

    def _update_metrics(self,r,agent_idx,round_agent_batches,round_agent_metrics,timing_raw,metrics):
        def _with_prefix(prefix: str, metric_dict: dict) -> dict:
            """Prefix metrics with agent identifier to avoid collisions."""
            return {f"{prefix}/{k}": v for k, v in metric_dict.items()}
        batch=round_agent_batches[r][agent_idx]
        agent_metrics=round_agent_metrics[r][agent_idx]
        agent_metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        agent_metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        agent_metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
        )
        metrics.update(_with_prefix(f"agent{agent_idx}/round{r}", agent_metrics))
     
    def mappo_fit(self):
        """Run multi-agent PPO algorithm."""
        from verl.utils.tracking import Tracking

        def _with_prefix(prefix: str, metric_dict: dict) -> dict:
            """Prefix metrics with agent identifier to avoid collisions."""
            return {f"{prefix}/{k}": v for k, v in metric_dict.items()}
        
        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        num_agents = int(ma.get("num_agents", 1))
        num_rounds = int(ma.get("num_rounds", 1))
        agent_keys = list(self.train_dataloaders.keys())

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fns is not None and self.config.trainer.get("val_before_train", True):
            # TODO: check the logic of _validate()
            # TODO: multi-agent
            val_metrics = self._multi_agent_validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

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
            for batch_tuple in zip(*(self.train_dataloaders[k] for k in agent_keys)):
                round_agent_batches=[[None for _ in range(num_agents)] for _ in range(num_rounds)]
                metrics = {}
                timing_raw = {}
                round_agent_metrics=[[{} for _ in range(num_agents)] for _ in range(num_rounds)]

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                is_last_step = self.global_steps >= self.total_training_steps
                step_durations = []

                # discussion setting
                batch_size=self.config.data.train_batch_size
                histories=[""]*batch_size
                # rollout
                for r in range(num_rounds):
                    this_round = [""] * batch_size
                    from concurrent.futures import ThreadPoolExecutor
                    futures=[]
                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx,agent_key in enumerate(agent_keys):
                            batch_dict=batch_tuple[agent_idx]
                            futures.append(
                                executor.submit(
                                    self._single_agent_rollout,
                                    agent_idx,
                                    agent_key,
                                    batch_dict,
                                    histories,
                                    r,
                                    timing_raw,
                                    round_agent_metrics
                                )
                            )               
                    results = [f.result() for f in futures]
                    for agent_idx, (resp_texts, batch) in enumerate(results):
                        this_round = [
                            old + f"\nAgent {agent_idx}: {new}"
                            for old, new in zip(this_round, resp_texts)
                        ]
                        round_agent_batches[r][agent_idx]=batch
                    
                    histories[:]=this_round
                    # histories[:] = [
                    #     old + f"\n[Round {r}]: {new}"
                    #     for old, new in zip(histories, this_round)
                    # ]
                    # DEBUG
                    # if r==num_rounds-1:
                    #     print(histories[0])

                # backpropagate reward
                self.back_propogate_reward(num_rounds,num_agents,round_agent_batches,gamma=1)
                
                # apply kl in reward
                if self.config.algorithm.use_kl_in_reward:
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx in range(num_agents):
                                batch=round_agent_batches[r][agent_idx]
                                futures.append(
                                    executor.submit(
                                        apply_kl_penalty,
                                        batch,
                                        self.kl_ctrl_in_reward,
                                        self.config.algorithm.kl_penalty
                                    )
                                )
                        results = [f.result() for f in futures]
                        for agent_idx,(batch,kl_metrics) in enumerate(results):
                            round_agent_metrics[r][agent_idx].update(kl_metrics)
                else:
                    for r in range(num_rounds):
                        for agent_idx in range(num_agents):
                            batch=round_agent_batches[r][agent_idx]
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                # compute advantages, executed on the driver process
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                for r in range(num_rounds):
                    futures=[]
                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx in range(num_agents):
                            batch=round_agent_batches[r][agent_idx]
                            futures.append(
                                executor.submit(
                                    compute_advantage,
                                    batch,
                                    self.config.algorithm.adv_estimator,
                                    self.config.algorithm.gamma,
                                    self.config.algorithm.lam,
                                    self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo,
                                    self.config.algorithm
                                )
                            )
                    results = [f.result() for f in futures]
                    for agent_idx,batch in enumerate(results):
                        round_agent_batches[r][agent_idx]=batch
                # update critic
                if self.use_critic:
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx,agent_key in enumerate(agent_keys):
                                futures.append(
                                    executor.submit(
                                        self._update_critic,
                                        r,
                                        agent_idx,
                                        agent_key,
                                        round_agent_batches,
                                        timing_raw,
                                        round_agent_metrics
                                    )
                                )
                        results = [f.result() for f in futures]
                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    for r in range(num_rounds):
                        futures=[]
                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx,agent_key in enumerate(agent_keys):
                                futures.append(
                                    executor.submit(
                                        self._update_actor,
                                        r,
                                        agent_idx,
                                        agent_key,
                                        round_agent_batches,
                                        timing_raw,
                                        round_agent_metrics
                                    )
                                )
                        results = [f.result() for f in futures]
                # collect metrics per agent with prefix
                for r in range(num_rounds):
                    futures=[]
                    with ThreadPoolExecutor(max_workers=num_agents) as executor:        
                        for agent_idx in range(num_agents):
                            futures.append(
                                executor.submit(
                                    self._update_metrics,
                                    r,
                                    agent_idx,
                                    round_agent_batches,
                                    round_agent_metrics,
                                    timing_raw,
                                    metrics
                                )
                            )
                        results = [f.result() for f in futures]
                # validate
                if (
                    self.val_reward_fns is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._multi_agent_validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )

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

                if step_durations:
                    steps_duration = max(step_durations)
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    for agent_key in agent_keys:
                        self.actor_rollout_wgs[agent_key].dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                # trigger dataset hooks if present
                for agent_idx, agent_key in enumerate(agent_keys):
                    train_dataset = self.train_datasets.get(agent_key, None)
                    if hasattr(train_dataset, "on_batch_end"):
                        train_dataset.on_batch_end(batch=round_agent_batches[0][agent_idx])
    # def mappo_fit(self):
    #     """Run multi-agent PPO algorithm."""
    #     from verl.utils.tracking import Tracking

    #     ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
    #     num_agents = int(ma.get("num_agents", 1))
    #     agent_keys = list(self.train_dataloaders.keys())

    #     logger = Tracking(
    #         project_name=self.config.trainer.project_name,
    #         experiment_name=self.config.trainer.experiment_name,
    #         default_backend=self.config.trainer.logger,
    #         config=OmegaConf.to_container(self.config, resolve=True),
    #     )
    #     def _with_prefix(prefix: str, metric_dict: dict) -> dict:
    #         """Prefix metrics with agent identifier to avoid collisions."""
    #         return {f"{prefix}/{k}": v for k, v in metric_dict.items()}

    #     self.global_steps = 0

    #     # load checkpoint before doing anything
    #     self._load_checkpoint()

    #     # perform validation before training
    #     if self.val_reward_fns is not None and self.config.trainer.get("val_before_train", True):
    #         # TODO: check the logic of _validate()
    #         val_metrics = self._validate()
    #         assert val_metrics, f"{val_metrics=}"
    #         pprint(f"Initial validation metrics: {val_metrics}")
    #         logger.log(data=val_metrics, step=self.global_steps)
    #         if self.config.trainer.get("val_only", False):
    #             return

    #     # add tqdm
    #     progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    #     # we start from step 1
    #     self.global_steps += 1
    #     last_val_metrics = None
    #     self.max_steps_duration = 0

    #     prev_step_profile = False
    #     curr_step_profile = (
    #         self.global_steps in self.config.global_profiler.steps
    #         if self.config.global_profiler.steps is not None
    #         else False
    #     )
    #     next_step_profile = False

    #     for epoch in range(self.config.trainer.total_epochs):
    #         for batch_tuple in zip(*(self.train_dataloaders[k] for k in agent_keys)):
    #             round_agent_batches: dict[str, DataProto] = {}
    #             metrics = {}
    #             timing_raw = {}

    #             with marked_timer("start_profile", timing_raw):
    #                 self._start_profiling(
    #                     not prev_step_profile and curr_step_profile
    #                     if self.config.global_profiler.profile_continuous_steps
    #                     else curr_step_profile
    #                 )
    #             is_last_step = self.global_steps >= self.total_training_steps
    #             step_durations = []

    #             # discussion setting
    #             batch_size=self.config.data.train_batch_size
    #             histories=[""]*batch_size
    #             rounds=3
    #             for r in range(rounds):
    #                 this_round = [""] * batch_size
    #                 for agent_idx,agent_key in enumerate(agent_keys):
    #                     # Per-agent bookkeeping and metrics
    #                     agent_metrics = {}
    #                     agent_timing: dict[str, float] = {}

    #                     batch_dict=batch_tuple[agent_idx]
    #                     batch: DataProto = DataProto.from_single_dict(batch_dict)
    #                     # if not the first round
    #                     if r>0:
    #                         questions, _ =self._extract_prompts_and_questions(batch,agent_key)
    #                         system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please collaborate with the other agent to help the user. Provide a well−reasoned response that not only considers your own previous solution but also takes into account answers from other agents. If you believe your previous answer was incorrect, feel free to revise it. However, avoid repeating the same answer you or other agents have already provided."
    #                         discussion_prompt=f"The discussion history is as follows, where you are Agent {agent_idx}:"
    #                         self._build_input_ids_from_histories(system_prompt,discussion_prompt,questions,histories,batch,agent_key,max_history_tokens=4096)
    #                     # add uid to batch
    #                     if "uid" not in batch.non_tensor_batch:
    #                         batch.non_tensor_batch["uid"] = np.array(
    #                             [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    #                         )
                        
    #                     gen_batch = self._get_gen_batch(batch)
                        
    #                     # pass global_steps to trace
    #                     gen_batch.meta_info["global_steps"] = self.global_steps
    #                     gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

    #                     reward_extra_infos_dict: dict = {}
    #                     with marked_timer("step", agent_timing):
    #                         # generate a batch
    #                         with marked_timer("gen", timing_raw, color="red"):
    #                             if not self.async_rollout_mode:
    #                                 gen_batch_output = self.actor_rollout_wgs[agent_key].generate_sequences(gen_batch)
    #                             else:
    #                                 gen_batch_output = self.async_rollout_managers[agent_key].generate_sequences(gen_batch)

    #                             timing_raw.update(gen_batch_output.meta_info["timing"])
    #                             gen_batch_output.meta_info.pop("timing", None)

    #                         # TODO: figure out the meaning of this step (calculate adv)
    #                         if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
    #                             reward_fn = None
    #                             if isinstance(self.reward_fns, dict):
    #                                 reward_fn = self.reward_fns.get(agent_key, None)
    #                             elif hasattr(self, "reward_fn"):
    #                                 reward_fn = self.reward_fn
    #                             if reward_fn is None:
    #                                 raise ValueError("A reward_fn is required for REMAX advantage estimation in MAPPO.")
                            
    #                             with marked_timer("gen_max", agent_timing, color="purple"):
    #                                 gen_baseline_batch = deepcopy(gen_batch)
    #                                 gen_baseline_batch.meta_info["do_sample"] = False
    #                                 if not self.async_rollout_mode:
    #                                     gen_baseline_output = self.actor_rollout_wgs[agent_key].generate_sequences(
    #                                         gen_baseline_batch
    #                                     )
    #                                 else:
    #                                     gen_baseline_output = self.async_rollout_managers[agent_key].generate_sequences(
    #                                         gen_baseline_batch
    #                                     )
    #                                 batch = batch.union(gen_baseline_output)
    #                                 reward_baseline_tensor = reward_fn(batch)
    #                                 reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

    #                                 batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

    #                                 batch.batch["reward_baselines"] = reward_baseline_tensor

    #                                 del gen_baseline_batch, gen_baseline_output

    #                         # repeat to align with repeated responses in rollout
    #                         batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
    #                         batch = batch.union(gen_batch_output)

    #                         if "response_mask" not in batch.batch.keys():
    #                             batch.batch["response_mask"] = compute_response_mask(batch)

    #                         if self.config.trainer.balance_batch:
    #                             self._balance_batch(
    #                                 batch,
    #                                 metrics=agent_metrics,
    #                                 logging_prefix=f"global_seqlen_agent{agent_idx}",
    #                             )

    #                         # compute global_valid tokens
    #                         batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    #                         with marked_timer("reward", agent_timing, color="yellow"):
    #                             if self.use_rm and "rm_scores" not in batch.batch.keys():
    #                                 reward_tensor = self.rm_wg.compute_rm_score(batch)
    #                                 batch = batch.union(reward_tensor)

    #                             reward_fn = None
    #                             if isinstance(self.reward_fns, dict):
    #                                 reward_fn = self.reward_fns.get(agent_key, None)
    #                             elif hasattr(self, "reward_fn"):
    #                                 reward_fn = self.reward_fn

    #                             if self.config.reward_model.launch_reward_fn_async:
    #                                 future_reward = compute_reward_async.remote(data=batch, reward_fn=reward_fn)
    #                             else:
    #                                 reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)

    #                         # recompute old_log_probs
    #                         with marked_timer("old_log_prob", agent_timing, color="blue"):
    #                             old_log_prob = self.actor_rollout_wgs[agent_key].compute_log_prob(batch)
    #                             entropys = old_log_prob.batch["entropys"]
    #                             response_masks = batch.batch["response_mask"]
    #                             loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    #                             entropy_agg = agg_loss(
    #                                 loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
    #                             )
    #                             old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
    #                             agent_metrics.update(old_log_prob_metrics)
    #                             old_log_prob.batch.pop("entropys")
    #                             batch = batch.union(old_log_prob)

    #                         if self.use_reference_policy:
    #                             # compute reference log_prob
    #                             with marked_timer("ref", agent_timing, color="olive"):
    #                                 if not self.ref_in_actor:
    #                                     ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    #                                 else:
    #                                     ref_log_prob = self.actor_rollout_wgs[agent_key].compute_ref_log_prob(batch)
    #                                 batch = batch.union(ref_log_prob)

    #                         # compute values
    #                         if self.use_critic:
    #                             with marked_timer("values", agent_timing, color="cyan"):
    #                                 values = self.critic_wgs[agent_key].compute_values(batch)
    #                                 batch = batch.union(values)          

    #                         with marked_timer("adv", agent_timing, color="brown"):
    #                             # we combine with rule-based rm
    #                             reward_extra_infos_dict = reward_extra_infos_dict or {}
    #                             if self.config.reward_model.launch_reward_fn_async:
    #                                 reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
    #                             batch.batch["token_level_scores"] = reward_tensor

    #                             if reward_extra_infos_dict:
    #                                 batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            
    #                             if self.config.algorithm.use_kl_in_reward:
    #                                 batch, kl_metrics = apply_kl_penalty(
    #                                     batch,
    #                                     kl_ctrl=self.kl_ctrl_in_reward,
    #                                     kl_penalty=self.config.algorithm.kl_penalty,
    #                                 )
    #                                 agent_metrics.update(kl_metrics)
    #                             else:
    #                                 batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                                
    #                             # compute advantages, executed on the driver process
    #                             norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
    #                             batch = compute_advantage(
    #                                 batch,
    #                                 adv_estimator=self.config.algorithm.adv_estimator,
    #                                 gamma=self.config.algorithm.gamma,
    #                                 lam=self.config.algorithm.lam,
    #                                 num_repeat=self.config.actor_rollout_ref.rollout.n,
    #                                 norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    #                                 config=self.config.algorithm,
    #                             )
    #                         # update critic
    #                         if self.use_critic:
    #                             with marked_timer("update_critic", timing_raw, color="pink"):
    #                                 critic_output = self.critic_wgs[agent_key].update_critic(batch)
    #                             critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    #                             agent_metrics.update(critic_output_metrics)

    #                         # implement critic warmup
    #                         if self.config.trainer.critic_warmup <= self.global_steps:
    #                             # update actor
    #                             with marked_timer("update_actor", agent_timing, color="red"):
    #                                 batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
    #                                 actor_output = self.actor_rollout_wgs[agent_key].update_actor(batch)
    #                             actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    #                             agent_metrics.update(actor_output_metrics)

    #                         # Log rollout generations if enabled
    #                         rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
    #                         if rollout_data_dir:
    #                             with marked_timer("dump_rollout_generations", agent_timing, color="green"):
    #                                 agent_tokenizer = (
    #                                     self.tokenizers[agent_key] if isinstance(self.tokenizers, dict) else self.tokenizers
    #                                 )
    #                                 inputs = agent_tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    #                                 outputs = agent_tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    #                                 scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
    #                                 sample_gts = [
    #                                     item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
    #                                     for item in batch
    #                                 ]
    #                                 reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
    #                                 if "request_id" in batch.non_tensor_batch:
    #                                     reward_extra_infos_to_dump.setdefault(
    #                                         "request_id",
    #                                         batch.non_tensor_batch["request_id"].tolist(),
    #                                     )
    #                                 self._dump_generations(
    #                                     inputs=inputs,
    #                                     outputs=outputs,
    #                                     gts=sample_gts,
    #                                     scores=scores,
    #                                     reward_extra_infos_dict=reward_extra_infos_to_dump,
    #                                     dump_path=rollout_data_dir,
    #                                 )
                                    

    #                     # collect metrics per agent with prefix
    #                     agent_metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
    #                     agent_metrics.update(compute_timing_metrics(batch=batch, timing_raw=agent_timing))
    #                     n_gpus = self.resource_pool_manager.get_n_gpus()
    #                     agent_metrics.update(
    #                         compute_throughout_metrics(batch=batch, timing_raw=agent_timing, n_gpus=n_gpus)
    #                     )
    #                     metrics.update(_with_prefix(f"agent{agent_idx}", agent_metrics))
    #                     round_agent_batches[agent_key] = batch

    #                     if "step" in agent_timing:
    #                         step_durations.append(agent_timing["step"])

    #                     # update round history
    #                     resp_ids = batch.batch["responses"]
    #                     resp_texts = self.tokenizers[agent_key].batch_decode(resp_ids,skip_special_tokens=True)
    #                     this_round = [
    #                         old + f"\nAgent {agent_idx}: {new}" 
    #                         for old, new in zip(this_round, resp_texts)
    #                     ]
    #                     del gen_batch_output, batch

    #                 # update global history
    #                 histories[:] = [
    #                     old + f"\n[Round {r}]: {new}"
    #                     for old, new in zip(histories, this_round)
    #                 ]
    #                 # DEBUG
    #                 if r==rounds-1:
    #                     print(histories[0])

    #             # validate
    #             if (
    #                 self.val_reward_fns is not None
    #                 and self.config.trainer.test_freq > 0
    #                 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
    #             ):
    #                 with marked_timer("testing", timing_raw, color="green"):
    #                     val_metrics: dict = self._validate()
    #                     if is_last_step:
    #                         last_val_metrics = val_metrics
    #                 metrics.update(val_metrics)

    #             # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
    #             esi_close_to_expiration = should_save_ckpt_esi(
    #                 max_steps_duration=self.max_steps_duration,
    #                 redundant_time=self.config.trainer.esi_redundant_time,
    #             )

    #             if self.config.trainer.save_freq > 0 and (
    #                 is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
    #             ):
    #                 if esi_close_to_expiration:
    #                     print("Force saving checkpoint: ESI instance expiration approaching.")
    #                 with marked_timer("save_checkpoint", timing_raw, color="green"):
    #                     self._save_checkpoint()

    #             with marked_timer("stop_profile", timing_raw):
    #                 next_step_profile = (
    #                     self.global_steps + 1 in self.config.global_profiler.steps
    #                     if self.config.global_profiler.steps is not None
    #                     else False
    #                 )
    #                 self._stop_profiling(
    #                     curr_step_profile and not next_step_profile
    #                     if self.config.global_profiler.profile_continuous_steps
    #                     else curr_step_profile
    #                 )
    #                 prev_step_profile = curr_step_profile
    #                 curr_step_profile = next_step_profile

    #             if step_durations:
    #                 steps_duration = max(step_durations)
    #                 self.max_steps_duration = max(self.max_steps_duration, steps_duration)
    #             # training metrics
    #             metrics.update(
    #                 {
    #                     "training/global_step": self.global_steps,
    #                     "training/epoch": epoch,
    #                 }
    #             )
    #             # TODO: make a canonical logger that supports various backend
    #             logger.log(data=metrics, step=self.global_steps)

    #             progress_bar.update(1)
    #             self.global_steps += 1

    #             if (
    #                 hasattr(self.config.actor_rollout_ref.actor, "profiler")
    #                 and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
    #             ):
    #                 for agent_key in agent_keys:
    #                     self.actor_rollout_wgs[agent_key].dump_memory_snapshot(
    #                         tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
    #                     )

    #             if is_last_step:
    #                 pprint(f"Final validation metrics: {last_val_metrics}")
    #                 progress_bar.close()
    #                 return

    #             # trigger dataset hooks if present
    #             for agent_key in agent_keys:
    #                 train_dataset = self.train_datasets.get(agent_key, None)
    #                 if hasattr(train_dataset, "on_batch_end"):
    #                     train_dataset.on_batch_end(batch=round_agent_batches[agent_key])

class RayZOTrainer:
    """Distributed zo trainer using Ray."""

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
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        log_dir = self.config.trainer.default_local_dir or "."
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.zo_accuracy_log_path = os.path.join(log_dir, "zo_accuracy.csv")
        self._init_accuracy_log()

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: change to default model path
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
        # load actor
        print(f"[INFO] Loading actor checkpoint from: {actor_path}")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")


    def _init_accuracy_log(self):
        log_dir = os.path.dirname(self.zo_accuracy_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(self.zo_accuracy_log_path):
            with open(self.zo_accuracy_log_path, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["global_step", "accuracy"])


    def _append_accuracy_metric(self, global_step: int, accuracy: float):
        with open(self.zo_accuracy_log_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([global_step, accuracy])


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
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "dialogue_history"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


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
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
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

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
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


    def truncate_history(self, history: str, max_len: int) -> str:
        """truncate history"""
        ids = self.tokenizer.encode(history, add_special_tokens=False)
        if len(ids) <= max_len:
            return history
        ids = ids[-max_len:]  
        return self.tokenizer.decode(ids, skip_special_tokens=True)


    def init_workers(self):
        """Init actor rollout worker (multi-agent setting).
        
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """

        # resource pool
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # actor rollout
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        # TODO: check RayClassWithInitArgs, whether we create the models here.
        # TODO: update the names to agent_idx
        # The config is saved in actor_rollout_cls
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls


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

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )
    # TODO: Implement the update logic
    def zo_step(self,batch,timing_raw):
        # 1. sample the random seed for sampling z
        zo_random_seed = np.random.randint(1000000000)
        # 2. zo_eps
        zo_eps=0.01
        # 3. positive perturbation
        batch_copy=deepcopy(batch)
        self.actor_rollout_wg.zo_perturb_parameters(zo_random_seed,scaling_factor=1,zo_eps=zo_eps)
        loss1=self.zo_forward(batch_copy,timing_raw)
        del batch_copy
        # 4. negative perturbation
        batch_copy=deepcopy(batch)
        self.actor_rollout_wg.zo_perturb_parameters(zo_random_seed,scaling_factor=-2,zo_eps=zo_eps)
        loss2 = self.zo_forward(batch_copy,timing_raw)
        del batch_copy
        # 5. get gradients
        projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

        # 6. set model back to the un-perturbed version
        self.actor_rollout_wg.zo_perturb_parameters(zo_random_seed,scaling_factor=1,zo_eps=zo_eps)

        # 7. log loss info
        print(f"Global steps: {self.global_steps}, accuracy: -{loss1}, -{loss2}")
        return projected_grad, zo_random_seed
    
    def zo_update(self,projected_grad,lora_model,zo_random_seed,weight_decay=0):
        # Reset the random seed for sampling zs
        torch.manual_seed(zo_random_seed)
        named_parameters_to_optim = [
            (name, param)
            for name, param in lora_model.named_parameters()
            if param.requires_grad
        ]
        for name, param in named_parameters_to_optim:
            # resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - lora_model._get_learning_rate() * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lora_model._get_learning_rate() * (projected_grad * z)


    def zo_forward(self,batch,timing_raw):
        # rollout
        gen_batch = self._get_gen_batch(batch)
        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

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
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)
            # Debug
            # resp_ids = batch.batch["responses"]
            # responses = self.tokenizer.batch_decode(resp_ids,skip_special_tokens=True)
            # print(responses[0])

            with marked_timer("reward", timing_raw, color="yellow"):
                # compute reward model score
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            scores = reward_tensor.sum(-1).cpu().tolist()
            loss=-np.mean(scores)

            return loss

    def zo_perturb_parameters(self,lora_model,zo_eps,random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        # add self.random_seed
        torch.manual_seed(random_seed if random_seed is not None else 42)
        named_parameters_to_optim = [
            (name, param)
            for name, param in lora_model.named_parameters()
            if param.requires_grad
        ]
        for name, param in named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data.add_(scaling_factor * z * zo_eps)

    def _get_learning_rate(self):
        base_lr=1e-4
        lr=base_lr/np.sqrt(self.global_steps)
        if lr <= 5e-5:
            lr=5e-5
        return lr


    
    def load_lora_model(self):
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        # 1. load the base model
        model_path=self.config.actor_rollout_ref.model.path
        base_model=AutoModelForCausalLM.from_pretrained(model_path)
        # 2. free the parameters of the base model:
        for param in base_model.parameters():
            param.requires_grad = False
        # 3. load lora adapter
        lora_path=self.config.actor_rollout_ref.model.lora_adapter_path
        lora_model=PeftModel.from_pretrained(base_model, lora_path, adapter_name="adapter_saved")
        return lora_model

    def debug_generate_with_lora(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
    ):
        """Debug helper: load the LoRA model and generate from provided `input_ids`.

        This bypasses rollout workers so you can quickly verify the adapter output.
        Returns generated token ids and, when a tokenizer is present, decoded strings.
        """

        # Load LoRA-wrapped model directly
        lora_model = self.load_lora_model()
        lora_model.eval()

        # Generation args
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature is not None and temperature > 0,
        }
        if temperature is not None and temperature > 0:
            gen_kwargs["temperature"] = temperature
        if hasattr(self, "tokenizer") and getattr(self, "tokenizer", None) is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            generated = lora_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = None
        if hasattr(self, "tokenizer") and getattr(self, "tokenizer", None) is not None:
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated, decoded


    def fit(self):
        """Run ZO loop."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

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
        # lora_model=self.load_lora_model()
        # self.actor_rollout_wg.print_trainable_params()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # input_id=batch.batch["input_ids"]
                # attention_mask=batch.batch["attention_mask"]
                # generated, decoded=self.debug_generate_with_lora(input_id,attention_mask)
                # print(decoded[0])
                # self.zo_forward(batch,timing_raw)
                lr=self._get_learning_rate()
                projected_grad,zo_random_seed=self.zo_step(batch,timing_raw)
                self.actor_rollout_wg.zo_update(projected_grad,zo_random_seed,lr)
                is_last_step = self.global_steps >= self.total_training_steps
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
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                self.global_steps += 1
                progress_bar.update(1)



class RayRiskAverseTrainer(RayMAPPOTrainer):
    """MAPPO variant with shared discussion prompts and an adversarial agent."""

    def _shared_discussion_prompt(self) -> str:
        """Use a unified discussion prompt for all agents."""
        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        return ma.get("discussion_prompt", "The discussion history is as follows:")
    def _apply_kl_penalty(self, data: DataProto, data_ref: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
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
            data.batch["old_log_probs"], data_ref.batch["old_log_probs"], kl_penalty=kl_penalty
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
    def _build_input_ids_from_histories(
        self,
        system_prompt,
        discussion_prompt,
        questions,
        histories,
        batch: DataProto,
        agent_key,
        max_history_tokens,
    ):
        shared_prompt = self._shared_discussion_prompt()
        return super()._build_input_ids_from_histories(
            system_prompt, shared_prompt, questions, histories, batch, agent_key, max_history_tokens
        )

    def mappo_fit(self):
        """Run multi-agent PPO with adversarial agent and shared discussion prompts."""
        from verl.utils.tracking import Tracking

        def _with_prefix(prefix: str, metric_dict: dict) -> dict:
            """Prefix metrics with agent identifier to avoid collisions."""
            return {f"{prefix}/{k}": v for k, v in metric_dict.items()}

        ma = OmegaConf.select(self.config, "multi_agent", default={}) or {}
        num_agents = int(ma.get("num_agents", 1))
        num_rounds = int(ma.get("num_rounds", 1))
        risk_coef = float(ma.get("risk_coef", 1.0))
        agent_keys = list(self.train_dataloaders.keys())

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fns is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._multi_agent_validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

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
            for batch_tuple in zip(*(self.train_dataloaders[k] for k in agent_keys)):
                round_agent_batches = [[None for _ in range(num_agents)] for _ in range(num_rounds)]
                metrics = {}
                timing_raw = {}
                round_agent_metrics = [[{} for _ in range(num_agents)] for _ in range(num_rounds)]

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                is_last_step = self.global_steps >= self.total_training_steps
                step_durations = []

                batch_size = self.config.data.train_batch_size
                histories = [""] * batch_size
                for r in range(num_rounds):
                    this_round = [""] * batch_size
                    from concurrent.futures import ThreadPoolExecutor

                    futures = []
                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx, agent_key in enumerate(agent_keys):
                            batch_dict = batch_tuple[agent_idx]
                            futures.append(
                                executor.submit(
                                    self._single_agent_rollout,
                                    agent_idx,
                                    agent_key,
                                    batch_dict,
                                    histories,
                                    r,
                                    timing_raw,
                                    round_agent_metrics,
                                )
                            )
                    results = [f.result() for f in futures]
                    for agent_idx, (resp_texts, batch) in enumerate(results):
                        this_round = [
                            old + f"\nAgent {agent_idx}: {new}" for old, new in zip(this_round, resp_texts)
                        ]
                        round_agent_batches[r][agent_idx] = batch

                    histories[:] = this_round
                for r in range(num_rounds):
                    round_agent_batches[r][0].batch["token_level_scores"] = - round_agent_batches[r][1].batch["token_level_scores"]
                    round_agent_batches[r][0],kl_metrics=self._apply_kl_penalty(round_agent_batches[r][0],round_agent_batches[r][1], self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty)
                    round_agent_metrics[r][0].update(kl_metrics)

                if self.config.algorithm.use_kl_in_reward:
                    for r in range(num_rounds):
                        non_adv_agents = [idx for idx in range(num_agents) if idx != 0]
                        futures = []
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx in non_adv_agents:
                                batch = round_agent_batches[r][agent_idx]
                                futures.append(
                                    executor.submit(
                                        apply_kl_penalty,
                                        batch,
                                        self.kl_ctrl_in_reward,
                                        self.config.algorithm.kl_penalty,
                                    )
                                )
                        results = [f.result() for f in futures]
                        for agent_idx, (batch, kl_metrics) in zip(non_adv_agents, results):
                            round_agent_batches[r][agent_idx] = batch
                            round_agent_metrics[r][agent_idx].update(kl_metrics)
                else:
                    for r in range(num_rounds):
                        for agent_idx in range(num_agents):
                            batch = round_agent_batches[r][agent_idx]
                            if agent_idx == 0 and "token_level_rewards" in batch.batch:
                                continue
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                for r in range(num_rounds):
                    futures = []
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx in range(num_agents):
                            batch = round_agent_batches[r][agent_idx]
                            futures.append(
                                executor.submit(
                                    compute_advantage,
                                    batch,
                                    self.config.algorithm.adv_estimator,
                                    self.config.algorithm.gamma,
                                    self.config.algorithm.lam,
                                    self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo,
                                    self.config.algorithm,
                                )
                            )
                    results = [f.result() for f in futures]
                    for agent_idx, batch in enumerate(results):
                        round_agent_batches[r][agent_idx] = batch

                if self.use_critic:
                    for r in range(num_rounds):
                        futures = []
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx, agent_key in enumerate(agent_keys):
                                futures.append(
                                    executor.submit(
                                        self._update_critic,
                                        r,
                                        agent_idx,
                                        agent_key,
                                        round_agent_batches,
                                        timing_raw,
                                        round_agent_metrics,
                                    )
                                )
                        results = [f.result() for f in futures]

                if self.config.trainer.critic_warmup <= self.global_steps:
                    for r in range(num_rounds):
                        futures = []
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=num_agents) as executor:
                            for agent_idx, agent_key in enumerate(agent_keys):
                                futures.append(
                                    executor.submit(
                                        self._update_actor,
                                        r,
                                        agent_idx,
                                        agent_key,
                                        round_agent_batches,
                                        timing_raw,
                                        round_agent_metrics,
                                    )
                                )
                        results = [f.result() for f in futures]

                for r in range(num_rounds):
                    futures = []
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=num_agents) as executor:
                        for agent_idx in range(num_agents):
                            futures.append(
                                executor.submit(
                                    self._update_metrics,
                                    r,
                                    agent_idx,
                                    round_agent_batches,
                                    round_agent_metrics,
                                    timing_raw,
                                    metrics,
                                )
                            )
                        results = [f.result() for f in futures]

                if (
                    self.val_reward_fns is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._multi_agent_validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )

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

                if step_durations:
                    steps_duration = max(step_durations)
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    for agent_key in agent_keys:
                        self.actor_rollout_wgs[agent_key].dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                for agent_idx, agent_key in enumerate(agent_keys):
                    train_dataset = self.train_datasets.get(agent_key, None)
                    if hasattr(train_dataset, "on_batch_end"):
                        train_dataset.on_batch_end(batch=round_agent_batches[0][agent_idx])

