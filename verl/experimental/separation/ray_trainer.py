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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip


class SeparateRayPPOTrainer(RayPPOTrainer):
    """
    Support for the initialization and fit process of Ray Trainer in the resource-separated scenario:
        - Fully async policy
        - One-step off-policy
    """

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
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )
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
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

    def init_workers(self):
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
        self._init_async_rollout_manager()

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    def _create_worker_classes(self):
        self._create_actor_rollout_classes()
        self._create_critic_class()
        self._create_reference_policy_class()
        self._create_reward_model_class()

    def _create_actor_rollout_classes(self):
        raise NotImplementedError

    def _create_critic_class(self):
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    def _create_reference_policy_class(self):
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
                # profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    def _create_reward_model_class(self):
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    def _init_worker_groups(self):
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
        self.all_wg = all_wg

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = self.all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

    def _init_reward_loop(self):
        if self.use_reward_loop:
            # create reward loop manager
            if self.use_reward_loop:
                from verl.experimental.reward_loop import RewardLoopManager

                # initalize reward loop manager
                # reward model (colocate or standalone): get resource_pool
                # no reward model: resource_pool = None
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

    def _init_async_rollout_manager(self):
        pass

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.

        !!!
        The logic of fit is consistent with that of fit_refactor;
        if any modifications are made, apply them to both methods simultaneously.
        """
        from omegaconf import OmegaConf

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
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

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

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self.epoch = epoch
                self.fit_step(batch_dict)
                if self.is_last_step:
                    return

    def fit_step(self, batch_dict: Any = None):
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
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = self._fit_get_batch(batch_dict)
            batch = self._fit_generate(batch)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_weights()
            self._fit_dump_data(batch)

        self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_torch_memory()
        self._fit_experimental(batch)
        self._fit_postprocess_step()

    def _fit_prepare_step(self):
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
        self.is_last_step = self.global_steps >= self.total_training_steps

    def _fit_start_profile(self):
        timing_raw = self.timing_raw
        with marked_timer("start_profile", timing_raw):
            self._start_profiling(
                not self.prev_step_profile and self.curr_step_profile
                if self.config.global_profiler.profile_continuous_steps
                else self.curr_step_profile
            )

    def _fit_get_batch(self, batch_dict: dict) -> DataProto:
        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        # add uid
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        return batch

    def _fit_generate(self, batch: DataProto = None) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        gen_batch = self._get_gen_batch(batch)
        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        with marked_timer("gen", timing_raw, color="red"):
            if not self.async_rollout_mode:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
            else:
                if self.curr_step_profile:
                    self.async_rollout_manager.start_profile(global_step=self.global_steps)
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                self.checkpoint_manager.sleep_replicas()
                if self.curr_step_profile:
                    self.async_rollout_manager.stop_profile()

            timing_raw.update(gen_batch_output.meta_info["timing"])
            gen_batch_output.meta_info.pop("timing", None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with marked_timer("gen_max", timing_raw, color="purple"):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                if not self.async_rollout_mode:
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                else:
                    if self.curr_step_profile:
                        self.async_rollout_manager.start_profile()
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                    self.checkpoint_manager.sleep_replicas()
                    if self.curr_step_profile:
                        self.async_rollout_manager.stop_profile()
                batch = batch.union(gen_baseline_output)
                # compute reward model score on batch
                rm_scores = None
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    batch_reward = self._compute_reward_colocate(batch)
                    batch = batch.union(batch_reward)

                # Compute or extract reward for REMAX baseline
                reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                keys_to_pop = set(gen_baseline_output.batch.keys())
                if rm_scores is not None:
                    keys_to_pop.update(rm_scores.batch.keys())
                batch.pop(batch_keys=list(keys_to_pop))

                batch.batch["reward_baselines"] = reward_baseline_tensor

                del rm_scores, gen_baseline_batch, gen_baseline_output
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
        # get images_seqlens
        images_seqlens_all = []
        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
            if "image_grid_thw" not in multi_modal_input.keys():
                continue
            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
        batch.meta_info["images_seqlens"] = images_seqlens_all
        return batch

    def _fit_compute_reward(self, batch: DataProto) -> DataProto:
        timing_raw = self.timing_raw
        with marked_timer("reward", timing_raw, color="yellow"):
            # compute reward model score
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                batch_reward = self._compute_reward_colocate(batch)
                batch = batch.union(batch_reward)

            # Compute or extract reward_tensor and reward_extra_infos_dict for training
            reward_tensor, reward_extra_infos_dict = extract_reward(batch)
            self.reward_tensor = reward_tensor
            self.reward_extra_infos_dict = reward_extra_infos_dict
        return batch

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        # Operating Mode Selection:
        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

            apply_bypass_mode(
                batch=batch,
                rollout_corr_config=rollout_corr_config,
                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
            )
        else:  # Recompute old_log_probs
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
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
                old_log_prob.batch.pop("entropys")
                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                    router_mode = getattr(self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled")
                    if router_mode == "R2":
                        batch.batch.pop("routed_experts")
                    else:
                        old_log_prob.batch.pop("routed_experts")
                batch = batch.union(old_log_prob)
                if "rollout_log_probs" in batch.batch.keys():
                    # TODO: we may want to add diff of probs too.
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))

        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
        return batch

    def _fit_compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        timing_raw = self.timing_raw
        if self.use_reference_policy:
            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        return batch

    def _fit_compute_critic(self, batch: DataProto) -> DataProto:
        timing_raw = self.timing_raw
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)
        return batch

    def _fit_compute_advantage(self, batch) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        reward_tensor = self.reward_tensor
        reward_extra_infos_dict = self.reward_extra_infos_dict

        with marked_timer("adv", timing_raw, color="brown"):
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
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

            # Compute rollout correction: IS weights, rejection sampling, and metrics
            # Only runs in decoupled mode (computes once per batch using stable π_old)
            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            if (
                rollout_corr_config is not None
                and "rollout_log_probs" in batch.batch
                and not bypass_recomputing_logprobs  # Only in decoupled mode
            ):
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                # Compute IS weights, apply rejection sampling, compute metrics
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                # IS and off-policy metrics already have rollout_corr/ prefix
                metrics.update(is_metrics)

            # compute advantages, executed on the driver process
            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor

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

    def _fit_update_critic(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)
        return batch

    def _fit_update_actor(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)

            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return batch

    def _fit_update_weights(self):
        timing_raw = self.timing_raw
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update weights from trainer to rollout
            with marked_timer("update_weights", timing_raw, color="red"):
                self.checkpoint_manager.update_weights()

    def _fit_dump_data(self, batch: DataProto):
        timing_raw = self.timing_raw
        reward_extra_infos_dict = self.reward_extra_infos_dict
        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

    def _fit_validate(self):
        metrics = self.metrics
        timing_raw = self.timing_raw
        if self.config.trainer.test_freq > 0 and (
            self.is_last_step or self.global_steps % self.config.trainer.test_freq == 0
        ):
            with marked_timer("testing", timing_raw, color="green"):
                val_metrics: dict = self._validate()
                if self.is_last_step:
                    self.last_val_metrics = val_metrics
            metrics.update(val_metrics)

    def _fit_save_checkpoint(self):
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
            self.is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                # self.checkpoint_manager.sleep_replicas()
                self._save_checkpoint()
                # wake replicas to avoid OOM during checkpoint saving
                # TODO: Check separation is needed.
                # self.checkpoint_manager.update_weights()

    def _fit_stop_profile(self):
        timing_raw = self.timing_raw
        with marked_timer("stop_profile", timing_raw):
            self.next_step_profile = (
                self.global_steps + 1 in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            self._stop_profiling(
                self.curr_step_profile and not self.next_step_profile
                if self.config.global_profiler.profile_continuous_steps
                else self.curr_step_profile
            )
            self.prev_step_profile = self.curr_step_profile
            self.curr_step_profile = self.next_step_profile

    def _fit_collect_metrics(self, batch):
        metrics = self.metrics
        timing_raw = self.timing_raw

        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        # compute variance proxy metrics
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

    def _fit_torch_memory(self):
        if (
            hasattr(self.config.actor_rollout_ref.actor, "profiler")
            and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
        ):
            self.actor_rollout_wg.dump_memory_snapshot(
                tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
            )

    def _fit_experimental(self, batch):
        # this is experimental and may be changed/removed in the future in favor of a general-purpose one
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)

        # this is experimental and may be changed/removed in the future
        # in favor of a general-purpose data buffer pool
        if hasattr(self.train_dataset, "on_batch_end"):
            # The dataset may be changed after each training batch
            self.train_dataset.on_batch_end(batch=batch)

    def _fit_postprocess_step(self):
        metrics = self.metrics
        timing_raw = self.timing_raw

        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)

        # TODO: make a canonical logger that supports various backend
        self.logger.log(data=metrics, step=self.global_steps)
        self.progress_bar.update(1)
        self.global_steps += 1
        if self.is_last_step:
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
            pprint(f"Final validation metrics: {self.last_val_metrics}")
            self.progress_bar.close()
