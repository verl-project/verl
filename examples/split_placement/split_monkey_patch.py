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
An naive implementation of split placment example
"""

import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    marked_timer,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics


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

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    # we start from step 1
    self.global_steps += 1
    last_val_metrics = None

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw):
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with marked_timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        # compute reward model score on batch
                        rm_scores = None
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            rm_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_scores)
                        reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        keys_to_pop = set(gen_baseline_output.batch.keys())
                        if rm_scores is not None:
                            keys_to_pop.update(rm_scores.batch.keys())
                        batch.pop(batch_keys=list(keys_to_pop))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del rm_scores, gen_baseline_batch, gen_baseline_output

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw):
                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    # we combine with rule-based rm
                    reward_tensor, _ = compute_reward(batch, self.reward_fn)
                    batch.batch["token_level_scores"] = reward_tensor

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
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

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor_call", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                else:
                    actor_output = None

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic_call", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    # NOTE: make sure you set blocking=False in update_actor and update_crtic in the worker class
                    with marked_timer("update_actor_critic", timing_raw):
                        critic_output = critic_output.get()
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                if actor_output is not None:
                    actor_output = actor_output.get()
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if self.global_steps >= self.total_training_steps:
                pprint(f"Final validation metrics: {last_val_metrics}")
                return

            self.global_steps += 1
