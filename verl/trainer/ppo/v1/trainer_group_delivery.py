# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Synchronous GRPO on separate GPU pools, with group-level gradient accumulation."""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.utils import Role, need_critic
from verl.trainer.ppo.v1.group_delivery import GroupDeliveryReplayBuffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.workers.engine_workers_group_delivery import GroupDeliveryActorWorker
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_group_delivery_config(config: DictConfig) -> None:
    """Reject features that change the fixed-batch objective or update boundary."""
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    conditions = {
        "actor.strategy=megatron": actor.strategy == "megatron",
        "actor_rollout_ref.hybrid_engine=False": not config.actor_rollout_ref.hybrid_engine,
        "dedicated rollout GPUs": rollout.nnodes > 0 and rollout.n_gpus_per_node > 0,
        "a distributed checkpoint engine": rollout.checkpoint_engine.backend != "naive",
        "algorithm.adv_estimator=grpo": config.algorithm.adv_estimator == "grpo",
        "rollout.n > 1": rollout.n > 1,
        "actor.ppo_epochs=1": actor.ppo_epochs == 1,
        "actor.ppo_mini_batch_size=data.train_batch_size": actor.ppo_mini_batch_size == config.data.train_batch_size,
        "actor.policy_loss.loss_mode=vanilla": actor.policy_loss.loss_mode == "vanilla",
        "no critic": not need_critic(config),
        "no distillation": not is_distillation_enabled(config.get("distillation")),
        "algorithm.use_kl_in_reward=False (KL loss is supported)": not config.algorithm.use_kl_in_reward,
        "no dynamic group filtering": not OmegaConf.select(config, "algorithm.filter_groups.enable", default=False),
        "no rollout skipping": not OmegaConf.select(config, "skip.rollout_tq.enable", default=False),
        "no custom sampler": not OmegaConf.select(config, "trainer.v1.sampler.custom_sampler.path"),
        "no score centering": not OmegaConf.select(
            config, "algorithm.rollout_correction.score_centering", default=False
        ),
        "no batch-dependent rollout correction": not (
            OmegaConf.select(config, "algorithm.rollout_correction.rollout_is")
            or OmegaConf.select(config, "algorithm.rollout_correction.rollout_rs")
        ),
        "no actor score centering": not OmegaConf.select(
            config, "actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering", default=False
        ),
        "no actor warmup": config.trainer.critic_warmup == 0,
        "positive polling interval": config.trainer.v1.group_delivery.poll_interval > 0,
        "positive train batch": config.data.train_batch_size > 0,
    }
    reward_model = config.reward.reward_model
    conditions["a separate reward model pool"] = not reward_model.enable or reward_model.enable_resource_pool
    supported_losses = {"token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"}
    conditions["a supported mean loss aggregation"] = actor.loss_agg_mode in supported_losses
    if actor.loss_agg_mode == "seq-mean-token-sum-norm":
        conditions["a positive fixed actor.loss_scale_factor"] = (
            actor.loss_scale_factor is not None and actor.loss_scale_factor > 0
        )
    failed = [description for description, ok in conditions.items() if not ok]
    if failed:
        raise ValueError("group_delivery requires: " + "; ".join(failed))


@register_trainer("group_delivery")
class PPOTrainerGroupDelivery(PPOTrainerSeparateAsync):
    """Reuse standalone resource setup, with no prefetch, partial rollout, or early updates.

    Each controller iteration delivers one complete GRPO group to all actor ranks.
    Padding satisfies DP dispatch; the worker stores already finalized gradients
    outside Megatron's schedule buffers. The last group opens the optimizer barrier.
    """

    def __init__(self, config: DictConfig):
        validate_group_delivery_config(config)
        self._enable_hybrid_replicas = False
        # Deliberately bypass the async initializer and all switching/prefetch state.
        # Define the switch config it would have set, with switching forced off, so
        # inherited hooks that read it fail safe instead of raising AttributeError.
        self.hybrid_rollout_config = replace(
            omega_conf_to_dataclass(config.trainer.v1.separate_async.hybrid_rollout),
            enable_switch=False,
        )
        PPOTrainer.__init__(self, config)

    def _build_replay_buffer(self):
        return GroupDeliveryReplayBuffer(
            trainer_mode="group_delivery",
            trainer_config=self.config.trainer.v1.group_delivery,
            max_off_policy_threshold=None,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            poll_interval=self.config.trainer.v1.group_delivery.poll_interval,
        )

    def _init_resource_pool_mgr(self):
        PPOTrainer._init_resource_pool_mgr(self)
        self.role_worker_mapping[Role.Actor] = ray.remote(GroupDeliveryActorWorker)

    def get_llm_client(self):
        return self.standalone_server_manager.get_client(client_cls=LLMServerClient)

    def _reissue_inflight_prompts(self, partition_id="train") -> int:
        # Checkpoints are only taken after the accumulation window was committed.
        # A restored async checkpoint with pending prompts has different semantics.
        partitions = tq.kv_list(partition_id) or {}
        tags = partitions.get(partition_id, {})
        if any(tag.get("is_prompt", False) for tag in tags.values()):
            raise RuntimeError("group_delivery cannot resume a checkpoint containing unconsumed prompt groups")
        if tags:
            tq.kv_clear(keys=list(tags), partition_id=partition_id)
        return 0

    def on_init_end(self):
        # The dedicated rollout replicas are created with load_format=dummy and
        # hold no checkpoint weights; publish them before the first rollout.
        self.standalone_checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        """No warmup or next-step prompts may run under the current policy."""

    def on_step_begin(self):
        """The fixed batch is submitted by step(), after opening the gradient window."""

    def on_validate_begin(self):
        """Dedicated rollout replicas are already available for validation."""

    def on_sample_begin(self):
        """Rollout continues while the actor consumes completed groups."""

    def on_sample_end(self):
        """Keep rollout replicas running until the fixed batch finishes."""

    def _record_event(self, event: str, **details) -> None:
        """Emit controller timestamps for overlap and update-boundary verification."""
        logger.info(
            "group_delivery_event %s",
            json.dumps(dict(event=event, step=self.global_steps, monotonic_s=time.monotonic(), **details)),
        )

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._record_event("weight_sync_begin")
            self._pending_sync_metrics = self.standalone_checkpoint_manager.update_weights(self.global_steps)
            self._record_event("weight_sync_end")

    def _compute_old_log_prob(self, batch, metrics):
        # Actor parameters remain unchanged throughout the entire batch.
        return PPOTrainer._compute_old_log_prob(self, batch, metrics)

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        # There is no optimizer mini-batch inside a group. Only align to DP here;
        # the engine owns token packing and PP/VPP micro-batch alignment.
        actor = self.config.actor_rollout_ref.actor
        multiple = dp_size
        if not actor.use_dynamic_bsz:
            multiple = math.lcm(multiple, dp_size * actor.ppo_micro_batch_size_per_gpu)
        rollout = self.config.actor_rollout_ref.rollout
        if not rollout.log_prob_use_dynamic_bsz:
            multiple = math.lcm(multiple, dp_size * rollout.log_prob_micro_batch_size_per_gpu)
        if self.use_reference_policy:
            ref = self.config.actor_rollout_ref.ref
            ref_dp = dp_size
            if not self.ref_in_actor:
                wg = self.ref_policy_wg
                if "ref" not in wg._dispatch_info:
                    wg._dispatch_info["ref"] = wg._query_dispatch_info("ref")
                ref_dp = max(wg._dispatch_info["ref"]) + 1
            multiple = math.lcm(multiple, ref_dp)
            if not ref.log_prob_use_dynamic_bsz:
                multiple = math.lcm(multiple, ref_dp * ref.log_prob_micro_batch_size_per_gpu)
        return multiple

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        prompts = self._next_train_batch()
        group_ids = list(prompts["uid"])
        if "__rollout_n__" in prompts.keys():
            raise ValueError("group_delivery requires the configured, fixed rollout.n for every prompt")
        self.replay_buffer.begin_batch(group_ids, self.global_steps, self.config.actor_rollout_ref.rollout.n)
        keys, tags = [], []
        aggregator = MetricsAggregator()
        self._overlapped_groups = 0
        self._record_event("batch_begin", groups=len(group_ids))
        try:
            self.actor_rollout_wg.begin_group_batch(group_ids)
            self._submit_batch_to_rollout(prompts)
            for _ in group_ids:
                group_metrics = {}
                batch = self._step_once(group_metrics, timing_raw, sample_batch_size=1)
                keys.extend(batch.keys)
                tags.extend(batch.tags)
                aggregator.add_step_metrics(group_metrics, sample_count=int(batch.extra_info["group_delivery_weight"]))
            self.replay_buffer.finish_batch()
            with marked_timer("optimizer_step", timing_raw, color="red"):
                self._record_event("optimizer_begin")
                results = self.actor_rollout_wg.finish_group_batch()
                self._record_event("optimizer_end")
        except Exception:
            # Log the original failure before touching the workers: if one rank
            # died inside a collective (e.g. a CUDA OOM), its peer spins there
            # and any cleanup RPC to that peer never returns. Without this line
            # the traceback is lost behind the blocked cleanup.
            logger.exception("group_delivery step failed; aborting the batch")
            # Both cleanups must be attempted; preserve the original failure.
            cleanups = (
                ("abort_group_batch", self.actor_rollout_wg.abort_group_batch),
                ("abort_replicas", self.standalone_checkpoint_manager.abort_replicas),
            )
            for name, cleanup in cleanups:
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    executor.submit(cleanup).result(timeout=60)
                except TimeoutError:
                    logger.exception("group_delivery cleanup %s timed out; worker is stuck", name)
                except Exception:
                    logger.exception("group_delivery cleanup %s failed", name)
                finally:
                    executor.shutdown(wait=False)
            raise
        metrics.update(aggregator.get_aggregated_metrics())
        # ONE_TO_ALL returns one result per rank; the optimizer computes a global norm.
        # PP rank zero need not own outputs; pick each metric from its first output rank.
        for result in results:
            for key, value in result.items():
                if value is not None:
                    metrics.setdefault(f"actor/{key}", value)
        metrics["training/groups_backward_started_during_rollout"] = self._overlapped_groups
        metrics["training/delivered_groups"] = len(group_ids)
        metrics["training/optimizer_steps"] = 1
        return KVBatchMeta(partition_id="train", keys=keys, tags=tags)

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        actor = self.config.actor_rollout_ref.actor
        group_ids = {
            key.split("_", 1)[0] for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding")
        }
        if len(group_ids) != 1:
            raise RuntimeError("Expected exactly one real prompt group before backward")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["response_mask"])
        lengths = [float(mask.sum()) for mask in data["response_mask"].unbind()]
        sequences = sum(length > 0 for length in lengths)
        weight = sum(lengths) if actor.loss_agg_mode == "token-mean" else sequences
        if weight <= 0:
            raise RuntimeError("A synchronous group has no valid response tokens")
        batch.extra_info.update(
            group_delivery_id=group_ids.pop(),
            group_delivery_weight=weight,
            global_batch_size=sequences,
            calculate_entropy=actor.calculate_entropy or actor.entropy_coeff != 0.0,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        pending = sum(
            tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
            for tag in (tq.kv_list("train") or {}).get("train", {}).values()
        )
        self._overlapped_groups += int(pending > 0)
        self._record_event("backward_begin", group=batch.extra_info["group_delivery_id"], pending_groups=pending)
        output = self.actor_rollout_wg.backward_group(batch)
        self._record_event("backward_end", group=batch.extra_info["group_delivery_id"])
        metrics.update(reduce_metrics(rename_dict(output["metrics"], "actor/")))
        return batch
