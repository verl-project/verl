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
import logging
import os
import time
from collections import deque
from enum import Enum

import ray
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.trainer.config import HybridRolloutSwitchConfig
from verl.trainer.ppo.utils import Role, need_reward_model
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HybridEngineMode(Enum):
    TRAINER = "trainer"
    ROLLOUT = "rollout"


@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are separate, trainer may switch to rollout if idle.
    2. Partial rollout is enabled.
    """

    def __init__(self, config: DictConfig):
        train_batch_size = config.data.train_batch_size
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        parameter_sync_step = config.trainer.v1.separate_async.parameter_sync_step
        assert train_batch_size == parameter_sync_step * ppo_mini_batch_size, (
            f"train_batch_size must equal parameter_sync_step * ppo_mini_batch_size in separate async "
            f"training, but got train_batch_size={train_batch_size}, "
            f"parameter_sync_step={parameter_sync_step}, ppo_mini_batch_size={ppo_mini_batch_size}"
        )
        assert config.actor_rollout_ref.rollout.nnodes > 0, "nnodes must be > 0 in separate async training"
        assert config.actor_rollout_ref.rollout.n_gpus_per_node > 0, (
            "n_gpus_per_node must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive", (
            "please use nccl/nixl/mooncake, etc. backend for separate async training"
        )
        if need_reward_model(config):
            assert config.reward.reward_model.enable_resource_pool, (
                "Colocate reward model (reward.reward_model.enable_resource_pool=False) is not supported "
                "in separate async mode, because the standalone rollout never pauses to free GPU memory. "
                "Use standalone mode (reward.reward_model.enable_resource_pool=True) instead."
            )

        super().__init__(config)
        self.hybrid_rollout_config: HybridRolloutSwitchConfig = omega_conf_to_dataclass(
            self.config.trainer.v1.separate_async.hybrid_rollout
        )
        if self.hybrid_rollout_config.enable_switch:
            # No support for PD disaggregation for switching
            rollout_cfg = self.config.get("actor_rollout_ref", {}).get("rollout", {})
            disaggregation_cfg = rollout_cfg.get("disaggregation", {})
            if bool(disaggregation_cfg.get("enabled", False)):
                raise ValueError(
                    "trainer.v1.separate_async.hybrid_rollout.enable_switch does not support rollout disaggregation: "
                    "step-boundary redistribution relies on paused replicas queueing new requests"
                )
            # Custom samplers own their polling. One that cannot wait on buffer depth would hand the
            # GPUs back before the step has anything to train on, which is worse than not switching.
            required_methods = ("wait_for_sampleable", "get_sampleable_count")
            if any(not hasattr(self.replay_buffer, method) for method in required_methods):
                raise TypeError(
                    f"{type(self.replay_buffer).__name__} must implement {required_methods} when "
                    "trainer.v1.separate_async.hybrid_rollout.enable_switch=True"
                )
            self._init_hybrid_rollout_state()

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        # Replace ActorRolloutRefWorker with DetachActorWorker to get CPU save/restore
        # capability needed for Decoupled PPO when parameter_sync_step > 1.
        # The base class adds exactly one of ActorRolloutRef or ActorRollout to the mapping.
        if Role.ActorRolloutRef in self.role_worker_mapping:
            self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(DetachActorWorker)
        elif Role.ActorRollout in self.role_worker_mapping:
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(DetachActorWorker)

    def _init_hybrid_rollout_state(self) -> None:
        config = self.hybrid_rollout_config
        self._switch_threshold_ratio = config.switch_threshold_ratio
        self._idle_steps = 0
        self._calm_steps = 0
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self._sample_start = time.perf_counter()
        self._step_threshold = 0
        self._wait_seconds = 0.0
        self._wait_samples = 0
        self._to_rollout_costs: deque[float] = deque(maxlen=config.switch_cost_window_size)
        self._to_trainer_costs: deque[float] = deque(maxlen=config.switch_cost_window_size)
        rollout_cfg = self.config.get("actor_rollout_ref", {}).get("rollout", {})
        trainer_cfg = self.config.trainer
        hybrid_gpus = trainer_cfg.nnodes * trainer_cfg.n_gpus_per_node
        standalone_gpus = rollout_cfg.nnodes * rollout_cfg.n_gpus_per_node
        # TODO: Use a more accurate or dynamic scaling factor.
        self._scaling_factor = (hybrid_gpus + standalone_gpus) / standalone_gpus

    def _setup(self):
        super()._setup()

        # initialize standalone rollout
        # TODO: make initialization parallel with super().init()
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
        self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, start_rank=hybrid_num_replicas
        )
        rollout_config = self.config.actor_rollout_ref.rollout
        if rollout_config.prometheus.enable:
            server_addresses = (
                self.llm_server_manager.server_addresses + self.standalone_server_manager.server_addresses
            )
            update_prometheus_config(rollout_config.prometheus, server_addresses, rollout_config.name)

        # create checkpoint engine manager for trainer and standalone rollout
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        # hybrid engine is in rollout mode after initialization
        self.current_mode = HybridEngineMode.ROLLOUT
        self.add_replicas_to_balancer()

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Version-aware old_log_probs computation for Decoupled PPO.

        In bypass mode, delegates to the base class (copies rollout_log_probs directly).
        In Decoupled mode, uses save_model_to_cpu / restore_model_from_cpu to ensure
        all mini-batches within a parameter_sync_step cycle use the same stable π_old.

        - local_trigger_step == 0: Current weights are π_old → save to CPU, compute directly.
        - local_trigger_step >= 1: Save current weights → restore π_old → compute → restore current.
        """
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            return super()._compute_old_log_prob(batch, metrics)

        if self.local_trigger_step == 0:
            self.actor_rollout_wg.save_model_to_cpu(0)
            return super()._compute_old_log_prob(batch, metrics)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(0)
            result = super()._compute_old_log_prob(batch, metrics)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
            return result

    def get_llm_client(self):
        # get server client from standalone rollout
        return self.standalone_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def _rollout_server_managers(self) -> list:
        managers = super()._rollout_server_managers()
        standalone = getattr(self, "standalone_server_manager", None)
        if standalone is not None:
            managers.append(standalone)
        return managers

    def on_init_end(self):
        # update weights after loading checkpoint
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        self._add_async_warmup_batches(self.config.trainer.v1.separate_async.num_warmup_batches)

    def on_validate_begin(self):
        if self.current_mode == HybridEngineMode.TRAINER:
            logger.info("Switching hybrid engine to rollout mode for validation")
            self.switch_to_rollout()

    def on_step_begin(self):
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self._step_threshold = 0
        if self.hybrid_rollout_config.enable_switch:
            self.timing_raw["switch_wait"] = 0.0
        if self.current_mode != HybridEngineMode.ROLLOUT:
            return
        if not self.hybrid_rollout_config.enable_switch:
            self._timed_switch_to_trainer()
            return

        self._step_threshold = self._switch_threshold()
        sampleable_count = self.replay_buffer.get_sampleable_count(self.global_steps, "train")
        if sampleable_count >= self._step_threshold:
            self._timed_switch_to_trainer()

    def _timed_switch_to_trainer(self) -> None:
        """Switch Hybrid back to training and record the full remove/abort/sleep cost."""
        switch_start = time.perf_counter()
        with marked_timer("switch_to_trainer", self.timing_raw, color="cyan"):
            self.switch_to_trainer()
        if self.hybrid_rollout_config.enable_switch:
            self._to_trainer_costs.append(time.perf_counter() - switch_start)

    def on_sample_begin(self):
        if self.hybrid_rollout_config.enable_switch:
            sampleable = self.replay_buffer.get_sampleable_count(self.global_steps, "train")
            mini_batch_size = self.config.data.train_batch_size // self.parameter_sync_step
            self._step_wait_samples += max(0, mini_batch_size - sampleable)
        self._sample_start = time.perf_counter()

    def on_sample_end(self):
        self._step_sample_wait_seconds += time.perf_counter() - self._sample_start

    def prepare_step(self) -> dict:
        metrics = super().prepare_step()
        metrics.update(self._wait_for_sampleable_and_switch())
        return metrics

    def _wait_for_sampleable_and_switch(self) -> dict:
        if self.current_mode != HybridEngineMode.ROLLOUT:
            return {}

        logger.info(f"Lending hybrid engine to generation until {self._step_threshold} groups are sampleable")
        with marked_timer("switch_wait", self.timing_raw, color="yellow"):
            _, eviction_metrics = self.replay_buffer.wait_for_sampleable(
                self.global_steps, "train", self._step_threshold
            )
        self._timed_switch_to_trainer()
        return eviction_metrics

    def _switch_threshold(self) -> int:
        """Sampleable prompts before switching to trainer, floored at one mini-batch."""
        train_batch_size = self.config.data.train_batch_size
        mini_batch_size = train_batch_size // self.parameter_sync_step
        target = round(self._switch_threshold_ratio * train_batch_size)
        return min(max(target, mini_batch_size), train_batch_size)

    def _step_had_idle(self) -> bool:
        """Whether waiting for sampleable prompts."""
        poll_interval = getattr(self.replay_buffer, "poll_interval", 2.0)
        return self._step_sample_wait_seconds > poll_interval

    def _adapt_switch_threshold(self, had_idle: bool) -> None:
        config = self.hybrid_rollout_config
        if had_idle:
            self._calm_steps = 0
            self._idle_steps = min(self._idle_steps + 1, config.switch_threshold_release_steps)
            if self._idle_steps < config.switch_threshold_release_steps:
                return
            self._switch_threshold_ratio = min(1.0, self._switch_threshold_ratio + config.switch_threshold_step_up)
            return

        self._idle_steps = 0
        self._calm_steps = min(self._calm_steps + 1, config.switch_threshold_release_steps)
        if self._calm_steps < config.switch_threshold_release_steps:
            return
        min_ratio = 1.0 / self.parameter_sync_step
        self._switch_threshold_ratio = max(min_ratio, self._switch_threshold_ratio - config.switch_threshold_step_down)

    def _effective_switch_cost(self) -> float | None:
        if not self._to_rollout_costs or not self._to_trainer_costs:
            return None
        return sum(self._to_rollout_costs) / len(self._to_rollout_costs) + sum(self._to_trainer_costs) / len(
            self._to_trainer_costs
        )

    def on_step_end(self):
        config = self.hybrid_rollout_config
        should_switch = False
        prepare_seconds = 0.0
        decision_metrics: dict[str, float] = {}
        if config.enable_switch:
            ratio_used = self._switch_threshold_ratio
            had_idle = self._step_had_idle()
            if self._step_wait_samples > 0 and self._step_sample_wait_seconds > 0:
                self._wait_seconds += self._step_sample_wait_seconds
                self._wait_samples += self._step_wait_samples
            if config.adaptive_switch_threshold:
                self._adapt_switch_threshold(had_idle)

            decision_threshold = self._switch_threshold()
            sampleable_count = self.replay_buffer.get_sampleable_count(self.global_steps + 1, "train")
            remaining = max(0, decision_threshold - sampleable_count)
            per_sample_time = self._wait_seconds / self._wait_samples if self._wait_samples > 0 else None
            effective_switch_cost = self._effective_switch_cost()
            benefit = (
                remaining * per_sample_time * (1.0 - 1.0 / self._scaling_factor)
                if per_sample_time is not None
                else None
            )
            should_switch = (
                self.global_steps < self.total_training_steps
                and remaining > 0
                and (benefit is None or effective_switch_cost is None or benefit > effective_switch_cost)
            )
            decision_metrics = {
                "separate_async/switch/threshold_ratio": ratio_used,
                "separate_async/switch/wait_samples": float(self._step_wait_samples),
                "separate_async/switch/idle": float(had_idle),
                "separate_async/decision/sampleable_count": float(sampleable_count),
                "separate_async/decision/remaining": float(remaining),
                "separate_async/decision/should_switch_to_rollout": float(should_switch),
            }
            if per_sample_time is not None:
                decision_metrics["separate_async/decision/per_sample_time_seconds"] = per_sample_time
            if effective_switch_cost is not None:
                decision_metrics["separate_async/decision/effective_switch_cost_seconds"] = effective_switch_cost

        if should_switch:
            prepare_start = time.perf_counter()
            # Accumulate preparation and wake-up under one user-facing transition metric.
            with marked_timer("switch_to_rollout", self.timing_raw, color="cyan"):
                self.add_replicas_to_balancer()
                self.clear_sticky_cache()
            prepare_seconds = time.perf_counter() - prepare_start

        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = dict(
                self.standalone_checkpoint_manager.update_weights(self.global_steps) or {}
            )

        if config.enable_switch:
            self._pending_sync_metrics.update(decision_metrics)

            if should_switch:
                switch_start = time.perf_counter()
                with marked_timer("switch_to_rollout", self.timing_raw, color="cyan"):
                    logger.info("Switching hybrid engine to rollout mode for the next step")
                    self.switch_to_rollout(already_registered=True)
                self._to_rollout_costs.append(prepare_seconds + time.perf_counter() - switch_start)

    def _get_n_gpus_for_throughput(self) -> int:
        """Include standalone rollout GPUs in the throughput denominator."""
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        rollout_gpus = (
            self.config.actor_rollout_ref.rollout.n_gpus_per_node * self.config.actor_rollout_ref.rollout.nnodes
        )
        return trainer_gpus + rollout_gpus

    def switch_to_rollout(self, *, already_registered: bool = False):
        """Install committed weights and make Hybrid replicas available for generation."""
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()
        if not already_registered:
            self.add_replicas_to_balancer()
        self.current_mode = HybridEngineMode.ROLLOUT

    def switch_to_trainer(self):
        """Stop routing to Hybrid, abort partial requests, and return its GPU memory to training."""
        self.remove_replicas_from_balancer()
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()
        self.current_mode = HybridEngineMode.TRAINER

    def add_replicas_to_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        servers = dict(
            zip(self.llm_server_manager.server_addresses, self.llm_server_manager.server_handles, strict=True)
        )
        ray.get(global_load_balancer.add_servers.remote(servers))

    def remove_replicas_from_balancer(self):
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))

    def clear_sticky_cache(self) -> dict:
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        return ray.get(global_load_balancer.clear_sticky_cache.remote())
