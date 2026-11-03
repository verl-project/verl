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
"""Pieces shared by v1 trainers whose rollout runs on its own GPUs (separate_async, decoupled_async)."""

from copy import deepcopy

import ray
from transfer_queue import KVBatchMeta

from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.trainer.ppo.utils import Role
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager
from verl.workers.rollout.utils import update_prometheus_config


class StandaloneRolloutMixin:
    """Standalone rollout replicas and Decoupled PPO's stable π_old; mix in before PPOTrainer."""

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        # Replace ActorRolloutRefWorker with DetachActorWorker to get CPU save/restore
        # capability needed for Decoupled PPO when parameter_sync_step > 1.
        # The base class uses a pure Actor role when hybrid replicas are disabled.
        if Role.Actor in self.role_worker_mapping:
            self.role_worker_mapping[Role.Actor] = ray.remote(DetachActorWorker)
            if Role.RefPolicy in self.role_worker_mapping:
                self.role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)
        elif Role.ActorRolloutRef in self.role_worker_mapping:
            self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(DetachActorWorker)
        elif Role.ActorRollout in self.role_worker_mapping:
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(DetachActorWorker)

    def _setup_standalone_rollout(self) -> None:
        """Launch the standalone replicas, ranked after any hybrid replicas."""
        # TODO: make initialization parallel with super().init()
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
        # Standalone replicas have dedicated GPUs; keep their memory budget
        # separate from the hybrid replicas that share GPUs with training.
        standalone_config = deepcopy(self.config)
        standalone_rollout_config = standalone_config.actor_rollout_ref.rollout
        standalone_memory = standalone_rollout_config.get("standalone_gpu_memory_utilization")
        if standalone_memory is not None:
            standalone_rollout_config.gpu_memory_utilization = standalone_memory
        self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
            config=standalone_config, start_rank=hybrid_num_replicas
        )
        rollout_config = self.config.actor_rollout_ref.rollout
        if rollout_config.prometheus.enable:
            server_addresses = (
                self.llm_server_manager.server_addresses + self.standalone_server_manager.server_addresses
            )
            update_prometheus_config(rollout_config.prometheus, server_addresses, rollout_config.name)

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

    def _get_n_gpus_for_throughput(self) -> int:
        """Include standalone rollout GPUs in the throughput denominator."""
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        rollout_gpus = (
            self.config.actor_rollout_ref.rollout.n_gpus_per_node * self.config.actor_rollout_ref.rollout.nnodes
        )
        return trainer_gpus + rollout_gpus

    def get_llm_client(self):
        # get server client from standalone rollout
        return self.standalone_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def _rollout_server_managers(self) -> list:
        managers = super()._rollout_server_managers()
        standalone = getattr(self, "standalone_server_manager", None)
        if standalone is not None:
            managers.append(standalone)
        return managers
