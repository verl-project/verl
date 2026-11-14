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

import ray
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.trainer.ppo.utils import need_reward_model
from verl.trainer.ppo.v1.prompt_dispatcher import PromptDispatcher
from verl.trainer.ppo.v1.standalone_rollout import StandaloneRolloutMixin
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("decoupled_async")
class PPOTrainerDecoupledAsync(StandaloneRolloutMixin, PPOTrainer):
    """Asynchronous PPO trainer that never drives rollout.

    The trainer only publishes weight versions and enqueues prompts in TransferQueue:
    1. Each rollout replica's WeightUpdateAgent loads newer versions on its own schedule.
    2. A PromptDispatcher starts enqueued prompts once every replica has the version they require.
    3. The replay buffer evicts groups by the weight version that generated them.
    """

    _enable_hybrid_replicas = False

    def __init__(self, config: DictConfig):
        mode_config = config.trainer.v1.decoupled_async
        rollout = config.actor_rollout_ref.rollout
        assert rollout.checkpoint_engine.backend == "transport_pull", (
            "decoupled_async needs actor_rollout_ref.rollout.checkpoint_engine.backend=transport_pull"
        )
        assert rollout.name == "vllm", "decoupled_async supports vLLM rollout only"
        assert rollout.nnodes > 0 and rollout.n_gpus_per_node > 0, "decoupled_async needs standalone rollout GPUs"
        train_batch_size = config.data.train_batch_size
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        assert train_batch_size == mode_config.parameter_sync_step * ppo_mini_batch_size, (
            f"train_batch_size ({train_batch_size}) must equal parameter_sync_step "
            f"({mode_config.parameter_sync_step}) * ppo_mini_batch_size ({ppo_mini_batch_size})"
        )
        assert mode_config.min_version_lag >= 1
        threshold = config.trainer.v1.sampler.max_off_policy_threshold
        assert threshold is None or mode_config.min_version_lag < threshold, (
            "min_version_lag must stay below max_off_policy_threshold, or rollout produces samples that are dropped"
        )
        if need_reward_model(config):
            assert config.reward.reward_model.enable_resource_pool, (
                "decoupled_async needs a standalone reward model (reward.reward_model.enable_resource_pool=True)"
            )
        super().__init__(config)
        self.mode_config = mode_config
        self._published_version = -1
        self.weight_update_agents: list = []
        self._weight_update_agent_runs: list = []
        self.prompt_dispatcher = None
        self._prompt_dispatcher_run = None

    def _setup(self):
        super()._setup()
        self._setup_standalone_rollout()
        self._start_weight_update_agents()
        # A failed agent or dispatcher stops new samples; fail the job instead of waiting in sample().
        self.replay_buffer.health_check_fn = self._check_rollout

    def _start_weight_update_agents(self) -> None:
        from verl.checkpoint_engine.transport.update_agent import WeightUpdateAgent

        engine_kwargs = self.config.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs
        registry_name = engine_kwargs.get("transport_pull", {}).get("registry_name", "verl_weight_registry")
        agent_cls = ray.remote(num_cpus=0)(WeightUpdateAgent)
        self.weight_update_agents = [
            agent_cls.remote(
                servers=replica.servers,
                workers=replica.workers,
                registry_name=registry_name,
                min_version_lag=self.mode_config.min_version_lag,
                poll_interval=self.mode_config.poll_interval,
            )
            for replica in self.standalone_server_manager.get_replicas()
        ]
        # Each loop runs until stopped; a ref only resolves early if its actor fails.
        self._weight_update_agent_runs = [agent.run.remote() for agent in self.weight_update_agents]

    def _start_prompt_dispatcher(self) -> None:
        dispatcher_cls = ray.remote(num_cpus=0)(PromptDispatcher)
        self.prompt_dispatcher = dispatcher_cls.remote(
            agent_loop_workers=self.agent_loop_manager.agent_loop_workers,
            update_agents=self.weight_update_agents,
            max_inflight_prompts=self.mode_config.max_inflight_prompts,
            poll_interval=self.mode_config.dispatch_poll_interval,
        )
        self._prompt_dispatcher_run = self.prompt_dispatcher.run.remote()

    def _rollout_runs(self) -> list:
        runs = list(self._weight_update_agent_runs)
        if self._prompt_dispatcher_run is not None:
            runs.append(self._prompt_dispatcher_run)
        return runs

    def _check_rollout(self) -> None:
        finished, _ = ray.wait(self._rollout_runs(), num_returns=1, timeout=0)
        if finished:
            ray.get(finished)  # re-raises the actor's error
            raise RuntimeError("A weight update agent or the prompt dispatcher exited unexpectedly")

    def _min_weight_version(self) -> int:
        # Replicas may trail the newest version by min_version_lag - 1 before they update.
        return self._published_version - (self.mode_config.min_version_lag - 1)

    def _publish_weights(self) -> dict:
        """Publish the current weights as version global_steps; replicas load it themselves."""
        self._check_rollout()
        ray.get(self.actor_rollout_wg.update_weights(global_steps=self.global_steps))
        self._published_version = self.global_steps
        states = ray.get([agent.get_state.remote() for agent in self.weight_update_agents])
        versions = [state["version"] for state in states]
        metrics = {
            "weight_update/published_version": self.global_steps,
            "weight_update/min_replica_version": min(versions),
            "weight_update/max_replica_version": max(versions),
        }
        for key in ("weight_update/prefetch_seconds", "weight_update/pause_seconds"):
            values = [state[key] for state in states if key in state]
            if values:
                metrics[f"{key}_max"] = max(values)
        if self.prompt_dispatcher is not None:
            dispatcher = ray.get(self.prompt_dispatcher.get_state.remote())
            metrics["dispatcher/inflight"] = dispatcher["inflight"]
            metrics["dispatcher/waiting_for_version"] = dispatcher["waiting_for_version"]
        return metrics

    def _submit_batch_to_rollout(self, batch: TensorDict) -> int:
        """Enqueue prompts for the dispatcher; nothing is sent to rollout from here."""
        tag = {
            "is_prompt": True,
            "status": "pending",
            "global_steps": self.global_steps,
            "min_weight_version": self._min_weight_version(),
        }
        tq.kv_batch_put(
            keys=list(batch["uid"]),
            partition_id="train",
            tags=[dict(tag) for _ in range(len(batch))],
            fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
        )
        return len(batch)

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        """Put checkpointed pending/running prompt groups back in the queue for the dispatcher."""
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {})
        inflight_uids = [
            key
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        ]
        if not inflight_uids:
            return 0
        inflight_uid_set = set(inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in inflight_uid_set
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)
        tag = {
            "is_prompt": True,
            "status": "pending",
            "global_steps": self.global_steps,
            "min_weight_version": self._min_weight_version(),
        }
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=[dict(tag) for _ in inflight_uids])
        logger.info(
            f"Re-queued {len(inflight_uids)} in-flight prompts; cleared {len(old_trajectory_keys)} trajectories"
        )
        return len(inflight_uids)

    def on_init_end(self):
        self._publish_weights()
        # Generation must not start on weights older than the checkpoint that was loaded.
        timeout = self.mode_config.init_timeout
        ray.get([agent.wait_for_version.remote(self.global_steps, timeout) for agent in self.weight_update_agents])

    def on_validate_begin(self):
        # Validation sends prompts straight to the agent loop workers, without the dispatcher's version
        # gating, so first make every replica serve the weights being validated.
        timeout = self.mode_config.init_timeout
        ray.get(
            [agent.wait_for_version.remote(self._published_version, timeout) for agent in self.weight_update_agents]
        )

    def on_train_begin(self):
        self._start_prompt_dispatcher()
        self._add_async_warmup_batches(self.mode_config.num_warmup_batches)

    def on_sample_end(self):
        # Other modes pause or sleep rollout here; in this mode rollout keeps generating.
        return

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = self._publish_weights()

    def on_train_end(self):
        super().on_train_end()
        actors = list(self.weight_update_agents)
        if self.prompt_dispatcher is not None:
            actors.append(self.prompt_dispatcher)
        ray.get([actor.stop.remote() for actor in actors])
        # Each loop returns within one poll interval, or after the update it is applying.
        ray.get(self._rollout_runs(), timeout=300)
