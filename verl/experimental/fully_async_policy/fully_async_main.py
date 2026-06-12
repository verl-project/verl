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

"""FullyAsyncTaskRunner: Unified entry point for fully async PPO training.

Supports two data transfer backends (selected via config.transfer_queue.enable):
- transfer_queue.enable=False (default): MessageQueue + FullyAsyncRollouter/FullyAsyncTrainer
- transfer_queue.enable=True:  TransferQueue + ReplayBuffer + TQFullyAsyncRollouter/TQFullyAsyncTrainer
"""

import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()
        self._use_tq = False
        # Class references (resolved in run() based on transfer_queue.enable)
        self._rollouter_cls = None
        self._trainer_cls = None

    def run(self, config):
        # Detect backend mode from config and resolve class references
        self._use_tq = config.transfer_queue.enable
        if self._use_tq:
            from verl.experimental.fully_async_policy.fully_async_rollouter_tq import FullyAsyncRollouterTQ
            from verl.experimental.fully_async_policy.fully_async_trainer_tq import FullyAsyncTrainerTQ

            self._rollouter_cls = ray.remote(FullyAsyncRollouterTQ).options(num_cpus=10, max_concurrency=100)
            self._trainer_cls = ray.remote(FullyAsyncTrainerTQ).options(num_cpus=10, max_concurrency=100)

            # Initialize TQ in the main process FIRST with config.transfer_queue
            # This ensures all subsequent tq.init() calls (in Rollouter, Trainer, RB actors)
            # connect to the SAME shared TransferQueueController instance.
            try:
                import transfer_queue as tq

                tq_config = OmegaConf.to_container(getattr(config, "transfer_queue", {}), resolve=True)
                print(f"[ASYNC MAIN] Initializing TQ with config: {tq_config}", flush=True)
                tq.init(tq_config)
                print("[ASYNC MAIN] TQ initialized in main process", flush=True)
            except Exception as e:
                print(f"[ASYNC MAIN] TQ init warning: {e}", flush=True)
        else:
            from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
            from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

            self._rollouter_cls = ray.remote(FullyAsyncRollouter).options(num_cpus=10, max_concurrency=100)
            self._trainer_cls = ray.remote(FullyAsyncTrainer).options(num_cpus=10, max_concurrency=100)

        mode_label = "TQ" if self._use_tq else "MQ"
        print(f"[ASYNC MAIN] Starting fully async PPO training (mode={mode_label})...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        print("[ASYNC MAIN] Creating Trainer first (needed for hybrid worker group injection)...")
        self._create_trainer(config)

        print("[ASYNC MAIN] Injecting trainer's worker group into rollouter for hybrid replicas...")
        self._setup_hybrid_worker_group(config)

        print("[ASYNC MAIN] Creating Rollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # ======== Create data channel (MQ or RB) ========
        self._create_data_channel(config)

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_data_channel(self, config):
        """Create the data channel: MessageQueue (default) or ReplayBuffer (TQ mode)."""
        if self._use_tq:
            self._create_replay_buffer(config)
        else:
            self._create_message_queue(config)

    def _create_message_queue(self, config):
        """Create MessageQueue + MessageQueueClient (original path)."""
        from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient

        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

    def _create_replay_buffer(self, config):
        """Create ReplayBuffer Ray Actor (TQ mode) with dual-layer slot config.

        Layer 1 (Physical): max_pending_slots = max_concurrent_samples
            Limits simultaneous in-flight samples (OOM guard).
            Maps to Rollouter's max_concurrent_samples (e.g. GPU / (TP * PP) * 16).

        Layer 2 (Version window): max_version_slots = max_required_samples
            Limits total slots per model version (staleness guard).
            Maps to Rollouter's max_required_samples
            (= required_samples * trigger_parameter_sync_step).
        """
        from verl.experimental.fully_async_policy.replay_buffer import ReplayBuffer

        # Layer 1: Physical concurrency limit
        max_concurrent_samples = ray.get(self.components["rollouter"].get_max_concurrent_samples.remote())
        # Layer 2: Version window (staleness) limit
        max_required_samples = ray.get(self.components["rollouter"].get_max_required_samples.remote())

        print(
            f"[ASYNC MAIN] Creating ReplayBuffer... "
            f"max_pending_slots(physical)={max_concurrent_samples}, "
            f"max_version_slots(staleness)={max_required_samples}"
        )
        replay_buffer = ReplayBuffer.remote(
            max_version_slots=max_required_samples,
            max_pending_slots=max_concurrent_samples,
        )
        self.components["replay_buffer"] = replay_buffer

        ray.get(self.components["rollouter"].set_replay_buffer.remote(replay_buffer))
        ray.get(self.components["trainer"].set_replay_buffer.remote(replay_buffer))

    def _create_rollouter(self, config) -> None:
        """Create rollouter: FullyAsyncRollouter (default) or TQFullyAsyncRollouter (TQ mode)."""

        rollouter = self._rollouter_cls.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        """Create trainer: FullyAsyncTrainer (default) or TQFullyAsyncTrainer (TQ mode)."""
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = self._trainer_cls.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] Trainer created and initialized successfully")

    def _setup_hybrid_worker_group(self, config) -> None:
        """
        Extract the trainer's actor_rollout_wg and store it for later injection
        into the rollouter. This WG backs the hybrid rollout replicas
        used during trainer-side validation (use_trainer_do_validate).
        """
        trainer = self.components["trainer"]
        if config.async_training.use_trainer_do_validate:
            trainer_wg = ray.get(trainer.get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                f"[ASYNC MAIN] Hybrid worker group extracted from trainer "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )
        else:
            print("[ASYNC MAIN] use_trainer_do_validate=False, skipping hybrid worker group setup")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        print(f"[ASYNC MAIN] rollouter handle: {self.components['rollouter']}", flush=True)
        print(f"[ASYNC MAIN] trainer handle: {self.components['trainer']}", flush=True)
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()
        print(
            f"[ASYNC MAIN] fit.remote() submitted, rollouter_future={rollouter_future},"
            f" trainer_future={trainer_future}",
            flush=True,
        )

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
