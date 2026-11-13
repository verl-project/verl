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

import asyncio
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
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

    def run(self, config):
        import os
        import sys
        import time
        import traceback

        debug_path = "/root/work/tiankuan/zqc/logs/fully_async_actor_traceback.log"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        def debug_write(message):
            with open(debug_path, "a", buffering=1) as f:
                f.write(
                    f"\\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"PID={os.getpid()} {message}\\n"
                )
                f.flush()
                os.fsync(f.fileno())

        debug_write("TASK RUNNER ENTER")
        print("[ASYNC MAIN] Starting fully async PPO training...", flush=True)

        try:
            debug_write("BEFORE _initialize_components")
            self._initialize_components(config)
            debug_write("AFTER _initialize_components")

            debug_write("BEFORE _run_training_loop")
            self._run_training_loop()
            debug_write("TASK RUNNER FINISHED SUCCESSFULLY")

        except BaseException as exc:
            debug_write(
                f"TASK RUNNER FAILED: {type(exc).__name__}: {exc!r}"
            )

            with open(debug_path, "a", buffering=1) as f:
                traceback.print_exc(file=f)
                f.flush()
                os.fsync(f.fileno())

            print(
                f"[ASYNC MAIN] Full traceback saved to {debug_path}",
                file=sys.stderr,
                flush=True,
            )
            raise

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

        self._initialize_model_components(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _initialize_model_components(self, config) -> None:
        needs_hybrid = config.async_training.get("use_trainer_do_validate", False) or config.async_training.get(
            "use_dynamic_resource_scheduling", False
        )
        if config.async_training.get("parallel_initialization", False) and not needs_hybrid:
            print("[ASYNC MAIN] Initializing Trainer and Rollouter in parallel...")
            trainer_init = self._create_trainer(config, wait_for_init=False)
            rollouter_init = self._create_rollouter(config, wait_for_init=False)
            # Both init RPCs must be submitted before waiting. Weight synchronization
            # and fit still happen only after both components have initialized.
            ray.get([trainer_init, rollouter_init])
            ray.get(self.components["rollouter"].set_max_required_samples.remote())
            print("[ASYNC MAIN] Trainer and Rollouter initialized successfully")
        else:
            # Hybrid rollout needs the trainer's initialized worker group.
            if needs_hybrid and config.async_training.get("parallel_initialization", False):
                print("[ASYNC MAIN] Hybrid rollout requires sequential initialization")
            self._create_trainer(config)
            self._setup_hybrid_worker_group(config)
            self._create_rollouter(config)

    def _create_rollouter(self, config, wait_for_init=True):
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = FullyAsyncRollouter.remote(
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

        self.components["rollouter"] = rollouter
        init_future = rollouter.init_workers.remote()
        if wait_for_init:
            ray.get(init_future)
            ray.get(rollouter.set_max_required_samples.remote())
            print("[ASYNC MAIN] Rollouter created and initialized successfully")
        return init_future

    def _create_trainer(self, config, wait_for_init=True):
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        self.components["trainer"] = trainer
        init_future = trainer.init_workers.remote()
        if wait_for_init:
            ray.get(init_future)
            print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")
        return init_future

    def _setup_hybrid_worker_group(self, config) -> None:
        """Extract the trainer's actor_rollout_wg for hybrid rollout replicas.

        Hybrid replicas are needed when *either*:
        - ``use_trainer_do_validate`` is enabled (trainer-side validation), or
        - ``use_dynamic_resource_scheduling`` is enabled (dynamic resource scheduling).

        The extracted worker group is stored in ``self.components["hybrid_worker_group"]``
        and injected into the rollouter before ``init_workers()`` is called.
        """
        trainer = self.components["trainer"]
        needs_hybrid = config.async_training.use_trainer_do_validate or config.async_training.get(
            "use_dynamic_resource_scheduling", False
        )
        if needs_hybrid:
            trainer_wg = ray.get(trainer.get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                f"[ASYNC MAIN] Hybrid worker group extracted from trainer "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )
        else:
            print(
                "[ASYNC MAIN] Neither use_trainer_do_validate nor use_dynamic_resource_scheduling enabled, "
                "skipping hybrid worker group setup"
            )

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

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
            asyncio.run(self.components["message_queue_client"].clear_queue())
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
