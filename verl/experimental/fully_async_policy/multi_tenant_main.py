"""Multi-tenant fully async training entry point.

Extends the fully_async_main orchestrator to support multiple tenants,
each training their own LoRA adapter on a shared base model.
"""

import asyncio
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import TenantConfig, parse_tenants
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.fully_async_policy.multi_tenant_rollouter import MultiTenantRollouter
from verl.experimental.fully_async_policy.multi_tenant_trainer import MultiTenantTrainer
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class MultiTenantTaskRunner:
    """Ray remote class for executing multi-tenant distributed PPO training."""

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[MT MAIN] Starting multi-tenant fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[MT MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Parse tenant configurations
        tenants_str = config.multi_tenant.tenants
        tenant_configs = parse_tenants(tenants_str)
        self.components["tenant_configs"] = tenant_configs
        print(f"[MT MAIN] Parsed {len(tenant_configs)} tenants: {[t.name for t in tenant_configs]}")

        print("[MT MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[MT MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        from concurrent.futures import ThreadPoolExecutor

        print("[MT MAIN] Creating MultiTenantRollouter and MultiTenantTrainer in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            trainer_future = executor.submit(self._create_trainer, config)
            trainer_future.result()

            rollouter_future = executor.submit(self._create_rollouter, config)
            rollouter_future.result()

        # Sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"[MT MAIN] total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # Create per-tenant message queues
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[MT MAIN] Creating per-tenant MessageQueues... max_queue_size {max_queue_size}")

        tenant_queues = {}
        tenant_queue_clients = {}
        for tc in tenant_configs:
            mq = MessageQueue.remote(config, max_queue_size)
            tenant_queues[tc.name] = mq
            tenant_queue_clients[tc.name] = MessageQueueClient(mq)

        self.components["tenant_queues"] = tenant_queues
        self.components["tenant_queue_clients"] = tenant_queue_clients

        # Set per-tenant queue clients on rollouter and trainer
        ray.get(self.components["rollouter"].set_tenant_queue_clients.remote(tenant_queue_clients))
        ray.get(self.components["trainer"].set_tenant_queue_clients.remote(tenant_queue_clients))

        # Load checkpoints
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        # Setup parameter synchronization
        print("[MT MAIN] Setting up parameter synchronization...")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # Initial weight sync (base model via NCCL)
        print("[MT MAIN] Initial base model param sync...")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        # Initialize per-tenant LoRA adapters on vLLM replicas
        print("[MT MAIN] Initializing per-tenant LoRA adapters on vLLM...")
        ray.get(self.components["trainer"].init_tenant_adapters_on_rollout.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[MT MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        print("[MT MAIN] Starting create rollouter...")
        rollouter = MultiTenantRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            tenant_configs=self.components["tenant_configs"],
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[MT MAIN] MultiTenantRollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[MT MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = MultiTenantTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            tenant_configs=self.components["tenant_configs"],
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[MT MAIN] MultiTenantTrainer created and initialized successfully")

    def _run_training_loop(self):
        self.running = True

        print("[MT MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[MT MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[MT MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[MT MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            # Clear all tenant queues
            for name, client in self.components["tenant_queue_clients"].items():
                asyncio.run(client.clear_queue())
            print("[MT MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    assert config.async_training.use_trainer_do_validate is False, "use_trainer_do_validate is not ready to use."

    if not hasattr(config, "multi_tenant") or not config.multi_tenant.get("tenants"):
        raise RuntimeError("must set multi_tenant.tenants config (e.g. 'alice:train.parquet:val.parquet,bob:...')")

    from time import time

    start_time = time()
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=MultiTenantTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
