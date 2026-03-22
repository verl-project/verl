"""Multi-tenant trainer that extends FullyAsyncTrainer.

Polls per-tenant MessageQueues, trains whichever tenant is ready first,
swaps LoRA adapter weights between tenants, and syncs per-tenant adapters
to vLLM replicas.
"""

import logging
import os
import time
from typing import Any

import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import (
    TenantConfig,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainerBase, TrainingStopException
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)

# Tenant version keys for save_model_to_cpu (high base to avoid collision with MIS correction)
_TENANT_VERSION_BASE = 100000


@ray.remote(num_cpus=10)
class MultiTenantTrainer(FullyAsyncTrainerBase):
    """Multi-tenant trainer: polls per-tenant queues, swaps adapters, per-tenant weight sync."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
        tenant_configs: list[TenantConfig] = None,
    ):
        if tenant_configs is None or len(tenant_configs) == 0:
            raise ValueError("tenant_configs must be provided and non-empty")

        self.tenant_configs = tenant_configs
        self.tenant_lora_map = {tc.name: tc.lora_int_id for tc in tenant_configs}

        # Per-tenant queue clients (set by main)
        self.tenant_queue_clients: dict[str, MessageQueueClient] = {}

        # Currently active tenant on the training GPUs
        self.active_tenant: str | None = None

        # Per-tenant training state (param versions, step counters)
        self.tenant_param_versions: dict[str, int] = {tc.name: 0 for tc in tenant_configs}

        # Track which tenant was last trained (for metrics)
        self.last_trained_tenant: str | None = None

        # Per-tenant termination tracking
        self.tenant_terminated: dict[str, bool] = {tc.name: False for tc in tenant_configs}

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=device_name,
        )

    def set_tenant_queue_clients(self, tenant_queue_clients: dict[str, MessageQueueClient]):
        """Set per-tenant message queue clients."""
        self.tenant_queue_clients = tenant_queue_clients

    # Override: not used in multi-tenant mode
    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        pass

    def _tenant_version_key(self, tenant_name: str) -> int:
        """Get a unique version key for save/restore_model_to/from_cpu."""
        idx = next(i for i, tc in enumerate(self.tenant_configs) if tc.name == tenant_name)
        return _TENANT_VERSION_BASE + idx

    def _init_tenant_model_states(self):
        """Save the initial model state for all tenants.

        All tenants start with the same initial adapter weights (from the base model + random LoRA init).
        We save a copy per tenant so we can swap between them later.
        """
        print("[MTTrainer] Initializing per-tenant model states on CPU...")
        for tc in self.tenant_configs:
            self.actor_rollout_wg.save_model_to_cpu(self._tenant_version_key(tc.name))
            print(f"[MTTrainer] Saved initial state for tenant '{tc.name}'")

        self.active_tenant = self.tenant_configs[0].name
        print(f"[MTTrainer] Active tenant set to '{self.active_tenant}'")

    def _switch_tenant(self, new_tenant: str):
        """Swap LoRA adapter weights to a different tenant."""
        if self.active_tenant == new_tenant:
            return

        print(f"[MTTrainer] Switching from tenant '{self.active_tenant}' to '{new_tenant}'")
        switch_start = time.time()

        # Save current tenant's model state
        if self.active_tenant is not None:
            self.actor_rollout_wg.save_model_to_cpu(self._tenant_version_key(self.active_tenant))

        # Restore new tenant's model state
        self.actor_rollout_wg.restore_model_from_cpu(self._tenant_version_key(new_tenant))
        self.active_tenant = new_tenant

        switch_time = time.time() - switch_start
        print(f"[MTTrainer] Tenant switch completed in {switch_time:.2f}s")

    async def _get_samples_from_queue(self):
        """Override: poll all tenant queues, pick the first with enough samples.

        Returns:
            tuple: (epoch, batch) or (None, None) if all tenants terminated.
        """
        print(
            f"[MTTrainer] Polling {len(self.tenant_queue_clients)} tenant queues "
            f"for {self.required_samples} samples...",
            flush=True,
        )

        poll_start = time.time()
        poll_count = 0

        while True:
            # Check if all tenants have terminated
            if all(self.tenant_terminated.values()):
                print("[MTTrainer] All tenant queues terminated")
                return None, None

            # Poll each active tenant's queue
            for tc in self.tenant_configs:
                if self.tenant_terminated[tc.name]:
                    continue

                queue_client = self.tenant_queue_clients[tc.name]
                stats = queue_client.get_statistics_sync()
                queue_size = stats["queue_size"]

                if queue_size >= self.required_samples:
                    # This tenant has enough samples — collect them
                    print(
                        f"[MTTrainer] Tenant '{tc.name}' ready with {queue_size} samples "
                        f"(need {self.required_samples})"
                    )

                    # Switch to this tenant's adapter
                    self._switch_tenant(tc.name)
                    self.last_trained_tenant = tc.name

                    # Collect samples from this tenant's queue
                    queue_samples = []
                    while len(queue_samples) < self.required_samples:
                        sample, queue_len = queue_client.get_sample_sync()
                        if sample is None:
                            print(f"[MTTrainer] Tenant '{tc.name}' terminated during collection")
                            self.tenant_terminated[tc.name] = True
                            break
                        queue_samples.append(sample)

                    if len(queue_samples) < self.required_samples:
                        print(
                            f"[MTTrainer] Tenant '{tc.name}' didn't provide enough samples: "
                            f"{len(queue_samples)}/{self.required_samples}"
                        )
                        continue

                    total_wait_time = time.time() - poll_start

                    queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
                    if self.config.trainer.balance_batch:
                        batch = assemble_batch_from_rollout_samples(
                            queue_samples, self.tokenizer, self.config, self._balance_batch
                        )
                    else:
                        batch = assemble_batch_from_rollout_samples(
                            queue_samples, self.tokenizer, self.config, None
                        )

                    batch.meta_info["fully_async/total_wait_time"] = total_wait_time
                    batch.meta_info["fully_async/active_tenant"] = tc.name
                    return 0, batch

            # No tenant ready yet, brief sleep before retrying
            poll_count += 1
            if poll_count % 100 == 0:
                active_tenants = [tc.name for tc in self.tenant_configs if not self.tenant_terminated[tc.name]]
                sizes = {}
                for name in active_tenants:
                    s = self.tenant_queue_clients[name].get_statistics_sync()
                    sizes[name] = s["queue_size"]
                print(f"[MTTrainer] Waiting for samples... Queue sizes: {sizes}")
            time.sleep(0.1)

    async def _fit_update_weights(self):
        """Override: sync the active tenant's LoRA adapter to vLLM replicas."""
        if self.local_trigger_step != 1:
            return

        if self.active_tenant is None:
            return

        tenant_name = self.active_tenant
        lora_int_id = self.tenant_lora_map[tenant_name]

        with marked_timer("timing_s/param_sync", self.timing_raw):
            # First: do the base model NCCL sync (same for all tenants, effectively a no-op for LoRA)
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)

            # Second: extract and sync this tenant's LoRA adapter to vLLM replicas
            await self._sync_tenant_lora_to_rollout(tenant_name)

        print(
            f"[MTTrainer] _fit_update_weights for tenant '{tenant_name}' (lora_int_id={lora_int_id}), "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f}s "
            f"current_param_version: {self.current_param_version}"
        )

        # Update tenant-specific param version
        self.tenant_param_versions[tenant_name] = self.current_param_version

        # Reset staleness in rollouter
        timing_raw = ray.get(self.rollouter.reset_staleness.remote())
        self.logger.log(data=timing_raw, step=self.current_param_version)

        # Log aggregated training metrics with tenant info
        agg_metrics = self.metrics_aggregator.get_aggregated_metrics()
        agg_metrics["fully_async/active_tenant_lora_id"] = lora_int_id
        self.logger.log(data=agg_metrics, step=self.current_param_version)
        self.metrics_aggregator.reset()

    async def _sync_tenant_lora_to_rollout(self, tenant_name: str):
        """Extract the current LoRA adapter and send it to vLLM replicas for this tenant."""
        lora_int_id = self.tenant_lora_map[tenant_name]

        # Extract LoRA weights from the training model (runs on all workers, take rank 0's result)
        results = self.actor_rollout_wg.get_lora_adapter_weights()
        lora_state_dict, peft_config = results[0]

        if lora_state_dict is None:
            print(f"[MTTrainer] Warning: no LoRA adapter found for tenant '{tenant_name}'")
            return

        print(
            f"[MTTrainer] Syncing LoRA adapter for tenant '{tenant_name}' "
            f"(lora_int_id={lora_int_id}, {len(lora_state_dict)} params) to vLLM replicas"
        )

        # Get vLLM server handles from the rollouter's replicas
        replicas = ray.get(self.rollouter.get_replicas.remote())
        sync_futures = []
        for replica in replicas:
            sync_futures.append(
                replica.server_handle.add_tenant_lora.remote(lora_int_id, peft_config, lora_state_dict)
            )

        ray.get(sync_futures)
        print(f"[MTTrainer] LoRA sync complete for tenant '{tenant_name}'")

    async def init_tenant_adapters_on_rollout(self):
        """Initialize all tenant LoRA adapters on vLLM replicas.

        Called once at startup after the base model is synced. Each tenant gets the
        same initial adapter weights (the model's initial LoRA state).
        """
        # Also initialize per-tenant model states on CPU
        self._init_tenant_model_states()

        # Extract initial LoRA weights (same for all tenants)
        results = self.actor_rollout_wg.get_lora_adapter_weights()
        lora_state_dict, peft_config = results[0]

        if lora_state_dict is None:
            print("[MTTrainer] Warning: no LoRA adapter found — multi-tenant requires LoRA")
            return

        replicas = ray.get(self.rollouter.get_replicas.remote())

        for tc in self.tenant_configs:
            print(f"[MTTrainer] Loading initial LoRA for tenant '{tc.name}' (lora_int_id={tc.lora_int_id})")
            sync_futures = []
            for replica in replicas:
                sync_futures.append(
                    replica.server_handle.add_tenant_lora.remote(tc.lora_int_id, peft_config, lora_state_dict)
                )
            ray.get(sync_futures)

        print(f"[MTTrainer] All {len(self.tenant_configs)} tenant LoRA adapters initialized on vLLM")

    def _collect_metrics_from_samples(self, batch, metrics):
        """Override: add tenant info to metrics."""
        super()._collect_metrics_from_samples(batch, metrics)
        if hasattr(batch, "meta_info") and batch.meta_info:
            tenant_name = batch.meta_info.get("fully_async/active_tenant")
            if tenant_name:
                metrics["fully_async/active_tenant"] = tenant_name
                metrics["fully_async/active_tenant_lora_id"] = self.tenant_lora_map.get(tenant_name, -1)

    def _save_checkpoint(self):
        """Override: save per-tenant checkpoints."""
        for tc in self.tenant_configs:
            local_global_step_folder = os.path.join(
                self.config.trainer.default_local_dir,
                tc.name,
                f"global_step_{self.tenant_param_versions[tc.name]}",
            )
            actor_local_path = os.path.join(local_global_step_folder, "actor")

            # Switch to this tenant's adapter before saving
            self._switch_tenant(tc.name)

            max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                None,
                self.tenant_param_versions[tc.name],
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )
            print(f"[MTTrainer] Saved checkpoint for tenant '{tc.name}' at {actor_local_path}")

        # Save critic (shared across tenants)
        if self.use_critic:
            critic_folder = os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
            )
            critic_local_path = os.path.join(critic_folder, str(Role.Critic))
            max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None)
            self.critic_wg.save_checkpoint(
                critic_local_path, None, self.current_param_version, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # Save rollouter dataloader state
        global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )
        ray.get(self.rollouter.save_checkpoint.remote(global_step_folder))

        # Write latest checkpoint marker
        local_latest = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest, "w") as f:
            f.write(str(self.current_param_version))
