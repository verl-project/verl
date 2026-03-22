"""Multi-tenant rollouter that extends FullyAsyncRollouter.

Creates per-tenant dataloaders, interleaves samples across tenants,
and routes generated samples to per-tenant MessageQueues.
"""

import asyncio
import time
from pprint import pformat

import numpy as np
import ray

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    TenantConfig,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRolllouterBase
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTenantRollouter(FullyAsyncRolllouterBase):
    """Multi-tenant rollouter: per-tenant data loading, generation with tenant LoRA, per-tenant queues."""

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

        # Use first tenant's data files for parent class init (it creates default dataloaders).
        # We'll override with per-tenant dataloaders below.
        from omegaconf import open_dict

        with open_dict(config):
            config.data.train_files = tenant_configs[0].train_file
            config.data.val_files = tenant_configs[0].val_file

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=device_name,
        )

        # Create per-tenant dataloaders, overriding the parent's single dataloader
        self._create_tenant_dataloaders(config, tokenizer, processor)

        # Per-tenant queue clients (set later by main)
        self.tenant_queue_clients: dict[str, MessageQueueClient] = {}

    def _create_tenant_dataloaders(self, config, tokenizer, processor):
        """Create per-tenant train dataloaders and validation datasets."""
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        self.tenant_dataloaders = {}
        self.tenant_val_datasets = {}

        for tc in self.tenant_configs:
            train_dataset = create_rl_dataset(
                tc.train_file,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("train_max_samples", -1),
            )
            val_dataset = create_rl_dataset(
                tc.val_file,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
            train_sampler = create_rl_sampler(config.data, train_dataset)

            # Create dataloader for this tenant
            from torchdata.stateful_dataloader import StatefulDataLoader

            train_dataloader = StatefulDataLoader(
                dataset=train_dataset,
                batch_size=config.data.gen_batch_size,
                num_workers=config.data.get("dataloader_num_workers", 0),
                sampler=train_sampler,
                drop_last=True,
                collate_fn=collate_fn,
            )

            self.tenant_dataloaders[tc.name] = train_dataloader
            self.tenant_val_datasets[tc.name] = val_dataset

            print(
                f"[MTRollouter] Created dataloader for tenant '{tc.name}': "
                f"train={len(train_dataset)} samples, val={len(val_dataset)} samples"
            )

        # Recalculate total rollout steps based on smallest tenant dataset
        min_steps = min(len(dl) for dl in self.tenant_dataloaders.values())
        # Total steps = per-tenant steps * num_tenants * epochs
        self.total_rollout_steps = min_steps * len(self.tenant_configs) * config.trainer.total_epochs
        if config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[MTRollouter] Total rollout steps (all tenants): {self.total_rollout_steps}")

    async def set_tenant_queue_clients(self, tenant_queue_clients: dict[str, MessageQueueClient]):
        """Set per-tenant message queue clients."""
        async with self.lock:
            self.tenant_queue_clients = tenant_queue_clients

    # Override: use the first tenant's queue as the default (for compatibility)
    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        pass  # Not used in multi-tenant mode

    def _create_continuous_tenant_iterator(self):
        """Create a continuous data iterator that interleaves across tenants and epochs.

        Yields (epoch, tenant_config, batch_dict) tuples, round-robin across tenants.
        """
        for epoch in range(self.config.trainer.total_epochs):
            # Create iterators for all tenants
            tenant_iters = {}
            for tc in self.tenant_configs:
                tenant_iters[tc.name] = iter(self.tenant_dataloaders[tc.name])

            # Round-robin across tenants until all are exhausted
            active_tenants = list(self.tenant_configs)
            while active_tenants:
                next_active = []
                for tc in active_tenants:
                    try:
                        batch_dict = next(tenant_iters[tc.name])
                        yield epoch, tc, batch_dict
                    except StopIteration:
                        print(f"[MTRollouter] Tenant '{tc.name}' exhausted in epoch {epoch}")
                        continue
                    next_active.append(tc)
                active_tenants = next_active

    async def _feed_samples(self):
        """Override: feed samples from all tenants, tagged with tenant_id and lora_int_id."""
        tenant_iterator = self._create_continuous_tenant_iterator()

        for epoch, tc, batch_dict in tenant_iterator:
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            # Tag the DataProto with tenant's lora_int_id for multi-LoRA generation
            full_batch.meta_info["_lora_int_id"] = tc.lora_int_id

            sample_id = f"sample_{tc.name}_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
                tenant_id=tc.name,
            )

            await self.pending_queue.put(rollout_sample)

            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[MTRollouter][Feed] Maximum count reached, stopping: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[MTRollouter][Feed] Sample addition complete, {self.global_steps} samples added")

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Override: generate with tenant LoRA and route to tenant's queue."""
        tenant_id = rollout_sample.tenant_id

        # Generate — the DataProto already has _lora_int_id in meta_info
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
        rollout_sample.full_batch = ret
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()

        # Route to the correct tenant's queue
        queue_client = self.tenant_queue_clients[tenant_id]
        success = await queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        if success:
            self.total_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    async def fit(self):
        """Override: check tenant queues are set before starting."""
        print("[MTRollouter] Starting MultiTenantRollouter...")

        if not self.tenant_queue_clients:
            raise ValueError("Tenant queue clients not set. Call set_tenant_queue_clients() first.")

        # Set running state
        async with self.lock:
            self.paused = False
            self.running = True

        # Reuse parent's streaming generation logic
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[MTRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[MTRollouter] MultiTenantRollouter fit completed")

    async def _streaming_generation_main(self):
        """Override: send termination signal to ALL tenant queues on completion."""
        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        print(f"[MTRollouter] Start streaming mode, max concurrent samples: {self.max_concurrent_samples}")

        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[MTRollouter] Sample feed completed")
            await self.processor_task
            print("[MTRollouter] Streaming process completed")
            await self.pending_queue.join()
            print("[MTRollouter] pending_queue joined")

        except Exception as e:
            print(f"[MTRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # Send finish signal to ALL tenant queues
            for name, client in self.tenant_queue_clients.items():
                await client.put_sample(sample=None)
                print(f"[MTRollouter] Sent termination signal to tenant '{name}' queue")

        async with self.lock:
            self.running = False

    async def _should_pause_generation(self) -> bool:
        """Override: check ALL tenant queues for fullness."""
        # Check if any tenant queue is full
        for name, client in self.tenant_queue_clients.items():
            queue_stats = client.get_statistics_sync()
            queue_size = queue_stats["queue_size"]
            if queue_size >= self.max_queue_size:
                if not self.paused:
                    print(
                        f"[MTRollouter][ShouldPause] Tenant '{name}' queue full: "
                        f"size={queue_size}, max={self.max_queue_size}"
                    )
                return True

        # Check global staleness
        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    f"[MTRollouter][ShouldPause] Global staleness: "
                    f"{self.staleness_samples} >= {self.max_required_samples}"
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        """Override: include per-tenant queue stats."""
        stats = {
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        for name, client in self.tenant_queue_clients.items():
            queue_stats = client.get_statistics_sync()
            stats[f"monitor/queue/{name}_queue_size"] = queue_stats["queue_size"]

        return stats
