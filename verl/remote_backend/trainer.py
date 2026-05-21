"""`RayPPOTrainer` subclass that drives a `RemoteBackend` instead of an
in-process engine. Resolves the backend by name, enforces single-forwarder
constraints when required, threads the reconnect handle into `wg_kwargs`.
"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import Dataset, Sampler

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry
from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role, WorkerType


class RemoteBackendTrainer(RayPPOTrainer):
    """PPO trainer that delegates train/rollout/log-prob/sync/ckpt to a
    :class:`RemoteBackend`.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        backend: Optional[RemoteBackend] = None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        if backend is None:
            backend_name = config.trainer.get("remote_backend")
            if not backend_name:
                raise ValueError(
                    "RemoteBackendTrainer requires trainer.remote_backend "
                    "to be set (e.g. trainer.remote_backend=arctic)."
                )
            backend = RemoteBackendRegistry.create(backend_name, config)
        self.backend: RemoteBackend = backend

        self._enforce_single_forwarder_if_required()

        self.use_gpu = False
        self.wg_kwargs["main_config"] = config
        self.wg_kwargs["backend_handle"] = self.backend.reconnect_handle()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _enforce_single_forwarder_if_required(self) -> None:
        """Honour the backend's :meth:`RemoteBackend.requires_single_forwarder`
        declaration: assert ``n_gpus_per_node × nnodes == 1`` and a single
        rollout replica. See the ABC docstring for the rationale."""
        if not self.backend.requires_single_forwarder():
            return

        n_gpus = self.config.trainer.n_gpus_per_node
        nnodes = self.config.trainer.nnodes
        if n_gpus * nnodes != 1:
            raise AssertionError(
                f"Remote backend {type(self.backend).__name__!r} requires a "
                "single forwarder worker, but the verl-side worker group "
                f"would have n_gpus_per_node={n_gpus} × nnodes={nnodes} = "
                f"{n_gpus * nnodes} workers. With more than one, ONE_TO_ALL "
                "calls duplicate against the backend and mesh-dispatched "
                "calls fragment the global batch. Set "
                "trainer.n_gpus_per_node=1 and trainer.nnodes=1 (the backend "
                "owns its own GPUs and parallelism) or override "
                "RemoteBackend.requires_single_forwarder() in your backend "
                "and validate the config yourself."
            )

        rollout_replicas = self._inferred_rollout_replica_count()
        if rollout_replicas != 1:
            raise AssertionError(
                f"Remote backend {type(self.backend).__name__!r} requires a "
                f"single rollout replica until the multi-replica path is "
                f"validated; got {rollout_replicas}. Set "
                "actor_rollout_ref.rollout.agent.num_workers=1 or override "
                "RemoteBackend.requires_single_forwarder() in your backend."
            )

    def _inferred_rollout_replica_count(self) -> int:
        """Best-effort introspection of the configured rollout-replica count.

        Different rollout backends use different config keys; we look at the
        few we know about and default to 1.
        """
        rollout_cfg = self.config.actor_rollout_ref.rollout
        for key_path in (("agent", "num_workers"), ("num_workers",), ("replicas",)):
            cur = rollout_cfg
            for part in key_path:
                cur = cur.get(part) if hasattr(cur, "get") else None
                if cur is None:
                    break
            if isinstance(cur, int):
                return cur
        return 1

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def destroy(self) -> None:
        if getattr(self, "backend", None) is not None:
            # `RemoteBackend.destroy` is async (so adapters can await
            # remote RPCs without blocking event loops). The trainer's
            # `destroy` is called from synchronous shutdown paths, so
            # bridge with `asyncio.run`.
            import asyncio

            asyncio.run(self.backend.destroy())
            self.backend = None
