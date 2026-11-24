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
import asyncio
import concurrent.futures
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.import_utils import import_external_libs
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, RolloutReplica, get_rollout_class
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__name__)

# Fault-tolerant weight-sync tunables (see enable_fault_tolerance): bounded
# commit wait, post-failure detection grace, max attempts, finalize bound.
_BOUNDED_ABORT_S = 120.0
_PROBE_GRACE_S = 30.0
_MAX_SYNC_ATTEMPTS = 3
_FINALIZE_TIMEOUT_S = 30.0

# Zero-syncable-members wait (single-replica / all-restarting deployments):
# poll cadence and overall bound while the restart pipeline runs. These waits
# are not sync attempts — there is nothing to transfer to while nobody is
# syncable, so they must not consume the attempt budget.
_EMPTY_MEMBERSHIP_POLL_S = 10.0
_RECOVERY_WAIT_TIMEOUT_S = 1800.0


@dataclass
class TensorMeta:
    name: str
    """The name of the weight tensor."""
    shape: torch.Size
    """The shape of the weight tensor."""
    dtype: torch.dtype
    """The dtype of the weight tensor."""
    chunk_offset: int
    """The chunk offset of the weight tensor."""
    chunk_size: int
    """The chunk size of the weight tensor."""
    offset: int
    """The offset of the weight tensor in the bucket."""


class CheckpointEngineRegistry:
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    # Engine modules whose import failed, keyed by module name. Each engine pulls
    # its own transport dependency (cupy for nccl/nixl, nixl, torch_npu for hccl,
    # ...) and `verl.checkpoint_engine` imports them all optionally, so a missing
    # dependency would otherwise only show up as an unregistered backend.
    _import_errors: dict[str, ImportError] = {}

    def register(backend: str):
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def record_import_error(cls, module: str, error: ImportError):
        """Record an engine module that could not be imported.

        Args:
            module: The name of the checkpoint engine module.
            error: The import error raised by the module.
        """
        cls._import_errors[module] = error

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        """Get the checkpoint engine class.

        Args:
            backend: The backend of the checkpoint engine.

        Returns:
            The checkpoint engine class.
        """
        if backend not in cls._registry:
            message = f"Checkpoint engine {backend} not registered, registered backends: {sorted(cls._registry)}"
            if cls._import_errors:
                unavailable = ", ".join(f"{module}: {error}" for module, error in sorted(cls._import_errors.items()))
                message += f". Engine modules that failed to import: {unavailable}"
            raise ValueError(message)
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        return cls.get(backend)(*args, **kwargs)


class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from actor to rollout.

    In actor process:
    >>> actor = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(actor.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    # Whether send_weights() publishes an immutable trainer-side snapshot independently
    # of rollout consumers. When True, the manager may return after publishing while
    # rollout commits the snapshot in the background. Coupled transports such as NCCL
    # and NIXL must keep this False so update_weights() waits for rollout commit.
    support_publish_weights = False

    # How receive_weights yields weights to the server adapter:
    #   "named_tensors" -- (name, tensor) pairs, bucketed into full-tensor loads.
    #   "delta_flush"   -- (named_tensors, is_last) flushes; the seed may be dense,
    #                       while steady updates carry sparse patches.
    wire_format = "named_tensors"

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare checkpoint engine before each step send_weights/receive_weights.

        1. Allocate weight bucket.
        2. [Optional] Register weight bucket for RDMA.
        3. Return metadata to build communication topology: master ip:port, register RDMA description, etc.

        Args:
            worker_group: The worker group that the checkpoint engine will be used.

        Returns:
            A dictionary that contains the metadata of the worker group.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology between all workers.

        Args:
            actor_wg_world_size: The world size of the actor worker group.
            rollout_world_size: The world size of the rollout replica.
            metadata: A list of metadata `prepare` from all workers.

        Returns:
            A tuple of two dictionaries that contains the communication topology for actor and rollout worker group.
            Each dict value should be a list argument equal to the world size of the worker group to dispatch to
            `init_process_group`.

            ```
            world_size = rollout.world_size + actor_wg.world_size
            kwargs = {
                "rank": list(range(world_size)),
                "world_size": [world_size] * world_size,
                "master_metadata": [metadata[0]] * world_size,
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Init process group for checkpoint engine.

        Args:
            **kwargs: Keyword arguments from `build_topology`.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights.

        1. Free weight bucket.
        1. [Optional] Deregister weight bucket for RDMA.
        2. [Optional] Destroy process group.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Args:
            global_steps: Optional trainer step/version associated with this weight update.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for actor and rollout colocated on same GPU.

    In actor process:
    >>> engine = ColocatedCheckpointEngine()
    >>> actor = Actor()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(actor.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def __init__(self, bucket_size: int, is_master: bool = False) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        raise NotImplementedError

    def init_process_group(self, **kwargs):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError

    def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        self.weights = weights

    def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Args:
            global_steps: Optional trainer step/version associated with this weight update.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None


class CheckpointEngineWorker(Worker):
    """CheckpointEngineWorker colocated with inference engine's WorkerProc on same GPU.

    Args:
        rollout_config: The rollout configuration.
        model_config: The model configuration.
        server_adapter: The server adapter to update weights.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rollout_config = rollout_config
        self.model_config = model_config

        self.server_adapter: BaseRollout = server_adapter
        backend = self.rollout_config.checkpoint_engine.backend
        if backend == "delta_sharded" and self.rollout_config.name not in {"sglang", "vllm"}:
            raise NotImplementedError(
                f"checkpoint_engine.backend={backend!r} has no delta weight consumer for "
                f"rollout.name={self.rollout_config.name!r}; use sglang or vllm"
            )
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        # If custom_backend_module is set, import it so plugins can register
        # in CheckpointEngineRegistry before the backend is instantiated.
        import_external_libs(self.rollout_config.checkpoint_engine.custom_backend_module or None)
        self.checkpoint_engine: CheckpointEngine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )
        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
        await self.server_adapter.update_weights(
            weights,
            global_steps=global_steps,
            wire_format=getattr(self.checkpoint_engine, "wire_format", "named_tensors"),
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_replica_rank(self) -> int:
        """Get replica rank from the underlying rollout server adapter."""
        return self.server_adapter.replica_rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def is_leader_rank(self) -> bool:
        """Get leader rank flag from the underlying rollout server adapter."""
        return self.server_adapter.is_leader_rank


_worker_cls = ray.remote(CheckpointEngineWorker)


@dataclass
class WeightUpdateTicket:
    version: int | None
    trainer_published: list[ray.ObjectRef]
    rollout_committed: concurrent.futures.Future

    async def wait_published(self):
        """Wait until trainer workers have published this weight version."""
        return await asyncio.gather(*self.trainer_published)

    async def wait_committed(self):
        """Wait until rollout workers have installed this weight version."""
        return await asyncio.wrap_future(self.rollout_committed)


class CheckpointEngineManager:
    """Checkpoint engine manager to coordinate weight synchronization between actor and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In actor, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.

    ```
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
    │ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
    │ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
    │    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
    | ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
    │ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
    │ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
    └────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
         v                                        |    |    |    |    |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
    ```

    Design notes:

    1. Coupled strong synchronization (``support_publish_weights=False``), such as NCCL/NIXL:
       the trainer sender and rollout receiver must participate in the same transfer. The manager waits for the
       previous update to commit, aborts in-flight generation, releases the rollout KV cache, builds the communication
       topology, and starts trainer send plus rollout receive. ``update_weights`` returns only after rollout installs
       the version, both sides finalize the transfer, and rollout generation resumes.

    2. Decoupled publish/commit (``support_publish_weights=True``), such as a trainer D2H snapshot followed by an
       asynchronous rollout RDMA read: after synchronously building the control-plane topology, the manager starts
       trainer publish and schedules rollout commit on its background event loop. ``update_weights`` returns once the
       immutable trainer snapshot is published; in parallel, the background task aborts generation, releases the KV
       cache, fetches and installs that version, finalizes the transfer, and resumes rollout.

    The manager allows at most one uncommitted version. Every new update first waits for the previous rollout commit,
    which bounds snapshot lifetime and prevents versions from being installed out of order.

    Args:
        config: The checkpoint engine config.
        actor_wg: The actor worker group (the training side that produces weights).
        replicas: The list of rollout replicas.
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        actor_wg: RayWorkerGroup,
        replicas: list[RolloutReplica],
    ) -> None:
        self.config = config
        self.backend = config.backend
        import_external_libs(self.config.custom_backend_module or None)
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        self.actor_wg = actor_wg
        self.replicas = replicas

        self._update_ticket = None
        # Fault-tolerance state, inert until enable_fault_tolerance() (the
        # rollout-restart wiring): fail-fast sync, no fencing. Membership is
        # written only via sync_membership().
        self._fault_tolerant = False
        self._pre_sync_hook = None
        self._sync_succeeded_callback = None
        self._group_generation = 0
        self._retry_epoch = 0
        self._fatal_error: str | None = None
        self._background_loop = asyncio.new_event_loop()
        self._background_loop_ready = threading.Event()
        self._background_loop_thread = threading.Thread(
            target=self._run_background_loop,
            name="checkpoint-engine-manager",
            daemon=True,
        )
        self._background_loop_thread.start()
        if not self._background_loop_ready.wait(timeout=5):
            self.close()
            raise RuntimeError("Failed to start CheckpointEngineManager background event loop.")

    def enable_fault_tolerance(
        self,
        pre_sync_hook=None,
        sync_succeeded_callback=None,
    ) -> None:
        """Activate wait-and-retry weight sync (idempotent; called once by the
        rollout-restart wiring).

        ``pre_sync_hook``: async, reconciles membership before every attempt
        (a fresh detection round on the manager, then
        manager snapshot -> :meth:`sync_membership`).
        ``sync_succeeded_callback``: fired after a successful sync so the
        manager promotes rebuilt replicas; a lost notification is retried by
        the next sync.
        """
        self._fault_tolerant = True
        self._pre_sync_hook = pre_sync_hook
        self._sync_succeeded_callback = sync_succeeded_callback

    @staticmethod
    def _replica_key(replica: RolloutReplica) -> str:
        return f"{replica.replica_rank}{getattr(replica, 'name_suffix', '') or ''}"

    def sync_membership(self, snapshot) -> None:
        """Adopt the manager-authoritative membership snapshot (replicas,
        fencing generation, fatal flag) — the only write boundary for this
        state."""
        self.replicas = list(snapshot.replicas)
        self._group_generation = snapshot.group_generation
        self._fatal_error = snapshot.fatal_error

    def _run_background_loop(self):
        asyncio.set_event_loop(self._background_loop)
        self._background_loop_ready.set()
        self._background_loop.run_forever()

    def close(self):
        """Stop and close the background event loop."""
        loop = getattr(self, "_background_loop", None)
        thread = getattr(self, "_background_loop_thread", None)
        if loop is None or thread is None:
            return

        if thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
        if not loop.is_closed():
            loop.close()

    def build_process_group(self, rollout: RayWorkerGroup):
        """Build process group for actor worker group and rollout replicas."""
        actor_wg = self.actor_wg

        # 1. prepare all workers
        metadata = ray.get(
            actor_wg.execute_checkpoint_engine(["prepare"] * actor_wg.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )

        # 2. build communication topology between all workers
        actor_wg_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            actor_wg.world_size, rollout.world_size, metadata
        )
        for k, v in actor_wg_kwargs.items():
            assert len(v) == actor_wg.world_size, f"actor_wg_kwargs[{k}] must have length of {actor_wg.world_size}"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] must have length of {rollout.world_size}"

        actor_wg_kwargs["method"] = ["init_process_group"] * actor_wg.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size

        # Membership fencing: generation = manager conviction count + retry
        # epoch, so a membership change or a failed attempt builds under a
        # never-used name (each worker destroys its previous group on the
        # rename). Zero (or fault tolerance disabled) sends no extra kwargs.
        fence = self._group_generation + self._retry_epoch
        if self._fault_tolerant and fence > 0:
            actor_wg_kwargs["group_generation"] = [fence] * actor_wg.world_size
            rollout_kwargs["group_generation"] = [fence] * rollout.world_size

        # 3. init process group between all workers
        ray.get(
            actor_wg.execute_checkpoint_engine(**actor_wg_kwargs) + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )

    def add_replicas(self, replicas: list[RolloutReplica]):
        """Add rollout replicas to the manager for elastic scale up, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to add.
        """
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        replicas_set = set(replicas)
        self.replicas = [r for r in self.replicas if r not in replicas_set]

    @auto_await
    async def sleep_replicas(self):
        """Sleep all rollout replicas: free weight and kv_cache device memory."""
        await asyncio.gather(*[r.sleep() for r in self.replicas])

    @auto_await
    async def wake_up_replicas(self):
        """Resume all rollout replicas: recover kv_cache and weights device memory."""
        await asyncio.gather(*[r.wake_up() for r in self.replicas])

    @auto_await
    async def abort_replicas(self, reject_request: bool = False):
        """Abort all in-flight requests on every replica.

        Args:
            reject_request: Fail requests arriving behind the closed gate instead of
                parking them, for replicas that will not resume generation soon.
        """
        await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])

    @auto_await
    async def resume_generation_replicas(self):
        """Resume generation on all replicas after abort_all_requests."""
        await asyncio.gather(*[r.resume_generation() for r in self.replicas])

    @auto_await
    async def release_kv_cache_replicas(self):
        """Release kv_cache of all rollout replicas before NCCL weight sync.

        Unlike sleep_replicas(), this only frees the kv_cache and leaves model
        weights untouched, so the NCCL transfer can write directly into the
        existing weight buffers.  Call resume_kv_cache_replicas() after sync.
        """
        await asyncio.gather(*[r.release_kv_cache() for r in self.replicas])

    @auto_await
    async def resume_kv_cache_replicas(self):
        """Restore kv_cache of all rollout replicas after NCCL weight sync.

        Counterpart to release_kv_cache_replicas().
        """
        await asyncio.gather(*[r.resume_kv_cache() for r in self.replicas])

    @staticmethod
    def _merge_sync_metrics(results: list[Any]) -> dict[str, Any]:
        sync_metrics = {}
        for result in results:
            if isinstance(result, dict):
                sync_metrics.update(result)
        return sync_metrics

    @auto_await
    async def wait_for_weight_update(self):
        """Wait until the rollout replicas hold the weights of the last ``update_weights``.

        With a backend that publishes weights (``support_publish_weights=True``), ``update_weights`` returns
        once the trainer has published them, and the rollout commit finishes on the background loop. Trainers
        call this before anything that must see the new weights (validation) and before the training loop
        exits. For coupled backends the last update is already committed, so this returns at once.
        """
        if self._update_ticket is not None:
            await self._update_ticket.wait_committed()

    @auto_await
    async def update_weights(self, global_steps: int = None):
        """Update weights from actor worker group to rollout replicas.

        Args:
            global_steps: The global steps of the actor worker group.

        With fault tolerance (see :meth:`enable_fault_tolerance`) this is
        the wait-and-retry loop over member failures; otherwise the
        fail-fast pipeline.
        """
        if not self._fault_tolerant:
            return await self._update_weights_pipeline(global_steps)
        return await self._update_weights_with_retry(global_steps)

    async def _update_weights_with_retry(self, global_steps: int = None):
        """Fatal gate + membership reconcile, then bounded attempts; an empty
        syncable set waits for recovery (nobody to sync to — the loop must
        not burn the attempt budget on doomed empty-group builds); after a
        failure, hold for the detection grace window only and re-attempt with
        the surviving group (the entry reconcile re-probes membership; a
        fenced-out replica rejoins at the following sync)."""
        attempts = 0
        while True:
            self._gate_on_fatal()
            await self._pull_membership()
            await self._wait_for_syncable_replicas()
            try:
                return await self._update_weights_pipeline(global_steps)
            except Exception as exc:
                # Drop the abandoned ticket (it never resolves) and rename
                # the next attempt's group: a failed attempt must not be
                # retried on its own communicator.
                self._update_ticket = None
                self._retry_epoch += 1
                attempts += 1
                if attempts >= _MAX_SYNC_ATTEMPTS:
                    raise RuntimeError(f"weight sync failed after {attempts} attempts (last: {exc})") from exc
                logger.warning(
                    "weight sync attempt %d/%d failed (%s: %s) — retrying after the detection grace window",
                    attempts,
                    _MAX_SYNC_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                )
                await self._wait_for_detection()

    def _gate_on_fatal(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"rollout restart is in a fatal state: {self._fatal_error}")

    async def _pull_membership(self) -> None:
        """Best-effort membership pull; a broken pull leaves membership at
        the last snapshot."""
        if self._pre_sync_hook is None:
            return
        try:
            await self._pre_sync_hook()
        except Exception:
            logger.warning("membership reconcile failed (non-fatal)", exc_info=True)

    async def _wait_for_detection(self) -> None:
        """Hold off the retry only for the fault-detection grace window: the
        next attempt's entry reconcile runs a fresh detection round, so the
        surviving group syncs without waiting for any rebuild to settle — a
        fenced-out replica rejoins at the following sync. Fails closed on
        the fatal flag."""
        await asyncio.sleep(_PROBE_GRACE_S)
        await self._pull_membership()
        self._gate_on_fatal()

    async def _wait_for_syncable_replicas(self) -> None:
        """Hold while the syncable member set is empty (every replica
        mid-recovery or fenced out): with nobody to sync to, a fast retry
        would only burn the attempt budget on doomed empty-group builds and
        fail the step before recovery lands. Terminal states: the fatal flag
        (all replicas DEAD) and the overall recovery-wait bound. Waits here
        are not sync attempts."""
        if self.replicas:
            return
        logger.warning("no syncable rollout replicas at the weight-sync entry — waiting for recovery")
        deadline = time.monotonic() + _RECOVERY_WAIT_TIMEOUT_S
        while not self.replicas:
            self._gate_on_fatal()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"no syncable rollout replicas recovered within {_RECOVERY_WAIT_TIMEOUT_S:.0f}s")
            await asyncio.sleep(_EMPTY_MEMBERSHIP_POLL_S)
            await self._pull_membership()

    async def _update_weights_pipeline(self, global_steps: int = None):
        # 0. update weights for sync training with colocated actor and rollout
        # No global_steps check here: the only caller without one wakes replicas back up on unchanged
        # weights, so the current salt still matches what the external KV store holds.
        if self.backend == "naive":
            await asyncio.gather(*self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend))
            return {}

        # Wait previous update to be committed
        if self._update_ticket is not None:
            await self._update_ticket.wait_committed()

        self._update_ticket = await self.submit_update(global_steps=global_steps)
        if self.backend_cls.support_publish_weights:
            published = await self._update_ticket.wait_published()
            return self._merge_sync_metrics(published)
        else:
            return await self._update_ticket.wait_committed()

    async def submit_update(self, global_steps: int | None, rollout_quiesced: bool = False) -> WeightUpdateTicket:
        # An empty member set must fail here with a clear error — an empty
        # worker list would surface as an opaque single-controller TypeError.
        if not self.replicas:
            raise RuntimeError("no rollout replicas available for the weight sync (empty syncable member set)")
        # Salted external KV caches tell weights apart only by global_steps, and the shared KV store
        # is no longer wiped, so a weight change without a new version would serve stale KV.
        if global_steps is None and any(r.config.external_kv_cache_invalidation == "salt" for r in self.replicas):
            raise ValueError("external_kv_cache_invalidation='salt' requires global_steps on every weight update")

        # 1. create a temporay worker group for all replicas
        workers = []
        for replica in self.replicas:
            workers.extend(replica.workers)
        rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))

        if not self.backend_cls.support_publish_weights:
            # Coupled transports allocate receive buffers and start collectives in
            # build_process_group(), so preserve the original ordering: quiesce
            # rollout and release its KV cache before either of those operations.
            # With fault tolerance the quiesce is best-effort per replica.
            await self._quiesce_replicas()

        # 2. build process group
        self.build_process_group(rollout)

        # 3. trainer publish weights
        trainer_published = self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend)

        # 4. rollout consume weights
        rollout_committed = asyncio.run_coroutine_threadsafe(
            self._commit_update(
                rollout=rollout,
                trainer_published=trainer_published,
                global_steps=global_steps,
            ),
            self._background_loop,
        )
        return WeightUpdateTicket(
            version=global_steps,
            trainer_published=trainer_published,
            rollout_committed=rollout_committed,
        )

    async def _quiesce_replicas(self) -> None:
        """Abort + release KV cache: strict (default) or best-effort per
        replica (fault tolerance)."""
        if not self._fault_tolerant:
            await self.abort_replicas()
            await self.release_kv_cache_replicas()
            return
        await asyncio.gather(*(self._best_effort_abort(r) for r in list(self.replicas)))
        await asyncio.gather(*(self._best_effort_release(r) for r in list(self.replicas)))

    async def _best_effort_abort(self, replica: RolloutReplica) -> None:
        try:
            await replica.abort_all_requests()
        except Exception:
            logger.warning("abort_all_requests failed for replica %s (non-fatal)", self._replica_key(replica))

    async def _best_effort_release(self, replica: RolloutReplica) -> None:
        try:
            await replica.release_kv_cache()
        except Exception:
            logger.warning("release_kv_cache failed for replica %s (non-fatal)", self._replica_key(replica))

    async def _commit_update(
        self,
        rollout: RayWorkerGroup,
        trainer_published: list[ray.ObjectRef],
        global_steps: int | None,
    ):
        """Install a published weight version on rollout workers."""
        if self.backend_cls.support_publish_weights:
            await self._quiesce_replicas()

        rollout_committed = rollout.update_weights(global_steps=global_steps)
        if not self._fault_tolerant:
            return await self._commit_strict(rollout, trainer_published, rollout_committed)
        return await self._commit_all_or_nothing(rollout, trainer_published, rollout_committed)

    async def _commit_strict(
        self,
        rollout: RayWorkerGroup,
        trainer_published: list[ray.ObjectRef],
        rollout_committed: list,
    ):
        results = await asyncio.gather(*trainer_published, *rollout_committed)

        await asyncio.gather(
            *self.actor_wg.execute_checkpoint_engine(["finalize"] * self.actor_wg.world_size),
            *rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size),
        )

        await self.resume_kv_cache_replicas()
        await self.resume_generation_replicas()
        return self._merge_sync_metrics(results[: self.actor_wg.world_size])

    async def _commit_all_or_nothing(
        self,
        rollout: RayWorkerGroup,
        trainer_published: list[ray.ObjectRef],
        rollout_committed: list,
    ):
        """All-or-nothing commit: the transfer completes within the bounded
        abort or the attempt fails and the wait-retry loop owns the follow-up
        under a fresh fenced name. The version stamp moves only on a complete
        transfer, so a failed attempt leaves every replica serving its last
        installed version."""
        futures = [asyncio.ensure_future(ref) for ref in (*trainer_published, *rollout_committed)]
        try:
            results = await asyncio.wait_for(asyncio.gather(*futures), timeout=_BOUNDED_ABORT_S)
        except Exception as exc:
            # A worker raised or stalled (a dead peer's collective holds the
            # survivors): fail the attempt — the retry renames the group and
            # each worker destroys its previous group on the rename.
            raise RuntimeError(
                f"weight sync commit failed or stalled within {_BOUNDED_ABORT_S}s ({type(exc).__name__}: {exc})"
            ) from exc

        finalize_refs = self.actor_wg.execute_checkpoint_engine(
            ["finalize"] * self.actor_wg.world_size
        ) + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        try:
            await asyncio.wait_for(asyncio.gather(*finalize_refs, return_exceptions=True), timeout=_FINALIZE_TIMEOUT_S)
        except asyncio.TimeoutError:
            # A stuck worker's finalize queues behind its blocked collective;
            # the straggler is the manager's polling business.
            logger.warning("engine finalize did not finish within 30s (stuck worker assumed); continuing")

        await self._guarded_resume(list(self.replicas))
        self._notify_sync_succeeded()
        return self._merge_sync_metrics(results[: len(trainer_published)])

    async def _guarded_resume(self, replicas: list[RolloutReplica]) -> None:
        """Resume KV cache and generation per replica, swallowing per-replica
        failures (a dead replica must not block the others)."""
        for replica in replicas:
            key = self._replica_key(replica)
            try:
                await replica.resume_kv_cache()
                await replica.resume_generation()
            except Exception:
                logger.warning("resume failed for replica %s (non-fatal)", key)

    def _notify_sync_succeeded(self) -> None:
        """Fire-and-forget: let the manager promote rebuilt replicas; a lost
        notification is retried by the next sync."""
        if self._sync_succeeded_callback is None:
            return
        try:
            self._sync_succeeded_callback()
        except Exception:
            logger.warning("sync-succeeded notification failed (non-fatal)", exc_info=True)


async def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None], bucket_size: int, meta_only: bool = False
) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor | None], None]:
    """Split the weight into chunks.

    Args:
        weights: The weights generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the weight chunk metadata and the buffer.
    """
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(
                name=name,
                shape=weight.shape,
                dtype=weight.dtype,
                chunk_offset=chunk_offset,
                chunk_size=chunk_size,
                offset=None,
            )
            yield (tensor_meta, None if meta_only else buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size


async def merge_weight_chunks(
    chunks: Generator[tuple[TensorMeta, torch.Tensor], None, None], bucket_size: int
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    """Merge the weight chunks into the original weight.

    Args:
        chunks: The chunks generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the name of the weight tensor and the tensor itself.
    """
    merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
    async for tensor_meta, chunk in chunks:
        assert chunk.dtype == torch.uint8, f"Chunk dtype must be uint8, but got {chunk.dtype}"
        nbytes = tensor_meta.shape.numel() * tensor_meta.dtype.itemsize

        # weight is small enough to fit in one bucket
        if nbytes <= bucket_size:
            assert merge_weight is None, f"Weight must be None, but got {merge_name}"
            name, weight = tensor_meta.name, chunk.view(tensor_meta.dtype).view(tensor_meta.shape)
            yield (name, weight)
            continue

        if merge_weight is None:
            assert tensor_meta.chunk_offset == 0, f"Chunk offset must be 0, but got {tensor_meta}"
            merge_name, merge_weight = (
                tensor_meta.name,
                torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=chunk.device),
            )
            merge_buffer = merge_weight.view(-1).view(torch.uint8)
            merge_offset = 0

        assert tensor_meta.name == merge_name
        assert merge_offset == tensor_meta.chunk_offset
        merge_buffer[tensor_meta.chunk_offset : tensor_meta.chunk_offset + tensor_meta.chunk_size] = chunk
        merge_offset += tensor_meta.chunk_size
        if tensor_meta.chunk_offset + tensor_meta.chunk_size == nbytes:
            yield (merge_name, merge_weight)
            merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
