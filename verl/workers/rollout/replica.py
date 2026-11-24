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
import logging
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional

import ray
from omegaconf import DictConfig
from pydantic import BaseModel
from ray.actor import ActorHandle

from verl.plugin.platform import get_platform
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.fault_tolerance import (
    ReplicaRestartError,
    WeightSourceOverride,
    _get_actor_state_dict,
    verify_process_dead,
)

logger = logging.getLogger(__file__)

# Shared by every replica this process creates, and fresh per run, so a cache entry left in an
# external KV store by an earlier run never matches even when it reaches the same global step.
_PREFIX_CACHE_NAMESPACE = uuid.uuid4().hex


def prefix_cache_salt(config: RolloutConfig, global_steps: Optional[int]) -> Optional[str]:
    """Engine cache_salt for a request served by weights of ``global_steps``.

    None under ``external_kv_cache_invalidation="reset"``, where verl resets caches instead.
    The server computes the salt after its pause gate, so a request parked across a weight
    update is salted with the weights it actually runs on.
    """
    if config.external_kv_cache_invalidation != "salt":
        return None
    # Set by RolloutReplica; without it, salts from different runs would collide in a shared store.
    assert config.prefix_cache_namespace, "prefix_cache_namespace is unset; servers must be launched by RolloutReplica"
    return f"{config.prefix_cache_namespace}:{global_steps}"


class TokenOutput(BaseModel):
    token_ids: list[int]
    """response token ids"""
    log_probs: Optional[list[float]] = None
    """logprobs of response token ids"""
    routed_experts: Optional[Any] = None
    """routed experts of response token ids"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'length', 'aborted', 'engine_failed', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


def engine_failure_receipt(global_steps: Optional[int], partial_output: Any = None) -> TokenOutput:
    """Build the ``engine_failed`` progress receipt for an in-flight request.

    Completeness-first: the accumulated prefix is kept only when its per-token
    logprobs are complete; anything partial drops the whole prefix — recompute
    beats corrupt data. The server serves this receipt when the engine core
    dies mid-request (or is already known dead); the client recognizes it and
    bounded-replays (request-path self-healing only — the client reports
    nothing, detection is the manager's polling channel).
    """
    token_ids: list[int] = []
    log_probs: Optional[list[float]] = None

    outputs = getattr(partial_output, "outputs", None) or []
    if outputs:
        out = outputs[0]
        token_ids = list(getattr(out, "token_ids", None) or [])
        raw = getattr(out, "logprobs", None)
        if token_ids and isinstance(raw, list) and len(raw) >= len(token_ids):
            try:
                # strict=False: logprobs may legitimately trail tokens; the length
                # check above already guarantees every kept token has its entry.
                log_probs = [float(entry[token_id].logprob) for entry, token_id in zip(raw, token_ids, strict=False)]
            except (KeyError, TypeError, AttributeError, ValueError):
                log_probs = None

    if log_probs is None:
        # Incomplete or unparsable progress: drop the prefix entirely.
        token_ids = []

    return TokenOutput(
        token_ids=token_ids,
        log_probs=log_probs,
        routed_experts=None,
        stop_reason="engine_failed",
        num_preempted=None,
        extra_fields={"engine_failed": True, "global_steps": global_steps},
    )


class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"


WITNESS_RETRY_WINDOW_S = 30.0  # bounded retry window absorbing the Ray GCS -> OS death lag


def _witness_server_process_dead(worker, pid: int, create_time: float) -> bool:
    """Executed inside a checkpoint-engine worker actor via ``__ray_call__``: same-node witness
    probe (module-level so it serializes to the worker)."""
    return verify_process_dead(pid, create_time)


async def _witness_server_dead(old_workers: list, pid: int, create_time: float) -> None:
    """Confirm the server process is really dead before relaunching.

    The surviving checkpoint-engine workers share the server actor's node and
    witness the OS-level death; the driver-local probe is the single-node
    fallback. Ray's GCS can publish an actor DEAD before the OS process exits,
    so the probe retries within a bounded window — fail-closed once the
    deadline passes.
    """
    deadline = time.monotonic() + WITNESS_RETRY_WINDOW_S
    while True:
        for worker in old_workers:
            try:
                dead = await asyncio.wait_for(
                    worker.__ray_call__.remote(_witness_server_process_dead, pid, create_time), timeout=5.0
                )
            except Exception:
                continue
            if dead:
                return
        if verify_process_dead(pid, create_time):
            return
        if time.monotonic() >= deadline:
            raise ReplicaRestartError(
                f"witness probe could not confirm server process death (pid={pid}) within "
                f"{WITNESS_RETRY_WINDOW_S:g}s; refusing to relaunch"
            )
        await asyncio.sleep(1.0)


async def _wait_actor_terminal(handle, timeout: float = 10.0) -> None:
    """Bounded wait for a ``ray.kill``-ed actor to leave ALIVE — Ray GCS can
    publish DEAD before the OS process dies; this sequences killing old
    actors against respawning under the same names."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Blocking GCS query — keep it off the manager actor's event loop.
        state_dict = await asyncio.to_thread(_get_actor_state_dict, handle)
        if state_dict is None or state_dict.get("state") != "ALIVE":
            return
        await asyncio.sleep(0.2)
    raise ReplicaRestartError(f"actor did not leave the ALIVE state within {timeout}s after ray.kill")


class RolloutReplica(ABC):
    """Rollout replica is an individual server instance, which may be deployed on single or multiple nodes.
    It is equivalent to launch server in each node with command line:

    SGLang:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 2 ...
    python -m sglang.launch_server --node-rank 1 --nnode 2 ...
    ```

    vLLM:
    ```
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 0 ...
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 ...
    ```

    Args:
        replica_rank: int, rank of this rollout replica.
        config: RolloutConfig, full config.
        model_config: DictConfig, model config.
        gpus_per_node: int, number of gpus per node.
    """

    _restart_in_progress: bool = False
    """Re-entry guard for restart(); the supervisor's per-replica mutex is the
    primary defense, this is the belt-and-braces one."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ) -> None:
        self.replica_rank = replica_rank
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        if self.config.prefix_cache_namespace is None:
            self.config.prefix_cache_namespace = _PREFIX_CACHE_NAMESPACE
        self.model_config: HFModelConfig = model_config

        self.world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.gpus_per_node = gpus_per_node
        self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0, (
            f"world_size {self.world_size} must be divisible by gpus_per_node {self.gpus_per_replica_node}"
        )
        self.nnodes = self.world_size // self.gpus_per_replica_node
        self.is_reward_model = is_reward_model
        self.is_teacher_model = is_teacher_model
        self.name_suffix = f"_{name_suffix}" if name_suffix else ""

        self.rollout_mode: RolloutMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: RayResourcePool = None
        self.bundle_indices: list[int] = []

        self.servers: list[ActorHandle] = []
        self._server_address: str = None
        self._server_handle: ActorHandle = None

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    async def init_hybrid_colocated(self, worker_group: RayWorkerGroup, resource_pool: RayResourcePool):
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
            bundle_indices: list[int], bundle indices for this rollout replica.
        """
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        self.resource_pool = resource_pool
        self.bundle_indices = [self.replica_rank * self.world_size + idx for idx in range(self.world_size)]
        await self.launch_servers()

    # TODO(sgm): this should be the default solution, but need to make the RolloutMode more clear.
    async def init_colocated(self, resource_pool: RayResourcePool):
        """Init colocated rollout server, rollout engine and hybrid engine colocated in same ray placement group
        but in separate processes.

        Args:
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
        """
        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool
        use_gpu = self.rollout_worker_use_gpu()

        if self.is_reward_model:
            name_prefix = f"rollout_reward_colocate_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            name_prefix = f"rollout_teacher_colocate_{self.replica_rank}{self.name_suffix}"
        else:
            name_prefix = f"rollout_colocate_{self.replica_rank}{self.name_suffix}"

        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=name_prefix,
            use_gpu=use_gpu,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    async def init_standalone(self, resource_pool: Optional[RayResourcePool] = None):
        """Init standalone rollout server.

        Args:
            resource_pool: Existing pool to attach. If omitted, create a new
                per-replica pool as before. Callers that already sliced a parent
                pool (e.g. async-RL) pass the slice here so replica placement
                stays under their ResourcePoolManager.
        """
        self.rollout_mode = RolloutMode.STANDALONE
        if resource_pool is None:
            if self.is_reward_model:
                resource_pool_name = f"rollout_pool_reward_{self.replica_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                resource_pool_name = f"rollout_pool_teacher_{self.replica_rank}{self.name_suffix}"
            else:
                resource_pool_name = f"rollout_pool_{self.replica_rank}{self.name_suffix}"
            resource_pool_spec = {
                resource_pool_name: [self.gpus_per_replica_node] * self.nnodes,
            }
            resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resource_pool_spec,
                mapping=None,
                max_colocate_count=2,
            )
            resource_pool_manager.create_resource_pool()
            resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]
        self.resource_pool = resource_pool

        # create worker group for this rollout
        # These workers share each bundle's device with the rollout engine's own workers. Where a
        # device cannot be shared between processes (TPU), they must not claim it.
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=self._standalone_worker_name_prefix(),
            use_gpu=get_platform().supports_colocated_worker_groups(),
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    def _standalone_worker_name_prefix(self) -> str:
        if self.is_reward_model:
            return f"rollout_reward_standalone_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            return f"rollout_teacher_standalone_{self.replica_rank}{self.name_suffix}"
        return f"rollout_standalone_{self.replica_rank}{self.name_suffix}"

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        from verl.checkpoint_engine.base import CheckpointEngineWorker

        rollout_worker_actor_cls = ray.remote(CheckpointEngineWorker)

        return RayClassWithInitArgs(
            cls=rollout_worker_actor_cls,
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    @abstractmethod
    async def launch_servers(self):
        """Launch http server in each node.

        A restart relaunch hands the checkpoint cold-start weight source down
        through this launch chain; backends without restart support keep the
        zero-argument signature.
        """
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        """Get rollout server address for OpenAI chat completion."""
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        """Get rollout server handle for Token-in-token-out generation."""
        return self._server_handle

    def rollout_worker_use_gpu(self) -> bool:
        return True

    async def wake_up(self):
        """Wake up each rollout server."""
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def sleep(self):
        """Sleep each rollout server."""
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self, reject_request: bool = False):
        """Partial rollout: abort and save all unfinished requests in each rollout server.

        Args:
            reject_request: Fail requests that arrive while generation is blocked instead
                of holding them until the next resume_generation(). Pass True when the
                replica is leaving the load balancer and no resume is coming soon.
                Backends that cannot intercept their own admission path log a warning
                and ignore it.
        """
        await asyncio.gather(
            *[server.abort_all_requests.remote(reject_request=reject_request) for server in self.servers]
        )

    async def resume_generation(self):
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def clear_kv_cache(self):
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.clear_kv_cache.remote() for server in self.servers])

    async def release_kv_cache(self):
        """Release only the kv_cache GPU memory, keeping model weights in place."""
        await asyncio.gather(*[server.release_kv_cache.remote() for server in self.servers])

    async def resume_kv_cache(self):
        """Restore the kv_cache GPU memory after a weight sync."""
        await asyncio.gather(*[server.resume_kv_cache.remote() for server in self.servers])

    async def check_health(self) -> None:
        """Liveness probe: the head server's ``check_health`` (raises when the
        engine is gone); a replica without a live server actor is a restart
        error rather than a silent pass."""
        if not self.servers:
            raise ReplicaRestartError(f"replica {self.replica_rank}: no live server actor to probe")
        await self.servers[0].check_health.remote()

    async def restart(
        self,
        *,
        cleanup_timeout_s: float = 60.0,
        server_kill_timeout_s: float = 10.0,
        weight_source: Optional[WeightSourceOverride] = None,
    ) -> None:
        """Unified whole-replica restart (orchestrated by the supervisor —
        distinct from Ray's per-actor auto-restart).

        Sequence: verified server shutdown -> server actor kill with bounded
        terminal wait -> witness probe while the checkpoint-engine workers
        are alive -> checkpoint-engine kill + terminal wait -> same-pool
        respawn (bundles survive actor death) -> relaunch of the server
        actors.

        ``weight_source`` is the checkpoint cold-start source: it rides the
        launch chain; backends without restart support keep the
        zero-argument signature. The caller (supervisor) owns the LB fence,
        the group-generation bump, and the VERIFYING gate; on any failure
        the replica keeps no half-registered state and stays ISOLATED.
        """
        if self._restart_in_progress:
            raise ReplicaRestartError(f"replica {self.replica_rank}: restart already in progress")
        if self.rollout_mode != RolloutMode.STANDALONE or self.nnodes != 1:
            raise NotImplementedError(
                f"restart() requires a single-node STANDALONE replica (mode={self.rollout_mode}, nnodes={self.nnodes})"
            )
        if self.resource_pool is None:
            raise ReplicaRestartError(f"replica {self.replica_rank}: no resource pool to respawn onto")

        launch_kwargs = {"weight_source": weight_source} if weight_source is not None else {}
        self._restart_in_progress = True
        try:
            old_server = self.servers[0] if self.servers else None
            old_workers = list(self.workers)

            server_identity = None
            if old_server is not None:
                # S1: verified shutdown — engine-owned children exit under the
                # accounting ladder; a dead actor's RayActorError is swallowed
                # (the terminal wait below proves the process gone instead).
                try:
                    server_identity = await asyncio.wait_for(old_server.shutdown.remote(), timeout=cleanup_timeout_s)
                except (ray.exceptions.RayActorError, asyncio.TimeoutError):
                    logger.warning(
                        "server actor shutdown unavailable (likely already dead); "
                        "continuing teardown without a witness identity"
                    )
                # S2: kill the server actor; the bounded terminal wait is required
                # because Ray GCS can publish DEAD before the OS process dies.
                ray.kill(old_server, no_restart=True)
                await _wait_actor_terminal(old_server, timeout=server_kill_timeout_s)
            if server_identity is not None:
                # S3: witness probe — must run while the checkpoint-engine
                # workers are still alive (they share the server actor's node).
                pid, create_time = server_identity
                await _witness_server_dead(old_workers, pid, create_time)
            # S4: kill the remaining checkpoint-engine actors — a fault that
            # killed one worker is not disprovable for its siblings (clean slate).
            for worker in old_workers:
                try:
                    ray.kill(worker, no_restart=True)
                except Exception:
                    logger.warning("failed to kill a checkpoint-engine worker; terminal wait follows", exc_info=True)
            await asyncio.gather(*[_wait_actor_terminal(w, timeout=server_kill_timeout_s) for w in old_workers])
            # S5: respawn the checkpoint-engine workers on the same resource pool.
            worker_group = RayWorkerGroup(
                resource_pool=self.resource_pool,
                ray_cls_with_init=self.get_ray_class_with_init_args(),
                bin_pack=False,
                name_prefix=self._standalone_worker_name_prefix(),
                use_gpu=get_platform().supports_colocated_worker_groups(),
                device_name=get_device_name(),
            )
            self.workers = worker_group.workers
            # S6: relaunch — the weight source rides the launch chain into the
            # new server's constructor; the replica re-registers with the LB
            # only after VERIFYING.
            self._reset_server_state()
            self._pre_relaunch_cleanup()
            await self.launch_servers(**launch_kwargs)
        except BaseException:
            self._reset_server_state()
            raise
        finally:
            self._restart_in_progress = False

    def _reset_server_state(self) -> None:
        """Drop all server registration state (no half-updated fields survive a failed restart)."""
        self.servers = []
        self._server_handle = None
        self._server_address = None

    def _pre_relaunch_cleanup(self) -> None:
        """Engine-specific pre-relaunch cleanup (e.g. stale IPC sockets); no-op by default."""
        return None  # noqa: B027 — optional hook, not an abstract obligation

    async def start_profile(self, **kwargs):
        """Start profiling on the replica."""
        await asyncio.gather(*[server.start_profile.remote(**kwargs) for server in self.servers])

    async def stop_profile(self):
        """Stop profiling on the replica."""
        await asyncio.gather(*[server.stop_profile.remote() for server in self.servers])


class RolloutReplicaRegistry:
    """Factory for managing rollout replica implementations."""

    _registry: dict[str, Callable[[], type[RolloutReplica]]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], type[RolloutReplica]]) -> None:
        """Register a new rollout replica type."""
        cls._registry[name] = loader

    @classmethod
    def get(cls, name: str) -> type[RolloutReplica]:
        """Get a rollout replica class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown rollout mode: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()


# Loader functions for built-in types
def _load_vllm():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    return vLLMReplica


def _load_sglang():
    try:
        import vllm  # noqa: F401
    except ImportError:
        import sys
        import types
        from unittest.mock import Mock

        mock_vllm = types.ModuleType("vllm")

        mock_custom_ops = types.ModuleType("vllm._custom_ops")
        mock_custom_ops.scaled_fp8_quant = Mock()
        mock_vllm._custom_ops = mock_custom_ops

        mock_model_executor = types.ModuleType("vllm.model_executor")
        mock_layers = types.ModuleType("vllm.model_executor.layers")
        mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

        class GeluAndMul:  # noqa: N801
            pass

        class SiluAndMul:  # noqa: N801
            pass

        mock_activation.GeluAndMul = GeluAndMul
        mock_activation.SiluAndMul = SiluAndMul
        mock_layers.activation = mock_activation
        mock_model_executor.layers = mock_layers
        mock_vllm.model_executor = mock_model_executor

        sys.modules["vllm"] = mock_vllm
        sys.modules["vllm._custom_ops"] = mock_custom_ops
        sys.modules["vllm.model_executor"] = mock_model_executor
        sys.modules["vllm.model_executor.layers"] = mock_layers
        sys.modules["vllm.model_executor.layers.activation"] = mock_activation

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

    return SGLangReplica


def _load_trtllm():
    from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

    return TRTLLMReplica


# Register built-in types
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
RolloutReplicaRegistry.register("trtllm", _load_trtllm)


def get_rollout_replica_class(rollout: str, disaggregation_enabled: bool = False) -> type[RolloutReplica]:
    """Resolve a replica class by backend name.

    PD-disaggregated rollouts reuse the base backend name (``sglang`` /
    ``vllm``); the dispatch here picks the PD class only when the caller
    asserts ``disaggregation_enabled=True`` (sourced from
    ``RolloutConfig.disaggregation.enabled``). Validation in
    ``RolloutConfig.__post_init__`` rejects the flag for backends without a
    PD class.
    """
    if disaggregation_enabled:
        if rollout == "sglang":
            # _load_sglang side-effect: installs vllm mocks needed by SGLangPDReplica's
            # transitive imports. Cheap if already installed.
            RolloutReplicaRegistry.get("sglang")
            from verl.workers.rollout.sglang_rollout.sglang_pd_replica import SGLangPDReplica

            return SGLangPDReplica
        if rollout == "vllm":
            from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

            return vLLMPDReplica
        raise NotImplementedError(
            f"PD disaggregation is only supported with rollout in ('sglang', 'vllm'); got {rollout!r}."
        )
    return RolloutReplicaRegistry.get(rollout)
