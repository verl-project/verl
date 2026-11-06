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
import contextlib
import logging
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.plugin.platform import get_platform
from verl.utils.device import get_resource_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutReplica
from verl.workers.rollout.runtime import (
    VLLMServerLauncher,
    probe_rollout_runtime,
    resolve_rollout_runtime_env,
    runtime_identity,
)

logger = logging.getLogger(__name__)


class vLLMReplica(RolloutReplica):
    """vLLM replica controller, importable without the vLLM package installed."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.runtime_env = None
        self.runtime_expected = None
        self.runtime_reports = []
        self._server_launchers = []
        if self.config.runtime.py_executable is None:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

            self.server_class = ray.remote(vLLMHttpServer)
        else:
            self.server_class = None

    def _require_shared_runtime(self):
        if self.config.runtime.py_executable is not None:
            raise ValueError("rollout.runtime requires standalone rollout; hybrid/colocated modes are unsupported.")

    async def init_hybrid(self, worker_group):
        """Initialize fused workers only when they share the controller runtime."""
        self._require_shared_runtime()
        await super().init_hybrid(worker_group)

    async def init_hybrid_colocated(self, worker_group, resource_pool):
        """Initialize fused colocated workers without a distinct interpreter."""
        self._require_shared_runtime()
        await super().init_hybrid_colocated(worker_group, resource_pool)

    async def init_colocated(self, resource_pool):
        """Initialize separate colocated workers in the default shared runtime."""
        self._require_shared_runtime()
        await super().init_colocated(resource_pool)

    async def init_standalone(self, resource_pool=None):
        """Start standalone rollout and clean up this replica on startup failure."""
        if self.config.runtime.py_executable is None:
            return await super().init_standalone(resource_pool)
        if get_platform().device_name != "cuda" or self.is_reward_model or self.is_teacher_model:
            raise ValueError("rollout.runtime currently supports standalone CUDA actor rollouts only.")
        owns_pool = resource_pool is None
        try:
            await asyncio.wait_for(super().init_standalone(resource_pool), self.config.runtime.startup_timeout)
        except BaseException:
            for actor in self.servers + self.workers + self._server_launchers:
                with contextlib.suppress(Exception):
                    ray.kill(actor, no_restart=True)
            self.servers.clear()
            self.workers.clear()
            self._server_launchers.clear()
            if owns_pool and self.resource_pool is not None:
                for pg in self.resource_pool.pgs or []:
                    ray.util.remove_placement_group(pg)
            raise

    async def _preflight_standalone(self):
        if self.config.runtime.py_executable is None:
            return
        if self.config.checkpoint_engine.backend != "nccl":
            raise ValueError("rollout.runtime currently requires checkpoint_engine.backend=nccl.")
        # Keep checkpoint workers' normal Ray device visibility. The HTTP
        # actors alone need the platform's no-set-device flags (see launch_servers).
        self.runtime_env = resolve_rollout_runtime_env(self.config.runtime, ray.get_runtime_context().runtime_env)
        self.runtime_expected = runtime_identity()
        pgs = self.resource_pool.get_placement_groups(device_name=get_platform().device_name)
        bundles = [
            node_id
            for pg in pgs
            for _, node_id in sorted(ray.util.placement_group_table(pg)["bundles_to_node_id"].items())
        ]
        # Sub-pools retain their parent's groups; only probe their own bundle slice.
        start = getattr(self.resource_pool, "start_bundle_index", 0)
        node_ids = set(bundles[start : start + self.resource_pool.world_size])
        refs = [
            probe_rollout_runtime.options(
                runtime_env=self.runtime_env,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
            ).remote(
                self.runtime_expected,
                self.config.checkpoint_engine.backend,
                self.config.checkpoint_engine.custom_backend_module,
            )
            for node_id in node_ids
        ]
        try:
            self.runtime_reports = await asyncio.gather(*refs)
            logger.info("Standalone rollout runtime preflight: %s", self.runtime_reports)
        except BaseException:
            for ref in refs:
                ray.cancel(ref, force=True)
            raise

    def _get_worker_runtime_env(self) -> dict:
        return self.runtime_env or {}

    async def _create_server(self, actor_options: dict, server_kwargs: dict):
        if self.runtime_env is None:
            return self.server_class.options(**actor_options).remote(**server_kwargs)
        actor_options["runtime_env"] = resolve_rollout_runtime_env(
            self.config.runtime, self.runtime_env, actor_options["runtime_env"]["env_vars"]
        )
        launcher = (
            ray.remote(VLLMServerLauncher)
            .options(
                num_cpus=0,
                lifetime="non_detached",
                runtime_env=actor_options["runtime_env"],
                scheduling_strategy=actor_options["scheduling_strategy"],
            )
            .remote(self.runtime_expected)
        )
        self._server_launchers.append(launcher)
        try:
            return await launcher.create.remote(actor_options, server_kwargs)
        except BaseException:
            # Killing the owner also covers creation that lost its return
            # handle. No lookup by name can accidentally kill another launch.
            with contextlib.suppress(Exception):
                ray.kill(launcher, no_restart=True)
            self._server_launchers.remove(launcher)
            raise

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}{self.name_suffix}"
            env_vars = {
                **{var: "1" for var in get_platform().ray_noset_envvars()},
                **get_platform().rollout_env_vars(),
            }

            server = await self._create_server(
                actor_options={
                    "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                    "runtime_env": {"env_vars": env_vars},
                    "name": name,
                    "max_concurrency": self.config.ray_actor_max_concurrency,
                },
                server_kwargs={
                    "config": self.config,
                    "model_config": self.model_config,
                    "rollout_mode": self.rollout_mode,
                    "workers": workers,
                    "replica_rank": self.replica_rank,
                    "node_rank": node_rank,
                    "gpus_per_node": gpus_per_replica_node,
                    "nnodes": nnodes,
                    "cuda_visible_devices": node_cuda_visible_devices,
                },
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(
        self,
        reject_request: bool = False,
        reset_prefix_cache: bool = True,
        abort_only: bool = False,
    ) -> dict[str, Any]:
        """Abort in-flight requests on every server.

        abort_only=False (default) also closes admission. See
        vLLMHttpServer.abort_all_requests(). reset_prefix_cache and
        reject_request apply only when abort_only is False.
        """
        results = await asyncio.gather(
            *[
                server.abort_all_requests.remote(
                    reject_request=reject_request,
                    reset_prefix_cache=reset_prefix_cache,
                    abort_only=abort_only,
                )
                for server in self.servers
            ]
        )

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def resume_generation(self):
        """Resume the DP engines before admitting requests on any server."""
        # KV-cache wake-up may already have resumed scheduling. Keep admission
        # closed during this resume too: new requests could otherwise start a
        # DP wave while only some ranks are still in the resume collective.
        await self.servers[0].resume_engine_generation.remote()
        await asyncio.gather(*[server.open_submission_gate.remote() for server in self.servers])

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}

    async def snapshot(self) -> dict[str, Any]:
        """Return live KV-cache and scheduler observations from the head server.

        Colocated / non-PD replicas report the head HTTP server. PD is out of
        scope: prefill and decode engines have independent KV pools.
        """
        # TODO: aggregate prefill + decode snapshots for PD replicas.
        if self.config.disaggregation.enabled:
            raise NotImplementedError("vLLMReplica.snapshot() does not support PD disaggregation")
        return await self.servers[0].snapshot.remote()

    async def release_kv_cache(self):
        # Drain all in-flight requests so that vLLM worker threads go idle
        # before we touch engine.release_kv_cache()
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.release_kv_cache.remote() for server in self.servers])

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix (e.g. 'vllm_')."""
        return "vllm_"
