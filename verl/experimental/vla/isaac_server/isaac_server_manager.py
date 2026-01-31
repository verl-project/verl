# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
Isaac Sim Server Manager

Manages multiple IsaacServers across GPUs and pipeline stages.

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                 IsaacServerManager                                 │
    │                                                                    │
    │  Stage 0 (actors[0]):          Stage 1 (actors[1]):               │
    │  ┌──────────────────────┐      ┌──────────────────────┐           │
    │  │ Actor 0 (GPU 0)      │      │ Actor 0 (GPU 0)      │           │
    │  │ Tasks: 0-4           │      │ Tasks: 0-4           │           │
    │  ├──────────────────────┤      ├──────────────────────┤           │
    │  │ Actor 1 (GPU 1)      │      │ Actor 1 (GPU 1)      │           │
    │  │ Tasks: 5-9           │      │ Tasks: 5-9           │           │
    │  ├──────────────────────┤      ├──────────────────────┤           │
    │  │ ...                  │      │ ...                  │           │
    │  └──────────────────────┘      └──────────────────────┘           │
    └────────────────────────────────────────────────────────────────────┘

Key Concepts:
    - Each stage has its own set of servers (physical isolation between stages)
    - Each server handles a subset of tasks
    - All servers in a stage share the same task → server mapping
    - Batched operations are parallelized via ray.get() on multiple servers

Usage:
    manager = IsaacServerManager(
        num_stages=2,
        num_actors_per_stage=8,
        num_tasks=40,
        group_size=8,
    )
    manager.initialize()  # Creates and initializes all actors

    # Step on specific stage
    result = manager.step_batched(server_requests, stage_id=0)

    # Reset all stages
    results = manager.reset_batched(server_requests, stage_id=0)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import ray

from .isaac_server import IsaacServer

logger = logging.getLogger(__name__)


class IsaacServerManager:
    """
    Manager for multiple IsaacServers across stages and GPUs.

    Design:
    - Creates num_stages × num_servers_per_stage servers
    - Each stage is physically isolated (separate server instances)
    - Provides batched step/reset operations for efficiency

    Thread Safety:
    - Manager methods are thread-safe for concurrent calls from different stages
    - Uses ThreadPoolExecutor for parallel ray.get() calls
    """

    def __init__(
        self,
        num_stages: int = 2,
        num_servers_per_stage: int = 8,
        num_tasks: int = 40,
        group_size: int = 8,
        env_id: str = "Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        render_last_only: bool = True,
        camera_height: int = 256,
        camera_width: int = 256,
        placement_group=None,
        accelerator_type: Optional[str] = None,
        runtime_env: Optional[dict] = None,
    ):
        """
        Initialize the actor manager.

        Args:
            num_stages: Number of pipeline stages (each gets its own server set)
            num_servers_per_stage: Number of servers per stage (typically = num GPUs)
            num_tasks: Total number of tasks (split across actors)
            group_size: Number of envs per task (fixed for all tasks)
            env_id: Isaac Lab environment ID
            render_last_only: Only render last step of action chunks
            camera_height: Camera image height
            camera_width: Camera image width
            placement_group: Optional Ray placement group for scheduling
            accelerator_type: Optional accelerator type label (e.g., "sim")
            runtime_env: Optional Ray runtime environment for servers
        """
        self.num_stages = num_stages
        self.num_servers_per_stage = num_servers_per_stage
        self.num_tasks = num_tasks
        self.group_size = group_size
        self.env_id = env_id
        self.render_last_only = render_last_only
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.placement_group = placement_group
        self.accelerator_type = accelerator_type
        self.runtime_env = runtime_env

        # Total envs = num_tasks * group_size
        self._total_envs = num_tasks * group_size

        # Calculate task distribution across servers
        # Tasks are distributed evenly; some servers get +1 task if remainder > 0
        self.tasks_per_server = num_tasks // num_servers_per_stage
        self.remainder_tasks = num_tasks % num_servers_per_stage

        # servers[stage_id][server_rank] = IsaacServer handle
        self.servers: list[list[ray.ObjectRef]] = []

        # Global task mapping (same for all stages)
        # task_id → server_rank
        self._task_to_server: dict[int, int] = {}
        # task_id → local env indices on that server
        self._task_to_env_indices: dict[int, list[int]] = {}
        # server_rank → task_offset
        self._server_task_offsets: list[int] = []

        self._initialized = False

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=num_servers_per_stage)

        # Calculate GPU requirements
        # Each GPU is shared by num_stages servers (time-sharing across pipeline stages)
        # e.g., 2 stages → 2 servers per GPU → 0.5 GPU/server
        servers_per_gpu = num_stages
        self.gpu_per_server = 1.0 / servers_per_gpu
        total_gpus_needed = num_servers_per_stage  # One GPU per server rank, shared by stages

        logger.info(
            f"IsaacServerManager created: "
            f"{num_stages} stages × {num_servers_per_stage} servers = "
            f"{num_stages * num_servers_per_stage} total servers, "
            f"{num_tasks} tasks × {group_size} group_size = {self._total_envs} total_envs"
        )
        if self.remainder_tasks > 0:
            logger.info(
                f"Task distribution: servers 0-{self.remainder_tasks - 1} handle {self.tasks_per_server + 1} tasks, "
                f"servers {self.remainder_tasks}-{num_servers_per_stage - 1} handle {self.tasks_per_server} tasks"
            )
        logger.info(
            f"GPU allocation: {servers_per_gpu} servers/GPU, "
            f"{self.gpu_per_server:.2f} GPU/server, "
            f"total GPUs needed: {total_gpus_needed}"
        )

    def initialize(self) -> bool:
        """
        Create and initialize all servers.

        This creates num_stages × num_servers_per_stage servers,
        each running on its own GPU (managed by Ray).

        Returns:
            True if all servers initialized successfully
        """
        if self._initialized:
            logger.warning("Manager already initialized")
            return True

        logger.info("Creating IsaacServers...")

        # Build task distribution
        self._build_task_mapping()

        # Create servers for each stage
        for stage_id in range(self.num_stages):
            stage_servers = []
            for server_rank in range(self.num_servers_per_stage):
                # Calculate tasks for this server
                if server_rank < self.remainder_tasks:
                    server_num_tasks = self.tasks_per_server + 1
                    task_offset = server_rank * (self.tasks_per_server + 1)
                else:
                    server_num_tasks = self.tasks_per_server
                    task_offset = (
                        self.remainder_tasks * (self.tasks_per_server + 1)
                        + (server_rank - self.remainder_tasks) * self.tasks_per_server
                    )

                # Build server options
                # Each server gets 1/num_stages GPU so that all stages can time-share the same GPUs
                # Example: 2 stages × 8 servers/stage = 16 servers, each with 0.5 GPU = 8 GPUs total
                server_options = {"num_gpus": self.gpu_per_server}

                # Add runtime_env if provided (same as EnvWorker uses)
                if self.runtime_env is not None:
                    server_options["runtime_env"] = self.runtime_env

                # Add placement group if provided
                if self.placement_group is not None:
                    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

                    bundle_idx = server_rank % len(self.placement_group.bundle_specs)
                    server_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_idx,
                    )

                # Add accelerator type if specified
                if self.accelerator_type is not None:
                    server_options["resources"] = {self.accelerator_type: 1e-4}

                # Server's total envs = num_tasks * group_size
                server_total_envs = server_num_tasks * self.group_size

                # Create server
                server = IsaacServer.options(**server_options).remote(
                    env_id=self.env_id,
                    num_tasks=server_num_tasks,
                    group_size=self.group_size,  # Fixed envs per task
                    actor_rank=server_rank,
                    task_offset=task_offset,
                    render_last_only=self.render_last_only,
                    camera_height=self.camera_height,
                    camera_width=self.camera_width,
                    stage_id=stage_id,
                )
                stage_servers.append(server)

                logger.info(
                    f"Created server: stage={stage_id}, rank={server_rank}, "
                    f"tasks={task_offset}-{task_offset + server_num_tasks - 1}, "
                    f"envs={server_total_envs}, gpu={self.gpu_per_server:.2f}"
                )

            self.servers.append(stage_servers)

        # Initialize all servers in parallel
        logger.info("Initializing all servers...")
        init_futures = []
        for stage_id, stage_servers in enumerate(self.servers):
            for server in stage_servers:
                init_futures.append(server.init_env.remote())

        # Wait for all initializations
        try:
            results = ray.get(init_futures)
            for result in results:
                if result.get("status") != "ok":
                    logger.error(f"Server initialization failed: {result}")
                    return False
        except Exception as e:
            logger.error(f"Server initialization error: {e}")
            return False

        # Verify servers are ready
        _ = ray.get(self.servers[0][0].get_task_mapping.remote())

        self._initialized = True
        logger.info(
            f"All servers initialized: "
            f"{self.num_stages * self.num_servers_per_stage} servers, "
            f"{self._total_envs} envs per stage"
        )

        return True

    def _build_task_mapping(self):
        """Build global task → server mapping."""
        self._task_to_server.clear()
        self._task_to_env_indices.clear()
        self._server_task_offsets.clear()

        for server_rank in range(self.num_servers_per_stage):
            # Calculate tasks for this server
            if server_rank < self.remainder_tasks:
                server_num_tasks = self.tasks_per_server + 1
                task_offset = server_rank * (self.tasks_per_server + 1)
            else:
                server_num_tasks = self.tasks_per_server
                task_offset = (
                    self.remainder_tasks * (self.tasks_per_server + 1)
                    + (server_rank - self.remainder_tasks) * self.tasks_per_server
                )

            self._server_task_offsets.append(task_offset)

            # Map global task_ids to this server
            # local_task_id is 0-based within the server
            for local_task_id in range(server_num_tasks):
                global_task_id = task_offset + local_task_id
                self._task_to_server[global_task_id] = server_rank

                # Local env indices for this task on the server
                # Each task has group_size envs, indexed locally within the server
                self._task_to_env_indices[global_task_id] = list(
                    range(local_task_id * self.group_size, (local_task_id + 1) * self.group_size)
                )

        logger.debug(f"Task mapping built: {self.num_tasks} tasks across {self.num_servers_per_stage} servers")

    def get_server_for_task(self, global_task_id: int) -> int:
        """Get the server rank that handles a given global task ID."""
        return self._task_to_server.get(global_task_id, 0)

    def get_env_indices_for_task(self, global_task_id: int) -> list:
        """
        Get LOCAL env indices for a given global task ID.

        These are indices local to the server that handles this task.
        """
        return self._task_to_env_indices.get(global_task_id, [])

    def step(
        self,
        actions: np.ndarray,
        env_indices: list,
        server_rank: int,
        stage_id: int,
        render_last_only: bool = True,
    ) -> dict:
        """
        Send step command to a specific server.

        Args:
            actions: Actions array
            env_indices: LOCAL env indices on that server
            server_rank: Which server
            stage_id: Which pipeline stage
            render_last_only: Only render last step of action chunks

        Returns:
            Response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        if stage_id >= len(self.servers) or server_rank >= len(self.servers[stage_id]):
            raise ValueError(f"Invalid stage_id={stage_id} or server_rank={server_rank}")

        return ray.get(self.servers[stage_id][server_rank].step.remote(actions, env_indices))

    def reset(
        self,
        env_indices: list,
        server_rank: int,
        stage_id: int,
        stabilize: bool = True,
    ) -> dict:
        """
        Send reset command to a specific server.

        Args:
            env_indices: LOCAL env indices on that server
            server_rank: Which server
            stage_id: Which pipeline stage
            stabilize: Whether to run stabilization steps

        Returns:
            Response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        if stage_id >= len(self.servers) or server_rank >= len(self.servers[stage_id]):
            raise ValueError(f"Invalid stage_id={stage_id} or server_rank={server_rank}")

        return ray.get(self.servers[stage_id][server_rank].reset.remote(env_indices, stabilize))

    def step_batched(
        self,
        server_requests: dict[int, tuple[np.ndarray, list]],
        stage_id: int,
        render_last_only: bool = True,
    ) -> dict[int, dict]:
        """
        Send step commands to multiple servers CONCURRENTLY.

        Uses ray.get() for parallel execution across servers.

        Args:
            server_requests: Dict mapping server_rank -> (actions, env_indices)
                e.g., {0: (actions_0, [0,1,2]), 2: (actions_2, [8,9,10])}
            stage_id: Which pipeline stage
            render_last_only: Only render last step of action chunks

        Returns:
            Dict mapping server_rank -> response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Submit all step calls
        futures = {}
        for server_rank, (actions, indices) in server_requests.items():
            if server_rank >= len(self.servers[stage_id]):
                logger.error(f"Invalid server_rank={server_rank}")
                continue
            future = self.servers[stage_id][server_rank].step.remote(actions, indices)
            futures[server_rank] = future

        # Wait for all results
        results = {}
        for server_rank, future in futures.items():
            try:
                results[server_rank] = ray.get(future)
            except Exception as e:
                logger.error(f"Step failed on server {server_rank}: {e}")
                results[server_rank] = {"status": "error", "message": str(e)}

        return results

    def reset_batched(
        self,
        server_requests: dict[int, list],
        stage_id: int,
        stabilize: bool = True,
    ) -> dict[int, dict]:
        """
        Send reset commands to multiple servers CONCURRENTLY.

        Args:
            server_requests: Dict mapping server_rank -> env_indices
                e.g., {0: [0,1,2,3], 2: [8,9,10,11]}
            stage_id: Which pipeline stage
            stabilize: Whether to run stabilization steps

        Returns:
            Dict mapping server_rank -> response dict
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Submit all reset calls
        futures = {}
        for server_rank, indices in server_requests.items():
            if server_rank >= len(self.servers[stage_id]):
                logger.error(f"Invalid server_rank={server_rank}")
                continue
            future = self.servers[stage_id][server_rank].reset.remote(indices, stabilize)
            futures[server_rank] = future

        # Wait for all results
        results = {}
        for server_rank, future in futures.items():
            try:
                results[server_rank] = ray.get(future)
            except Exception as e:
                logger.error(f"Reset failed on server {server_rank}: {e}")
                results[server_rank] = {"status": "error", "message": str(e)}

        return results

    def reset_all_stages_batched(
        self,
        server_requests: dict[int, list],
        stabilize: bool = True,
    ) -> dict[int, dict[int, dict]]:
        """
        Send reset commands to all stages with the same server_requests.

        This is useful for initial reset where all stages need the same env states.

        Args:
            server_requests: Dict mapping server_rank -> env_indices
            stabilize: Whether to run stabilization steps

        Returns:
            Dict mapping stage_id -> (server_rank -> response dict)
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Submit all reset calls for all stages
        futures_by_stage = {}
        for stage_id in range(self.num_stages):
            futures = {}
            for server_rank, indices in server_requests.items():
                if server_rank >= len(self.servers[stage_id]):
                    continue
                future = self.servers[stage_id][server_rank].reset.remote(indices, stabilize)
                futures[server_rank] = future
            futures_by_stage[stage_id] = futures

        # Wait for all results
        results = {}
        for stage_id, futures in futures_by_stage.items():
            stage_results = {}
            for server_rank, future in futures.items():
                try:
                    stage_results[server_rank] = ray.get(future)
                except Exception as e:
                    logger.error(f"Reset failed on stage {stage_id} server {server_rank}: {e}")
                    stage_results[server_rank] = {"status": "error", "message": str(e)}
            results[stage_id] = stage_results

        return results

    @property
    def num_tasks(self) -> int:
        """Get total number of tasks."""
        return self._num_tasks

    @num_tasks.setter
    def num_tasks(self, value: int):
        self._num_tasks = value

    @property
    def total_envs(self) -> int:
        """Get total number of envs per stage."""
        return self._total_envs

    @property
    def initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    def close(self):
        """Close all servers and clean up resources."""
        logger.info("Closing all servers...")

        # Close thread pool
        self._executor.shutdown(wait=True)

        # Close all servers
        close_futures = []
        for stage_servers in self.servers:
            for server in stage_servers:
                close_futures.append(server.close.remote())

        # Wait for all closes
        if close_futures:
            try:
                ray.get(close_futures)
            except Exception as e:
                logger.warning(f"Error closing servers: {e}")

        # Kill servers
        for stage_servers in self.servers:
            for server in stage_servers:
                try:
                    ray.kill(server)
                except Exception:
                    pass

        self.servers = []
        self._initialized = False

        logger.info("All servers closed")

    def __del__(self):
        """Clean up on deletion."""
        if self._initialized:
            self.close()
