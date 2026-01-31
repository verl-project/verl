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
Isaac Lab Simulation Server (Ray-based)

A Ray Actor that runs Isaac Lab multi-task simulation on a single GPU.
Benefits of using Ray servers:
- Unified resource management by Ray
- Simplified deployment (no manual server startup)
- Efficient data transfer via Ray's object store

Architecture:
    - One IsaacServer per GPU
    - Each Server handles a subset of tasks
    - Multiple Servers managed by IsaacServerManager (one manager per stage)

Usage:
    # Create server (Ray will schedule to available GPU)
    server = IsaacServer.options(num_gpus=1).remote(
        env_id="Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        num_tasks=5,
        group_size=8,  # Envs per task
        actor_rank=0,
        task_offset=0,
    )

    # Initialize (must call before step/reset)
    ray.get(server.init_env.remote())

    # Use step/reset
    result = ray.get(server.step.remote(actions, env_indices))
"""

import logging
import os
import shutil
from typing import Any, Optional

import numpy as np
import ray

# NOTE: DO NOT import torch here!
# Isaac Sim (AppLauncher) must initialize CUDA before any GPU libraries are loaded.
# If torch is imported at module level, it will initialize CUDA with default settings,
# causing conflicts with Isaac Sim's GPU resource allocation and rendering pipeline.
# Instead, import torch inside methods after AppLauncher is initialized (see init_env()).
# Subsequent imports are cached by Python (no overhead), but ensure correct initialization order.

logger = logging.getLogger("IsaacServer")


def setup_per_process_caches(server_rank: int, stage_id: int = 0, clear_cache: bool = False):
    """Setup per-process cache directories to avoid locking conflicts.

    Prevents OptiX/shader cache conflicts.
    Must be called BEFORE importing any Isaac/Omniverse modules.

    Important: Each (stage_id, server_rank) pair needs unique cache directories
    to avoid conflicts between different stage servers.

    Args:
        server_rank: The rank of this server within a stage.
        stage_id: The stage ID (for multi-stage training).
        clear_cache: If True, clear existing cache directories before setup.
                     This ensures clean state but increases startup time.
    """

    # Use stage_id/server_rank to create unique cache paths
    cache_suffix = f"stage_{stage_id}/rank_{server_rank}"

    def _setup_cache_dir(base_path: str, clear: bool) -> str:
        """Create cache directory, optionally clearing existing content."""
        cache_dir = os.path.join(base_path, cache_suffix)
        if clear and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    # === OptiX Cache ===
    optix_base = "/tmp/optix_cache"  # Don't inherit from env - always use fresh
    optix_rank_cache = _setup_cache_dir(optix_base, clear_cache)
    os.environ["OPTIX_CACHE_PATH"] = optix_rank_cache
    os.environ["OPTIX7_CACHE_PATH"] = optix_rank_cache

    # === NVIDIA Driver Shader Cache ===
    shader_base = "/tmp/nv_shader_cache"  # Don't inherit from env
    shader_rank_cache = _setup_cache_dir(shader_base, clear_cache)
    os.environ["__GL_SHADER_DISK_CACHE"] = "1"
    os.environ["__GL_SHADER_DISK_CACHE_PATH"] = shader_rank_cache
    os.environ["__GL_SHADER_DISK_CACHE_SKIP_CLEANUP"] = "1"

    # === Omniverse Kit Cache (includes shader cache) ===
    ov_base = "/tmp/ov_cache"  # Don't inherit from env
    ov_rank_cache = _setup_cache_dir(ov_base, clear_cache)
    os.environ["OMNI_KIT_CACHE_DIR"] = ov_rank_cache
    os.environ["OMNI_USER_CACHE_DIR"] = ov_rank_cache  # Also set user cache
    os.environ["CARB_DATA_PATH"] = os.path.join(ov_rank_cache, "carb")  # Carbonite data

    # === XDG Base Directories ===
    # Note: Kit in Isaac Sim container uses hardcoded paths (/isaac-sim/kit/data/)
    # which ignore XDG variables. The kvdb conflict warning is benign - Kit
    # gracefully disables kvdb when another process holds the lock.
    # This doesn't affect training or rendering, only settings persistence.
    xdg_base = "/tmp/xdg_home"
    xdg_rank_home = _setup_cache_dir(xdg_base, clear_cache)
    os.environ["XDG_DATA_HOME"] = os.path.join(xdg_rank_home, "data")
    os.environ["XDG_CONFIG_HOME"] = os.path.join(xdg_rank_home, "config")
    os.environ["XDG_CACHE_HOME"] = os.path.join(xdg_rank_home, "cache")
    os.makedirs(os.environ["XDG_DATA_HOME"], exist_ok=True)
    os.makedirs(os.environ["XDG_CONFIG_HOME"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

    # Log cache configuration for debugging
    cleared_msg = " (cleared)" if clear_cache else ""
    logger.info(f"[Stage {stage_id} Rank {server_rank}] Cache directories configured{cleared_msg}:")
    logger.info(f"  OptiX:  {optix_rank_cache}")
    logger.info(f"  Shader: {shader_rank_cache}")
    logger.info(f"  OV Kit: {ov_rank_cache}")
    logger.info(f"  XDG:    {xdg_rank_home}")


@ray.remote(num_gpus=1)
class IsaacServer:
    """
    Ray Actor for Isaac Lab multi-task simulation.

    Managed by Ray for unified resource management.

    Key features:
    1. Uses Ray's server methods directly
    2. GPU allocated by Ray scheduler (num_gpus=1)
    3. Data transferred via Ray's object store (automatic serialization)
    4. Lifecycle managed by Ray (no manual start/stop)

    Thread Safety:
        Ray servers are single-threaded by default, so all method calls
        are serialized.
    """

    def __init__(
        self,
        env_id: str = "Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        num_tasks: int = 5,
        group_size: int = 8,  # Fixed envs per task
        actor_rank: int = 0,
        task_offset: int = 0,
        render_last_only: bool = True,
        camera_height: int = 256,
        camera_width: int = 256,
        stage_id: int = 0,
    ):
        """
        Initialize the Isaac Sim Server.

        Args:
            env_id: Gymnasium environment ID for multi-task Isaac env
            num_tasks: Number of tasks this server handles
            group_size: Number of parallel envs per task (fixed for all tasks)
            actor_rank: Rank of this server (0 to num_servers-1)
            task_offset: Global task ID offset for this server
            render_last_only: If True, only render on the last step of action chunks
            camera_height: Camera image height
            camera_width: Camera image width
            stage_id: Pipeline stage ID (for logging and identification)
        """
        self.env_id = env_id
        self.num_tasks = num_tasks
        self.group_size = group_size
        self.actor_rank = actor_rank
        self.task_offset = task_offset
        self.render_last_only = render_last_only
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.stage_id = stage_id

        # Total envs = num_tasks * group_size
        self.total_envs = num_tasks * group_size

        # Task ID to env indices mapping (local to this server)
        # task_id here is LOCAL (0 to num_tasks-1)
        self.task_to_env_indices = {
            task_id: list(range(task_id * group_size, (task_id + 1) * group_size)) for task_id in range(num_tasks)
        }

        # Will be initialized in init_env()
        self.env = None
        self.simulation_app = None
        self.device = None
        self.action_dim = None
        self._initialized = False

        logger.info(
            f"[Stage {stage_id} Server {actor_rank}] Created: "
            f"{num_tasks} tasks (offset={task_offset}), "
            f"group_size={group_size}, {self.total_envs} envs"
        )

    def init_env(self) -> dict:
        """
        Initialize the Isaac Lab environment.

        This is separated from __init__ because Isaac Lab initialization
        requires GPU context which may not be ready during server creation.

        Returns:
            dict with status and environment info
        """
        if self._initialized:
            return {
                "status": "ok",
                "message": "Already initialized",
                "num_tasks": self.num_tasks,
                "group_size": self.group_size,
                "total_envs": self.total_envs,
            }

        # Setup per-process caches (unique per stage + actor to avoid conflicts)
        # Set ISAAC_CLEAR_CACHE=1 env var to clear caches on startup (slower but cleaner)
        clear_cache = os.environ.get("ISAAC_CLEAR_CACHE", "0") == "1"
        setup_per_process_caches(self.actor_rank, self.stage_id, clear_cache=clear_cache)

        # Set environment variables for Isaac Lab config
        # GROUP_SIZE = envs per task (not total envs!)
        os.environ["GROUP_SIZE"] = str(self.group_size)
        os.environ["TASK_OFFSET"] = str(self.task_offset)
        os.environ["NUM_TASKS"] = str(self.num_tasks)

        logger.info(f"[Stage {self.stage_id} Actor {self.actor_rank}] Initializing Isaac environment: {self.env_id}")

        # Import torch after Isaac Sim initialization (see module-level comment for details)
        # This is the first real import in the process - torch will initialize CUDA here.
        # Subsequent imports in other methods are no-ops (cached lookup only).
        import torch

        # Detect GPU - Ray assigns GPU via CUDA_VISIBLE_DEVICES, so device is always "cuda:0" from actor's view
        num_gpus = torch.cuda.device_count()
        self.device = "cuda:0" if num_gpus > 0 else "cpu"

        logger.info(f"[Stage {self.stage_id} Actor {self.actor_rank}] Visible GPUs: {num_gpus}, using {self.device}")

        # Import Isaac Lab components - follow IsaacEnv pattern exactly
        import gymnasium as gym
        from isaaclab.app import AppLauncher

        # Use simple kwargs initialization (same as IsaacEnv)
        launch_args = {"headless": True, "enable_cameras": True}

        # Optional: Flush NVIDIA driver shader cache to fix rendering noise caused by cache overflow
        # Set ISAAC_FLUSH_SHADER_CACHE=1 if you see:
        #   - Noisy/corrupted rendered images
        #   - Warning: "ShaderCache: 95 percent of the driver's shadercache size limit has been reached"
        # Note: This makes startup slower (cache rebuild) but fixes render quality
        # After one flush, you can remove the env var as cache will be clean
        if os.environ.get("ISAAC_FLUSH_SHADER_CACHE", "0") == "1":
            logger.info(
                f"[Stage {self.stage_id} Actor {self.actor_rank}] "
                "Flushing shader cache (startup will be slower but fixes render quality)"
            )
            extra_args = ["--/renderer/shadercache/driverDiskCache/flush=true"]
            launch_args["extra_args"] = extra_args

        app_launcher = AppLauncher(**launch_args)
        self.simulation_app = app_launcher.app

        # Force franka registration (same as IsaacEnv)
        import isaaclab_playground.tasks.manipulation.libero.config.franka  # noqa

        # Now import Isaac Lab task utilities
        from isaaclab_tasks.utils import parse_env_cfg

        # Parse environment config (same as IsaacEnv._init_env)
        env_cfg = parse_env_cfg(self.env_id, num_envs=self.total_envs)

        # Configure environment (following IsaacEnv pattern exactly)
        env_cfg.env_name = f"stage{self.stage_id}_actor{self.actor_rank}"
        env_cfg.sim.device = self.device
        env_cfg.sim.physx.enable_ccd = True
        env_cfg.terminations.time_out = None
        env_cfg.observations.policy.concatenate_terms = False

        # Override camera dimensions if supported
        if hasattr(env_cfg, "camera_height"):
            env_cfg.camera_height = self.camera_height
            env_cfg.camera_width = self.camera_width
            if hasattr(env_cfg, "recreate_cameras"):
                env_cfg.recreate_cameras()
            logger.info(
                f"[Stage {self.stage_id} Actor {self.actor_rank}] Set camera: {self.camera_width}x{self.camera_height}"
            )

        # Set task offset if supported
        if hasattr(env_cfg, "task_offset"):
            env_cfg.task_offset = self.task_offset

        # Ensure correct num_envs
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = self.total_envs

        # Create environment (same as IsaacEnv)
        self.env = gym.make(self.env_id, cfg=env_cfg).unwrapped
        self.action_dim = self.env.action_space.shape[-1]

        # Get decimation (number of sim steps per action) for efficient rendering
        # Set render_interval = decimation to render only once per action (at the last sim step)
        self.decimation = self.env.cfg.decimation

        # Verify envs created
        actual_num_envs = self.env.num_envs
        if actual_num_envs != self.total_envs:
            logger.warning(
                f"[Stage {self.stage_id} Actor {self.actor_rank}] "
                f"Env count mismatch: requested {self.total_envs}, got {actual_num_envs}"
            )
            self.total_envs = actual_num_envs
            # Rebuild task mapping
            self.group_size = actual_num_envs // self.num_tasks if self.num_tasks > 0 else actual_num_envs
            self.task_to_env_indices = {
                task_id: list(range(task_id * self.group_size, (task_id + 1) * self.group_size))
                for task_id in range(self.num_tasks)
            }

        # Initial reset
        self.env.reset()
        self._initialized = True

        logger.info(
            f"[Stage {self.stage_id} Actor {self.actor_rank}] "
            f"Initialized: {self.total_envs} envs, action_dim={self.action_dim}"
        )

        return {
            "status": "ok",
            "num_tasks": self.num_tasks,
            "group_size": self.group_size,
            "total_envs": self.total_envs,
            "action_dim": self.action_dim,
            "task_offset": self.task_offset,
            "actor_rank": self.actor_rank,
            "stage_id": self.stage_id,
        }

    def get_task_mapping(self) -> dict:
        """Return the task ID to env indices mapping."""
        return {
            "status": "ok",
            "task_to_env_indices": self.task_to_env_indices,
            "num_tasks": self.num_tasks,
            "group_size": self.group_size,
            "total_envs": self.total_envs,
            "task_offset": self.task_offset,
            "actor_rank": self.actor_rank,
            "stage_id": self.stage_id,
        }

    def step(self, actions: np.ndarray, env_indices: list) -> dict:
        """
        Execute step on specified environments.

        Args:
            actions: Actions array, shape (len(env_indices), action_dim) or
                     (len(env_indices), num_chunks, action_dim) for chunked actions
            env_indices: List of env indices to step

        Returns:
            dict with obs, rewards, terminations, truncations, infos
        """
        if not self._initialized:
            raise RuntimeError("Actor not initialized. Call init_env() first.")

        # Import torch (cached lookup, no overhead - see module-level comment)
        import torch

        actions = np.array(actions)

        logger.debug(
            f"[Stage {self.stage_id} Actor {self.actor_rank}] "
            f"Step: {len(env_indices)} envs, indices={env_indices[:5]}..."
        )

        # Check if actions have chunk dimension: [num_envs, num_chunks, action_dim]
        if len(actions.shape) == 3:
            return self._handle_chunk_step(actions, env_indices)

        # Single step: [num_envs, action_dim]
        full_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
        full_actions[env_indices] = torch.tensor(actions, device=self.device, dtype=torch.float32)

        # Step all envs
        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

        # Extract results for requested env indices
        response = {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
            "rewards": rewards[env_indices].cpu().numpy(),
            "terminations": terminations[env_indices].cpu().numpy(),
            "truncations": truncations[env_indices].cpu().numpy(),
            "infos": self._extract_infos(infos, env_indices),
        }

        return response

    def _handle_chunk_step(self, chunk_actions: np.ndarray, env_indices: list) -> dict:
        """
        Handle action chunk: execute each action in the chunk sequentially.

        Args:
            chunk_actions: [num_envs, num_actions, action_dim] - one action chunk containing multiple actions
            env_indices: list of env indices to step

        Returns:
            dict with accumulated rewards and final obs
        """
        # Import torch (cached lookup, no overhead - see module-level comment)
        import torch

        num_actions = chunk_actions.shape[1]

        action_rewards = []
        action_terminations = []
        action_truncations = []

        # Skip rendering by setting a very large interval for all intermediate actions if render_last_only is enabled
        if self.render_last_only and hasattr(self.env.unwrapped, "cfg"):
            self.env.unwrapped.cfg.sim.render_interval = 999999

        for action_idx in range(num_actions):
            is_last_action = action_idx == num_actions - 1

            # Enable rendering for the last action to get observations
            # Set to decimation to render only once at the end of the action (avoids multiple renders per action)
            if is_last_action and self.render_last_only and hasattr(self.env.unwrapped, "cfg"):
                self.env.unwrapped.cfg.sim.render_interval = self.decimation

            # Get current action from the chunk
            actions = chunk_actions[:, action_idx, :]

            # Build full action tensor
            full_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
            full_actions[env_indices] = torch.tensor(actions, device=self.device, dtype=torch.float32)

            # Step all envs
            obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

            # Collect results for this action
            action_rewards.append(rewards[env_indices].cpu().numpy())
            action_terminations.append(terminations[env_indices].cpu().numpy())
            action_truncations.append(truncations[env_indices].cpu().numpy())

        # Stack action results: [num_envs, num_actions]
        stacked_rewards = np.stack(action_rewards, axis=1)
        stacked_terminations = np.stack(action_terminations, axis=1)
        stacked_truncations = np.stack(action_truncations, axis=1)

        return {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
            "rewards": stacked_rewards,
            "terminations": stacked_terminations,
            "truncations": stacked_truncations,
            "infos": self._extract_infos(infos, env_indices),
        }

    def reset(self, env_indices: Optional[list] = None, stabilize: bool = True) -> dict:
        """
        Reset specified environments.

        Args:
            env_indices: List of env indices to reset (None for all)
            stabilize: Whether to run stabilization steps (10 zero-action steps)

        Returns:
            dict with obs
        """
        if not self._initialized:
            raise RuntimeError("Actor not initialized. Call init_env() first.")

        # Import torch (cached lookup, no overhead - see module-level comment)
        import torch

        if env_indices is None:
            env_indices = list(range(self.total_envs))

        logger.debug(f"[Stage {self.stage_id} Actor {self.actor_rank}] Reset: {len(env_indices)} envs")

        # Validate env_indices
        actual_num_envs = self.env.unwrapped.num_envs
        max_idx = max(env_indices) if env_indices else 0
        if max_idx >= actual_num_envs:
            raise RuntimeError(f"env_indices out of bounds: max={max_idx}, but env only has {actual_num_envs} envs")

        # Use Isaac Lab's internal _reset_idx for partial reset
        reset_env_ids = torch.tensor(env_indices, device=self.device, dtype=torch.long)
        self.env.unwrapped._reset_idx(reset_env_ids)

        # Stabilize: run zero-action steps to let physics settle
        if stabilize:
            zero_actions = torch.zeros(self.total_envs, self.action_dim, device=self.device)
            for _ in range(10):
                obs, _, _, _, infos = self.env.step(zero_actions)
        else:
            obs = self.env.unwrapped.observation_manager.compute()

        return {
            "status": "ok",
            "obs": self._extract_obs(obs, env_indices),
        }

    def ping(self) -> dict:
        """Health check."""
        return {
            "status": "ok",
            "message": "pong",
            "initialized": self._initialized,
            "stage_id": self.stage_id,
            "actor_rank": self.actor_rank,
        }

    def close(self) -> dict:
        """Clean up resources."""
        logger.info(f"[Stage {self.stage_id} Actor {self.actor_rank}] Closing...")

        if self.env:
            self.env.close()
            self.env = None

        if self.simulation_app:
            self.simulation_app.close()
            self.simulation_app = None

        self._initialized = False

        return {"status": "ok", "message": "Actor closed"}

    def _to_cpu_numpy(self, value: Any) -> Any:
        """Recursively convert any CUDA tensors to CPU numpy arrays."""
        # Import torch (cached lookup, no overhead - see module-level comment)
        import torch

        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        elif isinstance(value, dict):
            return {k: self._to_cpu_numpy(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return type(value)(self._to_cpu_numpy(v) for v in value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            if hasattr(value, "cpu") and callable(value.cpu):
                try:
                    return value.cpu().numpy()
                except Exception:
                    pass
            return value

    def _extract_obs(self, obs: Any, env_indices: list) -> dict:
        """Extract observations for specified env indices."""
        # Import torch (cached lookup, no overhead - see module-level comment)
        import torch

        if isinstance(obs, dict):
            result = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value[env_indices].cpu().numpy()
                elif isinstance(value, dict):
                    result[key] = self._extract_obs(value, env_indices)
                elif isinstance(value, np.ndarray):
                    result[key] = value[env_indices]
                else:
                    result[key] = self._to_cpu_numpy(value)
            return result
        elif isinstance(obs, torch.Tensor):
            return obs[env_indices].cpu().numpy()
        elif isinstance(obs, np.ndarray):
            return obs[env_indices]
        else:
            return self._to_cpu_numpy(obs)

    def _extract_infos(self, infos: dict, env_indices: list) -> dict:
        """Extract infos for specified env indices."""
        # Import torch (cached lookup, no overhead - see module-level comment)
        import torch

        result = {}
        if infos:
            for key, value in infos.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[0] >= max(env_indices) + 1:
                        result[key] = value[env_indices].cpu().numpy()
                    else:
                        result[key] = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    if value.ndim > 0 and value.shape[0] >= max(env_indices) + 1:
                        result[key] = value[env_indices]
                    else:
                        result[key] = value
                elif isinstance(value, dict):
                    result[key] = self._extract_infos(value, env_indices)
                else:
                    result[key] = self._to_cpu_numpy(value)
        return result
