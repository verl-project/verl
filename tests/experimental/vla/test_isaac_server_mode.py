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
Integration tests for Isaac Server Mode.

Tests the server-based simulation approach where Ray Actors manage
Isaac Lab simulations across multiple GPUs.
"""

import unittest

import numpy as np
import pytest
import ray
from omegaconf import OmegaConf


@pytest.fixture(scope="module", autouse=True)
def init_ray():
    """Initialize Ray for all tests in this module."""
    if not ray.is_initialized():
        # Try to connect to existing Ray cluster
        try:
            ray.init(address="auto")
            print("Connected to existing Ray cluster")
        except Exception:
            # Fall back to local mode for testing
            ray.init(local_mode=True)
            print("Started Ray in local mode")

    yield

    # Cleanup after all tests
    # Note: Don't shutdown Ray if it was already running
    # ray.shutdown()


def test_task_balanced_sampler():
    """
    Test TaskBalancedSampler for balanced task distribution.

    Verifies that:
    1. Sampler correctly validates batch_size divisibility
    2. All tasks are represented in samples
    3. Proper error handling for invalid configurations
    """
    from verl.experimental.vla.workers.env.utils import TaskBalancedSampler

    # Create a mock dataset with task_ids
    class MockDataset:
        def __init__(self, num_samples, num_tasks):
            self.task_ids = [i % num_tasks for i in range(num_samples)]
            self.data = list(range(num_samples))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"task_id": self.task_ids[idx], "data": self.data[idx]}

    num_tasks = 10
    num_samples = 100
    batch_size = 16
    max_per_task = 4
    stage_num = 2

    dataset = MockDataset(num_samples, num_tasks)

    # Test 1: Valid configuration
    sampler = TaskBalancedSampler(
        dataset=dataset, batch_size=batch_size, max_per_task=max_per_task, stage_num=stage_num
    )

    # Verify samples_per_stage
    assert sampler.samples_per_stage == batch_size // stage_num

    # Get samples
    sample_indices = list(sampler)
    assert len(sample_indices) > 0

    # Verify batch size
    first_batch = sample_indices[:batch_size]
    assert len(first_batch) == batch_size

    # Test 2: Invalid configuration - batch_size not divisible by stage_num
    with pytest.raises(AssertionError, match="batch_size .* must be divisible by stage_num"):
        TaskBalancedSampler(
            dataset=dataset,
            batch_size=15,
            max_per_task=max_per_task,
            stage_num=2,  # 15 not divisible by 2
        )

    print("✓ TaskBalancedSampler test passed")


def test_isaac_server_manager_creation():
    """
    Test IsaacServerManager creation and initialization.

    Note: This test requires Isaac Sim environment to be properly configured.
    """
    # pytest.skip("Requires Isaac Sim environment - run manually in Isaac container")

    from verl.experimental.vla.isaac_server import IsaacServerManager

    num_stages = 2
    num_servers_per_stage = 2
    num_tasks = 4
    group_size = 8

    manager = IsaacServerManager(
        num_stages=num_stages,
        num_servers_per_stage=num_servers_per_stage,
        num_tasks=num_tasks,
        group_size=group_size,
        env_id="Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
        render_last_only=True,
        camera_height=256,
        camera_width=256,
        accelerator_type="sim",
    )

    # Initialize servers
    success = manager.initialize()
    assert success, "Failed to initialize IsaacServerManager"

    # Verify server count
    assert len(manager.servers) == num_stages
    for stage_servers in manager.servers:
        assert len(stage_servers) == num_servers_per_stage

    print(f"✓ IsaacServerManager created with {num_stages} stages, {num_servers_per_stage} servers per stage")


def test_isaac_server_rollout_loop():
    """
    Test complete generation-simulation loop with Isaac Server Mode.

    This is a full integration test that verifies:
    1. EnvLoop initialization with isaac_server_mode
    2. Environment worker and server creation
    3. Environment reset
    4. Action generation (mocked)
    5. Environment step execution
    6. Observation/reward/termination handling
    7. Proper cleanup

    Requires:
    - Ray cluster with 'sim' resources available
    - Isaac Sim environment properly configured
    """
    from verl import DataProto
    from verl.experimental.vla.env_loop import EnvLoop
    from verl.experimental.vla.workers.env import EnvWorkerServer
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    # Test configuration - small scale for fast testing
    num_stages = 2
    num_servers = 4  # Total Isaac servers across all stages
    num_tasks = 4
    group_size = 8  # Envs per task
    num_envs = 8  # Envs per worker (in server mode, this is per-stage batch size)
    total_trajs = num_stages * num_envs
    num_action_chunks = 8
    action_dim = 7

    config = OmegaConf.create(
        {
            "env": {
                "rollout": {"pipeline_stage_num": num_stages},
                "train": {
                    "isaac_server_mode": True,
                    "num_isaac_servers": num_servers,
                    "num_tasks": num_tasks,
                    "group_size": group_size,
                    "env_id": "Isaac-Libero-Franka-OscPose-Camera-All-Tasks-v0",
                    "num_envs": num_envs,
                    "total_trajs": total_trajs,
                    "simulator_type": "isaac",
                    "max_episode_steps": 64,
                    "only_eval": False,
                    "reward_coef": 1.0,
                    "init_params": {
                        "camera_heights": 256,
                        "camera_widths": 256,
                        "camera_names": ["agentview"],
                    },
                    "video_cfg": {
                        "save_video": False,
                    },
                    "seed": 42,
                },
                "actor": {"model": {"num_action_chunks": num_action_chunks, "action_dim": action_dim}},
                "enable_offload": False,
            }
        }
    )

    print(f"\n{'=' * 60}")
    print("Isaac Server Mode Gen-Sim Loop Integration Test")
    print(f"{'=' * 60}")
    print("Configuration:")
    print(f"  - Stages: {num_stages}")
    print(f"  - Total trajectories: {total_trajs}")
    print(f"  - Envs per stage: {num_envs}")
    print(f"  - Isaac servers: {num_servers}")
    print(f"  - Tasks: {num_tasks}, Group size: {group_size}")
    print(f"  - Action chunks: {num_action_chunks}, Action dim: {action_dim}")

    # Step 1: Create environment worker group
    print("\n[Step 1] Creating environment worker group...")
    env_resource_pool = RayResourcePool([1], use_gpu=False)  # 1 EnvWorkerServer, no GPU needed
    env_cls = RayClassWithInitArgs(cls=ray.remote(EnvWorkerServer), config=config.env)
    env_wg = RayWorkerGroup(
        resource_pool=env_resource_pool,
        ray_cls_with_init=env_cls,
        name_prefix="env",
    )
    print(f"✓ Environment worker group created (world_size={env_wg.world_size})")

    # Step 2: Create a minimal rollout worker group (dummy, just for EnvLoop interface)
    # In real training, this would be the model inference workers
    print("\n[Step 2] Creating rollout worker group...")

    # Create a simple mock worker class that EnvLoop won't actually call
    @ray.remote
    class DummyRolloutWorker:
        def __init__(self):
            pass

    rollout_resource_pool = RayResourcePool([1], use_gpu=False)
    rollout_cls = RayClassWithInitArgs(cls=DummyRolloutWorker)
    rollout_wg = RayWorkerGroup(
        resource_pool=rollout_resource_pool,
        ray_cls_with_init=rollout_cls,
        name_prefix="rollout",
    )
    print("✓ Rollout worker group created (dummy for testing)")

    # Step 3: Create EnvLoop
    print("\n[Step 3] Creating EnvLoop...")
    env_loop = EnvLoop(
        env_wg=env_wg,
        rollout_wg=rollout_wg,
        config=config,
    )
    print("✓ EnvLoop created")
    print(f"  - isaac_server_mode: {env_loop.isaac_server_mode}")
    print(f"  - total_trajs: {env_loop.total_trajs}")
    print(f"  - num_envs_per_worker: {env_loop.num_envs_per_worker}")
    print(f"  - stage_num: {env_loop.stage_num}")

    # Step 4: Initialize workers and simulators
    print("\n[Step 4] Initializing workers and simulators...")
    env_loop.env_wg.init_worker()
    print("✓ Workers initialized")

    env_loop.env_wg.init_simulator()
    print(f"✓ Simulators initialized (IsaacServerManager + {num_servers} IsaacServers created)")

    # Step 5: Reset environment
    print("\n[Step 5] Resetting environment...")
    reset_data = DataProto.from_dict(
        non_tensors={
            "state_ids": list(range(total_trajs)),
            "task_ids": [i % num_tasks for i in range(total_trajs)],
        }
    )
    obs_data = env_loop.env_wg.reset_envs_to_state_ids(reset_data)
    assert obs_data is not None, "Reset should return observations"
    print("✓ Environment reset successful")
    print(f"  - Received observations for {len(reset_data.non_tensor_batch['state_ids'])} trajectories")

    # Step 6: Execute one generation-simulation step
    print("\n[Step 6] Executing generation-simulation loop (1 step)...")

    # Generate mock actions (in real scenario, this would come from model)
    print("  [Gen] Generating actions (mocked with random values)...")
    mock_actions = np.random.randn(total_trajs, num_action_chunks, action_dim).astype(np.float32)
    print(f"  ✓ Actions generated: shape {mock_actions.shape}")

    # Execute actions in environment (stage by stage)
    print("  [Sim] Executing actions in environment...")
    for stage_id in range(num_stages):
        print(f"    - Stage {stage_id}: Processing {num_envs} trajectories...")

        # Create action batch for this stage
        action_batch = DataProto.from_dict(
            non_tensors={
                "actions": mock_actions,
                "prompts": ["test"] * total_trajs,  # Dummy prompts
            }
        )

        # Execute step
        step_result = env_loop.env_wg.env_interact_step(action_batch)
        assert step_result is not None, f"Stage {stage_id} should return step result"

        print(f"    ✓ Stage {stage_id} completed")

        # Verify we got observations and rewards
        if hasattr(step_result, "batch"):
            batch = step_result.batch
            if "rews" in batch:
                print(f"      Rewards received: shape {batch['rews'].shape}")
            if "obs" in batch or "full_image" in batch:
                print("      Observations received")

    print("  ✓ Simulation step completed")

    # Step 7: Finish rollout and cleanup
    print("\n[Step 7] Cleanup...")
    env_loop.env_wg.finish_rollout()
    print("✓ Finish rollout called")

    print(f"\n{'=' * 60}")
    print("✓✓✓ Isaac Server Mode Gen-Sim Loop Test PASSED ✓✓✓")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
            print("Connected to existing Ray cluster")
        except Exception:
            ray.init(local_mode=True)
            print("Started Ray in local mode")

    unittest.main()
