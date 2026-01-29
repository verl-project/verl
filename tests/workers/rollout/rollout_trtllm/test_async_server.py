# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import ray
import torch
from ray.util import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.single_controller.ray import RayResourcePool, SubRayResourcePool
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMHttpServer, TRTLLMReplica


class TestTRTLLMReplica:
    def test_placement_group_with_sub_ray_resource_pool(self):
        """
        Scenario: SubRayResourcePool, 1 node, 8 GPUs, TP=4, replica_rank=1
        SubRayResourcePool pre-assigns start_bundle_index=4 for replica 1.
        Expected: Replica 1 gets bundles [4, 5, 6, 7]
        """
        with patch("verl.workers.rollout.trtllm_rollout.trtllm_async_server.ray"):
            mock_config = MagicMock()
            mock_config.tensor_model_parallel_size = 4
            mock_config.data_parallel_size = 1
            mock_config.pipeline_model_parallel_size = 1

            replica = TRTLLMReplica(
                replica_rank=1,
                config=mock_config,
                model_config=MagicMock(),
                gpus_per_node=8,
            )

            mock_pg = MagicMock()
            mock_pg.bundle_count = 8

            mock_resource_pool = MagicMock(spec=SubRayResourcePool)
            mock_resource_pool.pgs = [mock_pg]
            mock_resource_pool.subgroup_world_size = 4
            mock_resource_pool.start_bundle_index = 4

            replica.resource_pool = mock_resource_pool
            replica.world_size = 4  # TP=4

            pgs, bundle_indices = replica.get_pgs_and_bundle_indices()

            assert len(pgs) == 1
            assert pgs[0] == mock_pg
            assert len(bundle_indices) == 1
            assert bundle_indices[0] == [4, 5, 6, 7]

    def test_placement_group_with_ray_resource_pool(self):
        """
        Scenario: RayResourcePool, 1 node, 8 GPUs, TP=2, replica_rank=1
        RayResourcePool calculates: local_bundle_index = world_size * replica_rank = 2 * 1 = 2
        Expected: Replica 1 gets bundles [2, 3]
        """
        with patch("verl.workers.rollout.trtllm_rollout.trtllm_async_server.ray"):
            mock_config = MagicMock()
            mock_config.tensor_model_parallel_size = 2
            mock_config.data_parallel_size = 1
            mock_config.pipeline_model_parallel_size = 1

            replica = TRTLLMReplica(
                replica_rank=1,
                config=mock_config,
                model_config=MagicMock(),
                gpus_per_node=8,
            )

            mock_pg = MagicMock()
            mock_pg.bundle_count = 8

            mock_resource_pool = MagicMock()
            mock_resource_pool.pgs = [mock_pg]

            replica.resource_pool = mock_resource_pool
            replica.world_size = 2  # TP=2

            pgs, bundle_indices = replica.get_pgs_and_bundle_indices()

            assert len(pgs) == 1
            assert pgs[0] == mock_pg
            assert len(bundle_indices) == 1
            assert bundle_indices[0] == [2, 3]


class TestTRTLLMHttpServer:
    def test_async_memory_management(self):
        """Test TRT-LLM async memory management (sleep) reduces memory usage."""
        from hydra import compose, initialize_config_dir

        try:
            os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
            ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

            config_dir = os.path.abspath("verl/verl/trainer/config")
            if not os.path.exists(config_dir):
                config_dir = os.path.abspath("verl/trainer/config")

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                config = compose(config_name="ppo_trainer")

            config.trainer.n_gpus_per_node = 1
            config.trainer.nnodes = 1
            config.actor_rollout_ref.model.path = os.path.expanduser(
                "/lustre/fsw/coreai_dlalgo_llm/erinh/llm-models/Qwen2.5-1.5B-Instruct"
            )
            config.actor_rollout_ref.rollout.name = "trtllm"
            config.actor_rollout_ref.rollout.mode = "async"
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
            config.actor_rollout_ref.rollout.free_cache_engine = True  # Enable memory management

            rollout_config = config.actor_rollout_ref.rollout
            model_config = config.actor_rollout_ref.model

            resource_pool = RayResourcePool(
                process_on_nodes=[1],
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="test_rollout",
            )
            pgs = resource_pool.get_placement_groups()
            bundle_indices = [[0]]

            pg_data = placement_group_table(pgs[0])
            node_id = pg_data["bundles_to_node_id"][bundle_indices[0][0]]

            server = TRTLLMHttpServer.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name="trtllm_server_test_0",
            ).remote(
                config=rollout_config,
                model_config=model_config,
                is_reward_model=False,
                rollout_mode=RolloutMode.COLOCATED,
                workers=[],
                replica_rank=0,
                max_colocate_count=1,
                pgs=pgs,
                bundle_indices=bundle_indices,
            )

            ray.get(server.launch_server.remote())
            device_ids = ray.get(server.report_device_ids.remote())
            print(f"TRTLLM device UUIDs: {device_ids}")

            def _uuid_to_device_index(device_uuid: str) -> int | None:
                for idx in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(idx)
                    uuid = getattr(props, "uuid", None)
                    if uuid is None:
                        # fall back to rank 0
                        return 0
                    if isinstance(uuid, bytes):
                        uuid_str = uuid.decode("utf-8", errors="ignore")
                    else:
                        uuid_str = str(uuid)
                    if uuid_str == device_uuid or uuid_str in device_uuid:
                        print(f"Mapped device UUID {device_uuid} to torch device index {idx}")
                        return idx
                return 0

            def get_gpu_memory_mb_for_device(device_uuid: str) -> float:
                device_index = _uuid_to_device_index(device_uuid)
                prev_device = torch.cuda.current_device()
                torch.cuda.set_device(device_index)
                mem_free, mem_total = torch.cuda.mem_get_info()
                torch.cuda.set_device(prev_device)
                return (mem_total - mem_free) / (1024**2)

            baseline_memory_mb = get_gpu_memory_mb_for_device(device_ids[0])
            print(f"   Baseline memory: {baseline_memory_mb:.2f} MB")

            ray.get(server.sleep.remote())
            time.sleep(2)

            sleep_memory_mb = get_gpu_memory_mb_for_device(device_ids[0])
            memory_freed_mb = baseline_memory_mb - sleep_memory_mb
            print(f"   Memory after sleep: {sleep_memory_mb:.2f} MB")
            print(f"   Memory freed: {memory_freed_mb:.2f} MB")

            assert memory_freed_mb >= baseline_memory_mb * 0.6, (
                f"Expected sleep() to free >=60% of baseline memory. "
                f"Baseline: {baseline_memory_mb:.2f} MB, freed: {memory_freed_mb:.2f} MB."
            )

        finally:
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)
