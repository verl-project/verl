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

            resource_pool = SubRayResourcePool(
                placement_groups=[mock_pg],
                start_bundle_index=4,
                subgroup_world_size=4,
            )

            replica.resource_pool = resource_pool
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

            resource_pool = RayResourcePool(
                process_on_nodes=[8],
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="test_rollout",
            )
            resource_pool.pgs = [mock_pg]

            replica.resource_pool = resource_pool
            replica.world_size = 2  # TP=2

            pgs, bundle_indices = replica.get_pgs_and_bundle_indices()

            assert len(pgs) == 1
            assert pgs[0] == mock_pg
            assert len(bundle_indices) == 1
            assert bundle_indices[0] == [2, 3]


class TestTRTLLMHttpServer:
    @staticmethod
    def _build_rollout_config(*, response_length: int | None = None, free_cache_engine: bool = False):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = 1
        config.trainer.nnodes = 1
        model_root = os.path.expanduser(os.getenv("TRTLLM_TEST_MODEL_PATH_ROOT", "~/models"))
        config.actor_rollout_ref.model.path = os.path.join(model_root, "Qwen/Qwen2.5-0.5B-Instruct")
        config.actor_rollout_ref.rollout.name = "trtllm"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        if response_length is not None:
            config.actor_rollout_ref.rollout.response_length = response_length
        if free_cache_engine:
            config.actor_rollout_ref.rollout.free_cache_engine = True

        return config.actor_rollout_ref.rollout, config.actor_rollout_ref.model

    @staticmethod
    def _create_server(rollout_config, model_config, *, name: str):
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

        return TRTLLMHttpServer.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
            name=name,
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

    def test_async_generate(self):
        """Test TRT-LLM generate method with real model."""
        try:
            os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
            ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

            rollout_config, model_config = self._build_rollout_config(response_length=50)

            server = self._create_server(
                rollout_config,
                model_config,
                name="trtllm_server_test_generate",
            )

            ray.get(server.launch_server.remote())

            # Test generate with a simple prompt
            prompt_ids = [1, 2, 3, 4, 5]  # Simple test prompt
            sampling_params = {
                "temperature": 1.0,
                "top_k": 0,
                "logprobs": 1,
            }
            request_id = "test_request_1"

            result = ray.get(server.generate.remote(prompt_ids, sampling_params, request_id))

            print(f"Result: {result}")
            # Verify the result structure
            assert hasattr(result, "token_ids"), "Result should have token_ids attribute"
            assert hasattr(result, "log_probs"), "Result should have log_probs attribute"
            assert isinstance(result.token_ids, list), "token_ids should be a list"
            assert len(result.token_ids) > 0, "Generated tokens should not be empty"

            # Verify logprobs are returned when requested
            assert result.log_probs is not None, "log_probs should not be None when requested"
            assert len(result.log_probs) == len(result.token_ids), "log_probs length should match token_ids"

            print(f"Generated {len(result.token_ids)} tokens")
            print(f"Token IDs: {result.token_ids[:10]}...")  # Print first 10 tokens
            print(f"Log probs: {result.log_probs[:10]}...")  # Print first 10 log probs

        finally:
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)

    def test_async_memory_management(self):
        """Test TRT-LLM async memory management (sleep) reduces memory usage."""
        try:
            os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
            ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

            rollout_config, model_config = self._build_rollout_config(free_cache_engine=True)

            server = self._create_server(
                rollout_config,
                model_config,
                name="trtllm_server_test_0",
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
