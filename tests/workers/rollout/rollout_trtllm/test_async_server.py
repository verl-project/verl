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

from unittest.mock import MagicMock, patch

from verl.single_controller.ray import SubRayResourcePool
from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica


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
