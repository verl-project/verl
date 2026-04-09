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
import asyncio
import os
import subprocess

import ray

from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica


class TestInterNodeTRTLLMRollout:
    def test_inter_node_trtllm_rollout(self):
        """Test TRT-LLM rollout with TP=8 spanning 2 simulated nodes (4 GPUs each).

        On CI with 8 GPUs on 1 node, we set gpus_per_node=4 so init_standalone()
        creates resource_pool_spec=[4, 4] (2 placement groups), exercising the
        inter-node code path in get_pgs_and_bundle_indices() and launch_servers().
        """
        from hydra import compose, initialize_config_dir

        prev_rank = os.environ.get("RANK")
        os.environ["RANK"] = "0"

        try:
            os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
            ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

            config_dir = os.path.abspath("verl/verl/trainer/config")
            if not os.path.exists(config_dir):
                config_dir = os.path.abspath("verl/trainer/config")

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                config = compose(config_name="ppo_trainer")

            # Inter-node settings: 8 GPUs split as 2 "nodes" x 4 GPUs
            config.trainer.n_gpus_per_node = 4
            config.trainer.nnodes = 2
            model_root = os.path.expanduser(os.getenv("TRTLLM_TEST_MODEL_PATH_ROOT", "~/models"))
            config.actor_rollout_ref.model.path = os.path.join(model_root, "Qwen/Qwen2.5-0.5B-Instruct")
            config.actor_rollout_ref.rollout.name = "trtllm"
            config.actor_rollout_ref.rollout.mode = "async"
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = 8  # spans 2 "nodes"

            replica = TRTLLMReplica(
                replica_rank=0,
                config=config.actor_rollout_ref.rollout,
                model_config=config.actor_rollout_ref.model,
                gpus_per_node=4,  # key: 4 not 8, so nnodes=2 inside replica
            )

            asyncio.run(replica.init_standalone())

            # Verify inter-node setup: 8 workers across 2 PGs
            assert len(replica.workers) == 8
            assert replica._server_address is not None

            # Verify leader rank
            worker0 = replica.workers[0]
            worker7 = replica.workers[7]
            replica_rank = ray.get(worker0.get_replica_rank.remote())
            is_leader_0 = ray.get(worker0.is_leader_rank.remote())
            is_leader_7 = ray.get(worker7.is_leader_rank.remote())

            assert replica_rank == 0
            assert is_leader_0 is True
            assert is_leader_7 is False

            # Verify token generation
            prompt_ids = [1, 2, 3, 4, 5]
            sampling_params = {"temperature": 1.0, "top_k": 0, "logprobs": 1}
            result = ray.get(replica._server_handle.generate.remote(prompt_ids, sampling_params, "test_inter_node_1"))

            print(f"Result: {result}")
            assert hasattr(result, "token_ids"), "Result should have token_ids attribute"
            assert hasattr(result, "log_probs"), "Result should have log_probs attribute"
            assert isinstance(result.token_ids, list), "token_ids should be a list"
            assert len(result.token_ids) > 0, "Generated tokens should not be empty"
            assert result.log_probs is not None, "log_probs should not be None when requested"
            assert len(result.log_probs) == len(result.token_ids), "log_probs length should match token_ids"

            print(f"Generated {len(result.token_ids)} tokens")
            print(f"Token IDs: {result.token_ids[:10]}...")
            print(f"Log probs: {result.log_probs[:10]}...")

        finally:
            if prev_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = prev_rank
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)
