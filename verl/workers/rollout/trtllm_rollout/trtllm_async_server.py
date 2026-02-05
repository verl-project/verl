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
import logging
import os
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util import placement_group_table
from ray.util.placement_group import PlacementGroup

from verl.single_controller.ray import RayClassWithInitArgs, SubRayResourcePool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.trtllm_rollout.trtllm_rollout import ServerAdapter
from verl.workers.rollout.utils import get_max_position_embeddings, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote
class TRTLLMHttpServer:
    """TensorRT LLM HTTP server in single node.

    Args:
        config (DictConfig): full config.
        model_config (HFModelConfig): model config.
        is_reward_model (bool): whether this is a reward model.
        rollout_mode (RolloutMode): rollout mode.
        workers (list[ActorHandle]): list of rollout workers.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        max_colocate_count (int): max colocate count.
        pgs (list[PlacementGroup]): placement groups.
        bundle_indices (list[list[int]]): bundle indices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        is_reward_model: bool,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        max_colocate_count: int,
        pgs: list[PlacementGroup] = None,
        bundle_indices: list[list[int]] = None,
    ):
        os.environ["TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL"] = "1"
        assert torch.cuda.is_available(), "TRTLLM http server should run on GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.is_reward_model = is_reward_model
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.max_colocate_count = max_colocate_count
        self.pgs = pgs
        self.bundle_indices = bundle_indices

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        logger.info(f"TRTLLMHttpServer, replica_rank: {self.replica_rank}")

        self.sampling_args = {
            "detokenize": False,
            "end_id": -1,
            "pad_id": self.model_config.hf_config.pad_token_id,
            "stop_token_ids": [self.model_config.hf_config.eos_token_id],
            "include_stop_str_in_output": True,
        }

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self):
        from tensorrt_llm import AsyncLLM
        from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
        from tensorrt_llm.serve import OpenAIServer

        engine_kwargs = self.config.get("engine_kwargs", {}).get("trtllm", {}) or {}
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=True,
            free_gpu_memory_fraction=self.config.gpu_memory_utilization,
        )

        per_worker_gpu_share = 1.0 / self.max_colocate_count

        llm_kwargs = {
            "model": self.model_config.local_path,
            "backend": "pytorch",
            "orchestrator_type": "ray",
            "ray_worker_extension_cls": "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
            "kv_cache_config": kv_cache_config,
            "max_seq_len": self.config.max_model_len,
            "max_batch_size": self.config.max_num_seqs,
            "max_num_tokens": self.config.max_num_batched_tokens,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "trust_remote_code": self.model_config.trust_remote_code,
            "placement_groups": self.pgs,
            "placement_bundle_indices": self.bundle_indices,
            "per_worker_gpu_share": per_worker_gpu_share,
            "enable_sleep": True,
            "allreduce_strategy": "NCCL",
            "sampler_type": "TRTLLMSampler",
            **engine_kwargs,
        }

        if self.is_reward_model:
            llm_kwargs.update(
                {
                    "cuda_graph_config": None,
                    "disable_overlap_scheduler": True,
                }
            )
        else:
            llm_kwargs.update(
                {
                    "cuda_graph_config": CudaGraphConfig(
                        enable_padding=True,
                        batch_sizes=self.config.cudagraph_capture_sizes,
                        max_batch_size=0 if self.config.cudagraph_capture_sizes else self.config.max_num_seqs,
                    )
                }
            )

        self.llm = await AsyncLLM(**llm_kwargs)

        trtllm_server = OpenAIServer(
            llm=self.llm,
            model=self.model_config.local_path,
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
        )
        app = trtllm_server.app
        self._server_port, self._server_task = await run_unvicorn(app, None, self._server_address)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        assert image_data is None and video_data is None, "Multimodality is not yet supported in TRTLLMHttpServer."

        from tensorrt_llm.llmapi import SamplingParams

        max_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids))
        sampling_params["max_tokens"] = max_tokens
        sampling_params["logprobs"] = 1 if sampling_params.pop("logprobs", False) else None
        if sampling_params["top_k"] == -1:
            sampling_params["top_k"] = 0
        sampling_params.update(self.sampling_args)

        trt_llm_sampling_params = SamplingParams(**sampling_params)
        outputs = await self.llm.generate_async(
            inputs=prompt_ids,
            sampling_params=trt_llm_sampling_params,
        )

        token_ids = outputs.outputs[0].token_ids
        log_probs = None
        if trt_llm_sampling_params.logprobs is not None:
            log_probs = [list(d.values())[0].logprob for d in outputs.outputs[0].logprobs]
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        if self.rollout_mode == RolloutMode.COLOCATED:
            await self.llm.resume(tags=ServerAdapter.get_full_tags())
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            await self.llm.release(tags=ServerAdapter.get_full_tags())
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.llm.release(tags=ServerAdapter.get_full_tags())
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def report_device_ids(self) -> list[str]:
        """Report GPU device UUIDs from TRT-LLM workers."""
        return await self.llm.collective_rpc(
            "report_device_id",
            unique_reply_rank=0,
        )


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class TRTLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ) -> None:
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.node_ip = ray.util.get_node_ip_address().strip("[]")

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
            replica_rank=self.replica_rank,
        )
        return worker_dict_cls

    def rollout_worker_use_gpu(self) -> bool:
        return False

    def get_pgs_and_bundle_indices(self) -> tuple[list[PlacementGroup], list[list[int]]]:
        """Get placement groups and bundle indices for the replica."""

        start_pg_index = 0
        local_bundle_index = 0

        # For SubRayResourcePool, the replica is assigned sub pool specific for this replica.
        if isinstance(self.resource_pool, SubRayResourcePool):
            assert self.resource_pool.subgroup_world_size == self.world_size, (
                "Subgroup world size must be equal to world size"
            )
            local_bundle_index = self.resource_pool.start_bundle_index
        # For RayResourcePool, the replica is assigned to entire resource pool.
        # We need to find start pg index and local bundle index based on replica rank.
        else:
            local_bundle_index = self.world_size * self.replica_rank

        while local_bundle_index >= self.resource_pool.pgs[start_pg_index].bundle_count:
            start_pg_index += 1
            local_bundle_index -= self.resource_pool.pgs[start_pg_index].bundle_count
        assert (
            start_pg_index < len(self.resource_pool.pgs)
            and local_bundle_index < self.resource_pool.pgs[start_pg_index].bundle_count
        ), "Start pg index or local bundle index out of range"

        # Global Bundle View for Replica x 2 & TP=4:
        # ┌───────────────────┬───────────────────┐
        # │ Placement Group 0 │ Placement Group 1 │
        # ├────┬────┬────┬────┼────┬────┬────┬────┤
        # │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
        # └────┴────┴────┴────┴────┴────┴────┴────┘
        #   └───────────────┘   └───────────────┘
        #       Replica 0           Replica 1
        #       (4 GPUs)            (4 GPUs)

        left_bundle_count = self.world_size

        pgs = []
        bundle_indices = []

        for pg in self.resource_pool.pgs[start_pg_index:]:
            if left_bundle_count == 0:
                break

            left_bundle_count_in_pg = min(left_bundle_count, pg.bundle_count - local_bundle_index)
            pg_bundle_indices = [local_bundle_index + idx for idx in range(left_bundle_count_in_pg)]
            pgs.append(pg)
            bundle_indices.append(pg_bundle_indices)
            left_bundle_count -= left_bundle_count_in_pg
            local_bundle_index = 0

        assert left_bundle_count == 0, "all bundle indices should be assigned"

        return pgs, bundle_indices

    async def launch_servers(self):
        assert self.nnodes == 1, "TRTLLMReplica doesn't support multiple nodes for single replica yet."
        assert self.resource_pool.pgs is not None, "placement groups are not initialized"

        pgs, bundle_indices = self.get_pgs_and_bundle_indices()

        # Check server process should be launched on the same node as first bundle of first pg.
        first_pg_data = placement_group_table(pgs[0])
        node_id = first_pg_data["bundles_to_node_id"][bundle_indices[0][0]]
        print(f"TRTLLMReplica: {self.replica_rank}")
        print(f"pg node_id: {node_id}")
        print(f"pgs: {pgs}")
        print(f"bundle_indices: {bundle_indices}")

        # TRTLLMReplica is a 1:1 map from replica to TRTLLMHttpServer.
        name = (
            f"trtllm_server_{self.replica_rank}"
            if not self.is_reward_model
            else f"trtllm_server_reward_{self.replica_rank}"
        )

        server = TRTLLMHttpServer.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
            name=name,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            is_reward_model=self.is_reward_model,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
            max_colocate_count=self.resource_pool.max_colocate_count,
            pgs=pgs,
            bundle_indices=bundle_indices,
        )
        self.servers.append(server)

        # launch http server in each node
        await asyncio.gather(*[server.launch_server.remote() for server in self.servers])

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
