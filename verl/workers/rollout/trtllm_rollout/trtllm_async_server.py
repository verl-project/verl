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
from dataclasses import dataclass
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group

from verl.plugin.platform import get_platform
from verl.runtime import ClassWithInitArgs, Worker
from verl.single_controller.ray.resource_pool import RayResourcePool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.net_utils import get_local_ip_address, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, qwen2_5_vl_dedup_image_tokens, run_uvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_TRTLLM_RAY_NODE_PIN = 1e-4


@dataclass(frozen=True)
class TRTLLMExecutorPlacement:
    """TRT-LLM-private native placement; never exposed by ResourcePool."""

    placement_groups: tuple[PlacementGroup, ...]
    bundle_indices: tuple[tuple[int, ...], ...]

    def close(self) -> None:
        for group in self.placement_groups:
            ray.util.remove_placement_group(group)


def _create_trtllm_executor_placement(pool: RayResourcePool) -> TRTLLMExecutorPlacement:
    """Create the native PG contract required by TRT-LLM's inner Ray executor."""
    node_addresses = {
        str(node["NodeID"]): str(node["NodeManagerAddress"]) for node in ray.nodes() if node.get("Alive", True)
    }
    resource_name = get_platform().ray_resource_name()
    groups: list[PlacementGroup] = []
    try:
        for node_index in range(pool.nnodes):
            node_id = pool.node_ids[node_index * pool.processes_per_node]
            try:
                node_address = node_addresses[node_id]
            except KeyError as exc:
                raise RuntimeError(f"TRT-LLM executor host {node_id!r} is not a live Ray node") from exc
            bundle = {
                "CPU": _TRTLLM_RAY_NODE_PIN,
                resource_name: 1,
                f"node:{node_address}": _TRTLLM_RAY_NODE_PIN,
            }
            group = placement_group(
                bundles=[bundle.copy() for _ in range(pool.processes_per_node)],
                strategy="STRICT_PACK",
            )
            groups.append(group)
        ray.get([group.ready() for group in groups])
    except BaseException:
        for group in groups:
            ray.util.remove_placement_group(group)
        raise
    return TRTLLMExecutorPlacement(
        placement_groups=tuple(groups),
        bundle_indices=tuple(tuple(range(pool.processes_per_node)) for _ in groups),
    )


def _resolve_chat_stop_tokens(model_config) -> tuple[int, list[int]]:
    """Return (end_id, stop_token_ids) for TorchSampler.

    Both TRTLLM's samplers stops only on end_id.  For chat-format prompts the model
    naturally ends each assistant turn with a chat-end token (e.g. <|im_end|>
    for Qwen, <|eot_id|> for Llama-3) that is *different* from the base-model
    eos_token_id.  If end_id is set to the base eos the sampler ignores the
    chat-end token and the model loops into a second turn, inflating response
    lengths until max_tokens is hit.

    For models without a distinct chat-end token the return values are
    identical to the current default (end_id = hf_config.eos_token_id).
    """
    eos_token_id = model_config.hf_config.eos_token_id
    all_stop_ids: list[int] = list(eos_token_id) if isinstance(eos_token_id, list) else [eos_token_id]

    if model_config.generation_config is not None:
        gen_eos = model_config.generation_config.eos_token_id
        if gen_eos is not None:
            for t in gen_eos if isinstance(gen_eos, list) else [gen_eos]:
                if t not in all_stop_ids:
                    all_stop_ids.append(t)

    chat_end_id = None
    if model_config.tokenizer is not None:
        _chat_stop_strings = ["<|im_end|>", "<|eot_id|>", "<|end_of_turn|>"]
        _added_vocab = model_config.tokenizer.get_added_vocab()
        for stop_str in _chat_stop_strings:
            if stop_str in _added_vocab:
                tid = _added_vocab[stop_str]
                if tid not in all_stop_ids:
                    all_stop_ids.append(tid)
                if chat_end_id is None:
                    chat_end_id = tid

    primary_end_id = chat_end_id if chat_end_id is not None else eos_token_id
    logger.warning(f"TRT-LLM stop token IDs: {all_stop_ids}, end_id: {primary_end_id}")
    return primary_end_id, all_stop_ids


class TRTLLMHttpServer(Worker):
    """TensorRT LLM HTTP server in single node.

    Args:
        config (DictConfig): full config.
        model_config (HFModelConfig): model config.
        is_reward_model (bool): whether this is a reward model.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        placement: Ray placement groups and bundle indices used by TRT-LLM's Ray executor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        is_reward_model: bool,
        rollout_mode: RolloutMode,
        replica_rank: int,
        placement: TRTLLMExecutorPlacement,
        env_vars: dict[str, str] | None = None,
    ):
        if "WORLD_SIZE" in os.environ:
            super().__init__()
        os.environ.update(env_vars or {})
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
        self.replica_rank = replica_rank
        self.placement = placement
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None
        # Set when generation is allowed; cleared during weight sync to block new requests.
        self._generation_allowed = asyncio.Event()
        self._generation_allowed.set()

        self.profiler_controller = self._init_profiler_controller()

        # Non-HYBRID with load_format=dummy normally needs to load from disk (auto).
        # Exception: FP8 has no on-disk ckpt; weights are filled during first sync (keep dummy).
        if (
            self.rollout_mode != RolloutMode.HYBRID
            and self.config.load_format == "dummy"
            and self.config.quantization != "fp8"
        ):
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        self.is_vlm_model = (
            self.model_config.hf_config is not None and hasattr(self.model_config.hf_config, "vision_config")
        ) or hasattr(self.model_config, "vision_config")

        # used for http server
        self._server_address = get_local_ip_address()
        self._server_port = None

        logger.info(f"TRTLLMHttpServer, replica_rank: {self.replica_rank}")

        _end_id, _stop_ids = _resolve_chat_stop_tokens(self.model_config)

        logger.info(f"TRT-LLM resolved end_id={_end_id}, stop_ids={_stop_ids}")

        self._use_torch_sampler = bool(int(os.environ.get("TLLM_USE_TORCHSAMPLER", "0")))

        if self._use_torch_sampler:
            self.sampling_args = {
                "detokenize": True,
                "end_id": _end_id,
                "stop_token_ids": _stop_ids,
                "pad_id": self.model_config.hf_config.pad_token_id,
                "include_stop_str_in_output": True,
            }
        else:
            self.sampling_args = {
                "detokenize": False,
                "end_id": -1,
                "pad_id": self.model_config.hf_config.pad_token_id,
                "stop_token_ids": _stop_ids,
                "include_stop_str_in_output": True,
            }
        logger.info(f"use_torch_sampler={self._use_torch_sampler}, sampling_args={self.sampling_args}")

    async def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self):
        from tensorrt_llm import AsyncLLM
        from tensorrt_llm.llmapi import CapacitySchedulerPolicy, CudaGraphConfig, KvCacheConfig, SchedulerConfig

        try:
            from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType, SleepConfig
        except ImportError:
            ExecutorMemoryType = None
            SleepConfig = None
        from tensorrt_llm.serve import OpenAIServer

        assert self.config.pipeline_model_parallel_size == 1, "pipeline_model_parallel_size > 1 is not supported yet"

        engine_kwargs = self.config.get("engine_kwargs", {}).get("trtllm", {}) or {}
        # Pop kv_cache_config from engine_kwargs to merge into KvCacheConfig constructor,
        # otherwise **engine_kwargs unpacking in llm_kwargs would overwrite the entire
        # KvCacheConfig object, losing free_gpu_memory_fraction and enable_block_reuse.
        kv_cache_overrides = engine_kwargs.pop("kv_cache_config", {})
        kv_cache_kwargs = {
            "enable_block_reuse": self.config.enable_prefix_caching,
            "free_gpu_memory_fraction": self.config.gpu_memory_utilization,
            **kv_cache_overrides,
        }
        kv_cache_config = KvCacheConfig(**kv_cache_kwargs)

        per_worker_gpu_share = _TRTLLM_RAY_NODE_PIN

        quantization = self.config.quantization
        if quantization is not None:
            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                engine_kwargs["model_kwargs"] = {"quantization_config": FP8_BLOCK_QUANT_KWARGS}
                if self.config.load_format != "dummy":
                    raise ValueError("FP8 quantization is only supported for dummy load format")
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")

        llm_kwargs = {
            "model": self.model_config.local_path,
            "backend": "pytorch",
            "dtype": self.config.dtype,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "orchestrator_type": "ray",
            "kv_cache_config": kv_cache_config,
            "max_seq_len": self.config.max_model_len,
            "max_batch_size": self.config.max_num_seqs,
            "max_num_tokens": self.config.max_num_batched_tokens,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_model_parallel_size,
            "moe_expert_parallel_size": self.config.expert_parallel_size,
            "moe_tensor_parallel_size": self.config.moe_tensor_parallel_size,
            "load_format": self.config.load_format,
            "trust_remote_code": self.model_config.trust_remote_code,
            "placement_groups": list(self.placement.placement_groups),
            "placement_bundle_indices": [list(indices) for indices in self.placement.bundle_indices],
            "per_worker_gpu_share": per_worker_gpu_share,
            "sleep_config": SleepConfig(
                restore_modes={
                    ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "NONE",
                    ExecutorMemoryType.KV_CACHE: "NONE",
                }
            )
            if self.config.enable_sleep_mode and SleepConfig is not None
            else None,
            "allreduce_strategy": "NCCL",
            "sampler_type": "TorchSampler" if self._use_torch_sampler else "TRTLLMSampler",
            **engine_kwargs,
        }

        self_defined_extension = {
            "ray_worker_extension_cls": "verl.workers.rollout.trtllm_rollout.trtllm_worker_extension.WorkerExtension",
        }
        if self.is_vlm_model:
            llm_kwargs.update(self_defined_extension)
        else:
            # TODO: once TRT-LLM WorkerExtension includes wait_for_engine_idle,
            # replace with "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension" directly.
            llm_kwargs.update(
                {
                    "ray_worker_extension_cls": (
                        "verl.workers.rollout.trtllm_rollout.trtllm_worker_extension.RlhfWorkerExtension"
                    ),
                }
            )

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
                    ),
                    "scheduler_config": SchedulerConfig(
                        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
                    ),
                }
            )

        self.llm = await AsyncLLM(**llm_kwargs)
        import inspect

        init_params = inspect.signature(OpenAIServer.__init__).parameters
        if "generator" in init_params:
            trtllm_server = OpenAIServer(
                generator=self.llm,
                model=self.model_config.local_path,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
        else:
            trtllm_server = OpenAIServer(
                llm=self.llm,
                model=self.model_config.local_path,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )

        app = trtllm_server.app
        self._server_port, self._server_task = await run_uvicorn(app, None, self._server_address)

    async def generate(
        self,
        prompt_ids: str | list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        from tensorrt_llm.llmapi import SamplingParams

        max_tokens = min(
            self.config.response_length,
            self.config.prompt_length + self.config.response_length - len(prompt_ids),
        )
        max_tokens = max(0, min(max_tokens, self.config.max_model_len - len(prompt_ids)))
        sampling_params["max_tokens"] = max_tokens
        # TorchSampler: logprobs=0 means sampled-token logprob; TRTLLMSampler: logprobs=1
        _want_logprobs = sampling_params.pop("logprobs", False)
        if self._use_torch_sampler:
            sampling_params["logprobs"] = 0 if _want_logprobs else None
        else:
            sampling_params["logprobs"] = 1 if _want_logprobs else None
        if sampling_params["top_k"] == -1:
            sampling_params["top_k"] = 0
        sampling_params.update(self.sampling_args)

        trt_llm_sampling_params = SamplingParams(**sampling_params)
        if audio_data is not None:
            raise NotImplementedError("TRT-LLM rollout does not support audio inputs yet.")

        await self._generation_allowed.wait()
        if self.is_vlm_model and (image_data or video_data):
            deduped_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
            org_prompt = self.llm.tokenizer.decode(deduped_ids)
            input_dict = {
                "prompt": org_prompt,
                "multi_modal_data": {},
                "mm_processor_kwargs": dict(mm_processor_kwargs or {}),
            }
            if image_data:
                input_dict["multi_modal_data"]["image"] = image_data
            if video_data:
                input_dict["multi_modal_data"]["video"] = video_data

            outputs = await self.llm.generate_async(
                inputs=input_dict,
                sampling_params=trt_llm_sampling_params,
            )
        else:
            outputs = await self.llm.generate_async(
                inputs=prompt_ids,
                sampling_params=trt_llm_sampling_params,
            )
        token_ids = outputs.outputs[0].token_ids
        log_probs = None
        if outputs.outputs[0].logprobs is not None:
            # When logprobs=1, TRT-LLM returns only the sampled token's logprob at each position.
            # Extract log_probs before checking finish_reason so cancelled (partial) requests also
            # return log_probs for their already-generated tokens.
            log_probs = [list(d.values())[0].logprob for d in outputs.outputs[0].logprobs]
        if outputs.outputs[0].finish_reason == "cancelled":
            return TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason="aborted",
                extra_fields={"global_steps": self.global_steps},
            )
        return TokenOutput(token_ids=token_ids, log_probs=log_probs, extra_fields={"global_steps": self.global_steps})

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def supports_partial_loading(self) -> bool:
        results = await self.llm.collective_rpc("supports_partial_loading")
        return all(results) if isinstance(results, list) else bool(results)

    async def abort_all_requests(self):
        """Abort all in-flight requests and block new ones. Call resume_generation() to unblock."""
        self._generation_allowed.clear()
        await self.llm.pause_generation()
        # TODO: remove once TRT-LLM is upgraded to a version where pause_generation()
        # drains internally (https://github.com/NVIDIA/TensorRT-LLM/pull/13784).
        await self.llm.collective_rpc("wait_for_engine_idle")
        if self.config.enable_prefix_caching:
            await self.llm.collective_rpc("reset_prefix_cache")

    async def resume_generation(self):
        """Unblock new generation requests after abort_all_requests()."""
        await self.llm.resume_generation()
        self._generation_allowed.set()

    async def clear_kv_cache(self):
        """Invalidate prefix cache entries after weight update."""
        await self.llm.collective_rpc("reset_prefix_cache")

    async def release_kv_cache(self):
        """Release only kv_cache GPU memory, keeping model weights intact.

        This is used during weight sync to free GPU memory for new weights.
        """
        if not self.config.free_cache_engine:
            return
        await self.llm.release(tags=["kv_cache"])

    async def resume_kv_cache(self):
        """Restore kv_cache GPU memory after a weight sync. Counterpart to release_kv_cache()."""
        await self.llm.resume(tags=["kv_cache"])

    async def wake_up(self):
        from verl.workers.rollout.trtllm_rollout.trtllm_rollout import ServerAdapter

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        if self.rollout_mode == RolloutMode.COLOCATED:
            await self.llm.resume(tags=ServerAdapter.get_full_tags())
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        from verl.workers.rollout.trtllm_rollout.trtllm_rollout import ServerAdapter

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

    async def start_profile(self, **kwargs):
        if self.profiler_controller.check_enable() and self.profiler_controller.check_this_rank():
            await self.llm.collective_rpc("start_profile")

    async def stop_profile(self):
        if self.profiler_controller.check_enable() and self.profiler_controller.check_this_rank():
            await self.llm.collective_rpc("stop_profile")

    def _init_profiler_controller(self) -> DistProfiler:
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            elif profiler_config.tool == "nsys":
                # nsys config lives in global_tool_config, not tool_config
                from verl.utils.profiler.config import NsightToolConfig

                raw = (profiler_config.global_tool_config or {}).get("nsys")
                tool_config = omega_conf_to_dataclass(raw) if raw is not None else NsightToolConfig()
            elif profiler_config.tool is not None:
                logger.warning(f"trtllm rollout: unsupported profiler tool '{profiler_config.tool}', disabling")
                profiler_config = None
        return DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)


class TRTLLMReplica(RolloutReplica):
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
        if is_teacher_model:
            raise NotImplementedError("TRTLLMReplica doesn't support teacher model yet.")
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.node_ip = get_local_ip_address()
        self._executor_placement: TRTLLMExecutorPlacement | None = None

    def close(self) -> None:
        try:
            super().close()
        finally:
            placement, self._executor_placement = self._executor_placement, None
            if placement is not None:
                placement.close()

    def rollout_worker_use_gpu(self) -> bool:
        return False

    def _create_executor_placement(self) -> TRTLLMExecutorPlacement:
        """Create TRT-LLM's engine-private native Ray placement."""
        if not isinstance(self.resource_pool, RayResourcePool):
            raise NotImplementedError("TRT-LLM's Ray executor requires the Ray Runtime backend")
        if self._executor_placement is None:
            self._executor_placement = _create_trtllm_executor_placement(self.resource_pool)
        return self._executor_placement

    async def launch_servers(self):
        if self.resource_pool is None:
            raise RuntimeError("rollout worker placement is not initialized")
        executor_placement = self._create_executor_placement()
        print(f"TRTLLMReplica: {self.replica_rank}")

        _server_env_vars = {var: "1" for var in get_platform().ray_noset_envvars()}
        _server_env_vars.update(get_platform().rollout_env_vars())
        # Propagate profiling env vars to the Ray actor so that RayExecutor
        # (instantiated inside TRTLLMHttpServer) picks them up for inner workers.
        for _prof_var in (
            "TLLM_ENABLE_NSYS",
            "TLLM_NSYS_OUTPUT_DIR",
            "TLLM_USE_TORCHSAMPLER",
        ):
            if _val := os.environ.get(_prof_var):
                _server_env_vars[_prof_var] = _val
        group = await self._create_server_worker_group(
            ClassWithInitArgs(
                TRTLLMHttpServer,
                config=self.config,
                model_config=self.model_config,
                is_reward_model=self.is_reward_model,
                rollout_mode=self.rollout_mode,
                replica_rank=self.replica_rank,
                placement=executor_placement,
                env_vars=_server_env_vars,
            ),
            source_pool=self.resource_pool.slice(0),
        )
        server = group.remote()
        self.servers.append(server)

        # launch http server in each node
        await asyncio.gather(*[server.submit("launch_server") for server in self.servers])

        # get http server address from first server
        server_address, server_port = await self.servers[0].submit("get_server_address")
        self._server_handle = self.servers[0]
        await self._set_server_endpoints([self._server_handle] * self.world_size)
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
