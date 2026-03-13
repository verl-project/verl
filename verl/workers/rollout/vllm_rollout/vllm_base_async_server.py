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
"""Base classes shared by vLLMHttpServer/vLLMReplica and vLLMOmniHttpServer/vLLMOmniReplica.

All logic that is identical (or near-identical) between the standard vLLM and
vLLM-Omni engines lives here.  Each engine subclass only overrides the parts
that genuinely differ.

Classes
-------
BaseVLLMHttpServer
    Template-method base for per-node HTTP servers.  Provides __init__,
    launch_server, and all utility/lifecycle helpers.

BaseVLLMReplica
    Base for multi-node replicas.  Provides launch_servers, sleep,
    abort_all_requests, and abort_request.
"""
import argparse
import asyncio
import json
import logging
import os
from pprint import pprint
from typing import Any, Callable, Optional

import ray
from ray.actor import ActorHandle

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils import FlexibleArgumentParser

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config, get_vllm_max_lora_rank

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# BaseVLLMHttpServer
# ---------------------------------------------------------------------------


class BaseVLLMHttpServer:
    """Template-method base for vLLM and vLLM-Omni per-node HTTP servers.

    Lifecycle
    ---------
    1. __init__      – common setup; calls _init_config / _init_model_config /
                       _validate_configs / _post_init hooks.
    2. launch_server – template method: builds CLI args, parses them, then calls
                       run_server (node_rank==0) or run_headless (workers).
    3. run_server / run_headless / generate  – abstract; implemented by subclasses.

    Override points (all have sensible defaults)
    ---------------------------------------------
    _init_config, _init_model_config, _validate_configs, _post_init
    _get_engine_kwargs_key, _preprocess_engine_kwargs
    _get_override_generation_config, _apply_quantization
    _get_worker_extension_cls, _update_extra_server_args
    _get_cli_modules, _get_cli_description, _postprocess_parsed_args
    _get_wake_up_tags, _sleep_hybrid
    clear_kv_cache, wait_for_requests_to_drain
    abort_all_requests, resume_generation
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices

        self.config = self._init_config(config)
        self.model_config = self._init_model_config(model_config)
        self._validate_configs()

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        self._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_config(self, config):
        return omega_conf_to_dataclass(config)

    def _init_model_config(self, model_config):
        """Initialise model_config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(model_config)

    def _validate_configs(self) -> None:
        """Validate config/model_config after initialisation. No-op by default."""
        pass

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Called at the end of __init__. Default logs server metadata."""
        logger.info(
            f"{self.__class__.__name__}, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    # -----------------------------------------------------------------------
    # Common concrete methods (identical between vLLM and vLLM-Omni)
    # -----------------------------------------------------------------------

    def get_master_address(self):
        """Return (master_address, master_port, dp_rpc_port) for data-parallel setup."""
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Return (server_address, server_port) of the running HTTP server."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    # -----------------------------------------------------------------------
    # launch_server – template method
    # -----------------------------------------------------------------------

    def _get_engine_kwargs_key(self) -> str:
        """Return the key under config.engine_kwargs for this engine (e.g. 'vllm')."""
        raise NotImplementedError

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Mutate engine_kwargs in-place before the CLI args dict is built. No-op by default."""
        pass

    def _get_override_generation_config(self) -> dict:
        """Return the override_generation_config dict. Default is empty (no overrides)."""
        return {}

    def _apply_quantization(self) -> tuple[Optional[str], dict]:
        """Process quantization config. Returns (quantization_str, hf_overrides).

        Default raises NotImplementedError when quantization is requested.
        Override in subclasses that support quantization (e.g. vLLMHttpServer).
        """
        quantization = self.config.quantization
        hf_overrides: dict = {}
        if quantization is not None:
            raise NotImplementedError(f"{self.__class__.__name__} does not support quantization yet.")
        return quantization, hf_overrides

    def _get_worker_extension_cls(self) -> str:
        """Return the fully-qualified colocate worker extension class name."""
        raise NotImplementedError

    def _update_extra_server_args(self, args: dict) -> None:
        """Inject engine-specific entries into the CLI args dict. No-op by default.

        Called immediately after the common args dict is built. Useful for adding
        skip_tokenizer_init, profiler args, MTP speculative-decoding args, etc.
        """
        pass

    def _get_cli_modules(self) -> list:
        """Return the list of CLI command modules used for argument parsing."""
        raise NotImplementedError

    def _get_cli_description(self) -> str:
        """Return the description string for the CLI argument parser."""
        raise NotImplementedError

    def _postprocess_parsed_args(self, server_args: argparse.Namespace) -> None:
        """Post-process parsed CLI args in-place. No-op by default."""
        pass

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. Build engine-specific CLI args
        engine_kwargs = self.config.get("engine_kwargs", {}).get(self._get_engine_kwargs_key(), {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        self._preprocess_engine_kwargs(engine_kwargs)

        # Override default generation config; subclasses populate temperature, top_k, etc.
        override_generation_config = self._get_override_generation_config()
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization, hf_overrides = self._apply_quantization()

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = engine_kwargs.get("decode_context_parallel_size", 1) or 1
        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"

        compilation_config = json.dumps(compilation_config)

        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.replica_rank + self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        # Inject engine-specific args (skip_tokenizer_init, profiler, mtp, etc.)
        self._update_extra_server_args(args)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = (
                self.model_config.lora_rank
            )  # FIXME: fallback to lora_rank for now, we should unify lora settings.

        if self.model_config.lora.get("merge", False):
            lora_rank = 0

        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if self.model_config.lora.get("fully_sharded_loras", False):
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        server_args = ["serve", self.model_config.local_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        # 2. Parse CLI args using engine-specific modules
        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        self._postprocess_parsed_args(server_args)

        # 3. Launch server
        if self.node_rank == 0:
            self._master_sock.close()
            self._dp_rpc_sock.close()
            self._dp_master_sock.close()
            await self.run_server(server_args)
        else:
            # TODO: avoid connect before master_sock close
            await asyncio.sleep(3)
            await self.run_headless(server_args)

    # -----------------------------------------------------------------------
    # Abstract: must be implemented by subclasses
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        """Initialise the engine client and start the HTTP server (master node)."""
        raise NotImplementedError

    async def run_headless(self, args: argparse.Namespace):
        """Run as a worker node without an HTTP server."""
        raise NotImplementedError

    async def generate(self, *args, **kwargs):
        """Execute generation for a single request."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # wake_up / sleep – template methods with hooks
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        """Return the tags passed to engine.wake_up(). Default includes kv_cache."""
        return ["kv_cache", "weights"]

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is woken up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.wake_up(tags=self._get_wake_up_tags())
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def _sleep_hybrid(self):
        """HYBRID mode sleep. Default: sleep(level=1). Override for lora-aware logic."""
        await self.engine.sleep(level=1)

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            await self._sleep_hybrid()
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    # -----------------------------------------------------------------------
    # Profiler, cache, and lifecycle utilities
    # -----------------------------------------------------------------------

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.engine.stop_profile()

    async def clear_kv_cache(self):
        """Reset KV/prefix cache. Override in subclasses with a live engine."""
        pass

    async def set_global_steps(self, global_steps: int):
        """Set the global step counter for the current model weights."""
        self.global_steps = global_steps

    async def wait_for_requests_to_drain(self):
        """Block until all in-flight requests finish. Override in subclasses."""
        pass

    # -----------------------------------------------------------------------
    # Request abort / resume
    # -----------------------------------------------------------------------

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        Uses the low-level output-processor path (common to all engine versions).
        vLLMHttpServer overrides this to use engine.pause_generation() on vLLM >= 0.12.

        Returns:
            dict[str, Any]: {aborted_count, request_ids}
        """
        try:
            # Atomic snapshot to avoid race conditions with the engine thread
            request_states_snapshot = list(self.engine.output_processor.request_states.items())
            request_ids = [req_id for req_id, _ in request_states_snapshot]

            if not request_ids:
                return {"aborted_count": 0, "request_ids": []}

            # Signal each request as aborted via its output queue
            from vllm.v1.engine import FinishReason

            for _, req_state in request_states_snapshot:
                request_output = req_state.make_request_output(
                    [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                )
                req_state.queue.put(request_output)

            self.engine.output_processor.abort_requests(request_ids)
            await self.engine.engine_core.abort_requests_async(request_ids)

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def resume_generation(self):
        """Resume generation after abort_all_requests. No-op by default."""
        if self.node_rank != 0:
            return

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: {aborted, request_id} or {aborted, error}
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}


# ---------------------------------------------------------------------------
# BaseVLLMReplica
# ---------------------------------------------------------------------------


class BaseVLLMReplica(RolloutReplica):
    """Base for vLLMReplica and vLLMOmniReplica.

    Provides the full launch_servers() implementation and common request
    management (sleep, abort_all_requests, abort_request).

    Subclasses only need to:
      1. Set self.server_class = ray.remote(YourHttpServer) in __init__.
      2. Implement _get_server_name_prefix() (returns 'vllm_' or 'vllm_omni_').
      3. Optionally override _validate_launch_requirements() for version checks.
    """

    def _validate_launch_requirements(self) -> None:
        """Validate requirements before launching. No-op by default."""
        pass

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix (e.g. 'vllm_' or 'vllm_omni_')."""
        raise NotImplementedError

    async def launch_servers(self):
        """Launch the HTTP server actor on each node of this replica."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        self._validate_launch_requirements()

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

        # Create one server actor per node, pinned via NodeAffinitySchedulingStrategy
        prefix = self._get_server_name_prefix()
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            name = (
                f"{prefix}server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"{prefix}server_reward_{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    }
                },
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # Broadcast master address and launch all nodes concurrently
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # Record the HTTP endpoint of the master server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Drain in-flight requests, then sleep all servers."""
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing requests across every server in this replica.

        Returns:
            dict[str, Any]: Aggregated {aborted_count, request_ids, server_results}.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids: list = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Fans out to all servers since routing is unknown.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result from whichever server owned the request.
        """
        # TODO(petersh6): abort only the server that owns the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}
