from __future__ import annotations

import json
import logging
import os
from typing import Any

import ray
import sglang
import sglang.srt.entrypoints.engine
from packaging import version

from verl.experimental.decoupled_spec.config import DraftConfig, build_draft_model_config
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.net_utils import get_free_port
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.sglang_rollout import _set_envs_and_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DraftSGLangServer:
    """A minimal sidecar SGLang server for decoupled-spec drafting.

    Unlike rollout replicas, this actor is launched directly with Ray GPU resources
    and is intentionally not wired into checkpoint / rollout worker lifecycles.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        draft_config: DraftConfig,
        server_index: int,
    ):
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config, dataclass_type=RolloutConfig)
        self.draft_config = draft_config
        self.server_index = server_index
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port, _ = get_free_port(self._server_address)

        draft_model_cfg = build_draft_model_config(model_config, draft_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(draft_model_cfg, dataclass_type=HFModelConfig)
        self._engine = None
        self._launch_server()

    def _launch_server(self) -> None:
        engine_kwargs = dict(self.rollout_config.get("engine_kwargs", {}).get("sglang", {}) or {})
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.rollout_config.get("quantization", None)

        if quantization is not None and quantization != "fp8":
            raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")

        fp8_block_quant_kwargs = {}
        if quantization == "fp8":
            if version.parse(sglang.__version__) < version.parse("0.5.5"):
                raise ValueError("sglang>=0.5.5 is required for fp8 quantization")
            fp8_block_quant_kwargs = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }

        args = {
            "model_path": self.model_config.local_path,
            "host": self._server_address,
            "port": self._server_port,
            "dtype": self.rollout_config.dtype,
            "mem_fraction_static": self.rollout_config.gpu_memory_utilization,
            "disable_cuda_graph": self.rollout_config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.draft_config.tp_size,
            "dp_size": 1,
            "ep_size": 1,
            "node_rank": 0,
            "load_format": self.draft_config.load_format,
            "nnodes": 1,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.rollout_config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.rollout_config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        self._engine = AsyncHttpServerAdapter(
            launch_server=True,
            max_connections=self.rollout_config.server.max_connections,
            timeout=self.rollout_config.server.timeout,
            max_attempts=self.rollout_config.server.max_attempts,
            retry_delay=self.rollout_config.server.retry_delay,
            max_start_wait_time=self.rollout_config.server.max_start_wait_time,
            **args,
        )

    def get_server_address(self) -> tuple[str, int]:
        return self._server_address, self._server_port

    def get_server_handle(self) -> str:
        return f"{self._server_address}:{self._server_port}"

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "server_index": self.server_index,
            "node_id": ray.get_runtime_context().get_node_id(),
            "server_address": self.get_server_handle(),
        }
