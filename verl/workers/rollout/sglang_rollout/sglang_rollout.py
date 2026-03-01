# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import AsyncIterator, Generator

import ray
import sglang.srt.entrypoints.engine
import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # Enable faulthandler in subprocesses
    os.environ["PYTHONFAULTHANDLER"] = "1"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        if config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        super().__init__(config, model_config, device_mesh)
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # device_mesh is needed to gather cuda ipc handle to update weights
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

        # Only init http server adapter in tp rank 0
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        engine_kwargs = (self.config.get("engine_kwargs", {}) or {}).get("sglang", {}) or {}
        adapter_kwargs = {}
        server_args_fields = {f.name for f in dataclasses.fields(ServerArgs)}
        if "api_key" in server_args_fields:
            adapter_kwargs["api_key"] = engine_kwargs.get("api_key", None)
        if "admin_api_key" in server_args_fields:
            adapter_kwargs["admin_api_key"] = engine_kwargs.get("admin_api_key", None)
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=host,
            port=server_port,
            launch_server=False,
            **adapter_kwargs,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        await self._init_server_adapter()

        update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
        if self.config.get("quantization", None) == "fp8":
            from verl.utils.sglang.sglang_fp8_utils import quant_weights_by_name

            logger.info("Convert bf16 weights to fp8 format before loading")
            weights = quant_weights_by_name(
                weights,
                self.model_config.hf_config.quantization_config,
                dtype=self.model_config.hf_config.dtype,
            )
            for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                await sgl_update_weights(
                    engine=self._engine,
                    params_batch=params_batch,
                    device_mesh_key="infer_tp",
                    device_mesh=self.device_mesh,
                )
        else:
            engine_kwargs = (self.config.get("engine_kwargs", {}) or {}).get("sglang", {}) or {}
            enable_lora = bool(engine_kwargs.get("enable_lora", False))
            if not enable_lora:
                async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                    await sgl_update_weights(
                        engine=self._engine,
                        params_batch=params_batch,
                        device_mesh_key="infer_tp",
                        device_mesh=self.device_mesh,
                    )
            else:
                lora_items: list[tuple[str, torch.Tensor]] = []

                async def _base_weight_iter() -> AsyncIterator[tuple[str, torch.Tensor]]:
                    async for name, t in ensure_async_iterator(weights):
                        if "lora_" in name:
                            lora_items.append((name, t))
                        else:
                            yield name, t

                async for params_batch in get_named_tensor_buckets(_base_weight_iter(), update_weights_bucket_bytes):
                    await sgl_update_weights(
                        engine=self._engine,
                        params_batch=params_batch,
                        device_mesh_key="infer_tp",
                        device_mesh=self.device_mesh,
                    )

                if lora_items:
                    lora_name = f"verl_policy_{self.replica_rank}_{self.node_rank}"
                    # Build a PEFT-compatible adapter config for SGLang.
                    # Note: SGLang needs explicit module names to infer LoRA hidden dims.
                    target_modules_val = engine_kwargs.get("lora_target_modules", None)
                    target_modules = None
                    if target_modules_val:
                        if isinstance(target_modules_val, str):
                            try:
                                target_modules = json.loads(target_modules_val)
                            except Exception:
                                s = target_modules_val.strip()
                                if s.startswith("[") and s.endswith("]"):
                                    items = [p.strip().strip("\"'") for p in s[1:-1].split(",")]
                                    items = [p for p in items if p]
                                    target_modules = items
                        else:
                            try:
                                target_modules = list(target_modules_val)
                            except Exception:
                                target_modules = None
                    if not (
                        isinstance(target_modules, list)
                        and target_modules
                        and all(isinstance(x, str) for x in target_modules)
                    ):
                        target_modules = [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ]
                    from peft import LoraConfig, TaskType  # type: ignore

                    peft_cfg = LoraConfig(
                        r=int(self.model_config.lora_rank),
                        lora_alpha=int(self.model_config.lora_alpha),
                        target_modules=target_modules,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=True,
                    ).to_dict()
                    peft_cfg["task_type"] = peft_cfg["task_type"].value if peft_cfg.get("task_type") else None
                    peft_cfg["peft_type"] = peft_cfg["peft_type"].value if peft_cfg.get("peft_type") else None
                    peft_cfg["target_modules"] = list(peft_cfg.get("target_modules") or target_modules)
                    (adapter_dir / "adapter_config.json").write_text(json.dumps(peft_cfg), encoding="utf-8")

                    from safetensors.torch import save_file  # type: ignore

                    lora_state = {k: v.detach().cpu() for k, v in lora_items}
                    required_bytes = sum(int(v.numel()) * int(v.element_size()) for v in lora_state.values())
                    safety_margin = 8 * 1024 * 1024  # 8 MiB
                    adapter_root = Path("/dev/shm")
                    try:
                        free_bytes = int(shutil.disk_usage(adapter_root).free)
                    except Exception:
                        free_bytes = 0
                    if free_bytes < required_bytes + safety_margin:
                        logger.warning(
                            f"Not enough space in {adapter_root} for LoRA adapter "
                            f"(need={required_bytes}B free={free_bytes}B); falling back to /tmp."
                        )
                        adapter_root = Path("/tmp")

                    adapter_dir = adapter_root / f"verl_sglang_lora_{self.replica_rank}_{self.node_rank}"
                    adapter_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                    try:
                        adapter_dir.chmod(0o700)
                    except OSError as e:
                        logger.warning(f"Failed to chmod LoRA adapter dir '{adapter_dir}' to 0700: {e}")
                    (adapter_dir / "adapter_config.json").write_text(json.dumps(peft_cfg), encoding="utf-8")
                    tmp_path = adapter_dir / "adapter_model.safetensors.tmp"
                    out_path = adapter_dir / "adapter_model.safetensors"
                    save_file(lora_state, str(tmp_path))
                    os.replace(tmp_path, out_path)

                    try:
                        await self._engine.unload_lora_adapter(lora_name)
                    except Exception as e:
                        logger.warning(f"Failed to unload LoRA adapter '{lora_name}', proceeding with load: {e}")
                    await self._engine.load_lora_adapter(lora_name=lora_name, lora_path=str(adapter_dir), pinned=False)

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
