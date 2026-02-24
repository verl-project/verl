# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Utility functions for the Automodel engine integration."""

import torch
import torch.distributed

from verl.utils.device import get_device_id, get_torch_device


def get_dp_rank(device_mesh, include_cp=False):
    """Get data-parallel rank from device mesh."""
    if device_mesh is None:
        return 0
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh.get_local_rank("dp_cp")
    return device_mesh.get_local_rank("dp")


def get_tp_rank(device_mesh):
    """Get tensor-parallel rank from device mesh."""
    if device_mesh is None or "tp" not in device_mesh.mesh_dim_names or device_mesh["tp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("tp")


def get_pp_rank(device_mesh):
    """Get pipeline-parallel rank from device mesh."""
    if device_mesh is None or "pp" not in device_mesh.mesh_dim_names or device_mesh["pp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("pp")


def get_dp_group_size(device_mesh, include_cp=False):
    """Get data-parallel group size from device mesh."""
    if device_mesh is None:
        return torch.distributed.get_world_size()
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh["dp_cp"].size()
    if "dp" in device_mesh.mesh_dim_names:
        return device_mesh["dp"].size()
    return torch.distributed.get_world_size()


def maybe_fully_shard_optimizer(model, optimizer, model_wrapper):
    """Call fully_shard_optimizer for MegatronFSDP strategy."""
    from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager

    if isinstance(model_wrapper, MegatronFSDPManager) and torch.distributed.get_world_size() > 1:
        from megatron_fsdp.fully_shard import fully_shard_optimizer

        fully_shard_optimizer(model, optimizer)


def build_model_wrapper_from_engine_config(engine_config, world_size):
    """Construct an Automodel model wrapper

    Args:
        engine_config: AutomodelEngineConfig instance.
        world_size: Total number of processes in the job.

    Returns:
        A model wrapper instance with device meshes configured.
    """
    strategy = engine_config.distributed_strategy

    if strategy == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy
        from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

        wrapper = FSDP2Manager(
            tp_size=engine_config.tp_size,
            pp_size=engine_config.pp_size,
            cp_size=engine_config.cp_size,
            ep_size=engine_config.ep_size,
            activation_checkpointing=engine_config.activation_checkpointing,
            world_size=world_size,
            mp_policy=mp_policy,
        )

    elif strategy == "megatron_fsdp":
        from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager

        wrapper = MegatronFSDPManager(
            tp_size=engine_config.tp_size,
            cp_size=engine_config.cp_size,
            activation_checkpointing=engine_config.activation_checkpointing,
            world_size=world_size,
        )

    elif strategy == "ddp":
        from nemo_automodel.components.distributed.ddp import DDPManager

        wrapper = DDPManager(
            world_size=world_size,
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    else:
        raise ValueError(f"Unsupported distributed_strategy: {strategy}")

    return wrapper


def build_automodel_model(model_config, engine_config, model_wrapper):
    """Build a model using NeMoAutoModelForCausalLM.from_pretrained().

    Args:
        model_config: HFModelConfig with model path and settings.
        engine_config: AutomodelEngineConfig with distributed settings.
        model_wrapper: Model wrapper (FSDP2Manager/MegatronFSDPManager/DDPManager).

    Returns:
        A HuggingFace model with Automodel's distributed infrastructure applied.
    """
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM

    kwargs = {}

    if engine_config.enable_fp8:
        from nemo_automodel.components.fp8.config import FP8Config

        kwargs["fp8_config"] = FP8Config()

    if engine_config.enable_compile:
        from nemo_automodel.components.compile.config import CompileConfig

        kwargs["compile_config"] = CompileConfig()

    # Force use of HF model implementation for Qwen/Llama models.
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(
        model_config.path, trust_remote_code=model_config.trust_remote_code
    )
    _arch = (getattr(_cfg, "architectures", None) or [""])[0].lower()
    if "qwen" in _arch or "llama" in _arch:
        kwargs["force_hf"] = True

    # Pass TP/CP sizes so from_pretrained() can apply internal overrides.
    kwargs["tp_size"] = engine_config.tp_size
    kwargs["cp_size"] = engine_config.cp_size

    kwargs["attn_implementation"] = engine_config.attn_implementation

    from verl.utils.torch_dtypes import PrecisionType
    kwargs["torch_dtype"] = PrecisionType.to_dtype(engine_config.model_dtype)

    model = NeMoAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_config.path,
        model_wrapper=model_wrapper,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )

    return model


@torch.no_grad()
def offload_automodel_model_to_cpu(model, empty_cache=True):
    """Offload an FSDP2-wrapped model to CPU.

    Same pattern as VeOmni's offload_veomni_model_to_cpu.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_automodel_model_to_gpu(model):
    """Load model back to GPU."""
    device = get_device_id()
    model.to(device)


@torch.no_grad()
def offload_automodel_optimizer(optimizer):
    """Offload optimizer state to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_automodel_optimizer(optimizer, device_id):
    """Load optimizer state back to GPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
