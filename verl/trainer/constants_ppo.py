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

import os

import torch

from verl.trainer.runtime_env import process_runtime_env
from verl.utils.device import get_device_capability

_major, _ = get_device_capability()
# Opt-in GB200 NCCL WAR: set TLLM_DISABLE_NVLS_MNNVL=1 in the launch shell to disable
# both NCCL_NVLS_ENABLE and NCCL_MNNVL_ENABLE on Blackwell. Required by async-RL
# Megatron on GB200 nodes without IMEX (mbridge all_gather raises NCCL 801).
_gb200_nccl_env = {}
if (_major or 0) >= 10 and os.environ.get("TLLM_DISABLE_NVLS_MNNVL", "0") == "1":
    _gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}

# On ROCm, Ray 2.x force-clears accelerator visibility for num_gpus=0 actors
# (e.g. the SGLang server actor), leaving them unable to see any GPU. Disable
# that override so the actor keeps its HIP visibility. Scoped to ROCm to avoid
# changing Ray's default behavior on other platforms.
_rocm_ray_env = {}
if torch.version.hip is not None:
    _rocm_ray_env = {"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0"}

# CUDA_DEVICE_MAX_CONNECTIONS=1 is only needed by the Megatron engine on
# pre-Blackwell Hopper/Ampere GPUs (compute capability 8.x/9.x) to serialize
# kernel launches for TP/CP communication overlap. Blackwell/GB200 (>=10) does
# not need it, and Torch-FSDP2 / Megatron-FSDP must NOT set it to 1. So we set
# it conditionally at runtime (see get_ppo_runtime_env), not unconditionally.
#
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/skills/mcore-run-on-slurm/SKILL.md#cuda_device_max_connections
_is_hopper_or_ampere = (_major or 0) in (8, 9)


def _uses_strategy(config, actor_config, critic_config, strategy: str) -> bool:
    """Return whether either selected trainable model uses ``strategy``."""
    if actor_config is None and config is not None:
        actor_config = config.actor_rollout_ref.actor
    if critic_config is None and config is not None:
        critic_config = config.critic
    return any(section is not None and section.get("strategy") == strategy for section in (actor_config, critic_config))


PPO_RUNTIME_ENV = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
    # TODO: disable compile cache due to cache corruption issue
    # https://github.com/vllm-project/vllm/issues/31199
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    # Needed for multi-processes colocated on same NPU device
    # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    **_gb200_nccl_env,
    **_rocm_ray_env,
}


def get_ppo_runtime_env(config=None, *, actor_config=None, critic_config=None) -> dict[str, str]:
    """Build the backend-neutral Worker environment for PPO training.

    Controller overrides are copied explicitly so every backend creates the
    same Worker environment regardless of its launcher process ancestry.

    Args:
        config: Optional legacy training config.
        actor_config: Actor config selected by topology ``config_key``.
        critic_config: Critic config selected by topology ``config_key``.
    """
    env_vars = PPO_RUNTIME_ENV.copy()
    # Only Megatron on Hopper/Ampere needs CUDA_DEVICE_MAX_CONNECTIONS=1.
    if _is_hopper_or_ampere and _uses_strategy(config, actor_config, critic_config, "megatron"):
        env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if _uses_strategy(config, actor_config, critic_config, "mindspeed"):
        env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars = process_runtime_env(env_vars)
    # Always forward these at call-time, not import-time.
    for key in ("VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "VERL_RL_INSIGHT_ENABLE"):
        env_vars[key] = os.environ.get(key, "0")
    # Forward only when set: empty string breaks vLLM ParallelConfig int parsing.
    for key in (
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "FLASH_ATTENTION_DETERMINISTIC",
        "NCCL_DETERMINISTIC",
        "NCCL_ALGO",
    ):
        val = os.environ.get(key)
        if val is not None:
            env_vars[key] = val
    return env_vars


def get_ppo_ray_runtime_env(config=None, *, actor_config=None, critic_config=None) -> dict[str, object]:
    """Compatibility wrapper returning the historical Ray runtime_env mapping."""
    return {"env_vars": get_ppo_runtime_env(config, actor_config=actor_config, critic_config=critic_config)}


PPO_RAY_RUNTIME_ENV = {"env_vars": PPO_RUNTIME_ENV}
