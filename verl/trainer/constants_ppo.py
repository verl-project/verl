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
# it conditionally at runtime (see get_ppo_ray_runtime_env), not unconditionally.
#
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/skills/mcore-run-on-slurm/SKILL.md#cuda_device_max_connections
_is_hopper_or_ampere = (_major or 0) in (8, 9)


def _uses_megatron(config) -> bool:
    """Return True if any trainable engine in the config uses the Megatron strategy."""
    if config is None:
        return False
    from omegaconf import OmegaConf

    for key in ("actor_rollout_ref.actor.strategy", "critic.strategy"):
        if OmegaConf.select(config, key, default=None) == "megatron":
            return True
    return False


def _uses_mindspeed(config) -> bool:
    """Return True if any trainable engine in the config uses the mindspeed strategy."""
    if config is None:
        return False
    from omegaconf import OmegaConf

    for key in ("actor_rollout_ref.actor.strategy", "critic.strategy"):
        if OmegaConf.select(config, key, default=None) == "mindspeed":
            return True
    return False


PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
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
    },
}


def _apply_uv_run_runtime_env(runtime_env):
    """Start Ray workers in the driver's uv environment without shipping the CWD.

    When an ancestor process is `uv run`, ray.init() applies Ray's uv hook. The hook
    sets py_executable so workers run under the same `uv run` invocation as the driver
    (wanted), and it also defaults working_dir to os.getcwd() (not wanted: that uploads
    the whole repo and relocates every worker into a copy of it, where `uv run` would
    build a second venv instead of reusing .venv).

    verl used to opt out of the working_dir default by passing working_dir=None, but
    Ray >= 2.55 validates the value and raises "TypeError: path_or_uri must be a string"
    on None. Ray documents the alternative used here: flag off
    RAY_ENABLE_UV_RUN_RUNTIME_ENV and call the hook manually.
    """
    try:
        from ray._private import ray_constants
        from ray._private.runtime_env.uv_runtime_env_hook import _get_uv_run_cmdline, hook
    except ImportError:
        return runtime_env

    if not getattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False) or not _get_uv_run_cmdline():
        return runtime_env

    # A URI rather than a path, so the hook neither injects os.getcwd() nor checks that
    # pyproject.toml lives under a working_dir that is dropped right after.
    runtime_env = hook({**runtime_env, "working_dir": "verl://unset"})
    runtime_env.pop("working_dir", None)
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    return runtime_env


def get_ppo_ray_runtime_env(config=None):
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.

    Args:
        config: Optional training config. When the engine strategy is Megatron
            and the GPU is Hopper/Ampere, CUDA_DEVICE_MAX_CONNECTIONS=1 is set.
    """
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    # Only Megatron on Hopper/Ampere needs CUDA_DEVICE_MAX_CONNECTIONS=1.
    if _is_hopper_or_ampere and _uses_megatron(config):
        runtime_env["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if _uses_mindspeed(config):
        runtime_env["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # Always forward these at call-time, not import-time.
    for key in ("PYTHONHASHSEED", "VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "VERL_RL_INSIGHT_ENABLE"):
        runtime_env["env_vars"][key] = os.environ.get(key, "0")
    # Forward PYTHONPATH to Ray workers so packages exposed only via PYTHONPATH (e.g. the
    # Megatron-LM baked into the CI image at /workspace/Megatron-LM, which is not installed
    # into site-packages) stay importable. Workers do not inherit the driver's PYTHONPATH when
    # Ray is started out-of-band (e.g. `ray start --head`), so pass it through explicitly.
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        runtime_env["env_vars"]["PYTHONPATH"] = pythonpath
    return _apply_uv_run_runtime_env(runtime_env)
