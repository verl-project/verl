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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.utils.device import get_device_capability

_major, _ = get_device_capability()
# Opt-in GB200 NCCL WAR: set TLLM_DISABLE_NVLS_MNNVL=1 in the launch shell to disable
# both NCCL_NVLS_ENABLE and NCCL_MNNVL_ENABLE on Blackwell. Required by async-RL
# Megatron on GB200 nodes without IMEX (mbridge all_gather raises NCCL 801).
_gb200_nccl_env = {}
if (_major or 0) >= 10 and os.environ.get("TLLM_DISABLE_NVLS_MNNVL", "0") == "1":
    _gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        # PYTHONHASHSEED must be propagated via Ray runtime_env so that hash()
        # is deterministic across all Ray actors and subprocesses (e.g. NaiveRouter)
        # when full_determinism is enabled.  "0" means random hash seed (same as
        # the Python default), so non-deterministic runs are completely unaffected.
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0"),
        # VERL_FULL_DETERMINISM and VLLM_BATCH_INVARIANT are set in vLLMHttpServer
        # when rollout.full_determinism=True, but those os.environ changes only
        # affect the current actor process.  Forward them via runtime_env so all
        # Ray actors (including RM vLLM servers) receive them from the start.
        "VERL_FULL_DETERMINISM": os.environ.get("VERL_FULL_DETERMINISM", "0"),
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        **_gb200_nccl_env,
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # Ray workers do NOT inherit the driver's os.environ — only runtime_env
    # vars are propagated.  Remove keys that the driver already has set ONLY
    # when they are purely advisory (e.g. VLLM_LOGGING_LEVEL) so we avoid
    # double-setting.  But always forward PYTHONHASHSEED regardless, because
    # the driver setting it in os.environ does not reach Ray workers.
    for key in list(runtime_env["env_vars"].keys()):
        # These env vars MUST always be forwarded via runtime_env because
        # Ray workers do not inherit the driver's os.environ.  Even if the
        # driver has them set, removing them from runtime_env would cause
        # workers to miss them entirely.
        if key in ("PYTHONHASHSEED", "VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT"):
            continue
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
