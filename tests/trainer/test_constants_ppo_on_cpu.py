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

import os
from unittest.mock import patch

from omegaconf import OmegaConf

from verl.trainer.constants_ppo import NVTX_INJECTION_ENV, get_ppo_ray_runtime_env

_INHERITED = "/usr/local/cuda/lib64/libcupti.so"


def _config(tool):
    return OmegaConf.create({"global_profiler": {"tool": tool}})


def test_torch_profiling_overrides_inherited_nvtx_injection():
    """An image-level NVTX injection into libcupti steals the CUPTI slot Kineto needs."""
    with patch.dict(os.environ, {NVTX_INJECTION_ENV: _INHERITED}, clear=True):
        env_vars = get_ppo_ray_runtime_env(_config("torch"))["env_vars"]

    # Overridden rather than dropped, and non-empty so a startup hook using setdefault
    # cannot put libcupti back.
    assert env_vars[NVTX_INJECTION_ENV] not in ("", _INHERITED)


def test_other_tools_leave_nvtx_injection_untouched():
    with patch.dict(os.environ, {NVTX_INJECTION_ENV: _INHERITED}, clear=True):
        env_vars = get_ppo_ray_runtime_env(_config("nsys"))["env_vars"]

    assert NVTX_INJECTION_ENV not in env_vars


def test_nvtx_injection_override_can_be_opted_out():
    env = {NVTX_INJECTION_ENV: _INHERITED, "VERL_KEEP_NVTX_INJECTION": "1"}
    with patch.dict(os.environ, env, clear=True):
        env_vars = get_ppo_ray_runtime_env(_config("torch"))["env_vars"]

    assert NVTX_INJECTION_ENV not in env_vars
