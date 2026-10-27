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
"""The Ray worker environment that run_deepseek_v4_veomni.sh passes through ray_kwargs.ray_init.runtime_env.

The recipe runs as is; a ``python`` shim on PATH prints the trainer's arguments instead of launching it.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[2] / "examples" / "grpo_trainer" / "run_deepseek_v4_veomni.sh"
PREFIX = "+ray_kwargs.ray_init.runtime_env.env_vars."


def _worker_env(tmp_path, hip=None, env=None, cpus=None, gfx_target_version=None):
    shim = tmp_path / "bin" / "python"
    shim.parent.mkdir(exist_ok=True)
    shim.write_text(
        "#!/bin/bash\n"
        'if [ "${1:-}" = -m ] && [ "${2:-}" = verl.trainer.main_ppo ]; then shift 2; printf "%s\\n" "$@"; exit 0; fi\n'
        f'exec {sys.executable} "$@"\n'
    )
    shim.chmod(0o755)
    if gfx_target_version is not None:
        # The recipe's awk reads this KFD topology (a CPU node, then a GPU node) instead of the host's.
        properties = tmp_path / "properties"
        properties.write_text(f"gfx_target_version 0\ngfx_target_version {gfx_target_version}\n")
        awk = shim.parent / "awk"
        awk.write_text(
            "#!/bin/bash\n"
            f'args=(); for a in "$@"; do [[ $a == /sys/class/kfd/* ]] && a={properties}; args+=("$a"); done\n'
            f'exec {shutil.which("awk")} "${{args[@]}}"\n'
        )
        awk.chmod(0o755)
    torch = tmp_path / "torch"
    torch.mkdir(exist_ok=True)
    (torch / "__init__.py").write_text(f"class version:\n    hip = {hip!r}\n")
    run_env = {
        **{k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "NCCL_DEBUG", "VLLM_ROCM_USE_AITER")},
        "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HDFS_ROOT": str(tmp_path),
        "DATA_ROOT": str(tmp_path),
        **(env or {}),
    }
    out = subprocess.run(
        ["bash", str(RECIPE)],
        env=run_env,
        capture_output=True,
        text=True,
        check=True,
        preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None,
    ).stdout
    pairs = (line[len(PREFIX) :].split("=", 1) for line in out.splitlines() if line.startswith(PREFIX))
    return {key: value.strip('"') for key, value in pairs}


def test_cuda_gets_only_omp_threads(tmp_path):
    assert list(_worker_env(tmp_path)) == ["OMP_NUM_THREADS"]


def test_rocm_defaults_and_explicit_values(tmp_path):
    env = _worker_env(tmp_path, hip="7.2")
    assert env["NCCL_DEBUG"] == "VERSION"

    # A value set in the launching shell is forwarded even where the GPU architecture is unknown.
    env = _worker_env(tmp_path, hip="7.2", env={"VLLM_ROCM_USE_AITER": "0", "OMP_NUM_THREADS": "3"})
    assert env["VLLM_ROCM_USE_AITER"] == "0" and env["OMP_NUM_THREADS"] == "3"


def test_rocm_enables_aiter_on_gfx94x_only(tmp_path):
    assert _worker_env(tmp_path, hip="7.2", gfx_target_version=90402)["VLLM_ROCM_USE_AITER"] == "1"  # gfx942
    assert "VLLM_ROCM_USE_AITER" not in _worker_env(tmp_path, hip="7.2", gfx_target_version=90500)  # gfx950


def test_omp_threads_follow_the_cpu_affinity(tmp_path):
    one_cpu = {min(os.sched_getaffinity(0))}  # a restricted runner may not include CPU 0
    assert _worker_env(tmp_path, cpus=one_cpu)["OMP_NUM_THREADS"] == "1"


def test_keys_of_the_job_runtime_env_are_left_to_the_job(tmp_path):
    # Ray rejects a key that both the submitted job's runtime env and ray.init's runtime env set.
    job = {"runtime_env": {"env_vars": {"OMP_NUM_THREADS": "4"}}}
    env = _worker_env(tmp_path, env={"RAY_JOB_CONFIG_JSON_ENV_VAR": json.dumps(job), "OMP_NUM_THREADS": "4"})
    assert "OMP_NUM_THREADS" not in env
