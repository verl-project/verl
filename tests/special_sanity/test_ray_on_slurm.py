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

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_SCRIPT = REPO_ROOT / "examples" / "tutorial" / "slurm" / "ray_on_slurm.slurm"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _launcher_env(tmp_path: Path, hostnames: str, expected_nodes: int = 2) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.log"

    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
if [[ "$1" != "show" || "$2" != "hostnames" ]]; then
    echo "unexpected scontrol arguments: $*" >&2
    exit 2
fi
if [[ -n "${FAKE_SCONTROL_HOSTS:-}" ]]; then
    printf '%s\n' "$FAKE_SCONTROL_HOSTS"
fi
""",
    )
    _write_executable(
        fake_bin / "srun",
        """#!/bin/bash
printf '%s\n' "$*" >>"$FAKE_SRUN_LOG"
if [[ " $* " == *" hostname --ip-address "* ]]; then
    printf '10.0.0.1\n'
fi
""",
    )
    _write_executable(fake_bin / "sleep", "#!/bin/bash\nexit 0\n")

    env = os.environ.copy()
    env.update(
        {
            "FAKE_SCONTROL_HOSTS": hostnames,
            "FAKE_SRUN_LOG": str(srun_log),
            "LC_ALL": "C",
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "RAY_NETWORK_INTERFACE": "",
            "SLURM_CPUS_PER_TASK": "4",
            "SLURM_GPUS_PER_NODE": "1",
            "SLURM_JOB_NODELIST": "node-[a-b]",
            "SLURM_JOB_NUM_NODES": str(expected_nodes),
            "SLURM_NNODES": str(expected_nodes),
        }
    )
    return env, srun_log


def _run_launcher(tmp_path: Path, hostnames: str, expected_nodes: int = 2) -> tuple[subprocess.CompletedProcess, Path]:
    env, srun_log = _launcher_env(tmp_path, hostnames, expected_nodes)
    result = subprocess.run(
        ["bash", str(SLURM_SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result, srun_log


def test_launcher_passes_each_hostname_to_srun(tmp_path: Path):
    result, srun_log = _run_launcher(tmp_path, "node-a\nnode-b")

    assert result.returncode == 0, result.stderr
    commands = srun_log.read_text(encoding="utf-8").splitlines()
    head_command = next(command for command in commands if "ray start --head" in command)
    worker_command = next(command for command in commands if "ray start --address" in command)
    assert "-w node-a" in head_command
    assert "-w node-b" in worker_command
    assert "Starting HEAD at node-a" in result.stdout
    assert "Starting WORKER 1 at node-b" in result.stdout


def test_launcher_rejects_node_count_mismatch(tmp_path: Path):
    result, srun_log = _run_launcher(tmp_path, "node-a")

    assert result.returncode == 1
    assert "Expected 2 Slurm nodes, but scontrol returned 1" in result.stderr
    assert not srun_log.exists()


def test_launcher_rejects_empty_node_list(tmp_path: Path):
    result, srun_log = _run_launcher(tmp_path, "")

    assert result.returncode == 1
    assert "scontrol did not return any nodes" in result.stderr
    assert not srun_log.exists()
