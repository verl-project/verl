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

"""E2E test: KVCAware router via run_infer.sh — full agent loop with routing.

Launches ``run_infer.sh`` (KVCAware router is hardcoded in parallel_infer.py),
waits for completion, then checks:
  1. Routing decisions produced ("routed to server")
  2. COMBINED scoring (not falling back to random)
  3. Mean RM Score printed (end-to-end completion)
  4. Trajectory logs produced (agent loop actually ran)

This is a GPU test (needs real vLLM + GPU + model + dataset).
"""

from __future__ import annotations

import os
import subprocess

import pytest
import yaml

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", ".."))
_RUN_INFER = os.path.join(_PROJECT_ROOT, "examples", "kvc_aware_router", "run_infer.sh")
_AGENT_CONFIG = os.path.join(_PROJECT_ROOT, "examples", "kvc_aware_router", "agent_config_simulated.yaml")
_MODEL = os.environ.get("VLLM_MODEL", "/data1/models/Qwen/Qwen3-4B-Instruct-2507")
_DATASET = os.environ.get("SWEBENCH_DATASET", "/data1/hgq/uni-agent/scripts/swe_bench_verified_modal.parquet")
_LOG_DIR = "/tmp/e2e_router_logs"


def _get_traj_dir() -> str:
    """Read log_dir from the agent config YAML."""
    with open(_AGENT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return cfg[0]["log_dir"]


def _run_infer(timeout: int = 600) -> str:
    """Run run_infer.sh with KVCAware router. Returns log content."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_file = os.path.join(_LOG_DIR, "router_e2e.log")

    # GPU config: CUDA_VISIBLE_DEVICES controls which GPUs Ray/vLLM see;
    # --n-gpus-per-node must match the count.
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    num_gpus = len(cuda_vis.split(","))
    cmd = [
        "bash",
        _RUN_INFER,
        _MODEL,
        _DATASET,
        _AGENT_CONFIG,
        "--num-workers",
        "1",
        "--n-gpus-per-node",
        str(num_gpus),
        "--tensor-parallel-size",
        "2",
        "--max-samples",
        "4",
        "--n",
        "2",
        "--max-model-len",
        "8192",
    ]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = cuda_vis

    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return open(log_file).read()


class TestKVCAwareRouterE2E:
    """E2E: run_infer.sh with KVCAware router — full agent loop."""

    def test_kvc_aware_router_full_e2e(self):
        """
        Feature: KVCAware router end-to-end via run_infer.sh
        Description: run full agent loop with --router-config-path, verify:
          - routing decisions ("routed to server" >= 1)
          - COMBINED scoring (not random fallback)
          - Mean RM Score printed (end-to-end completion)
          - trajectory logs produced (interaction_result.json exists)
        """
        log = _run_infer()

        # 1. Routing decisions
        assert "routed to server" in log, "No routing decisions in log"
        routing_count = log.count("routed to server")
        assert routing_count >= 1, f"Expected >=1 routing decision, got {routing_count}"

        # 2. COMBINED scoring (not random fallback)
        assert "COMBINED" in log, "No COMBINED scoring — strategy may have failed"

        # 3. End-to-end completion
        assert "Mean RM Score" in log, "run_infer.sh did not complete (no RM Score)"

        # 4. Trajectory logs produced (log_dir from agent config yaml)
        traj_dir = _get_traj_dir()
        traj_count = len(
            [d for d in os.listdir(traj_dir) if os.path.isfile(os.path.join(traj_dir, d, "interaction_result.json"))]
        )
        assert traj_count > 0, f"No trajectory logs in {traj_dir}"
