# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.coverage.route_pr import select_workflows

ROUTER = Path(__file__).resolve().parents[2] / "scripts/ci/coverage/route_pr.py"
DIFF = Path(__file__).resolve().parent / "fixtures/ascend_pr.diff"

# requirements-npu.txt selects three workflows. The SFT workflow explicitly
# includes gsm8k_multiturn_sft.py after excluding examples/**.
PATH_WORKFLOWS = [
    "e2e_ascend.yml",
    "e2e_ppo_trainer_megatron_vllm_ascend.yml",
    "e2e_sft_llm_ascend.yml",
    "e2e_v1_separate_async_ascend.yml",
]


@pytest.mark.parametrize(
    ("paths", "recommended", "expected"),
    [
        pytest.param([], [], [], id="empty-diff"),
        pytest.param(
            ["verl/workers/engine_workers.py"],
            [],
            [
                "e2e_ascend.yml",
                "e2e_ppo_trainer_megatron_vllm_2_ascend.yml",
                "e2e_ppo_trainer_megatron_vllm_ascend.yml",
                "e2e_ppo_trainer_veomni_vllm_ascend.yml",
                "e2e_sft_llm_ascend.yml",
                "e2e_v1_separate_async_ascend.yml",
                "model_ascend.yml",
                "npu_unit_tests.yml",
                "reward_model_vllm_ascend.yml",
                "vllm_ascend.yml",
            ],
            id="engine-workers-only",
        ),
        pytest.param(
            ["requirements-npu.txt", "examples/data_preprocess/gsm8k_multiturn_sft.py"],
            ["e2e_sft_llm_ascend.yml", "unknown.yml", "model_ascend.yml"],
            [*PATH_WORKFLOWS, "model_ascend.yml"],
            id="merge-allowed-recommendations",
        ),
    ],
)
def test_select_workflows(paths, recommended, expected):
    assert select_workflows(paths, recommended).selected == expected


def test_route_pr_path_fallback(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "user.name=CI", "-c", "user.email=ci@example.org", "-c", "commit.gpgSign=false", *args],
            cwd=tmp_path,
            text=True,
        ).strip()

    # Exercise the CI entry point's path fallback when the coverage selector fails.
    # The router gets changed paths from `git diff BASE_SHA...HEAD_SHA`, so commit
    # the fixture patch on top of an empty base to give it both revisions.
    git("init", "--quiet")
    git("commit", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("apply", "--cached", str(DIFF))
    git("commit", "-m", "apply PR diff")

    output = tmp_path / "github_output"
    env = {
        **os.environ,
        "BASE_SHA": base,
        "HEAD_SHA": git("rev-parse", "HEAD"),
        "SELECTOR_OUTCOME": "failure",
        "COVERAGE_WORKFLOWS": "",
        "BASELINE_RUN": "",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }
    subprocess.run([sys.executable, str(ROUTER)], cwd=tmp_path, env=env, check=True)

    key, value = output.read_text(encoding="utf-8").strip().split("=", 1)
    assert key == "workflows"
    assert json.loads(value) == PATH_WORKFLOWS
