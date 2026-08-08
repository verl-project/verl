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
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "examples/on_policy_distillation_trainer/run_qwen3_vl_2b_megatron.sh"


def test_qwen3_vl_2b_recipe_resolves_validated_four_npu_defaults(tmp_path):
    wrapper = tmp_path / RECIPE.name
    wrapper.write_text(RECIPE.read_text(encoding="utf-8"), encoding="utf-8")
    wrapper.chmod(0o755)

    base = tmp_path / "run_qwen3_vl_megatron.sh"
    base.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
for name in ACTOR_TP ROLLOUT_TP TEACHER_TP TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE \\
    TEST_FREQ ROLLOUT_GPU_MEMORY_UTILIZATION TEACHER_GPU_MEMORY_UTILIZATION \
    STUDENT_MODEL TEACHER_MODEL MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH ROLLOUT_SEED; do
    printf 'ENV:%s=%s\\n' "$name" "${!name}"
done
printf 'ARG:%s\\n' "$@"
""",
        encoding="utf-8",
    )
    base.chmod(0o755)

    result = subprocess.run(
        ["bash", str(wrapper), "trainer.test_freq=7"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )

    assert {
        "ENV:ACTOR_TP=1",
        "ENV:ROLLOUT_TP=1",
        "ENV:TEACHER_TP=1",
        "ENV:TRAIN_BATCH_SIZE=128",
        "ENV:PPO_MINI_BATCH_SIZE=128",
        "ENV:TEST_FREQ=5",
        "ENV:ROLLOUT_GPU_MEMORY_UTILIZATION=0.20",
        "ENV:TEACHER_GPU_MEMORY_UTILIZATION=0.20",
        "ENV:STUDENT_MODEL=Qwen/Qwen3-VL-2B-Instruct",
        "ENV:TEACHER_MODEL=Qwen/Qwen3-VL-4B-Instruct",
        "ENV:MAX_PROMPT_LENGTH=1024",
        "ENV:MAX_RESPONSE_LENGTH=2048",
        "ENV:ROLLOUT_SEED=42",
    }.issubset(result.stdout.splitlines())
    assert "ARG:actor_rollout_ref.rollout.max_num_seqs=128" in result.stdout
    assert "ARG:distillation.teacher_models.teacher_model.inference.max_num_seqs=1" in result.stdout
    assert "ARG:actor_rollout_ref.actor.use_dynamic_bsz=False" in result.stdout
    assert "ARG:actor_rollout_ref.rollout.n=1" in result.stdout
    assert "ARG:actor_rollout_ref.rollout.seed=42" in result.stdout
    assert "ARG:distillation.distillation_loss.topk=64" in result.stdout
    assert "ARG:distillation.distillation_loss.use_task_rewards=False" in result.stdout
    assert "ARG:distillation.distillation_loss.use_policy_gradient=True" in result.stdout
    assert "ARG:data.shuffle=False" in result.stdout
    assert "ARG:trainer.val_before_train=True" in result.stdout
    assert result.stdout.splitlines()[-1] == "ARG:trainer.test_freq=7"
