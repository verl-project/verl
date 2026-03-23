#!/usr/bin/env bash
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

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-"${HOME}/data/gsm8k/train.parquet"}"
VAL_FILE="${VAL_FILE:-"${HOME}/data/gsm8k/test.parquet"}"
EXP_NAME="${EXP_NAME:-qwen2.5-3b-sdpo}"

python3 -m verl.trainer.main_ppo \
  --config-name=sdpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  trainer.experiment_name="${EXP_NAME}"
