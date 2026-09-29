#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the verl project.
set -Eeuo pipefail
DOCTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCTEST_HELPER_PATH="${DOCTEST_DIR}/scripts/doctest_helper.py"
VERIFY_RUNTIME_PATH="${DOCTEST_DIR}/scripts/verify_runtime.py"
source "${DOCTEST_DIR}/scripts/common.sh"
REPO_ROOT="${DOCTEST_REPO_ROOT:-$(cd "${DOCTEST_DIR}/../../.." && pwd)}"
CASES=(fsdp2_vllm megatron_vllm fsdp2_sglang megatron_sglang)
function validate_case_selection() {
  local case_name="$1"
  local marker="quickstart-${case_name}"
  local block
  local script
  block="$("${DOCTEST_PYTHON}" "${DOCTEST_HELPER_PATH}" extract "${marker}")" || return $?
  bash -n <<<"${block}"
  script="$(grep -Eo 'tests/special_npu/quick_start/[A-Za-z0-9_.-]+\.sh' <<<"${block}" | head -n 1)"
  [[ -n "${script}" ]] || die "No quick_start script referenced by marker ${marker}"
  [[ -f "${REPO_ROOT}/${script}" ]] || die "Quick Start script not found: ${script}"
  bash -n "${REPO_ROOT}/${script}"
}
function cleanup_quickstart() {
  local exit_code=$?
  log_info "Quick Start doctest finished with exit code ${exit_code}."
  return "${exit_code}"
}
function run_quickstart() {
  local case_name="$1"
  trap cleanup_quickstart EXIT
  require_command "${DOCTEST_PYTHON}"
  require_command bash
  validate_case_selection "${case_name}"
  check_shell_block quickstart-environment
  check_shell_block quickstart-model
  check_shell_block quickstart-data
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log_info "DRY_RUN=1: validated selection ${case_name} without launching training."
    return 0
  fi
  require_command python3
  export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
  export NDEVICES_PER_NODE="${NDEVICES_PER_NODE:-8}"
  pushd "${REPO_ROOT}" >/dev/null
  run_shell_block quickstart-environment
  run_shell_block quickstart-model
  if [[ -n "${TRAIN_FILE:-}" || -n "${TEST_FILE:-}" ]]; then
    [[ -n "${TRAIN_FILE:-}" && -n "${TEST_FILE:-}" ]] || die "Set TRAIN_FILE and TEST_FILE together."
  else
    run_shell_block quickstart-data
    TRAIN_FILE="${GSM8K_OUTPUT_DIR:-$HOME/data/gsm8k}/train.parquet"
    TEST_FILE="${GSM8K_OUTPUT_DIR:-$HOME/data/gsm8k}/test.parquet"
  fi
  export TRAIN_FILE TEST_FILE
  python3 "${VERIFY_RUNTIME_PATH}" \
    --model-path "${MODEL_PATH:-}" \
    --train-file "${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}" \
    --test-file "${TEST_FILE:-$HOME/data/gsm8k/test.parquet}" \
    --min-devices "${NDEVICES_PER_NODE:-0}"
  run_shell_block "quickstart-${case_name}"
  popd >/dev/null
}
[[ $# -eq 1 && "${1:-}" =~ ^(fsdp2_vllm|megatron_vllm|fsdp2_sglang|megatron_sglang)$ ]] ||
  die "Usage: $0 {fsdp2_vllm|megatron_vllm|fsdp2_sglang|megatron_sglang}"
run_quickstart "$1"
