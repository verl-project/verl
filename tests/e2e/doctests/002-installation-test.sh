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
REPO_ROOT="$(cd "${DOCTEST_DIR}/../../.." && pwd)"
WORK_DIR=""
OWN_WORK_DIR=0
function detect_os() {
  [[ -r /etc/os-release ]] || die "Cannot detect the operating system: /etc/os-release is missing."
  local os_id
  os_id="$(. /etc/os-release && echo "${ID,,}")"
  case "${os_id}" in
    ubuntu) echo ubuntu ;;
    openeuler) echo openeuler ;;
    *) die "Unsupported operating system '${os_id}'. Expected Ubuntu or openEuler." ;;
  esac
}
function run_installation_check() {
  log_info "Validating documentation markers and doctest config (read-only)..."
  "${DOCTEST_PYTHON}" "${DOCTEST_HELPER_PATH}" validate
  local marker
  for marker in installation-prerequisites-ubuntu installation-prerequisites-openeuler \
    installation-vllm-environment installation-sglang-environment installation-checkout \
    installation-vllm-install installation-sglang-install; do
    check_shell_block "${marker}"
  done
  log_info "Installation documentation check passed."
}
function run_installation_source() {
  local backend="${INSTALL_BACKEND:-vllm}"
  local os_name
  local case_name="${INSTALL_CASE:-fsdp2_${backend}}"
  local conda_base
  [[ "${ALLOW_INSTALL:-0}" == "1" ]] || die "Source installation requires ALLOW_INSTALL=1 in a disposable container."
  [[ "${backend}" == vllm || "${backend}" == sglang ]] || die "INSTALL_BACKEND must be vllm or sglang."
  case "${case_name}" in
    "fsdp2_${backend}") export USE_MEGATRON=0 ;;
    "megatron_${backend}") export USE_MEGATRON=1 ;;
    *) die "INSTALL_CASE must be fsdp2_${backend} or megatron_${backend}." ;;
  esac
  run_installation_check
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    bash "${DOCTEST_DIR}/001-quickstart-test.sh" "${case_name}"
    log_info "DRY_RUN=1: source installation plan validated; no packages installed."
    return 0
  fi
  require_command git
  require_command conda
  os_name="$(detect_os)"
  conda_base="$(conda info --base)"
  [[ -f "${conda_base}/etc/profile.d/conda.sh" ]] || die "Cannot find conda.sh under ${conda_base}."
  set +u
  source "${conda_base}/etc/profile.d/conda.sh"
  set -u
  if [[ -n "${INSTALL_WORKDIR:-}" ]]; then
    WORK_DIR="${INSTALL_WORKDIR}"
    mkdir -p "${WORK_DIR}"
    [[ -z "$(ls -A "${WORK_DIR}")" ]] || die "INSTALL_WORKDIR must be empty: ${WORK_DIR}"
  else
    WORK_DIR="$(mktemp -d)"
    OWN_WORK_DIR=1
  fi
  WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
  trap cleanup_installation EXIT
  export VERL_CONDA_PREFIX="${WORK_DIR}/conda-env"
  export VERL_REPOSITORY="${VERL_REPOSITORY:-${REPO_ROOT}}"
  if [[ "${VERL_REPOSITORY}" == "${REPO_ROOT}" ]]; then
    export VERL_REVISION="${VERL_REVISION:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
  fi
  run_shell_block "installation-prerequisites-${os_name}"
  pushd "${WORK_DIR}" >/dev/null
  run_shell_block "installation-${backend}-environment"
  run_shell_block installation-checkout
  run_shell_block "installation-${backend}-install"
  python3 "${VERIFY_RUNTIME_PATH}" --check-imports --backend "${backend}"
  DOCTEST_REPO_ROOT="${WORK_DIR}/verl" bash "${DOCTEST_DIR}/001-quickstart-test.sh" "${case_name}"
  set +u
  conda deactivate
  set -u
  popd >/dev/null
}
function cleanup_installation() {
  local exit_code=$?
  if [[ "${OWN_WORK_DIR}" == "1" && -n "${WORK_DIR}" && -d "${WORK_DIR}" && "${KEEP_INSTALL_WORKDIR:-0}" != "1" ]]; then
    rm -rf -- "${WORK_DIR}" || log_info "Could not remove installation work directory: ${WORK_DIR}"
  fi
  return "${exit_code}"
}
[[ $# -eq 1 && "${1:-}" =~ ^(check|source)$ ]] ||
  die "Usage: $0 {check|source}"
case "$1" in
  check) run_installation_check ;;
  source) run_installation_source ;;
esac
