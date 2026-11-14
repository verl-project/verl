#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
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
#
# Runs verl unit tests on a TPU host, one pytest process per file: a TPU chip belongs to one process at a time, so
# tests that use it cannot share a process. Tests that start Ray get their own local Ray instance, not the RayCluster
# the host may belong to. run_tpu_e2e_ci.sh (test tpu_platform) runs it in the TPU container for TPU_UNIT_TESTS, and
# verify_tpu_e2e_log.py (test unit) checks its output.
#
# Usage, at the root of a verl checkout, with verl-hardware-plugin importable:
#   VERL_PLATFORM=tpu VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin \
#       bash tests/special_tpu/run_tpu_unit_tests.sh tests/trainer/ppo/test_rollout_corr.py ...

set -uo pipefail

failed=()
for test_file in "$@"; do
    echo "[TPU CI] pytest ${test_file}"
    rc=0
    env -u RAY_ADDRESS RAY_TMPDIR="$(mktemp -d)" python3 -m pytest -q -rA -p no:cacheprovider "${test_file}" || rc=$?
    echo "[TPU CI] pytest ${test_file}: exit code ${rc}"
    if [[ "${rc}" -ne 0 ]]; then
        failed+=("${test_file}")
    fi
done
if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "[TPU CI] unit tests failed on TPU: ${failed[*]}"
    exit 1
fi
