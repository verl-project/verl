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
DOCTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
function usage() {
  echo "Usage:"
  echo "  $0 quickstart {fsdp2_vllm|megatron_vllm|fsdp2_sglang|megatron_sglang}"
  echo "  $0 installation {check|source}"
}
[[ $# -eq 2 ]] || { usage; exit 1; }
case "$1:$2" in
  quickstart:fsdp2_vllm|quickstart:megatron_vllm|quickstart:fsdp2_sglang|quickstart:megatron_sglang)
    worker=001-quickstart-test.sh
    ;;
  installation:check|installation:source)
    worker=002-installation-test.sh
    ;;
  *)
    usage
    exit 1
    ;;
esac
exec bash "${DOCTEST_DIR}/${worker}" "$2"
