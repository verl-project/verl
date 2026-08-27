# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

#!/usr/bin/env bash

set -e -x
torchrun --nproc-per-node=4 --standalone tests/special_distributed/test_tensor_dict.py
torchrun --nproc-per-node=4 --standalone tests/special_distributed/test_torch_functional.py
torchrun --nproc-per-node=4 --standalone -m pytest -svv \
    tests/special_distributed/test_megatron_dynamic_cp_features.py
# Regression for verl#5995 (FSDP2 + CPUOffloadPolicy state_dict crash). Only
# needs 2 ranks to exercise CPUOffloadPolicy sharding.
torchrun --nproc-per-node=2 --standalone tests/special_distributed/test_fsdp2_cpu_offload_state_dict.py
# Regression for colocated FSDP2 full-parameter model transfers. It locks in
# PyTorch's implicit pinned allocation for non-blocking D2H Module.to().
torchrun --nproc-per-node=2 --standalone tests/special_distributed/test_fsdp2_pinned_model_transfer.py
# Regression for FSDP1 param_offload under an unsharded strategy (fsdp_size=1).
# Only needs 2 ranks to build the one-rank shard group that degrades to NO_SHARD.
torchrun --nproc-per-node=2 --standalone tests/special_distributed/test_fsdp1_no_shard_offload.py
