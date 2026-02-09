# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
This CI test is used for checking whether device api usage is irregular, suggest using api in `verl/utils/device.py`.
Search targets include .py files in verl/recipe and verl/verl.
Some files that must contain ".cuda", "cuda" or "nccl" keyword is pre-defined in whitelist below.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

# directory or file path must contain keyword ".cuda" or "cuda"
CUDA_KEYWORD_CHECK_WHITELIST = [
    "verl/utils/device.py",
    "verl/utils/torch_functional.py",  # import flash_attn only on cuda
    "verl/utils/profiler/nvtx_profile.py",  # appear in NsightSystemsProfiler
    "verl/utils/profiler/torch_profile.py",  # appear in TorchProfiler
    "verl/utils/profiler/config.py",  # appear in TorchProfilerToolConfig
    "verl/utils/kernel/linear_cross_entropy.py",  # appear in nvidia nvtx
    "verl/utils/rendezvous/ray_backend.py",  # appear in cupy importance
    "verl/single_controller/ray/base.py",  # appear in default device_name
    "verl/trainer/ppo/ray_trainer.py",  # appear in default device_name
    "verl/experimental/transfer_queue/ray_trainer.py",  # appear in docstring as default device_name
    "verl/experimental/one_step_off_policy/ray_trainer.py",  # appear in docstring as default device_name
    "verl/utils/reward_score/sandbox_fusion/utils.py",  # appear in sandbox language type
    "verl/workers/reward_model/megatron/reward_model.py",  # appear in default device_name
    "verl/third_party/torch/distributed/_state_dict_utils.py",  # torch monkey patch fixes
    "verl/third_party/torch/distributed/checkpoint/state_dict.py",  # torch monkey patch fixes
    "verl/workers/engine/base.py",  # appear in default device_name
    "verl/workers/engine/utils.py",  # appear in enable_full_determinism
    "verl/workers/engine/fsdp/transformer_impl.py",  # appear in default device_name
    "verl/workers/engine/veomni/transformer_impl.py",  # appear in default device_name
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py",  # appear in config.cudagraph_capture_sizes
    "verl/workers/rollout/sglang_rollout/async_sglang_server.py",  # manually set CUDA_VISIBLE_DEVICES
    "verl/workers/rollout/trtllm_rollout/trtllm_async_server.py",  # appear in config.cudagraph_capture_sizes
    "verl/workers/rollout/replica.py",  # appear in default device_name
    "verl/checkpoint_engine",  # checkpoint engine backend are device specific
]

# directory or file path must contain keyword "nccl"
NCCL_KEYWORD_CHECK_WHITELIST = [
    "verl/utils/device.py",
    "verl/third_party/sglang/parallel_state.py",  # appear in default backend
    "verl/recipe/fully_async_policy/param_sync.py",  # fully_async_policy in default backend
]

SEARCH_WHITELIST = CUDA_KEYWORD_CHECK_WHITELIST + NCCL_KEYWORD_CHECK_WHITELIST

SEARCH_KEYWORDS = [".cuda", '"cuda"', '"nccl"']


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:
        path_in_str = str(path.absolute())

        # judge whether current path is in pre-defined search whitelist or not.
        path_in_whitelist = False

        for sw in SEARCH_WHITELIST:
            # for easy debugging in non-linux system
            sw = sw.replace("/", os.sep)
            if sw in path_in_str:
                print(f"[SKIP] File {path_in_str} is in device api usage check whitelist, checking is skipped.")
                path_in_whitelist = True
                break

        if path_in_whitelist:
            continue

        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            find_invalid_device_management = False

            for sk in SEARCH_KEYWORDS:
                if sk in file_content:
                    find_invalid_device_management = True
                    break

            print(
                f"[CHECK] File {path_in_str} is detected for device api usage check, check result: "
                f"{'success' if not find_invalid_device_management else f'failed, because detect {sk}'}."
            )

            assert not find_invalid_device_management, (
                f'file {path_in_str} contains .cuda/"cuda"/"nccl" usage, please use api in '
                f"verl/utils/device.py directly."
            )
