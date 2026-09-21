# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Verify installed bytes and the source-validated refit lifecycle contract."""

import hashlib
import json
from importlib import import_module
from importlib.metadata import distribution, version
from pathlib import Path

from apply_vllm_nvfp4 import AFTER as VLLM_NVFP4_PATCH_SHA256
from patch_megatron_fa4 import verify_patched_source


def main():
    attention_source = distribution("megatron-core").locate_file("megatron/core/transformer/attention.py")
    verify_patched_source(Path(attention_source).read_text())
    expected_versions = {
        "transformer-engine": "2.18.0",
        "transformer-engine-cu13": "2.18.0",
        "transformer-engine-torch": "2.18.0",
        "vllm": "0.29.0",
        "megatron-core": "0.19.0",
    }
    actual_versions = {name: version(name) for name in expected_versions}
    if actual_versions != expected_versions:
        raise RuntimeError(f"Unexpected dependency versions: {actual_versions}")
    # PyPI and the official cu130 index use these two audited distribution
    # identities for the same Torch release. Do not normalize other packages.
    actual_versions["torch"] = version("torch")
    if actual_versions["torch"] not in {"2.13.0", "2.13.0+cu130"}:
        raise RuntimeError(f"Unexpected Torch distribution: {actual_versions['torch']}")
    torch = import_module("torch")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Unexpected Torch CUDA build: {torch.version.cuda}")
    native_extensions = {
        name: str(import_module(name).__file__)
        for name in ("flash_attn_2_cuda", "fused_weight_gradient_mlp_cuda", "amp_C")
    }
    nvfp4_source = distribution("vllm").locate_file("vllm/model_executor/layers/quantization/online/nvfp4.py")
    if hashlib.sha256(Path(nvfp4_source).read_bytes()).hexdigest() != VLLM_NVFP4_PATCH_SHA256:
        raise RuntimeError("Apply the vLLM NVFP4 compatibility patch before using this runtime")
    print(
        "DELIVERY_RUNTIME_GUARD_PASS",
        json.dumps(
            {
                "versions": actual_versions,
                "torch_cuda": torch.version.cuda,
                "native_extensions": native_extensions,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
