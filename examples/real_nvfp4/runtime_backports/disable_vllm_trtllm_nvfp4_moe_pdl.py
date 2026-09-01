#!/usr/bin/env python3
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
"""Disable PDL only for vLLM's FlashInfer TRTLLM NVFP4 MoE calls.

This is a fail-closed diagnostic patch for the intermittent SM100 startup
kernel hang. It does not disable CUDA graphs or change the MoE backend.
"""

from importlib.metadata import distribution, version


def main() -> None:
    assert version("vllm") == "0.26.0", version("vllm")
    path = distribution("vllm").locate_file("vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py")
    text = path.read_text()

    anchor = """            per_token_scale=per_token_scale,
            output=output,
            tune_max_num_tokens=min("""
    replacement = """            per_token_scale=per_token_scale,
            output=output,
            enable_pdl=False,
            tune_max_num_tokens=min("""
    if replacement not in text:
        count = text.count(anchor)
        assert count == 1, f"modular call: expected one v0.26 block, found {count}"
        text = text.replace(anchor, replacement)
    else:
        assert anchor not in text

    anchor = """            activation_type=activation_to_flashinfer_int(activation),
            per_token_scale=per_token_scale,
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
        )[0]"""
    replacement = """            activation_type=activation_to_flashinfer_int(activation),
            per_token_scale=per_token_scale,
            enable_pdl=False,
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
        )[0]"""
    if replacement not in text:
        count = text.count(anchor)
        assert count == 1, f"monolithic call: expected one v0.26 block, found {count}"
        text = text.replace(anchor, replacement)
    else:
        assert anchor not in text

    path.write_text(text)
    verified = path.read_text()
    assert verified.count("enable_pdl=False,") == 2
    print(
        "REAL_NVFP4_VLLM_TRTLLM_MOE_PDL_OFF_PASS",
        path,
        flush=True,
    )


if __name__ == "__main__":
    main()
