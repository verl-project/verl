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

import pytest
import torch
from compressed_tensors.compressors import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from verl.utils.qat.quantizer import QATQuantizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_qat_quantizer_stream_round_trip(dtype, mode, target):
    torch.manual_seed(0)
    params = {}
    for layer in range(2):
        for i, projection in enumerate(["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj"]):
            name = f"model.layers.{layer}.{projection}"
            weight = torch.randn(16, 32, dtype=dtype, device="cuda") * (i + 1)
            weight[0, :16] = 0
            params[f"{name}.weight"] = weight
            if mode == "w4a4":
                params[f"{name}.input_global_scale"] = torch.tensor([2.0], device="cuda")
                params[f"{name}.input_amax"] = torch.tensor([1344.0], device="cuda")
    params["lm_head.weight"] = torch.randn(16, 32, dtype=dtype, device="cuda")
    original = {key: value.clone() for key, value in params.items()}
    quantizer = QATQuantizer(mode=mode, device=torch.device("cuda"), param_dtype=dtype)
    output = dict(quantizer.quantize_with_fusion(iter(params.items()), target_device=torch.device(target)))
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="float", symmetric=True, strategy="tensor_group", group_size=16),
    )
    expected_keys = {"lm_head.weight"}
    for key, weight in params.items():
        torch.testing.assert_close(weight, original[key], rtol=0, atol=0)
        if not key.endswith(".weight") or key == "lm_head.weight":
            continue
        name = key.removesuffix(".weight")
        state = {
            suffix: output[f"{name}.{suffix}"] for suffix in ["weight_packed", "weight_scale", "weight_global_scale"]
        }
        expected_keys.update(f"{name}.{suffix}" for suffix in state)
        assert state["weight_packed"].shape == (16, 16)
        assert state["weight_packed"].dtype == torch.uint8
        assert state["weight_scale"].dtype == torch.float8_e4m3fn
        restored = NVFP4PackedCompressor.decompress(state, scheme)["weight"]

        # Independent nearest-level FP4 reference, using the exported two-level scale.
        scale = state["weight_scale"].float() / state["weight_global_scale"]
        scale = scale.repeat_interleave(16, dim=-1)
        normalized = weight.to(target).float() / scale
        boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=target)
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=target)
        codes = torch.bucketize(normalized.abs(), boundaries)
        # E2M1 uses round-to-nearest-even at exactly representable midpoints.
        for boundary_index in (1, 3, 5):
            codes = torch.where(normalized.abs() == boundaries[boundary_index], boundary_index + 1, codes)
        expected = levels[codes] * normalized.sign() * scale
        torch.testing.assert_close(restored, expected.to(restored.dtype), rtol=0, atol=0)
        if mode == "w4a4":
            expected_keys.add(f"{name}.input_global_scale")
            torch.testing.assert_close(
                output[f"{name}.input_global_scale"], params[f"{name}.input_global_scale"].to(target)
            )

    for layer in range(2):
        for group in (["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]):
            names = [f"model.layers.{layer}.{projection}" for projection in group]
            expected_scale = 2688.0 / max(params[f"{name}.weight"].abs().max().float().item() for name in names)
            for name in names:
                torch.testing.assert_close(
                    output[f"{name}.weight_global_scale"], torch.tensor([expected_scale], device=target)
                )
    assert set(output) == expected_keys
    assert all(value.device.type == target for value in output.values())
    torch.testing.assert_close(output["lm_head.weight"], params["lm_head.weight"].to(target), rtol=0, atol=0)
