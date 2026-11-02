# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

pytest.importorskip("compressed_tensors")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    _compress_weight,
    _compress_weight_with_compressed_tensors,
    _scheme_for_preset,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="TileLang FP8/FP4 export kernels need an SM90 or later GPU",
)


@pytest.mark.parametrize(
    ("preset", "shape"),
    [
        ("FP8_BLOCK", (130, 257)),
    ],
)
def test_export_kernel_matches_compressed_tensors_bitwise(preset, shape):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight[0].zero_()
    weight[1].mul_(1e-8)
    scheme = _scheme_for_preset(preset, ["Linear"])

    actual = _compress_weight(weight, scheme)
    expected = _compress_weight_with_compressed_tensors(weight, scheme)

    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))


@pytest.mark.parametrize(
    ("preset", "scale_name", "expected_scale_bytes"),
    [
        ("MXFP8", "weight_scale", (127, 127, 128, 129)),
        ("MXFP4", "weight_scale", (134, 134, 134, 135)),
    ],
)
def test_mx_export_kernel_uses_deepseek_rounding(preset, scale_name, expected_scale_bytes):
    weight = (
        torch.tensor((400.0, 448.0, 500.0, 1000.0), device="cuda", dtype=torch.bfloat16)
        .unsqueeze(-1)
        .expand(-1, 32)
        .contiguous()
    )
    scheme = _scheme_for_preset(preset, ["Linear"])

    actual = _compress_weight(weight, scheme)

    assert torch.equal(
        actual[scale_name].reshape(-1),
        torch.tensor(expected_scale_bytes, device="cuda", dtype=torch.uint8),
    )


def test_nvfp4_export_keeps_compressed_tensors_path(monkeypatch):
    weight = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
    scheme = _scheme_for_preset("NVFP4", ["Linear"])
    expected = _compress_weight_with_compressed_tensors(weight, scheme)
    calls = []

    def fallback(*args, **kwargs):
        calls.append(True)
        return expected

    monkeypatch.setattr("verl.utils.qat.compressed_tensors._compress_weight_with_compressed_tensors", fallback)

    actual = _compress_weight(weight, scheme)

    assert calls == [True]
    assert actual is expected
