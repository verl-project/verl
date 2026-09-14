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
"""CPU tests for the MXFP8 post-refit self-check (dequant reference + probe)."""

import pytest
import torch

from verl.utils.mxfp8_refit_check import assert_mxfp8_linear_matches, mxfp8_dequantize, probe_mxfp8_linear

N, K = 16, 128


def _fake_layer(seed=0):
    g = torch.Generator().manual_seed(seed)
    # fp8-representable values: small integers scaled by a per-block power of two
    q = torch.randint(-6, 7, (N, K), generator=g).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 130, (N, K // 32), generator=g).to(torch.uint8)  # 2^-7 .. 2^2
    return q, scale


def test_dequantize_applies_per_block_power_of_two():
    q = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [126, 130]], dtype=torch.uint8)  # 2^0, 2^1 / 2^-1, 2^3
    w = mxfp8_dequantize(q, scale)
    assert torch.equal(w[0, :32], torch.ones(32)) and torch.equal(w[0, 32:], torch.full((32,), 2.0))
    assert torch.equal(w[1, :32], torch.full((32,), 0.5)) and torch.equal(w[1, 32:], torch.full((32,), 8.0))


def test_dequantize_rejects_wrong_scale_shape():
    q = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        mxfp8_dequantize(q, torch.zeros(2, 3, dtype=torch.uint8))


def test_probe_passes_for_a_correct_kernel():
    q, scale = _fake_layer()
    w = mxfp8_dequantize(q, scale)

    def good_apply(x):  # what a healthy kernel computes, up to activation quantization noise
        return (x.float() @ w.t()).to(torch.bfloat16)

    rel = probe_mxfp8_linear(good_apply, q, scale)
    assert rel < 0.02  # bf16 rounding only; real kernels add ~4% from activation quantization
    assert assert_mxfp8_linear_matches("l0", good_apply, q, scale, engine="test") == pytest.approx(rel)


def test_probe_catches_stale_scales():
    q, scale = _fake_layer()
    stale = mxfp8_dequantize(q, scale + 1)  # every block descaled by 2x: fresh weights, old scale layout

    def stale_apply(x):
        return (x.float() @ stale.t()).to(torch.bfloat16)

    assert probe_mxfp8_linear(stale_apply, q, scale) > 0.9
    with pytest.raises(RuntimeError, match="refit self-check failed"):
        assert_mxfp8_linear_matches("l0", stale_apply, q, scale, engine="test")


def test_probe_catches_nan_output():
    q, scale = _fake_layer()

    def nan_apply(x):
        return torch.full((x.shape[0], N), float("nan"), dtype=torch.bfloat16)

    assert probe_mxfp8_linear(nan_apply, q, scale) == float("inf")


def test_check_is_skipped_not_failed_when_apply_is_incompatible():
    q, scale = _fake_layer()

    def broken_apply(x):
        raise TypeError("unexpected keyword")

    rel = assert_mxfp8_linear_matches("l0", broken_apply, q, scale, engine="test")
    assert rel != rel  # nan: skipped, not raised


def test_env_disable_and_tolerance(monkeypatch):
    from verl.utils import mxfp8_refit_check as m

    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK", "0")
    assert not m.refit_check_enabled()
    monkeypatch.setenv("VERL_MXFP8_REFIT_CHECK_TOL", "0.5")
    assert m.refit_check_tolerance() == 0.5
