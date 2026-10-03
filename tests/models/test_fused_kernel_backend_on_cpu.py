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

import warnings

import pytest

from verl.models.transformers.monkey_patch import _resolve_fused_kernels_backend, patch_forward_with_backends


@pytest.mark.parametrize("npu_available", [False, True])
def test_triton_backend_resolution(monkeypatch, npu_available):
    """Triton is preserved on CUDA/CPU and falls back on Ascend NPU."""
    seen = {}

    def fake_is_npu_available(*, check_device):
        seen["check_device"] = check_device
        return npu_available

    monkeypatch.setattr("verl.utils.device.is_torch_npu_available", fake_is_npu_available)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = _resolve_fused_kernels_backend("triton")

    assert seen["check_device"] is False
    assert backend == ("torch" if npu_available else "triton")
    if npu_available:
        assert len(caught) == 1
        assert "falling back to the torch backend" in str(caught[0].message)
    else:
        assert not caught


@pytest.mark.parametrize("backend", [None, "torch", "liger"])
def test_non_triton_backend_is_unchanged(monkeypatch, backend):
    """Only the unsupported NPU Triton combination is rewritten."""
    monkeypatch.setattr("verl.utils.device.is_torch_npu_available", lambda **_: True)
    assert _resolve_fused_kernels_backend(backend) == backend


def test_npu_triton_request_uses_torch_forward(monkeypatch):
    """The public backend patch must select the safe implementation on NPU."""

    class FakeModel:
        config = type("Config", (), {"model_type": "unsupported"})()

        def forward(self):
            return None

    monkeypatch.setattr("verl.utils.device.is_torch_npu_available", lambda **_: True)
    original_forward = FakeModel.forward
    model = FakeModel()

    with pytest.warns(RuntimeWarning, match="falling back to the torch backend"):
        patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="triton")

    from verl.models.transformers import dense_common

    assert FakeModel.forward is dense_common.forward_with_torch_backend
    assert model._verl_fused_kernels_backend == "torch"
    assert FakeModel.forward is not original_forward
