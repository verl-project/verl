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

import torch
import torch.nn as nn

from verl.utils.torch_dtypes import cast_module_params_to_dtype


class _RotaryLikeModel(nn.Module):
    """Mimics a model holding an FP32 rotary frequency buffer plus parameters."""

    def __init__(self):
        super().__init__()
        dim, theta = 8, 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.weight = nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
        self.step = nn.Parameter(torch.arange(4, dtype=torch.int64), requires_grad=False)


def test_fp32_buffers_keep_their_precision():
    model = _RotaryLikeModel()
    before = model.inv_freq.clone()
    cast_module_params_to_dtype(model, torch.bfloat16)
    assert model.inv_freq.dtype == torch.float32
    assert torch.equal(model.inv_freq, before)


def test_floating_point_parameters_are_cast():
    model = _RotaryLikeModel()
    cast_module_params_to_dtype(model, torch.bfloat16)
    assert model.weight.dtype == torch.bfloat16


def test_meta_initialized_parameters_are_cast():
    model = nn.Linear(4, 4, dtype=torch.float32, device="meta")
    cast_module_params_to_dtype(model, torch.bfloat16)
    assert model.weight.dtype == torch.bfloat16


def test_non_floating_tensors_are_untouched():
    model = _RotaryLikeModel()
    model.register_buffer("counts", torch.arange(4, dtype=torch.int64))
    cast_module_params_to_dtype(model, torch.bfloat16)
    assert model.step.dtype == torch.int64
    assert model.counts.dtype == torch.int64


def test_tied_parameter_identity_is_preserved():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Parameter(torch.ones(3, 3, dtype=torch.float32))
            self.head = nn.Linear(3, 3, bias=False)
            self.head.weight = self.embed

    model = Tied()
    embed = model.embed
    cast_module_params_to_dtype(model, torch.bfloat16)
    assert model.embed is embed
    assert model.head.weight is embed
    assert model.embed.dtype == torch.bfloat16
def test_blanket_module_cast_rounds_fp32_buffers():
    """Document the regression this helper avoids (verl-project/verl#8154).

    A blanket ``nn.Module.to(dtype)`` also converts registered buffers, so the
    FP32 ``inv_freq`` values are rounded and cannot be recovered; the
    parameter-only cast keeps them bit-identical.
    """
    reference = _RotaryLikeModel().inv_freq.clone()

    blanket = _RotaryLikeModel()
    blanket.to(torch.bfloat16)
    assert blanket.inv_freq.dtype == torch.bfloat16
    assert not torch.equal(blanket.inv_freq.float(), reference)

    protected = _RotaryLikeModel()
    cast_module_params_to_dtype(protected, torch.bfloat16)
    assert protected.inv_freq.dtype == torch.float32
    assert torch.equal(protected.inv_freq, reference)
