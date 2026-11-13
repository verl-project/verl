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
"""Unit tests for the backend dispatch in verl.utils.attention_utils."""

import sys
import types
from unittest import mock

import pytest
import torch

from verl.utils import attention_padding_utils, attention_utils


@pytest.fixture
def reset_resolved_functions():
    """The resolved functions live in module globals; clear them around each test."""
    attention_utils._index_first_axis = None
    attention_utils._pad_input = None
    attention_utils._rearrange = None
    attention_utils._unpad_input = None
    yield
    attention_utils._index_first_axis = None
    attention_utils._pad_input = None
    attention_utils._rearrange = None
    attention_utils._unpad_input = None


def _fake_flash_attn():
    """A stand-in for the flash_attn package, which has no wheel on non-CUDA devices."""
    pkg = types.ModuleType("flash_attn")
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    for name in ("index_first_axis", "pad_input", "rearrange", "unpad_input"):
        setattr(bert_padding, name, mock.Mock(name=name, return_value=f"flash_{name}"))
    pkg.bert_padding = bert_padding
    return {"flash_attn": pkg, "flash_attn.bert_padding": bert_padding}


def test_prefers_flash_attn_when_installed(reset_resolved_functions):
    """With flash-attn importable, all four entry points dispatch to it."""
    with (
        mock.patch.dict(sys.modules, _fake_flash_attn()),
        mock.patch("verl.utils.device.is_torch_npu_available", return_value=False),
    ):
        assert attention_utils.index_first_axis() == "flash_index_first_axis"
        assert attention_utils.pad_input() == "flash_pad_input"
        assert attention_utils.rearrange() == "flash_rearrange"
        assert attention_utils.unpad_input() == "flash_unpad_input"


def test_falls_back_to_padding_utils_without_flash_attn(reset_resolved_functions):
    """Without flash-attn, dispatch goes to the pure-torch port rather than a local duplicate."""
    # A None entry in sys.modules makes `import flash_attn...` raise ImportError.
    with (
        mock.patch.dict(sys.modules, {"flash_attn": None, "flash_attn.bert_padding": None}),
        mock.patch("verl.utils.device.is_torch_npu_available", return_value=False),
    ):
        index_first_axis, pad_input, rearrange, unpad_input = attention_utils._get_attention_functions()

    assert index_first_axis is attention_padding_utils.index_first_axis
    assert pad_input is attention_padding_utils.pad_input
    assert rearrange is attention_padding_utils.rearrange
    assert unpad_input is attention_padding_utils.unpad_input


def test_npu_uses_padding_utils_over_flash_attn(reset_resolved_functions):
    """NPU keeps its existing behaviour: the port wins even if flash-attn is importable."""
    with (
        mock.patch.dict(sys.modules, _fake_flash_attn()),
        mock.patch("verl.utils.device.is_torch_npu_available", return_value=True),
    ):
        index_first_axis, *_ = attention_utils._get_attention_functions()

    assert index_first_axis is attention_padding_utils.index_first_axis


def test_padding_utils_unpad_pad_round_trip():
    """pad_input(unpad_input(x)) must restore the padded tensor and report correct metadata."""
    hidden_states = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    unpadded, indices, cu_seqlens, max_seqlen, seqused = attention_padding_utils.unpad_input(
        hidden_states, attention_mask
    )

    assert unpadded.shape == (3, 4)
    assert max_seqlen == 2
    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(seqused, torch.tensor([2, 1], dtype=torch.int32))

    repadded = attention_padding_utils.pad_input(unpadded, indices, batch=2, seqlen=3)
    torch.testing.assert_close(repadded, hidden_states * attention_mask.unsqueeze(-1))


def test_padding_utils_index_first_axis_is_differentiable():
    """index_first_axis is an autograd.Function; gradients must flow back to the unselected rows."""
    tensor = torch.randn(4, 3, requires_grad=True)
    indices = torch.tensor([0, 2])

    attention_padding_utils.index_first_axis(tensor, indices).sum().backward()

    expected = torch.zeros(4, 3)
    expected[indices] = 1.0
    torch.testing.assert_close(tensor.grad, expected)
