# Copyright contributors to the PEAR project
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

import itertools
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def test_packed_convolution_matches_the_per_sequence_convolution():
    from verl.models.transformers.qwen3_5 import _packed_causal_conv1d

    torch.manual_seed(0)
    channels, width, lengths = 256, 4, [7, 300, 33]
    conv = nn.Conv1d(channels, channels, width, groups=channels, padding=width - 1).cuda().to(torch.bfloat16)
    x = torch.randn(1, channels, sum(lengths), device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    expected = torch.cat([F.silu(conv(seg)[:, :, : seg.shape[-1]]) for seg in x.split(lengths, dim=2)], dim=-1)

    actual = _packed_causal_conv1d(SimpleNamespace(conv1d=conv), x, cu_seqlens, cu_seqlens.cpu().long())
    assert actual.shape == expected.shape
    assert torch.allclose(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
