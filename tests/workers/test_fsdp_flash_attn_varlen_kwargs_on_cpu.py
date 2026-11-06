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

"""Explicit varlen ``FlashAttentionKwargs`` for packed micro-batches in the FSDP engine.

``build_flash_attn_varlen_kwargs`` must describe exactly the segmentation that transformers would
otherwise re-derive from ``position_ids`` in every attention layer, so these tests compare it with
``prepare_fa_kwargs_from_position_ids`` on the same packed layout, including the right-pad segment
appended by ``pad_packed_inputs``.

The ``veomni.*`` imports done by ``transformer_impl`` at module load are stubbed with ``MagicMock``
as in ``test_packed_pad_to_length_on_cpu.py``.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.offloading",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.moe_router_replay",
    "veomni.utils.seqlen_pos_transform_utils",
    "veomni.models.checkpoint_tensor_loading",
    "veomni.models.checkpoint_tensor_loading.get_checkpoint_tensor_converter",
):
    sys.modules.setdefault(_mod, MagicMock())

from verl.workers.engine.fsdp.transformer_impl import _supports_flash_attn_varlen_kwargs  # noqa: E402
from verl.workers.engine.utils import build_flash_attn_varlen_kwargs, pad_packed_inputs  # noqa: E402


def _packed_position_ids(seqlens: list[int]) -> torch.Tensor:
    return torch.cat([torch.arange(n) for n in seqlens]).unsqueeze(0)


def _cu_seqlens(seqlens: list[int]) -> torch.Tensor:
    return torch.tensor([0, *seqlens]).cumsum(0)


@pytest.mark.parametrize(
    "seqlens, pad_size",
    [
        ([5, 3], 0),
        ([5, 3], 4),
        ([1, 7, 1, 2], 0),
        ([1, 7, 1, 2], 9),
        ([16], 0),
        ([16], 3),
    ],
)
def test_matches_boundaries_derived_from_position_ids(seqlens, pad_size):
    input_ids = torch.ones(1, sum(seqlens), dtype=torch.long)
    _, position_ids = pad_packed_inputs(input_ids, _packed_position_ids(seqlens), pad_size)
    (ref_cu_q, ref_cu_k), (ref_max_q, ref_max_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    kwargs = build_flash_attn_varlen_kwargs(_cu_seqlens(seqlens), pad_size=pad_size)

    assert kwargs["cu_seq_lens_q"].dtype == torch.int32
    assert torch.equal(kwargs["cu_seq_lens_q"], ref_cu_q)
    assert torch.equal(kwargs["cu_seq_lens_k"], ref_cu_k)
    # Python ints, so neither transformers nor flash-attn needs a device sync to read them.
    assert type(kwargs["max_length_q"]) is int and type(kwargs["max_length_k"]) is int
    assert kwargs["max_length_q"] == kwargs["max_length_k"] == int(ref_max_q) == int(ref_max_k)


def test_empty_micro_batch_passes_nothing():
    assert build_flash_attn_varlen_kwargs(torch.zeros(1, dtype=torch.long)) == {}


def _module(attn_implementation, **config_kwargs):
    return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn_implementation, **config_kwargs))


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "flash_attention_3"])
def test_enabled_for_text_models_on_flash_attention(attn_implementation):
    assert _supports_flash_attn_varlen_kwargs(_module(attn_implementation))


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", "flex_attention", None])
def test_disabled_for_other_attention_implementations(attn_implementation):
    assert not _supports_flash_attn_varlen_kwargs(_module(attn_implementation))


def test_disabled_for_vision_language_models():
    assert not _supports_flash_attn_varlen_kwargs(_module("flash_attention_2", vision_config=SimpleNamespace()))


def test_disabled_without_a_config():
    assert not _supports_flash_attn_varlen_kwargs(SimpleNamespace())
