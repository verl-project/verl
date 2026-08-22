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
"""FP8 padding alignment for THD preprocessing, incl. the mxfp8 recipe."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

import verl.utils.device as device_module


def _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size: int = 1, cp_size: int = 1, cp_rank: int = 0):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")

    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_context_parallel_group = lambda: object()
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size

    class PackedSeqParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    packed_seq_params.PackedSeqParams = PackedSeqParams

    core.parallel_state = parallel_state
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_fp8_padding", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_fp8_thd_align_size(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)

    # Default (blockwise & friends): per-seq lcm(16, align), total align*128.
    assert mcore_util._compute_fp8_thd_align_size(1) == (16, 128)
    assert mcore_util._compute_fp8_thd_align_size(2) == (16, 256)
    assert mcore_util._compute_fp8_thd_align_size(2, "blockwise") == (16, 256)
    assert mcore_util._compute_fp8_thd_align_size(12) == (48, 12 * 128)

    # mxfp8 scales in 32-element blocks: per-seq lcm(32, align).
    assert mcore_util._compute_fp8_thd_align_size(1, "mxfp8") == (32, 128)
    assert mcore_util._compute_fp8_thd_align_size(2, "mxfp8") == (32, 256)
    assert mcore_util._compute_fp8_thd_align_size(12, "mxfp8") == (96, 12 * 128)


def test_get_fp8_padding_options(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)

    class FakeConfig:
        fp8 = None
        fp8_recipe = "mxfp8"

    # fp8 disabled: no padding, recipe not forwarded.
    assert mcore_util.get_fp8_padding_options(FakeConfig()) == (False, None)

    config = FakeConfig()
    config.fp8 = "e4m3"
    assert mcore_util.get_fp8_padding_options(config) == (True, "mxfp8")

    config.fp8 = "hybrid"
    config.fp8_recipe = "blockwise"
    assert mcore_util.get_fp8_padding_options(config) == (True, "blockwise")


def _padded_seqlens(packed_seq_params) -> torch.Tensor:
    return packed_seq_params.cu_seqlens_q_padded.diff()


def test_preprocess_packed_seqs_mxfp8_alignment(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)

    batch_size, seq_len = 2, 100
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, :17] = True
    attention_mask[1, :50] = True

    _, params_default = mcore_util.preprocess_packed_seqs(
        input_ids, attention_mask, pre_process=True, use_fp8_padding=True
    )
    seqlens_default = _padded_seqlens(params_default)
    # Every sequence 16-aligned, total 128-aligned.
    assert (seqlens_default % 16 == 0).all()
    assert params_default.cu_seqlens_q_padded[-1] % 128 == 0

    _, params_mxfp8 = mcore_util.preprocess_packed_seqs(
        input_ids, attention_mask, pre_process=True, use_fp8_padding=True, fp8_recipe="mxfp8"
    )
    seqlens_mxfp8 = _padded_seqlens(params_mxfp8)
    # Every sequence 32-aligned, total still 128-aligned.
    assert (seqlens_mxfp8 % 32 == 0).all()
    assert params_mxfp8.cu_seqlens_q_padded[-1] % 128 == 0


def test_preprocess_thd_engine_mxfp8_alignment(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)

    rows = [
        torch.arange(17, dtype=torch.long),
        torch.arange(50, dtype=torch.long),
    ]
    input_ids = torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    _, params_mxfp8, _ = mcore_util.preprocess_thd_engine(
        input_ids, pre_process=True, use_fp8_padding=True, fp8_recipe="mxfp8"
    )
    seqlens_mxfp8 = _padded_seqlens(params_mxfp8)
    assert (seqlens_mxfp8 % 32 == 0).all()
    assert params_mxfp8.cu_seqlens_q_padded[-1] % 128 == 0
