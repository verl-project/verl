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
"""Regression coverage for verl#6492."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

import verl.utils.device as device_module


def _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size: int = 4):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")

    parallel_state.get_context_parallel_world_size = lambda: 1
    parallel_state.get_context_parallel_rank = lambda: 0
    parallel_state.get_context_parallel_group = lambda: None
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
    spec = importlib.util.spec_from_file_location("mcore_util_regression", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_tensor(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _check_topk_preprocess(monkeypatch, device: torch.device):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    topk = 64

    logprob_rows = [
        torch.arange(3 * topk, dtype=torch.float32, device=device).reshape(3, topk),
        torch.arange(2 * topk, dtype=torch.float32, device=device).reshape(2, topk) + 1000,
    ]
    teacher_logprobs = _nested_tensor(logprob_rows)

    logprobs_bshd, attention_mask, position_ids = mcore_util.preprocess_bshd_engine(teacher_logprobs)

    assert logprobs_bshd.shape == (2, 4, topk)
    assert logprobs_bshd.device.type == device.type
    assert attention_mask.device.type == device.type
    assert position_ids.shape == (2, 4)
    torch.testing.assert_close(logprobs_bshd[0, :3], logprob_rows[0])
    torch.testing.assert_close(logprobs_bshd[1, :2], logprob_rows[1])
    torch.testing.assert_close(logprobs_bshd[0, 3], torch.zeros(topk, dtype=torch.float32, device=device))
    torch.testing.assert_close(logprobs_bshd[1, 2:], torch.zeros(2, topk, dtype=torch.float32, device=device))
    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[True, True, True, False], [True, True, False, False]], device=device),
    )
    torch.testing.assert_close(
        position_ids,
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long, device=device),
    )

    id_rows = [
        torch.arange(3 * topk, dtype=torch.long, device=device).reshape(3, topk),
        torch.arange(2 * topk, dtype=torch.long, device=device).reshape(2, topk) + 2000,
    ]
    teacher_ids = _nested_tensor(id_rows)
    ids_bshd, ids_attention_mask, _ = mcore_util.preprocess_bshd_engine(teacher_ids)

    assert ids_bshd.shape == (2, 4, topk)
    assert ids_bshd.dtype == torch.long
    torch.testing.assert_close(ids_bshd[0, :3], id_rows[0])
    torch.testing.assert_close(ids_bshd[1, :2], id_rows[1])
    torch.testing.assert_close(ids_attention_mask, attention_mask)


def test_preprocess_bshd_engine_preserves_1d_input_shape_on_cpu(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    rows = [
        torch.tensor([11, 12, 13], dtype=torch.long),
        torch.tensor([21, 22], dtype=torch.long),
    ]
    input_ids = _nested_tensor(rows)

    input_ids_bshd, attention_mask, position_ids = mcore_util.preprocess_bshd_engine(input_ids)

    assert input_ids_bshd.shape == (2, 4)
    torch.testing.assert_close(input_ids_bshd[0], torch.tensor([11, 12, 13, 0], dtype=torch.long))
    torch.testing.assert_close(input_ids_bshd[1], torch.tensor([21, 22, 0, 0], dtype=torch.long))
    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
    )
    torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long))


def test_preprocess_bshd_engine_preserves_topk_dense_dim_on_cpu(monkeypatch):
    _check_topk_preprocess(monkeypatch, torch.device("cpu"))


def test_preprocess_bshd_engine_preserves_topk_dense_dim_on_gpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    _check_topk_preprocess(monkeypatch, torch.device("cuda"))


def test_preprocess_thd_engine_pads_to_minimum_rows(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    input_ids = _nested_tensor([torch.arange(100, dtype=torch.long)])

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        min_local_rows=128,
    )

    assert local_ids.shape == (1, 128)
    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 128]
    torch.testing.assert_close(local_ids[0, :100], torch.arange(100, dtype=torch.long))
    torch.testing.assert_close(local_ids[0, 100:], torch.zeros(28, dtype=torch.long))
    torch.testing.assert_close(local_positions[0, :100], torch.arange(100, dtype=torch.long))
    torch.testing.assert_close(local_positions[0, 100:], torch.zeros(28, dtype=torch.long))


def test_preprocess_thd_engine_uses_flat_packed_cu_seqlens(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=4)
    input_ids = _nested_tensor([torch.arange(5, dtype=torch.long), torch.arange(3, dtype=torch.long) + 10])
    cu_seqlens = torch.tensor([0, 2, 5, 7, 8], dtype=torch.int32)

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        packed_cu_seqlens=cu_seqlens,
    )

    expected_padded_cu = torch.tensor([0, 2, 8, 10, 12], dtype=torch.int32)
    torch.testing.assert_close(packed_seq_params.cu_seqlens_q, expected_padded_cu)
    torch.testing.assert_close(packed_seq_params.cu_seqlens_kv, expected_padded_cu)
    torch.testing.assert_close(
        getattr(packed_seq_params, mcore_util._VERL_ROW_CU_SEQLENS_PADDED),
        torch.tensor([0, 8, 12], dtype=torch.int32),
    )
    torch.testing.assert_close(local_ids[0, :5], torch.arange(5, dtype=torch.long))
    torch.testing.assert_close(local_ids[0, 5:8], torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(local_ids[0, 8:11], torch.arange(3, dtype=torch.long) + 10)
    torch.testing.assert_close(local_ids[0, 11], torch.tensor(0, dtype=torch.long))
    torch.testing.assert_close(local_positions[0], torch.tensor([0, 1, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1]))

    output = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
    restored = mcore_util.postprocess_thd_engine(output, packed_seq_params, input_ids, batch_size=2)
    restored_rows = restored.unbind()
    torch.testing.assert_close(restored_rows[0].squeeze(-1), torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(restored_rows[1].squeeze(-1), torch.arange(8, 11, dtype=torch.float32))


def test_preprocess_thd_engine_merges_per_row_padded_cu_seqlens(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=4)
    input_ids = _nested_tensor([torch.arange(5, dtype=torch.long), torch.arange(3, dtype=torch.long) + 10])
    cu_seqlens = torch.tensor(
        [
            [0, 2, 5, 5],
            [0, 2, 3, 3],
        ],
        dtype=torch.int32,
    )

    _, packed_seq_params, _ = mcore_util.preprocess_thd_engine(
        input_ids,
        packed_cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        packed_seq_params.cu_seqlens_q,
        torch.tensor([0, 2, 8, 10, 12], dtype=torch.int32),
    )


def test_preprocess_thd_engine_rolls_within_packed_cu_seqlens(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=4)
    input_ids = _nested_tensor([torch.arange(5, dtype=torch.long), torch.arange(3, dtype=torch.long) + 10])
    cu_seqlens = torch.tensor([0, 2, 5, 7, 8], dtype=torch.int32)

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        need_roll=True,
        packed_cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        packed_seq_params.cu_seqlens_q,
        torch.tensor([0, 2, 8, 10, 12], dtype=torch.int32),
    )
    torch.testing.assert_close(local_ids[0], torch.tensor([1, 0, 3, 4, 0, 0, 0, 2, 11, 10, 0, 12]))
    torch.testing.assert_close(local_positions[0], torch.tensor([1, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1, 0]))


def test_preprocess_thd_engine_rejects_cu_seqlens_without_row_boundaries(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=4)
    input_ids = _nested_tensor([torch.arange(5, dtype=torch.long), torch.arange(3, dtype=torch.long) + 10])
    cu_seqlens = torch.tensor([0, 2, 7, 8], dtype=torch.int32)

    with pytest.raises(ValueError, match="must include nested input row boundaries"):
        mcore_util.preprocess_thd_engine(input_ids, packed_cu_seqlens=cu_seqlens)
