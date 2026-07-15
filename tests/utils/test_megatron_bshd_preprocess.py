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
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_dynamic_data_context_parallel_groups = lambda group_size: ("dcp", group_size)

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


def test_preprocess_thd_engine_pads_short_topk_sequence_dimension(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    topk = 64
    teacher_logprobs = _nested_tensor([torch.arange(topk, dtype=torch.float32).reshape(1, topk)])

    packed, packed_seq_params, position_ids = mcore_util.preprocess_thd_engine(
        teacher_logprobs,
        local_cp_size=2,
    )

    assert packed.shape == (1, 2, topk)
    torch.testing.assert_close(packed, torch.zeros_like(packed))
    torch.testing.assert_close(position_ids, torch.tensor([[1, 0]], dtype=torch.long))
    assert packed_seq_params.local_cp_size == 2


def test_dcp_local_thd_postprocess_matches_local_token_mask(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)

    input_ids = _nested_tensor([torch.arange(16, dtype=torch.long)])
    packed_seq_params = types.SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 16], dtype=torch.int32),
        cp_group=object(),
    )
    local_output = torch.tensor([[10.0, 11.0, 12.0, 13.0]])

    postprocessed = mcore_util.postprocess_thd_engine_local(
        local_output,
        packed_seq_params,
        input_ids,
        batch_size=1,
        local_cp_size=4,
    )
    local_mask = mcore_util.build_thd_local_token_mask(
        packed_seq_params,
        input_ids,
        batch_size=1,
        local_cp_size=4,
    )
    compact = mcore_util.postprocess_thd_engine_local(
        local_output,
        packed_seq_params,
        input_ids,
        batch_size=1,
        local_cp_size=4,
        compact=True,
    )
    local_indices = mcore_util.build_thd_local_token_indices(
        packed_seq_params,
        input_ids,
        batch_size=1,
        local_cp_size=4,
    )
    full_seq_lens = mcore_util.build_thd_full_seq_lens(input_ids)

    expected = torch.zeros(16)
    expected[[2, 3, 12, 13]] = torch.tensor([10.0, 11.0, 12.0, 13.0])
    expected_mask = torch.zeros(16, dtype=torch.bool)
    expected_mask[[2, 3, 12, 13]] = True

    torch.testing.assert_close(postprocessed.unbind()[0], expected)
    torch.testing.assert_close(local_mask.unbind()[0], expected_mask)
    torch.testing.assert_close(compact.unbind()[0], torch.tensor([10.0, 11.0, 12.0, 13.0]))
    torch.testing.assert_close(local_indices.unbind()[0], torch.tensor([2, 3, 12, 13]))
    torch.testing.assert_close(full_seq_lens.unbind()[0], torch.tensor([16]))


def test_dcp_local_thd_metadata_exactly_covers_unaligned_sequence(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    cp_rank = 0
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: cp_rank)
    input_ids = _nested_tensor([torch.arange(100, 109, dtype=torch.long)])
    all_indices = []

    for rank in range(4):
        cp_rank = rank
        packed, packed_seq_params, _ = mcore_util.preprocess_thd_engine(input_ids, local_cp_size=4)
        compact = mcore_util.postprocess_thd_engine_local(
            packed,
            packed_seq_params,
            input_ids,
            batch_size=1,
            local_cp_size=4,
            compact=True,
        ).unbind()[0]
        indices = mcore_util.build_thd_local_token_indices(
            packed_seq_params,
            input_ids,
            batch_size=1,
            local_cp_size=4,
        ).unbind()[0]

        torch.testing.assert_close(compact, input_ids.unbind()[0][indices])
        all_indices.append(indices)

    torch.testing.assert_close(torch.cat(all_indices).sort().values, torch.arange(9))
