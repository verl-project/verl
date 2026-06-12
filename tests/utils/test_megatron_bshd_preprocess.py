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
    packed_seq_params.PackedSeqParams = type("PackedSeqParams", (), {})

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


def _load_thd_engine_util(monkeypatch, tp_size: int = 1, cp_size: int = 2, cp_rank: int = 0):
    """Load mcore util with cp_size/cp_rank configured for preprocess_thd_engine tests.

    Kept separate from _load_mcore_util_with_stubbed_megatron to avoid touching
    the existing bshd-engine test helpers.
    """
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")

    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    packed_seq_params.PackedSeqParams = type("PackedSeqParams", (), {"__init__": lambda self, **kw: None})

    core.parallel_state = parallel_state
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_thd", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # warning_once is a verl-specific logger extension; fall back to standard warning
    if not hasattr(module.logger, "warning_once"):
        module.logger.warning_once = module.logger.warning
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


# ---------------------------------------------------------------------------
# Regression tests for FP8 + CP + SFT bugfix
#
# Root cause: when use_fp8_padding=True, seqlen_padded_i >> align_size, but the
# old code guarded padding with `d.numel() < align_size` and filled only up to
# `align_size` tokens.  For higher cp_ranks the first chunk slice
# `d[half_seqlen*cp_rank : half_seqlen*(cp_rank+1)]` can reach indices beyond
# d.shape[0], causing an IndexError.  The fix pads d to seqlen_padded_i and
# adds bounds-safe slice helpers.
# ---------------------------------------------------------------------------


def _make_thd_input(seqlens: list[int]) -> torch.Tensor:
    """Build a 1-D jagged nested tensor with token id == position index."""
    rows = [torch.arange(s, dtype=torch.long) for s in seqlens]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def test_preprocess_thd_engine_fp8_cp_no_crash_on_cpu(monkeypatch):
    """FP8+CP: preprocess_thd_engine must not raise IndexError for any cp_rank.

    Reproduces the pre-fix OOB crash:
      tp_size=1, cp_size=2, use_fp8_padding=True, seqlen_orig=5
        align_size      = tp_size * cp_size * 2 = 4
        per_seq_align   = lcm(16, 4) = 16   (from _compute_fp8_thd_align_size)
        seqlen_padded   = 512  (16 per-seq + 496 total-align extra)
        seqlen_per_rank = 256, half_seqlen = 128

      cp_rank=1: first chunk slice = d[128:256] but d.shape[0]==5 → OOB
        old code: `if d.numel() < align_size` (5 < 4 is False) → no pad → crash
        new code: `if d.numel() < seqlen_padded_i` (5 < 512)   → pad to 512 → OK
    """
    for cp_rank in range(2):
        mcore_util = _load_thd_engine_util(monkeypatch, tp_size=1, cp_size=2, cp_rank=cp_rank)
        # seqlen=5 is intentionally much shorter than seqlen_padded to trigger the bug
        input_ids = _make_thd_input([5])
        ids_rmpad, _, pos_ids = mcore_util.preprocess_thd_engine(
            input_ids, pre_process=True, need_roll=False, use_fp8_padding=True
        )
        assert ids_rmpad is not None
        assert pos_ids is not None


def test_preprocess_thd_engine_fp8_cp_correct_content_on_cpu(monkeypatch):
    """FP8+CP: zigzag chunks must contain the correct token ids after the fix.

    With tp_size=1, cp_size=2, seqlen_orig=200, use_fp8_padding=True:
      per_seq_align = lcm(16, 4) = 16, total_align = 4*128 = 512
      seqlen_padded = 512  (208 rounded up to 512 by total_align)
      seqlen_per_rank = 256,  half_seqlen = 128

    cp_rank=0: first chunk = tokens[0:128],   remain = tokens[384:512] (all padding)
    cp_rank=1: first chunk = tokens[128:200] (valid) + zeros[200:256],
               remain = tokens[256:200] → remain_len=0, all zero

    Token ids equal their position index, so we assert exact values.
    """
    seqlen_orig = 200
    half_seqlen = 128  # seqlen_per_rank=256 // 2

    # ---- cp_rank=0 ----
    mcore_util_r0 = _load_thd_engine_util(monkeypatch, tp_size=1, cp_size=2, cp_rank=0)
    ids_r0, _, pos_r0 = mcore_util_r0.preprocess_thd_engine(
        _make_thd_input([seqlen_orig]), pre_process=True, need_roll=False, use_fp8_padding=True
    )
    ids_r0 = ids_r0[0]  # [seqlen_per_rank]
    pos_r0 = pos_r0[0]
    # first chunk [0:128]: token ids and positions = 0..127
    torch.testing.assert_close(ids_r0[:half_seqlen], torch.arange(0, half_seqlen, dtype=torch.long))
    torch.testing.assert_close(pos_r0[:half_seqlen], torch.arange(0, half_seqlen, dtype=torch.long))
    # remain chunk [128:256]: tokens[384:512] don't exist (seqlen=200), must be zero
    assert ids_r0[half_seqlen:].sum().item() == 0, "remain chunk for cp_rank=0 should be all-zero padding"

    # ---- cp_rank=1 ----
    mcore_util_r1 = _load_thd_engine_util(monkeypatch, tp_size=1, cp_size=2, cp_rank=1)
    ids_r1, _, pos_r1 = mcore_util_r1.preprocess_thd_engine(
        _make_thd_input([seqlen_orig]), pre_process=True, need_roll=False, use_fp8_padding=True
    )
    ids_r1 = ids_r1[0]  # [seqlen_per_rank]
    pos_r1 = pos_r1[0]
    valid_in_first = seqlen_orig - half_seqlen  # 200 - 128 = 72 valid tokens
    # first chunk: valid tokens [128:200], padding [200:256]
    torch.testing.assert_close(ids_r1[:valid_in_first], torch.arange(half_seqlen, seqlen_orig, dtype=torch.long))
    torch.testing.assert_close(pos_r1[:valid_in_first], torch.arange(half_seqlen, seqlen_orig, dtype=torch.long))
    assert ids_r1[valid_in_first:half_seqlen].sum().item() == 0, "padding after valid first-chunk tokens"
    # remain chunk [128:256]: remain_start=256, remain_end=min(384, 200)=200 < 256 → remain_len=0
    assert ids_r1[half_seqlen:].sum().item() == 0, "remain chunk for cp_rank=1 should be all-zero"
