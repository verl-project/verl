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
"""Low-precision padding alignment for THD preprocessing, incl. the mxfp8 recipe, and its use by top-k distillation."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    parallel_state.get_dynamic_data_context_parallel_groups = lambda group_size: f"dynamic-cp-group-of-{group_size}"
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size

    class PackedSeqParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    packed_seq_params.PackedSeqParams = PackedSeqParams

    # Mirrors megatron.core.fp8_utils.get_fp8_align_size for CI hosts without megatron.
    fp8_utils = types.ModuleType("megatron.core.fp8_utils")
    fp8_utils.get_fp8_align_size = lambda fp8_recipe: 32 if fp8_recipe == "mxfp8" else 16

    core.parallel_state = parallel_state
    core.fp8_utils = fp8_utils
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setitem(sys.modules, "megatron.core.fp8_utils", fp8_utils)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_fp8_padding", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_quant_thd_align_size(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)

    # 16-token recipes (blockwise & friends): per-seq lcm(16, align), total align*128.
    assert mcore_util._compute_quant_thd_align_size(1, 16) == (16, 128)
    assert mcore_util._compute_quant_thd_align_size(2, 16) == (16, 256)
    assert mcore_util._compute_quant_thd_align_size(12, 16) == (48, 12 * 128)

    # mxfp8 scales in 32-element blocks: per-seq lcm(32, align).
    assert mcore_util._compute_quant_thd_align_size(1, 32) == (32, 128)
    assert mcore_util._compute_quant_thd_align_size(2, 32) == (32, 256)
    assert mcore_util._compute_quant_thd_align_size(12, 32) == (96, 12 * 128)


def test_get_thd_quant_align_size(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)

    class FakeConfig:
        fp8 = None
        fp8_recipe = "mxfp8"

    # No fp8 training: no low-precision padding, whatever the recipe field says.
    assert mcore_util.get_thd_quant_align_size(FakeConfig()) is None

    config = FakeConfig()
    config.fp8 = "e4m3"
    assert mcore_util.get_thd_quant_align_size(config) == 32

    config.fp8 = "hybrid"
    config.fp8_recipe = "blockwise"
    assert mcore_util.get_thd_quant_align_size(config) == 16


def _padded_seqlens(packed_seq_params) -> torch.Tensor:
    return packed_seq_params.cu_seqlens_q_padded.diff()


def test_preprocess_packed_seqs_mxfp8_alignment(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)

    batch_size, seq_len = 2, 100
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, :17] = True
    attention_mask[1, :50] = True

    _, params_none = mcore_util.preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
    # No low-precision padding: the sequences keep their lengths (TP=1, CP=1).
    assert _padded_seqlens(params_none).tolist() == [17, 50]

    _, params_default = mcore_util.preprocess_packed_seqs(
        input_ids, attention_mask, pre_process=True, quant_align_size=16
    )
    seqlens_default = _padded_seqlens(params_default)
    # Every sequence 16-aligned, total 128-aligned.
    assert (seqlens_default % 16 == 0).all()
    assert params_default.cu_seqlens_q_padded[-1] % 128 == 0

    _, params_mxfp8 = mcore_util.preprocess_packed_seqs(
        input_ids, attention_mask, pre_process=True, quant_align_size=32
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

    _, params_mxfp8, _ = mcore_util.preprocess_thd_engine(input_ids, pre_process=True, quant_align_size=32)
    seqlens_mxfp8 = _padded_seqlens(params_mxfp8)
    assert (seqlens_mxfp8 % 32 == 0).all()
    assert params_mxfp8.cu_seqlens_q_padded[-1] % 128 == 0


# Top-k distillation packs the teacher tensors itself and asserts that they line up with the student logits,
# which come from inputs packed with the low-precision padding.

_DISTILLATION_CONFIG = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=-20.0, tail_mass_eps=1e-6))


def _load_megatron_distillation_losses(monkeypatch, mcore_util):
    # the losses module packs with verl.models.mcore.util: use the copy loaded against stubbed megatron
    monkeypatch.setitem(sys.modules, "verl.models.mcore.util", mcore_util)
    path = Path(__file__).parents[2] / "verl" / "trainer" / "distillation" / "megatron" / "losses.py"
    spec = importlib.util.spec_from_file_location("megatron_distillation_losses_quant_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _ShapeOnlyKL:
        """Stands in for the vocab-parallel KL, which needs TP groups: these tests check the packing."""

        @staticmethod
        def apply(student_logits, *_args):
            return (torch.zeros(student_logits.shape[:2]),) * 6

    monkeypatch.setattr(module, "_VocabParallelKLDivergence", _ShapeOnlyKL)
    return module


def _nested(seqlens, *dense_shape, dtype=torch.float32):
    return torch.nested.as_nested_tensor(
        [torch.zeros(n, *dense_shape, dtype=dtype) for n in seqlens], layout=torch.jagged
    )


@pytest.mark.parametrize(
    "seqlen, quant_align_size, rows",
    [
        (16, 32, 256),  # mxfp8: eight 16-token sequences, each padded to 32 rows: 256 rows against 128 unpadded
        (17, 16, 256),  # the gap predates MXFP8: under the 16-token FP8 recipes, 17 tokens pad to 32 rows
    ],
)
def test_distillation_topk_thd_packs_teacher_with_the_student_padding(monkeypatch, seqlen, quant_align_size, rows):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    losses = _load_megatron_distillation_losses(monkeypatch, mcore_util)
    seqlens = [seqlen] * 8

    student_ids, *_ = mcore_util.preprocess_thd_engine(
        _nested(seqlens, dtype=torch.long), pre_process=True, quant_align_size=quant_align_size
    )
    assert student_ids.shape[:2] == (1, rows)
    student_logits = torch.zeros(1, rows, 4)
    teacher_log_probs, teacher_ids = _nested(seqlens, 2), _nested(seqlens, 2, dtype=torch.long)

    outputs = losses.compute_forward_kl_topk(
        student_logits,
        teacher_log_probs,
        teacher_ids,
        _DISTILLATION_CONFIG,
        "thd",
        quant_align_size=quant_align_size,
    )
    assert outputs["distillation_losses"].shape == (1, rows)

    # without the student's padding the teacher tensors have 8 * seqlen rows
    with pytest.raises(AssertionError):
        losses.compute_forward_kl_topk(student_logits, teacher_log_probs, teacher_ids, _DISTILLATION_CONFIG, "thd")


def test_distillation_topk_bshd_packs_teacher_with_the_student_padding(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    losses = _load_megatron_distillation_losses(monkeypatch, mcore_util)
    seqlens = [17] * 8

    student_ids, *_ = mcore_util.preprocess_bshd_engine(
        _nested(seqlens, dtype=torch.long), pre_process=True, use_fp8_padding=True
    )
    assert student_ids.shape[:2] == (8, 32)
    student_logits = torch.zeros(8, 32, 4)
    teacher_log_probs, teacher_ids = _nested(seqlens, 2), _nested(seqlens, 2, dtype=torch.long)

    outputs = losses.compute_forward_kl_topk(
        student_logits, teacher_log_probs, teacher_ids, _DISTILLATION_CONFIG, "bshd", quant_align_size=32
    )
    assert outputs["distillation_losses"].shape == (8, 32)

    with pytest.raises(AssertionError):
        losses.compute_forward_kl_topk(student_logits, teacher_log_probs, teacher_ids, _DISTILLATION_CONFIG, "bshd")


@pytest.mark.parametrize("strategy", ["megatron", "fsdp"])
def test_compute_topk_loss_passes_the_student_packing_to_the_megatron_loss_only(monkeypatch, strategy):
    import verl.trainer.distillation as distillation_pkg
    import verl.trainer.distillation.losses as distillation_losses

    received = []

    def fake_loss(*, student_logits, teacher_topk_log_probs, teacher_topk_ids, config, data_format, **kwargs):
        received.append(kwargs)
        return {"distillation_losses": torch.zeros(student_logits.shape[:2])}

    package = types.ModuleType(f"verl.trainer.distillation.{strategy}")
    package.losses = types.ModuleType(f"verl.trainer.distillation.{strategy}.losses")
    package.losses.compute_forward_kl_topk = fake_loss
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.losses.__name__, package.losses)
    monkeypatch.setattr(distillation_pkg, strategy, package, raising=False)

    distillation_losses.compute_topk_loss(
        SimpleNamespace(strategy=strategy),
        SimpleNamespace(distillation_loss=SimpleNamespace(loss_mode="forward_kl_topk")),
        {"teacher_logprobs": None, "teacher_ids": None},
        torch.zeros(1, 8, 4),
        "thd",
        quant_align_size=32,
        cp_layout="contiguous",
        local_cp_size=2,
    )
    packing = {"quant_align_size": 32, "cp_layout": "contiguous", "local_cp_size": 2}
    assert received == [packing if strategy == "megatron" else {}]


# The teacher's rows must hold the same tokens as the student's rows, not only the same number of rows: the tests below
# mark every token with its own number (0 is padding) and compare the rows the KL receives with the student's rows.


def _marked(seqlens, *dense_shape, dtype=torch.long):
    """Nested tensor in which every token holds its own number (1, 2, ... across the batch) in all its entries."""
    rows, start = [], 1
    for n in seqlens:
        tokens = torch.arange(start, start + n).to(dtype)
        rows.append(tokens.reshape(n, *[1] * len(dense_shape)).expand(n, *dense_shape).clone())
        start += n
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _teacher_rows_seen_by_the_kl(monkeypatch, losses):
    seen = {}

    class _RecordingKL:
        @staticmethod
        def apply(student_logits, teacher_log_probs, teacher_ids, *_args):
            seen["log_probs"], seen["ids"] = teacher_log_probs, teacher_ids
            return (torch.zeros(student_logits.shape[:2]),) * 6

    monkeypatch.setattr(losses, "_VocabParallelKLDivergence", _RecordingKL)
    return seen


def _pack_student_and_teacher(monkeypatch, mcore_util, seqlens, **packing):
    """Packs the student's tokens and runs the top-k loss with the same packing; returns the student rows and what the
    KL received."""
    losses = _load_megatron_distillation_losses(monkeypatch, mcore_util)
    seen = _teacher_rows_seen_by_the_kl(monkeypatch, losses)
    student_tokens, *_ = mcore_util.preprocess_thd_engine(_marked(seqlens), pre_process=True, **packing)
    teacher_log_probs, teacher_ids = _marked(seqlens, 2, dtype=torch.float32), _marked(seqlens, 2)
    student_logits = torch.zeros(*student_tokens.shape, 4)
    losses.compute_forward_kl_topk(
        student_logits, teacher_log_probs, teacher_ids, _DISTILLATION_CONFIG, "thd", **packing
    )
    return student_tokens, seen


@pytest.mark.parametrize("seqlen", [17, 32])
@pytest.mark.parametrize("quant_align_size", [None, 16, 32])
@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("cp_layout", ["zigzag", "contiguous"])
def test_distillation_topk_thd_teacher_rows_hold_the_student_tokens(
    monkeypatch, cp_layout, cp_rank, quant_align_size, seqlen
):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2, cp_rank=cp_rank)
    student_tokens, seen = _pack_student_and_teacher(
        monkeypatch, mcore_util, [seqlen] * 16, quant_align_size=quant_align_size, cp_layout=cp_layout
    )
    for k in range(2):
        assert torch.equal(seen["ids"][..., k], student_tokens)
        assert torch.equal(seen["log_probs"][..., k], student_tokens.to(torch.float32))


def test_distillation_topk_thd_teacher_split_zigzag_against_a_contiguous_student_misaligns_rows(monkeypatch):
    """What the CP layout prevents: the student split contiguously (dsv4_hybrid), the teacher with the zigzag default.
    The shapes agree, so the assertion in the loss passes, but most rows belong to other tokens."""
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2, cp_rank=0)
    seqlens = [17] * 16
    student_tokens, *_ = mcore_util.preprocess_thd_engine(
        _marked(seqlens), pre_process=True, quant_align_size=32, cp_layout="contiguous"
    )
    _, seen = _pack_student_and_teacher(monkeypatch, mcore_util, seqlens, quant_align_size=32, cp_layout="zigzag")
    teacher_tokens = seen["ids"][..., 0]
    valid = student_tokens != 0
    assert teacher_tokens.shape == student_tokens.shape == (1, 256)
    assert int(valid.sum()) == 136
    assert int((teacher_tokens[valid] != student_tokens[valid]).sum()) == 128


@pytest.mark.parametrize("group_rank", [0, 1])
@pytest.mark.parametrize("cp_layout", ["zigzag", "contiguous"])
def test_distillation_topk_thd_teacher_rows_follow_the_dynamic_cp_group(monkeypatch, cp_layout, group_rank):
    """Dynamic CP packs a micro-batch over a group of local_cp_size ranks (2 here), not the whole CP group (4)."""
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=4, cp_rank=3)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: group_rank)
    seqlens = [17] * 16
    student_tokens, seen = _pack_student_and_teacher(
        monkeypatch, mcore_util, seqlens, quant_align_size=32, cp_layout=cp_layout, local_cp_size=2
    )
    assert torch.equal(seen["ids"][..., 0], student_tokens)

    # packed over the whole CP group instead, the teacher rows hold other tokens
    _, seen_static = _pack_student_and_teacher(
        monkeypatch, mcore_util, seqlens, quant_align_size=32, cp_layout=cp_layout
    )
    assert not torch.equal(seen_static["ids"][..., 0], student_tokens)
