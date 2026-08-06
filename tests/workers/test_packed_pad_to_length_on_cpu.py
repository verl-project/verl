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

"""Bucket-aligned packed-sequence padding (``pad_to_length``) for the FSDP and VeOmni engines.

Covers the three pieces that make a packed micro-batch shape-stable:

* ``pad_packed_inputs`` — the tensor-level right-pad, including the
  ``position_ids`` convention that turns the pad into its own varlen
  segment instead of an extension of the last real sequence.
* ``FSDPEngine._get_packed_pad_size`` — rounding the packed length up to
  the next bucket boundary. Lives on the FSDP base, so ``VeOmniEngine``
  inherits it.
* ``FSDPEngine._gather_and_unpad_packed`` — the no-SP branch that trims
  the pad back off before the outputs are re-jagged.

The ``veomni.*`` imports that ``transformer_impl`` performs at module
load are stubbed with ``MagicMock`` the same way
``test_router_replay_engine_helpers_on_cpu.py`` does, so this runs on the
plain CPU unit-test image.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

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


from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead  # noqa: E402
from verl.workers.engine.utils import pad_packed_inputs  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead  # noqa: E402

ENGINE_CLASSES = pytest.mark.parametrize(
    "engine_cls",
    [FSDPEngineWithLMHead, VeOmniEngineWithLMHead],
    ids=["fsdp", "veomni"],
)


def _make_engine(engine_cls, pad_to_length: bool, bucket: int = 1024):
    """Bare instance — ``_get_packed_pad_size`` only reads the two attributes set here,
    and a real ``__init__`` would need a process group and VeOmni's parallel_state."""
    engine = engine_cls.__new__(engine_cls)
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = bucket
    return engine


class TestPadPackedInputs:
    def test_pad_size_zero_is_a_noop(self):
        input_ids = torch.arange(6).unsqueeze(0)
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])

        padded_ids, padded_pos = pad_packed_inputs(input_ids, position_ids, 0)

        assert padded_ids is input_ids
        assert padded_pos is position_ids

    def test_pad_tokens_form_their_own_varlen_segment(self):
        input_ids = torch.arange(1, 7).unsqueeze(0)
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])

        padded_ids, padded_pos = pad_packed_inputs(input_ids, position_ids, 4)

        assert padded_ids.tolist() == [[1, 2, 3, 4, 5, 6, 0, 0, 0, 0]]
        # position_ids restart at 0, so cu_seqlens sees one trailing segment of
        # length 4 rather than a 3-token sequence that grew to 7.
        assert padded_pos.tolist() == [[0, 1, 2, 0, 1, 2, 0, 1, 2, 3]]

    def test_mrope_position_ids_are_padded_per_rope_dim(self):
        input_ids = torch.arange(3).unsqueeze(0)
        position_ids = torch.arange(3).view(1, 1, 3).repeat(4, 1, 1)  # (rope_dim, 1, total_nnz)

        padded_ids, padded_pos = pad_packed_inputs(input_ids, position_ids, 2)

        assert padded_ids.shape == (1, 5)
        assert padded_pos.shape == (4, 1, 5)
        for rope_dim in range(4):
            assert padded_pos[rope_dim, 0].tolist() == [0, 1, 2, 0, 1]

    def test_pad_value_and_dtype_are_preserved(self):
        """Temperature divides the logits, so its pad has to be 1 rather than the default 0."""
        temperature = torch.full((1, 3), 0.7)

        padded, _ = pad_packed_inputs(temperature, None, 2, pad_value=1)

        assert padded.dtype == torch.float32
        assert torch.equal(padded, torch.tensor([[0.7, 0.7, 0.7, 1.0, 1.0]]))


@ENGINE_CLASSES
class TestGetPackedPadSize:
    """``VeOmniEngine`` subclasses ``FSDPEngine`` without going through its ``__init__``,
    so both engines are checked against the same inherited helper."""

    def test_disabled_returns_zero(self, engine_cls):
        engine = _make_engine(engine_cls, pad_to_length=False)

        assert engine._get_packed_pad_size(700) == 0

    def test_rounds_up_to_the_next_bucket(self, engine_cls):
        engine = _make_engine(engine_cls, pad_to_length=True, bucket=1024)

        assert engine._get_packed_pad_size(700) == 324

    def test_exact_multiple_needs_no_pad(self, engine_cls):
        engine = _make_engine(engine_cls, pad_to_length=True, bucket=1024)

        assert engine._get_packed_pad_size(2048) == 0

    def test_micro_batch_over_the_token_budget_is_still_padded(self, engine_cls):
        """The reason for bucketing rather than a single target length: ``rearrange_micro_batches``
        sizes the micro-batch count from the token budget but partitions samples to balance
        attention workload, so a skewed micro-batch can exceed the budget. It still lands on a
        bucket boundary instead of keeping an arbitrary shape."""
        engine = _make_engine(engine_cls, pad_to_length=True, bucket=1024)

        assert engine._get_packed_pad_size(12000) == 12288 - 12000

    def test_padded_length_is_always_a_bucket_multiple(self, engine_cls):
        engine = _make_engine(engine_cls, pad_to_length=True, bucket=640)

        for packed in (1, 639, 640, 641, 10239, 10240, 12000):
            assert (packed + engine._get_packed_pad_size(packed)) % 640 == 0


@ENGINE_CLASSES
class TestDistillationGuard:
    def test_topk_distillation_micro_batch_is_rejected(self, engine_cls):
        engine = _make_engine(engine_cls, pad_to_length=True)
        micro_batch = TensorDict({}, batch_size=[])
        micro_batch.set_non_tensor("distillation_use_topk", True)

        with pytest.raises(RuntimeError, match="not supported with top-K distillation"):
            engine.prepare_model_inputs(micro_batch)


class TestGatherAndUnpadPacked:
    """Without Ulysses SP there is nothing to all-gather, so the helper degenerates to
    trimming the suffix that ``prepare_model_inputs`` appended."""

    @pytest.fixture
    def engine(self):
        engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
        engine.use_ulysses_sp = False
        return engine

    def test_trims_the_pad_suffix(self, engine):
        tensor = torch.arange(10)

        assert engine._gather_and_unpad_packed(tensor, 4).tolist() == [0, 1, 2, 3, 4, 5]

    def test_zero_pad_returns_the_input_unchanged(self, engine):
        tensor = torch.arange(10)

        assert engine._gather_and_unpad_packed(tensor, 0) is tensor
