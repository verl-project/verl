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

It also covers the FSDP engine's ``response_only_lm_head`` option on packed
micro-batches, with and without that padding.

The ``veomni.*`` imports that ``transformer_impl`` performs at module
load are stubbed with ``MagicMock`` the same way
``test_router_replay_engine_helpers_on_cpu.py`` does, so this runs on the
plain CPU unit-test image.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict
from transformers import Qwen3Config, Qwen3ForCausalLM

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


import verl.utils.experimental.torch_functional as experimental_F  # noqa: E402
import verl.utils.torch_functional as verl_F  # noqa: E402
from verl.models.transformers.dense_common import _keep_rows, forward_with_torch_backend  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.utils.dataset.dataset_utils import DatasetPadMode  # noqa: E402
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead  # noqa: E402
from verl.workers.engine.utils import pad_packed_inputs  # noqa: E402
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead  # noqa: E402
from verl.workers.utils.padding import no_padding_2_padding  # noqa: E402

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


class TestResponseOnlyLmHead:
    """``response_only_lm_head`` projects only the response rows and returns the same values."""

    PROMPTS = [[5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]
    RESPONSES = [[20, 21], [22], [23, 24, 25]]

    @staticmethod
    def _jagged(rows):
        return torch.nested.as_nested_tensor([torch.tensor(row, dtype=torch.long) for row in rows], layout=torch.jagged)

    def _micro_batch(self, nested_prompts=True, responses=RESPONSES, **flags):
        sequences = [p + r for p, r in zip(self.PROMPTS, responses, strict=True)]
        fields = {
            "input_ids": self._jagged(sequences),
            "position_ids": self._jagged([list(range(len(s))) for s in sequences]),
        }
        if nested_prompts:
            fields.update(prompts=self._jagged(self.PROMPTS), responses=self._jagged(responses))
        else:  # left-padded prompts and right-padded responses, with an attention mask
            pw, rw = max(map(len, self.PROMPTS)), max(map(len, responses))
            fields["prompts"] = torch.tensor([[0] * (pw - len(p)) + p for p in self.PROMPTS])
            fields["responses"] = torch.tensor([r + [0] * (rw - len(r)) for r in responses])
            fields["attention_mask"] = torch.tensor(
                [
                    [0] * (pw - len(p)) + [1] * len(p + r) + [0] * (rw - len(r))
                    for p, r in zip(self.PROMPTS, responses, strict=True)
                ]
            )
        micro_batch = TensorDict(fields, batch_size=[])
        defaults = {"use_remove_padding": True, "pad_mode": DatasetPadMode.NO_PADDING, "temperature": 0.7}
        tu.assign_non_tensor(micro_batch, **{**defaults, "calculate_entropy": True, **flags})
        return micro_batch

    @staticmethod
    def _engine(response_only_lm_head, model=None, **attributes):
        """Only what prepare_model_inputs/outputs read; a real engine needs a process group."""
        engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
        engine.engine_config = SimpleNamespace(
            response_only_lm_head=response_only_lm_head,
            entropy_checkpointing=False,
            entropy_from_logits_with_chunking=False,
        )
        engine.module, engine.compute_entropy_from_logits = model, verl_F.entropy_from_logits
        engine.pad_to_length, engine.use_ulysses_sp, engine.forward_accepts_logits_to_keep = False, False, True
        engine.__dict__.update(attributes)
        return engine

    @staticmethod
    def _model(fused):
        """A tiny Qwen3; the same weights on every call."""
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        )  # fmt: skip
        model = Qwen3ForCausalLM(config).double()
        if fused:  # the forward that apply_monkey_patch installs for use_fused_kernels
            model.forward = forward_with_torch_backend.__get__(model)
        return model

    @pytest.mark.parametrize("nested_prompts", [True, False], ids=["nested", "padded"])
    @pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
    def test_matches_the_full_lm_head(self, nested_prompts, fused, monkeypatch):
        monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)  # CUDA-only
        projected_rows, results = {}, {}
        for response_only in (False, True):
            model = self._model(fused)
            micro_batch = self._micro_batch(nested_prompts, use_fused_kernels=fused, calculate_sum_pi_squared=not fused)
            # pad_to_length pads the 16 packed tokens to 24, so the outputs are also scattered into padding rows.
            engine = self._engine(response_only, model, pad_to_length=True, pad_to_length_bucket=12)
            model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
            output = model(**model_inputs, use_cache=False)
            projected_rows[response_only] = (output.log_probs if fused else output.logits).shape[1]
            outputs = engine.prepare_model_outputs(output, output_args, micro_batch, logits_processor_func=None)
            responses = {name: no_padding_2_padding(value, micro_batch) for name, value in outputs.items()}
            sum(value.sum() for value in responses.values()).backward()
            results[response_only] = responses, [parameter.grad for parameter in model.parameters()]
        assert projected_rows == {False: 24, True: sum(map(len, self.RESPONSES))}
        torch.testing.assert_close(results[True], results[False])

    @pytest.mark.parametrize(
        "flags, attributes",
        [
            ({"use_remove_padding": False}, {}),
            ({}, {"use_ulysses_sp": True}),
            ({}, {"engine_config": SimpleNamespace(response_only_lm_head=True, strategy="fsdp_turbo")}),
            ({"distillation_use_topk": True}, {}),
            ({"score_centering": True}, {}),
            ({}, {"forward_accepts_logits_to_keep": False}),
        ],
        ids=[
            "padded_input",
            "ulysses_sp",
            "fsdp_turbo",
            "topk_distillation",
            "score_centering",
            "forward_without_logits_to_keep",
        ],
    )
    def test_unsupported_setups_are_refused(self, flags, attributes):
        with pytest.raises(ValueError, match="response_only_lm_head is not supported"):
            self._engine(True, **attributes).prepare_model_inputs(self._micro_batch(**flags))

    def test_micro_batch_without_prompts_and_responses_is_refused(self):
        """SFT micro-batches carry input_ids and loss_mask, without the prompt/response split the rows come from."""
        micro_batch = self._micro_batch().exclude("prompts", "responses")
        with pytest.raises(ValueError, match="without prompts and responses"):
            self._engine(True).prepare_model_inputs(micro_batch)

    @pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
    def test_micro_batch_without_response_tokens_projects_every_row(self, fused, monkeypatch):
        """With no response rows to keep, the full LM head runs, as with the option off."""
        monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)  # CUDA-only
        model = self._model(fused)
        results = {}
        for response_only in (False, True):
            micro_batch = self._micro_batch(responses=[[], [], []], use_fused_kernels=fused)
            engine = self._engine(response_only, model)
            model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
            assert "logits_to_keep" not in model_inputs
            output = model(**model_inputs, use_cache=False)
            outputs = engine.prepare_model_outputs(output, output_args, micro_batch, logits_processor_func=None)
            results[response_only] = {name: value.values() for name, value in outputs.items()}
        torch.testing.assert_close(results[True], results[False])

    def test_one_response_row_from_a_flat_fused_output(self):
        """The Triton fused kernel returns flat (n,) outputs, so a single kept row arrives as shape (1,)."""
        micro_batch = self._micro_batch(responses=[[20], [], []], use_fused_kernels=True)
        engine = self._engine(True)
        model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
        assert model_inputs["logits_to_keep"].tolist() == [2]  # the row that predicts token 20
        output = SimpleNamespace(log_probs=torch.tensor([-0.5]), entropy=torch.tensor([0.25]))
        outputs = engine.prepare_model_outputs(output, output_args, micro_batch, logits_processor_func=None)
        expected = torch.zeros(11)  # 4 + 5 + 2 packed tokens
        assert torch.equal(outputs["log_probs"].values(), expected.index_fill(0, torch.tensor([2]), -0.5))
        assert torch.equal(outputs["entropy"].values(), expected.index_fill(0, torch.tensor([2]), 0.25))

    def test_logits_to_keep_zero_leaves_the_fused_inputs_unchanged(self):
        """The fused forwards' default; their callers may pass 1-D shift_labels, which the Triton kernel accepts."""
        hidden_states, shift_labels = torch.randn(1, 5, 4), torch.arange(5)
        kept_hidden_states, kept_labels = _keep_rows(hidden_states, shift_labels, 0)
        assert kept_hidden_states is hidden_states and kept_labels is shift_labels
