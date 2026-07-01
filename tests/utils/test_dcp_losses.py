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

from types import SimpleNamespace

import torch
from tensordict import TensorDict

import verl.models.mcore.model_forward_fused as fused_forward_module
from verl.trainer.ppo.core_algos import agg_loss
from verl.workers.config import CriticConfig
from verl.workers.engine.megatron.transformer_impl import _apply_dcp_local_token_mask_for_loss
from verl.workers.utils.losses import value_loss
from verl.workers.utils.padding import no_padding_2_padding


def _nested(parts):
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def test_engine_applies_dcp_local_mask_before_value_loss():
    data = TensorDict(
        {
            "values": torch.zeros(1, 3),
            "returns": torch.zeros(1, 3),
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "values": _nested([torch.tensor([0.0, 10.0, 100.0, 1000.0, 10000.0])]),
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)
    _loss, metrics = value_loss(CriticConfig(strategy="megatron", ppo_micro_batch_size_per_gpu=1), model_output, data)

    assert metrics["critic/vpred_mean"] == torch.tensor(505.0).item()
    assert "_dcp_local_token_mask" not in model_output


def test_engine_applies_dcp_local_mask_before_sft_loss_shift():
    data = TensorDict(
        {
            "loss_mask": _nested([torch.tensor([False, False, True, True, True])]),
        },
        batch_size=[1],
    )
    model_output = {
        "_dcp_local_token_mask": _nested([torch.tensor([False, True, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)

    shifted = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
    expected = torch.tensor([False, True, False, True, False])
    torch.testing.assert_close(shifted, expected)


def test_engine_compacts_dcp_local_mask_for_forward_only_sft_loss():
    data = TensorDict(
        {
            "loss_mask": _nested([torch.tensor([True, False, True, False, False, True])]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": _nested([torch.tensor([1.0, 2.0, 3.0])]),
        "_dcp_local_token_mask": _nested([torch.tensor([True, True, False, False, True, False])]),
    }

    _apply_dcp_local_token_mask_for_loss(model_output, data)

    assert data["loss_mask"].values().numel() == model_output["log_probs"].values().numel()
    shifted = torch.roll(data["loss_mask"].values(), shifts=-1, dims=0)
    expected = torch.tensor([False, True, True])
    torch.testing.assert_close(shifted, expected)


def test_dcp_no_padding_uses_response_mask_for_response_span():
    data = TensorDict(
        {
            "response_mask": torch.ones(1, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[False, False, False, True, False]]),
        },
        batch_size=[1],
    )
    model_output = _nested([torch.tensor([0.0, 10.0, 100.0, 1000.0, 10000.0])])

    padded = no_padding_2_padding(model_output, data)

    torch.testing.assert_close(padded, torch.tensor([[10.0, 100.0, 1000.0]]))


def test_dcp_no_padding_preserves_internal_zero_response_span():
    data = TensorDict(
        {
            "_dcp_response_lengths": torch.tensor([5]),
            "response_mask": torch.tensor([[True, False, True, False, True]]),
        },
        batch_size=[1],
    )
    model_output = _nested([torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])])

    padded = no_padding_2_padding(model_output, data)

    torch.testing.assert_close(padded, torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]]))


def test_sequence_token_counts_reconstruct_global_sequence_mean():
    losses = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    full_mask = torch.ones_like(losses, dtype=torch.bool)
    rank0_mask = torch.tensor([[True, False, False, True]])
    rank1_mask = torch.tensor([[False, True, True, False]])
    token_counts = torch.tensor([4])

    full_loss = agg_loss(losses, full_mask, "seq-mean-token-mean", global_batch_size=1)
    rank0_loss = agg_loss(
        losses,
        rank0_mask,
        "seq-mean-token-mean",
        global_batch_size=1,
        sequence_token_counts=token_counts,
    )
    rank1_loss = agg_loss(
        losses,
        rank1_mask,
        "seq-mean-token-mean",
        global_batch_size=1,
        sequence_token_counts=token_counts,
    )

    torch.testing.assert_close(rank0_loss + rank1_loss, full_loss)


def test_fused_forward_propagates_dynamic_cp_metadata(monkeypatch):
    packed_seq_params = SimpleNamespace()
    preprocess_local_cp_sizes = []
    postprocess_calls = []

    def fake_preprocess(value, *, pre_process, need_roll=False, use_fp8_padding=False, local_cp_size=None):
        preprocess_local_cp_sizes.append(local_cp_size)
        return torch.tensor([[1, 2]], dtype=value.dtype), packed_seq_params, None

    def fake_local_postprocess(
        output,
        _packed_seq_params,
        _input_ids,
        _batch_size,
        *,
        post_process,
        local_cp_size,
        compact,
    ):
        postprocess_calls.append((local_cp_size, compact))
        return _nested([output.reshape(-1)])

    monkeypatch.setattr(fused_forward_module, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(fused_forward_module, "postprocess_thd_engine_local", fake_local_postprocess)
    monkeypatch.setattr(
        fused_forward_module,
        "build_thd_local_token_indices",
        lambda *_args, **_kwargs: _nested([torch.tensor([0, 1])]),
    )
    monkeypatch.setattr(
        fused_forward_module,
        "build_thd_full_seq_lens",
        lambda *_args, **_kwargs: _nested([torch.tensor([2])]),
    )
    monkeypatch.setattr(
        fused_forward_module,
        "build_thd_local_token_mask",
        lambda *_args, **_kwargs: _nested([torch.tensor([True, True])]),
    )

    class FakeModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                log_probs=torch.tensor([[1.0, 2.0]]),
                entropy=torch.tensor([[3.0, 4.0]]),
            )

    forward = fused_forward_module.fused_forward_model_engine()
    output = forward(
        model=FakeModel(),
        input_ids=_nested([torch.tensor([1, 2])]),
        labels=_nested([torch.tensor([1, 2])]),
        multi_modal_inputs={},
        temperature=1.0,
        calculate_entropy=True,
        pad_token_id=0,
        local_cp_size=2,
        return_dcp_local_token_mask=True,
        dcp_local_output_only=True,
        dcp_compact_output_only=True,
    )

    assert preprocess_local_cp_sizes == [2, 2]
    assert postprocess_calls == [(2, True), (2, True)]
    assert set(output) == {
        "log_probs",
        "entropy",
        "_dcp_local_token_indices",
        "_dcp_full_seq_lens",
        "_dcp_local_token_mask",
    }
