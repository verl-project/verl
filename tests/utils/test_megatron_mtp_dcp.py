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

import pytest
import torch

pytest.importorskip("megatron")

import verl.models.mcore.mtp_patch as mtp_patch
import verl.utils.megatron_utils as megatron_utils


class _OutputLayer(torch.nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return hidden_states, None


def _model(**overrides):
    config = {
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.1,
        "calculate_per_token_loss": True,
        "use_mup": False,
    }
    config.update(overrides)
    return SimpleNamespace(
        share_embeddings_and_output_weights=False,
        post_process=True,
        config=SimpleNamespace(**config),
        output_layer=_OutputLayer(),
        training=False,
        compute_language_model_loss=lambda labels, _logits: torch.ones_like(labels, dtype=torch.float32),
    )


def _postprocess(model, packed_seq_params):
    return mtp_patch._megatron_gptmodel_postprocess(
        model,
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(2, 1, 4),
        input_ids=torch.arange(4).reshape(1, 4),
        position_ids=None,
        labels=torch.arange(4).reshape(1, 4),
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        loss_mask=torch.ones(1, 4),
        packed_seq_params=packed_seq_params,
    )


def test_process_mtp_loss_uses_dynamic_group_and_token_scale(monkeypatch):
    dynamic_group = object()
    model = _model()
    model.cp_group = object()
    packed = SimpleNamespace(
        cp_group=dynamic_group,
        _verl_mtp_loss_normalization_factor=4107 / 4081,
    )
    captured = {}

    monkeypatch.setattr(mtp_patch, "_HAS_PROCESS_MTP_LOSS", True)
    monkeypatch.setattr(
        mtp_patch,
        "_PROCESS_MTP_LOSS_PARAMS",
        {"hidden_states", "config", "cp_group", "packed_seq_params"},
    )

    def fake_process(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"][:1]

    monkeypatch.setattr(mtp_patch, "_process_mtp_loss", fake_process, raising=False)
    _postprocess(model, packed)

    assert captured["cp_group"] is dynamic_group
    assert captured["packed_seq_params"] is packed
    assert captured["config"].mtp_loss_scaling_factor == pytest.approx(0.1 * 4107 / 4081)
    assert model.config.mtp_loss_scaling_factor == 0.1


def test_detached_embeddings_roll_with_dynamic_group(monkeypatch):
    dynamic_group = object()
    calls = []

    def fake_roll(tensor, *, cp_group, **kwargs):
        calls.append((cp_group, kwargs.get("fill_value")))
        return tensor, tensor.sum()

    monkeypatch.setattr(mtp_patch, "roll_tensor", fake_roll)
    monkeypatch.setattr(mtp_patch, "_ROLL_TENSOR_HAS_FILL_VALUE", True)
    layer = SimpleNamespace(cp_group=object(), _get_embeddings_has_padding_mask=True)
    mtp_patch._patched_get_embeddings_for_detach(
        layer,
        input_ids=torch.arange(4).reshape(1, 4),
        position_ids=torch.arange(4).reshape(1, 4),
        embedding=lambda input_ids, position_ids: torch.stack([input_ids, position_ids], dim=-1).float(),
        hidden_states=torch.ones(4, 1, 2),
        packed_seq_params=SimpleNamespace(cp_group=dynamic_group),
        padding_mask=torch.zeros(1, 4, dtype=torch.bool),
    )

    assert calls == [
        (dynamic_group, None),
        (dynamic_group, None),
        (dynamic_group, True),
    ]


@pytest.mark.parametrize(
    ("tracker", "expected_scale"),
    [
        ({"loss_sums": object(), "calculate_per_token_loss": True}, 1.0),
        ({"loss_values": object()}, 0.25),
    ],
)
def test_mtp_metric_scale_matches_mcore_tracker(monkeypatch, tracker, expected_scale):
    captured = {}

    class _LoggingHelper:
        @staticmethod
        def track_mtp_metrics(*, loss_scale, total_loss_dict, **_kwargs):
            captured["scale"] = loss_scale
            total_loss_dict["mtp_1 loss"] = torch.tensor(loss_scale)

    _LoggingHelper.tracker = tracker
    monkeypatch.setattr(megatron_utils, "MTPLossLoggingHelper", _LoggingHelper)

    metrics = megatron_utils.get_megatron_mtp_loss(n_micro_batch=4)
    assert captured["scale"] == expected_scale
    assert metrics == {"mtp_losses/mtp_1_loss": expected_scale}
