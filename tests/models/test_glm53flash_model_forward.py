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
"""Engine forward of VLMs that take the full packed THD stream (GLM-5.3-Flash)."""

from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore import model_forward


def _nested(*rows):
    return torch.nested.nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


class _LanguageModel:
    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(8, 1, 4), torch.ones(1, 4)


class _Wrapper:
    pre_process = True
    post_process = True
    config = SimpleNamespace(fp8=None)

    def __init__(self):
        self.language_model = _LanguageModel()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.language_model.forward(**{k: v for k, v in kwargs.items() if k != "attention_mask"})


@pytest.fixture
def full_stream_engine(monkeypatch):
    physical = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed = SimpleNamespace(
        cu_seqlens_q=physical,
        cu_seqlens_kv=physical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
    )
    full_ids = torch.tensor([[1, 2, 3, 0, 4, 5, 0, 0]])
    monkeypatch.setattr(
        model_forward,
        "preprocess_thd_engine",
        lambda *_args, **_kwargs: (torch.zeros(1, 4, dtype=torch.long), packed, None),
    )
    monkeypatch.setattr(model_forward, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    monkeypatch.setattr(model_forward, "preprocess_vlm_thd_engine", lambda *_args, **_kwargs: (full_ids, None, None))
    monkeypatch.setattr(model_forward, "postprocess_thd_engine", lambda output, *_args, **_kwargs: output)
    return full_ids


def test_full_stream_vlm_gets_unsharded_stream_and_skips_mtp_loss(full_stream_engine):
    model = _Wrapper()

    output = model_forward.gptmodel_forward_model_engine(
        model, _nested([1, 2, 3], [4, 5]), {}, vision_model=True, data_format="thd"
    )

    wrapper_kwargs = model.calls[-1]
    assert wrapper_kwargs["input_ids"] is full_stream_engine
    assert wrapper_kwargs["attention_mask"] is None
    assert wrapper_kwargs["position_ids"] is None
    assert model.language_model.calls[-1]["compute_mtp_loss"] is False
    assert "forward" not in vars(model.language_model)
    # The model's (output, CP-sliced loss mask) pair is reduced to the output.
    assert output.shape == (8, 1, 4)


def test_full_stream_vlm_trains_mtp_on_full_stream_supervision(full_stream_engine, monkeypatch):
    labels = _nested([2, 3, 9], [5, 9])
    loss_mask = _nested([1, 1, 0], [1, 0])
    monkeypatch.setattr(model_forward, "_convert_to_nested_tensor", lambda value, lengths: labels)
    monkeypatch.setattr(model_forward, "_build_mtp_loss_mask_nested", lambda value, lengths, mask: loss_mask)
    model = _Wrapper()

    model_forward.gptmodel_forward_model_engine(
        model,
        _nested([1, 2, 3], [4, 5]),
        {},
        vision_model=True,
        data_format="thd",
        mtp_enable_train=True,
        logits_processor_args={"label": object(), "loss_mask": object()},
    )

    language_model_kwargs = model.language_model.calls[-1]
    assert language_model_kwargs["compute_mtp_loss"] is True
    # Full-stream supervision, shifted by one token like the unsharded engine layout.
    torch.testing.assert_close(language_model_kwargs["labels"], torch.tensor([[3, 9, 0, 5, 9, 0, 0, 2]]))
    torch.testing.assert_close(language_model_kwargs["loss_mask"], torch.tensor([[1, 0, 0, 1, 0, 0, 0, 1]]))


def test_full_stream_vlm_rejects_bshd(monkeypatch):
    monkeypatch.setattr(model_forward, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    monkeypatch.setattr(
        model_forward,
        "preprocess_bshd_engine",
        lambda *_args, **_kwargs: (torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool), None),
    )
    with pytest.raises(ValueError, match="data_format='thd'"):
        model_forward.gptmodel_forward_model_engine(
            _Wrapper(), _nested([1, 2, 3], [4, 5]), {}, vision_model=True, data_format="bshd"
        )
