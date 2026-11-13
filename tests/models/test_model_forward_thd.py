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

from verl.models.mcore import model_forward
from verl.models.mcore import util as mcore_util


@pytest.fixture
def cp2(monkeypatch):
    monkeypatch.setattr(mcore_util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_rank", lambda: 0)
    # The tests exercise packing and wrapper output handling without launching a process group.
    monkeypatch.setattr(model_forward, "postprocess_thd_engine", lambda output, *_args, **_kwargs: output)


@pytest.mark.parametrize("vision_model", [False, True])
def test_full_stream_glm_rejects_bshd_before_packing(monkeypatch, vision_model):
    model = SimpleNamespace(pre_process=True, post_process=True, config=SimpleNamespace(fp8=None))
    monkeypatch.setattr(model_forward, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    monkeypatch.setattr(
        model_forward,
        "preprocess_bshd_engine",
        lambda *args, **kwargs: pytest.fail("GLM BSHD must be rejected before packing"),
    )
    with pytest.raises(ValueError, match="GLM-5.3-Flash requires data_format='thd'"):
        model_forward.gptmodel_forward_model_engine(
            model, torch.zeros(1, 2, dtype=torch.long), {}, data_format="bshd", vision_model=vision_model
        )


def test_short_dsv41_cp_batch_aligns_logits_labels_and_teacher_payload(cp2):
    input_ids = torch.nested.nested_tensor([torch.arange(100)], layout=torch.jagged)
    temperatures = torch.nested.nested_tensor([torch.ones(100)], layout=torch.jagged)
    teacher_ids = torch.nested.nested_tensor([torch.ones(100, 2, dtype=torch.long)], layout=torch.jagged)
    teacher_logprobs = torch.nested.nested_tensor([torch.full((100, 2), -1.0)], layout=torch.jagged)

    class Model:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None, dsv4_version="v4.1", context_parallel_size=2, csa_window_size=128)

        def __call__(self, input_ids, **kwargs):
            assert input_ids.shape == (1, 128)
            padding_mask = kwargs["padding_mask"]
            assert padding_mask.dtype == torch.bool
            assert padding_mask.shape == input_ids.shape
            assert not padding_mask[0, :100].any()
            assert padding_mask[0, 100:].all()
            return torch.zeros(*input_ids.shape, 8)

    def process(logits, label, temperature, teacher_topk_ids, teacher_topk_logprobs):
        assert logits.shape[:2] == label.shape == temperature.shape == (1, 128)
        assert teacher_topk_ids.shape == teacher_topk_logprobs.shape == (1, 128, 2)
        torch.testing.assert_close(label[0, :100], torch.roll(torch.arange(100), shifts=-1))
        assert torch.isneginf(teacher_topk_logprobs[0, 100:]).all()
        return {"log_probs": logits[..., 0]}

    output = model_forward.gptmodel_forward_model_engine(
        Model(),
        input_ids,
        {},
        cp_layout="contiguous",
        logits_processor=process,
        logits_processor_args={
            "label": input_ids,
            "temperature": temperatures,
            "teacher_topk_ids": teacher_ids,
            "teacher_topk_logprobs": teacher_logprobs,
        },
    )
    assert output["log_probs"].shape == (1, 128)


@pytest.mark.parametrize("use_logits_processor", [False, True])
def test_full_stream_mtp_cp_discards_the_wrappers_returned_loss_mask(cp2, monkeypatch, use_logits_processor):
    input_ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3, 4])], layout=torch.jagged)
    loss_mask = torch.nested.nested_tensor([torch.tensor([0.0, 1.0, 1.0, 1.0])], layout=torch.jagged)
    logits = torch.zeros(1, 2, 8)
    monkeypatch.setattr(model_forward, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    monkeypatch.setattr(mcore_util, "takes_full_packed_thd_vlm_inputs", lambda model: True)

    class LanguageModel:
        def forward(self, *, compute_mtp_loss):
            assert compute_mtp_loss is True
            return logits

    class FullStreamModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.language_model = LanguageModel()

        def __call__(self, input_ids, labels, loss_mask, **kwargs):
            # Like GLM53FlashModel, accept the full stream and return a CP-local mask with the output.
            assert input_ids.shape == labels.shape == loss_mask.shape == (1, 4)
            return self.language_model.forward(), loss_mask[:, [0, 3]]

    def process(output, label):
        assert output is logits
        assert label.shape == logits.shape[:2]
        return {"logits": output}

    model = FullStreamModel()
    output = model_forward.gptmodel_forward_model_engine(
        model,
        input_ids,
        {},
        vision_model=True,
        mtp_enable_train=True,
        logits_processor=process if use_logits_processor else None,
        logits_processor_args={"label": input_ids, "loss_mask": loss_mask},
    )
    assert (output["logits"] if use_logits_processor else output) is logits
    assert "forward" not in vars(model.language_model)
