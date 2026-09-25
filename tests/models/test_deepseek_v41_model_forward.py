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

from verl.models.mcore import model_forward, registry


def test_deepseek_v41_bshd_uses_padding_mask_instead_of_attention_mask(monkeypatch):
    valid_tokens = torch.tensor([[True, True, False], [True, True, True]])
    dense_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    position_ids = torch.arange(3).expand(2, -1)
    captured = {}

    monkeypatch.setattr(
        model_forward,
        "preprocess_bshd_engine",
        lambda *_args, **_kwargs: (dense_ids, valid_tokens, position_ids),
    )
    monkeypatch.setattr(model_forward, "postprocess_bshd_engine", lambda output, *_args, **_kwargs: output)

    class Model:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None, dsv4_version="v4.1", vision_config=SimpleNamespace(downsample_ratio=3))

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros(2, 3, 8)

    hf_config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
    )
    forward = registry.get_mcore_engine_forward_fn(hf_config)
    forward(Model(), dense_ids, {}, data_format="bshd")

    assert captured["attention_mask"] is None
    assert torch.equal(captured["padding_mask"], ~valid_tokens)


def test_deepseek_v41_bshd_converts_processor_tensors_to_native_images(monkeypatch):
    image_token_id = 129264
    dense_ids = torch.tensor(
        [
            [10] + [image_token_id] * 9 + [12],
            [20, 21] + [0] * 9,
        ]
    )
    valid_tokens = torch.tensor(
        [
            [True] * 11,
            [True, True] + [False] * 9,
        ]
    )
    position_ids = torch.arange(11).expand(2, -1)
    captured = {}

    monkeypatch.setattr(
        model_forward,
        "preprocess_bshd_engine",
        lambda *_args, **_kwargs: (dense_ids, valid_tokens, position_ids),
    )
    monkeypatch.setattr(model_forward, "postprocess_bshd_engine", lambda output, *_args, **_kwargs: output)

    class Model:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None, dsv4_version="v4.1", vision_config=SimpleNamespace(downsample_ratio=3))

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros(2, 11, 8)

    types = torch.full((2, 11), -1, dtype=torch.long)
    types[0, 1:5] = torch.tensor([0, 1, 2, 3])
    types[0, 5:10] = torch.tensor([0, 1, 1, 2, 3])
    patches = torch.arange(27 * 3 * 2 * 2, dtype=torch.bfloat16).reshape(27, 3, 2, 2)
    multi_modal_inputs = {
        "pixel_values": patches,
        "image_grid_hws": torch.tensor([[3, 3], [3, 6]]),
        "vision_token_types": types,
    }
    hf_config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        image_token_id=image_token_id,
        vision_config=SimpleNamespace(downsample_ratio=3),
    )
    forward = registry.get_mcore_engine_forward_fn(hf_config)
    forward(Model(), dense_ids, multi_modal_inputs, data_format="bshd")

    assert "pixel_values" not in captured
    assert len(captured["images"]) == 2
    assert captured["images"][1] == []
    first, second = captured["images"][0]
    assert (first.start, first.n_vit_h, first.n_vit_w) == (1, 3, 3)
    assert (second.start, second.n_vit_h, second.n_vit_w) == (5, 3, 6)
    torch.testing.assert_close(first.patches, patches[:9])
    torch.testing.assert_close(second.patches, patches[9:])
    torch.testing.assert_close(first.types, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(second.types, torch.tensor([0, 1, 1, 2, 3]))
    assert captured["attention_mask"] is None
    torch.testing.assert_close(captured["padding_mask"], ~valid_tokens)
