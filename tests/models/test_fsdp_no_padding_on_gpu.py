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
"""GPU regression coverage for issue #6278.

These tests exercise the FSDP language-model engine path with
``pad_mode=NO_PADDING`` and ``use_remove_padding=False``. The issue was that
the model attention mask was built from response-only ``loss_mask`` rows, then
used as a full prompt+response sequence mask.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("Requires CUDA", allow_module_level=True)


def _nested(rows: list[list[int]], device: str) -> torch.Tensor:
    return torch.nested.as_nested_tensor(
        [torch.tensor(row, device=device, dtype=torch.long) for row in rows],
        layout=torch.jagged,
    )


def _make_micro_batch(device: str):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    input_ids = _nested([[11, 12, 13, 14, 15], [21, 22, 23]], device)
    position_ids = _nested([[0, 1, 2, 3, 4], [0, 1, 2]], device)
    prompts = _nested([[11, 12, 13], [21, 22]], device)
    responses = _nested([[14, 15], [23]], device)

    # This is intentionally response-only. The old attention-mask path used this
    # tensor and produced masks of lengths [2, 1] instead of full sequence [5, 3].
    loss_mask = torch.nested.as_nested_tensor(
        [
            torch.ones(2, device=device, dtype=torch.int64),
            torch.ones(1, device=device, dtype=torch.int64),
        ],
        layout=torch.jagged,
    )

    micro_batch = TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "prompts": prompts,
            "responses": responses,
            "loss_mask": loss_mask,
            "temperature": torch.ones(2, device=device),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=False,
        use_fused_kernels=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=0,
        max_response_len=2,
        max_response_length=2,
    )
    return micro_batch


def _fsdp_engine_with_lm_head_cls():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="NPU not support router replay for now.", category=UserWarning)
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    return FSDPEngineWithLMHead


def _legacy_attention_mask_from_response_loss_mask(
    loss_mask: torch.Tensor, batch_size: int, max_seq_len: int
) -> torch.Tensor:
    """Mirror the pre-fix mask construction to make the regression explicit."""
    attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
    attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
    return torch.nested.to_padded_tensor(attention_mask, padding=0, output_size=(batch_size, max_seq_len))


def test_prepare_model_inputs_uses_full_sequence_attention_mask_on_gpu():
    device = "cuda"
    FSDPEngineWithLMHead = _fsdp_engine_with_lm_head_cls()
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    micro_batch = _make_micro_batch(device)

    model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
    legacy_attention_mask = _legacy_attention_mask_from_response_loss_mask(
        loss_mask=micro_batch["loss_mask"],
        batch_size=micro_batch.batch_size[0],
        max_seq_len=model_inputs["input_ids"].shape[1],
    )

    expected_attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    expected_legacy_attention_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        device=device,
        dtype=torch.int32,
    )

    torch.testing.assert_close(legacy_attention_mask, expected_legacy_attention_mask)
    assert not torch.equal(legacy_attention_mask, expected_attention_mask)
    torch.testing.assert_close(model_inputs["attention_mask"], expected_attention_mask)
    assert model_inputs["attention_mask"].device.type == "cuda"
    assert model_inputs["attention_mask"].dtype == torch.int32
    assert model_inputs["input_ids"].shape == (2, 5)
    assert output_args["input_ids_rmpad_rolled"].shape == (8,)


def test_prepare_model_outputs_can_be_sliced_back_to_response_shape_on_gpu():
    from verl.utils.torch_functional import logprobs_from_logits
    from verl.workers.utils.padding import no_padding_2_padding

    device = "cuda"
    FSDPEngineWithLMHead = _fsdp_engine_with_lm_head_cls()
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    micro_batch = _make_micro_batch(device)
    _, output_args = engine.prepare_model_inputs(micro_batch)

    torch.manual_seed(0)
    vocab_size = 32
    max_seq_len = 5
    logits = torch.randn(2, max_seq_len, vocab_size, device=device)

    model_output = engine.prepare_model_outputs(
        output=SimpleNamespace(logits=logits.clone()),
        output_args=output_args,
        micro_batch=micro_batch,
        logits_processor_func=None,
    )
    padded_log_probs = no_padding_2_padding(model_output["log_probs"], micro_batch)

    flat_logits = torch.cat([logits[0, :5], logits[1, :3]], dim=0)
    expected_full_log_probs = logprobs_from_logits(flat_logits, output_args["input_ids_rmpad_rolled"])
    expected = torch.stack(
        [
            expected_full_log_probs[2:4],
            F.pad(expected_full_log_probs[6:7], (0, 1)),
        ],
        dim=0,
    )

    assert model_output["log_probs"].is_nested
    assert padded_log_probs.shape == (2, 2)
    torch.testing.assert_close(padded_log_probs, expected)
