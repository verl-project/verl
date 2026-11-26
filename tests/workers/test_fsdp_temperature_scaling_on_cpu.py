# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0(the "License");
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
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp import transformer_impl as fsdp
from verl.workers.engine.fsdp.transformer_impl import (
    _is_scalar_unit_temperature,
    _scale_logits_by_temperature,
)


@pytest.mark.parametrize("temperature", [1, 1.0])
def test_scalar_unit_temperature_is_detected(temperature):
    assert _is_scalar_unit_temperature(temperature) is True


@pytest.mark.parametrize("temperature", [0.7, 2.0, torch.tensor(1.0)])
def test_non_unit_or_tensor_temperature_uses_general_path(temperature):
    assert _is_scalar_unit_temperature(temperature) is False


def test_unit_temperature_preserves_logits_storage_and_gradient():
    base = torch.randn(1, 4, 8, requires_grad=True)
    logits_view = base.squeeze(0)

    scaled = _scale_logits_by_temperature(
        logits_view,
        torch.ones(4, 1),
        is_unit_temperature=True,
    )

    assert scaled is logits_view
    scaled.sum().backward()
    torch.testing.assert_close(base.grad, torch.ones_like(base))


def test_non_unit_temperature_is_out_of_place_and_has_correct_gradient():
    logits = torch.randn(4, 8, requires_grad=True)
    temperature = torch.full((4, 1), 0.5)

    scaled = _scale_logits_by_temperature(
        logits,
        temperature,
        is_unit_temperature=False,
    )

    assert scaled is not logits
    torch.testing.assert_close(scaled, logits * 2)
    scaled.sum().backward()
    torch.testing.assert_close(logits.grad, torch.full_like(logits, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,entropy,grad_enabled,uses_chunks",
    [
        ([137], False, True, True),
        ([137, 73], False, True, False),
        ([137], True, True, False),
        ([137], False, False, False),
    ],
)
def test_dense_fsdp_logprobs(dtype, lengths, entropy, grad_enabled, uses_chunks):
    generator = torch.Generator().manual_seed(42)
    vocab = 257
    base = torch.randn(len(lengths), 145, vocab, dtype=dtype, generator=generator)
    labels = torch.randint(vocab, (sum(lengths),), generator=generator)
    input_ids = torch.nested.nested_tensor(
        [torch.zeros(length, dtype=torch.long) for length in lengths], layout=torch.jagged
    )
    micro = TensorDict({"input_ids": input_ids}, batch_size=[len(lengths)])
    tu.assign_non_tensor(micro, use_remove_padding=False, calculate_entropy=entropy)
    args = {"temperature": torch.full((len(lengths),), 0.7), "input_ids_rmpad_rolled": labels}
    engine = SimpleNamespace(engine_config=SimpleNamespace(batch_invariant=False, entropy_checkpointing=False))

    def forward(chunk_size):
        engine.engine_config.log_prob_temperature_chunk_size = chunk_size
        logits = base.clone().requires_grad_(grad_enabled)
        with torch.set_grad_enabled(grad_enabled):
            output = fsdp.FSDPEngineWithLMHead.prepare_model_outputs(
                engine, SimpleNamespace(logits=logits), args, micro, None
            )
        outputs = {key: value.values() for key, value in output.items()}
        grad = None
        if grad_enabled:
            weights = torch.linspace(-1, 1, sum(lengths), dtype=dtype)
            grad = torch.autograd.grad((outputs["log_probs"] * weights).sum(), logits)[0]
        return outputs, grad

    baseline, baseline_grad = forward(0)
    with patch.object(
        fsdp, "_logprobs_with_chunked_temperature", wraps=fsdp._logprobs_with_chunked_temperature
    ) as chunk:
        actual, actual_grad = forward(64)
        assert chunk.call_count == int(uses_chunks)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    if grad_enabled:
        torch.testing.assert_close(actual_grad, baseline_grad, atol=0, rtol=0)
