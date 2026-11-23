# Copyright 2026 Individual Contributor
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
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _engine(chunking=True, checkpointing=False, chunk_size=3):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        entropy_from_logits_with_chunking=chunking,
        entropy_from_logits_chunk_size=chunk_size,
        entropy_checkpointing=checkpointing,
        batch_invariant=False,
    )
    engine.compute_entropy_from_logits = (
        verl_F.entropy_from_logits_with_chunking if chunking else verl_F.entropy_from_logits
    )
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    return engine


@pytest.mark.parametrize("shape", [(11, 17), (1, 11, 17), (2, 11, 17)])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("chunking", [False, True])
def test_entropy_values_and_gradients(shape, checkpointing, chunking):
    torch.manual_seed(42)
    logits = torch.randn(shape, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    actual = _engine(chunking, checkpointing)._compute_entropy(logits)
    expected = verl_F.entropy_from_logits(reference)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_chunk_size_bounds_softmax_in_forward_and_recomputation(monkeypatch, checkpointing, packed):
    # Slicing the sequence dimension prevents flattening batch and sequence into
    # one view. Dense chunking must not introduce a full-vocabulary reshape copy.
    base = torch.randn(2, 15, 17, requires_grad=True)
    logits = base[:, 2:13, :]
    assert not logits.is_contiguous()
    if packed:
        logits = logits[0]
    calls = []
    original = torch.nn.functional.softmax

    def record_softmax(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        assert value.ndim == 2
        assert value.shape[0] <= 3
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softmax", record_softmax)
    entropy = _engine(checkpointing=checkpointing)._compute_entropy(logits)
    forward_calls = len(calls)
    entropy.sum().backward()
    assert forward_calls == (4 if packed else 8)
    assert len(calls) > forward_calls if checkpointing else len(calls) == forward_calls
    assert base.grad is not None and torch.isfinite(base.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_entropy_inference_preserves_input(dtype):
    logits = torch.randn(2, 11, 17, dtype=dtype)
    original = logits.clone()
    with torch.no_grad():
        actual = _engine()._compute_entropy(logits)
    torch.testing.assert_close(actual, verl_F.entropy_from_logits(logits.float()))
    torch.testing.assert_close(logits, original, rtol=0, atol=0)


@pytest.mark.parametrize("remove_padding", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_prepare_model_outputs_uses_chunking(remove_padding, checkpointing):
    lengths = [7, 5]
    input_ids = torch.nested.as_nested_tensor([torch.arange(n) % 17 for n in lengths], layout=torch.jagged)
    batch = TensorDict({"input_ids": input_ids}, batch_size=[2])
    tu.assign_non_tensor(
        batch,
        use_remove_padding=remove_padding,
        use_fused_kernels=False,
        calculate_entropy=True,
        # Isolate entropy postprocessing from the separate log-probability kernel.
        distillation_only=True,
    )
    shape = (1, sum(lengths), 17) if remove_padding else (2, max(lengths), 17)
    logits = torch.randn(shape, requires_grad=True)
    expected = verl_F.entropy_from_logits(logits.detach())
    engine = _engine(checkpointing=checkpointing)
    calls = []

    def entropy_fn(value, *, chunk_size):
        calls.append((tuple(value.shape), chunk_size))
        return verl_F.entropy_from_logits_with_chunking(value, chunk_size=chunk_size)

    engine.compute_entropy_from_logits = entropy_fn
    args = {
        "temperature": torch.ones(2),
        "temperature_rmpad": torch.ones(sum(lengths)),
        "temperature_is_one": True,
        "input_ids_rmpad_rolled": input_ids.values().roll(-1),
        "pad_size": 0,
    }
    result = engine.prepare_model_outputs(SimpleNamespace(logits=logits), args, batch, None)
    expected = expected[0] if remove_padding else torch.cat([expected[i, :n] for i, n in enumerate(lengths)])
    torch.testing.assert_close(result["entropy"].values(), expected)
    assert calls and all(len(shape) == 2 and size == 3 for shape, size in calls)
    result["entropy"].values().sum().backward()
    assert torch.isfinite(logits.grad).all()
    if not remove_padding:
        assert torch.count_nonzero(logits.grad[1, 5:]) == 0


@pytest.mark.parametrize("checkpointing", [False, True])
def test_compiled_entropy_matches_eager(checkpointing):
    engine = _engine(checkpointing=checkpointing)
    engine.compute_entropy_from_logits = torch.compile(
        engine.compute_entropy_from_logits, backend="aot_eager", dynamic=True
    )
    logits = torch.randn(2, 11, 17, requires_grad=True)
    expected_logits = logits.detach().clone().requires_grad_(True)
    actual = engine._compute_entropy(logits)
    expected = verl_F.entropy_from_logits(expected_logits)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, expected_logits.grad, atol=1e-6, rtol=1e-5)
