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

import inspect
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch.utils._python_dispatch import TorchDispatchMode

from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as vf
from verl.workers.engine.fsdp import transformer_impl as impl


@pytest.fixture(autouse=True)
def cpu_logprob_backend(monkeypatch):
    # GPU hosts may have flash-attn installed even when this test uses CPU inputs.
    monkeypatch.setattr(vf, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(vf, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False)


def engine(batch_invariant=False):
    obj = object.__new__(impl.FSDPEngineWithLMHead)
    obj.engine_config = SimpleNamespace(
        entropy_checkpointing=False,
        entropy_from_logits_with_chunking=False,
        entropy_from_logits_chunk_size=3,
        batch_invariant=batch_invariant,
    )
    obj.compute_entropy_from_logits = vf.entropy_from_logits
    obj.ulysses_sequence_parallel_size = 1
    obj.use_ulysses_sp = False
    return obj


class NoVocabularyCat(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.cat.default:
            assert all(t.ndim <= 1 for t in args[0]), "must not concatenate vocabulary-sized logits"
        return func(*args, **(kwargs or {}))


def make_batch(lengths, **flags):
    ids = torch.nested.as_nested_tensor([torch.arange(n) % 13 for n in lengths], layout=torch.jagged)
    batch = TensorDict({"input_ids": ids}, batch_size=[len(lengths)])
    tu.assign_non_tensor(batch, use_remove_padding=False, use_fused_kernels=False, **flags)
    args = {
        "temperature": torch.ones(len(lengths)),
        "temperature_is_one": True,
        "input_ids_rmpad_rolled": ids.values().roll(-1),
    }
    return batch, args


@pytest.mark.parametrize("lengths,padded", [([7], 7), ([5], 9), ([7, 3, 5], 9), ([0, 5, 0], 7), ([0, 0], 3)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_logprobs_values_gradients_and_no_vocabulary_cat(lengths, padded, strided, dtype):
    torch.manual_seed(17)
    base = torch.randn(len(lengths), padded * 2, 26, dtype=dtype, requires_grad=True)
    reference = base.detach().clone().requires_grad_()
    logits = base[:, ::2, ::2] if strided else base[:, :padded, :13]
    ref_logits = reference[:, ::2, ::2] if strided else reference[:, :padded, :13]
    labels = torch.arange(sum(lengths)) % 13
    with NoVocabularyCat():
        actual = impl._logprobs_from_dense_logits(logits, labels, torch.tensor(lengths))
    packed = torch.cat([row[:n] for row, n in zip(ref_logits, lengths, strict=True)])
    expected = torch.log_softmax(packed, -1).gather(-1, labels[:, None]).squeeze(-1)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    tolerance = {"atol": 0.02, "rtol": 0.02} if dtype == torch.bfloat16 else {"atol": 1e-6, "rtol": 1e-5}
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(base.grad, reference.grad, **tolerance)


@pytest.mark.parametrize("entropy,sum_pi_squared", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_engine_shared_logits_losses_and_model_backward(
    monkeypatch, entropy, sum_pi_squared, temperature, checkpointing
):
    torch.manual_seed(5)
    batch, args = make_batch([7, 3], calculate_entropy=entropy, calculate_sum_pi_squared=sum_pi_squared)
    args.update(temperature=torch.tensor([temperature, 1.0]), temperature_is_one=temperature == 1.0)
    base = torch.randn(2, 7, 13, requires_grad=True)
    reference = base.detach().clone().requires_grad_()
    # Softmax backward saves its output: protect model-saved logits too, even
    # when neither entropy nor an auxiliary loss is requested by the engine.
    logits = base.softmax(-1)
    saved_logits = logits.detach().clone()
    calls = []
    original = impl.logprobs_from_logits

    def tracked(x, labels, **kwargs):
        calls.append(kwargs["inplace_backward"])
        assert x.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr() or temperature != 1.0
        return original(x, labels, **kwargs)

    monkeypatch.setattr(impl, "logprobs_from_logits", tracked)
    with NoVocabularyCat():
        obj = engine()
        obj.engine_config.entropy_checkpointing = checkpointing
        output = obj.prepare_model_outputs(SimpleNamespace(logits=logits), args, batch, None)
    scaled = reference.softmax(-1) / args["temperature"][:, None, None]
    packed = torch.cat([scaled[0], scaled[1, :3]])
    ref = {"log_probs": vf.logprobs_from_logits(packed, args["input_ids_rmpad_rolled"], inplace_backward=False)}
    if entropy:
        ref["entropy"] = vf.entropy_from_logits(packed)
    if sum_pi_squared:
        ref["sum_pi_squared"] = vf.calculate_sum_pi_squared_from_logits(packed)
    loss, ref_loss = 0, 0
    for key in ref:
        torch.testing.assert_close(output[key].values(), ref[key], atol=1e-6, rtol=1e-5)
        weights = torch.randn_like(ref[key])
        loss = loss + (output[key].values() * weights).sum()
        ref_loss = ref_loss + (ref[key] * weights).sum()
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(base.grad, reference.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits, saved_logits, atol=0, rtol=0)
    assert calls == [False, False]


@pytest.mark.parametrize("topk,centering", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("distillation_only", [False, True])
def test_packed_processor_contract_preserved(monkeypatch, topk, centering, distillation_only):
    batch, args = make_batch(
        [7, 3], distillation_use_topk=topk, score_centering=centering, distillation_only=distillation_only
    )
    logits = torch.randn(2, 7, 13, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    packed_ref = torch.cat([reference[0], reference[1, :3]])
    calls = []

    def processor(*, student_logits, data):
        calls.append(student_logits.shape)
        assert data is batch
        torch.testing.assert_close(student_logits.squeeze(0), packed_ref)
        return {"aux": student_logits.square().sum(-1)}

    def forbidden(*args, **kwargs):
        pytest.fail("packed logits consumers must retain their existing path")

    monkeypatch.setattr(impl, "_logprobs_from_dense_logits", forbidden)
    out = engine().prepare_model_outputs(SimpleNamespace(logits=logits), args, batch, processor)
    assert calls == [torch.Size([1, 10, 13])]
    assert ("log_probs" not in out) == distillation_only
    actual_loss = out["aux"].values().sum()
    expected_loss = packed_ref.square().sum()
    if not distillation_only:
        expected_lp = vf.logprobs_from_logits(packed_ref, args["input_ids_rmpad_rolled"], inplace_backward=False)
        torch.testing.assert_close(out["log_probs"].values(), expected_lp)
        actual_loss = actual_loss + out["log_probs"].values().sum()
        expected_loss = expected_loss + expected_lp.sum()
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_entropy_only_does_not_pack_logits_or_compute_logprobs(monkeypatch):
    batch, args = make_batch([7, 3], calculate_entropy=True, distillation_only=True)

    def forbidden(*args, **kwargs):
        pytest.fail("distillation_only must not compute logprobs")

    monkeypatch.setattr(impl, "_logprobs_from_dense_logits", forbidden)
    with NoVocabularyCat():
        result = engine().prepare_model_outputs(SimpleNamespace(logits=torch.randn(2, 7, 13)), args, batch, None)
    assert list(result) == ["entropy"]


@pytest.mark.skipif(
    "batch_invariant" not in inspect.signature(vf.logprobs_from_logits).parameters,
    reason="this source baseline predates batch_invariant",
)
def test_batch_invariant_uses_existing_sampler_formula(monkeypatch):
    batch, args = make_batch([7, 3], batch_invariant=True)
    args.update(temperature=torch.tensor([0.7, 1.2]), temperature_is_one=False)
    logits = torch.randn(2, 7, 13, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()

    def forbidden(*args, **kwargs):
        pytest.fail("batch-invariant sampler formula must keep its own path")

    monkeypatch.setattr(impl, "_logprobs_from_dense_logits", forbidden)
    out = (
        engine(batch_invariant=True)
        .prepare_model_outputs(SimpleNamespace(logits=logits), args, batch, None)["log_probs"]
        .values()
    )
    scaled = reference.float() / args["temperature"][:, None, None]
    packed = torch.cat([scaled[0], scaled[1, :3]])
    expected = packed.log_softmax(-1).gather(-1, args["input_ids_rmpad_rolled"][:, None]).squeeze(-1)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inference_preserves_dense_input(dtype):
    logits = torch.randn(2, 7, 13, dtype=dtype)
    original = logits.clone()
    labels = torch.arange(10) % 13
    with torch.no_grad(), NoVocabularyCat():
        actual = impl._logprobs_from_dense_logits(logits, labels, torch.tensor([7, 3]))
    packed = torch.cat([logits[0], logits[1, :3]])
    expected = vf.logprobs_from_logits(packed, labels, inplace_backward=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits, original, atol=0, rtol=0)
