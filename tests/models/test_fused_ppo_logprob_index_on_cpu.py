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
"""Unit tests for response-only vocabulary projection in fused PPO kernels (issue #8649)."""

import contextlib
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from verl.models.transformers import dense_common, glm4v, qwen2_vl, qwen3_5, qwen3_vl
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
from verl.workers.utils.losses import ppo_loss, sft_loss
from verl.workers.utils.padding import (
    compute_response_logprob_index,
    left_right_2_no_padding,
    no_padding_2_padding,
    response_from_nested,
)

HIDDEN_SIZE = 16
VOCAB_SIZE = 64

ALL_ADAPTERS = [
    dense_common.forward_with_torch_backend,
    dense_common.forward_with_triton_backend,
    qwen3_5.forward_with_torch_backend,
    qwen3_5.forward_with_triton_backend,
    qwen3_vl.forward_with_torch_backend,
    qwen3_vl.forward_with_triton_backend,
    qwen2_vl.forward_with_torch_backend,
    qwen2_vl.forward_with_triton_backend,
    glm4v.forward_with_torch_backend,
    glm4v.forward_with_triton_backend,
]

TORCH_ADAPTERS = {
    dense_common.forward_with_torch_backend,
    qwen3_5.forward_with_torch_backend,
    qwen3_vl.forward_with_torch_backend,
    qwen2_vl.forward_with_torch_backend,
    glm4v.forward_with_torch_backend,
}


def _make_nested_ppo_micro_batch(
    prompt_lens: list[int],
    response_lens: list[int],
    vocab_size: int = VOCAB_SIZE,
    calculate_entropy: bool = True,
) -> TensorDict:
    torch.manual_seed(123)
    bsz = len(prompt_lens)
    max_resp = max(max(response_lens), 1)
    prompts = [torch.randint(0, vocab_size, (plen,)) for plen in prompt_lens]
    responses = [torch.randint(0, vocab_size, (rlen,)) for rlen in response_lens]
    seqs = [torch.cat([p, r], dim=0) for p, r in zip(prompts, responses, strict=True)]
    positions = [torch.arange(s.numel()) for s in seqs]
    resp_masks = [torch.ones(rlen, dtype=torch.long) for rlen in response_lens]
    old_lps = [-torch.rand(rlen, dtype=torch.float32) for rlen in response_lens]
    advantages = [torch.randn(rlen, dtype=torch.float32) for rlen in response_lens]

    td = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(seqs, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(positions, layout=torch.jagged),
            "prompts": torch.nested.as_nested_tensor(prompts, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(responses, layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor(resp_masks, layout=torch.jagged),
            "loss_mask": torch.nested.as_nested_tensor(resp_masks, layout=torch.jagged),
            "old_log_probs": torch.nested.as_nested_tensor(old_lps, layout=torch.jagged),
            "advantages": torch.nested.as_nested_tensor(advantages, layout=torch.jagged),
        },
        batch_size=[bsz],
    )
    tu.assign_non_tensor(
        td,
        use_remove_padding=True,
        use_fused_kernels=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        temperature=1.0,
        calculate_entropy=calculate_entropy,
        dp_size=1,
        batch_num_tokens=max(sum(response_lens), 1),
        global_batch_size=bsz,
        max_response_len=max_resp,
    )
    return td


# ---------------------------------------------------------------------------
# 1. Index builder & NaN-canary verification
# ---------------------------------------------------------------------------


def test_compute_response_logprob_index_nested_and_nan_canary():
    prompt_lens = [13, 5, 29, 7]
    response_lens = [6, 11, 4, 9]
    td = _make_nested_ppo_micro_batch(prompt_lens, response_lens)

    idx = compute_response_logprob_index(td)
    assert idx is not None
    assert idx.dtype == torch.long

    expected = []
    cursor = 0
    for plen, rlen in zip(prompt_lens, response_lens, strict=True):
        seq_end = cursor + plen + rlen
        expected.extend(range(seq_end - rlen - 1, seq_end - 1))
        cursor = seq_end
    assert idx.tolist() == expected

    # Canary test: fill every non-indexed token with NaN and verify downstream
    # response extractors never read outside `idx`.
    total_nnz = td["input_ids"].values().numel()
    full_values = torch.randn(total_nnz, dtype=torch.float32)
    canary_values = torch.full((total_nnz,), float("nan"), dtype=torch.float32)
    canary_values[idx] = full_values[idx]

    offsets = td["input_ids"].offsets()
    full_nested = torch.nested.nested_tensor_from_jagged(full_values, offsets)
    canary_nested = torch.nested.nested_tensor_from_jagged(canary_values, offsets)

    padded_ref = no_padding_2_padding(full_nested, td)
    padded_canary = no_padding_2_padding(canary_nested, td)
    assert torch.isfinite(padded_canary).all()
    torch.testing.assert_close(padded_canary, padded_ref)

    nested_ref = response_from_nested(full_nested, td["response_mask"])
    nested_canary = response_from_nested(canary_nested, td["response_mask"])
    assert torch.isfinite(nested_canary.values()).all()
    torch.testing.assert_close(nested_canary.values(), nested_ref.values())


def test_compute_response_logprob_index_padded_prompts_responses():
    batch_size = 3
    max_prompt_len = 12
    max_resp_len = 8
    max_seq_len = max_prompt_len + max_resp_len
    prompt_lens = [7, 12, 4]
    resp_lens = [5, 3, 8]

    input_ids = torch.randint(1, VOCAB_SIZE, (batch_size, max_seq_len))
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    response_mask = torch.zeros(batch_size, max_resp_len, dtype=torch.long)
    for i, (plen, rlen) in enumerate(zip(prompt_lens, resp_lens, strict=True)):
        attention_mask[i, max_prompt_len - plen : max_prompt_len] = 1
        attention_mask[i, max_prompt_len : max_prompt_len + rlen] = 1
        response_mask[i, :rlen] = 1

    td = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :max_prompt_len],
            "responses": input_ids[:, max_prompt_len:],
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1),
        },
        batch_size=[batch_size],
    )
    td_no_pad = left_right_2_no_padding(td)
    # Keep 2D padded prompts/responses + attention_mask to exercise the non-nested branch.
    td_no_pad["prompts"] = input_ids[:, :max_prompt_len]
    td_no_pad["responses"] = input_ids[:, max_prompt_len:]
    td_no_pad["attention_mask"] = attention_mask

    idx = compute_response_logprob_index(td_no_pad)
    assert idx is not None

    expected = []
    cursor = 0
    for plen, rlen in zip(prompt_lens, resp_lens, strict=True):
        seq_end = cursor + plen + rlen
        expected.extend(range(seq_end - rlen - 1, seq_end - 1))
        cursor = seq_end
    assert idx.tolist() == expected


def test_compute_response_logprob_index_edge_cases():
    # Mixed empty and non-empty responses
    td_mixed = _make_nested_ppo_micro_batch(prompt_lens=[6, 4, 5], response_lens=[3, 0, 2])
    idx_mixed = compute_response_logprob_index(td_mixed)
    assert idx_mixed is not None
    # Sample 0: len=9, [5,6,7]; Sample 1: len=4, []; Sample 2: start=13, len=7, end=20 -> [17,18]
    assert idx_mixed.tolist() == [5, 6, 7, 17, 18]

    # All-empty responses -> empty index tensor
    td_empty = _make_nested_ppo_micro_batch(prompt_lens=[5, 7], response_lens=[0, 0])
    idx_empty = compute_response_logprob_index(td_empty)
    assert idx_empty is not None
    assert idx_empty.numel() == 0

    # Zero-length prompt -> unsafe to index seq_end - resp_len - 1, must return None
    td_zero_prompt = _make_nested_ppo_micro_batch(prompt_lens=[0, 5], response_lens=[3, 2])
    assert compute_response_logprob_index(td_zero_prompt) is None

    # Missing prompts/responses -> None
    td_missing = _make_nested_ppo_micro_batch(prompt_lens=[4], response_lens=[2])
    td_missing.pop("prompts")
    assert compute_response_logprob_index(td_missing) is None


# ---------------------------------------------------------------------------
# 2. Adapter routing, shape preservation, VLM kwarg isolation, and gradients
# ---------------------------------------------------------------------------


class _FakeConfig:
    output_attentions = False
    output_hidden_states = False


class _FakeBaseOutput:
    def __init__(self, hidden: torch.Tensor):
        self._hidden = hidden
        self.hidden_states = None
        self.past_key_values = None
        self.attentions = None

    def __getitem__(self, idx):
        if idx == 0:
            return self._hidden
        raise IndexError(idx)


def _make_fake_lm(capture_model_kwargs: dict[str, Any] | None = None) -> torch.nn.Module:
    class FakeBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

        def forward(self, input_ids=None, **kwargs):
            if capture_model_kwargs is not None:
                capture_model_kwargs.update(kwargs)
            return _FakeBaseOutput(self.embed(input_ids).to(torch.float32))

    class FakeLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()
            self.model = FakeBaseModel()
            self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    torch.manual_seed(7)
    return FakeLM()


@contextlib.contextmanager
def _patch_adapters_for_routing(capture_model_kwargs: dict[str, Any]):
    captured: dict[str, Any] = {}

    def fake_fused_linear_forward(self, hidden_states, vocab_weights, input_ids, temperature=1.0):
        captured["hidden_shape"] = tuple(hidden_states.shape)
        captured["labels"] = input_ids.detach().clone()
        # Real FusedLinearForPPO returns 2D (1, K)
        vals = input_ids.to(torch.float32) + 0.5
        return vals, vals * 0.1

    def fake_linear_ce(hidden_states, vocab_weights, input_ids, temperature, reduction, *_args, **kwargs):
        captured["hidden_shape"] = tuple(hidden_states.shape)
        captured["labels"] = input_ids.detach().clone()
        # Real Triton linear_cross_entropy returns 1D (K,)
        vals = input_ids.squeeze(0).to(torch.float32) + 0.5
        return vals, vals * 0.1

    def fake_vlm_body(self, input_ids, **kwargs):
        capture_model_kwargs.update(kwargs)
        return _FakeBaseOutput(self.model.embed(input_ids).to(torch.float32))

    with (
        mock.patch("verl.utils.experimental.torch_functional.FusedLinearForPPO.forward", new=fake_fused_linear_forward),
        mock.patch("verl.utils.kernel.linear_cross_entropy.linear_cross_entropy", new=fake_linear_ce),
        mock.patch("verl.models.transformers.qwen2_vl.qwen2_vl_forward", new=fake_vlm_body),
        mock.patch("verl.models.transformers.glm4v.glm4v_forward", new=fake_vlm_body),
    ):
        yield captured


@pytest.mark.parametrize("forward_fn", ALL_ADAPTERS, ids=lambda f: f"{f.__module__.split('.')[-1]}.{f.__name__}")
def test_all_adapters_slice_response_tokens_and_do_not_leak_kwarg(forward_fn):
    total_nnz = 20
    input_ids = torch.randint(0, VOCAB_SIZE, (1, total_nnz))
    position_ids = torch.arange(total_nnz).unsqueeze(0)
    shift_labels = torch.arange(100, 100 + total_nnz, dtype=torch.long).unsqueeze(0)
    logprob_index = torch.tensor([3, 4, 5, 14, 15, 16, 17], dtype=torch.long)

    model_kwargs_seen: dict[str, Any] = {}
    model = _make_fake_lm(capture_model_kwargs=model_kwargs_seen)

    with _patch_adapters_for_routing(model_kwargs_seen) as captured:
        out = forward_fn(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            logprob_index=logprob_index,
            temperature=1.0,
            return_dict=True,
        )

    assert "logprob_index" not in model_kwargs_seen, "logprob_index leaked into base/VLM model **kwargs"
    assert captured["hidden_shape"] == (1, logprob_index.numel(), HIDDEN_SIZE)
    torch.testing.assert_close(captured["labels"], shift_labels.index_select(-1, logprob_index))

    expected_rank = (1, total_nnz) if forward_fn in TORCH_ADAPTERS else (total_nnz,)
    assert out.log_probs.shape == expected_rank
    assert out.entropy.shape == expected_rank

    lp_flat = out.log_probs.squeeze(0)
    ent_flat = out.entropy.squeeze(0)
    torch.testing.assert_close(lp_flat[logprob_index], shift_labels[0, logprob_index].to(torch.float32) + 0.5)
    torch.testing.assert_close(ent_flat[logprob_index], (shift_labels[0, logprob_index].to(torch.float32) + 0.5) * 0.1)

    outside_mask = torch.ones(total_nnz, dtype=torch.bool)
    outside_mask[logprob_index] = False
    assert torch.all(lp_flat[outside_mask] == 0)
    assert torch.all(ent_flat[outside_mask] == 0)


@pytest.mark.parametrize("forward_fn", ALL_ADAPTERS, ids=lambda f: f"{f.__module__.split('.')[-1]}.{f.__name__}")
def test_all_adapters_handle_empty_logprob_index(forward_fn):
    total_nnz = 12
    input_ids = torch.randint(0, VOCAB_SIZE, (1, total_nnz))
    position_ids = torch.arange(total_nnz).unsqueeze(0)
    shift_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    empty_index = torch.empty(0, dtype=torch.long)

    model_kwargs_seen: dict[str, Any] = {}
    model = _make_fake_lm(capture_model_kwargs=model_kwargs_seen)
    model.train()

    with _patch_adapters_for_routing(model_kwargs_seen) as captured:
        out = forward_fn(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            shift_labels=shift_labels,
            logprob_index=empty_index,
            temperature=1.0,
            return_dict=True,
        )

    assert "hidden_shape" not in captured, "Kernel should not be called when logprob_index is empty"
    assert out.log_probs.squeeze(0).shape == (total_nnz,)
    assert torch.all(out.log_probs == 0)
    assert out.log_probs.requires_grad


def test_torch_fused_ppo_forward_backward_numerical_equivalence():
    prompt_lens = [11, 7, 15]
    response_lens = [5, 8, 4]
    td = _make_nested_ppo_micro_batch(prompt_lens, response_lens, calculate_entropy=True)
    idx = compute_response_logprob_index(td)
    assert idx is not None

    input_ids_rmpad = td["input_ids"].values().unsqueeze(0)
    position_ids_rmpad = td["position_ids"].values().unsqueeze(0)
    rolled_labels = torch.cat([torch.roll(seq, shifts=-1, dims=0) for seq in td["input_ids"].unbind()]).unsqueeze(0)
    offsets = td["input_ids"].offsets()
    actor_cfg = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=3, entropy_coeff=0.02)

    def run_pass(logprob_index):
        model = _make_fake_lm()
        model.train()
        out = dense_common.forward_with_torch_backend(
            model,
            input_ids=input_ids_rmpad,
            position_ids=position_ids_rmpad,
            shift_labels=rolled_labels,
            logprob_index=logprob_index,
            temperature=1.2,
            return_dict=True,
        )
        model_output = {
            "log_probs": torch.nested.nested_tensor_from_jagged(out.log_probs.squeeze(0), offsets),
            "entropy": torch.nested.nested_tensor_from_jagged(out.entropy.squeeze(0), offsets),
        }
        loss, _ = ppo_loss(config=actor_cfg, model_output=model_output, data=td, dp_group=None)
        loss.backward()
        return (
            loss.detach(),
            no_padding_2_padding(model_output["log_probs"], td).detach(),
            no_padding_2_padding(model_output["entropy"], td).detach(),
            model.model.embed.weight.grad.detach().clone(),
            model.lm_head.weight.grad.detach().clone(),
        )

    loss_ref, lp_ref, ent_ref, dembed_ref, dhead_ref = run_pass(logprob_index=None)
    loss_opt, lp_opt, ent_opt, dembed_opt, dhead_opt = run_pass(logprob_index=idx)

    torch.testing.assert_close(loss_opt, loss_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(lp_opt, lp_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ent_opt, ent_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dembed_opt, dembed_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dhead_opt, dhead_ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. FSDPEngineWithLMHead & FSDPEngineWithValueHead safety-gate tests
# ---------------------------------------------------------------------------


def _make_test_fsdp_engine(module: torch.nn.Module, engine_cls=FSDPEngineWithLMHead, use_ulysses_sp: bool = False):
    engine = engine_cls.__new__(engine_cls)
    engine.module = module
    engine.pad_to_length = False
    engine.use_ulysses_sp = use_ulysses_sp
    engine.ulysses_sequence_parallel_size = 2 if use_ulysses_sp else 1
    engine._autocast_dtype = torch.float32
    engine.engine_config = SimpleNamespace(
        batch_invariant=False,
        entropy_checkpointing=False,
        entropy_from_logits_with_chunking=False,
    )
    engine.get_data_parallel_group = lambda: None
    engine._get_packed_pad_size = lambda _n: 0
    engine._gather_and_unpad_packed = lambda x, pad_size: x[: x.shape[0] - pad_size] if pad_size else x
    return engine


class _RecordingPatchedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs: dict[str, Any] = {}

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask=None,
        position_ids=None,
        shift_labels=None,
        logprob_index=None,
        **kwargs,
    ):
        self.last_kwargs = {
            "shift_labels": shift_labels,
            "logprob_index": logprob_index,
            **kwargs,
        }
        seq_len = input_ids.shape[-1]
        z = torch.zeros((1, seq_len), dtype=torch.float32, requires_grad=True)
        return SimpleNamespace(log_probs=z, entropy=z.clone(), logits=torch.zeros((1, seq_len, 1), requires_grad=True))


class _RecordingGenericKwargsModule(torch.nn.Module):
    """Simulates a non-patched or VeOmni module whose forward only has **kwargs."""

    def __init__(self):
        super().__init__()
        self.last_kwargs: dict[str, Any] = {}

    def forward(self, input_ids: torch.LongTensor, **kwargs):
        self.last_kwargs = dict(kwargs)
        seq_len = input_ids.shape[-1]
        z = torch.zeros((1, seq_len), dtype=torch.float32, requires_grad=True)
        return SimpleNamespace(log_probs=z, entropy=z.clone())


def test_fsdp_engine_enables_logprob_index_for_ppo_and_logprob_inference(monkeypatch):
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_id", lambda: "cpu")
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_name", lambda: "cpu")

    base_mod = _RecordingPatchedModule()

    # Wrap inside FSDP1-style + PEFT-style shells to test _supports_logprob_index unwrapping
    class _PeftShell(torch.nn.Module):
        def get_base_model(self):
            return base_mod

        def forward(self, *args, **kwargs):
            return base_mod(*args, **kwargs)

    class _FsdpShell(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._fsdp_wrapped_module = _PeftShell()

        def forward(self, *args, **kwargs):
            return self._fsdp_wrapped_module(*args, **kwargs)

    engine = _make_test_fsdp_engine(_FsdpShell())
    td = _make_nested_ppo_micro_batch(prompt_lens=[8, 6], response_lens=[3, 4])
    expected_idx = compute_response_logprob_index(td)

    actor_cfg = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=2)
    ppo_loss_fn = partial(ppo_loss, config=actor_cfg)

    engine.forward_step(micro_batch=td, loss_function=ppo_loss_fn, forward_only=False)
    assert base_mod.last_kwargs["logprob_index"] is not None
    torch.testing.assert_close(base_mod.last_kwargs["logprob_index"], expected_idx)

    # Forward-only log-prob / ref-log-prob inference (loss_function=None, forward_only=True)
    base_mod.last_kwargs.clear()
    engine.forward_step(micro_batch=td, loss_function=None, forward_only=True)
    assert base_mod.last_kwargs["logprob_index"] is not None
    torch.testing.assert_close(base_mod.last_kwargs["logprob_index"], expected_idx)


@pytest.mark.parametrize(
    "gate_case",
    [
        "sft_loss",
        "distillation_use_topk",
        "distillation_only",
        "score_centering",
        "return_model_output",
        "ulysses_sp",
        "sp_size_gt_1",
        "no_fused_kernels",
        "generic_kwargs_model",
        "value_head_engine",
    ],
)
def test_fsdp_engine_disables_logprob_index_when_unsafe(monkeypatch, gate_case):
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_id", lambda: "cpu")
    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.get_device_name", lambda: "cpu")

    td = _make_nested_ppo_micro_batch(prompt_lens=[6, 5], response_lens=[3, 2])
    actor_cfg = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=2)
    loss_fn: Any = partial(ppo_loss, config=actor_cfg)
    forward_only = False

    if gate_case == "generic_kwargs_model":
        mod: torch.nn.Module = _RecordingGenericKwargsModule()
    else:
        mod = _RecordingPatchedModule()

    engine_cls = FSDPEngineWithValueHead if gate_case == "value_head_engine" else FSDPEngineWithLMHead
    engine = _make_test_fsdp_engine(mod, engine_cls=engine_cls, use_ulysses_sp=(gate_case == "ulysses_sp"))

    if gate_case == "sft_loss":
        loss_fn = partial(sft_loss, config=SimpleNamespace())
        forward_only = True
    elif gate_case == "distillation_use_topk":
        tu.assign_non_tensor(td, distillation_use_topk=True)
    elif gate_case == "distillation_only":
        tu.assign_non_tensor(td, distillation_only=True)
    elif gate_case == "score_centering":
        tu.assign_non_tensor(td, score_centering=True)
    elif gate_case == "return_model_output":
        tu.assign_non_tensor(td, return_model_output=True)
    elif gate_case == "sp_size_gt_1":
        tu.assign_non_tensor(td, sp_size=2)
    elif gate_case == "no_fused_kernels":
        tu.assign_non_tensor(td, use_fused_kernels=False)
    elif gate_case == "value_head_engine":
        loss_fn = None
        forward_only = True

    idx = engine._maybe_build_response_logprob_index(micro_batch=td, loss_function=loss_fn, forward_only=forward_only)
    assert idx is None

    if gate_case in ("return_model_output", "generic_kwargs_model", "value_head_engine"):
        engine.forward_step(micro_batch=td, loss_function=loss_fn, forward_only=forward_only)
        assert mod.last_kwargs.get("logprob_index") is None
