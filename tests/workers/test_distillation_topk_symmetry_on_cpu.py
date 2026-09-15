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
"""Regression guard for verl#6293.

The use_remove_padding=False branch of
FSDPEngineWithLMHead.prepare_model_outputs previously lacked the
distillation_use_topk handling that the use_remove_padding=True branch had,
so distillation outputs were silently dropped from model_output and the
downstream loss raised KeyError. This test invokes prepare_model_outputs on
a stub engine for both branches with distillation_use_topk=True and asserts
the distillation keys produced by logits_processor_func are propagated into
model_output as nested tensors in both cases.

``logprobs_from_logits`` is patched out: in CI environments where flash-attn
is installed, it dispatches to a Triton CrossEntropyLoss kernel that cannot
operate on CPU tensors. The substitute returns a dummy ``log_probs`` tensor
of the right shape, which is sufficient for this test — the contract under
test is the propagation of distillation keys, not the numerical correctness
of log-prob computation.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp.losses import (
    compute_forward_kl_topk as compute_fsdp_forward_kl_topk,
)
from verl.trainer.distillation.fsdp.losses import (
    compute_forward_kl_topk_tail as compute_fsdp_forward_kl_topk_tail,
)
from verl.trainer.distillation.fsdp.losses import tail_aware_kl_divergence
from verl.trainer.distillation.losses import compute_forward_kl_topk as collect_forward_kl_topk_metrics
from verl.trainer.distillation.tail_kl import compute_tail_aware_logit_gradient
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import DistillationLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = (
    "distillation_losses",
    "student_mass",
    "overlap_count",
    "overlap_token_advantage",
    "tail_loss",
)


def _make_engine_stub():
    """Bypass FSDPEngineWithLMHead.__init__; set only attributes that
    prepare_model_outputs touches in this test path (no SP, no fused kernels,
    no entropy)."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False

    class _EngineCfg:
        entropy_checkpointing = False

    eng.engine_config = _EngineCfg()
    return eng


def _make_logits_processor(keys):
    """Fake top-k distillation processor: returns one (1, total_nnz) tensor per key.

    The real processor (verl/trainer/distillation/losses.py) returns
    student_logits.shape[:2]; we mimic that contract.
    """

    def _proc(student_logits, data):
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("distillation_only", [False, True])
def test_distillation_outputs_emitted_in_both_padding_modes(use_remove_padding, distillation_only):
    """distillation_use_topk=True must populate distillation outputs into
    model_output regardless of use_remove_padding. See verl#6293.

    When distillation_only=True, log_probs must be omitted (supervised top-k path)."""
    bsz = 2
    seq_lengths_list = [3, 2]
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int64)
    total_nnz = int(seq_lengths.sum())

    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)]).to(torch.int64)

    flat_input_ids = torch.randint(0, _VOCAB_SIZE, (total_nnz,))
    input_ids_nested = torch.nested.nested_tensor_from_jagged(flat_input_ids, offsets=cu_seqlens)

    input_ids_rmpad_rolled = torch.randint(0, _VOCAB_SIZE, (total_nnz,))

    class _Output:
        pass

    output = _Output()

    if use_remove_padding:
        # True branch: output.logits shape (1, total_nnz, V), squeeze(0) -> (total_nnz, V).
        output.logits = torch.randn(1, total_nnz, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature_rmpad": torch.ones(total_nnz),
            # No SP and no static pad_to_length here, so nothing to trim off the packed tail.
            "pad_size": 0,
        }
    else:
        # False branch: output.logits shape (bsz, max_seqlen, V).
        max_seqlen = max(seq_lengths_list)
        output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature": torch.ones(bsz),
        }

    micro_batch = TensorDict({"input_ids": input_ids_nested}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=distillation_only,
        max_response_length=max(seq_lengths_list),
    )

    eng = _make_engine_stub()

    # Patch logprobs_from_logits because flash-attn's Triton CrossEntropyLoss
    # cannot operate on CPU tensors. The shape is what downstream code asserts
    # against (v.shape == log_probs.shape), and prepare_model_outputs reduces
    # both branches to a (total_nnz,) log_probs over the rmpad'ed logits.
    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits",
        return_value=torch.zeros(total_nnz),
    ):
        model_output = FSDPEngineWithLMHead.prepare_model_outputs(
            eng,
            output=output,
            output_args=output_args,
            micro_batch=micro_batch,
            logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS),
        )

    if distillation_only:
        assert "log_probs" not in model_output, (
            f"log_probs should be omitted when distillation_only=True "
            f"(use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )
    else:
        assert "log_probs" in model_output, (
            f"log_probs missing (use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
        )

    for k in _DISTILLATION_KEYS:
        assert k in model_output, (
            f"Distillation key '{k}' missing from model_output "
            f"(use_remove_padding={use_remove_padding}); "
            f"keys: {list(model_output.keys())}"
        )
        assert model_output[k].is_nested, (
            f"Expected '{k}' to be a nested tensor (use_remove_padding={use_remove_padding}); "
            f"got {type(model_output[k])}"
        )


def _nested_from_rows(rows):
    values = torch.tensor(rows)
    offsets = torch.tensor([0, len(rows)], dtype=torch.int64)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def test_forward_kl_topk_emits_overlap_metrics():
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 1.0, 0.0, 0.0],
            [8.0, 7.0, 0.0, 0.0, 9.0, 0.0],
            [9.0, 8.0, 7.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    teacher_ids = _nested_from_rows([[1, 2], [4, 5], [3, 4]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows(
        [
            [torch.log(torch.tensor(0.7)), torch.log(torch.tensor(0.2))],
            [torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.3))],
            [torch.log(torch.tensor(0.5)), torch.log(torch.tensor(0.4))],
        ]
    ).to(torch.float32)
    config = SimpleNamespace(distillation_loss=SimpleNamespace(log_prob_min_clamp=None))

    output = compute_fsdp_forward_kl_topk(
        student_logits=logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )

    torch.testing.assert_close(output["overlap_count"], torch.tensor([[2, 1, 0]]))

    student_log_probs = torch.log_softmax(logits, dim=-1)
    gathered_student = torch.gather(student_log_probs, dim=-1, index=teacher_ids.values().unsqueeze(0))
    teacher_log_probs = teacher_logprobs.values().unsqueeze(0)
    token_adv = -(teacher_log_probs.exp() * (teacher_log_probs - gathered_student))
    expected_ota = torch.tensor(
        [[token_adv[0, 0].mean(), token_adv[0, 1, 0], 0.0]],
        dtype=output["overlap_token_advantage"].dtype,
    )
    torch.testing.assert_close(output["overlap_token_advantage"], expected_ota)


def test_forward_kl_topk_metric_aggregation_for_overlap_outputs():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([0.1, 0.2, 0.3]),
        "student_mass": torch.tensor([0.9, 0.8, 0.7]),
        "teacher_mass": torch.tensor([0.95, 0.85, 0.75]),
        "overlap_count": torch.tensor([2, 1, 0]),
        "overlap_token_advantage": torch.tensor([-0.2, -0.4, 0.0]),
    }
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2))

    _, metrics = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )

    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(-0.3)


@pytest.mark.parametrize("use_chunked_topk", [False, True])
def test_forward_kl_topk_tail_matches_coarse_grained_kl_and_gradient(use_chunked_topk):
    """Tail-aware top-k is the exact KL after collapsing non-top-k tokens into one bucket."""
    teacher_probs = torch.tensor([0.30, 0.20, 0.18, 0.17, 0.15], dtype=torch.float64)
    student_probs = torch.tensor([0.50, 0.30, 0.08, 0.07, 0.05], dtype=torch.float64)
    teacher_ids = _nested_from_rows([[0, 1]]).to(torch.int64)
    teacher_logprobs = _nested_from_rows([teacher_probs[:2].log().tolist()]).to(torch.float64)
    student_logits = student_probs.log().reshape(1, 1, -1).requires_grad_(True)
    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            log_prob_min_clamp=None,
            tail_mass_eps=1e-12,
            use_chunked_topk=use_chunked_topk,
            chunked_topk_chunk_size=1,
        )
    )

    baseline = compute_fsdp_forward_kl_topk(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )["distillation_losses"]
    output = compute_fsdp_forward_kl_topk_tail(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_logprobs,
        teacher_topk_ids=teacher_ids,
        config=config,
        data_format="thd",
    )

    teacher_coarse = torch.cat([teacher_probs[:2], (1.0 - teacher_probs[:2].sum()).unsqueeze(0)])
    current_student_probs = student_logits.softmax(dim=-1).reshape(-1)
    student_coarse = torch.cat([current_student_probs[:2], (1.0 - current_student_probs[:2].sum()).unsqueeze(0)])
    expected = (teacher_coarse * (teacher_coarse.log() - student_coarse.log())).sum()

    assert baseline.item() < 0.0
    torch.testing.assert_close(output["distillation_losses"].squeeze(), expected.float())
    torch.testing.assert_close(output["tail_loss"].squeeze(), (expected - baseline.squeeze()).float())

    actual_grad = torch.autograd.grad(output["distillation_losses"].sum(), student_logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_forward_kl_topk_tail_equals_full_kl_when_topk_spans_vocab():
    teacher_logits = torch.tensor([[[2.0, 0.5, -0.3, 1.1]]], dtype=torch.float64)
    student_logits = torch.tensor([[[0.2, 1.3, -0.7, 0.4]]], dtype=torch.float64, requires_grad=True)
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    teacher_topk_logprobs, teacher_topk_ids = torch.topk(teacher_log_probs, k=teacher_logits.shape[-1], dim=-1)
    teacher_topk_logprobs = _nested_from_rows(teacher_topk_logprobs.reshape(-1, 4).tolist()).to(torch.float64)
    teacher_topk_ids = _nested_from_rows(teacher_topk_ids.reshape(-1, 4).tolist()).to(torch.int64)
    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            log_prob_min_clamp=None,
            tail_mass_eps=1e-12,
            use_chunked_topk=False,
        )
    )

    output = compute_fsdp_forward_kl_topk_tail(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_topk_logprobs,
        teacher_topk_ids=teacher_topk_ids,
        config=config,
        data_format="thd",
    )
    student_log_probs = student_logits.log_softmax(dim=-1)
    expected = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    torch.testing.assert_close(output["distillation_losses"], expected.float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output["tail_loss"], torch.zeros_like(output["tail_loss"]), atol=1e-6, rtol=0)


def test_forward_kl_topk_tail_rejects_per_entry_logprob_clamp():
    with pytest.raises(ValueError, match="requires log_prob_min_clamp=None"):
        DistillationLossConfig(loss_mode="forward_kl_topk_tail", log_prob_min_clamp=-10.0)


@pytest.mark.parametrize("tail_mass_eps", [0.0, 1.0, -1e-6])
def test_forward_kl_topk_tail_rejects_invalid_tail_epsilon(tail_mass_eps):
    with pytest.raises(ValueError, match="tail_mass_eps must be in"):
        DistillationLossConfig(
            loss_mode="forward_kl_topk_tail",
            log_prob_min_clamp=None,
            tail_mass_eps=tail_mass_eps,
        )


def test_forward_kl_topk_tail_adds_tail_for_fused_mass_outputs():
    """Fused engines can expose differentiable masses and let the common collector add the tail bucket."""
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    head_loss = torch.tensor([-0.10, -0.05, 0.0], requires_grad=True)
    student_mass = torch.tensor([0.80, 0.70, 0.60], requires_grad=True)
    teacher_mass = torch.tensor([0.50, 0.40, 0.30])
    model_output = {
        "distillation_losses": head_loss,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="forward_kl_topk_tail",
            tail_mass_eps=1e-6,
            topk=2,
        )
    )

    losses, metrics = collect_forward_kl_topk_metrics(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    losses[data["response_mask"]].sum().backward()

    assert metrics["distillation/head_loss"] == pytest.approx(-0.075)
    assert metrics["distillation/tail_loss"] > 0.0
    assert student_mass.grad is not None
    assert torch.count_nonzero(student_mass.grad[:2]) == 2


def test_forward_kl_topk_tail_rejects_nondifferentiable_fused_mass():
    data = TensorDict(
        {
            "prompts": torch.tensor([[101]]),
            "responses": torch.tensor([[11]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "response_mask": torch.tensor([[1]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    model_output = {
        "distillation_losses": torch.tensor([0.1, 0.0], requires_grad=True),
        "student_mass": torch.tensor([0.8, 0.0]),
        "teacher_mass": torch.tensor([0.5, 0.0]),
    }
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="forward_kl_topk_tail",
            tail_mass_eps=1e-6,
            topk=2,
        )
    )

    with pytest.raises(RuntimeError, match="requires differentiable student_mass"):
        collect_forward_kl_topk_metrics(
            config=SimpleNamespace(),
            distillation_config=distillation_config,
            model_output=model_output,
            data=data,
        )


def test_tail_aware_analytic_logit_gradient_matches_autograd():
    teacher_probs = torch.tensor([0.30, 0.15, 0.14, 0.13, 0.18, 0.10], dtype=torch.float64)
    student_logits = torch.tensor([0.8, 0.3, -0.2, 0.1, -0.4, 0.6], dtype=torch.float64, requires_grad=True)
    student_probs = student_logits.softmax(dim=-1)
    teacher_ids = torch.tensor([0, 4], dtype=torch.int64)
    teacher_topk_probs = teacher_probs[teacher_ids]
    student_topk_mass = student_probs[teacher_ids].sum()
    teacher_topk_mass = teacher_topk_probs.sum()

    teacher_coarse = torch.cat([teacher_topk_probs, (1.0 - teacher_topk_mass).unsqueeze(0)])
    student_coarse = torch.cat([student_probs[teacher_ids], (1.0 - student_topk_mass).unsqueeze(0)])
    oracle_loss = (teacher_coarse * (teacher_coarse.log() - student_coarse.log())).sum()
    oracle_grad = torch.autograd.grad(oracle_loss, student_logits)[0]

    shard_grads = []
    shard_size = student_probs.numel() // 2
    for shard_start in (0, shard_size):
        shard_end = shard_start + shard_size
        in_shard = (teacher_ids >= shard_start) & (teacher_ids < shard_end)
        local_ids = teacher_ids - shard_start
        local_ids = local_ids.masked_fill(~in_shard, 0)
        local_teacher_probs = teacher_topk_probs.masked_fill(~in_shard, 0.0)
        shard_grads.append(
            compute_tail_aware_logit_gradient(
                student_probs=student_probs.detach()[shard_start:shard_end].unsqueeze(0),
                teacher_topk_probs=local_teacher_probs.unsqueeze(0),
                teacher_topk_indices=local_ids.unsqueeze(0),
                teacher_topk_mask=in_shard.unsqueeze(0),
                student_topk_mass=student_topk_mass.detach().unsqueeze(0),
                teacher_topk_mass=teacher_topk_mass.unsqueeze(0),
                tail_mass_eps=1e-12,
            ).squeeze(0)
        )
    analytic_grad = torch.cat(shard_grads)

    torch.testing.assert_close(analytic_grad, oracle_grad)


@pytest.mark.parametrize("topk", [1, 4, 16])
def test_forward_kl_topk_tail_is_coarse_grained_lower_bound(topk):
    torch.manual_seed(2026 + topk)
    teacher_log_probs = torch.randn(7, 16, dtype=torch.float64).log_softmax(dim=-1)
    student_log_probs = torch.randn(7, 16, dtype=torch.float64).log_softmax(dim=-1)
    teacher_topk_log_probs, teacher_topk_ids = torch.topk(teacher_log_probs, k=topk, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

    tail_aware_loss, *_ = tail_aware_kl_divergence(
        log_q=student_topk_log_probs,
        log_p=teacher_topk_log_probs,
        tail_mass_eps=1e-12,
    )
    full_kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    assert torch.all(tail_aware_loss >= -1e-6)
    assert torch.all(tail_aware_loss <= full_kl.float() + 1e-6)
