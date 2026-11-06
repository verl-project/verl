# Copyright 2026 Individual Contributor: Penghui Yang and the ResOPD authors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the sparse teacher transport and ResOPD loss dispatch."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager, _get_teacher_sampling_params
from verl.trainer.distillation.losses import collect_resopd, compute_topk_loss
from verl.trainer.distillation.resopd import resopd_loss
from verl.trainer.ppo.padding_utils import SYNTHETIC_PADDING_SEQ_LEN, construct_minimal_padding_template
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def loss_config(**overrides):
    args = dict(loss_mode="resopd", use_policy_gradient=False, loss_max_clamp=None, log_prob_min_clamp=None, topk=2)
    return DistillationLossConfig(**(args | overrides))


@pytest.mark.parametrize(
    "bad",
    [
        dict(use_policy_gradient=True),
        dict(loss_max_clamp=10),
        dict(log_prob_min_clamp=-10),
        dict(use_chunked_topk=True),
        dict(topk=0),
        dict(topk=None),
    ],
)
def test_invalid_configuration(bad):
    with pytest.raises(ValueError, match="ResOPD"):
        loss_config(**bad)


@pytest.mark.parametrize("mode", ["resopd", "k1", "forward_kl_topk"])
def test_teacher_request_keeps_two_tensor_contract(mode):
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_loss_config = loss_config(loss_mode=mode, use_policy_gradient=mode == "k1")
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(name="vllm", temperature=1.0))
    }
    width = {"resopd": 3, "forward_kl_topk": 2, "k1": 1}[mode]
    payload = dict(prompt_ids=[[0] * width] * 3, prompt_logprobs=[[-1.0] * width] * 3)
    client = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(extra_fields=payload)))
    manager.teacher_client = {"teacher": client}
    ids, logprobs = asyncio.run(manager.compute_teacher_logprobs_single([8, 2, 0]))
    assert ids.shape == logprobs.shape == (3, width)
    client.generate.assert_awaited_once()
    params = client.generate.call_args.kwargs["sampling_params"]
    assert params["prompt_logprobs"] == (0 if mode == "k1" else 2)
    assert params.get("include_prompt_token", False) == (mode == "resopd")


def test_unsupported_teacher_and_student_fail():
    teacher = SimpleNamespace(inference=SimpleNamespace(name="sglang", temperature=1.0))
    cfg = SimpleNamespace(distillation_loss=loss_config())
    with pytest.raises(NotImplementedError, match="vLLM"):
        _get_teacher_sampling_params(teacher, cfg.distillation_loss)
    with pytest.raises(NotImplementedError, match="eager"):
        compute_topk_loss(SimpleNamespace(strategy="megatron"), cfg, {}, None, "thd")
    with pytest.raises(ValueError, match="disable fused kernels"):
        collect_resopd(None, cfg, {}, {})


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni"])
def test_packed_teacher_scores_and_response_mask(strategy):
    # Column 2 is the observed token, including a head hit (duplicated ID).
    ids = torch.tensor([[[0, 0, 0], [0, 1, 2], [0, 1, 1], [0, 0, 0]], [[0, 1, 2], [0, 1, 0], [0, 0, 0], [0, 0, 0]]])
    scores = torch.tensor(
        [
            [[-1.0, -1.0, -1.0], [-0.7, -1.2, -2.0], [-0.6, -1.3, -1.3], [0.0, 0.0, 0.0]],
            [[-0.8, -1.1, -2.1], [-0.9, -1.0, -0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    data = TensorDict(
        dict(
            input_ids=torch.tensor([[0, 8, 2, 1], [7, 2, 0, 0]]),
            position_ids=torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]]),
            attention_mask=torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]]),
            prompts=torch.tensor([[0, 8], [7, 2]]),
            responses=torch.tensor([[2, 1], [0, 0]]),
            response_mask=torch.tensor([[1, 0], [1, 0]]),
            teacher_ids=ids,
            teacher_logprobs=scores,
        ),
        batch_size=2,
    )
    data = left_right_2_no_padding(data)
    assert data["teacher_ids"].values()[:, -1].tolist() == [2, 1, 0, 2, 0, 0]
    z = torch.tensor([[[0.2, 0.4, -0.3]] * 6], requires_grad=True)
    cfg = SimpleNamespace(distillation_loss=loss_config())
    output = compute_topk_loss(SimpleNamespace(strategy=strategy), cfg, data, z, "thd")
    losses, metrics = collect_resopd(None, cfg, {key: value.squeeze(0) for key, value in output.items()}, data)
    expected = torch.stack(
        [output["distillation_losses"][0, :2], torch.stack([output["distillation_losses"][0, 4], z.new_zeros(())])]
    )
    torch.testing.assert_close(losses, expected)
    assert metrics["distillation/resopd_head_hit"].aggregate() == 0.5
    (losses * data["response_mask"]).sum().backward()
    assert z.grad[0, [1, 2, 3, 5]].eq(0).all()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("rank", [0, 1])
def test_sequence_parallel_slicing_and_padding(rank):
    ids = torch.tensor([[0, 1, 2], [0, 1, 0], [0, 1, 1]])
    scores = torch.tensor([[-0.7, -1.2, -2.0], [-0.7, -1.2, -0.7], [-0.7, -1.2, -1.2]])
    data = {
        key: torch.nested.as_nested_tensor([value], layout=torch.jagged)
        for key, value in (("teacher_ids", ids), ("teacher_logprobs", scores))
    }
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    with (
        patch("verl.trainer.distillation.fsdp.losses.get_ulysses_sequence_parallel_world_size", return_value=2),
        patch("verl.utils.ulysses.dist.get_world_size", return_value=2),
        patch("verl.utils.ulysses.get_ulysses_sequence_parallel_rank", return_value=rank),
    ):
        output = compute_topk_loss(
            SimpleNamespace(strategy="fsdp"), SimpleNamespace(distillation_loss=loss_config()), data, logits, "thd"
        )
    ids = torch.cat([ids, torch.zeros_like(ids[:1])])[2 * rank : 2 * rank + 2].unsqueeze(0)
    scores = torch.cat([scores, torch.zeros_like(scores[:1])])[2 * rank : 2 * rank + 2].unsqueeze(0)
    expected = resopd_loss(logits, scores[..., :-1], ids[..., :-1], scores[..., -1:], ids[..., -1:])
    torch.testing.assert_close(output["distillation_losses"], expected["distillation_losses"])


def test_synthetic_padding_preserves_payload_width():
    source = {"teacher_ids": torch.ones(7, 3, dtype=torch.long), "teacher_logprobs": torch.ones(7, 3)}
    output, tag = construct_minimal_padding_template(source, {}, eos_token_id=0)
    assert tag["is_padding"] and output["response_mask"].sum() == 0
    for key in source:
        assert output[key].shape == (SYNTHETIC_PADDING_SEQ_LEN, 3)


def test_negative_surrogate_is_not_clipped():
    names = (
        "distillation_losses",
        "student_mass",
        "teacher_mass",
        "resopd_head_hit",
        "resopd_centered_event",
        "resopd_tail_log_ratio",
    )
    output = {key: torch.tensor([-2.0, 999.0], requires_grad=key == "distillation_losses") for key in names}
    data = {
        "prompts": torch.nested.as_nested_tensor([torch.tensor([7])], layout=torch.jagged),
        "responses": torch.nested.as_nested_tensor([torch.tensor([1])], layout=torch.jagged),
        "response_mask": torch.tensor([[1]]),
    }
    losses, _ = collect_resopd(None, None, output, data)
    assert losses.item() == -2.0
    losses.sum().backward()
    assert output["distillation_losses"].grad.tolist() == [1.0, 0.0]
