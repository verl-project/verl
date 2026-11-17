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
"""Semantic EOS for on-policy distillation (arXiv:2609.20511).

A base student and its post-trained teacher can end turns with different terminal tokens
(Qwen: ``<|endoftext|>`` vs ``<|im_end|>``).

Every student stop is then penalized by the teacher , because student use a different terminal token
and the student learns to never stop. With
``distillation_loss.semantic_eos_token_ids``
both models score a sampled stop for the whole terminal case (student + teacher).

"""

from types import SimpleNamespace

import pytest
import torch

from verl.experimental.teacher_loop.teacher_manager import merge_semantic_eos_logprobs
from verl.utils.torch_functional import semantic_eos_log_probs
from verl.workers.config import DistillationLossConfig
from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

EOS = [5, 7]


def _reference_class_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_softmax = torch.log_softmax(logits, dim=-1)
    sampled = log_softmax.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    terminal = torch.isin(labels, torch.tensor(EOS))
    return torch.where(terminal, torch.logsumexp(log_softmax[:, EOS], dim=-1), sampled)


def test_student_terminal_rows_score_whole_class_with_gradient():
    torch.manual_seed(0)
    labels = torch.tensor([1, 5, 3, 7, 0, 5])
    logits = torch.randn(len(labels), 10, dtype=torch.float64, requires_grad=True)
    ref_logits = logits.detach().clone().requires_grad_(True)
    weights = torch.randn(len(labels), dtype=torch.float64)

    sampled = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    merged = semantic_eos_log_probs(logits, labels, sampled, EOS)
    expected = _reference_class_log_probs(ref_logits, labels)
    (merged * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(merged, expected)
    torch.testing.assert_close(logits.grad, ref_logits.grad)


def test_student_without_terminal_labels_is_unchanged():
    log_probs = torch.randn(3)
    assert semantic_eos_log_probs(torch.randn(3, 10), torch.tensor([0, 1, 2]), log_probs, EOS) is log_probs


def test_teacher_merges_terminal_tokens_found_in_top_k():
    next_ids = torch.tensor([5, 2, 7, 5], dtype=torch.int32)
    token_lp = torch.tensor([-12.0, -0.5, -0.1, -9.0])
    # Row 0: the other terminal (7) is the teacher's top-1. Row 2: the sampled terminal is in its
    # own top-k and must be counted once. Row 3: the other terminal is outside the top-k.
    topk_ids = torch.tensor([[7, 1], [2, 3], [7, 5], [1, 2]], dtype=torch.int32)
    topk_lp = torch.tensor([[-0.01, -5.0], [-0.5, -2.0], [-0.1, -3.0], [-0.2, -2.0]])

    merged = merge_semantic_eos_logprobs(next_ids, token_lp, topk_ids, topk_lp, EOS)

    expected = torch.stack(
        [
            torch.logsumexp(torch.tensor([-12.0, -0.01]), 0),
            torch.tensor(-0.5),
            torch.logsumexp(torch.tensor([-0.1, -3.0]), 0),
            torch.tensor(-9.0),
        ]
    )
    torch.testing.assert_close(merged, expected)


def test_vllm_prompt_token_logprobs_keep_tokens_outside_top_k():
    def logprob(value, rank):
        return SimpleNamespace(logprob=value, rank=rank)

    # Token 9 at position 2 is outside the top-1 and only reported with rank 2 > k.
    output = SimpleNamespace(
        prompt_token_ids=[4, 3, 9],
        prompt_logprobs=[None, {3: logprob(-0.2, 1)}, {1: logprob(-0.1, 1), 9: logprob(-6.0, 2)}],
    )
    result = {}
    extract_prompt_logprobs(output, num_prompt_logprobs=1, result_dict=result)

    assert result["prompt_ids"] == [[3], [1], [0]]
    assert result["prompt_token_logprobs"] == [-0.2, -6.0, 0.0]

    result = {}
    extract_prompt_logprobs(output, num_prompt_logprobs=0, result_dict=result)
    assert "prompt_token_logprobs" not in result


def test_config_requests_teacher_top_k_only_when_needed():
    assert DistillationLossConfig(loss_mode="k1").teacher_prompt_logprobs == 0
    assert DistillationLossConfig(loss_mode="forward_kl_topk", topk=16).teacher_prompt_logprobs == 16
    config = DistillationLossConfig(loss_mode="k1", semantic_eos_token_ids=EOS, semantic_eos_teacher_topk=8)
    assert config.teacher_prompt_logprobs == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss_mode": "k1", "semantic_eos_token_ids": [5]},
        {"loss_mode": "k1", "semantic_eos_token_ids": [5, 5]},
        {"loss_mode": "forward_kl_topk", "semantic_eos_token_ids": EOS},
        {"loss_mode": "k1", "semantic_eos_token_ids": EOS, "semantic_eos_teacher_topk": 0},
    ],
)
def test_config_rejects_invalid_semantic_eos(kwargs):
    with pytest.raises(ValueError):
        DistillationLossConfig(**kwargs)


def test_v1_rollout_stops_on_every_terminal_token(monkeypatch):
    """The V1 worker overrides ``generate_sequences``; its rollouts must still stop on the whole class."""
    import asyncio

    from omegaconf import OmegaConf
    from tensordict import TensorDict

    from verl.trainer.ppo.v1 import agent_loop_tq

    rollout = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "calculate_log_probs": False, "topk_log_probs": 0}
    worker = object.__new__(agent_loop_tq.AgentLoopWorkerTQ.__ray_actor_class__)
    worker.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": rollout},
            "distillation": {"distillation_loss": {"semantic_eos_token_ids": EOS}},
        }
    )
    worker.distillation_enabled = True
    worker.background_tasks = set()
    captured = []

    async def fake_run_prompt(prompt, sampling_params, trajectory, trace=False):
        captured.append(sampling_params)

    async def fake_trajectory_info(global_steps, index, validate):
        return [{}] * len(index)

    worker._run_prompt = fake_run_prompt
    monkeypatch.setattr(agent_loop_tq, "get_trajectory_info", fake_trajectory_info)
    batch = TensorDict(
        {"global_steps": torch.tensor([0]), "index": torch.tensor([0]), "agent_name": ["single_turn_agent"]},
        batch_size=[1],
    )

    async def run():
        await worker.generate_sequences(batch)
        await asyncio.gather(*worker.background_tasks)

    asyncio.run(run())
    assert captured and captured[0]["stop_token_ids"] == EOS


def _eos_metrics(response_ids: list[list[int]]) -> dict:
    from tensordict import TensorDict

    from verl.trainer.distillation.losses import compute_distillation_loss_reverse_kl_estimator

    responses = torch.tensor(response_ids)
    prompts = torch.ones(responses.shape[0], 2, dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=1)
    next_ids = torch.cat([input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], dim=1)
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "attention_mask": torch.ones_like(input_ids),
            "response_mask": torch.ones_like(responses),
            "teacher_ids": next_ids.reshape(-1, 1),
            "teacher_logprobs": torch.full((input_ids.numel(), 1), -1.0),
        }
    )
    loss_config = DistillationLossConfig(loss_mode="k1", use_policy_gradient=True, semantic_eos_token_ids=EOS)
    model_output = {"log_probs": torch.full((input_ids.numel(),), -0.5)}
    _, metrics = compute_distillation_loss_reverse_kl_estimator(
        None, SimpleNamespace(distillation_loss=loss_config), model_output, data
    )
    return metrics


def test_eos_metrics_aggregate_across_dp_ranks_without_stops():
    """A micro-batch where every response is truncated must still emit the EOS metrics, or dp aggregation fails."""
    from verl.utils.metric.utils import Metric

    stopped = _eos_metrics([[3, 4, 5], [3, 7, 0]])
    truncated = _eos_metrics([[3, 4, 2], [1, 2, 3]])

    assert stopped.keys() == truncated.keys()
    count = Metric.aggregate_dp([stopped["distillation/eos_count"], truncated["distillation/eos_count"]])
    loss_sum = Metric.aggregate_dp([stopped["distillation/eos_loss_sum"], truncated["distillation/eos_loss_sum"]])
    assert loss_sum / count == pytest.approx(0.5)  # k1 at each stop: -0.5 - (-1.0)
