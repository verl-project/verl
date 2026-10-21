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
"""CPU coverage for score centering: config, math, vLLM head extraction, padding template and loss integration."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.core_algos import compute_policy_loss_bypass_mode, compute_policy_loss_reinforce
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.score_centering import (
    dummy_rollout_topk,
    pad_rollout_topk,
    score_centering_correction,
    score_centering_logits_processor,
    score_centering_ppo_loss,
    score_centering_weight_fn,
    topk_log_probs_from_logits,
)
from verl.utils import tensordict_utils as tu
from verl.utils.config import _validate_score_centering_config
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.utils import extract_response_topk_logprobs
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def test_score_centering_presets_are_bypass_reinforce():
    for cfg in (
        RolloutCorrectionConfig.bypass_pg_sc(),
        RolloutCorrectionConfig.bypass_pg_token_tis_sc(),
        RolloutCorrectionConfig.bypass_pg_token_icepop_sc(),
    ):
        assert cfg.score_centering and cfg.bypass_mode and cfg.loss_type == "reinforce"
    assert RolloutCorrectionConfig.bypass_pg_sc().rollout_is is None
    assert RolloutCorrectionConfig.bypass_pg_token_tis_sc(threshold=3.0).rollout_is_threshold == 3.0
    assert RolloutCorrectionConfig.bypass_pg_token_icepop_sc().rollout_is_threshold == "0.5_5.0"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bypass_mode=False, loss_type="reinforce"),
        dict(bypass_mode=True, loss_type="ppo_clip"),
        dict(bypass_mode=True, loss_type="reinforce", rollout_is="sequence"),
        dict(bypass_mode=True, loss_type="reinforce", rollout_is="token", rollout_is_batch_normalize=True),
    ],
)
def test_score_centering_rejects_unsupported_modes(kwargs):
    with pytest.raises(ValueError, match="score_centering"):
        RolloutCorrectionConfig(score_centering=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="sglang"),
        dict(calculate_log_probs=False),
        dict(top_p=0.9),
        dict(top_k=50),
        dict(logprobs_mode="raw_logprobs"),
    ],
)
def test_topk_log_probs_needs_the_vllm_sampling_distribution(kwargs):
    with pytest.raises(ValueError, match="topk_log_probs"):
        RolloutConfig(**{"name": "vllm", "topk_log_probs": 128, "calculate_log_probs": True, **kwargs})


def test_topk_log_probs_raises_vllm_max_logprobs():
    cfg = RolloutConfig(name="vllm", topk_log_probs=128, calculate_log_probs=True)
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 128
    cfg = RolloutConfig(
        name="vllm", topk_log_probs=32, calculate_log_probs=True, engine_kwargs={"vllm": {"max_logprobs": 64}}
    )
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 64
    cfg = RolloutConfig(
        name="vllm", topk_log_probs=128, calculate_log_probs=True, engine_kwargs={"vllm": {"max_logprobs": 20}}
    )
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 128


_RC = {"bypass_mode": True, "loss_type": "reinforce", "rollout_is": "token", "rollout_is_threshold": 2.0}


def _config(algorithm_rc=None, actor_rc=None, **overrides):
    config = {
        "algorithm": {"rollout_correction": {**_RC, "score_centering": True, **(algorithm_rc or {})}},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp", "policy_loss": {"loss_mode": "bypass_mode"}},
            "rollout": {"name": "vllm", "topk_log_probs": 128},
        },
        "distillation": {"enabled": False},
        "trainer": {"use_v1": True},
    }
    if actor_rc is not False:
        config["actor_rollout_ref"]["actor"]["policy_loss"]["rollout_correction"] = {
            **_RC,
            "score_centering": True,
            **(actor_rc or {}),
        }
    config = OmegaConf.create(config)
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    return config


def test_score_centering_config_accepts_consistent_settings():
    _validate_score_centering_config(_config())


def test_score_centering_config_skips_checks_when_off():
    _validate_score_centering_config(
        _config(algorithm_rc={"score_centering": False}, actor_rc=False, **{"trainer.use_v1": False})
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(actor_rc=False), "both"),
        (dict(algorithm_rc={"score_centering": False}), "both"),
        (dict(actor_rc={"rollout_is": "sequence"}), "score_centering"),
        (dict(actor_rc={"rollout_is_threshold": 3.0}), "match"),
        ({"trainer.use_v1": False}, "use_v1"),
        ({"actor_rollout_ref.actor.strategy": "megatron"}, "strategy"),
        ({"actor_rollout_ref.rollout.name": "sglang"}, "vllm"),
        ({"actor_rollout_ref.rollout.topk_log_probs": 0}, "topk_log_probs"),
        ({"distillation.enabled": True}, "distillation"),
    ],
)
def test_score_centering_config_rejects_misconfigurations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _validate_score_centering_config(_config(**kwargs))


def _full_vocab_centering(logits, sampler_log_probs):
    # Exact centering term over the whole vocabulary: sum_v sg[q_v] log p_v.
    log_p = torch.log_softmax(logits.float(), dim=-1)
    return (sampler_log_probs.exp().detach() * log_p).sum(-1)


def test_topk_log_probs_match_log_softmax_gather_with_gradient():
    torch.manual_seed(0)
    logits = torch.randn(5, 37, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.stack([torch.randperm(37)[:4] for _ in range(5)])
    out = topk_log_probs_from_logits(logits, ids, chunk_size=2)
    ref = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    weight = torch.randn_like(out)
    (out * weight).sum().backward()
    grad_chunked = logits.grad.clone()
    logits.grad = None
    (ref * weight).sum().backward()
    torch.testing.assert_close(grad_chunked.float(), logits.grad.float(), atol=1e-2, rtol=1e-2)


def test_correction_is_zero_on_policy():
    torch.manual_seed(1)
    logits = torch.randn(3, 11)
    log_p = torch.log_softmax(logits, dim=-1)
    ids = torch.topk(log_p, k=4, dim=-1).indices
    head = log_p.gather(-1, ids)
    correction, q_mass, p_mass = score_centering_correction(head, head, score_centering_weight_fn(None, 2.0))
    torch.testing.assert_close(correction, torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(q_mass, p_mass)


def test_topk_correction_gradient_equals_full_vocab_when_k_is_vocab():
    # With k = V the head/tail split is degenerate (rho = 0/0), so only the gradient of the
    # correction is meaningful; it must match the exact full-vocab centering term's gradient.
    torch.manual_seed(2)
    vocab = 9
    trainer_logits = torch.randn(4, vocab, requires_grad=True)
    sampler_log_probs = torch.log_softmax(torch.randn(4, vocab), dim=-1)
    ids = torch.arange(vocab).expand(4, vocab)
    head = topk_log_probs_from_logits(trainer_logits, ids)
    correction, _, _ = score_centering_correction(head, sampler_log_probs, score_centering_weight_fn(None, 2.0))
    ref = _full_vocab_centering(trainer_logits, sampler_log_probs)
    grad_a = torch.autograd.grad(correction.sum(), trainer_logits)[0]
    grad_b = torch.autograd.grad(ref.sum(), trainer_logits)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-5, rtol=1e-5)


def test_topk_correction_gradient_matches_tail_model():
    # With k < V the tail is modeled as rho * p; the expected gradient must equal
    # the exact gradient of sum_v sg[q_hat_v] log p_v with q_hat from the paper.
    torch.manual_seed(3)
    vocab, k = 13, 5
    trainer_logits = torch.randn(2, vocab, requires_grad=True)
    sampler_log_probs = torch.log_softmax(torch.randn(2, vocab), dim=-1)
    ids = torch.topk(sampler_log_probs, k=k, dim=-1).indices
    head = topk_log_probs_from_logits(trainer_logits, ids)
    correction, q_mass, p_mass = score_centering_correction(
        head, sampler_log_probs.gather(-1, ids), score_centering_weight_fn(None, 2.0)
    )
    log_p = torch.log_softmax(trainer_logits, dim=-1)
    p = log_p.exp().detach()
    q_hat = sampler_log_probs.exp().clone()
    in_head = torch.zeros_like(q_hat, dtype=torch.bool).scatter_(1, ids, True)
    rho = (1 - q_mass) / (1 - p_mass)
    q_hat = torch.where(in_head, q_hat, rho.unsqueeze(-1) * p)
    ref = (q_hat.detach() * log_p).sum(-1)
    grad_a = torch.autograd.grad(correction.sum(), trainer_logits)[0]
    grad_b = torch.autograd.grad(ref.sum(), trainer_logits)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-5, rtol=1e-5)


def test_drift_cancels_under_constant_reward():
    # Constant reward, sampler != trainer: vanilla REINFORCE has a nonzero expected gradient,
    # score centering makes it vanish (up to the tail model, exact here with k = V).
    torch.manual_seed(4)
    vocab = 7
    trainer_logits = torch.randn(1, vocab, requires_grad=True)
    sampler_log_probs = torch.log_softmax(torch.randn(1, vocab), dim=-1)
    ids = torch.arange(vocab).expand(1, vocab)
    head = topk_log_probs_from_logits(trainer_logits, ids)
    correction, _, _ = score_centering_correction(head, sampler_log_probs, score_centering_weight_fn(None, 2.0))
    log_p = torch.log_softmax(trainer_logits, dim=-1)
    q = sampler_log_probs.exp()
    expected_vanilla = torch.autograd.grad((q * log_p).sum(), trainer_logits, retain_graph=True)[0]
    expected_centered = torch.autograd.grad((q.detach() * log_p).sum() - correction.sum(), trainer_logits)[0]
    assert expected_vanilla.abs().max() > 1e-3
    torch.testing.assert_close(expected_centered, torch.zeros_like(expected_centered), atol=1e-6, rtol=0)


def test_weight_fn_matches_tis_and_icepop_rules():
    ratio = torch.tensor([0.1, 0.7, 1.0, 3.0, 9.0])
    torch.testing.assert_close(score_centering_weight_fn(None, 2.0)(ratio), torch.ones(5))
    torch.testing.assert_close(score_centering_weight_fn("token", 2.0)(ratio), ratio.clamp(max=2.0))
    torch.testing.assert_close(
        score_centering_weight_fn("token", "0.5_5.0")(ratio), torch.tensor([0.0, 0.7, 1.0, 3.0, 0.0])
    )


def test_composed_correction_uses_alpha_rho_f_one_over_rho():
    torch.manual_seed(5)
    head_p = torch.log_softmax(torch.randn(2, 6), dim=-1)[:, :3]
    head_q = torch.log_softmax(torch.randn(2, 6), dim=-1)[:, :3]
    tis = score_centering_weight_fn("token", 2.0)
    correction, q_mass, p_mass = score_centering_correction(head_p, head_q, tis)
    rho = (1 - q_mass) / (1 - p_mass)
    alpha = rho * tis(1 / rho)
    residual = head_q.exp() * tis((head_p - head_q).exp()) - alpha.unsqueeze(-1) * head_p.exp()
    torch.testing.assert_close(correction, (residual * head_p).sum(-1))


def test_dummy_rows_are_finite_distributions():
    ids, log_probs = dummy_rollout_topk(3, 4)
    assert ids.dtype == torch.int32 and log_probs.dtype == torch.float32
    assert ids.tolist() == [[0, 1, 2, 3]] * 3
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(3))


def test_pad_rollout_topk_places_heads_after_last_prompt_token():
    k = 2
    response_ids = [[10, 11], [12, 13], [14, 15]]
    response_log_probs = [[-0.1, -2.0]] * 3
    ids, log_probs = pad_rollout_topk(response_ids, response_log_probs, prompt_length=5)
    assert ids.shape == (8, k) and log_probs.shape == (8, k)
    assert ids[4:7].tolist() == response_ids
    assert ids[3].tolist() == [0, 1] and ids[7].tolist() == [0, 1]
    assert math.isclose(log_probs[0, 0].item(), -math.log(k), rel_tol=1e-6)


@dataclass
class _FlatLogprobs:
    """Minimal stand-in for vLLM ``FlatLogprobs``: ``[sampled, top-1, ..., top-k]`` per position."""

    token_ids: list[int]
    logprobs: list[float]
    num_positions: int

    def __len__(self):
        return self.num_positions


def test_extract_reads_sampled_token_and_head_by_rank():
    # position 0 samples the top-1 token; position 1 samples token 42, outside the head
    flat = _FlatLogprobs(
        token_ids=[7, 7, 3, 9, 42, 3, 7, 9],
        logprobs=[-0.1, -0.1, -1.5, -2.0, -8.0, -0.2, -1.0, -1.9],
        num_positions=2,
    )
    sampled, ids, log_probs = extract_response_topk_logprobs(flat, k=3)
    assert sampled == pytest.approx([-0.1, -8.0])
    assert ids.dtype == np.int32 and log_probs.dtype == np.float32
    assert ids.tolist() == [[7, 3, 9], [3, 7, 9]]
    np.testing.assert_allclose(log_probs, [[-0.1, -1.5, -2.0], [-0.2, -1.0, -1.9]], rtol=1e-6)


def test_extract_rejects_ragged_rows():
    flat = _FlatLogprobs(token_ids=[7, 7, 3], logprobs=[-0.1, -0.1, -1.5], num_positions=1)
    with pytest.raises(ValueError, match="entries per generated token"):
        extract_response_topk_logprobs(flat, k=3)


def test_extract_matches_vllm_dict_logprobs():
    vllm_logprobs = pytest.importorskip("vllm.logprobs")
    k, num_tokens = 4, 6
    generator = torch.Generator().manual_seed(0)
    flat, rows, actions = vllm_logprobs.FlatLogprobs(), [], []
    for position in range(num_tokens):
        values, order = torch.log_softmax(torch.randn(32, generator=generator), -1).sort(descending=True)
        # alternate between a sampled token inside the head and one outside it
        action = int(order[1] if position % 2 == 0 else order[k + 3])
        action_value = float(values[(order == action).nonzero().item()])
        token_ids, log_probs = [action] + order[:k].tolist(), [action_value] + values[:k].tolist()
        rank = int((values >= action_value).sum())
        vllm_logprobs.append_logprobs_for_next_position(flat, token_ids, log_probs, [None] * (k + 1), rank, k)
        vllm_logprobs.append_logprobs_for_next_position(rows, token_ids, log_probs, [None] * (k + 1), rank, k)
        actions.append(action)

    sampled, ids, log_probs = extract_response_topk_logprobs(flat, k)
    # vLLM's dict view is what the server read before switching to flat_logprobs
    assert sampled == [rows[i][actions[i]].logprob for i in range(num_tokens)]
    head = [sorted((v.rank, t, v.logprob) for t, v in row.items() if v.rank <= k) for row in rows]
    assert ids.tolist() == [[t for _, t, _ in row] for row in head]
    assert log_probs.tolist() == [[np.float32(v).item() for _, _, v in row] for row in head]


async def _generate_with_segments(monkeypatch, segments):
    from verl.workers.rollout import llm_server
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    pending = iter(segments)

    async def fake_generate(self, request_id, *, prompt_ids, **kwargs):
        return next(pending)

    async def no_wait(_delay, *args, **kwargs):
        return None

    # generate reaches the upstream through super(), so the base class is patched
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)
    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)
    client = FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)
    return await client.generate(request_id="req-0", prompt_ids=[1, 2, 3], sampling_params={})


@pytest.mark.asyncio
async def test_fully_async_client_concatenates_rollout_topk_across_resumes(monkeypatch):
    from verl.workers.rollout.replica import TokenOutput

    def head(base, n):
        return {
            "response_topk_ids": np.arange(base, base + 2 * n, dtype=np.int32).reshape(n, 2),
            "response_topk_log_probs": np.full((n, 2), -float(base), dtype=np.float32),
        }

    segments = [
        TokenOutput(token_ids=[101, 102], stop_reason="aborted", extra_fields=head(10, 2)),
        TokenOutput(token_ids=[103], stop_reason="completed", extra_fields=head(20, 1)),
    ]
    output = await _generate_with_segments(monkeypatch, segments)
    assert output.token_ids == [101, 102, 103]
    assert output.extra_fields["response_topk_ids"].tolist() == [[10, 11], [12, 13], [20, 21]]
    assert output.extra_fields["response_topk_log_probs"].tolist() == [[-10, -10], [-10, -10], [-20, -20]]
    assert output.extra_fields["response_topk_ids"].dtype == np.int32


@pytest.mark.asyncio
async def test_fully_async_client_without_rollout_topk_stays_unchanged(monkeypatch):
    from verl.workers.rollout.replica import TokenOutput

    output = await _generate_with_segments(monkeypatch, [TokenOutput(token_ids=[101], stop_reason="completed")])
    assert "response_topk_ids" not in output.extra_fields


def test_padding_template_uses_dummy_heads():
    k = 3
    sample = TensorDict(
        {
            "prompts": torch.zeros(4, dtype=torch.int64),
            "responses": torch.zeros(2, dtype=torch.int64),
            "input_ids": torch.zeros(6, dtype=torch.int64),
            "attention_mask": torch.ones(6, dtype=torch.int64),
            "response_mask": torch.ones(2, dtype=torch.int64),
            "position_ids": torch.arange(6),
            "rollout_topk_ids": torch.zeros(6, k, dtype=torch.int32),
            "rollout_topk_log_probs": torch.zeros(6, k),
        },
        batch_size=[],
    )
    template, _ = construct_minimal_padding_template(sample, {}, eos_token_id=0)
    seq_len = template["input_ids"].shape[0]
    assert template["rollout_topk_ids"].shape == (seq_len, k)
    torch.testing.assert_close(template["rollout_topk_log_probs"].exp().sum(-1), torch.ones(seq_len))


def _actor_config(rollout_correction):
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rollout_correction),
        loss_agg_mode="token-mean",
    )
    config.global_batch_info.update(dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None)
    return config


def test_reinforce_adds_advantage_times_correction():
    torch.manual_seed(0)
    log_prob = torch.randn(2, 4, requires_grad=True)
    rollout_log_prob = log_prob.detach() + 0.1
    advantages = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    correction = torch.randn(2, 4, requires_grad=True)
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    loss_sc, metrics = compute_policy_loss_reinforce(
        rollout_log_prob, log_prob, advantages, mask, "token-mean", config, sc_correction=correction
    )
    loss_plain, _ = compute_policy_loss_reinforce(rollout_log_prob, log_prob, advantages, mask, "token-mean", config)
    torch.testing.assert_close(loss_sc, loss_plain + (advantages * correction).mean())
    assert math.isclose(metrics["actor/sc_correction"], correction.mean().item(), rel_tol=1e-5)


def test_bypass_mode_dispatches_correction_to_reinforce():
    torch.manual_seed(1)
    log_prob = torch.randn(2, 3, requires_grad=True)
    old = log_prob.detach()
    adv = torch.randn(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    corr = torch.randn(2, 3)
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    loss, metrics = compute_policy_loss_bypass_mode(old, log_prob, adv, mask, "token-mean", config, sc_correction=corr)
    assert "actor/sc_correction" in metrics


def _nested_batch(lengths, k, vocab):
    ids = torch.nested.as_nested_tensor(
        [torch.randint(0, vocab, (n, k), dtype=torch.int32) for n in lengths], layout=torch.jagged
    )
    log_probs = torch.nested.as_nested_tensor(
        [
            torch.log_softmax(torch.randn(n, vocab), dim=-1).gather(-1, i.long())
            for n, i in zip(lengths, ids.unbind(), strict=False)
        ],
        layout=torch.jagged,
    )
    data = TensorDict({"rollout_topk_ids": ids, "rollout_topk_log_probs": log_probs}, batch_size=[len(lengths)])
    return data


def test_logits_processor_returns_per_token_scalars():
    torch.manual_seed(2)
    lengths, k, vocab = [3, 5], 4, 17
    data = _nested_batch(lengths, k, vocab)
    logits = torch.randn(1, sum(lengths), vocab, requires_grad=True)
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    out = score_centering_logits_processor(student_logits=logits, data=data, config=config)
    assert set(out) == {"sc_correction", "sc_sampler_head_mass", "sc_train_head_mass"}
    for v in out.values():
        assert v.shape == (1, sum(lengths))
    assert out["sc_correction"].requires_grad
    assert (out["sc_sampler_head_mass"] <= 1.0 + 1e-6).all()


@pytest.mark.parametrize(
    "preset, rollout_correction",
    [
        (RolloutCorrectionConfig.bypass_pg_sc(), {"bypass_mode": True, "loss_type": "reinforce"}),
        (
            RolloutCorrectionConfig.bypass_pg_token_icepop_sc(),
            {"bypass_mode": True, "loss_type": "reinforce", "rollout_is": "token", "rollout_is_threshold": "0.5_5.0"},
        ),
    ],
)
def test_logits_processor_reads_mapping_rollout_correction(preset, rollout_correction):
    torch.manual_seed(4)
    lengths, k, vocab = [3, 5], 4, 17
    data = _nested_batch(lengths, k, vocab)
    logits = torch.randn(1, sum(lengths), vocab)
    expected = score_centering_logits_processor(student_logits=logits, data=data, config=_actor_config(preset))
    out = score_centering_logits_processor(
        student_logits=logits, data=data, config=_actor_config({**rollout_correction, "score_centering": True})
    )
    for key in expected:
        torch.testing.assert_close(out[key], expected[key])


def test_ppo_loss_applies_score_centering_end_to_end():
    torch.manual_seed(3)
    prompts = torch.tensor([[0, 5, 6], [7, 8, 9]])
    responses = torch.tensor([[11, 12, 0], [13, 14, 15]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    response_mask = attention_mask[:, 3:]
    advantages = torch.randn(2, 3)
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp_min(0),
            "old_log_probs": -torch.rand(2, 3),
            "advantages": advantages,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum()), global_batch_size=2)
    data = left_right_2_no_padding(data)

    prompt_lens, response_lens = [2, 3], [2, 3]
    full = {
        "log_probs": [-torch.rand(4), -torch.rand(6)],
        "sc_correction": [torch.randn(4), torch.randn(6)],
        "sc_sampler_head_mass": [torch.rand(4), torch.rand(6)],
        "sc_train_head_mass": [torch.rand(4), torch.rand(6)],
    }
    model_output = {key: torch.nested.as_nested_tensor(rows, layout=torch.jagged) for key, rows in full.items()}

    def response_part(rows):
        # response token r is predicted by sequence position prompt_len - 1 + r
        return torch.stack(
            [
                torch.nn.functional.pad(row[p - 1 : p - 1 + r], (0, 3 - r))
                for row, p, r in zip(rows, prompt_lens, response_lens, strict=True)
            ]
        )

    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    loss, metrics = ppo_loss(config, model_output, data)

    mask = response_mask.bool()
    log_prob, correction = response_part(full["log_probs"]), response_part(full["sc_correction"])
    expected = (-advantages * log_prob + advantages * correction)[mask].mean()
    torch.testing.assert_close(loss, expected)
    for key in ("sc_correction", "sc_sampler_head_mass", "sc_train_head_mass"):
        expected_metric = response_part(full[key])[mask].mean().item()
        assert math.isclose(metrics[f"actor/{key}"].aggregate(), expected_metric, rel_tol=1e-5)


def test_ppo_loss_gradient_is_unchanged_on_policy():
    """With the sampler head equal to the trainer head, score centering changes neither the loss
    nor its gradient with respect to the logits."""
    torch.manual_seed(5)
    vocab, k = 9, 4
    prompts = torch.tensor([[0, 5, 6], [7, 8, 9 % vocab]])
    responses = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    response_mask = attention_mask[:, 3:]
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp_min(0),
            "advantages": torch.randn(2, 3),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=int(response_mask.sum()), global_batch_size=2)
    data = left_right_2_no_padding(data)
    cu_seqlens = data["input_ids"].offsets()
    nnz = int(cu_seqlens[-1])
    logits = torch.randn(nnz, vocab)
    # next-token log-probs of the unpadded sequence, as the engine computes them
    next_ids = torch.roll(data["input_ids"].values(), shifts=-1)
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    old_log_probs = torch.nested.nested_tensor_from_jagged(log_probs.detach(), cu_seqlens)
    data["old_log_probs"] = no_padding_2_padding(old_log_probs, data)
    topk_ids = torch.topk(logits, k=k, dim=-1).indices.to(torch.int32)
    data["rollout_topk_ids"] = torch.nested.nested_tensor_from_jagged(topk_ids, cu_seqlens)
    data["rollout_topk_log_probs"] = torch.nested.nested_tensor_from_jagged(
        topk_log_probs_from_logits(logits, topk_ids).detach(), cu_seqlens
    )

    def loss_and_grad(score_centering):
        config = _actor_config(
            RolloutCorrectionConfig(
                bypass_mode=True,
                loss_type="reinforce",
                rollout_is="token",
                rollout_is_threshold=2.0,
                score_centering=score_centering,
            )
        )
        logits_ = logits.clone().requires_grad_(True)
        log_probs_ = torch.log_softmax(logits_, dim=-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(log_probs_, cu_seqlens)}
        if score_centering:
            hook = score_centering_ppo_loss(config, student_logits=logits_.unsqueeze(0), data=data)
            model_output.update(
                {
                    key: torch.nested.nested_tensor_from_jagged(value.squeeze(0), cu_seqlens)
                    for key, value in hook.items()
                }
            )
        loss, metrics = score_centering_ppo_loss(config, model_output=model_output, data=data)
        loss.backward()
        return loss.detach(), logits_.grad, metrics

    loss_pg, grad_pg, _ = loss_and_grad(False)
    loss_sc, grad_sc, metrics = loss_and_grad(True)
    assert metrics["actor/sc_correction"].aggregate() == 0.0
    assert torch.equal(loss_sc, loss_pg)
    assert torch.equal(grad_sc, grad_pg)


def test_score_centering_loss_rejects_missing_correction():
    # e.g. an entrypoint that selects the loss but never sets the score_centering micro-batch flag
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_token_tis_sc())
    with pytest.raises(RuntimeError, match="logits processor did not run"):
        score_centering_ppo_loss(config, model_output={"log_probs": torch.zeros(1, 2)}, data=None)
