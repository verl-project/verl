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
"""Plumbing of ``kl_loss_type: topk_rb``: FSDP engine outputs, the padding round trip, and ``ppo_loss``.

The end-to-end test runs the real pipeline (ref engine -> left-right padding -> actor engine -> ppo_loss) and
the estimator on directly aligned tensors; agreement pins the alignment of the reference top-k to the logits.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.workers.engine.fsdp.transformer_impl as fsdp_impl
from verl.trainer.ppo.topk_rb_kl import compute_coarse_reverse_kl, compute_topk_log_probs, compute_topk_rb_kl
from verl.utils import tensordict_utils as tu
from verl.utils.config import validate_topk_rb_kl_config
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

V, K = 11, 3


def _nested(values, cu):
    return torch.nested.nested_tensor_from_jagged(values, offsets=cu)


def _engine_outputs(
    logits, labels, cu, flags, use_remove_padding=True, extra=None, temperature=1.0, batch_invariant=False
):
    """FSDPEngineWithLMHead.prepare_model_outputs on packed ``(N, V)`` logits, in either padding branch.

    ``logprobs_from_logits`` is patched to a plain gather (the flash-attn kernel needs CUDA) that also records
    the ``inplace_backward`` it was asked for.
    """
    if use_remove_padding:
        output = SimpleNamespace(logits=logits.unsqueeze(0))
        output_args = {"temperature_rmpad": torch.full((len(labels),), temperature), "pad_size": 0}
    else:
        output = SimpleNamespace(logits=_nested(logits, cu).to_padded_tensor(0.0))
        output_args = {"temperature": torch.full((len(cu) - 1,), temperature)}
    output_args.update(input_ids_rmpad_rolled=labels, temperature_is_one=temperature == 1.0)
    micro_batch = TensorDict({"input_ids": _nested(labels.roll(1), cu), **(extra or {})}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        **flags,
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, batch_invariant=batch_invariant)
    seen = {}

    def log_probs(logits, labels, inplace_backward, **kw):
        seen["inplace_backward"] = inplace_backward
        return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    with patch.object(fsdp_impl, "logprobs_from_logits", side_effect=log_probs):
        out = engine.prepare_model_outputs(output, output_args, micro_batch, logits_processor_func=None)
    return out, seen["inplace_backward"]


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_ref_pass_emits_its_topk(use_remove_padding):
    cu = torch.tensor([0, 4, 7])
    logits = torch.randn(7, V)
    out, _ = _engine_outputs(logits, torch.randint(0, V, (7,)), cu, {"ref_topk": K}, use_remove_padding)
    log_probs, ids = compute_topk_log_probs(logits, K)
    assert out["ref_topk_ids"].values().dtype == torch.int32
    torch.testing.assert_close(out["ref_topk_ids"].values().long(), ids)
    torch.testing.assert_close(out["ref_topk_log_probs"].values(), log_probs)


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_actor_pass_emits_coarse_terms_with_gradient(use_remove_padding):
    cu = torch.tensor([0, 5, 8])
    logits = torch.randn(8, V, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, V, (8,))
    ref_lp, ref_ids = compute_topk_log_probs(torch.randn(8, V, dtype=torch.float64), K)
    extra = {"ref_topk_ids": _nested(ref_ids.to(torch.int32), cu), "ref_topk_log_probs": _nested(ref_lp, cu)}
    out, inplace_backward = _engine_outputs(logits, labels, cu, {"kl_topk_rb": True}, use_remove_padding, extra)
    assert inplace_backward is False  # the coarse KL re-reads the logits in backward

    coarse, log_q_tail = compute_coarse_reverse_kl(logits, ref_ids, ref_lp)
    torch.testing.assert_close(out["kl_topk_coarse"].values(), coarse)
    torch.testing.assert_close(out["kl_topk_log_q_tail"].values(), log_q_tail)
    engine_objective = out["kl_topk_coarse"].values() * 1.3 + out["kl_topk_log_q_tail"].values()
    torch.testing.assert_close(
        torch.autograd.grad(engine_objective.sum(), logits)[0],
        torch.autograd.grad((coarse * 1.3 + log_q_tail).sum(), logits)[0],
    )


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_batch_invariant_topk_terms_use_the_log_prob_distribution(use_remove_padding):
    """batch_invariant log-probs scale fp32 unscaled logits by the temperature; with bf16 logits that differs from
    scaling in bf16, so the top-k terms must be computed from the same fp32-scaled logits."""
    cu = torch.tensor([0, 5, 8])
    logits = (torch.randn(8, V) * 3).to(torch.bfloat16).requires_grad_(True)
    ref_lp, ref_ids = compute_topk_log_probs(torch.randn(8, V), K)
    extra = {"ref_topk_ids": _nested(ref_ids.to(torch.int32), cu), "ref_topk_log_probs": _nested(ref_lp, cu)}
    out, _ = _engine_outputs(
        logits, torch.randint(0, V, (8,)), cu, {"kl_topk_rb": True}, use_remove_padding, extra, 0.7, True
    )
    expected, _ = compute_coarse_reverse_kl(logits.float() / 0.7, ref_ids, ref_lp)
    torch.testing.assert_close(out["kl_topk_coarse"].values(), expected, atol=1e-6, rtol=1e-6)
    bf16_scaled, _ = compute_coarse_reverse_kl(logits / torch.tensor(0.7, dtype=torch.bfloat16), ref_ids, ref_lp)
    assert (bf16_scaled - expected).abs().max() > 1e-4  # the fixture really distinguishes the two scalings


def test_inplace_backward_kept_without_topk_rb():
    _, inplace_backward = _engine_outputs(torch.randn(3, V), torch.randint(0, V, (3,)), torch.tensor([0, 3]), {})
    assert inplace_backward is True


@pytest.mark.parametrize("flags", [{"ref_topk": 4}, {"kl_topk_rb": True}])
@pytest.mark.parametrize("fused,pad_to_length", [(True, False), (False, True)])
def test_unsupported_engine_modes_are_rejected(flags, fused, pad_to_length):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = pad_to_length
    micro_batch = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(micro_batch, use_fused_kernels=fused, **flags)
    with pytest.raises(NotImplementedError, match="topk_rb"):
        engine.prepare_model_inputs(micro_batch)


def test_ulysses_rank_pairs_its_logits_with_the_matching_ref_rows():
    """5 packed rows, SP=2 -> padded to 6, rank 1 owns rows 3..5 (row 5 is padding)."""
    cu = torch.tensor([0, 2, 5])
    logits = torch.randn(6, V, dtype=torch.float64)
    ref_lp, ref_ids = compute_topk_log_probs(torch.randn(5, V, dtype=torch.float64), K)
    micro_batch = TensorDict(
        {"ref_topk_ids": _nested(ref_ids.to(torch.int32), cu), "ref_topk_log_probs": _nested(ref_lp, cu)},
        batch_size=[],
    )
    tu.assign_non_tensor(micro_batch, kl_topk_rb=True)

    def rank1_slice(value, dim, padding):
        assert value.shape[1] == 6  # padded like the token stream before slicing
        return value[:, 3:6]

    with patch.object(fsdp_impl, "slice_input_tensor", side_effect=rank1_slice):
        out = fsdp_impl._topk_kl_side_outputs(logits[3:6], micro_batch, pad_size=1, use_ulysses_sp=True)
    coarse, log_q_tail = compute_coarse_reverse_kl(logits[3:5], ref_ids[3:5], ref_lp[3:5])
    torch.testing.assert_close(out["kl_topk_coarse"][:2], coarse)
    torch.testing.assert_close(out["kl_topk_log_q_tail"][:2], log_q_tail)


def _batch(prompt_lens, response_lens, max_prompt, max_response, seed=0):
    """Left-padded prompts, right-padded responses, as the trainer holds them."""
    g = torch.Generator().manual_seed(seed)
    bsz = len(prompt_lens)
    prompts = torch.zeros(bsz, max_prompt, dtype=torch.long)
    responses = torch.zeros(bsz, max_response, dtype=torch.long)
    attention_mask = torch.zeros(bsz, max_prompt + max_response, dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        prompts[i, max_prompt - p :] = torch.randint(1, V, (p,), generator=g)
        responses[i, :r] = torch.randint(1, V, (r,), generator=g)
        attention_mask[i, max_prompt - p : max_prompt + r] = 1
    return TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], 1),
            "attention_mask": attention_mask,
            "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
            "response_mask": attention_mask[:, max_prompt:],
        },
        batch_size=[bsz],
    )


def test_ppo_loss_end_to_end_matches_the_estimator_on_aligned_tensors():
    batch = _batch([2, 3], [3, 2], max_prompt=3, max_response=3, seed=3)
    td = left_right_2_no_padding(batch.clone())
    cu = td["input_ids"].offsets()
    labels = td["input_ids"].values().roll(-1)  # next token per packed row
    policy_logits = torch.randn(len(labels), V, dtype=torch.float64)
    ref_logits = policy_logits + torch.randn(len(labels), V, dtype=torch.float64)

    def per_row(log_probs):  # log-prob of the label per packed row, as a response-padded tensor
        return no_padding_2_padding(_nested(log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1), cu), td)

    # reference pass; its nested full-sequence outputs go into the batch as stored by the trainer
    ref_out, _ = _engine_outputs(ref_logits, labels, cu, {"ref_topk": K})
    actor_td = td
    for key in ("ref_topk_log_probs", "ref_topk_ids"):
        actor_td[key] = ref_out[key]
    actor_td["ref_log_prob"] = per_row(ref_logits.log_softmax(-1)).float()
    actor_td["old_log_probs"] = per_row(policy_logits.log_softmax(-1)).float()
    actor_td["advantages"] = torch.zeros_like(actor_td["old_log_probs"])
    tu.assign_non_tensor(actor_td, dp_size=1, batch_num_tokens=None, global_batch_size=None)

    # actor pass and loss, with only the KL term active
    z = policy_logits.clone().requires_grad_(True)
    extra = {key: actor_td[key] for key in ("ref_topk_ids", "ref_topk_log_probs")}
    actor_out, _ = _engine_outputs(z, labels, cu, {"kl_topk_rb": True}, extra=extra)
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="token-mean",
        use_kl_loss=True,
        kl_loss_type="topk_rb",
        kl_topk=K,
        kl_loss_coef=1.0,
    )
    loss, metrics = ppo_loss(config, actor_out, actor_td)

    # the estimator directly on row-aligned tensors, token-mean over response tokens
    z2 = policy_logits.clone().requires_grad_(True)
    ref_lp, ref_ids = compute_topk_log_probs(ref_logits, K)
    coarse, log_q_tail = compute_coarse_reverse_kl(z2, ref_ids, ref_lp)
    log_prob = z2.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    ref_log_prob = ref_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    in_topk = (ref_ids == labels.unsqueeze(-1)).any(-1)
    per_token, value = compute_topk_rb_kl(
        coarse, log_q_tail, ref_lp[:, -1], in_topk, log_prob, ref_log_prob, log_prob.detach()
    )
    mask = batch["response_mask"].bool()
    expected_loss = no_padding_2_padding(_nested(per_token, cu), td)[mask].mean()
    expected_value = no_padding_2_padding(_nested(value, cu), td)[mask].mean()

    torch.testing.assert_close(
        torch.autograd.grad(loss, z)[0], torch.autograd.grad(expected_loss, z2)[0], atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        torch.as_tensor(metrics["kl_loss"].aggregate()).double(), expected_value, atol=1e-6, rtol=1e-5
    )


def test_config_rejects_empty_topk():
    with pytest.raises(ValueError, match="kl_topk"):
        ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            use_kl_loss=True,
            kl_loss_type="topk_rb",
            kl_topk=0,
        )


def _trainer_config(strategy="fsdp", fused=False, kl_in_reward=None, pad_to_length=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": True,
                    "kl_loss_type": "topk_rb",
                    "strategy": strategy,
                    "pad_to_length": pad_to_length,
                },
                "model": {"use_fused_kernels": fused},
            },
            "algorithm": {"use_kl_in_reward": kl_in_reward is not None, "kl_penalty": kl_in_reward or "kl"},
        }
    )


def test_startup_validation_rejects_unsupported_setups():
    validate_topk_rb_kl_config(_trainer_config())
    validate_topk_rb_kl_config(_trainer_config("megatron", pad_to_length=True))  # Megatron packs ref top-k like labels
    with pytest.raises(NotImplementedError, match="strategy"):
        validate_topk_rb_kl_config(_trainer_config("torchtitan"))
    with pytest.raises(ValueError, match="use_fused_kernels"):
        validate_topk_rb_kl_config(_trainer_config(fused=True))
    with pytest.raises(ValueError, match="kl_penalty"):
        validate_topk_rb_kl_config(_trainer_config(kl_in_reward="topk_rb"))
    with pytest.raises(NotImplementedError, match="pad_to_length"):
        validate_topk_rb_kl_config(_trainer_config("fsdp2", pad_to_length=True))
