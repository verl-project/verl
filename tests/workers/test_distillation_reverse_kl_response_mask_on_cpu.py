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
"""Regression for nested ``response_mask`` alignment in reverse-KL distillation.

``no_padding_2_padding`` pads student/teacher log-probs to ``max_response_len``,
while a nested ``response_mask`` is padded by ``to_padded_tensor`` to its own
max ragged length. Those baselines can disagree and trip the shape assert.
``align_response_mask`` pads with ``False`` or truncates trailing ``False``
cells so masked reductions stay unchanged.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import (
    align_padded_tensor,
    align_response_mask,
    compute_distillation_loss_range,
    compute_distillation_loss_reverse_kl_estimator,
    distillation_loss,
)
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import update_global_batch_info


def _make_distillation_config() -> DistillationConfig:
    loss_cfg = DistillationLossConfig(
        loss_mode="k1",
        topk=64,
        use_task_rewards=False,
        use_policy_gradient=True,
        loss_max_clamp=10.0,
        log_prob_min_clamp=-10.0,
    )
    return DistillationConfig(distillation_loss=loss_cfg)


def _build_inputs(prompt_lens, response_lens, mask_len_override=None):
    """Build packed reverse-KL inputs. ``mask_len_override`` decouples the nested
    mask's ``to_padded_tensor`` length from ``max(response_lens)``.

    Valid ``mask=True`` cells stay in the first ``r_i - 2`` slots so pad/truncate
    only touch trailing ``False`` positions.
    """
    bsz = len(prompt_lens)
    assert len(response_lens) == bsz
    total_nnz = sum(p + r for p, r in zip(prompt_lens, response_lens, strict=True))

    student_packed = torch.arange(total_nnz, dtype=torch.float32) * 0.01 + 0.3
    teacher_packed = (torch.arange(total_nnz, dtype=torch.float32) * 0.007 - 0.2).reshape(total_nnz, 1)

    prompts_nested = torch.nested.as_nested_tensor(
        [torch.zeros(p, dtype=torch.long) for p in prompt_lens], layout=torch.jagged
    )
    responses_nested = torch.nested.as_nested_tensor(
        [torch.zeros(r, dtype=torch.long) for r in response_lens], layout=torch.jagged
    )

    seg_lens = response_lens if mask_len_override is None else [mask_len_override] * bsz
    n_true_per_sample = [max(1, r - 2) for r in response_lens]
    mask_segs = []
    for i, length in enumerate(seg_lens):
        seg = torch.zeros(length)
        seg[: min(n_true_per_sample[i], length)] = 1.0
        mask_segs.append(seg)
    rmask_nested = torch.nested.nested_tensor(mask_segs, layout=torch.jagged)

    data = TensorDict(
        {
            "prompts": prompts_nested,
            "responses": responses_nested,
            "teacher_logprobs": teacher_packed,
            "response_mask": rmask_nested,
        },
        batch_size=[],
    )
    return {"log_probs": student_packed}, data


def _run(model_output, data, cfg):
    losses, metrics = compute_distillation_loss_reverse_kl_estimator(None, cfg, model_output, data)
    return losses.detach().clone(), float(metrics["distillation/abs_loss"].aggregate())


def _build_pg_inputs(length_offset):
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    model_output, data = _build_inputs(prompt_lens, response_lens)
    target_len = max(response_lens)
    source_len = target_len + length_offset

    old_log_prob_segs = []
    rollout_is_weight_segs = []
    for _ in response_lens:
        old_log_probs = torch.zeros(source_len)
        old_log_probs[: min(source_len, target_len)] = -0.1
        old_log_prob_segs.append(old_log_probs)

        rollout_is_weights = torch.zeros(source_len)
        rollout_is_weights[: min(source_len, target_len)] = 1.0
        rollout_is_weight_segs.append(rollout_is_weights)

    data["old_log_probs"] = torch.nested.nested_tensor(old_log_prob_segs, layout=torch.jagged)
    data["rollout_is_weights"] = torch.nested.nested_tensor(rollout_is_weight_segs, layout=torch.jagged)
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=None,
        global_batch_size=None,
    )
    return model_output, data


def test_reverse_kl_shape_consistent_is_noop():
    cfg = _make_distillation_config()
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    max_resp = max(response_lens)

    losses, abs_loss = _run(*_build_inputs(prompt_lens, response_lens), cfg)

    assert losses.shape == (len(prompt_lens), max_resp)
    assert torch.isfinite(losses).all()
    assert torch.isfinite(torch.tensor(abs_loss))


def test_reverse_kl_pad_shorter_matches_baseline():
    cfg = _make_distillation_config()
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    max_resp = max(response_lens)

    _, abs_base = _run(*_build_inputs(prompt_lens, response_lens), cfg)
    losses_pad, abs_pad = _run(*_build_inputs(prompt_lens, response_lens, mask_len_override=max_resp - 2), cfg)

    assert losses_pad.shape == (len(prompt_lens), max_resp)
    assert torch.isfinite(losses_pad).all()
    assert abs_pad == pytest.approx(abs_base, rel=0.0, abs=1e-10)


def test_reverse_kl_truncate_longer_masked_metric_stable():
    cfg = _make_distillation_config()
    prompt_lens = [3, 4, 3, 5]
    response_lens = [5, 7, 4, 6]
    max_resp = max(response_lens)

    _, abs_base = _run(*_build_inputs(prompt_lens, response_lens), cfg)
    losses_tr, abs_tr = _run(*_build_inputs(prompt_lens, response_lens, mask_len_override=max_resp + 3), cfg)

    assert losses_tr.shape == (len(prompt_lens), max_resp)
    assert torch.isfinite(losses_tr).all()
    assert abs_tr == pytest.approx(abs_base, rel=0.0, abs=1e-10)


def test_outer_loss_range_reuses_aligned_response_mask():
    losses = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    response_mask = torch.nested.nested_tensor(
        [torch.tensor([1, 1, 0, 0]), torch.tensor([1, 0, 0, 0])],
        layout=torch.jagged,
    )

    aligned = align_response_mask(response_mask, losses)
    metrics = compute_distillation_loss_range(losses, response_mask)

    assert aligned.shape == losses.shape
    assert aligned.dtype == torch.bool
    assert not aligned[:, 4:].any()
    assert float(metrics["distillation/loss_min"].aggregate()) == 0.0
    assert float(metrics["distillation/loss_max"].aggregate()) == 7.0


def test_align_response_mask_rejects_truncating_valid_tokens():
    losses = torch.zeros(1, 3)
    response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)

    with pytest.raises(ValueError, match="valid tokens beyond"):
        align_response_mask(response_mask, losses)


@pytest.mark.parametrize("length_offset", [-2, 2])
def test_distillation_loss_pg_aligns_nested_ppo_inputs(length_offset):
    actor_cfg = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True)
    distillation_cfg = _make_distillation_config()
    model_output, data = _build_pg_inputs(length_offset)

    loss, metrics = distillation_loss(actor_cfg, distillation_cfg, model_output, data)

    assert torch.isfinite(loss)
    assert "distillation/ppo_kl" in metrics
    assert "distillation/pg_clipfrac" in metrics


def test_align_padded_tensor_preserves_dtype_and_rejects_nonzero_tail():
    target = torch.zeros((2, 4), dtype=torch.float32)
    source = torch.ones((2, 2), dtype=torch.float64)

    aligned = align_padded_tensor(source, target, name="test tensor")

    assert aligned.shape == target.shape
    assert aligned.dtype == source.dtype
    assert aligned.device == source.device
    assert not aligned[:, 2:].any()

    with pytest.raises(ValueError, match="nonzero values beyond"):
        align_padded_tensor(torch.ones((2, 5)), target, name="test tensor")


def test_global_batch_info_is_refreshed_from_current_micro_batch():
    config = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True)
    config.global_batch_info["stale"] = 1
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(
        data,
        dp_size=4,
        batch_num_tokens=123,
        global_batch_size=16,
    )

    info = update_global_batch_info(config, data)
    unwrapped = {key: tu.unwrap_non_tensor_data(value) for key, value in info.items()}

    assert unwrapped == {
        "dp_size": 4,
        "batch_num_tokens": 123,
        "global_batch_size": 16,
        "loss_scale_factor": None,
    }
    assert config.global_batch_info == info
    assert "stale" not in config.global_batch_info
