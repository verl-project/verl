# Copyright 2026 Individual Contributor: Zupeng Wang
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

import math
from itertools import chain
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric, reduce_metrics
from verl.utils.py_functional import append_to_dict


def _make_case(padding, nested, overlap, device="cpu"):
    sample = {
        "prompts": torch.tensor([1]),
        "responses": torch.tensor([2, 3]),
        "attention_mask": torch.ones(3, dtype=torch.long),
        "response_mask": torch.ones(2, dtype=torch.long),
    }
    if padding:
        sample, tag = construct_minimal_padding_template(sample, {}, eos_token_id=0)
        assert tag["is_padding"]
    fields = {}
    for key in ("prompts", "responses", "attention_mask", "response_mask"):
        value = sample[key].to(device)
        fields[key] = (
            torch.nested.as_nested_tensor([value], layout=torch.jagged)
            if nested and key != "attention_mask"
            else value.unsqueeze(0)
        )
    data = TensorDict(fields, batch_size=[1])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=1)
    size = sample["attention_mask"].numel()
    outputs = {
        "distillation_losses": torch.full((size,), 0.4, device=device, requires_grad=True),
        "student_mass": torch.full((size,), 0.7, device=device),
        "teacher_mass": torch.full((size,), 0.8, device=device),
    }
    if overlap:
        outputs.update(
            overlap_count=torch.ones(size, device=device),
            overlap_token_advantage=torch.full((size,), -0.2, device=device),
        )
    actor_config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None)
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="forward_kl_topk", topk=2, use_policy_gradient=False, loss_max_clamp=None
        )
    )
    return actor_config, distillation_config, outputs, data


def _run_case(padding, nested=False, overlap=True):
    actor, config, outputs, data = _make_case(padding, nested, overlap)
    loss, metrics = distillation_loss(actor, config, outputs, data)
    return loss, metrics, outputs


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_padding_only_distillation_is_differentiable_zero(nested, overlap):
    loss, metrics, outputs = _run_case(True, nested, overlap)
    assert loss.requires_grad
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(outputs["distillation_losses"].grad, torch.zeros_like(outputs["distillation_losses"]))
    for metric in metrics.values():
        assert metric.values == [] if isinstance(metric, Metric) else metric == []


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_real_distillation_metrics_and_gradients_are_preserved(nested, overlap):
    loss, metrics, outputs = _run_case(False, nested, overlap)
    assert loss.item() == pytest.approx(0.4)
    loss.backward()
    torch.testing.assert_close(outputs["distillation_losses"].grad, torch.tensor([0.5, 0.5, 0.0]))
    reduced = {key: value.aggregate() if isinstance(value, Metric) else value for key, value in metrics.items()}
    assert reduced["distillation/loss_min"] == pytest.approx(0.4)
    assert reduced["distillation/loss_max"] == pytest.approx(0.4)
    for name, expected in (("student_mass", 0.7), ("teacher_mass", 0.8)):
        for suffix in ("", "_min", "_max"):
            assert reduced[f"distillation/{name}{suffix}"] == pytest.approx(expected)
    if overlap:
        assert reduced["distillation/overlap_ratio"] == pytest.approx(0.5)
        assert reduced["distillation/overlap_token_advantage"] == pytest.approx(-0.2)


def test_padding_does_not_change_metrics_across_microbatches_and_dp_ranks():
    _, real, _ = _run_case(False)
    _, empty, _ = _run_case(True)
    ranks = []
    # Match the worker's append -> all-gather -> DP reduction pipeline, including
    # a rank containing only synthetic padding and unequal observation counts.
    for microbatches in ((real, empty), (empty, real), (empty, empty)):
        rank_metrics = {}
        for metrics in microbatches:
            append_to_dict(rank_metrics, metrics)
        ranks.append(rank_metrics)
    reduced = {}
    for key in real:
        values = [rank[key] for rank in ranks]
        if isinstance(values[0], Metric):
            reduced[key] = Metric.aggregate_dp(values)
        else:
            reduced[key] = list(chain.from_iterable(values))
    reduced = reduce_metrics(reduced)
    for key, expected in real.items():
        expected = expected.aggregate() if isinstance(expected, Metric) else expected
        assert reduced[key] == pytest.approx(expected)


def test_all_padding_extrema_are_undefined_not_fabricated_zero():
    _, metrics, _ = _run_case(True)
    for metric in metrics.values():
        if isinstance(metric, Metric):
            assert math.isnan(Metric.aggregate_dp([metric, metric]))
